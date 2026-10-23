"""Train manuscript backbones and two-stage/dual-branch configurations."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import run_nonnegative_oneway as N
import run_e1_five_models as F
from gwno.data.dataset import load_sequence
from gwno.data.prepare import write_json
from known_recharge_v3 import install

ROOT = Path(__file__).resolve().parent
H, C, A = N.H, N.C, N.A
PREPARED = ROOT / 'prepared_formal_mf680_randomk_r3_repaired'


def load_data(settings):
    split = json.loads((PREPARED/'splits/d0_development.json').read_text())
    if [len(split[k]) for k in ['train', 'val', 'test']] != [70, 20, 10]:
        raise ValueError('Generate all 100 scenarios before formal training.')
    norm = json.loads((PREPARED/'normalization.json').read_text())
    data = {}
    for i, name in enumerate(split['train'] + split['val']):
        d = load_sequence(PREPARED, name, norm, 'cuda')
        d.update(H.chd_inputs(PREPARED, name, norm, 'cuda'))
        d['scenario_index'] = torch.tensor(i, device='cuda')
        data[name] = d
    count = sum(float(data[n]['active_mask'].sum()) * len(data[n]['head']) for n in split['train'])
    mean = sum(float(data[n]['chd_mask'].sum()) for n in split['train']) / count
    std = float(np.sqrt(mean * (1 - mean)))
    norm['forcing']['mean'][7], norm['forcing']['std'][7] = mean, std
    for d in data.values():
        d['forcing'][:, :, 7] = (d['chd_mask'].float() - mean) / std * d['active_mask']
    install(data, split['train'], PREPARED, norm, fit=True)
    write_json(ROOT/'runs_e1_v3/head_comparison/protocol.json',
               dict(normalization=norm, split=split, selection='validation head NRMSE',
                    learning_rate=settings['learning_rate'], test_used=False))
    write_json(ROOT/'runs_pde_ablation_e1/protocol.json',
               dict(normalization=norm, split=split, scales=settings['residual_scales'],
                    weights=[settings['flow_weight'], settings['transport_weight']]))
    return data, split, norm


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['head', 'guided-head', 'backbone-concentration', 'two-stage', 'dual'])
    p.add_argument('--model', choices=['unet', 'fno', 'afno', 'uno', 'deeponet'], default='afno')
    p.add_argument('--group', choices=['00', 'C', 'H', 'HC'], default='H')
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    p.add_argument('--epochs', type=int, default=2000)
    args = p.parse_args()
    if args.stage == 'dual' and args.group not in ['H', 'HC']:
        p.error('The manuscript dual-branch comparison contains H and HC.')
    if args.stage in ['guided-head', 'two-stage', 'dual'] and args.model != 'afno':
        p.error('The manuscript guidance/framework comparison uses AFNO.')
    if args.stage == 'backbone-concentration' and args.model == 'afno':
        p.error('Use two-stage --group 00 for the AFNO concentration baseline.')
    if not torch.cuda.is_available():
        p.error('The original training loops require a CUDA-enabled PyTorch installation.')
    torch.set_num_threads(4)
    settings = json.loads((ROOT.parent/'configs/paper.json').read_text())
    data, split, norm = load_data(settings)
    fit_args = SimpleNamespace(epochs=args.epochs, batch=8, hard_nonnegative=True)
    train_data = {n: data[n] for n in split['train']}
    val_data = {n: data[n] for n in split['val']}
    scales = settings['residual_scales']
    for seed in args.seeds:
        if args.stage in ['head', 'guided-head']:
            if args.stage == 'guided-head':
                A.NORM, A.SCALES, A.WH = norm, scales, settings['flow_weight']
                A.CACHES = {int(data[n]['scenario_index']): N.move(N.load_physics(PREPARED, n)) for n in split['train']}
                H.rollout = A.head_rollout
                folder = ROOT/f'runs_pde_ablation_e1/head_H/seed_{seed}'
            else:
                folder = ROOT/f'runs_e1_v3/head_comparison/{args.model}/lr_0.0003/seed_{seed}'
            H.fit(args.model, .0003, seed, fit_args, train_data, val_data, norm, split, folder)
        elif args.stage == 'backbone-concentration':
            item = dict(model='afno', lr=.0003)
            heads, checkpoint = C.load_head(item, seed, data, norm)
            C.fit(args.model, .0003, seed, item, fit_args, data, split, norm,
                  heads, checkpoint, ROOT/'runs_e1_five_models_20260915/concentration')
        else:
            N.SCHEDULE = 'sequential' if args.stage == 'two-stage' else 'dual'
            N.OUT = ROOT/'runs_nonnegative_oneway'/N.SCHEDULE
            weights = [settings['flow_weight'] if N.SCHEDULE == 'dual' else 0.,
                       settings['transport_weight'] if args.group in ['C', 'HC'] else 0.]
            caches = {n: N.move(N.load_physics(PREPARED, n)) for n in split['train']} if any(weights) else {}
            if N.SCHEDULE == 'sequential':
                N.HEAD_CHECKPOINT = str(ROOT/f'runs_pde_ablation_e1/head_H/seed_{seed}/best.pt') if args.group in ['H', 'HC'] else str(ROOT/f'runs_e1_v3/head_comparison/afno/lr_0.0003/seed_{seed}/best.pt')
                A.NORM = norm
                heads = A.load_heads(N.HEAD_CHECKPOINT, data)
                N.HEADS = torch.stack([heads[n] for n in data])
                del heads
            else:
                N.HEADS, N.HEAD_CHECKPOINT = None, None
            N.train(args.group, seed, data, split, norm, caches, weights, scales, args.epochs)


if __name__ == '__main__':
    main()
