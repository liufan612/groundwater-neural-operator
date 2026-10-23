"""Manuscript AFNO two-stage and dual-branch training with nonnegative states."""
import argparse
import copy
import json
import os
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch

import run_e1_head_v3 as H
import run_e1_concentration_v3 as C
from run_pde_ablation_e1 import move, write
from e23_physics import load as load_physics
from e23_engine import callbacks, physics_metrics
from pde_batch import FusedPDE
from gwno.models.factory import build_operator
from gwno.training.trainer import seed_all, early_stop_update
from gwno.evaluation.fields import field_metrics
from nonnegative_state import project_concentration
import e23_engine as E
import run_pde_ablation_e1 as A

ROOT = H.ROOT
OUT = ROOT / 'runs_nonnegative_oneway'
SCHEDULE = 'dual'
HEADS = None
HEAD_CHECKPOINT = None
KEEP = list(range(34)) + [36, 37]


def networks(seed):
    seed_all(seed)
    h = build_operator('afno', 36, 2, H.MODELS['afno']).cuda()
    seed_all(seed + 1000)
    c = build_operator('afno', 38, 2, C.CAPACITIES['afno']).cuda()
    # Preserve all common E1 initialization tensors; remove only h channels.
    lift = torch.nn.Conv2d(36, 88, 1).cuda()
    with torch.no_grad():
        lift.weight.copy_(c.lift.weight[:, KEEP])
        lift.bias.copy_(c.lift.bias)
    c.lift = lift
    return h, c


def trajectory(hnet, cnet, d, norm, pde=None, weights=(0., 0.), scales=None, backward=False):
    mask = d['active_mask'].bool()
    den = mask.sum((1, 2, 3))
    h = torch.where(d['chd_mask'][0], d['chd_values'][0], d['head'][0]) * mask
    c = project_concentration(d['concentration'][0], mask, norm)
    paths = {'head': [h.detach()], 'concentration': [c.detach()]}
    total = torch.zeros((len(h), 5), device=h.device, dtype=torch.float64)
    n = len(d['head']) - 1
    segment = []
    hs, cs = norm['head'], norm['concentration']
    fixed = HEADS[d['scenario_index']].transpose(0, 1) if SCHEDULE == 'sequential' else None
    for t in range(1, n + 1):
        boundary = d['chd_values'][t] * d['chd_mask'][t]
        hx = torch.cat([H.model_input(d, h, t), boundary], 1)
        cx = torch.cat([C.model_input(d, c, t, fixed[t] if fixed is not None else None), boundary], 1)
        hn = fixed[t] if fixed is not None else torch.where(d['chd_mask'][t], d['chd_values'][t], h + hnet(hx)) * mask
        cn = project_concentration(c + cnet(cx), mask, norm)
        lh = ((hn - d['head'][t]).square() * mask).sum((1, 2, 3)) / den
        lc = ((cn - d['concentration'][t]).square() * mask).sum((1, 2, 3)) / den
        ln = (torch.relu(-(cn + cs['mean'][0] / cs['std'][0])).square() * mask).sum((1, 2, 3)) / den
        lw, lp = torch.zeros_like(lh), torch.zeros_like(lc)
        if weights[0] or weights[1]:
            hp = [v * hs['std'][0] + hs['mean'][0] for v in (h, hn)]
            dt = d['times_days'][t, 0] - d['times_days'][t - 1, 0]
            if weights[0]:
                lw = pde.water(*hp, t, dt)
            if weights[1]:
                cp = [v * cs['std'][0] + cs['mean'][0] for v in (c, cn)]
                lp = pde.transport(*cp, *[v.detach() for v in hp], t, dt)
        loss = lc if fixed is not None else lh + lc
        if weights[0]:
            loss = loss + weights[0] * lw / scales['h']
        if weights[1]:
            loss = loss + weights[1] * lp / scales['c']
        total += torch.stack([lh, lc, ln, lw, lp], 1).detach() / n
        if backward:
            segment.append(loss.mean() / n)
        paths['head'].append(hn.detach())
        paths['concentration'].append(cn.detach())
        h, c = hn, cn
        if t % 5 == 0 or t == n:
            if backward:
                torch.stack(segment).sum().backward()
                segment = []
            h, c = h.detach(), c.detach()
    return {k: torch.stack(v) for k, v in paths.items()}, total


def validation(h, c, data, norm, details=False):
    h.eval()
    c.eval()
    errors, rows = [], []
    names = list(data)
    with torch.no_grad():
        for offset in range(0, len(names), 8):
            keys = names[offset:offset + 8]
            paths, loss = trajectory(h, c, H.batch([data[n] for n in keys]), norm)
            errors.extend(loss[:, :2].cpu().tolist())
            if details:
                for i, name in enumerate(keys):
                    d = data[name]
                    mask = d['active_mask'].cpu().numpy().astype(bool)
                    for variable in ['head', 'concentration']:
                        s = norm[variable]
                        p = paths[variable][:, i].cpu().numpy() * s['std'][0] + s['mean'][0]
                        y = d[variable].cpu().numpy() * s['std'][0] + s['mean'][0]
                        m = field_metrics(p, y, mask, s['std'][0])
                        layers = [field_metrics(p[:, j:j+1], y[:, j:j+1], mask[j:j+1], s['std'][0]) for j in range(2)]
                        rows.append(dict(scenario=name, variable=variable, **m, layers=layers,
                                         negative_amplitude=float(np.maximum(-p[1:, mask], 0).mean())))
    a = np.asarray(errors)
    return dict(system_mse=float(a.mean()), head_nrmse=float(np.sqrt(a[:, 0]).mean()),
                concentration_nrmse=float(np.sqrt(a[:, 1]).mean())), rows


def train(group, seed, data, split, norm, caches, weights, scales, epochs):
    folder = OUT / group / f'seed_{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'result.json').exists():
        return json.loads((folder / 'result.json').read_text())
    h, c = networks(seed)
    if SCHEDULE == 'sequential':
        seed_all(seed+1000)
        c = build_operator('afno',38,2,C.CAPACITIES['afno']).cuda()
        cp = torch.load(HEAD_CHECKPOINT,map_location='cuda',weights_only=False)
        h.load_state_dict(cp['model']);h.requires_grad_(False)
    opt = torch.optim.AdamW(([p for p in h.parameters() if p.requires_grad] + list(c.parameters())), lr=.0003, weight_decay=1e-6)
    start, wait, best, reference = 0, 0, float('inf'), float('inf')
    if (folder / 'last.pt').exists():
        cp = torch.load(folder / 'last.pt', map_location='cuda', weights_only=False)
        assert cp['normalization'] == norm and cp['weights'] == weights
        h.load_state_dict(cp['head'])
        c.load_state_dict(cp['concentration'])
        opt.load_state_dict(cp['optimizer'])
        start, best, reference, wait = cp['epoch'] + 1, cp['best'], cp['reference'], cp['wait']
    val = {n: data[n] for n in split['val']}
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start, epochs):
        if wait >= 100:
            break
        tick = time.time()
        h.train()
        c.train()
        names = list(split['train'])
        random.Random(seed + epoch).shuffle(names)
        losses = []
        for offset in range(0, len(names), 8):
            keys = names[offset:offset + 8]
            d = H.batch([data[n] for n in keys])
            pde = FusedPDE([caches[n] for n in keys], norm) if any(weights) else None
            opt.zero_grad(set_to_none=True)
            _, loss = trajectory(h, c, d, norm, pde, weights, scales, True)
            if not torch.isfinite(loss).all() or not all(p.grad is not None and torch.isfinite(p.grad).all() for net in (h, c) for p in net.parameters() if p.requires_grad):
                raise FloatingPointError('Nonfinite loss/gradient or disconnected branch')
            opt.step()
            losses.extend(loss.tolist())
        v, _ = validation(h, c, val, norm)
        if not all(np.isfinite(x) for x in v.values()):
            raise FloatingPointError('Nonfinite validation')
        improved = v['system_mse'] < best
        best = min(best, v['system_mse'])
        reference, wait = early_stop_update(v['concentration_nrmse'], reference, wait, 1e-4)
        row = dict(epoch=epoch, train=dict(zip(['head_mse', 'concentration_mse', 'negative_penalty', 'water_raw', 'transport_raw'], np.mean(losses, 0).tolist())),
                   validation=v, wait=wait, seconds=time.time()-tick)
        cp = dict(head=h.state_dict(), concentration=c.state_dict(), optimizer=opt.state_dict(), epoch=epoch,
                  best=best, reference=reference, wait=wait, normalization=norm, split=split,
                  weights=weights, scales=scales, seed=seed, head_inputs=36, concentration_inputs=38 if SCHEDULE=='sequential' else 36, schedule=SCHEDULE, hard_nonnegative=True, transport_head_detached=True, head_checkpoint=HEAD_CHECKPOINT)
        torch.save(cp, folder / 'last.pt')
        if improved:
            torch.save(cp, folder / 'best.pt')
        with (folder / 'history.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        write(folder / 'status.json', dict(status='RUNNING', group=group, seed=seed, pid=os.getpid(), **row))
        print(json.dumps(dict(group=group, seed=seed, **row)), flush=True)
    cp = torch.load(folder / 'best.pt', map_location='cuda', weights_only=False)
    h.load_state_dict(cp['head'])
    c.load_state_dict(cp['concentration'])
    selected, rows = validation(h, c, val, norm, True)
    write(folder / 'val_metrics.json', rows)
    write(folder / 'status.json', dict(status='EVALUATING_PHYSICS', group=group, seed=seed))
    phys = callbacks(ROOT / 'prepared_formal_mf680_randomk_r3_repaired', list(val), norm)
    E.trajectory = evaluation_adapter
    diagnostic = physics_metrics(h, c, val, phys, norm, 'parallel', scales, folder)
    summary = {var: {key: float(np.mean([r[key] for r in rows if r['variable'] == var])) for key in
                    ['r2', 'rmse', 'mae', 'nrmse', 'late_nrmse', 'negative_fraction', 'negative_amplitude']} for var in ['head', 'concentration']}
    history = [json.loads(s) for s in (folder / 'history.jsonl').read_text().splitlines()]
    result = dict(status='DONE', group=group, seed=seed, schedule=SCHEDULE, metrics=summary, physics=diagnostic,
                  selected=selected, best_epoch=cp['epoch'], epochs=len(history), logged_seconds=sum(r['seconds'] for r in history),
                  checkpoint=str(folder/'best.pt'), peak_memory_gb=torch.cuda.max_memory_allocated()/1e9,
                  real_parameters={v:sum(p.numel()*(2 if p.is_complex() else 1) for p in net.parameters()) for v, net in [('head',h),('concentration',c)]})
    write(folder / 'result.json', result)
    write(folder / 'status.json', result)
    return result



def evaluation_adapter(h,c,d,phys,norm,interface,phase,weights,scales,backward=False):
    paths, loss = trajectory(h,c,d,norm)
    keys=['head_mse','concentration_mse','negative_penalty','water_raw','transport_raw']
    return paths, dict(zip(keys,loss.mean(0).tolist()))
