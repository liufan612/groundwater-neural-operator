import csv
import copy
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import yaml
from ..data.dataset import load_sequence, model_input, stack_sequences
from ..data.prepare import write_json,STATIC,FORCING
from ..models.factory import build_operator
from ..models.dynamics import rollout
from ..evaluation.fields import field_metrics

def validate_resume_config(previous, current):
    old,new = copy.deepcopy(previous),copy.deepcopy(current)
    if new['training']['epochs'] < old['training']['epochs']:
        raise ValueError('Resume cannot reduce the epoch limit')
    for key in ['resume','epochs','early_stopping_patience','early_stopping_min_delta']:
        old['training'].pop(key,None); new['training'].pop(key,None)
    if old != new:
        raise ValueError('Resume may extend epochs/change early stopping only; model, data and optimizer must match')

def early_stop_update(score, reference, wait, min_delta):
    if score < reference-min_delta:
        return score,0
    return reference,wait+1

def update_run_index(root):
    fields=['experiment_id','block','model','seed','split_path','observation_set','config_path',
            'status','start_time','end_time','device','checkpoint_path','metrics_path',
            'best_epoch','validation_metric','gpu_hours','peak_memory_gb','error_message']
    rows=[]
    for path in sorted(Path(root).glob('*/*/seed_*/run_summary.json')):
        s=json.loads(path.read_text())
        rows.append({k:s.get(k,'') for k in fields})
    with (Path(root)/'run_index.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setup(config):
    prepared = Path(config['data']['prepared_dir'])
    split = json.loads((prepared/'splits/d0_development.json').read_text())
    norm = json.loads((prepared/'normalization.json').read_text())
    device = config.get('device','cuda' if torch.cuda.is_available() else 'cpu')
    return prepared,split,norm,device

def load_checkpoint(path, device):
    checkpoint = torch.load(path,map_location=device,weights_only=False)
    model = build_operator(checkpoint['model_name'],checkpoint['input_channels'],2,checkpoint['model_kwargs']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model,checkpoint

def train(config, variable='head', seed=42, train_names=None, val_names=None):
    seed_all(seed)
    prepared,split,norm,device = setup(config)
    cfg = config['training']
    batch_size = cfg['batch_size']
    if device.startswith('cuda'):
        torch.backends.cudnn.benchmark = True
    name = 'unet' if variable == 'head' else config['model']['name']
    kwargs = config['head_model'] if variable == 'head' else config['model']['kwargs']
    train_names = split['train'] if train_names is None else train_names
    val_names = split['val'] if val_names is None else val_names
    first = load_sequence(prepared,train_names[0],norm,device)
    channels = model_input(first,first[variable][0],1,first['head'][1] if variable=='concentration' else None).shape[1]
    model = build_operator(name,channels,2,kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    head_model = None
    head_path = None
    if variable == 'concentration':
        head_path = config['head']['checkpoint'].format(seed=seed)
        head_model,hcp = load_checkpoint(head_path,device)
        if hcp['normalization'] != norm or hcp['split'] != split:
            raise ValueError('Head checkpoint uses different data split or normalization')
        for p in head_model.parameters():
            p.requires_grad_(False)
    output = Path(config['runs_dir'])/config['experiment_id']/name/f'seed_{seed}'
    output.mkdir(parents=True,exist_ok=True)
    if (output/'last.pt').exists() and not cfg.get('resume',False):
        raise ValueError(f'Run already exists; use a new experiment_id or training.resume: {output}')
    best,wait,start_epoch = float('inf'),0,0
    stop_reference,best_epoch = float('inf'),None
    if cfg.get('resume',False):
        cp = torch.load(output/'last.pt',map_location=device,weights_only=False)
        validate_resume_config(cp['config'],config)
        if cp['normalization'] != norm or cp['split'] != split:
            raise ValueError('Resume data differ from checkpoint')
        model.load_state_dict(cp['model']); optimizer.load_state_dict(cp['optimizer'])
        start_epoch,best,wait = cp['epoch']+1,cp['best'],cp['wait']
        stop_reference=cp.get('stop_reference',best)
        if any(cp['config']['training'].get(k) != cfg.get(k) for k in
               ['early_stopping_patience','early_stopping_min_delta']):
            stop_reference,wait=best,0
        best_epoch=cp.get('best_epoch')
        if best_epoch is None:
            best_epoch=torch.load(output/'best.pt',map_location='cpu',weights_only=False)['epoch']
        random.setstate(cp['random_state']); torch.set_rng_state(cp['torch_rng'].cpu())
        if device.startswith('cuda'): torch.cuda.set_rng_state_all([state.cpu() for state in cp['cuda_rng']])
        if start_epoch>=cfg['epochs'] or wait>=cfg['early_stopping_patience']:
            return output
    (output/'resolved_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    started = time.time()
    if device.startswith('cuda'): torch.cuda.reset_peak_memory_stats()
    summary = dict(status='RUNNING',variable=variable,model=name,seed=seed,device=device,
                   checkpoint_path=str(output/'best.pt'),start_time=started,
                   experiment_id=config['experiment_id'],block=('B1' if config['experiment_id'].startswith('b1') else 'B2' if config['experiment_id'].startswith('b2') else 'M0_M1'),
                   split_path=str(prepared/'splits/d0_development.json'),observation_set='full_first_saved_frame',
                   config_path=str(output/'resolved_config.yaml'),metrics_path=str(output/'train_metrics.csv'))
    summary.update(resumed_from_epoch=start_epoch-1 if start_epoch else None,best_epoch=best_epoch,
                   epoch_limit=cfg['epochs'],early_stopping_patience=cfg['early_stopping_patience'],
                   early_stopping_min_delta=cfg.get('early_stopping_min_delta',0.))
    write_json(output/'run_summary.json',summary)
    update_run_index(config['runs_dir'])
    head_cache = {}
    physics_dir=config['data'].get('weak_balance_dir') if config['loss'].get('global_mass',0.) and variable=='concentration' else None
    # This 100-scenario dataset fits comfortably in device memory; read once per run.
    sequence_cache = {scenario:load_sequence(prepared,scenario,norm,device,physics_dir=physics_dir)
                      for scenario in dict.fromkeys(train_names+val_names)}
    local_weight=config['loss'].get('concentration_physics',0.) if variable=='concentration' else 0.
    local_callbacks={}
    if local_weight:
        from ..physics.local_pde import LocalTransportPDE
        for scenario in train_names:
            cache=torch.load(Path(config['data']['local_pde_dir'])/f'{scenario}.pt',map_location='cpu',weights_only=False)
            local_callbacks[scenario]=LocalTransportPDE(cache,norm['concentration']['std'][0],device)
    def head_for(scenario,data):
        if head_model is None: return None
        if scenario not in head_cache:
            with torch.no_grad():
                head_cache[scenario] = rollout(head_model,data,'head',residual=True)[0]
        return head_cache[scenario]
    def batch_for(names):
        sequences = [sequence_cache[n] for n in names]
        data = stack_sequences(sequences)
        heads = None if head_model is None else torch.stack(
            [head_for(n,s) for n,s in zip(names,sequences)],dim=1)
        return data,heads
    try:
        with (output/'train_metrics.csv').open('a' if start_epoch else 'w',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=['epoch','train_loss','val_scenario_nrmse','seconds'])
            if not start_epoch: writer.writeheader()
            for epoch in range(start_epoch,cfg['epochs']):
                epoch_start = time.perf_counter()
                order = list(train_names); random.shuffle(order)
                losses = []
                model.train()
                for offset in range(0,len(order),batch_size):
                    names = order[offset:offset+batch_size]
                    data,heads = batch_for(names)
                    optimizer.zero_grad(set_to_none=True)
                    _,loss = rollout(model,data,variable,heads,
                                     residual=config['model'].get('residual_update',True) if variable!='head' else True,
                                     bptt_steps=cfg['bptt_steps'],backward=True,
                                     nonnegative=config['loss']['nonnegative'] if variable=='concentration' else 0.,normalization=norm,
                                     global_mass=(config['loss'].get('global_mass',0.)*min(1.,(epoch+1)/config['loss'].get('global_mass_warmup',1)) if variable=='concentration' else 0.),
                                     local_pde=[local_callbacks[n] for n in names] if local_weight else None,
                                     local_pde_weight=local_weight,
                                     return_losses=True)
                    finite = torch.stack([torch.isfinite(loss).all()]+[
                        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all()
                    if not finite.item():
                        raise FloatingPointError('Nonfinite trajectory loss/gradient')
                    optimizer.step(); losses.append(loss)
                model.eval(); errors=[]
                with torch.no_grad():
                    for offset in range(0,len(val_names),batch_size):
                        data,heads=batch_for(val_names[offset:offset+batch_size])
                        _,loss=rollout(model,data,variable,heads,residual=True if variable=='head' else config['model'].get('residual_update',True),return_losses=True)
                        errors.append(loss.sqrt())
                score=torch.cat(errors).mean().item()
                if not np.isfinite(score): raise FloatingPointError('Nonfinite validation score')
                improved=score<best
                if improved: best,best_epoch=score,epoch
                stop_reference,wait=early_stop_update(score,stop_reference,wait,cfg.get('early_stopping_min_delta',0.))
                row=dict(epoch=epoch,train_loss=torch.cat(losses).mean().item(),val_scenario_nrmse=score,seconds=time.perf_counter()-epoch_start)
                writer.writerow(row); f.flush()
                checkpoint=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,best=best,wait=wait,
                                stop_reference=stop_reference,best_epoch=best_epoch,
                                model_name=name,model_kwargs=kwargs,input_channels=channels,normalization=norm,split=split,
                                variable_order=dict(static=STATIC,forcing=FORCING,layers='MODFLOW_top_to_bottom',
                                                    inputs=['static','forcing','active_mask','previous_state']+(['predicted_head'] if variable=='concentration' else [])),
                                split_path=str(prepared/'splits/d0_development.json'),config=config,variable=variable,
                                initialization='full_first_saved_frame',time_step_days=180,head_checkpoint=head_path,
                                random_state=random.getstate(),torch_rng=torch.get_rng_state(),
                                cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else [])
                torch.save(checkpoint,output/'last.pt')
                if improved:
                    torch.save(checkpoint,output/'best.pt')
                    summary['best_epoch']=epoch
                print(f'{variable} {name} seed={seed} {row}',flush=True)
                if wait>=cfg['early_stopping_patience']: break
        summary.update(status='DONE',best_validation_nrmse=best,validation_metric=best,last_epoch=epoch,
                       stop_reason='early_stopping' if wait>=cfg['early_stopping_patience'] else 'epoch_limit',
                       early_stopping_wait=wait)
    except Exception as exc:
        summary.update(status='FAILED',error_message=str(exc)); raise
    finally:
        summary.update(end_time=time.time(),elapsed_seconds=time.time()-started,
                       gpu_hours=(time.time()-started)/3600 if device.startswith('cuda') else 0.,
                       peak_memory_gb=torch.cuda.max_memory_allocated()/1e9 if device.startswith('cuda') else 0.)
        write_json(output/'run_summary.json',summary)
        update_run_index(config['runs_dir'])
    return output

def evaluate(config, checkpoint_path, subset='val'):
    prepared,split,norm,device=setup(config)
    model,cp=load_checkpoint(checkpoint_path,device)
    if cp['normalization'] != norm or cp['split'] != split:
        raise ValueError('Evaluation data differ from checkpoint')
    head_model = load_checkpoint(cp['head_checkpoint'],device)[0] if cp['head_checkpoint'] else None
    rows=[]
    with torch.no_grad():
        for scenario in split[subset]:
            data=load_sequence(prepared,scenario,norm,device)
            head=rollout(head_model,data,'head')[0] if head_model else None
            pred,_=rollout(model,data,cp['variable'],head,residual=True if cp['variable']=='head' else cp['config']['model'].get('residual_update',True))
            stat=norm[cp['variable']]
            p=pred.cpu().numpy()*stat['std'][0]+stat['mean'][0]
            y=data[cp['variable']].cpu().numpy()*stat['std'][0]+stat['mean'][0]
            rows.append(dict(scenario=scenario,**field_metrics(p,y,data['active_mask'].cpu().numpy().astype(bool),stat['std'][0])))
    output=Path(checkpoint_path).parent/f'{subset}_metrics.json'
    write_json(output,rows)
    return rows
