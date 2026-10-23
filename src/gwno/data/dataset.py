import numpy as np
import torch

def load_sequence(prepared, name, normalization, device, physics_dir=None):
    with np.load(prepared/'simulations'/f'{name}.npz') as z:
        data = {k:torch.as_tensor(z[k].copy(), device=device, dtype=torch.float32)
                for k in ['head','concentration','static','forcing','active_mask','times_days']}
    if physics_dir is not None:
        from pathlib import Path
        with np.load(Path(physics_dir)/f'{name}.npz') as z:
            if not np.array_equal(z['times_days'],data['times_days'].cpu().numpy()) or not np.all(z['mass_steps']==18):
                raise ValueError('Weak balance requires aligned saved times and 18 internal steps')
            data.update({k:torch.as_tensor(z[k].copy(),device=device,dtype=torch.float32)
                         for k in ['mass_left','mass_right','mass_inflow','mass_outflow']})
    mask = data['active_mask']
    for key in ['head','concentration','static','forcing']:
        stat = normalization[key]
        mean, std = [torch.tensor(stat[k], device=device) for k in ['mean','std']]
        if key == 'static':
            mean, std = mean[None,:,None,None], std[None,:,None,None]
            m = mask[:,None]
        elif key == 'forcing':
            mean, std = mean[None,None,:,None,None], std[None,None,:,None,None]
            m = mask[None,:,None]
        else:
            m = mask
        data[key] = (data[key]-mean)/std*m
    return data

def model_input(data, previous, t, head=None):
    """State at t-1 plus forcing of destination period t; no future target field."""
    h,w = previous.shape[-2:]
    if previous.ndim == 4:
        b = len(previous)
        items = [data['static'].reshape(b,-1,h,w), data['forcing'][t].reshape(b,-1,h,w),
                 data['active_mask'], previous]
        if head is not None:
            items.append(head)
        return torch.cat(items, dim=1)
    items = [data['static'].reshape(-1,h,w), data['forcing'][t].reshape(-1,h,w),
             data['active_mask'], previous]
    if head is not None:
        items.append(head)
    return torch.cat(items).unsqueeze(0)

def stack_sequences(sequences):
    """Time-major trajectory, batch-major static fields; retain each scenario mask."""
    return {key:torch.stack([s[key] for s in sequences],dim=1 if key in
            ['head','concentration','forcing','times_days','mass_left','mass_right',
             'mass_inflow','mass_outflow'] else 0) for key in sequences[0]}
