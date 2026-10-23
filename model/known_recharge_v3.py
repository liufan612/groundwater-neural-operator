"""Shared recharge covariate from prescribed RCHA arrays, in m/day."""
import json,re
from pathlib import Path
import numpy as np
import torch

def recharge(prepared,name,mask):
    with np.load(prepared/'simulations'/f'{name}.npz') as z:
        path=Path(json.loads(str(z['metadata']))['raw_model_path'])/'flow.rcha'
    content=path.read_text()
    if re.search(r'\b(irch|fixed_cell)\b',content,re.I):
        raise ValueError('Recharge receiver override requires explicit handling')
    periods={}
    for number,body in re.findall(r'BEGIN\s+period\s+(\d+)(.*?)END\s+period',content,re.I|re.S):
        match=re.search(r'recharge\s+INTERNAL\s+FACTOR\s+([\d.Ee+-]+)\s*\n(.*)',body,re.I|re.S)
        if match is None:raise ValueError(f'Unsupported recharge array in {path}')
        a=np.fromstring(match[2],sep=' ')
        if a.size!=150*250:raise ValueError(f'Recharge shape {a.size}')
        periods[int(number)-1]=a.reshape(30,5,50,5).mean((1,3))*float(match[1])
    active=mask.detach().cpu().numpy().astype(bool)
    result=np.zeros((30,2,30,50),np.float32);current=None
    for t in range(30):
        current=periods.get(t,current)
        if current is None:raise ValueError('Missing initial recharge')
        result[t,0]=current*active[0]
        result[t,1]=current*(~active[0]&active[1])
    return torch.as_tensor(result,device=mask.device)

def install(data,train_names,prepared,norm,fit):
    raw={n:recharge(prepared,n,d['active_mask']) for n,d in data.items()}
    if fit:
        values=torch.cat([raw[n][:,data[n]['active_mask'].bool()].flatten() for n in train_names])
        norm['forcing']['mean'][0]=float(values.mean())
        norm['forcing']['std'][0]=float(values.std(unbiased=False).clamp_min(1e-12))
    for n,d in data.items():
        d['forcing'][:,:,0]=(raw[n]-norm['forcing']['mean'][0])/norm['forcing']['std'][0]*d['active_mask']
