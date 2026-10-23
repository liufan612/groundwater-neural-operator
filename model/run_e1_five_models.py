"""Complete five-model screening; fixed protocol and validation-only selection."""
import argparse,json,os,time,traceback
from types import SimpleNamespace
import torch
import run_e1_head_v3 as H
import run_e1_concentration_v3 as C
from nonnegative_state import project_concentration

ROOT=H.ROOT
OUT=ROOT/'runs_e1_five_models_20260915'
EXTRA={'uno':{'width':24,'modes1':8,'modes2':8},'deeponet':{'branch_width':112,'trunk_dim':256}}
H.MODELS.update(EXTRA);C.CAPACITIES.update(EXTRA)

def hard_rollout(model,d,heads,norm,backward=False):
    mask=d['active_mask'].bool();den=mask.sum((1,2,3));truth=d['concentration']
    c=project_concentration(truth[0],mask,norm)
    paths=[c.detach()];mse=torch.zeros(len(c),device=c.device);segment=[]
    for t in range(1,len(truth)):
        x=torch.cat([C.model_input(d,c,t,heads[t]),d['chd_values'][t]*d['chd_mask'][t]],dim=1)
        c=project_concentration(c+model(x),mask,norm)
        loss=((c-truth[t]).square()*mask).sum((1,2,3))/den
        mse+=loss.detach()/(len(truth)-1);paths.append(c.detach())
        if backward:segment.append(loss.mean()/(len(truth)-1))
        if t%5==0 or t==len(truth)-1:
            if backward:torch.stack(segment).sum().backward();segment=[]
            c=c.detach()
    return torch.stack(paths),mse,mse

C.rollout=hard_rollout
