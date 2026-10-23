"""E1-matched sequential PDE ablation, shared head checkpoints, fused physics."""
import argparse,json,os,time,traceback,threading,subprocess
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import run_e1_head_v3 as H
import run_e1_concentration_v3 as C
from e23_physics import load as physics_cache
from pde_batch import FusedPDE
from gwno.models.factory import build_operator
from gwno.training.trainer import seed_all

ROOT=H.ROOT;OUT=ROOT/'runs_pde_ablation_e1';ORIGINAL_H=H.rollout;ORIGINAL_C=C.rollout
CACHES={};NORM={};WH=0.;WC=0.;SCALES={}

def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def move(x):
    if torch.is_tensor(x):return x.cuda()
    if isinstance(x,dict):return {k:move(v) for k,v in x.items()}
    if isinstance(x,list):return [move(v) for v in x]
    return x
def fused(d):return FusedPDE([CACHES[i] for i in d['scenario_index'].tolist()],NORM)

def head_rollout(model,d,backward=False):
    if not backward or not WH:return ORIGINAL_H(model,d,backward)
    pde=fused(d);mask=d['active_mask'].bool();den=mask.sum((1,2,3));truth=d['head']
    h=torch.where(d['chd_mask'][0],d['chd_values'][0],truth[0])*mask
    paths=[h.detach()];total=torch.zeros(len(h),device=h.device);segment=[]
    for t in range(1,len(truth)):
        x=torch.cat([H.model_input(d,h,t),d['chd_values'][t]*d['chd_mask'][t]],1)
        hn=torch.where(d['chd_mask'][t],d['chd_values'][t],h+model(x))*mask
        loss=((hn-truth[t]).square()*mask).sum((1,2,3))/den
        hs=NORM['head'];raw=pde.water(h*hs['std'][0]+hs['mean'][0],hn*hs['std'][0]+hs['mean'][0],t,d['times_days'][t,0]-d['times_days'][t-1,0])
        loss=loss+WH*raw/SCALES['h'];total+=loss.detach()/(len(truth)-1);paths.append(hn.detach());segment.append(loss.mean()/(len(truth)-1));h=hn
        if t%5==0 or t==len(truth)-1:torch.stack(segment).sum().backward();segment=[];h=h.detach()
    return torch.stack(paths),total

def concentration_rollout(model,d,heads,norm,backward=False):
    if not backward or not WC:return ORIGINAL_C(model,d,heads,norm,backward)
    pde=fused(d);mask=d['active_mask'].bool();den=mask.sum((1,2,3));truth=d['concentration'];c=truth[0]
    paths=[c.detach()];mse=torch.zeros(len(c),device=c.device);total=mse.clone();segment=[];hs=norm['head'];cs=norm['concentration']
    for t in range(1,len(truth)):
        x=torch.cat([C.model_input(d,c,t,heads[t]),d['chd_values'][t]*d['chd_mask'][t]],1)
        cn=(c+model(x))*mask
        ld=((cn-truth[t]).square()*mask).sum((1,2,3))/den
        ln=(torch.relu(-(cn+cs['mean'][0]/cs['std'][0])).square()*mask).sum((1,2,3))/den
        raw=pde.transport(c*cs['std'][0]+cs['mean'][0],cn*cs['std'][0]+cs['mean'][0],heads[t-1]*hs['std'][0]+hs['mean'][0],heads[t]*hs['std'][0]+hs['mean'][0],t,d['times_days'][t,0]-d['times_days'][t-1,0])
        loss=ld+.02*ln+WC*raw/SCALES['c'];mse+=ld.detach()/(len(truth)-1);total+=loss.detach()/(len(truth)-1);paths.append(cn.detach());segment.append(loss.mean()/(len(truth)-1));c=cn
        if t%5==0 or t==len(truth)-1:torch.stack(segment).sum().backward();segment=[];c=c.detach()
    return torch.stack(paths),mse,total

def load_heads(path,data):
    cp=torch.load(path,map_location='cuda',weights_only=False);assert cp['normalization']==NORM
    net=build_operator('afno',36,2,H.MODELS['afno']).cuda();net.load_state_dict(cp['model']);net.eval();paths={}
    with torch.no_grad():
        names=list(data)
        for start in range(0,len(names),8):
            keys=names[start:start+8];p,_=ORIGINAL_H(net,H.batch([data[n] for n in keys]))
            for i,n in enumerate(keys):paths[n]=p[:,i].clone()
    return paths
