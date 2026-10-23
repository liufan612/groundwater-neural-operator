"""E1 head architecture queue: shared boundary-aware supervised rollout."""
import argparse,json,random,time,os,traceback
from pathlib import Path
import numpy as np
import torch
from known_recharge_v3 import install
from gwno.data.dataset import load_sequence,model_input
from gwno.data.water_cache import prescribed_chd_covariates
from gwno.data.prepare import write_json
from gwno.models.factory import build_operator
from gwno.training.trainer import seed_all,early_stop_update
from gwno.evaluation.fields import field_metrics

ROOT=Path(__file__).resolve().parent
MODELS={'unet':{'base':24},'fno':{'width':32,'depth':4,'modes1':8,'modes2':8},'afno':{'width':88,'depth':4}}

def chd_inputs(prepared,name,norm,device):
    saved=ROOT/'prepared_local_pde_v1'/f'{name}.pt'
    if saved.exists():
        return {k:v[:,0] for k,v in prescribed_chd_covariates(torch.load(saved,map_location='cpu',weights_only=False),norm['head'],device).items()}
    # Read only the prescribed CHD text already used to create this data.
    with np.load(prepared/'simulations'/f'{name}.npz') as z:
        path=Path(json.loads(str(z['metadata']))['raw_model_path'])/'flow.chd'
    periods={};period=None
    for line in path.read_text().splitlines():
        words=line.split()
        if not words or words[0].startswith('#'):continue
        if len(words)>=3 and words[0].lower()=='begin' and words[1].lower()=='period':
            period=int(words[2])-1;periods[period]=[]
        elif words[0].lower()=='end':period=None
        elif period is not None:
            layer,row,col=map(int,words[:3]);value=float(words[3])
            periods[period].append(((0 if layer==1 else 1,(row-1)//5,(col-1)//5),value))
    values=torch.zeros(30,2,30,50,device=device);mask=torch.zeros_like(values,dtype=torch.bool);current=[]
    for t in range(30):
        current=periods.get(t,current);group={}
        for cell,value in current:group.setdefault(cell,[]).append(value)
        for cell,rows in group.items():
            if not np.allclose(rows,rows[0]):raise ValueError('Different prescribed heads in one exported boundary cell')
            mask[(t,)+cell]=True;values[(t,)+cell]=(rows[0]-norm['head']['mean'][0])/norm['head']['std'][0]
    return dict(chd_values=values,chd_mask=mask)

def batch(sequences):
    return {k:torch.stack([s[k] for s in sequences],dim=1 if k in ('head','concentration','forcing','times_days','chd_mask','chd_values') else 0) for k in sequences[0]}

def rollout(model,d,backward=False):
    mask=d['active_mask'].bool();den=mask.sum((1,2,3));truth=d['head']
    h=torch.where(d['chd_mask'][0],d['chd_values'][0],truth[0])*mask
    paths=[h.detach()];total=torch.zeros(len(h),device=h.device);segment=[]
    for t in range(1,len(truth)):
        x=torch.cat([model_input(d,h,t),d['chd_values'][t]*d['chd_mask'][t]],dim=1)
        h=torch.where(d['chd_mask'][t],d['chd_values'][t],h+model(x))*mask
        loss=((h-truth[t]).square()*mask).sum((1,2,3))/den
        total+=loss.detach()/(len(truth)-1);paths.append(h.detach())
        if backward:segment.append(loss.mean()/(len(truth)-1))
        if t%5==0 or t==len(truth)-1:
            if backward:torch.stack(segment).sum().backward();segment=[]
            h=h.detach()
    return torch.stack(paths),total

def fit(model_name,lr,seed,args,train,val,norm,split,folder):
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():return json.loads((folder/'result.json').read_text())
    seed_all(seed);model=build_operator(model_name,36,2,MODELS[model_name]).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-6)
    best=reference=float('inf');wait=0;start_epoch=0;started=time.time()
    if (folder/'last.pt').exists():
        cp=torch.load(folder/'last.pt',map_location='cuda',weights_only=False)
        model.load_state_dict(cp['model']);opt.load_state_dict(cp['optimizer'])
        best,reference,wait,start_epoch=cp['best'],cp['reference'],cp['wait'],cp['epoch']+1
    status=dict(status='RUNNING',model=model_name,lr=lr,seed=seed,pid=os.getpid())
    write_json(folder/'status.json',status)
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch,args.epochs):
        if wait>=100:break
        tick=time.time();names=list(train);random.Random(seed+epoch).shuffle(names);losses=[];model.train()
        for offset in range(0,len(names),args.batch):
            d=batch([train[n] for n in names[offset:offset+args.batch]])
            opt.zero_grad(set_to_none=True);_,loss=rollout(model,d,True)
            if not torch.isfinite(loss).all() or not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('Nonfinite training loss/gradient')
            opt.step();losses.extend(loss.tolist())
        model.eval();errors=[]
        with torch.no_grad():
            vn=list(val)
            for offset in range(0,len(vn),args.batch):
                _,loss=rollout(model,batch([val[n] for n in vn[offset:offset+args.batch]]));errors.extend(loss.sqrt().tolist())
        score=float(np.mean(errors))
        if not np.isfinite(score):raise FloatingPointError('Nonfinite validation')
        improved=score<best;best=min(best,score);reference,wait=early_stop_update(score,reference,wait,1e-4)
        row=dict(epoch=epoch,train_mse=float(np.mean(losses)),val_nrmse=score,seconds=time.time()-tick)
        with (folder/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        cp=dict(model=model.state_dict(),optimizer=opt.state_dict(),model_name=model_name,model_kwargs=MODELS[model_name],input_channels=36,normalization=norm,split=split,epoch=epoch,best=best,reference=reference,wait=wait,lr=lr,seed=seed,boundary_input=True)
        torch.save(cp,folder/'last.pt')
        if improved:torch.save(cp,folder/'best.pt')
        status.update(row);write_json(folder/'status.json',status)
        print(json.dumps(dict(model=model_name,lr=lr,seed=seed,**row)),flush=True)
    cp=torch.load(folder/'best.pt',map_location='cuda',weights_only=False);model.load_state_dict(cp['model']);model.eval();rows=[]
    with torch.no_grad():
        for name,d in val.items():
            path,_=rollout(model,batch([d]));p=path[:,0].cpu().numpy();y=d['head'].cpu().numpy();m=d['active_mask'].cpu().numpy().astype(bool);s=norm['head']['std'][0];mu=norm['head']['mean'][0]
            metrics=field_metrics(p*s+mu,y*s+mu,m,s)
            persistence=field_metrics(np.repeat(y[:1],len(y),axis=0)*s+mu,y*s+mu,m,s)
            layers=[field_metrics(p[:,i:i+1]*s+mu,y[:,i:i+1]*s+mu,m[i:i+1],s) for i in range(2)]
            rows.append(dict(scenario=name,**metrics,layers=layers,persistence=persistence))
    write_json(folder/'val_metrics.json',rows)
    result=dict(status='DONE',model=model_name,lr=lr,seed=seed,best_epoch=cp['epoch'],val_nrmse=best,elapsed_seconds=time.time()-started,parameters=sum(p.numel() for p in model.parameters()),real_parameters=sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters()),peak_memory_gb=torch.cuda.max_memory_allocated()/1e9)
    write_json(folder/'result.json',result);write_json(folder/'status.json',result);return result
