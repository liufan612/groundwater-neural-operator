"""E1 concentration search, seed confirmation, and top-two head cross-over."""
import argparse,json,os,random,time,traceback
from pathlib import Path
import numpy as np
import torch
from run_e1_head_v3 import ROOT,batch,chd_inputs,rollout as head_rollout
from known_recharge_v3 import install
from gwno.data.dataset import load_sequence,model_input
from gwno.data.prepare import write_json
from gwno.models.factory import build_operator
from gwno.training.trainer import seed_all,early_stop_update
from gwno.evaluation.fields import field_metrics

CAPACITIES={'unet':{'base':32},'fno':{'width':32,'depth':4,'modes1':8,'modes2':8},'afno':{'width':88,'depth':4}}



def load_head(item,seed,data,norm):
    path=ROOT/'runs_e1_v3/head_comparison'/item['model']/f'lr_{item["lr"]}'/f'seed_{seed}'/'best.pt'
    cp=torch.load(path,map_location='cuda',weights_only=False)
    assert cp['normalization']==norm
    model=build_operator(cp['model_name'],36,2,cp['model_kwargs']).cuda();model.load_state_dict(cp['model']);model.eval()
    paths={}
    with torch.no_grad():
        names=list(data)
        for offset in range(0,len(names),8):
            keys=names[offset:offset+8];p,_=head_rollout(model,batch([data[n] for n in keys]))
            for i,name in enumerate(keys):paths[name]=p[:,i].clone()
    del model
    return paths,str(path)

def rollout(model,d,heads,norm,backward=False):
    mask=d['active_mask'].bool();den=mask.sum((1,2,3));truth=d['concentration'];c=truth[0]
    paths=[c.detach()];mse=torch.zeros(len(c),device=c.device);total=mse.clone();segment=[]
    for t in range(1,len(truth)):
        x=torch.cat([model_input(d,c,t,heads[t]),d['chd_values'][t]*d['chd_mask'][t]],dim=1)
        c=(c+model(x))*mask
        ld=((c-truth[t]).square()*mask).sum((1,2,3))/den
        ln=(torch.relu(-(c+norm['concentration']['mean'][0]/norm['concentration']['std'][0])).square()*mask).sum((1,2,3))/den
        loss=ld+.02*ln
        mse+=ld.detach()/(len(truth)-1);total+=loss.detach()/(len(truth)-1);paths.append(c.detach())
        if backward:segment.append(loss.mean()/(len(truth)-1))
        if t%5==0 or t==len(truth)-1:
            if backward:torch.stack(segment).sum().backward();segment=[]
            c=c.detach()
    return torch.stack(paths),mse,total

def validation(model,val,heads,norm,bs):
    errors=[];model.eval()
    with torch.no_grad():
        names=list(val)
        for i in range(0,len(names),bs):
            keys=names[i:i+bs];d=batch([val[n] for n in keys]);h=torch.stack([heads[n] for n in keys],1)
            _,cm,_=rollout(model,d,h,norm)
            hm=(((h[1:]-d['head'][1:]).square()*d['active_mask']).sum((2,3,4))/d['active_mask'].sum((1,2,3))).mean(0)
            errors.extend(torch.stack([hm,cm],1).tolist())
    a=np.asarray(errors)
    return dict(system_mse=float(a.mean()),head_nrmse=float(np.sqrt(a[:,0]).mean()),concentration_nrmse=float(np.sqrt(a[:,1]).mean()))

def metrics(model,val,heads,norm):
    rows=[];model.eval()
    with torch.no_grad():
        for name,d in val.items():
            path,_,_=rollout(model,batch([d]),heads[name][:,None],norm)
            for variable,arr in [('head',heads[name]),('concentration',path[:,0])]:
                st=norm[variable];p=arr.cpu().numpy()*st['std'][0]+st['mean'][0];y=d[variable].cpu().numpy()*st['std'][0]+st['mean'][0];m=d['active_mask'].cpu().numpy().astype(bool)
                main=field_metrics(p,y,m,st['std'][0])
                layer=[field_metrics(p[:,j:j+1],y[:,j:j+1],m[j:j+1],st['std'][0]) for j in range(2)]
                negative_amplitude=float(np.maximum(-p[1:,m],0).mean())
                rows.append(dict(scenario=name,variable=variable,**main,layers=layer,negative_amplitude=negative_amplitude))
    return rows

def fit(cname,lr,seed,item,args,data,split,norm,heads,head_path,out):
    folder=out/f'head_{item["model"]}'/cname/f'lr_{lr}'/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():return json.loads((folder/'result.json').read_text())
    seed_all(seed+1000);model=build_operator(cname,38,2,CAPACITIES[cname]).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-6)
    best=reference=float('inf');wait=0;start_epoch=0;old_seconds=0.;started=time.time()
    if (folder/'last.pt').exists():
        cp=torch.load(folder/'last.pt',map_location='cuda',weights_only=False)
        assert cp['normalization']==norm and cp['head_checkpoint']==head_path
        model.load_state_dict(cp['model']);opt.load_state_dict(cp['optimizer']);best,reference,wait,start_epoch=cp['best'],cp['reference'],cp['wait'],cp['epoch']+1;old_seconds=cp.get('elapsed_seconds',0.)
    train={n:data[n] for n in split['train']};val={n:data[n] for n in split['val']}
    state=dict(status='RUNNING',pid=os.getpid(),head=item['model'],concentration=cname,lr=lr,seed=seed);write_json(folder/'status.json',state)
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch,args.epochs):
        if wait>=100:break
        tick=time.time();names=list(train);random.Random(seed+epoch).shuffle(names);losses=[];model.train()
        for i in range(0,len(names),args.batch):
            keys=names[i:i+args.batch];d=batch([train[n] for n in keys]);h=torch.stack([heads[n] for n in keys],1)
            opt.zero_grad(set_to_none=True);_,_,loss=rollout(model,d,h,norm,True)
            if not torch.isfinite(loss).all() or not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('Nonfinite concentration training')
            opt.step();losses.extend(loss.tolist())
        v=validation(model,val,heads,norm,args.batch);score=v['system_mse']
        if not np.isfinite(score):raise FloatingPointError('Nonfinite concentration validation')
        improved=score<best;best=min(score,best)
        # Early-stop tolerance remains in concentration NRMSE units as in E1 h.
        reference,wait=early_stop_update(v['concentration_nrmse'],reference,wait,1e-4)
        row=dict(epoch=epoch,train_loss=float(np.mean(losses)),validation=v,seconds=time.time()-tick)
        with (folder/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        cp=dict(model=model.state_dict(),optimizer=opt.state_dict(),model_name=cname,model_kwargs=CAPACITIES[cname],input_channels=38,hard_nonnegative=bool(getattr(args,"hard_nonnegative",False)),normalization=norm,split=split,head_checkpoint=head_path,epoch=epoch,best=best,reference=reference,wait=wait,seed=seed,lr=lr,elapsed_seconds=old_seconds+time.time()-started)
        torch.save(cp,folder/'last.pt')
        if improved:torch.save(cp,folder/'best.pt')
        state.update(row);write_json(folder/'status.json',state);print(json.dumps(dict(head=item['model'],C=cname,seed=seed,lr=lr,**row)),flush=True)
    selected=torch.load(folder/'best.pt',map_location='cuda',weights_only=False);model.load_state_dict(selected['model'])
    rows=metrics(model,val,heads,norm);write_json(folder/'val_metrics.json',rows)
    summary={var:{key:float(np.mean([r[key] for r in rows if r['variable']==var])) for key in ['r2','rmse','nrmse','late_nrmse','negative_fraction']} for var in ['head','concentration']}
    result=dict(status='DONE',hard_nonnegative=bool(getattr(args,"hard_nonnegative",False)),head=item['model'],concentration=cname,lr=lr,seed=seed,best_epoch=selected['epoch'],system_mse=best,metrics=summary,checkpoint=str(folder/'best.pt'),elapsed_seconds=old_seconds+time.time()-started,real_parameters=sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters()),peak_memory_gb=torch.cuda.max_memory_allocated()/1e9)
    write_json(folder/'result.json',result);write_json(folder/'status.json',result);return result
