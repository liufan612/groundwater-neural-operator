"""Matched-batch coupling/PDE experiments; full truth-free source rollouts."""
import json,random,time
from pathlib import Path
import numpy as np
import torch
from run_e1_head_v3 import ROOT,batch,MODELS
from run_e1_concentration_v3 import CAPACITIES
from e23_physics import load as load_physics,face_channels
from gwno.data.dataset import model_input
from gwno.models.factory import build_operator
from gwno.physics.local_water_pde import LocalWaterPDE
from gwno.physics.local_pde import LocalTransportPDE
from gwno.training.trainer import seed_all
from gwno.evaluation.fields import field_metrics
from gwno.physics.water import darcy_water_flux

def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2),encoding='utf-8')

def networks(system,interface,seed,device='cuda'):
    seed_all(seed)
    h=build_operator(system['head'],36,2,MODELS[system['head']]).to(device)
    # Independent deterministic C initialization: invariant to head implementation.
    seed_all(seed+10000)
    c=build_operator(system['concentration'],{'H':38,'Q':60,'parallel':36}[interface.removesuffix('_detach')],2,CAPACITIES[system['concentration']]).to(device)
    return h,c

def callbacks(prepared,names,norm,device='cuda'):
    result={}
    for name in names:
        cache=load_physics(prepared,name)
        w=LocalWaterPDE(cache['geometry'],cache['packets'],norm['head']['std'][0],device)
        c=LocalTransportPDE(cache,norm['concentration']['std'][0],device)
        result[name]=(w,c)
    return result

def trajectory(hnet,cnet,d,phys,norm,interface,phase,weights,scales,backward=False):
    detach_input=interface.endswith('_detach');interface=interface.removesuffix('_detach')
    mask=d['active_mask'].bool();den=mask.sum((1,2,3));n=len(d['head'])-1
    h=torch.where(d['chd_mask'][0],d['chd_values'][0],d['head'][0])*mask;c=d['concentration'][0]
    paths={'head':[h.detach()],'concentration':[c.detach()]};total=np.zeros(5);segment=[]
    for t in range(1,n+1):
        boundary=d['chd_values'][t]*d['chd_mask'][t]
        hx=torch.cat([model_input(d,h,t),boundary],1)
        if phase=='concentration':
            with torch.no_grad():hn=torch.where(d['chd_mask'][t],d['chd_values'][t],h+hnet(hx))*mask
        else:hn=torch.where(d['chd_mask'][t],d['chd_values'][t],h+hnet(hx))*mask
        cn=c
        if phase!='head':
            condition=hn.detach() if detach_input else hn
            cx=torch.cat([model_input(d,c,t,condition if interface=='H' else None),boundary],1)
            if interface=='Q':
                q=torch.stack([face_channels(condition[i]*norm['head']['std'][0]+norm['head']['mean'][0],p[0].g,scales['q']) for i,p in enumerate(phys)])
                cx=torch.cat([cx,q],1)
            cn=(c+cnet(cx))*mask
        lh=(((hn-d['head'][t]).square()*mask).sum((1,2,3))/den).mean()
        lc=(((cn-d['concentration'][t]).square()*mask).sum((1,2,3))/den).mean()
        ln=((torch.relu(-(cn+norm['concentration']['mean'][0]/norm['concentration']['std'][0])).square()*mask).sum((1,2,3))/den).mean()
        lw=lh.new_zeros(());lp=lh.new_zeros(())
        if weights[0] and phase!='concentration' or weights[1] and phase!='head':
            hs=norm['head'];cs=norm['concentration']
            for i,(water,transport) in enumerate(phys):
                h0,h1=[x[i]*hs['std'][0]+hs['mean'][0] for x in (h,hn)]
                dt=d['times_days'][t,i]-d['times_days'][t-1,i]
                if weights[0] and phase!='concentration':lw=lw+water(h0,h1,t,dt)/len(phys)
                if weights[1] and phase!='head':
                    c0,c1=[x[i]*cs['std'][0]+cs['mean'][0] for x in (c,cn)]
                    lp=lp+transport(c0,c1,h0,h1,t,dt)/len(phys)
        loss=(lh if phase=='head' else lc+.02*ln+weights[1]*lp/scales['c'])
        if phase=='joint':loss=loss+lh
        if phase!='concentration':loss=loss+weights[0]*lw/scales['h']
        if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite {phase} frame {t+1}')
        total+=np.array([float(x.detach()) for x in (lh,lc,ln,lw,lp)])/n
        if backward:segment.append(loss/n)
        paths['head'].append(hn.detach());paths['concentration'].append(cn.detach());h,c=hn,cn
        if t%5==0 or t==n:
            if backward:torch.stack(segment).sum().backward();segment=[]
            h,c=h.detach(),c.detach()
    return {k:torch.stack(v) for k,v in paths.items()},dict(zip(['head_mse','concentration_mse','negative_penalty','water_raw','transport_raw'],total.tolist()))

def evaluate(h,c,data,phys,norm,interface,scales,phase='joint',details=False):
    rows=[];h.eval();c.eval()
    with torch.no_grad():
        for name,d in data.items():
            paths,loss=trajectory(h,c,batch([d]),[phys[name]],norm,interface,phase,(0,0),scales)
            row=dict(scenario=name,**loss)
            if details:
                for variable in ['head','concentration']:
                    st=norm[variable];p=paths[variable][:,0].cpu().numpy()*st['std'][0]+st['mean'][0]
                    y=d[variable].cpu().numpy()*st['std'][0]+st['mean'][0];m=d['active_mask'].cpu().numpy().astype(bool)
                    row[variable]=dict(overall=field_metrics(p,y,m,st['std'][0]),late=field_metrics(p[19:],y[19:],m,st['std'][0]),
                        layers=[field_metrics(p[:,i:i+1],y[:,i:i+1],m[i:i+1],st['std'][0]) for i in range(2)])
                    if variable=='concentration':row['negative_fraction']=float((p[1:,m]<0).mean())
            rows.append(row)
    hm=float(np.mean([r['head_mse'] for r in rows]));cm=float(np.mean([r['concentration_mse'] for r in rows]))
    return dict(head_mse=hm,concentration_mse=cm,system_mse=(hm+cm)/2,rows=rows)

def train_phase(h,c,system,interface,phase,seed,train,val,phys,norm,weights,scales,epochs,folder,bs=2,time_limit=None):
    folder.mkdir(parents=True,exist_ok=True)
    parameters=[dict(params=h.parameters(),lr=system['head_lr'])] if phase=='head' else [dict(params=c.parameters(),lr=system['concentration_lr'])]
    if phase=='joint':parameters.insert(0,dict(params=h.parameters(),lr=system['head_lr']))
    opt=torch.optim.AdamW(parameters,weight_decay=1e-6);best=float('inf');started=time.time();start=0
    if (folder/'last.pt').exists():
        cp=torch.load(folder/'last.pt',weights_only=False,map_location='cuda');h.load_state_dict(cp['head']);c.load_state_dict(cp['concentration']);opt.load_state_dict(cp['optimizer']);start=cp['epoch']+1;best=cp['best']
    used=sum(json.loads(line)['seconds'] for line in (folder/'history.jsonl').read_text().splitlines()) if (folder/'history.jsonl').exists() else 0.
    for epoch in range(start,epochs):
        if time_limit is not None and used>=time_limit:break
        tick=time.time();h.train();c.train();names=list(train);random.Random(seed+epoch).shuffle(names);losses=[]
        for offset in range(0,len(names),bs):
            keys=names[offset:offset+bs];h.zero_grad(set_to_none=True);c.zero_grad(set_to_none=True)
            _,loss=trajectory(h,c,batch([train[n] for n in keys]),[phys[n] for n in keys],norm,interface,phase,weights,scales,True)
            if not all(p.grad is None or torch.isfinite(p.grad).all() for net in (h,c) for p in net.parameters()):raise FloatingPointError('Nonfinite gradient')
            opt.step();losses.append(loss)
        ev=evaluate(h,c,val,phys,norm,interface,scales,phase)
        score=ev['head_mse'] if phase=='head' else ev['system_mse'];improved=score<best;best=min(best,score)
        cp=dict(head=h.state_dict(),concentration=c.state_dict(),optimizer=opt.state_dict(),epoch=epoch,best=best)
        torch.save(cp,folder/'last.pt')
        if improved:torch.save(cp,folder/'best.pt')
        row=dict(epoch=epoch,validation={k:v for k,v in ev.items() if k!='rows'},train={k:float(np.mean([r[k] for r in losses])) for k in losses[0]},seconds=time.time()-tick)
        used+=row['seconds']
        with (folder/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        write(folder/'status.json',dict(status='RUNNING',phase=phase,**row));print(json.dumps(dict(folder=str(folder),**row)),flush=True)
    final=evaluate(h,c,val,phys,norm,interface,scales,phase,True);write(folder/'final_metrics.json',final)
    cp=torch.load(folder/'best.pt',weights_only=False,map_location='cuda');h.load_state_dict(cp['head']);c.load_state_dict(cp['concentration'])
    selected=evaluate(h,c,val,phys,norm,interface,scales,phase,True);write(folder/'selected_metrics.json',selected)
    completed=len((folder/'history.jsonl').read_text().splitlines())
    result=dict(status='DONE',best_epoch=cp['epoch'],epochs=completed,epoch_limit=epochs,seconds=time.time()-started,training_seconds=used,time_limit=time_limit,selected={k:v for k,v in selected.items() if k!='rows'},final={k:v for k,v in final.items() if k!='rows'})
    write(folder/'result.json',result);write(folder/'status.json',result);return result

def run(system,interface,schedule,seed,train,val,phys,norm,weights,scales,epochs,folder,time_limit=None):
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():return json.loads((folder/'result.json').read_text())
    write(folder/'protocol.json',dict(system=system,interface=interface,schedule=schedule,seed=seed,weights=weights,scales=scales,epochs_per_network=epochs,total_training_time_limit=time_limit,batch=2,bptt=5,train=list(train),validation=list(val),test_used=False,normalization=norm))
    h,c=networks(system,interface,seed);phases=['head','concentration'] if schedule=='sequential' else ['joint'];results={}
    for phase in phases:results[phase]=train_phase(h,c,system,interface,phase,seed,train,val,phys,norm,weights,scales,epochs,folder/phase,time_limit=None if time_limit is None else time_limit/len(phases))
    diagnostic=physics_metrics(h,c,val,phys,norm,interface,scales,folder)
    result=dict(interface=interface,schedule=schedule,seed=seed,weights=weights,physics=diagnostic,**results[phases[-1]])
    result['training_seconds']=sum(r['training_seconds'] for r in results.values())
    result['phase_results']=results
    write(folder/'result.json',result);return result

def physics_metrics(h,c,data,phys,norm,interface,scales,folder):
    rows=[];h.eval();c.eval()
    with torch.no_grad():
        for name,d in data.items():
            paths,_=trajectory(h,c,batch([d]),[phys[name]],norm,interface,'joint',(0,0),scales)
            hp=paths['head'][:,0]*norm['head']['std'][0]+norm['head']['mean'][0]
            cp=paths['concentration'][:,0]*norm['concentration']['std'][0]+norm['concentration']['mean'][0]
            ht=d['head']*norm['head']['std'][0]+norm['head']['mean'][0]
            w,tr=phys[name];g=w.g;records=[]
            for t in range(1,len(hp)):
                dt=d['times_days'][t]-d['times_days'][t-1]
                qp=darcy_water_flux(hp[t].flatten().double(),g);qt=darcy_water_flux(ht[t].flatten().double(),g)
                gradp=hp[t].flatten()[g['j']]-hp[t].flatten()[g['i']]
                gradt=ht[t].flatten()[g['j']]-ht[t].flatten()[g['i']]
                vertical=g['axis']==0
                records.append(dict(frame=t+1,water_residual=float(w(hp[t-1],hp[t],t,dt)),transport_residual=float(tr(cp[t-1],cp[t],hp[t-1],hp[t],t,dt)),
                    approximate_flux_mse=float((qp-qt).square().mean()),vertical_flux_mse=float((qp[vertical]-qt[vertical]).square().mean()),head_difference_mse=float((gradp-gradt).square().mean())))
            rows.append(dict(scenario=name,frames=records))
            np.savez_compressed(folder/f'prediction_{name}.npz',head=hp.cpu().numpy(),concentration=cp.cpu().numpy(),active_mask=d['active_mask'].cpu().numpy())
    summary={k:float(np.mean([f[k] for r in rows for f in r['frames']])) for k in rows[0]['frames'][0] if k!='frame'}
    write(folder/'physical_diagnostics.json',dict(summary=summary,scenarios=rows,interpretation='Approximate local discretization; flux reference uses validation h and the same Darcy operator, not observed solver flux or independent mass closure'))
    return summary
