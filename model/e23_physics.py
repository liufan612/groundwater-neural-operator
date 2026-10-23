"""Known-input geometry and compact boundary records for E2/E3 residuals."""
import json,re
from pathlib import Path
import numpy as np
import torch
from known_recharge_v3 import recharge
from gwno.physics.water import rectangular_water_geometry,darcy_water_flux
from gwno.physics.water_boundaries import prescribed_boundary_fluxes

ROOT=Path(__file__).resolve().parent

def compress(packet):
    p=dict(packet)
    # Combine records only when their nonlinear boundary parameters coincide.
    for name,keys,weight in [('evt',['cells','surface','rate','depth'],'area'),
                              ('riv',['cells','stage','rbot'],'cond')]:
        old=p[name];a=np.stack([old[k].numpy() for k in keys],1)
        unique,inverse=np.unique(a,axis=0,return_inverse=True)
        values=np.bincount(inverse,weights=old[weight].numpy())
        p[name]={k:torch.as_tensor(unique[:,i],dtype=torch.long if k=='cells' else torch.float64) for i,k in enumerate(keys)}
        p[name][weight]=torch.as_tensor(values,dtype=torch.float64)
    old=p['rch'];cells,inverse=torch.unique(old['cells'],return_inverse=True)
    q=torch.zeros(len(cells),dtype=torch.float64);q.index_add_(0,inverse,old['q'])
    p['rch']=dict(cells=cells,q=q)
    return p

def read_list(path,fields):
    periods={}
    for number,body in re.findall(r'BEGIN\s+period\s+(\d+)(.*?)END\s+period',path.read_text(),re.I|re.S):
        a=np.fromstring(body,sep=' ').reshape(-1,3+len(fields))
        cells=(a[:,0]>1).astype(int)*1500+((a[:,1].astype(int)-1)//5)*50+(a[:,2].astype(int)-1)//5
        record={k:torch.as_tensor(a[:,i+3].copy(),dtype=torch.float64) for i,k in enumerate(fields)}
        record.update(cells=torch.as_tensor(cells),area=torch.full((len(cells),),400.,dtype=torch.float64))
        if fields==['head']:
            u,inv=torch.unique(record['cells'],return_inverse=True)
            heads=torch.zeros(len(u),dtype=torch.float64);counts=torch.zeros_like(heads)
            heads.index_add_(0,inv,record['head']);counts.index_add_(0,inv,torch.ones_like(inv,dtype=torch.float64));heads/=counts
            assert torch.allclose(heads[inv],record['head'],atol=1e-8,rtol=0)
            record=dict(cells=u,head=heads)
        periods[int(number)-1]=record
    return [periods[max(k for k in periods if k<=t)] for t in range(30)]

def build(prepared,name):
    old=ROOT/'prepared_local_pde_v1'/f'{name}.pt'
    if old.exists():cache=torch.load(old,weights_only=False,map_location='cpu')
    else:
        with np.load(prepared/'simulations'/f'{name}.npz') as z:
            static=z['static'].copy();active=z['active_mask'].copy().astype(bool)
            model=Path(json.loads(str(z['metadata']))['raw_model_path'])
        k=static[:,0];bottom=static[:,2];top=static[0,1]
        shape=active.shape;convertible=np.broadcast_to(np.array([True,False])[:,None,None],shape)
        # Common source-library constitutive assumptions; no solver loading.
        g=rectangular_water_geometry(top,bottom,active,np.full(50,100.),np.full(30,100.),k,k,k,convertible,np.full(shape,1e-5))
        g.update(dx=torch.full((3000,),100.,dtype=torch.float64),dy=torch.full((3000,),100.,dtype=torch.float64),
                 porosity=torch.as_tensor((static[:,3]*active).ravel(),dtype=torch.float64))
        lists={p:read_list(model/f'flow.{p}',fields) for p,fields in [('wel',['q','concentration']),('riv',['stage','cond','rbot']),('evt',['surface','rate','depth']),('chd',['head'])]}
        rch=recharge(prepared,name,torch.as_tensor(active)).double().reshape(30,-1)*10000
        cells=torch.where(torch.as_tensor(active).flatten())[0]
        packets=[dict(size=3000,active=g['active'],rch=dict(cells=cells,q=rch[t,cells]),**{p:rows[t] for p,rows in lists.items()}) for t in range(30)]
        material={key:torch.full((3000,),value,dtype=torch.float64)*g['active'] for key,value in dict(alh=40.,alv=40.,ath1=4.,ath2=4.,atv=.1,diffc=0.).items()}
        cache=dict(geometry=g,packets=packets,transport=material,shape=shape,contract='Approximate source-library coefficients plus known prescribed inputs; no simulator output')
    cache['packets']=[compress(p) for p in cache['packets']]
    return cache

def load(prepared,name):
    path=ROOT/'prepared_e23_v3'/f'{name}.pt'
    if not path.exists():
        path.parent.mkdir(exist_ok=True);torch.save(build(prepared,name),path)
    return torch.load(path,weights_only=False,map_location='cpu')

def face_channels(head,g,scale):
    """12 channels/layer: +/- x,y,z outward flux then six face masks."""
    q=darcy_water_flux(head.reshape(-1).double(),g).to(head.dtype)/scale
    n=head.numel();out=head.new_zeros(n*6);mask=head.new_zeros(n*6)
    direction=(2-g['axis'])*2
    first=g['i']*6+direction;second=g['j']*6+direction+1
    out.index_add_(0,first,q);out.index_add_(0,second,-q)
    mask[first]=1;mask[second]=1
    return torch.cat([out.reshape(2,30,50,6).permute(0,3,1,2).reshape(12,30,50),mask.reshape(2,30,50,6).permute(0,3,1,2).reshape(12,30,50)])
