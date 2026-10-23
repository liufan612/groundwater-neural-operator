"""Disconnected-graph batch fusion, preserving per-scenario residual means."""
import torch
from gwno.physics.local_water_pde import LocalWaterPDE
from gwno.physics.local_pde import LocalTransportPDE,local_terms
from gwno.physics.water import water_residual
from gwno.physics.water_boundaries import impose_chd,prescribed_boundary_fluxes

def merge(caches):
    n=caches[0]['geometry']['active'].numel();b=len(caches)
    g={k:torch.cat([c['geometry'][k]+i*n if k in ['i','j'] else c['geometry'][k] for i,c in enumerate(caches)]) for k in caches[0]['geometry']}
    material={k:torch.cat([c['transport'][k] for c in caches]) for k in caches[0]['transport']}
    packets=[]
    for t in range(30):
        p=dict(size=b*n,active=g['active'])
        for name in ['wel','riv','evt','rch','chd']:
            p[name]={k:torch.cat([c['packets'][t][name][k]+i*n if k=='cells' else c['packets'][t][name][k] for i,c in enumerate(caches)]) for k in caches[0]['packets'][t][name]}
        packets.append(p)
    return dict(geometry=g,transport=material,packets=packets),b,n

class FusedPDE:
    def __init__(self,caches,norm):
        cache,self.b,self.n=merge(caches)
        self.w=LocalWaterPDE(cache['geometry'],cache['packets'],norm['head']['std'][0],'cuda')
        self.c=LocalTransportPDE(cache,norm['concentration']['std'][0],'cuda')
    def water(self,h0,h1,t,duration):
        w=self.w;g=w.g;p=w.packets[t];h0,h1=[x.reshape(-1).double() for x in (h0,h1)]
        h0=impose_chd(h0,w.packets[t-1]);h1=impose_chd(h1,p)
        a=torch.arange(18,device=h0.device,dtype=h0.dtype)[:,None]/18
        before=h0+(h1-h0)*a;after=impose_chd(h0+(h1-h0)*(a+1/18),p)
        chd=torch.zeros_like(g['active']);chd[p['chd']['cells']]=True
        residual,_=water_residual(before,after,g,sum(prescribed_boundary_fluxes(after,p).values()),duration/18,chd)
        scale=(g['storage_capacity']/duration+w.diagonal)*w.scale;valid=g['active']&~chd&(scale>0)
        values=torch.where(valid,(residual/scale.clamp_min(1e-30)).square(),0.).reshape(18,self.b,self.n)
        return (values.sum(-1)/valid.reshape(self.b,self.n).sum(-1)).mean(0)
    def transport(self,c0,c1,h0,h1,t,duration):
        c=self.c;c0,c1,h0,h1=[x.reshape(-1).double() for x in (c0,c1,h0,h1)]
        h0=impose_chd(h0,c.packets[t-1]);h1=impose_chd(h1,c.packets[t])
        a=torch.arange(18,device=c0.device,dtype=c0.dtype)[:,None]/18;b=a+1/18
        terms=local_terms(c0+(c1-c0)*a,c0+(c1-c0)*b,h0+(h1-h0)*a,h0+(h1-h0)*b,c.g,c.material,c.packets[t],duration/18)
        volume=terms['volume'];valid=c.g['active']&(volume>0)
        squared=(terms['residual']/torch.where(valid,volume,1.)/(c.scale/duration)).square()
        values=torch.where(valid,squared,0.).reshape(18,self.b,self.n)
        return (values.sum(-1)/valid.reshape(18,self.b,self.n).sum(-1)).mean(0)
