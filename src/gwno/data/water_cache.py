"""Cache known water geometry and historical boundary inputs, without states."""
import numpy as np
import torch
import flopy
from ..physics.water import rectangular_water_geometry
from ..physics.water_boundaries import known_water_boundaries,coarsen_water_boundaries


def build_water_cache(model_path,frames=8,factor=5,include_transport=False):
    sim=flopy.mf6.MFSimulation.load(sim_ws=str(model_path),verbosity_level=0)
    f=sim.get_model('flow');shape=f.dis.idomain.array.shape
    def horizontal(a):
        a=np.asarray(a);return a.reshape(*a.shape[:-2],30,factor,50,factor).mean((-3,-1))
    def material(a):
        a=horizontal(a)
        if len(a)>2:
            # Source repair replicated the same material into lower sublayers.
            if not np.allclose(a[1:],a[1:2],equal_nan=True):raise ValueError('Nonuniform lower material needs upscaling')
            a=a[:2]
        return a
    bottom=horizontal(f.dis.botm.array)[[0,-1]];k=f.npf.k.array
    geometry=rectangular_water_geometry(horizontal(f.dis.top.array),bottom,material(f.dis.idomain.array),
        f.dis.delr.array.reshape(50,factor).sum(1),f.dis.delc.array.reshape(30,factor).sum(1),material(k),
        material(k if f.npf.k22.array is None else f.npf.k22.array),material(k if f.npf.k33.array is None else f.npf.k33.array),
        material(f.npf.icelltype.array),material(f.sto.ss.array))
    assert np.all(f.sto.iconvert.array==0)
    packets=[coarsen_water_boundaries(known_water_boundaries(f,p),shape,factor) for p in range(frames)]
    result=dict(geometry=geometry,packets=packets,model_path=str(model_path),frames=frames,
                contract='Known input only; lower identical materials merged; no simulated h/C/FLOW in cache')
    if include_transport:
        transport=sim.get_model('transport')
        sources=[tuple(str(v).lower() for v in r) for r in transport.ssm.sources.get_data()]
        if sources!=[('wel-1','aux','concentration')]:raise ValueError('Local PDE requires the current WEL-only injected solute contract')
        shape=bottom.shape
        dx=f.dis.delr.array.reshape(50,factor).sum(1)
        dy=f.dis.delc.array.reshape(30,factor).sum(1)
        geometry['dx']=torch.tensor(np.broadcast_to(dx,shape).copy().ravel(),dtype=torch.float64)
        geometry['dy']=torch.tensor(np.broadcast_to(dy[None,:,None],shape).copy().ravel(),dtype=torch.float64)
        geometry['porosity']=torch.tensor(material(transport.mst.porosity.array).ravel(),dtype=torch.float64)
        geometry['porosity']=torch.where(geometry['active'],geometry['porosity'],0.)
        result['transport']={}
        defaults={'alv':'alh','ath2':'ath1','atv':'ath2'}
        for key in ['alh','alv','ath1','ath2','atv','diffc']:
            value=getattr(transport.dsp,key).array
            if value is None and key in defaults:
                result['transport'][key]=result['transport'][defaults[key]].clone()
                continue
            if value is None and key=='diffc':value=0.
            if value is None:raise ValueError(f'Mechanical dispersion requires {key}')
            result['transport'][key]=torch.tensor(material(np.broadcast_to(value,f.dis.idomain.array.shape)).ravel(),dtype=torch.float64)
            result['transport'][key]=torch.where(geometry['active'],result['transport'][key],0.)
        result['shape']=shape
    return result


def prescribed_chd_covariates(cache,normalization,device):
    values=torch.zeros(len(cache['packets']),1,2,30,50,device=device)
    masks=torch.zeros_like(values,dtype=torch.bool)
    for frame,packet in enumerate(cache['packets']):
        cells=packet['chd']['cells'].to(device)
        masks[frame].view(-1)[cells]=True
        values[frame].view(-1)[cells]=((packet['chd']['head'].to(device)-normalization['mean'][0])/normalization['std'][0]).to(values.dtype)
    return dict(chd_mask=masks,chd_values=values)
