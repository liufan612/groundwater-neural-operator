"""Differentiable rectangular-grid Darcy water fluxes in m3/day.

Confined diagonal-K faces use half-cell hydraulic resistances. Convertible
horizontal faces use saturated thickness in those resistances: this is an
explicit approximation, not MF6 NEWTON upstream saturation or XT3D.
Vertical conductance uses full thickness (no VARIABLECV).
"""
import numpy as np
import torch
from .balance import outward_flux


def rectangular_water_geometry(top,bottom,active,dx,dy,kx,ky,kz,convertible,ss,device='cpu'):
    bottom=np.asarray(bottom);shape=bottom.shape
    upper=np.concatenate([np.asarray(top)[None],bottom[:-1]])
    thickness=upper-bottom
    widths=np.broadcast_to(np.asarray(dx)[None,None,:],shape)
    heights=np.broadcast_to(np.asarray(dy)[None,:,None],shape)
    ids=np.arange(bottom.size).reshape(shape);active=np.asarray(active)>0
    pieces=[]
    for axis,k,spacing,face_width in [(2,kx,widths,heights),(1,ky,heights,widths),(0,kz,thickness,widths*heights)]:
        left=[slice(None)]*3;right=left.copy();left[axis]=slice(None,-1);right[axis]=slice(1,None)
        left,right=tuple(left),tuple(right);keep=active[left]&active[right]
        k=np.broadcast_to(k,shape)
        # Horizontal resistance still needs saturated thickness at runtime.
        area=face_width if axis==0 else face_width*thickness
        resistance=spacing/(2*k*area)
        pieces.append((ids[left][keep],ids[right][keep],resistance[left][keep],resistance[right][keep],np.full(keep.sum(),axis)))
    def tensor(value,dtype=torch.float64):return torch.as_tensor(np.array(value,copy=True),dtype=dtype,device=device)
    result={name:tensor(np.concatenate([p[i] for p in pieces]),torch.long if i in (0,1,4) else torch.float64)
            for i,name in enumerate(['i','j','ri','rj','axis'])}
    result.update(bottom=tensor(np.where(active,bottom,0.).ravel()),thickness=tensor(np.where(active,thickness,1.).ravel()),
        convertible=tensor(np.asarray(convertible).ravel(),torch.bool),active=tensor(active.ravel(),torch.bool),
        storage_capacity=tensor(np.where(active,np.broadcast_to(ss,shape)*thickness*widths*heights,0.).ravel()))
    return result


def darcy_water_flux(head,geometry):
    """head [..., cells] in metres; positive edge flux i -> j."""
    g=geometry;i,j=g['i'],g['j']
    thickness=torch.where(g['active'],g['thickness'],1.)
    saturation=torch.where(g['convertible'],((head-g['bottom'])/thickness).clamp(0,1),1.)
    si=torch.where(g['axis']==0,1.,saturation[...,i])
    sj=torch.where(g['axis']==0,1.,saturation[...,j])
    # Algebraic form avoids division by zero at dry cells.
    denominator=g['ri']*sj+g['rj']*si
    conductance=si*sj/torch.where(denominator>0,denominator,1.)
    return conductance*(head[...,i]-head[...,j])


def boundary_water_input(head,well_rate,recharge_rate,area,river_stage,river_bottom,river_conductance,
                         evt_surface,evt_depth,evt_max_rate):
    """Dense prescribed cell arrays. Rates: WEL m3/d, RCHA/EVT m/d.

    Recharge is already routed to its receiving cell; EVT arrays refer to
    package cell locations. CHD is treated separately as a fixed-head boundary.
    """
    river=river_conductance*(river_stage-torch.maximum(head,river_bottom))
    evap=evt_max_rate*area*((head-(evt_surface-evt_depth))/evt_depth).clamp(0,1)
    return well_rate+recharge_rate*area+river-evap


def water_residual(previous_head,next_head,geometry,external_input,dt,chd_mask):
    """SS-only storage + outward water - input; return free-cell residual.

    Implied CHD exchange closes its local equation; it is not an observed flux.
    Caller must impose the known CHD head before evaluating edge fluxes.
    """
    storage=geometry['storage_capacity']*(next_head-previous_head)/dt
    outward=outward_flux(darcy_water_flux(next_head,geometry),geometry['i'],geometry['j'],next_head.shape[-1])
    balance=storage+outward-external_input
    return torch.where(chd_mask,0.,balance)*geometry['active'],torch.where(chd_mask,balance,0.)
