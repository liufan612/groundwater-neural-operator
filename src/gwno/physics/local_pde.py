"""Cellwise FV transport PDE on the exported grid; not MF6 XT3D reconstruction.

Full dispersion tensor, shared face fluxes, SS-only MST storage, known sources.
Grid-oriented xyz uses increasing column/row/layer; q and gradients share axes.
"""
import torch
from .water import darcy_water_flux
from .water_boundaries import impose_chd,prescribed_boundary_fluxes
from .balance import outward_flux,advective_mass_flux,modflow_mst_storage_rate
from .tensor import modflow_dispersion_tensor


def face_geometry(head,g):
    i,j=g['i'],g['j'];axis=2-g['axis']
    sat=torch.where(g['convertible'],(head-g['bottom']).clamp_min(0).minimum(g['thickness']),g['thickness'])
    lengths=torch.stack([g['dx'],g['dy'],g['thickness']],-1)
    distance=(lengths[i,axis]+lengths[j,axis])/2
    horizontal_sat=(sat[...,i]+sat[...,j])/2
    area=torch.where(axis==0,(g['dy'][i]+g['dy'][j])/2*horizontal_sat,
        torch.where(axis==1,(g['dx'][i]+g['dx'][j])/2*horizontal_sat,g['dx'][i]*g['dy'][i]))
    return axis,distance,area,sat


def cell_vectors(edge_values,axis,i,j,n):
    """Average incident face components; one-sided at domain/mask boundaries."""
    result=edge_values.new_zeros(*edge_values.shape[:-1],n,3);counts=edge_values.new_zeros(n,3)
    for a in range(3):
        keep=axis==a
        for cells in [i[keep],j[keep]]:
            result[...,a].index_add_(-1,cells,edge_values[...,keep])
            counts[:,a].index_add_(0,cells,torch.ones_like(axis[keep],dtype=edge_values.dtype))
    return result/counts.clamp_min(1)


def local_terms(c0,c1,h0,h1,g,material,packet,dt):
    """Return mass/time terms per cell, including both sides of layer exchange."""
    i,j=g['i'],g['j'];n=c1.shape[-1]
    h1=impose_chd(h1,packet)
    axis,distance,area,sat=face_geometry(h1,g)
    q=darcy_water_flux(h1,g)
    specific=cell_vectors(q/area.clamp_min(1e-20),axis,i,j,n)
    tensor=modflow_dispersion_tensor(specific,g['porosity'],material['alh'],material['ath1'],
        material['alv'],material['ath2'],material['atv'],material['diffc'])
    normal_gradient=(c1[...,j]-c1[...,i])/distance
    gradient=cell_vectors(normal_gradient,axis,i,j,n)
    face_gradient=(gradient[...,i,:]+gradient[...,j,:])/2
    face_gradient=face_gradient.scatter(-1,axis[:,None].expand(*normal_gradient.shape,1),normal_gradient[...,None])
    face_tensor=(tensor[...,i,:,:]+tensor[...,j,:,:])/2
    dispersion=-area*(face_tensor[...,torch.arange(len(i),device=i.device),axis,:]*face_gradient).sum(-1)
    advection=advective_mass_flux(q,c1,i,j)
    rates=prescribed_boundary_fluxes(h1,packet)
    chd_mask=torch.zeros_like(g['active']);chd_mask[packet['chd']['cells']]=True
    water_storage=-g['storage_capacity']*(h1-h0)/dt*~chd_mask
    # Known-head boundary supply implied by the same local water discretization.
    chd_input=torch.where(chd_mask,outward_flux(q,i,j,n)-sum(rates.values()),0.)
    sinks=sum((-rate).clamp_min(0) for rate in rates.values())+(-chd_input).clamp_min(0)
    wel=packet['wel'];source=c1.new_zeros(n)
    source.index_add_(0,wel['cells'],wel['q'].clamp_min(0)*wel['concentration'])
    external=source-sinks*c1.clamp_min(0)
    volume=g['porosity']*g['dx']*g['dy']*sat*g['active']
    storage=modflow_mst_storage_rate(c0,c1,volume,water_storage,dt)
    adv=outward_flux(advection,i,j,n);dsp=outward_flux(dispersion,i,j,n)
    return dict(storage=storage,advection=adv,dispersion=dsp,external=external,
                residual=(storage+adv+dsp-external)*g['active'],volume=volume)


def local_residual_loss(residual,volume,active,concentration_scale,dt):
    """Square each cell before averaging. Units reduced to C per day first."""
    valid=active&(volume>0)
    density=residual[valid]/volume[valid]
    return (density/(concentration_scale/dt)).square().mean()


class LocalTransportPDE:
    def __init__(self,cache,concentration_scale,device):
        def move(value):
            if torch.is_tensor(value):return value.to(device)
            if isinstance(value,dict):return {k:move(v) for k,v in value.items()}
            if isinstance(value,list):return [move(v) for v in value]
            return value
        cache=move(cache);self.g=cache['geometry'];self.material=cache['transport'];self.packets=cache['packets']
        self.scale=concentration_scale
    def reference(self,c0,c1,h0,h1,t,duration):
        c0,c1,h0,h1=[x.reshape(-1).double() for x in [c0,c1,h0,h1]]
        h0=impose_chd(h0,self.packets[t-1]);h1=impose_chd(h1,self.packets[t])
        # Average squared LOCAL substep residuals, never square an integrated sum.
        losses=[];dt=duration/18
        for step in range(18):
            a=step/18;b=(step+1)/18
            terms=local_terms(c0+(c1-c0)*a,c0+(c1-c0)*b,h0+(h1-h0)*a,h0+(h1-h0)*b,
                self.g,self.material,self.packets[t],dt)
            losses.append(local_residual_loss(terms['residual'],terms['volume'],self.g['active'],self.scale,duration))
        return torch.stack(losses).mean()

    def __call__(self,c0,c1,h0,h1,t,duration):
        c0,c1,h0,h1=[x.reshape(-1).double() for x in [c0,c1,h0,h1]]
        h0=impose_chd(h0,self.packets[t-1]);h1=impose_chd(h1,self.packets[t])
        a=torch.arange(18,device=c0.device,dtype=c0.dtype)[:,None]/18
        b=a+1/18
        terms=local_terms(c0+(c1-c0)*a,c0+(c1-c0)*b,h0+(h1-h0)*a,h0+(h1-h0)*b,
            self.g,self.material,self.packets[t],duration/18)
        volume=terms['volume'];valid=self.g['active']&(volume>0)
        density=terms['residual']/torch.where(valid,volume,1.)
        squared=(density/(self.scale/duration)).square()
        return (torch.where(valid,squared,0.).sum(-1)/valid.sum(-1)).mean()
