"""Approximate local head residual from existing physical inputs and predictions.

No simulator calls or simulated FLOW budgets. Shared-face Darcy fluxes plus
storage/source balance; interpolation is quadrature, not an intermediate solve.
"""
import torch
from .water import water_residual
from .water_boundaries import impose_chd, prescribed_boundary_fluxes


class LocalWaterPDE:
    def __init__(self, geometry, packets, head_scale, device):
        def move(x):
            if torch.is_tensor(x): return x.to(device)
            if isinstance(x, dict): return {k: move(v) for k,v in x.items()}
            if isinstance(x, list): return [move(v) for v in x]
            return x
        self.g = move(geometry)
        self.packets = move(packets)
        self.scale = head_scale
        # Fixed geometric conductance scale: units m2/day. It does not depend
        # on a learned head, so the model cannot reduce loss by changing scale.
        g = self.g
        edge = 1/(g['ri']+g['rj'])
        self.diagonal = torch.zeros_like(g['storage_capacity'])
        self.diagonal.index_add_(0,g['i'],edge)
        self.diagonal.index_add_(0,g['j'],edge)

    def __call__(self, h0, h1, t, duration):
        h0,h1 = [h.reshape(-1).double() for h in (h0,h1)]
        g = self.g; packet = self.packets[t]
        h0 = impose_chd(h0,self.packets[t-1])
        h1 = impose_chd(h1,packet)
        a = torch.arange(18,device=h0.device,dtype=h0.dtype)[:,None]/18
        before = h0+(h1-h0)*a
        after = impose_chd(h0+(h1-h0)*(a+1/18),packet)
        chd = torch.zeros_like(g['active']); chd[packet['chd']['cells']] = True
        external = sum(prescribed_boundary_fluxes(after,packet).values())
        residual,_ = water_residual(before,after,g,external,duration/18,chd)
        rate_scale = (g['storage_capacity']/duration+self.diagonal)*self.scale
        valid = g['active'] & ~chd & (rate_scale>0)
        return (residual[...,valid]/rate_scale[valid]).square().mean()
