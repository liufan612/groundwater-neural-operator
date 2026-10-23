"""Known MF6 boundary inputs for differentiable water diagnostics.

Reads package input arrays only, never head files or simulated budgets.
Supports the current NEWTON structured grid, one-segment EVT and array RCH.
"""
import numpy as np
import torch


def inherited(data,period):
    keys=[key for key in data if key<=period]
    if not keys:raise ValueError('Missing initial boundary period')
    return data[max(keys)]


def known_water_boundaries(flow,period,device='cpu'):
    shape=flow.dis.idomain.array.shape;active=flow.dis.idomain.array>0
    ids=np.arange(np.prod(shape)).reshape(shape)
    area=flow.dis.delc.array[:,None]*flow.dis.delr.array[None,:]
    def tensor(a,dtype=torch.float64):return torch.as_tensor(np.array(a,copy=True),dtype=dtype,device=device)
    packet=dict(size=int(ids.size),active=tensor(active.ravel(),torch.bool))
    for name in ['wel','riv','evt','chd']:
        package=flow.get_package(name)
        if package is None:raise ValueError(f'Current water contract requires {name}')
        records=inherited(package.stress_period_data.get_data(),period)
        cells=np.array([tuple(row['cellid']) for row in records],dtype=int)
        fields={field:tensor(records[field]) for field in records.dtype.names if field!='cellid'}
        fields['cells']=tensor(ids[tuple(cells.T)],torch.long)
        fields['area']=tensor(area[cells[:,1],cells[:,2]])
        packet[name]=fields
        if name=='evt' and package.nseg.array not in (None,1):raise ValueError('Multi-segment EVT is not implemented')
    recharge=flow.get_package('RCHA-1')
    if recharge is None:raise ValueError('Array recharge package RCHA-1 was not loaded')
    rates=inherited(recharge.recharge.get_data(),period)
    irch_data=recharge.irch.get_data()
    start=np.zeros(shape[1:],dtype=int) if irch_data is None else np.asarray(inherited(irch_data,period),dtype=int)
    receiver=np.full(shape[1:],-1,dtype=int)
    for layer in range(shape[0]):
        take=(receiver<0)&active[layer]&(layer>=start)
        if recharge.fixed_cell.array:take&=(layer==start)
        receiver[take]=layer
    rr,cc=np.where(receiver>=0);ll=receiver[rr,cc]
    packet['rch']=dict(cells=tensor(ids[ll,rr,cc],torch.long),q=tensor((rates*area)[rr,cc]))
    return packet


def impose_chd(head,packet):
    result=head.clone();result[...,packet['chd']['cells']]=packet['chd']['head']
    return result


def coarsen_water_boundaries(packet,fine_shape,factor):
    """Retain each prescribed source record while mapping its receiving cell.

    RIV conductance and EVT area stay at record scale and are summed by the
    flux assembler. No averaging of nonlinear boundary parameters occurs.
    """
    layers,ny,nx=fine_shape;cy,cx=ny//factor,nx//factor
    def remap(cells):
        layer=cells//(ny*nx);row=(cells//nx)%ny;col=cells%nx
        if layers>2:layer=torch.where(layer>0,1,0)
        return layer*(cy*cx)+(row//factor)*cx+col//factor
    active=packet['active'].reshape(layers,cy,factor,cx,factor)
    if not torch.equal(active.any(2).any(3),active.all(2).all(3)):
        raise ValueError('Partial coarse active cells require a separate geometry contract')
    coarse_active=active.any(2).any(3)
    if layers>2:
        if not torch.equal(coarse_active[1:],coarse_active[1:2].expand_as(coarse_active[1:])):
            raise ValueError('Lower subdivisions must share the same active support')
        coarse_active=coarse_active[:2]
    result=dict(size=coarse_active.numel(),active=coarse_active.reshape(-1))
    for name in ['wel','riv','evt','rch','chd']:
        result[name]=dict(packet[name]);result[name]['cells']=remap(packet[name]['cells'])
    cells=result['chd']['cells'];values=result['chd']['head']
    unique,inverse=torch.unique(cells,return_inverse=True)
    heads=values.new_zeros(len(unique));counts=values.new_zeros(len(unique))
    heads.index_add_(0,inverse,values);counts.index_add_(0,inverse,torch.ones_like(values));heads/=counts
    if not torch.allclose(heads[inverse],values,atol=1e-10,rtol=0):
        raise ValueError('Different prescribed CHD heads occur within a coarse cell')
    result['chd']=dict(cells=unique,head=heads)
    return result


def prescribed_boundary_fluxes(head,packet):
    """Signed input m3/d per cell, without implied CHD supply/removal."""
    def assemble(cells,q):
        values=head.new_zeros(head.shape);values.index_add_(-1,cells,q.expand(*head.shape[:-1],len(cells)))
        return values
    wel,riv,evt,rch=[packet[k] for k in ['wel','riv','evt','rch']]
    h=head[...,riv['cells']]
    river=riv['cond']*(riv['stage']-torch.maximum(h,riv['rbot']))
    h=head[...,evt['cells']]
    evaporation=-evt['rate']*evt['area']*((h-(evt['surface']-evt['depth']))/evt['depth']).clamp(0,1)
    evap=assemble(evt['cells'],evaporation);recharge=assemble(rch['cells'],rch['q'])
    # RCH and EVT do not apply to prescribed-head cells in this MF6 contract.
    evap[...,packet['chd']['cells']]=0.;recharge[...,packet['chd']['cells']]=0.
    return dict(wel=assemble(wel['cells'],wel['q']),riv=assemble(riv['cells'],river),evt=evap,rch=recharge)


def integrate_prescribed_water(previous_head,next_head,packets,step_days):
    """m3 per cell, linear endpoint h with implicit-step right-end quadrature.

    packets contain known inputs at each substep. This approximates the
    unknown intermediate trajectory; it is not a new groundwater solve.
    """
    durations=previous_head.new_tensor(step_days)
    alpha=durations.cumsum(0)/durations.sum()
    if len(packets)!=len(durations) or not torch.all(durations>0):
        raise ValueError('One known packet per positive-duration step required')
    integrated={}
    for weight,dt,packet in zip(alpha,durations,packets):
        head=impose_chd((1-weight)*previous_head+weight*next_head,packet)
        for name,rate in prescribed_boundary_fluxes(head,packet).items():
            integrated[name]=integrated.get(name,0)+dt*rate
    return integrated
