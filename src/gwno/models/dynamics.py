import torch
from ..data.dataset import model_input
from ..physics.interval import coarse_global_residual

def rollout(model, data, variable, head=None, residual=True, bptt_steps=5, backward=False,
            nonnegative=0., normalization=None, return_losses=False, global_mass=0.,local_pde=None,local_pde_weight=0.):
    """Backward per segment; optimizer step belongs AFTER the whole trajectory."""
    truth, mask = data[variable], data['active_mask'].bool()
    state = truth[0]
    batched = state.ndim == 4
    dims = (1,2,3) if batched else (0,1,2)
    denominator = mask.sum(dim=dims)
    outputs, segment = [state.detach()], []
    total = torch.zeros_like(denominator,dtype=truth.dtype)
    for t in range(1,len(truth)):
        previous=state
        prediction = model(model_input(data,state,t,None if head is None else head[t]))
        if not batched: prediction = prediction[0]
        state = (state+prediction if residual else prediction)*mask
        loss = ((state-truth[t]).square()*mask).sum(dim=dims)/denominator
        if nonnegative:
            stat = normalization[variable]
            raw_scaled = state + stat['mean'][0]/stat['std'][0]
            loss = loss + nonnegative*(torch.relu(-raw_scaled).square()*mask).sum(dim=dims)/denominator
        if global_mass:
            if variable!='concentration':
                raise ValueError('Global solute balance applies only to concentration')
            stat=normalization[variable]
            before=(previous*stat['std'][0]+stat['mean'][0])*mask
            after=(state*stat['std'][0]+stat['mean'][0])*mask
            before=before.flatten(start_dim=1 if batched else 0)
            after=after.flatten(start_dim=1 if batched else 0)
            residual_mass=coarse_global_residual(before,after,*[
                data[k][t-1] for k in ['mass_left','mass_right','mass_inflow','mass_outflow']])
            loss=loss+global_mass*residual_mass.square()
        if local_pde_weight:
            if variable!='concentration' or head is None or local_pde is None:
                raise ValueError('Local transport PDE requires concentration, predicted head and known-input callbacks')
            stat=normalization['concentration'];hs=normalization['head']
            before=(previous*stat['std'][0]+stat['mean'][0])*mask
            after=(state*stat['std'][0]+stat['mean'][0])*mask
            h0=(head[t-1]*hs['std'][0]+hs['mean'][0])*mask
            h1=(head[t]*hs['std'][0]+hs['mean'][0])*mask
            if batched:
                local=torch.stack([callback(before[b],after[b],h0[b],h1[b],t,
                    data['times_days'][t,b]-data['times_days'][t-1,b]) for b,callback in enumerate(local_pde)])
            else:local=local_pde[0](before,after,h0,h1,t,data['times_days'][t]-data['times_days'][t-1])
            loss=loss+local_pde_weight*local
        segment.append(loss)
        total += loss.detach()/(len(truth)-1)
        outputs.append(state.detach())
        if len(segment) == bptt_steps or t == len(truth)-1:
            if backward:
                (torch.stack(segment).mean(dim=-1).sum()/(len(truth)-1) if batched else
                 torch.stack(segment).sum()/(len(truth)-1)).backward()
            state = state.detach()
            segment = []
    return torch.stack(outputs), total if return_losses else total.mean().item()
