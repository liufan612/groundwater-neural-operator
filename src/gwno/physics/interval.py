"""Weak MST integral for an explicitly linear-in-time concentration path.

Water quantities retain the internal flow time steps. This is an approximation
to the concentration trajectory, not an exact condensed MF6 transport solver.
Coefficients depend on water/geometry only, never on concentration labels.
"""
import torch


def linear_mst_coefficients(volume_next, storage_water_rate, times, cell_to_coarse=None,
                            n_coarse=None):
    """Return endpoint coefficients of the integrated MST mass change.

    volume_next and storage_water_rate: [steps, ..., fine_cells].
    times: [steps+1] including the interval's initial time, in days.
    storage_water_rate: volumetric rate into cell; zero in CHD cells.
    Optional cell_to_coarse prolongs each coarse concentration uniformly to
    its fine cells. Internal intercell fluxes cancel only in a GLOBAL sum.
    Per-cell values here are storage terms, not local conservation residuals.
    """
    dt=times[1:]-times[:-1]
    if volume_next.shape != storage_water_rate.shape or len(dt)!=len(volume_next):
        raise ValueError('Water arrays and internal times must align')
    if not torch.all(dt>0):
        raise ValueError('Internal times must increase')
    alpha=(times-times[0])/(times[-1]-times[0])
    shape=(-1,)+(1,)*(volume_next.ndim-1)
    old=alpha[:-1].reshape(shape)
    delta=(alpha[1:]-alpha[:-1]).reshape(shape)
    water_mass=storage_water_rate*dt.reshape(shape)
    left=(-volume_next*delta-water_mass*(1-old)).sum(dim=0)
    right=(volume_next*delta-water_mass*old).sum(dim=0)
    if cell_to_coarse is not None:
        def aggregate(x):
            result=x.new_zeros((*x.shape[:-1],n_coarse))
            return result.index_add(-1,cell_to_coarse,x)
        left,right=aggregate(left),aggregate(right)
    return left,right


def endpoint_mst_integral(concentration_prev, concentration_next, coefficients):
    """Global integrated storage mass; external source integral is separate."""
    left,right=coefficients
    return (left*concentration_prev+right*concentration_next).sum(dim=-1)


def linear_external_mass_integral(concentration_prev, concentration_next,
                                  inflow_mass_rate, outflow_water_rate, times):
    """Integrated SSM along linear C, sampled at backward-Euler step ends.

    Water arrays: [steps, ..., cells]; inflow is specified mass/time and
    outflow is nonnegative volume/time, aggregated separately per cell.
    Negative transported C does not extract mass (MF6 standard SSM rule).
    Inlet concentrations and actual water rates must come from forcings/flow,
    never from reference net solute-budget rates.
    """
    dt=times[1:]-times[:-1]
    if inflow_mass_rate.shape!=outflow_water_rate.shape or len(dt)!=len(outflow_water_rate):
        raise ValueError('Boundary rates and internal times must align')
    if not torch.all(dt>0) or torch.any(outflow_water_rate<0):
        raise ValueError('Times must increase and outflow magnitudes must be nonnegative')
    shape=(-1,)+(1,)*(outflow_water_rate.ndim-1)
    alpha=((times[1:]-times[0])/(times[-1]-times[0])).reshape(shape)
    c=(1-alpha)*concentration_prev+alpha*concentration_next
    return ((inflow_mass_rate-outflow_water_rate*c.clamp_min(0))*dt.reshape(shape)).sum(dim=(0,-1))


def coarse_global_residual(concentration_prev, concentration_next, left, right,
                           injected_mass, outflow_volume, steps=18):
    """Dimensionless global residual using injection-normalized coefficients.

    End states and coefficient arrays have [..., coarse_cells] layout. The
    coefficients describe endpoint-interpolated simulated FLOW, not predicted
    head. They are training auxiliaries only; no solute targets enter them.
    """
    alpha=torch.arange(1,steps+1,device=left.device,dtype=left.dtype)/steps
    alpha=alpha.reshape((-1,)+(1,)*left.ndim)
    path=(1-alpha)*concentration_prev+alpha*concentration_next
    extraction=(outflow_volume*path.clamp_min(0)).sum(dim=-1).mean(dim=0)
    return (left*concentration_prev+right*concentration_next-injected_mass).sum(dim=-1)+extraction
