"""Conservative control-volume convention: storage change + outward flux = source."""
import torch

def outward_flux(edge_flux, cells_from, cells_to, n_cells):
    """One signed flux per internal connection, positive from cell i to j.

    A vertical edge uses exactly the same rule as a horizontal edge.
    External sources and boundaries are supplied separately as net input rates.
    """
    shape=(*edge_flux.shape[:-1],n_cells)
    result=edge_flux.new_zeros(shape)
    result.index_add_(-1,cells_from,edge_flux)
    result.index_add_(-1,cells_to,-edge_flux)
    return result

def mass_balance(concentration_prev,concentration_next,pore_volume_prev,pore_volume_next,
                 dt,net_external_mass_rate,internal_outward_mass_rate):
    return (pore_volume_next*concentration_next-pore_volume_prev*concentration_prev)/dt + internal_outward_mass_rate-net_external_mass_rate

def advective_mass_flux(water_flux,concentration,cells_from,cells_to):
    """First-order upwind mass/time given signed volumetric water flux/time."""
    upstream=torch.where(water_flux>=0,concentration[...,cells_from],concentration[...,cells_to])
    return water_flux*upstream

def well_mass_rate(water_rate,injection_concentration,cell_concentration):
    return torch.where(water_rate>=0,water_rate*injection_concentration,water_rate*cell_concentration)

def modflow_ssm_mass_rate(water_rate,injection_concentration,cell_concentration):
    """MF6 6.6.1 standard SSM (not AUXMIXED), net mass rate into a cell.

    Outflow removes mass only when the cell concentration is nonnegative.
    This reproduces tsp-ssm.f90 ssm_term, including its negative-C switch;
    it does not clip the transported state or fix dispersion undershoots.
    """
    return torch.where(water_rate>=0,water_rate*injection_concentration,
                       water_rate*cell_concentration.clamp_min(0.))

def modflow_mst_storage_rate(concentration_prev,concentration_next,pore_volume_next,
                             flow_storage_rate,dt):
    """MODFLOW 6.6.1 MST accumulation (opposite sign to ratesto).

    flow_storage_rate is STO-SS + STO-SY, signed positive into the water cell.
    The effective old water volume is the NEW pore volume plus flow storage*dt,
    not the previous head-derived pore volume. See mst_cq_sto in gwt-mst.f90.
    """
    effective_old_volume=pore_volume_next+flow_storage_rate*dt
    return (pore_volume_next*concentration_next-effective_old_volume*concentration_prev)/dt
