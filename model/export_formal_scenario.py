"""Aggregate refined solver output and apply the manuscript's zero clipping."""
import json
from pathlib import Path
import numpy as np
import flopy
from gwno.data.prepare import write_json
from gwno.data.modflow import transport_budget_diagnostics


def export_scenario(root,output,scenario,model_path,factor,*,template=None):
    root,output,model_path=map(Path,(root,output,model_path))
    from template_inputs import coarse_inputs
    arrays = coarse_inputs(template or scenario)
    sim=flopy.mf6.MFSimulation.load(sim_ws=str(model_path),verbosity_level=0,load_only=['dis','npf','mst'])
    f=sim.get_model('flow');t=sim.get_model('transport');mask=f.dis.idomain.array>0
    with flopy.utils.HeadFile(str(model_path/'transport.ucn'),text='CONCENTRATION',precision='double') as reader:
        c=reader.get_alldata();times=np.array(reader.get_times())
    with flopy.utils.HeadFile(str(model_path/'flow.hds'),precision='double') as reader:h=reader.get_alldata()
    if len(c)!=30 or not np.array_equal(times,arrays['times_days']):raise ValueError('Incomplete or misaligned output frames')
    if not (np.isfinite(c[:,mask]).all() and np.isfinite(h[:,mask]).all()):raise ValueError('Nonfinite accepted fields')
    if max(abs(c[:,mask]).max(),abs(h[:,mask]).max())>1e20:raise ValueError('Dry/inactive sentinel reached exported active cells')
    bottom=f.dis.botm.array;top=np.concatenate([f.dis.top.array[None],bottom[:-1]])
    thickness=top-bottom;area=f.dis.delc.array[:,None]*f.dis.delr.array[None,:]
    saturated=np.where(f.npf.icelltype.array[None]!=0,np.clip(h-bottom,0,thickness),thickness)
    volume=np.where(mask,saturated*area*t.mst.porosity.array,0.)
    concentration=np.where(mask,c,0.)
    lower_parts=int(f.dis.nlay.array)-1
    def merge_layers(a,total):
        lower=a[:,1:].sum(1) if total else a[:,1:].mean(1)
        return np.stack([a[:,0],lower],axis=1)
    shape=(30,2,30,factor,50,factor)
    coarse_volume=merge_layers(volume,True).reshape(shape).sum((3,5))
    coarse_mass=merge_layers(volume*concentration,True).reshape(shape).sum((3,5))
    raw=np.divide(coarse_mass,coarse_volume,out=np.zeros_like(coarse_mass),where=coarse_volume>0)
    clipped=np.maximum(raw,0.)
    coarse_mask=arrays['active_mask']
    arrays['head']=np.where(coarse_mask,merge_layers(np.where(mask,h,0.),False).reshape(shape).mean((3,5)),0.).astype('float32')
    arrays['concentration_raw']=raw.astype('float32')
    arrays['concentration']=clipped.astype('float32')
    arrays['pore_volume']=coarse_volume.astype('float32')
    metadata=json.loads(str(arrays['metadata']))
    if template is not None:
        metadata.update(scenario=scenario,source_template=template)
    if (model_path/'target_well_forcing.npy').exists():
        wells=np.load(model_path/'target_well_forcing.npy')
        for i,field in enumerate(['well_rate','well_inj_concentration','well_mask']):
            arrays['forcing'][:,:,metadata['forcing_fields'].index(field)]=wells[:,:,i]
        metadata['source_mechanism']='prescribed_pulsed_well_injection'
    # Derive the ML K input from the actual solver material, never the old CSV.
    coarse_k=merge_layers(f.npf.k.array[None],False)[0].reshape(2,30,factor,50,factor).mean((2,4))
    arrays['static'][:,metadata['static_fields'].index('K')]=np.where(coarse_mask,coarse_k,0)
    if (model_path/'random_k.json').exists():
        metadata['random_k']=json.loads((model_path/'random_k.json').read_text())
        saved_k=np.load(model_path/'k_coarse.npy')
        if not np.allclose(coarse_k[coarse_mask],saved_k[coarse_mask],rtol=1e-7):
            raise ValueError('Solver K differs from generated coarse K')
    metadata.update(dataset=output.name.removeprefix('prepared_'),mf6_version='6.8.0',advection='UPSTREAM',dispersion='full XT3D',
        refinement_factor=factor,lower_layer_subdivisions=lower_parts,
        internal_dt_days=float(sim.tdis.perioddata.array['perlen'][0]/sim.tdis.perioddata.array['nstp'][0]),concentration_aggregation='saturated-pore-volume weighted across horizontal and vertical children',
        head_aggregation='arithmetic mean of equal-thickness lower sublayers and horizontal children',clipping='max(C,0) after aggregation',
        boundary='all convertible CHD heads at least bottom+1m',raw_model_path=str(model_path.resolve()))
    arrays['metadata']=json.dumps(metadata)
    budgets=transport_budget_diagnostics(model_path/'transport.cbc',model_path/'flow.dis.grb')
    residual=max(row['max_cell_relative_residual'] for row in budgets)
    if residual>1e-5:raise ValueError(f'Accepted raw solution has excessive budget residual: {residual}')
    report=dict(scenario=scenario,frames=30,active_cells=int(coarse_mask.sum()),
        raw_fine_min=float(c[:,mask].min()),raw_coarse_min=float(raw[:,coarse_mask].min()),
        raw_negative_mass_fraction=float((volume*np.maximum(-concentration,0.)).sum()/max((volume*np.maximum(concentration,0.)).sum(),1e-30)),
        clipped_coarse_cells=int((raw[:,coarse_mask]<0).sum()),
        clipping_added_mass_by_frame=((clipped-raw)*coarse_volume).sum((1,2,3)).tolist(),
        max_relative_raw_budget_residual=residual,
        note='Mass budgets describe raw simulation; zero-clipped exported concentration changes mass.')
    (output/'simulations').mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'simulations'/f'{scenario}.npz',**arrays)
    write_json(output/'reports'/f'{scenario}.json',report)
    return report
