"""Conservative horizontal refinement of one two-layer diagnostic source model.

Wells remain points at original cell centers (odd refinement factors). RIV
conductance is area-partitioned; recharge/ET flux density is unchanged. CHD
zones and piecewise-constant material geometry retain their original areas.
The +1 m wet-CHD diagnostic assumption is explicit, not a field calibration.
"""
import argparse
import subprocess
from pathlib import Path
import flopy
import numpy as np
from gwno.data.prepare import write_json

def refine_array(value,factor):
    if value is None:return None
    a=np.asarray(value)
    return np.repeat(np.repeat(a,factor,axis=-2),factor,axis=-1) if a.ndim>=2 else value

def map_records(data,factor,kind,bottom=None,convertible=None,minimum_chd_saturation=None):
    mapped={}
    for period,records in data.items():
        result=[]
        for record in records:
            layer,row,col=record['cellid']
            children=[(factor//2,factor//2)] if kind=='wel' else [(i,j) for i in range(factor) for j in range(factor)]
            values=[float(record[k]) for k in records.dtype.names[1:]]
            if kind=='riv':values[1]/=factor**2
            if kind=='chd' and convertible[layer,row,col]!=0:
                if minimum_chd_saturation is not None:
                    values[0]=max(values[0],float(bottom[layer,row,col]+minimum_chd_saturation))
                elif values[0]<=bottom[layer,row,col]:values[0]=float(bottom[layer,row,col]+1.)
            result.extend([((layer,row*factor+i,col*factor+j),*values) for i,j in children])
        mapped[period]=result
    return mapped

def pulse_wells(data,multipliers):
    """Expand inherited periods before changing positive injection only."""
    result={};current=None
    for period,multiplier in enumerate(multipliers):
        if period in data:current=data[period]
        if current is None:raise ValueError('Target templates require initial WEL records')
        records=current.copy()
        records['q']=np.where(records['q']>0,records['q']*multiplier,records['q'])
        result[period]=records
    return result

def run(factor,scenario='dual_42',transport_relaxation=1.,substeps=1,minimum_chd_saturation=None,transport_scaling='NONE',xt3d_off=False,transport_dbd=False,vertical_dispersivity=None,mf6_version='6.6.1',output_directory=None,random_k=False,*,k_seed=None,k_length_multiplier=1.,injection_multipliers=None,build_only=False):
    if factor<1 or factor%2!=1:raise ValueError('Odd positive factor required to preserve point-well centers')
    from solver import executable
    from template_inputs import ensure_templates
    root=Path(__file__).resolve().parent;exe=executable()
    source=flopy.mf6.MFSimulation.load(sim_ws=str(ensure_templates()/scenario),verbosity_level=0)
    f=source.get_model(f'gwf_{scenario}');t=source.get_model(f'gwt_{scenario}')
    suffix='' if transport_relaxation==1. else f'_relax_{transport_relaxation:g}'
    if substeps!=1:suffix+=f'_steps_{substeps}'
    if minimum_chd_saturation is not None:suffix+=f'_minsat_{minimum_chd_saturation:g}'
    if transport_scaling!='NONE':suffix+=f'_scale_{transport_scaling.lower()}'
    if xt3d_off:suffix+='_xt3d_off'
    if transport_dbd:suffix+='_dbd'
    if vertical_dispersivity is not None:suffix+=f'_atv_{vertical_dispersivity:g}'
    if mf6_version!='6.6.1':suffix+=f'_mf6_{mf6_version}'
    path=Path(output_directory) if output_directory is not None else root/'prepared/two_layer_refinement'/scenario/f'factor_{factor}{suffix}'
    path.mkdir(parents=True,exist_ok=True)
    sim=flopy.mf6.MFSimulation(sim_name='refinement',sim_ws=str(path),exe_name=str(exe))
    periods=source.tdis.perioddata.get_data().copy()
    periods['nstp']=substeps
    periods['tsmult']=1.
    flopy.mf6.ModflowTdis(sim,time_units='DAYS',nper=30,perioddata=periods)
    gwf=flopy.mf6.ModflowGwf(sim,modelname='flow',save_flows=True,newtonoptions='NEWTON UNDER_RELAXATION')
    gwt=flopy.mf6.ModflowGwt(sim,modelname='transport',save_flows=True)
    for model in [gwf,gwt]:
        damping=(dict(under_relaxation='SIMPLE',under_relaxation_gamma=transport_relaxation)
                 if model is gwt and transport_relaxation<1. else dict(under_relaxation='NONE'))
        if model is gwt and transport_dbd:
            damping=dict(under_relaxation='DBD',under_relaxation_theta=0.7,
                         under_relaxation_kappa=0.1,under_relaxation_gamma=0.1,
                         under_relaxation_momentum=0.)
        ims=flopy.mf6.ModflowIms(sim,filename=f'{model.name}.ims',print_option='SUMMARY',
            outer_dvclose=1e-6,outer_maximum=100,inner_maximum=300,**damping,
            inner_dvclose=1e-6,rcloserecord=[('inner_rclose',1e-6,None)],
            linear_acceleration='BICGSTAB',relaxation_factor=1.,
            scaling_method=transport_scaling if model is gwt else 'NONE',reordering_method='NONE')
        sim.register_ims_package(ims,[model.name])
    geometry=dict(nlay=2,nrow=30*factor,ncol=50*factor,
        delr=np.repeat(f.dis.delr.array/factor,factor),delc=np.repeat(f.dis.delc.array/factor,factor),
        top=refine_array(f.dis.top.array,factor),botm=refine_array(f.dis.botm.array,factor),
        idomain=refine_array(f.dis.idomain.array,factor))
    flopy.mf6.ModflowGwfdis(gwf,length_units='METERS',**geometry)
    flopy.mf6.ModflowGwtdis(gwt,length_units='METERS',**geometry)
    flopy.mf6.ModflowGwfic(gwf,strt=refine_array(f.ic.strt.array,factor))
    flopy.mf6.ModflowGwtic(gwt,strt=refine_array(t.ic.strt.array,factor))
    materials={k:getattr(f.npf,k).get_data() for k in ['k','k22','k33','icelltype']}
    if random_k:
        from random_k import generate_k
        new_k,k_metadata=generate_k(f,scenario,seed=k_seed,length_multiplier=k_length_multiplier)
        if k_seed is not None or k_length_multiplier!=1.:
            k_metadata.update(template=scenario,length_multiplier=k_length_multiplier)
        for direction in ['k22','k33']:
            if materials[direction] is not None:
                materials[direction]=materials[direction]/materials['k']*new_k
        materials['k']=new_k
        np.save(path/'k_coarse.npy',new_k)
        write_json(path/'random_k.json',k_metadata)
    flopy.mf6.ModflowGwfnpf(gwf,save_specific_discharge=True,**{k:refine_array(v,factor) for k,v in materials.items()})
    flopy.mf6.ModflowGwfsto(gwf,transient={0:True},**{k:refine_array(getattr(f.sto,k).get_data(),factor) for k in ['ss','sy','iconvert']})
    original_wel=f.wel.stress_period_data.get_data()
    if injection_multipliers is not None:
        if len(injection_multipliers)!=len(periods):raise ValueError('One injection multiplier per stress period required')
        original_wel=pulse_wells(original_wel,injection_multipliers)
        # Store prescribed coarse forcing for export; targets must not inherit old Q.
        well_forcing=np.zeros((30,2,3,30,50),dtype=np.float32)
        for period,records in original_wel.items():
            for record in records:
                layer,row,col=record['cellid']
                well_forcing[period,layer,:,row,col]=[record['q'],record['concentration'],1.]
        np.save(path/'target_well_forcing.npy',well_forcing)
        write_json(path/'injection_protocol.json',dict(multipliers=injection_multipliers,template=scenario))
    wel=map_records(original_wel,factor,'wel')
    riv=map_records(f.riv.stress_period_data.get_data(),factor,'riv')
    evt=map_records(f.evt.stress_period_data.get_data(),factor,'evt')
    chd=map_records(f.chd.stress_period_data.get_data(),factor,'chd',f.dis.botm.array,f.npf.icelltype.array,minimum_chd_saturation)
    flopy.mf6.ModflowGwfwel(gwf,pname='WEL-1',auxiliary=['CONCENTRATION'],stress_period_data=wel)
    flopy.mf6.ModflowGwfriv(gwf,pname='RIV-1',stress_period_data=riv)
    flopy.mf6.ModflowGwfevt(gwf,pname='EVT-1',stress_period_data=evt)
    flopy.mf6.ModflowGwfchd(gwf,pname='CHD-1',stress_period_data=chd)
    recharge={k:refine_array(v,factor) for k,v in f.rcha.recharge.get_data().items()}
    flopy.mf6.ModflowGwfrcha(gwf,pname='RCHA-1',recharge=recharge)
    flopy.mf6.ModflowGwtmst(gwt,porosity=refine_array(t.mst.porosity.array,factor))
    flopy.mf6.ModflowGwtadv(gwt,scheme='UPSTREAM')
    dispersion={k:refine_array(getattr(t.dsp,k).get_data(),factor) for k in ['diffc','alh','alv','ath1','ath2','atv']}
    if vertical_dispersivity is not None:dispersion['atv']=vertical_dispersivity
    flopy.mf6.ModflowGwtdsp(gwt,xt3d_off=xt3d_off,**dispersion)
    flopy.mf6.ModflowGwtssm(gwt,sources=[('WEL-1','AUX','CONCENTRATION')])
    flopy.mf6.ModflowGwfoc(gwf,head_filerecord='flow.hds',budget_filerecord='flow.bud',saverecord=[('HEAD','LAST'),('BUDGET','LAST')])
    flopy.mf6.ModflowGwtoc(gwt,concentration_filerecord='transport.ucn',budget_filerecord='transport.cbc',saverecord=[('CONCENTRATION','LAST'),('BUDGET','LAST')])
    flopy.mf6.ModflowGwfgwt(sim,exgmnamea='flow',exgmnameb='transport')
    # Check quantities that must not change with grid refinement.
    area=f.dis.delc.array[:,None]*f.dis.delr.array[None,:]
    errors={}
    for kind,original,mapped,column in [('well_Q',original_wel,wel,'q'),('river_conductance',f.riv.stress_period_data.get_data(),riv,'cond')]:
        index=1 if kind=='well_Q' else 2
        errors[kind]=max(abs(float(v[column].sum())-sum(r[index] for r in mapped[k])) for k,v in original.items())
    errors['well_mass_input']=max(abs(float((v['q']*v['concentration']).sum())-sum(r[1]*r[2] for r in wel[k])) for k,v in original_wel.items())
    fine_area=np.repeat(np.repeat(area/factor**2,factor,axis=0),factor,axis=1)
    errors['recharge_total']=max(abs(float(np.sum(v*area))-float(np.sum(recharge[k]*fine_area))) for k,v in f.rcha.recharge.get_data().items())
    errors['et_max_total']=max(abs(sum(float(r['rate'])*area[r['cellid'][1:]] for r in v)-sum(r[2]*fine_area[r[0][1:]] for r in evt[k])) for k,v in f.evt.stress_period_data.get_data().items())
    if max(errors.values())>1e-7:raise ValueError(f'Forcing mapping changed totals: {errors}')
    write_json(path/'mapping_report.json',dict(factor=factor,source=scenario,forcing_absolute_errors=errors,
        boundary_assumption=('original CHD zones retained; convertible dry CHD raised to bottom+1m for diagnosis'
            if minimum_chd_saturation is None else f'all convertible CHD clamped to at least bottom+{minimum_chd_saturation:g}m for diagnosis'),
        well_mapping='point at original center',dispersion='XT3D_OFF diagnostic' if xt3d_off else 'full XT3D',time_step_days=180/substeps,
        transport_relaxation=transport_relaxation,transport_scaling=transport_scaling,transport_dbd=transport_dbd,
        atv_override=vertical_dispersivity,mf6_version=mf6_version))
    sim.write_simulation(silent=True)
    if build_only:
        return path
    with (path/'execution.log').open('w') as log:
        process=subprocess.Popen([str(exe)],cwd=path,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        write_json(path/'execution_status.json',dict(status='RUNNING',pid=process.pid,mf6_version=mf6_version))
        process.wait()
        write_json(path/'execution_status.json',dict(status='DONE' if process.returncode==0 else 'FAILED',pid=process.pid,returncode=process.returncode,mf6_version=mf6_version))
    if process.returncode:raise RuntimeError(f'factor {factor} failed; see {path}/execution.log')
    reader=flopy.utils.HeadFile(str(path/'transport.ucn'),text='CONCENTRATION',precision='double');c=reader.get_alldata();reader.close()
    mask=geometry['idomain']>0
    summary=dict(factor=factor,frames=len(c),min_concentration=float(c[:,mask].min()),negative_fraction=float((c[:,mask]<0).mean()))
    if factor==1 and scenario=='dual_42' and substeps==1 and minimum_chd_saturation is None and not xt3d_off and vertical_dispersivity is None and mf6_version=='6.6.1':
        reference=root/'prepared/physics_pilot/upstream_wet_chd_diagnostic/dual_42.ucn'
        reader=flopy.utils.HeadFile(str(reference),text='CONCENTRATION',precision='double');ref=reader.get_alldata();reader.close()
        summary['reconstruction_max_error']=float(abs(c[:,mask]-ref[:,mask]).max())
        if summary['reconstruction_max_error']>1e-5:raise ValueError(f'Reconstruction differs from established pilot: {summary}')
    write_json(path/'result.json',summary);print(summary,flush=True)
    return path

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--factor',type=int,required=True)
    parser.add_argument('--scenario',default='dual_42')
    parser.add_argument('--transport-relaxation',type=float,default=1.)
    parser.add_argument('--substeps',type=int,default=1)
    parser.add_argument('--minimum-chd-saturation',type=float)
    parser.add_argument('--transport-scaling',choices=['NONE','DIAGONAL','L2NORM'],default='NONE')
    parser.add_argument('--xt3d-off',action='store_true')
    parser.add_argument('--transport-dbd',action='store_true')
    parser.add_argument('--atv',type=float,help='Transverse dispersivity for vertical flow; not horizontal-flow z spreading (ATH2)')
    parser.add_argument('--mf6-version',choices=['6.6.1','6.8.0'],default='6.6.1')
    args=parser.parse_args()
    if not 0<args.transport_relaxation<=1:parser.error('transport relaxation must be in (0,1]')
    if args.substeps<1:parser.error('substeps must be positive')
    if args.minimum_chd_saturation is not None and args.minimum_chd_saturation<=0:parser.error('minimum CHD saturation must be positive')
    if args.transport_dbd and args.transport_relaxation!=1.:parser.error('choose either SIMPLE or DBD relaxation')
    if args.atv is not None and args.atv<0:parser.error('dispersivity cannot be negative')
    run(args.factor,args.scenario,args.transport_relaxation,args.substeps,args.minimum_chd_saturation,args.transport_scaling,args.xt3d_off,args.transport_dbd,args.atv,args.mf6_version)
