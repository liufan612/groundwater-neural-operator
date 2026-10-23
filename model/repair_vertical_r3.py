"""Resolve confined lower-layer transport vertically, preserving full XT3D."""
import os
os.environ['OMP_NUM_THREADS']='1'
import json, subprocess, time, traceback
from pathlib import Path
import numpy as np
import flopy
from gwno.data.prepare import write_json

ROOT=Path(__file__).resolve().parent

def repair(scenario,parts=4,build_only=False):
    original=ROOT/'prepared_formal_mf680_randomk_r3/models'/scenario
    output=ROOT/f'prepared_formal_mf680_randomk_r3/repairs/{scenario}_lower{parts}'
    path=output/'model';path.mkdir(parents=True,exist_ok=True)
    state=dict(status='BUILDING',scenario=scenario,pid=os.getpid(),started=time.time(),promoted=False,
               change=f'Confined lower layer split into {parts}; full XT3D and original K retained')
    write_json(output/'status.json',state)
    try:
        sim=flopy.mf6.MFSimulation.load(sim_ws=str(original),verbosity_level=0)
        sim.set_sim_path(str(path))
        f=sim.get_model('flow');t=sim.get_model('transport')
        assert np.all(f.npf.icelltype.array[1]==0)
        def expand(a):
            if a is None:return None
            a=np.asarray(a)
            return np.concatenate([a[:1],np.repeat(a[1:2],parts,axis=0)]) if a.ndim==3 else a.copy()
        dis=f.dis
        bottom=dis.botm.array.copy()
        geometry=dict(nlay=1+parts,nrow=int(dis.nrow.array),ncol=int(dis.ncol.array),
            delr=dis.delr.array.copy(),delc=dis.delc.array.copy(),top=dis.top.array.copy(),
            botm=np.concatenate([bottom[:1],np.stack([bottom[0]+(bottom[1]-bottom[0])*(i+1)/parts for i in range(parts)])]),
            idomain=expand(dis.idomain.array),length_units='METERS')
        packages=[]
        for model,names in [(f,{'npf':['k','k22','k33','icelltype'],'sto':['ss','sy','iconvert'],'ic':['strt']}),
                            (t,{'mst':['porosity'],'ic':['strt'],'dsp':['diffc','alh','alv','ath1','ath2','atv']})]:
            for pkg,fields in names.items():
                obj=model.get_package(pkg)
                for field in fields:
                    data=getattr(obj,field).array
                    if data is not None:packages.append((obj,field,expand(data)))
        boundaries=[]
        for name,divide in [('wel','q'),('riv','cond'),('evt','rate'),('chd',None)]:
            pkg=f.get_package(name)
            if pkg is None:continue
            mapped={}
            for period,records in pkg.stress_period_data.get_data().items():
                rows=[]
                for record in records:
                    layer,row,col=record['cellid'];layers=[0] if layer==0 else range(1,parts+1)
                    for newlayer in layers:
                        item=record.copy();item['cellid']=(newlayer,row,col)
                        if layer==1 and divide:item[divide]/=parts
                        rows.append(tuple(item))
                mapped[period]=np.array(rows,dtype=records.dtype).view(np.recarray)
                if divide:
                    assert np.isclose(mapped[period][divide].sum(),records[divide].sum())
            boundaries.append((pkg,mapped))
        for model in [f,t]:
            model.remove_package('dis')
            constructor=flopy.mf6.ModflowGwfdis if model is f else flopy.mf6.ModflowGwtdis
            constructor(model,**geometry)
        for pkg,field,data in packages:getattr(pkg,field).set_data(data)
        for pkg,data in boundaries:pkg.stress_period_data.set_data(data)
        sim.write_simulation(silent=True)
        np.save(path/'k_coarse.npy',np.load(original/'k_coarse.npy'))
        write_json(path/'random_k.json',json.loads((original/'random_k.json').read_text()))
        write_json(path/'vertical_refinement.json',dict(lower_parts=parts,source=str(original),upper_parts=1))
        if build_only:
            return path
        from solver import executable
        exe=executable()
        with (path/'execution.log').open('w') as log:
            p=subprocess.Popen([str(exe)],cwd=path,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            state.update(status='RUNNING',mf6_pid=p.pid);write_json(output/'status.json',state)
            code=p.wait()
        if code:raise RuntimeError(f'MF6 exit {code}; see execution.log')
        with flopy.utils.HeadFile(str(path/'transport.ucn'),text='CONCENTRATION',precision='double') as reader:
            c=reader.get_alldata()
        mask=geometry['idomain']>0
        state.update(status='DONE',frames=len(c),fine_min=float(c[:,mask].min()),fine_max=float(c[:,mask].max()))
    except Exception:
        state.update(status='FAILED',error=traceback.format_exc());traceback.print_exc()
    state['ended']=time.time();write_json(output/'status.json',state)
    print(json.dumps(state),flush=True)
    return state

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('scenario');parser.add_argument('--parts',type=int,default=4);args=parser.parse_args()
    if args.parts<2:parser.error('Use at least two confined sublayers')
    repair(args.scenario,args.parts)
