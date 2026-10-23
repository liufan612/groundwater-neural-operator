"""Generate paired source-concentration timing cases from accepted MF6 inputs."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
import json
import shutil
import subprocess
import time
import traceback
from pathlib import Path
import numpy as np
import flopy
from export_formal_scenario import export_scenario
from gwno.data.prepare import write_json
from solver import executable

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'prepared_formal_mf680_randomk_r3_repaired'
OUT = ROOT / 'prepared_source_timing_v1'
EXE = executable()


def aggregate_wells(records):
    """Merge vertically subdivided wells, preserving flow and injected mass."""
    forcing = np.zeros((2, 3, 30, 50), np.float64)
    injection = np.zeros((2, 30, 50), np.float64)
    mass = np.zeros_like(injection)
    for row in records:
        layer, i, j = row['cellid']
        layer, i, j = (0 if layer == 0 else 1), i // 5, j // 5
        q, cin = float(row['q']), float(row['concentration'])
        forcing[layer, 0, i, j] += q
        forcing[layer, 1, i, j] = cin
        forcing[layer, 2, i, j] = 1
        injection[layer, i, j] += max(q, 0.)
        mass[layer, i, j] += max(q, 0.) * cin
    np.divide(mass, injection, out=forcing[:, 1], where=injection > 0)
    result = forcing.astype(np.float32)
    np.testing.assert_allclose(result[:, 0].sum(dtype=np.float64), records['q'].sum(), rtol=2e-7)
    np.testing.assert_allclose(np.maximum(result[:, 0], 0).astype(np.float64) * result[:, 1],
                               mass, rtol=2e-7, atol=1e-10)
    return result


def source_multipliers(mode, base_mass):
    """Preserve period 0 and set the manuscript timing/amplitude contrast."""
    weights = np.ones(len(base_mass))
    if mode in ['pulse', 'baseline_pulse']:
        weights[1:] = 0
        weights[7:13] = 1
    elif mode == 'intermittent':
        weights[1:] = (np.arange(len(base_mass)-1) % 6 < 3).astype(float)
    if mode in ['pulse', 'intermittent']:
        weights[1:] *= base_mass[1:].sum() / np.dot(base_mass[1:], weights[1:])
    elif mode == 'high_continuous':
        weights[1:] = base_mass[1:].sum() / base_mass[7:13].sum()
    return weights


def generate(template, mode, name):
    with np.load(SOURCE / 'simulations' / f'{template}.npz') as z:
        meta = json.loads(str(z['metadata']))
    source_path = Path(meta['raw_model_path'])
    path = OUT / 'models' / name
    path.mkdir(parents=True, exist_ok=True)
    sim = flopy.mf6.MFSimulation.load(sim_ws=str(source_path), verbosity_level=0)
    flow = sim.get_model('flow')
    transport = sim.get_model('transport')
    dt = sim.tdis.perioddata.array['perlen'].astype(float)
    original = flow.wel.stress_period_data.get_data()
    records = []
    current = None
    for t in range(len(dt)):
        current = original.get(t, current)
        records.append(current.copy())
    # First period is identical; source timing differs only after the common first frame.
    base_mass = np.array([np.sum(np.maximum(r['q'], 0)*r['concentration'])*d
                          for r, d in zip(records, dt)])
    weights = source_multipliers(mode, base_mass)
    forcing = np.zeros((30, 2, 3, 30, 50), np.float32)
    revised = {}
    for t, r in enumerate(records):
        r['concentration'] = np.where(r['q'] > 0, r['concentration']*weights[t], r['concentration'])
        revised[t] = r
        forcing[t] = aggregate_wells(r)
    changed_mass = np.array([np.sum(np.maximum(r['q'], 0)*r['concentration'])*d
                             for r, d in zip(records, dt)])
    if mode in ['continuous', 'pulse', 'intermittent']:
        np.testing.assert_allclose(changed_mass.sum(), base_mass.sum(), rtol=1e-12)
    flow.wel.stress_period_data.set_data(revised)
    sim.set_sim_path(str(path))
    sim.write_simulation(silent=True)
    for filename in ['random_k.json', 'k_coarse.npy']:
        if (source_path / filename).exists():
            shutil.copy2(source_path / filename, path / filename)
    np.save(path / 'target_well_forcing.npy', forcing)
    write_json(path / 'source_protocol.json', dict(template=template, mode=mode,
        concentration_multipliers=weights.tolist(), mass_by_period=changed_mass.tolist(),
        total_injected_mass=float(changed_mass.sum()), original_mass=float(base_mass.sum())))
    with (path / 'execution.log').open('w') as log:
        process = subprocess.Popen([str(EXE)], cwd=path, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        write_json(OUT / 'active.json', dict(scenario=name, solver_pid=process.pid, started=time.time()))
        code = process.wait()
    if code or 'Normal termination of simulation' not in (path / 'execution.log').read_text():
        raise RuntimeError(f'MODFLOW failed: {name}; see execution.log')
    report = export_scenario(ROOT, OUT, name, path, 5, template=template)
    file = OUT / 'simulations' / f'{name}.npz'
    with np.load(file) as z:
        arrays = {k: z[k].copy() for k in z.files}
    metadata = json.loads(str(arrays['metadata']))
    metadata.update(source_mechanism='prescribed_concentration_timing', source_timing_mode=mode,
                    source_dataset=SOURCE.name, evaluation_role='development_only')
    arrays['metadata'] = json.dumps(metadata)
    for i, field in enumerate(['well_rate', 'well_inj_concentration', 'well_mask']):
        np.testing.assert_array_equal(arrays['forcing'][:,:,metadata['forcing_fields'].index(field)], forcing[:,:,i])
    with np.load(SOURCE / 'simulations' / f'{template}.npz') as reference:
        np.testing.assert_allclose(arrays['head'], reference['head'], rtol=1e-5, atol=1e-4)
        np.testing.assert_allclose(arrays['concentration'][0], reference['concentration'][0], rtol=1e-5, atol=1e-4)
    if report['raw_negative_mass_fraction'] > .01:
        raise ValueError(f'Raw negative mass exceeds 1%: {name}')
    initial_max = float(np.max(transport.ic.strt.array))
    inlet_max = max(float(r['concentration'][r['q'] > 0].max()) for r in revised.values())
    active = arrays['active_mask'].astype(bool)
    peak = float(arrays['concentration_raw'][:,active].max())
    if peak > max(initial_max, inlet_max) + 1e-5*max(1., initial_max, inlet_max):
        raise ValueError(f'Concentration overshoot: {name}, {peak}')
    np.savez_compressed(file, **arrays)
    write_json(OUT / 'quality' / f'{name}.json', dict(accepted=True, peak=peak, **report))
