"""Reconstruct prescribed coarse-grid covariates from fixed MODFLOW inputs."""
from pathlib import Path
from zipfile import ZipFile
import json
import numpy as np
import flopy
from gwno.data.prepare import STATIC, FORCING

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / 'source_templates'


def ensure_templates():
    if not (TEMPLATES / 'dual_99/mfsim.nam').exists():
        with ZipFile(ROOT / 'configs/source_templates.zip') as archive:
            index = json.loads(archive.read('index.json'))
            for relative, source in index.items():
                destination = TEMPLATES / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(source))
    return TEMPLATES


def load_template(name):
    simulation = flopy.mf6.MFSimulation.load(
        sim_ws=str(ensure_templates() / name), verbosity_level=0)
    return simulation, simulation.get_model('gwf_' + name), simulation.get_model('gwt_' + name)


def coarse_inputs(name):
    """Known inputs only; solved head/concentration are supplied by the exporter.

    Preserve the paper's exported input convention: upward-increasing row index
    for the y feature, layerwise top elevations, explicit-period CHD
    masks, and a zero recharge placeholder. Training replaces the last two with
    inherited prescribed CHD and actual RCHA arrays before normalization.
    """
    sim, flow, transport = load_template(name)
    active = flow.dis.idomain.array > 0
    times = np.cumsum(sim.tdis.perioddata.array['perlen'])
    shape = (len(times),) + active.shape
    bottom = flow.dis.botm.array
    top = np.concatenate([flow.dis.top.array[None], bottom[:-1]])
    porosity = np.broadcast_to(transport.mst.porosity.array, active.shape)
    x = np.broadcast_to(np.cumsum(flow.dis.delr.array) - flow.dis.delr.array / 2, active.shape)
    y = np.broadcast_to((np.cumsum(flow.dis.delc.array) - flow.dis.delc.array / 2)[None, :, None], active.shape)
    static = np.stack([flow.npf.k.array, top, bottom, porosity, x, y], axis=1).astype(np.float32)
    static = np.where(active[:, None], static, 0.)
    forcing = np.zeros((len(times), 2, len(FORCING), 30, 50), np.float32)
    channels = {name: i for i, name in enumerate(FORCING)}
    for package, fields in [('evt', [('rate', 'ET')]),
                            ('riv', [('stage', 'river_stage'), ('cond', 'river_cond'), ('rbot', 'river_rbot')]),
                            ('wel', [('q', 'well_rate'), ('concentration', 'well_inj_concentration')]),
                            ('chd', [])]:
        period_data = flow.get_package(package).stress_period_data.get_data()
        if package == 'wel':
            current = None
            expanded = {}
            for period in range(len(times)):
                current = period_data.get(period, current)
                expanded[period] = current
            period_data = expanded
        for period, records in period_data.items():
            for record in records:
                layer, row, col = record['cellid']
                if not active[layer, row, col]:
                    continue
                for source, target in fields:
                    forcing[period, layer, channels[target], row, col] = record[source]
                if package in ['wel', 'chd']:
                    forcing[period, layer, channels['well_mask' if package == 'wel' else 'chd_mask'], row, col] = 1
    physical_top = np.concatenate([flow.dis.top.array[None], bottom[:-1]])
    area = flow.dis.delc.array[:, None] * flow.dis.delr.array[None, :]
    volume = np.where(active, (physical_top - bottom) * area, 0.).astype(np.float32)
    metadata = dict(scenario=name, source_mechanism='continuous_well_injection',
                    time_unit='days', length_unit='meters', concentration_unit='mg/L',
                    static_fields=STATIC, forcing_fields=FORCING,
                    initial_state='first_saved_frame_at_180_days', layer_order='MODFLOW_top_to_bottom')
    return dict(static=static, forcing=forcing, active_mask=active,
                times_days=times, cell_volume=volume, metadata=json.dumps(metadata))
