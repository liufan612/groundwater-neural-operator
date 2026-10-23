"""Read the corrected CSV once; retain MODFLOW layer/time/geometry semantics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import flopy

STATIC = ["K", "top", "bottom", "porosity", "x", "y"]
FORCING = ["recharge", "ET", "river_stage", "river_cond", "river_rbot",
           "well_rate", "well_inj_concentration", "chd_mask", "well_mask"]

def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")

def prepare_simulations(csv_path, modflow_root, output_dir):
    output = Path(output_dir)
    (output / "simulations").mkdir(parents=True, exist_ok=True)
    # CSV is ordered by scenario, but chunks can straddle scenario boundaries.
    pending = None
    records = []
    seen = set()
    def save(frame):
        name = str(frame.model_name.iloc[0])
        if name in seen:
            raise ValueError(f"CSV scenario is not contiguous: {name}")
        seen.add(name)
        sim = flopy.mf6.MFSimulation.load(sim_ws=str(Path(modflow_root) / name), verbosity_level=0)
        gwf, gwt = sim.get_model('gwf_' + name), sim.get_model('gwt_' + name)
        mask = gwf.dis.idomain.array > 0
        times = np.cumsum(sim.tdis.perioddata.array['perlen'])
        shape = (len(times),) + mask.shape
        idx = (frame.time_step.to_numpy(int), 1-frame.lytyp.to_numpy(int),
               frame.row.to_numpy(int), frame.col.to_numpy(int))
        present = np.zeros(shape, dtype=bool)
        present[idx] = True
        if len(frame) != present.sum() or not np.array_equal(present, np.broadcast_to(mask, shape)):
            raise ValueError(f"Missing/duplicate/inactive CSV cells: {name}")
        if not np.allclose(frame.time_days, times[idx[0]]):
            raise ValueError(f"Wrong saved-time alignment: {name}")
        def grid(column):
            a = np.zeros(shape, np.float32)
            a[idx] = frame[column].to_numpy(np.float32)
            if not np.isfinite(a[:, mask]).all():
                raise ValueError(f"Nonfinite {column}: {name}")
            return a
        head, conc = grid('head'), grid('concentration')
        porosity = np.broadcast_to(gwt.mst.porosity.array, mask.shape).astype(np.float32)
        static = np.stack([grid(k)[0] if k != 'porosity' else porosity for k in STATIC], axis=1)
        static *= mask[:, None]
        forcing = np.stack([grid(k) for k in FORCING], axis=2)
        bottom = np.nan_to_num(gwf.dis.botm.array)
        top = np.concatenate([np.nan_to_num(gwf.dis.top.array)[None], bottom[:-1]])
        area = gwf.dis.delc.array[:, None] * gwf.dis.delr.array[None, :]
        thickness = np.maximum(top-bottom, 0)
        sat = np.where(gwf.npf.icelltype.array[None] != 0,
                       np.clip(head-bottom, 0, thickness), thickness)
        cell_volume = (thickness * area * mask).astype(np.float32)
        pore_volume = (sat * area * porosity * mask).astype(np.float32)
        meta = dict(scenario=name, source_mechanism='continuous_well_injection',
                    time_unit=str(sim.tdis.time_units.get_data()),
                    length_unit=str(gwf.dis.length_units.get_data()), concentration_unit=None,
                    static_fields=STATIC, forcing_fields=FORCING,
                    initial_state='first_saved_frame_at_180_days', layer_order='MODFLOW_top_to_bottom')
        np.savez_compressed(output/'simulations'/f'{name}.npz', head=head, concentration=conc,
                            static=static, forcing=forcing, active_mask=mask, times_days=times,
                            cell_volume=cell_volume, pore_volume=pore_volume,
                            metadata=json.dumps(meta))
        records.append(dict(scenario=name, frames=len(times), active_cells=int(mask.sum()),
                            start_days=float(times[0]), end_days=float(times[-1])))
        print(f"prepared {name} ({len(records)})", flush=True)
    for chunk in pd.read_csv(csv_path, chunksize=200000):
        block = pd.concat([pending, chunk], ignore_index=True) if pending is not None else chunk
        last = block.model_name.iloc[-1]
        for _, frame in block[block.model_name != last].groupby('model_name', sort=False):
            save(frame)
        pending = block[block.model_name == last].copy()
    if pending is not None:
        save(pending)
    pd.DataFrame(records).to_csv(output/'scenario_metadata.csv', index=False)
    write_json(output/'preparation_report.json', dict(scenarios=len(records), csv=str(csv_path)))
    return records
