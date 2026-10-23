"""Generate the four source arms and optional matched-input intermittent cases."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import run_source_timing as R
from gwno.data.prepare import write_json

ROOT = Path(__file__).resolve().parent
ARMS = {'S00': 'continuous', 'S10': 'baseline_pulse', 'S01': 'high_continuous',
        'S11': 'pulse', 'intermittent': 'intermittent'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arms', nargs='+', choices=list(ARMS), default=['S00','S10','S01','S11'])
    p.add_argument('--bases', nargs='+', help='Default: all 20 validation base scenarios')
    args = p.parse_args()
    split = json.loads((R.SOURCE/'splits/d0_development.json').read_text())
    bases = args.bases or split['val']
    if not set(bases) <= set(split['val']):
        p.error('Source diagnostics use the validation bases specified in the manuscript.')
    R.OUT = ROOT/'prepared_source_arms'
    R.OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(R.SOURCE/'normalization.json', R.OUT/'normalization.json')
    for base in bases:
        for arm in args.arms:
            name = f'{base}_{arm}'
            if not (R.OUT/'quality'/f'{name}.json').exists():
                R.generate(base, ARMS[arm], name)
            print(name, 'exported', flush=True)
    rows = []
    for base in split['val']:
        for arm, mode in ARMS.items():
            name = f'{base}_{arm}'
            if (R.OUT/'quality'/f'{name}.json').exists():
                rows.append(dict(scenario=name, base_scenario=base, arm=arm, mode=mode,
                                 path=str(R.OUT/'simulations'/f'{name}.npz')))
    with (R.OUT/'dataset_index.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['scenario','base_scenario','arm','mode','path'])
        writer.writeheader(); writer.writerows(rows)
    write_json(R.OUT/'protocol.json', dict(arms=ARMS, bases=split['val'],
               initial_day=180, pulse_period_indices=list(range(7,13)),
               water_injection='unchanged', role='development diagnostics; no training',
               normalization='reuse training protocol normalization'))


if __name__ == '__main__':
    main()
