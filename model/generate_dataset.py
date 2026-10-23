"""Build the manuscript's 100 baseline scenarios and export the learning grid."""
import argparse
import json
from pathlib import Path
import numpy as np
from refine_two_layer_pilot import run as build
from repair_vertical_r3 import repair
from export_formal_scenario import export_scenario
from gwno.data.splits import make_splits
from gwno.data.prepare import write_json
import solver

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'prepared_formal_mf680_randomk_r3_repaired'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenarios', nargs='+', help='Default: dual_0 through dual_99')
    parser.add_argument('--build-only', action='store_true', help='Write solver inputs without running MODFLOW')
    args = parser.parse_args()
    config = json.loads((ROOT.parent/'configs/paper.json').read_text())
    names = args.scenarios or [f'dual_{i}' for i in range(100)]
    for name in names:
        if not args.build_only and (OUT/'reports'/f'{name}.json').exists():
            print(name, 'already exported', flush=True)
            continue
        original = ROOT/'prepared_formal_mf680_randomk_r3/models'/name
        model = build(5, name, substeps=18, minimum_chd_saturation=1.,
                      transport_scaling='DIAGONAL', mf6_version='6.8.0',
                      output_directory=original, random_k=True, build_only=True)
        parts = config['lower_subdivisions_by_scenario'][name]
        if parts > 1:
            model = repair(name, parts=parts, build_only=True)
            if not isinstance(model, Path):
                raise RuntimeError(f'Failed to refine {name}: {model}')
        if args.build_only:
            print(name, 'built', model, flush=True)
            continue
        solver.run(model)
        report = export_scenario(ROOT, OUT, name, model, 5)
        if report['raw_negative_mass_fraction'] > .01:
            # Do not mark a rejected case as completed on the next invocation.
            (OUT/'reports'/f'{name}.json').unlink()
            raise ValueError(f'{name}: raw negative mass fraction exceeds 1%')
        print(name, 'exported', flush=True)
    if not args.build_only and all((OUT/'reports'/f'dual_{i}.json').exists() for i in range(100)):
        split, _ = make_splits(OUT, config['split_seed'])
        write_json(OUT/'config.json', config)
        print('100 scenarios ready; split:', {k: len(split[k]) for k in ['train','val','test']})


if __name__ == '__main__':
    main()
