import numpy as np
from .prepare import write_json

def make_splits(prepared, seed=42):
    names = sorted(p.stem for p in (prepared/'simulations').glob('*.npz'))
    np.random.default_rng(seed).shuffle(names)
    a, b = int(.7*len(names)), int(.9*len(names))
    split = dict(seed=seed, role='D0_development_previously_viewed',
                 train=names[:a], val=names[a:b], test=names[b:])
    write_json(prepared/'splits'/'d0_development.json', split)
    # Merge batch moments in float64, using only training scenarios and active cells.
    moments = {}
    for name in split['train']:
        with np.load(prepared/'simulations'/f'{name}.npz') as z:
            mask = z['active_mask']
            arrays = {'head': z['head'][:, mask].reshape(-1, 1),
                      'concentration': z['concentration'][:, mask].reshape(-1, 1),
                      'static': z['static'].transpose(0,2,3,1)[mask],
                      'forcing': z['forcing'].transpose(0,1,3,4,2)[:,mask].reshape(-1,z['forcing'].shape[2])}
            for key, x in arrays.items():
                x = x.astype(np.float64)
                n, mean, m2 = moments.get(key, (0, np.zeros(x.shape[1]), np.zeros(x.shape[1])))
                k, avg = len(x), x.mean(0)
                delta = avg-mean
                moments[key] = (n+k, mean+delta*k/(n+k), m2+((x-avg)**2).sum(0)+delta**2*n*k/(n+k))
    norm = {key:dict(mean=mean.tolist(), std=np.maximum(np.sqrt(m2/n),1e-8).tolist(), count=n)
            for key,(n,mean,m2) in moments.items()}
    norm['training_scenarios'] = split['train']
    write_json(prepared/'normalization.json', norm)
    return split, norm
