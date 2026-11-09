"""B0 on the development test batches, matched hours: paired 27-day block-bootstrap CIs of S differences
(NN vs linear set 2a, vs linear + best constant offset, vs 27-day recurrence), per seed and for the 2-seed mean."""
import numpy as np, pandas as pd, torch
import folds
from eval_matched_baselines import linear_predictions, ace_hourly, metrics
from heldout_scores import boot_dS
from frozen_model import B0_RUN
dev = torch.device('cuda:0')
ace = ace_hourly()
parts = []
for f in range(1, 6):
    a = pd.read_csv(f'{B0_RUN}/seed42/fold{f}/test_predictions.csv', parse_dates=['time']).set_index('time')
    b = pd.read_csv(f'{B0_RUN}/seed43/fold{f}/test_predictions.csv', parse_dates=['time']).set_index('time')
    d = pd.DataFrame({'obs': a.obs, 'nn42': a.pred, 'nn43': b.pred.reindex(a.index)})
    d['linear'] = linear_predictions(f).reindex(d.index).to_numpy()
    d['recurrence'] = ace.reindex(d.index - pd.Timedelta(days=27)).to_numpy()
    parts.append(d)
D = pd.concat(parts).dropna()
D['nn_mean'] = (D.nn42 + D.nn43) / 2
o = D.obs.to_numpy()
offs = np.arange(-60, 61, 1.0)
best = min(offs, key=lambda c: metrics(o, D.linear.to_numpy() + c)['S'])
D['linear_off'] = D.linear + best
blk = ((D.index - D.index.min()) / pd.Timedelta(days=27)).astype(int).to_numpy()
print(f'matched hours {len(D)}; best linear offset (in-sample, on S) {best:+.0f} km/s')
for m in ['nn42', 'nn43', 'nn_mean', 'linear', 'linear_off', 'recurrence']:
    s = metrics(o, D[m].to_numpy()); print(f'{m:11s} S {s["S"]:.2f} RMSE {s["rmse"]:.2f} bias {s["bias"]:+.1f} r {s["r"]:.3f} TSS500 {s["tss500"]:.2f} TSS600 {s["tss600"]:.2f} HSS500 {s["hss500"]:.2f} HSS600 {s["hss600"]:.2f}')
for a in ['nn42', 'nn43', 'nn_mean']:
    for b in ['linear', 'linear_off', 'recurrence']:
        ds = boot_dS(o, (D[a].to_numpy() - o) ** 2, (D[b].to_numpy() - o) ** 2, blk, 1000, dev)
        print(f'S({a}) - S({b}) = {metrics(o, D[a].to_numpy())["S"] - metrics(o, D[b].to_numpy())["S"]:+.2f}  95% CI [{np.percentile(ds, 2.5):+.2f}, {np.percentile(ds, 97.5):+.2f}]')
