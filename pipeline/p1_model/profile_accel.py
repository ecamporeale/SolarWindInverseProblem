"""Loss profile of the learned radial acceleration (L, Vref) with NN1 held fixed.

For one fold: load the run's best checkpoint (plot_best_checkpoint.load_fold), compute NN1's boundary map
once per chunk, then for every (L, Vref) on a grid propagate it with DeltaV(r) = Vref (1 - exp(-(r-r0)/L))
(chunks batched 8 at a time) and score ACE over the scored windows (warm-up excluded) of the training and validation chunks.
Forward passes only, no training. Scores: RMSE and bias, and (since 2026-10-17) the training objective S
(train_joint_forecast.combined_score) on the percentile grid of the training chunks' observations, as in training.
RMSE alone is confounded by the loss: a model trained on S carries a positive bias, which a smaller acceleration
reduces, so the RMSE minimum moves toward weaker acceleration.

  python profile_accel.py --run runs/stage1.0d_lags12-24-48-72_mlp65-64-32 --fold 1 --cuda-index 0
Output: <run>/fold<k>/accel_profile.csv (L, Vref, split, n, sse, rmse, bias, S, trmse)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from forward_model import _inverse_softplus, steady_state_propagate
from helpers import sample_satellite_track
from plot_best_checkpoint import chunks_of, load_fold
from train_joint_forecast import _transient_skip_steps, combined_score, make_batch_record, percentile_grid

L_GRID = [10, 20, 35, 50, 75, 100, 150, 200, 300]
V_GRID = [0, 20, 40, 60, 80, 100, 150]
GROUP = 8                                   # chunks per batched physics call

p = argparse.ArgumentParser()
p.add_argument('--run', required=True)
p.add_argument('--fold', type=int, required=True)
p.add_argument('--cuda-index', type=int, default=0)
args = p.parse_args()
device = torch.device(f'cuda:{args.cuda_index}')
F = load_fold(args.fold, Path(args.run) / f'fold{args.fold}', device)
cfg, r, phi, fm = F['config'], F['r'], F['phi'], F['forward_model']
accel = fm.learned_radial_accel
L_tr, V_tr = (float(x) for x in accel.physical_parameters())
print(f'trained L = {L_tr:.3f} Rs, Vref = {V_tr:.3f} km/s', flush=True)

work = []
with torch.no_grad():
    for split in ('train', 'val'):
        for cid in chunks_of(F, split):
            rec = make_batch_record(F['specs'][cid], cfg, device)
            v_in = F['nn1'](rec['image_slice'])
            v0 = steady_state_propagate(v_in[:, 0], r, direction='outward', Omega=float(cfg.Omega))
            obs = rec['ace_obs']
            k0 = _transient_skip_steps(cfg, obs.shape[0])
            work.append((split, rec, v_in, v0, obs, k0))
o_tr = torch.cat([w[4][w[5]:] for w in work if w[0] == 'train'])
GRID = percentile_grid(o_tr[torch.isfinite(o_tr)].cpu().numpy())        # the training loss's fixed grid

rows = []
with torch.no_grad():
    for L in L_GRID:
        for V in V_GRID:
            accel.raw_L.data.fill_(float(_inverse_softplus(float(L))))
            accel.raw_Vref.data.fill_(float(_inverse_softplus(max(float(V), 1e-3))))
            acc = {s: [0.0, 0.0, 0] for s in ('train', 'val')}
            pairs = {s: ([], []) for s in ('train', 'val')}
            for g in range(0, len(work), GROUP):
                grp = work[g:g + GROUP]
                v_full_b = fm.forward_batched(torch.stack([w[2] for w in grp]), torch.stack([w[3] for w in grp]))
                for (split, rec, v_in, v0, obs, k0), v_full in zip(grp, v_full_b):
                    pred = sample_satellite_track(v_full, r, phi, rec['phi_obs_t'], rec['radius_obs_t'],
                                                  time_obs=rec['time_obs_t'], dt=cfg.dt)
                    o, q = obs[k0:], pred[k0:]
                    ok = torch.isfinite(o)
                    e = (q[ok] - o[ok]).double()
                    acc[split][0] += float((e ** 2).sum()); acc[split][1] += float(e.sum()); acc[split][2] += int(ok.sum())
                    pairs[split][0].append(o[ok]); pairs[split][1].append(q[ok])
            for s, (sse, se, n) in acc.items():
                S_, _, timse = combined_score(torch.cat(pairs[s][0]), torch.cat(pairs[s][1]), GRID)
                rows.append(dict(L=L, Vref=V, split=s, n=n, sse=sse, rmse=np.sqrt(sse / n), bias=se / n,
                                 S=float(S_), trmse=float(np.sqrt(float(timse)))))
            print(f'L={L:5.0f} Vref={V:5.0f}  train RMSE {rows[-2]["rmse"]:6.2f} S {rows[-2]["S"]:6.2f} (bias {rows[-2]["bias"]:+6.1f})  '
                  f'val RMSE {rows[-1]["rmse"]:6.2f} S {rows[-1]["S"]:6.2f} (bias {rows[-1]["bias"]:+6.1f})', flush=True)
out = Path(args.run) / f'fold{args.fold}' / 'accel_profile.csv'
pd.DataFrame(rows).assign(L_trained=L_tr, Vref_trained=V_tr).to_csv(out, index=False)
print(f'saved {out}')
