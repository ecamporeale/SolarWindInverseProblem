"""Part I comparison (2026-10-24), boundary-only version: the WSA 4.5 boundary propagated with B0's own solver over
B0's test chunks, so that NN1 and WSA are compared with the propagation held fixed.

For every fold and seed (42, 43) of runs/sweep4/B0: the fold is rebuilt as in training (plot_best_checkpoint.load_fold),
and each test chunk is run through forward_joint_pipeline exactly as B0's test predictions are made, except that NN1's
output is replaced by the WSA boundary on the model grid: at every model time step the latest WSA 4.5 map at or before
it (part1_wsa_profiles.py: ACE latitude, 128 Carrington longitudes; interpolated periodically to the model longitudes),
brought from WSA's 1-AU-equivalent speed to the inner boundary by inverting this fold's own solver (uniform inner speed
v_in → steady speed at 1 AU, monotone; the counterpart of HUXt's map_v_inwards for our solver, including its trained
radial acceleration). Warm-up and scored hours as B0 (plot_best_checkpoint.scored_predictions). NN2 is disabled in B0.

--link ballistic (2026-10-25): the same WSA boundary through the factorial test's ballistic link (F_ball's operator:
train_joint_forecast.kinematic_track, every parcel at its own speed plus the trained ΔV(r), Gaussian kernel σ = 2 h where
arrivals overlap), so that WSA's solver-vs-ballistic difference is measured with the operators of Section "what the link
adds"; the inversion of the boundary is the same (a steady flow has no interaction).

  python pipeline/p1_model/part1_wsa_solver_test.py [--cuda-index 0] [--link ballistic]
Output: runs/part1_wsa/solver_test.csv  time, fold, chunk, seed, obs, wsa_solver; runs/part1_wsa/solver_inversion.csv
        (--link ballistic: runs/part1_wsa/fball_test.csv, column wsa_fball)
"""
import argparse
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, 'lib')
from forward_model import steady_state_propagate                         # noqa: E402
from plot_best_checkpoint import chunks_of, load_fold, scored_predictions  # noqa: E402
from train_joint_forecast import forward_joint_pipeline, make_batch_record  # noqa: E402

RUN = 'runs/sweep4/B0'


def inversion_curve(F, v_grid=np.arange(150.0, 951.0, 10.0), days=12.0):
    """Steady speed at 1 AU (r closest to 215 Rs) for a uniform inner-boundary speed, with this fold's solver."""
    fm, r, cfg = F['forward_model'], F['r'], F['config']
    nt = int(days * 86400 / cfg.dt)
    i1 = int(torch.argmin(torch.abs(r - 215.0)))
    out = []
    with torch.no_grad():
        for v in v_grid:
            vin = torch.full((F['phi'].numel(), nt), float(v), device=F['device'])
            v0 = steady_state_propagate(vin[:, 0], r, direction='outward', Omega=float(cfg.Omega))
            vf = fm(vin, v0)
            out.append(float(vf[i1, :, -1].mean()))                           # v_full: (Nr, Nphi, Nt)
    return v_grid, np.array(out)


class WSABoundary(torch.nn.Module):
    """Stands in for NN1: returns the prepared boundary [Nphi, Nt] whatever the image slice."""

    def __init__(self, v):
        super().__init__(); self.v = v

    def forward(self, image_slice, *args):
        return self.v


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cuda-index', type=int, default=3)
    ap.add_argument('--link', choices=['solver', 'ballistic'], default='solver')
    a = ap.parse_args()
    col, out_csv = ('wsa_solver', 'solver_test.csv') if a.link == 'solver' else ('wsa_fball', 'fball_test.csv')
    dev = torch.device(f'cuda:{a.cuda_index}')
    W = np.load('runs/part1_wsa/wsa45_profiles.npz', allow_pickle=True)
    mt = pd.DatetimeIndex(W['map_time']); V1 = W['v1au'].astype(float); LON = W['lon']
    rows, inv = [], []
    for seed in (42, 43):
        for fold in range(1, 6):
            F = load_fold(fold, f'{RUN}/seed{seed}/fold{fold}', dev)
            vg, vo_raw = inversion_curve(F)                                   # with the solver (steady flow: same for both links)
            F['config'].propagation_mode = a.link                             # 'ballistic': F_ball's link
            vo = np.maximum.accumulate(vo_raw) + 1e-6 * np.arange(vo_raw.size)   # enforce monotone for the inversion
            inv.append(pd.DataFrame({'seed': seed, 'fold': fold, 'v_in': vg, 'v_1au': vo_raw, 'v_1au_monotone': vo}))
            print(f'seed {seed} fold {fold}: inversion max non-monotone step {max(0.0, -np.diff(vo_raw).min()):.2f} km/s', flush=True)
            phi = F['phi'].cpu().numpy() % (2 * np.pi)
            dt = F['config'].dt
            for cid in chunks_of(F, 'test'):
                rec = make_batch_record(F['specs'][cid], F['config'], dev)
                nt = rec['image_slice'].shape[-1]
                times = pd.Timestamp(rec['batch_start_time']) + pd.to_timedelta(np.arange(nt) * dt, unit='s')
                k = np.searchsorted(mt.values, times.values, side='right') - 1
                if (k < 0).any():
                    print(f'seed {seed} fold {fold} chunk {cid}: no WSA map before {times[0]}', flush=True); continue
                prof = {kk: np.interp(np.interp(phi, LON, V1[kk], period=2 * np.pi), vo, vg) for kk in np.unique(k)}
                vin = torch.tensor(np.stack([prof[kk] for kk in k], 1), dtype=torch.float32, device=dev)   # [Nphi, Nt]
                with torch.no_grad():
                    out = forward_joint_pipeline(rec, WSABoundary(vin), F['nn2'], F['forward_model'], F['r'], F['phi'], F['config'])
                d = scored_predictions(F, cid, rec, out).rename(columns={'pred': col})
                d['fold'], d['seed'] = fold, seed
                rows.append(d)
            print(f'seed {seed} fold {fold}: done', flush=True)
    if a.link == 'solver':
        pd.concat(inv).to_csv('runs/part1_wsa/solver_inversion.csv', index=False)
    D = pd.concat(rows, ignore_index=True)
    D.to_csv(f'runs/part1_wsa/{out_csv}', index=False)
    print(f'saved runs/part1_wsa/{out_csv}: {len(D)} rows')
