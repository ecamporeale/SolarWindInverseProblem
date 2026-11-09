"""Bias–variance–noise decomposition of the Part I models on B0's exact information set (2026-10-22; worklog
INTERPRETABLE_BOUNDARY.md §13). Reframes the look-alike analysis.

For squared error, MSE = E[Var(Y|X)] (noise) + E[(E[Y|X] − E f(X))²] (bias²) + E[Var f(X)] (variance).

Stage 1 — inputs. Every measured development ACE hour T (rows of the look-alike matching, knn_backmap_test: observed
speed v, source time t_src = T − 193.5 R☉ / v, Carrington longitude facing Earth at t_src) is given the 37 inputs B0
receives for that source cell (longitude cell of B0's 128-cell grid nearest the source longitude, model step nearest
t_src), captured at the input of B0's head while running B0's input pipeline on all 40 development chunks with fold 1's
feature statistics (one standardization for all rows). Cache: data/knn_degeneracy/source_cell_inputs37.npz.

Stage 2 — noise, by the Gamma test (Stefánsson, Končar & Jones 1997; 95 % interval from 300 resamples of 27-day
blocks; γ_1, the nearest-neighbour value without extrapolation, is an upper bound): for k = 1..10 nearest neighbours,
δ_k = mean |x_i − x_nn(k)|², γ_k = mean ½(y_i − y_nn(k))²; the intercept of the regression of γ_k on δ_k is the noise
variance E[Var(Y|X)]. Distances on the 32 continuous inputs, re-standardized over all rows; neighbours must have the
same availability pattern (four lag-image flags and the recurrence flag), come from another development batch and be
at least 13 d away (the look-alike rule). Per regime: queries binned by the 50-neighbour mean of y (an estimate of
E[Y|X], so the bins condition on the inputs, not the outcome), Gamma test within each bin.

Stage 3 — models (out-of-fold test predictions of each fold model, seeds 42 and 43; the headline prediction is their
mean): MSE of the 2-seed mean; variance = ¼ mean (f42 − f43)² (variance of a 2-seed mean, seed-to-seed only: the
variation with the training set across folds is not included, so this is a lower bound); bias² = MSE − noise −
variance, split into offset² (mean error squared) and the rest. Models: B0, B0 trained with the MSE loss, the 37-input linear map (F_lin), the
37-input NAM with 3 pairs (G_nam), the 4-coefficient linear model (Q_lin) and the 4-curve NAM (Q_nam0.3); the last two
use less information than the 37 inputs, so their bias² includes the information they do not use.

  python pipeline/p1_model/noise_decomposition.py [--cuda-index 0]      (KNN_ALIGN_BETA=0: fixed 96-h lag, suffix _beta0;
                                                                        NOISE_SOURCE=traced: source cell traced backward
                                                                        through B0's out-of-fold solution, suffix _traced)
Output: runs/sweep5_lasso/noise_decomposition.csv, runs/sweep5_lasso/gamma_test.csv
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import folds

_beta = float(os.environ.get('KNN_ALIGN_BETA', '1'))   # source-time alignment, as in knn_backmap_test.py (0 = fixed 96 h)
TRACED = os.environ.get('NOISE_SOURCE', 'ballistic') == 'traced'   # source cell traced through B0's own solution
TAG = '_traced' if TRACED else ('' if _beta == 1 else f'_beta{_beta:g}')
CACHE = f'data/knn_degeneracy/source_cell_inputs37{TAG}.npz'
FLAGS = [12, 19, 26, 33, 36]
CONT = [i for i in range(37) if i not in FLAGS]
MODELS = {'B0': 'runs/sweep4/B0', 'B0_mse': 'runs/sweep4/B0_mse', 'F_lin': 'runs/sweep4/F_lin', 'G_nam': 'runs/sweep4/G_nam',
          'Q_lin': 'runs/sweep5_lasso/Q_lin', 'Q_nam0.3': 'runs/sweep5_lasso/Q_nam0.3'}
KMAX, KMEAN, MIN_DT_D = 10, 50, 13.0


def build_inputs(dev):
    from plot_best_checkpoint import load_fold
    from train_joint_forecast import make_batch_record
    import knn_backmap_test as KB
    F = load_fold(1, Path('runs/sweep4/B0/seed42/fold1'), dev)
    cap = {}
    F['nn1'].head.register_forward_hook(lambda m, i, o: cap.__setitem__('x', i[0]))
    T, src, col, v = KB.T, KB.src, np.asarray(KB.col, dtype=float), np.asarray(KB.v, dtype=float)
    X = np.full((len(T), 37), np.nan, np.float32)
    jcell = np.rint(col / 360.0 * 128).astype(int) % 128
    for cid, spec in sorted(F['specs'].items(), key=lambda kv: kv[1]['batch_start_time']):
        rec = make_batch_record(spec, F['config'], dev)
        with torch.no_grad():
            F['nn1'](rec['image_slice'])
        x = cap['x']                                                    # [128, Nt, 37]
        s0 = pd.Timestamp(spec['batch_start_time'])
        n = np.rint((src - s0) / pd.Timedelta(minutes=30)).to_numpy()
        todo = np.flatnonzero(np.isnan(X[:, 0]) & (n >= 0) & (n < x.shape[1]))
        if len(todo):
            vals = x[torch.as_tensor(jcell[todo], device=x.device), torch.as_tensor(n[todo].astype(int), device=x.device)]
            vals = vals.cpu().numpy()
            ok = np.abs(vals).sum(1) > 0                                 # invalid cells are all-zero
            X[todo[ok]] = vals[ok]
        print(f'chunk {cid}: {np.isfinite(X[:, 0]).sum():,} rows filled', flush=True)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, X=X, T=T.values.astype('datetime64[ns]'), src=src.values.astype('datetime64[ns]'), v=v, jcell=jcell)


def _interp(V, rr, pp, tt, r0, drg, dphig):
    """Trilinear interpolation of V [Nr, Nphi, Nt] at (r [Rs], φ [rad], τ [steps]); φ periodic, r and τ clamped."""
    Nr, Nphi, Nt = V.shape
    x = torch.clamp((rr - r0) / drg, 0, Nr - 1.0001)
    y = torch.remainder(pp / dphig, Nphi)
    z = torch.clamp(tt, 0, Nt - 1.0001)
    i0, j0, k0 = x.floor().long(), y.floor().long(), z.floor().long()
    fx, fy, fz = x - i0, y - j0, z - k0
    i1, j1, k1 = i0 + 1, (j0 + 1) % Nphi, k0 + 1
    out = 0
    for ii, wx in ((i0, 1 - fx), (i1, fx)):
        for jj, wy in ((j0, 1 - fy), (j1, fy)):
            for kk, wz in ((k0, 1 - fz), (k1, fz)):
                out = out + wx * wy * wz * V[ii, jj, kk]
    return out


def trace_to_boundary(V, r, dt, Omega, r_start, phi_start, tau_start):
    """Characteristics of the solved transport (dr/dt = v, dφ/dt = −Ω: radial parcels drift toward decreasing
    Carrington longitude; the solver's azimuthal wave speed is −Ω), integrated backward in time (RK2, one model
    step at a time) from ACE to the inner boundary r[0]. Returns the source step τ_src and longitude φ_src (NaN
    where the parcel does not reach the boundary inside the chunk)."""
    r0, drg = float(r[0]), float(r[1] - r[0])
    dphig = 2 * np.pi / V.shape[1]
    c = float(dt) / 695700.0                                                  # Rs per (km/s · step)
    w = -float(Omega) * float(dt)                                             # rad per step (dφ/dt = −Ω)
    live = torch.isfinite(r_start) & torch.isfinite(phi_start) & torch.isfinite(tau_start)   # hours with a position
    rr = torch.where(live, r_start, torch.full_like(r_start, r0 + 1.0))
    pp = torch.where(live, phi_start, torch.zeros_like(phi_start)); tt = torch.where(live, tau_start, torch.zeros_like(tau_start))
    tau_src = torch.full_like(rr, float('nan')); phi_src = torch.full_like(rr, float('nan'))
    while bool(live.any()):
        v1 = _interp(V, rr, pp, tt, r0, drg, dphig)
        v2 = _interp(V, rr - 0.5 * c * v1, pp - 0.5 * w, tt - 0.5, r0, drg, dphig)
        rn, pn, tn = rr - c * v2, pp - w, tt - 1.0
        cross = live & (rn <= r0)
        f = (rr - r0) / (rr - rn)                                             # fraction of the step to r0
        tau_src = torch.where(cross, tt - f, tau_src); phi_src = torch.where(cross, pp - f * w, phi_src)
        live = live & ~cross & (tn >= 0)
        rr, pp, tt = torch.where(live, rn, rr), torch.where(live, pn, pp), torch.where(live, tn, tt)
    return tau_src, torch.remainder(phi_src, 2 * np.pi)


def build_inputs_traced(dev, seed=42):
    """Source cell of every ACE hour from B0's own solution: the parcel arriving at ACE at T is traced backward through
    the solved field of the out-of-fold model (the fold whose test batch contains the chunk) to 21.5 Rs. Inputs are
    captured as in build_inputs (fold 1's feature statistics), at the traced cell."""
    from plot_best_checkpoint import load_fold
    from train_joint_forecast import make_batch_record
    from forward_model import steady_state_propagate
    import knn_backmap_test as KB
    FD = {k: load_fold(k, Path(f'runs/sweep4/B0/seed{seed}/fold{k}'), dev) for k in folds.FOLDS}
    test_fold = {te: k for k, (tr, va, te) in folds.FOLDS.items()}
    F1 = FD[1]
    cap = {}
    F1['nn1'].head.register_forward_hook(lambda m, i, o: cap.__setitem__('x', i[0]))
    T, v = KB.T, np.asarray(KB.v, dtype=float)
    X = np.full((len(T), 37), np.nan, np.float32)
    src = np.full(len(T), np.datetime64('NaT'), 'datetime64[ns]'); lon = np.full(len(T), np.nan)
    warm = float(getattr(F1['config'], 'overlap_hours', 150.0)) * 3600.0
    for cid, spec in sorted(F1['specs'].items(), key=lambda kv: kv[1]['batch_start_time']):
        Fk = FD[test_fold[spec['dev_batch']]]
        rec = make_batch_record(Fk['specs'][cid], Fk['config'], dev)
        with torch.no_grad():
            u = Fk['nn1'](rec['image_slice'])                                 # [Nphi, Nt]
            v0 = steady_state_propagate(u[:, 0], Fk['r'], direction='outward', Omega=float(Fk['config'].Omega))
            V = Fk['forward_model'](u, v0)                                    # [Nr, Nphi, Nt]
            rec1 = make_batch_record(spec, F1['config'], dev)
            F1['nn1'](rec1['image_slice'])
        x = cap['x']                                                          # [128, Nt, 37]
        s0 = pd.Timestamp(spec['batch_start_time']); dt = float(Fk['config'].dt)
        t_obs = s0 + pd.to_timedelta(rec['time_obs_t'].cpu().numpy().astype(float), unit='s')
        rows = pd.Index(T).get_indexer(t_obs.round('h'))
        sel = np.flatnonzero((rows >= 0) & np.isnan(X[np.maximum(rows, 0), 0]))
        if len(sel):
            rs = rows[sel]
            tau, ph = trace_to_boundary(V, Fk['r'], dt, Fk['config'].Omega, rec['radius_obs_t'][sel].float(),
                                        rec['phi_obs_t'][sel].float(), rec['time_obs_t'][sel].float() / dt)
            tau, ph = tau.cpu().numpy(), ph.cpu().numpy()
            ok = np.isfinite(tau) & (tau * dt >= warm)                        # source after the chunk's warm-up
            j = np.rint(ph[ok] / (2 * np.pi) * x.shape[0]).astype(int) % x.shape[0]
            n = np.clip(np.rint(tau[ok]).astype(int), 0, x.shape[1] - 1)
            vals = x[torch.as_tensor(j, device=x.device), torch.as_tensor(n, device=x.device)].cpu().numpy()
            good = np.abs(vals).sum(1) > 0                                    # invalid cells are all-zero
            X[rs[ok][good]] = vals[good]
            src[rs[ok]] = (s0 + pd.to_timedelta(tau[ok] * dt, unit='s')).values
            lon[rs[ok]] = np.rad2deg(ph[ok])
        print(f'chunk {cid}: {np.isfinite(X[:, 0]).sum():,} rows filled', flush=True)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, X=X, T=T.values.astype('datetime64[ns]'), src=src, v=v, src_lon=lon)


def knn(Xc, pattern, t_days, batch, dev, k):
    """Indices [n, k] of the k nearest admissible neighbours (same pattern, other batch, |ΔT| ≥ 13 d), squared distances."""
    n = len(Xc)
    idx = np.full((n, k), -1, np.int64); d2 = np.full((n, k), np.inf, np.float32)
    Xt = torch.tensor(Xc, device=dev); tt = torch.tensor(t_days, device=dev); bt = torch.tensor(batch, device=dev)
    for p in np.unique(pattern):
        members = np.flatnonzero(pattern == p)
        if len(members) < k + 1:
            continue
        M = torch.as_tensor(members, device=dev)
        for a in range(0, len(members), 2048):
            q = M[a:a + 2048]
            D = torch.cdist(Xt[q], Xt[M]) ** 2
            bad = (bt[q][:, None] == bt[M][None, :]) | ((tt[q][:, None] - tt[M][None, :]).abs() < MIN_DT_D)
            D[bad] = float('inf')
            dv, di = torch.topk(D, k, dim=1, largest=False)
            idx[members[a:a + 2048]] = members[di.cpu().numpy()]
            d2[members[a:a + 2048]] = dv.cpu().numpy()
    return idx, d2


def gamma(y, idx, d2, rows):
    """Gamma-test intercept (noise variance) and slope on the query rows; also δ_k, γ_k."""
    dk, gk = [], []
    rows = rows[np.isfinite(d2[rows, KMAX - 1]) & (idx[rows, KMAX - 1] >= 0)]        # queries with all KMAX neighbours
    for k in range(KMAX):
        dk.append(d2[rows, k].mean()); gk.append(0.5 * np.mean((y[rows] - y[idx[rows, k]]) ** 2))
    b, a = np.polyfit(dk, gk, 1)
    return a, b, dk, gk


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cuda-index', type=int, default=0)
    a = ap.parse_args()
    dev = torch.device(f'cuda:{a.cuda_index}')
    if not os.path.exists(CACHE):
        build_inputs_traced(dev) if TRACED else build_inputs(dev)
    C = np.load(CACHE)
    X, T, v = C['X'], pd.DatetimeIndex(C['T']), C['v']
    ok = np.isfinite(X).all(1) & np.isfinite(v)
    X, T, y = X[ok], T[ok], v[ok]
    print(f'rows with source-cell inputs: {ok.sum():,} of {len(ok):,}')
    pattern = ((X[:, FLAGS] > 0).astype(int) * (2 ** np.arange(len(FLAGS)))).sum(1)
    Xc = X[:, CONT]
    Xc = (Xc - Xc.mean(0)) / np.where(Xc.std(0) > 0, Xc.std(0), 1.0)
    t_days = ((T - T[0]) / pd.Timedelta(days=1)).to_numpy()
    batch = np.searchsorted(folds.EDGES.values, T.values, side='right') - 1           # development batch 0..4
    idx, d2 = knn(Xc, pattern, t_days, batch, dev, KMAX)                # Gamma test: 10 nearest, every pattern group
    idx50, _ = knn(Xc, pattern, t_days, batch, dev, KMEAN)                # regimes only: 50-neighbour mean of y
    rows = np.arange(len(y))
    A, B, dk, gk = gamma(y, idx, d2, rows)
    okr = rows[np.isfinite(d2[:, KMAX - 1]) & (idx[:, KMAX - 1] >= 0)]
    blk = (t_days[okr] // 27).astype(int); ub = np.unique(blk); grp = {b_: okr[blk == b_] for b_ in ub}
    rng = np.random.default_rng(0)
    boot = [gamma(y, idx, d2, np.concatenate([grp[b_] for b_ in rng.choice(ub, len(ub))]))[0] for _ in range(300)]
    lo95, hi95 = np.sqrt(np.percentile(boot, [2.5, 97.5]))
    print(f'noise RMS 95 % (27-day block bootstrap, 300): [{lo95:.1f}, {hi95:.1f}] km/s; nearest-neighbour bound '
          f'{np.sqrt(gk[0]):.1f} km/s')
    print('Gamma test, all hours: δ_k ' + ' '.join(f'{x:.2f}' for x in dk) + ' | γ_k ' + ' '.join(f'{x:.0f}' for x in gk))
    print(f'noise variance {A:.0f} km²/s² (noise RMS {np.sqrt(max(A, 0)):.1f} km/s); slope {B:.1f}; '
          f'var(y) {y.var():.0f} (s.d. {y.std():.1f})')
    ymean = np.array([y[idx50[i][idx50[i] >= 0]].mean() if (idx50[i] >= 0).any() else np.nan for i in range(len(y))])  # ≈ E[Y|X]
    fin = np.isfinite(ymean)
    edges = np.quantile(ymean[fin], [0, 0.5, 0.75, 0.9, 0.975, 1])
    regime = np.where(fin, np.clip(np.searchsorted(edges, np.nan_to_num(ymean), side='right') - 1, 0, 4), -1)
    G = [dict(regime='all', n=len(y), noise_var=A, noise_rms_lo95=lo95, noise_rms_hi95=hi95, slope=B, **{f'delta{k + 1}': dk[k] for k in range(KMAX)},
              **{f'gamma{k + 1}': gk[k] for k in range(KMAX)})]
    noise = {'all': A}
    for r in range(5):
        Ar, Br, dkr, gkr = gamma(y, idx, d2, np.flatnonzero(regime == r))
        lab = f'E[Y|X] {edges[r]:.0f}–{edges[r + 1]:.0f}'
        noise[r] = Ar
        G.append(dict(regime=lab, n=int((regime == r).sum()), noise_var=Ar, slope=Br,
                      **{f'delta{k + 1}': dkr[k] for k in range(KMAX)}, **{f'gamma{k + 1}': gkr[k] for k in range(KMAX)}))
    pd.DataFrame(G).to_csv(f'runs/sweep5_lasso/gamma_test{TAG}.csv', index=False)

    P = pd.DataFrame({'time': T, 'y': y, 'regime': regime})
    for key, path in MODELS.items():
        for s in (42, 43):
            q = pd.concat([pd.read_csv(f'{path}/seed{s}/fold{f}/test_predictions.csv', parse_dates=['time'])[['time', 'pred']]
                           for f in range(1, 6)]).drop_duplicates('time').rename(columns={'pred': f'{key}_{s}'})
            P = P.merge(q, on='time', how='left')
    P = P.dropna()
    print(f'rows with all model predictions: {len(P):,}')
    out = []
    for reg in ['all'] + list(range(5)):
        g = P if reg == 'all' else P[P.regime == reg]
        nv = noise[reg]
        for key in MODELS:
            f42, f43 = g[f'{key}_42'].to_numpy(), g[f'{key}_43'].to_numpy()
            f = 0.5 * (f42 + f43)
            e = f - g.y.to_numpy()
            mse, var = np.mean(e ** 2), 0.25 * np.mean((f42 - f43) ** 2)
            b2 = mse - nv - var
            out.append(dict(regime='all' if reg == 'all' else G[reg + 1]['regime'], n=len(g), model=key, rmse=np.sqrt(mse),
                            noise_rms=np.sqrt(max(nv, 0)), var_rms=np.sqrt(var), bias2_rms=np.sign(b2) * np.sqrt(abs(b2)),
                            offset=e.mean(), shape_bias_rms=np.sign(b2 - e.mean() ** 2) * np.sqrt(abs(b2 - e.mean() ** 2)),
                            noise_share=nv / mse, var_share=var / mse, bias2_share=b2 / mse))
    R = pd.DataFrame(out)
    R.to_csv(f'runs/sweep5_lasso/noise_decomposition{TAG}.csv', index=False)
    pd.set_option('display.width', 250)
    print(R.round(2).to_string(index=False))
