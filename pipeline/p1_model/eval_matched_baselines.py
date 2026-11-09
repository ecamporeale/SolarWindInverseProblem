"""NN pipeline vs baselines on exactly the same scored test hours (development folds, folds.py).

For each fold: the run's best checkpoint is run on the fold's test-batch chunks (plot_best_checkpoint.load_fold),
giving hourly (time T, ACE obs, NN prediction) over the scored windows (warm-up excluded). Baselines at the same T:
  linear set-2a   OLS on the 8 set-2a pixel features (data/final_feature_matrix_pixel.pkl) of the image at T - 4 d
                  (the PAPER_1 regression); fitted per fold on rows whose image time AND target time both lie in
                  the fold's training batches; rows exist only where all candidate features were finite.
  27-day recurrence  hourly ACE speed at T - 27 d.
  climatology     mean ACE speed over the fold's training batches.
Truth for every method = the NN's scored ACE series, restricted to measured hours (hours whose ACE value was
interpolated are dropped; before 2026-10-07 they were included). Metrics add the combined score
S = sqrt(RMSE * sqrt(TIMSE)) with percentile TIMSE on a grid from the observations of the same row set.
Matched set = hours where NN, linear and recurrence all exist. Paired 27-day block-bootstrap 95% CIs of
RMSE(NN) - RMSE(baseline), pooled over folds.

  python eval_matched_baselines.py --run runs/stage1.0a_baseline [--cuda-index 5]
Outputs: <run>/fold<k>/test_predictions.csv (cached; delete to recompute), <run>/matched_baselines.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import folds

SET_2A = ['A: area f=0.30, w=20, off=0', 'B: ecliptic band |lat|<=15', 'A: area f=0.35, w=12.5, off=0',
          'D: mean log(I/med) of dark pixels, band |lat|<=30', 'B: hole-boundary distance, off=+10',
          'A: ESWF definition (f=0.35, w=7.5, off=0)', 'H: weak-field fraction <10 G, w=7.5, |lat|<=30',
          'H: mean |B_r|, w=7.5, |lat|<=30']
LEAD = pd.Timedelta(days=4)
B, BLOCK_DAYS = 2000, 27


def nn_test_predictions(run, fold, device):
    path = Path(run) / f'fold{fold}' / 'test_predictions.csv'
    if path.exists():
        return pd.read_csv(path, parse_dates=['time'])
    from plot_best_checkpoint import chunks_of, load_fold, scored_predictions
    F = load_fold(fold, Path(run) / f'fold{fold}', device)
    d = pd.concat([scored_predictions(F, cid) for cid in chunks_of(F, 'test')], ignore_index=True)
    d.to_csv(path, index=False)
    return d


def ace_hourly():
    from config import Config
    from main import load_observations
    o = load_observations(Config())
    o = o[o.Spacecraft_ID == 'ACE']
    return o.set_index(pd.to_datetime(o.Time))['Speed'].resample('1h').mean()


def linear_predictions(fold):
    X = pd.read_pickle('data/final_feature_matrix_pixel.pkl')
    t_img = pd.to_datetime(X.obstime).dt.round('h')
    T = t_img + LEAD
    b_img, b_T = folds.batch_of(t_img.to_numpy()), folds.batch_of(T.to_numpy())
    tr_b, _, te_b = folds.FOLDS[fold]
    tr = np.isin(b_img, tr_b) & np.isin(b_T, tr_b)
    A = np.column_stack([np.ones(len(X)), X[SET_2A].to_numpy(float)])
    coef = np.linalg.lstsq(A[tr], X.y.to_numpy(float)[tr], rcond=None)[0]
    te = b_T == te_b
    return pd.Series(A[te] @ coef, index=T[te].to_numpy()).groupby(level=0).mean()


def metrics(o, p, grid=None):
    """grid: fixed TIMSE percentile grid (train_joint_forecast.percentile_grid); default: from o itself."""
    from train_joint_forecast import combined_score, percentile_grid
    e = p - o
    S, _, timse = combined_score(torch.tensor(o), torch.tensor(p), percentile_grid(o) if grid is None else grid)
    out = dict(S=float(S), trmse=float(np.sqrt(float(timse))), rmse=np.sqrt(np.mean(e ** 2)), bias=np.mean(e), r=np.corrcoef(o, p)[0, 1], amp=np.std(p) / np.std(o))
    for v in (500, 600):
        hit, miss = np.sum((p >= v) & (o >= v)), np.sum((p < v) & (o >= v))
        fa, cn = np.sum((p >= v) & (o < v)), np.sum((p < v) & (o < v))
        pod = hit / (hit + miss) if hit + miss else np.nan
        pofd = fa / (fa + cn) if fa + cn else np.nan
        out[f'tss{v}'] = pod - pofd
        den = (hit + miss) * (miss + cn) + (hit + fa) * (fa + cn)
        out[f'hss{v}'] = 2.0 * (hit * cn - miss * fa) / den if den else np.nan
    return out


def block_ci(d, a, b_, seed=0):
    """Paired block bootstrap of RMSE(a) - RMSE(b_) over 27-day blocks (pooled rows)."""
    rng = np.random.default_rng(seed)
    blk = ((d.time - d.time.min()) / pd.Timedelta(days=BLOCK_DAYS)).astype(int).to_numpy()
    ids = np.unique(blk)
    sa = pd.Series((d[a] - d.obs) ** 2).groupby(blk).agg(['sum', 'count'])
    sb = pd.Series((d[b_] - d.obs) ** 2).groupby(blk).sum()
    diffs = []
    for _ in range(B):
        pick = rng.choice(ids, len(ids))
        n = sa['count'].loc[pick].sum()
        diffs.append(np.sqrt(sa['sum'].loc[pick].sum() / n) - np.sqrt(sb.loc[pick].sum() / n))
    return np.percentile(diffs, [2.5, 97.5])


def matched_table(run, device, ace=None):
    """Hourly table over all folds: time, obs, nn, linear, recurrence, climatology, fold (measured hours only)."""
    ace = ace_hourly() if ace is None else ace
    pooled = []
    for fold in sorted(folds.FOLDS):
        d = nn_test_predictions(run, fold, device).rename(columns={'pred': 'nn'})
        d['time'] = pd.to_datetime(d.time).dt.round('h')
        d = d[ace.reindex(d.time).notna().to_numpy()].reset_index(drop=True)          # measured hours only
        d['recurrence'] = ace.reindex(d.time - pd.Timedelta(days=27)).to_numpy()
        d['linear'] = linear_predictions(fold).reindex(d.time).to_numpy()
        tr_b = folds.FOLDS[fold][0]
        d['climatology'] = float(ace[np.isin(folds.batch_of(ace.index.to_numpy()), tr_b)].mean())
        d['fold'] = fold
        pooled.append(d)
    return pd.concat(pooled, ignore_index=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--cuda-index', type=int, default=5)
    args = p.parse_args()
    device = torch.device(f'cuda:{args.cuda_index}' if torch.cuda.is_available() else 'cpu')
    D = matched_table(args.run, device)
    rows = []
    for fold, d in D.groupby('fold', sort=True):
        m = d.dropna(subset=['linear', 'recurrence'])
        rows.append(dict(fold=fold, set='all scored', method='NN', n=len(d), **metrics(d.obs.to_numpy(), d.nn.to_numpy())))
        for meth in ('nn', 'linear', 'recurrence', 'climatology'):
            rows.append(dict(fold=fold, set='matched', method={'nn': 'NN'}.get(meth, meth), n=len(m),
                             **metrics(m.obs.to_numpy(), m[meth].to_numpy())))
    M = D.dropna(subset=['linear', 'recurrence'])
    rows.append(dict(fold='pooled', set='all scored', method='NN', n=len(D), **metrics(D.obs.to_numpy(), D.nn.to_numpy())))
    for meth in ('nn', 'linear', 'recurrence', 'climatology'):
        rows.append(dict(fold='pooled', set='matched', method={'nn': 'NN'}.get(meth, meth), n=len(M),
                         **metrics(M.obs.to_numpy(), M[meth].to_numpy())))
    res = pd.DataFrame(rows)
    res.to_csv(Path(args.run) / 'matched_baselines.csv', index=False)
    pd.set_option('display.width', 200)
    print(res.round(3).to_string(index=False))
    print(f'\nmatched hours: {len(M)} of {len(D)} scored test hours')
    for b_ in ('linear', 'recurrence', 'climatology'):
        lo, hi = block_ci(M.reset_index(drop=True), 'nn', b_)
        dr = metrics(M.obs.to_numpy(), M.nn.to_numpy())['rmse'] - metrics(M.obs.to_numpy(), M[b_].to_numpy())['rmse']
        print(f'RMSE(NN) - RMSE({b_}) = {dr:+.2f} km/s  95% CI [{lo:+.2f}, {hi:+.2f}]')
