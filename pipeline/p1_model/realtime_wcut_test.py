"""Where does error persistence stop helping? Paired block bootstrap of dS(tau) = S(errpers_ens) - S(lastseen_ens)
for every lead hour, from realtime_scores.py's rows (out-of-fold w). Resampling unit: 27-day blocks of issue
times (pooled over folds); S, including its percentile TIMSE grid, is recomputed in every resample.
Cut-off = first lead from which the 95% CI upper bound of dS stays >= 0 for all longer leads
(error persistence never again significantly better than the NN alone).

  python realtime_wcut_test.py [--dir runs/realtime_dev] [--B 1000] [--cuda-index 0]
Output: <dir>/wcut_bootstrap.csv (lead, n, dS, lo, hi, frac_boot_better)
"""
import argparse

import numpy as np
import pandas as pd
import torch

BLOCK_DAYS, K = 27, 100


def S_batch(o, e2):
    """S for each column: o [N, B] observations, e2 [N, B] squared errors (rows = resampled points)."""
    q = torch.quantile(o, torch.linspace(0, 1, K + 1, device=o.device, dtype=o.dtype), dim=0)      # [K+1, B]
    width = q[1:] - q[:-1]
    above = (o[None, :, :] >= q[:-1, None, :]).to(o.dtype)                                          # [K, N, B]
    cnt = above.sum(1)
    tail = (above * e2[None]).sum(1) / cnt.clamp(min=1.0)
    use = (cnt > 0) & (width > 0)
    timse = (width * tail * use).sum(0) / (o.max(0).values - o.min(0).values)
    return (e2.mean(0) * timse) ** 0.25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='runs/realtime_dev')
    ap.add_argument('--B', type=int, default=1000)
    ap.add_argument('--chunk', type=int, default=50, help='bootstrap replicates per GPU batch')
    ap.add_argument('--cuda-index', type=int, default=0)
    args = ap.parse_args()
    dev = torch.device(f'cuda:{args.cuda_index}')
    D = pd.read_pickle(f'{args.dir}/rows_lastseen.pkl')
    D = D[np.isfinite(D.obs) & np.isfinite(D.lastseen_ens) & np.isfinite(D.errpers_ens)]
    t_first = pd.Timestamp(D.issue.min())
    D['block'] = ((pd.to_datetime(D.issue) - t_first) / pd.Timedelta(days=BLOCK_DAYS)).astype(int)
    rng = np.random.default_rng(0)
    blocks = np.sort(D.block.unique())
    picks = [rng.choice(blocks, len(blocks)) for _ in range(args.B)]            # same block draws for every lead
    rows = []
    for lead, g in D.groupby('lead'):
        o = torch.tensor(g.obs.to_numpy(float), device=dev, dtype=torch.float64)
        e_a = torch.tensor((g.errpers_ens - g.obs).to_numpy(float) ** 2, device=dev, dtype=torch.float64)
        e_b = torch.tensor((g.lastseen_ens - g.obs).to_numpy(float) ** 2, device=dev, dtype=torch.float64)
        dS0 = float(S_batch(o[:, None], e_a[:, None]) - S_batch(o[:, None], e_b[:, None]))
        by_block = {b: np.flatnonzero(g.block.to_numpy() == b) for b in blocks}
        idxs = [np.concatenate([by_block[b] for b in pk]) for pk in picks]
        n_max = max(len(i) for i in idxs)
        ds = []
        for c in range(0, args.B, args.chunk):
            sub = idxs[c:c + args.chunk]
            # pad each replicate to n_max by repeating its own rows (cyclic), then weight the pads out
            I = torch.tensor(np.stack([np.resize(i, n_max) for i in sub], 1), device=dev)
            w = torch.tensor(np.stack([np.arange(n_max) < len(i) for i in sub], 1), device=dev)
            ds.append(_dS_padded(o, e_a, e_b, I, w))
        ds = torch.cat(ds).cpu().numpy()
        lo, hi = np.percentile(ds, [2.5, 97.5])
        rows.append(dict(lead=lead, n=len(g), dS=dS0, lo=lo, hi=hi, frac_boot_better=float(np.mean(ds < 0))))
        if lead % 12 == 0:
            print(f'lead {lead:3d} h: dS {dS0:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]', flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(f'{args.dir}/wcut_bootstrap.csv', index=False)
    sig = R.hi < 0                                                               # significantly better
    last_sig = int(R.lead[sig].max()) if sig.any() else 0
    print(f'last lead where error persistence is significantly better (95% CI below 0): {last_sig} h; '
          f'cut-off (w = 0 from) {last_sig + 1} h')
    print(f'leads > {last_sig} h where it is significantly WORSE (CI above 0): {int((R.lo > 0)[R.lead > last_sig].sum())}')


def _dS_padded(o, e_a, e_b, I, w):
    """dS for replicates with unequal lengths: rows beyond a replicate's length are excluded by giving them
    NaN observations, handled through nan-aware quantiles and masks."""
    oo = o[I].clone(); oo[~w] = float('nan')
    ea, eb = e_a[I], e_b[I]
    return _S_nan(oo, ea) - _S_nan(oo, eb)


def _S_nan(o, e2):
    m = torch.isfinite(o)
    q = torch.nanquantile(o, torch.linspace(0, 1, K + 1, device=o.device, dtype=o.dtype), dim=0)
    width = q[1:] - q[:-1]
    oz = torch.where(m, o, torch.full_like(o, -1e30))
    above = (oz[None] >= q[:-1, None, :]).to(o.dtype)
    e2m = torch.where(m, e2, torch.zeros_like(e2))
    cnt = above.sum(1)
    tail = (above * e2m[None]).sum(1) / cnt.clamp(min=1.0)
    use = (cnt > 0) & (width > 0)
    vmax = torch.where(m, o, torch.full_like(o, -1e30)).max(0).values
    vmin = torch.where(m, o, torch.full_like(o, 1e30)).min(0).values
    timse = (width * tail * use).sum(0) / (vmax - vmin)
    return (e2m.sum(0) / m.sum(0) * timse) ** 0.25


if __name__ == '__main__':
    main()
