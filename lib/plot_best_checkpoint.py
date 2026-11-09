"""Load a fold's best.pt and redo the training diagnostic plots (boundary map + tracks) for its chunks.

Rebuilds the fold exactly as train_joint_forecast.py --fold does (same chunks, rescaling and feature
statistics from the training batches), loads NN1 / NN2 / learned acceleration from the checkpoint, runs
the forward pipeline without gradients and calls save_batch_diagnostic_plots. Also prints each chunk's
scored RMSE (warm-up excluded) and the pooled value, which should match val_rmse_scored at best_epoch.
load_fold / scored_predictions are reused by eval_matched_baselines.py.

  python plot_best_checkpoint.py --fold 1 [--split val|test|train] [--chunks 16,17] [--cuda-index 5]
Outputs: <run-dir>/plots_best/
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import folds
import nn1_inputs
from config import Config
from main import load_observations
from train_joint_forecast import (ACEResidualHead, FeatureVelocityMLP, PropagationForwardModel, _transient_skip_steps,
                                  forward_joint_pipeline, make_batch_record, save_batch_diagnostic_plots, setup_grid)


def load_fold(fold, run_dir, device, checkpoint='best.pt', chunks_per_batch=8, overlap_hours=150.0,
              spacecraft='ACE', feature_mlp_radius=0, nn2_hidden=64, no_speed_interp=None):
    """Rebuild a fold as in training and load its checkpoint. Returns a dict of everything needed to run chunks."""
    ckpt = torch.load(Path(run_dir) / checkpoint, map_location=device, weights_only=False)
    print(f"{Path(run_dir) / checkpoint}: epoch {ckpt['epoch']}, best_epoch {ckpt['best_epoch']}, "
          f"best_val_score {ckpt['best_val_score']:.4g}", flush=True)
    config = Config()
    config.device = str(device)
    config.overlap_hours = float(overlap_hours)
    config.spacecraft_include = [s.strip() for s in spacecraft.split(',') if s.strip()]
    config.feature_mlp_radius = int(feature_mlp_radius)
    grid = ckpt.get('grid', {'r_outer': 215.0, 'phi_grid_periodic': False})   # older checkpoints: legacy grid
    config.r_outer, config.phi_grid_periodic = float(grid['r_outer']), bool(grid['phi_grid_periodic'])
    config.Omega = float(grid.get('Omega', 2.97e-6))           # recorded since 2026-10-16; older checkpoints: 2.97e-6
    config.propagation_mode = grid.get('propagation_mode', 'solver')       # factorial test of the solver (2026-10-17)
    config.kinematic_lag_h = float(grid.get('kinematic_lag_h', 96.0))
    config.ballistic_sigma_h = float(grid.get('ballistic_sigma_h', 2.0))
    gap = grid.get('max_interp_gap_hours_speed', None)          # recorded since 2026-10-11
    if no_speed_interp is None:
        no_speed_interp = gap == 0
    if no_speed_interp:
        config.max_interp_gap_hours_speed = 0                       # measured hours only, as trained with --no-speed-interp
    config.nn1_lag_hours = ckpt['nn1_state_dict'].get('lag_hours', torch.zeros(0)).tolist()
    config.nn1_extra_channels = [nn1_inputs.EXTRA_NAMES[i] for i in ckpt['nn1_state_dict'].get('extra_codes', torch.zeros(0, dtype=torch.long)).tolist()]
    sd1 = ckpt['nn1_state_dict']
    if 'head.red_groups' in sd1:                                # one input per quantity (boundary.ReducedInputsHead)
        from boundary import GROUP_NAMES
        config.nn1_reduce = [GROUP_NAMES[i] for i in sd1['head.red_groups'].tolist()]
        config.nn1_head = 'nam' if 'head.inner.nam_key' in sd1 else 'hinge' if 'head.inner.hinge_lo' in sd1 else 'mlp'
        if config.nn1_head == 'hinge':                          # range buffers; starting points are overwritten by the load
            lo, hi = sd1['head.inner.hinge_lo'].tolist(), sd1['head.inner.hinge_hi'].tolist()
            config.hinge_spec = dict(lo=lo, hi=hi, init1=[(a + b) / 3 for a, b in zip(lo, hi)],
                                     init2=[2 * (a + b) / 3 for a, b in zip(lo, hi)])
        config.nam_smooth = float(sd1.get('head.smooth', torch.tensor(0.0)))
        if config.nn1_head == 'nam':
            config.nam_hidden = tuple(sd1['head.inner.nam_hidden'].tolist())
        config.mlp_hidden_dims = ()
    elif 'head.nam_key' in sd1:                                 # neural additive model head (boundary.NAMHead)
        config.nn1_head = 'nam'
        config.nam_key = sd1['head.nam_key'].tolist()
        config.nam_pairs = [tuple(p) for p in sd1['head.nam_pairs'].tolist()]
        config.nam_hidden = tuple(sd1['head.nam_hidden'].tolist())
        config.mlp_hidden_dims = ()
    else:
        w = sorted((int(k.split('.')[1]), v.shape[0]) for k, v in sd1.items()
                   if k.startswith('head.') and k.endswith('.weight'))
        config.mlp_hidden_dims = tuple(n for _, n in w[:-1])    # hidden widths as trained
    config.nn1_drop_hmi = bool(int(ckpt['nn1_state_dict'].get('drop_hmi_flag', torch.tensor(0))))
    config.nn1_dropout = float(ckpt['nn1_state_dict'].get('dropout_p', torch.tensor(0.0)))   # layer layout as trained
    rec_offs = ckpt['nn2_state_dict'].get('rec_offsets_h', torch.zeros(0)).tolist()
    config.cv_split, config.box_days = grid.get('cv_split', 'b5'), grid.get('box_days', None)   # recorded since 2026-10-15
    config.nn1_dataset_dir = grid.get('nn1_dataset_dir', 'data/nn1_dataset')
    config.nn1_heldout_dir = grid.get('nn1_heldout_dir', 'data/nn1_dataset_heldout')
    nn1_inputs.DATA = Path(config.nn1_dataset_dir)
    roles = None if config.cv_split == 'b5' else folds.fold_intervals(fold, config.cv_split)
    train_b = folds.FOLDS[fold][0] if roles is None else roles['train']
    obs_df = load_observations(config).sort_values('Time').reset_index(drop=True)
    obs_df = obs_df[obs_df['Spacecraft_ID'].isin(config.spacecraft_include)].reset_index(drop=True)
    if rec_offs:                                                  # NN2 recurrence inputs, as in training
        config.nn2_rec_offsets_h = rec_offs
        config.nn2_rec_series = obs_df[obs_df['Spacecraft_ID'] == 'ACE'].set_index('Time')['Speed'].sort_index().resample('1h').mean()
    specs = nn1_inputs.build_fold_batch_specs(config, obs_df, list(train_b), chunks_per_batch=chunks_per_batch,
                                              roles=roles, box_days=config.box_days)

    r, phi, t, dr, dphi = setup_grid(config)
    nn1 = FeatureVelocityMLP(config).to(device)
    nn2 = ACEResidualHead(history_steps=max(1, int(round(ckpt['history_hours']))), hidden=nn2_hidden, rec_offsets_h=rec_offs).to(device)
    forward_model = PropagationForwardModel(config, r, phi, dr, dphi).to(device)
    nn1.load_state_dict(ckpt['nn1_state_dict'])          # carries its own fill speed (450 for older checkpoints)
    nn2.load_state_dict(ckpt['nn2_state_dict'])
    accel = getattr(forward_model, 'learned_radial_accel', None)
    if ckpt.get('forward_extra_state_dict') is not None and accel is not None:
        accel.load_state_dict(ckpt['forward_extra_state_dict'], strict=False)
    for m in (nn1, nn2, forward_model):
        m.eval()
    return dict(fold=fold, ckpt=ckpt, config=config, specs={s['batch_idx']: s for s in specs}, obs_df=obs_df,
                r=r, phi=phi, nn1=nn1, nn2=nn2, forward_model=forward_model, device=device)


def chunks_of(F, split):
    if F['config'].cv_split != 'b5':                                  # specs carry their role name
        return [cid for cid, s in F['specs'].items() if s['dev_batch'] == split]
    train_b, val_b, test_b = folds.FOLDS[F['fold']]
    role = {'train': list(train_b), 'val': [val_b], 'test': [test_b]}[split]
    return [cid for cid, s in F['specs'].items() if s['dev_batch'] in role]


def run_chunk(F, cid):
    rec = make_batch_record(F['specs'][cid], F['config'], F['device'])
    with torch.no_grad():
        out = forward_joint_pipeline(rec, F['nn1'], F['nn2'], F['forward_model'], F['r'], F['phi'], F['config'])
    return rec, out


def scored_predictions(F, cid, rec=None, out=None):
    """Hourly (time, obs, pred) over the chunk's scored window (warm-up excluded, finite obs only)."""
    if rec is None:
        rec, out = run_chunk(F, cid)
    obs, pred = rec['ace_obs'].cpu().numpy(), out['ace_final'].cpu().numpy()
    times = rec['batch_start_time'] + pd.to_timedelta(rec['obs_tracks'][0]['time_obs_seconds'], unit='s')
    k0 = _transient_skip_steps(F['config'], obs.shape[0])
    ok = np.isfinite(obs[k0:])
    return pd.DataFrame({'time': np.asarray(times[k0:])[ok], 'obs': obs[k0:][ok], 'pred': pred[k0:][ok], 'chunk': cid})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--fold', type=int, required=True)
    p.add_argument('--run-dir', default=None, help='default runs/stage1.0a_baseline/fold<k>')
    p.add_argument('--checkpoint', default='best.pt', help='file inside --run-dir')
    p.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    p.add_argument('--chunks', default=None, help='comma-separated chunk ids (default: all chunks of the split)')
    p.add_argument('--chunks-per-batch', type=int, default=8)
    p.add_argument('--overlap-hours', type=float, default=150.0)
    p.add_argument('--spacecraft', default='ACE')
    p.add_argument('--feature-mlp-radius', type=int, default=0)
    p.add_argument('--nn2-hidden', type=int, default=64)
    p.add_argument('--cuda-index', type=int, default=5)
    args = p.parse_args()

    run_dir = Path(args.run_dir or f'runs/stage1.0a_baseline/fold{args.fold}')
    device = torch.device(f'cuda:{args.cuda_index}' if torch.cuda.is_available() else 'cpu')
    F = load_fold(args.fold, run_dir, device, args.checkpoint, args.chunks_per_batch, args.overlap_hours,
                  args.spacecraft, args.feature_mlp_radius, args.nn2_hidden)
    chunk_ids = [int(c) for c in args.chunks.split(',')] if args.chunks else chunks_of(F, args.split)
    plot_dir = run_dir / 'plots_best'
    sse = n = 0
    for cid in chunk_ids:
        rec, out = run_chunk(F, cid)
        save_batch_diagnostic_plots(rec, out, F['r'], F['phi'], F['config'], plot_dir, split_name=args.split,
                                    epoch=F['ckpt']['epoch'])
        d = scored_predictions(F, cid, rec, out)
        err = d.pred.to_numpy(float) - d.obs.to_numpy(float)
        sse += float((err ** 2).sum()); n += len(d)
        rmse = float(np.sqrt((err ** 2).mean())) if len(d) else float('nan')
        s = F['specs'][cid]
        print(f"  chunk {cid:2d} ({s['batch_start_time'][:10]} -> {s['batch_end_time'][:10]}): "
              f"scored RMSE {rmse:6.1f} km/s over {len(d)} h", flush=True)
    print(f"pooled scored RMSE ({args.split}, {len(chunk_ids)} chunks): {np.sqrt(sse / max(n, 1)):.2f} km/s; plots in {plot_dir}")
