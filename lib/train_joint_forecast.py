"""
Joint whole-batch inverse + forecast training pipeline.

Pipeline:
    AIA/HMI -> NN1 (inner boundary, IntensityToVelocityMLP)
             -> physics propagation (PropagationForwardModel, trainable learned
                radial acceleration, exactly as in the inverse pipeline)
             -> NN2 residual head (ACEResidualHead), same-timestep:
                ace_final(T) = ace_pred(T) + delta, where delta is predicted
                from a single causal window ace_pred[T-history_steps, ..., T]
                (the physics pipeline's own prediction history, inclusive of
                T itself). NN2 never sees real ACE observations and takes no
                explicit lead-time input: it only reshapes the physics
                model's own local trajectory, which is itself derived from
                images via corotation and so is available causally at any
                forecast lead time. ace_pred(T) itself comes straight from
                the completed physics forward pass, not from NN2.

This is a new, independent branch forked from train_final.py at the user's
request. It does NOT depend on train_final.py, main.py, loss.py, or
inverse_solver's batch_inverse_solver() at runtime, and does not modify
their behavior: main.py, train_final.py, and loss.py are all untouched.

Key differences from train_final.py's stage-2 pipeline:
  - No rolling windows. Each batch is the full computational box (the whole
    Nt-step, multi-hundred-day window produced by inverse_solver's batch
    geometry) and is propagated through physics exactly once per training
    step, both for NN1's boundary loss and NN2's residual correction.
  - The boundary loss is computed fresh every step from a live forward pass
    (multi-spacecraft data misfit + phi/t curvature regularization, same
    structure as loss.py's LossFunction), not a supervised regression
    against a previously-saved v_inner. Unlike loss.py's LossFunction, the
    per-track data misfit uses average TIMSE (tail-weighted, using that
    track's own per-batch precomputed weights) instead of plain MSE, so it
    sits on the same scale as the forecast loss below and
    lambda_boundary/lambda_forecast behave as their nominal values suggest.
  - The forecast loss (for NN2's corrected primary-track prediction) is also
    average TIMSE, using a lookup precomputed once per batch from that
    batch's own primary-track observations -- not a single lookup shared
    across train/val/test.
  - The physics forward model's learned radial acceleration is trainable
    (jointly optimized with NN1/NN2), matching the original inverse code's
    optimizer grouping, instead of being frozen.
  - Batch-level split only: batches (not rolling windows within a batch) are
    assigned to train/val/test. Config's default grid (Nt=30618) yields
    exactly 4 batches; the default split is train=[0,1], val=[2], test=[3].
    Batch length is overridable per-run via --batch-days (e.g. 60 for ~2-month
    batches, ~44 of them over the full dataset) without touching config.py's
    Nt or main.py. --split-fractions does a chronological block split (e.g.
    "0.6,0.2,0.2") over however many batches that produces, instead of the
    explicit --train/val/test-batches index lists.
  - Both losses skip the v0-transient window at the start of each track
    (same overlap_hours-derived skip_steps as loss.py's LossFunction): v0 is
    a numerical necessity for solving the PDE, not something the boundary
    condition controls, so observations before the boundary-driven signal
    has had time to reach the observation point aren't scored.
  - Training batches are NOT averaged into one gradient step. Empirically,
    train batch 0 and batch 1's NN1 gradients had cosine similarity -0.93 --
    nearly directly opposing -- so averaging them destructively cancelled
    ~2/3 of each batch's signal every step, actively driving NN1's output
    toward a flat, low-variance collapse over training. Each train batch now
    gets its own optimizer step (order shuffled per epoch), with gradient
    clipping (max_norm=1.0, matching stage-1's optimization.py, which this
    script otherwise never touches).

Usage:
    python train_joint_forecast.py --epochs 300
"""

import argparse
import copy
import math
import os
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from boundary import IntensityToVelocityMLP, ThresholdVelocityModel, ContextualVelocityMLP, MultiFeatureThresholdModel, PolynomialVelocityModel, ShallowReluVelocityModel, MergedTemporalVelocityModel, FeatureVelocityMLP
from config import Config
from forward_model import PropagationForwardModel, steady_state_propagate
from helpers import sample_satellite_track, save_boundary_condition_plot, setup_grid
from inverse_solver import _build_obs_tracks_for_batch, _load_all_image_data, _prepare_batch_specs
from main import load_observations
from parallel import _get_primary_track

SEED = 42

# Make nohup/redirected runs emit progress immediately.
print = partial(print, flush=True)

CHECKPOINT_SCHEMA = "joint_forecast_v1"


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Joint whole-batch NN1 (boundary) + physics + NN2 (residual forecast) trainer"
    )
    parser.add_argument("--checkpoint", default="joint_forecast_best.pt", help="Best-checkpoint path")
    parser.add_argument("--resume", nargs="?", const="joint_forecast_best.pt", default=None, help="Resume checkpoint")
    parser.add_argument("--output-model", default="joint_forecast_final.pt", help="Final output model path")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs")
    parser.add_argument("--save-every", type=int, default=20, help="Save rolling checkpoint every N epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for NN1+NN2")
    parser.add_argument("--accel-lr-scale", type=float, default=None,
                         help="LR scale for the physics model's learned radial acceleration "
                              "(defaults to config.learned_radial_accel_lr_scale)")
    parser.add_argument("--disable-adaptive-lr", action="store_true",
                         help="Disable ReduceLROnPlateau (on by default, matching main.py's inverse "
                              "pipeline's config.adaptive_lr). When enabled, --lr/--accel-lr-scale are "
                              "starting values, not fixed for the whole run.")
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.8,
                         help="LR multiplier applied (to every param group) when val_forecast plateaus")
    parser.add_argument("--lr-scheduler-patience", type=int, default=20,
                         help="Epochs with no val_forecast improvement before reducing LR. This is "
                              "separate from --patience (early stopping) -- typically smaller, so LR "
                              "drops a few times before training gives up entirely.")
    parser.add_argument("--lr-scheduler-min-lr", type=float, default=1e-5, help="Floor for learning rate")

    parser.add_argument("--train-batches", default="0,1", help="Comma-separated train batch indices")
    parser.add_argument("--val-batches", default="2", help="Comma-separated val batch indices")
    parser.add_argument("--test-batches", default="3", help="Comma-separated test batch indices")
    parser.add_argument("--spacecraft", default=None,
                        help="Comma-separated Spacecraft_ID values to keep in the data-misfit loss "
                             "(e.g. 'ACE'). Default: every spacecraft in the observation file. "
                             "Batch geometry (start times, Nt, overlap) is computed from the FULL "
                             "observation set before this filter, so the split is unchanged by it.")
    parser.add_argument("--batch-days", type=float, default=None,
                         help="Override batch length in days (e.g. 60 for ~2-month batches). This "
                              "sets config.Nt for this process only, via _prepare_batch_specs' "
                              "existing model_nt/stride machinery -- config.py's Nt (and therefore "
                              "main.py's pipeline) is never touched. Default: None, keeps the "
                              "existing config.Nt-derived batch geometry (638 days, 4 batches).")
    parser.add_argument("--batch-overlap-hours", type=float, default=None,
                         help="Override config.overlap_hours: hours shared between consecutive "
                              "batches (controls stride) and hours discarded from the start of "
                              "every batch's loss to skip the v0 transient (see "
                              "_transient_skip_steps). Default: None, keeps config.py's value (100.0).")
    parser.add_argument("--split-fractions", default=None,
                         help="Comma-separated train,val,test fractions (e.g. '0.6,0.2,0.2'). "
                              "Chronological block split over ALL available batches in time order: "
                              "earliest fraction -> train, next -> val, remainder -> test. Overrides "
                              "--train-batches/--val-batches/--test-batches when given. Mutually "
                              "exclusive with --split-rotating-months.")
    parser.add_argument("--split-rotating-months", default=None,
                         help="Comma-separated train,val,test MONTH counts (e.g. '8,3,2'), repeating "
                              "as a cycle over the whole dataset instead of one single chronological "
                              "block each -- every split gets batches spread across every year/era "
                              "instead of being confined to one contiguous span. See "
                              "split_batches_by_rotating_months's docstring for why the specific "
                              "calendar months assigned to each split drift cycle to cycle when the "
                              "total doesn't evenly divide 12. Overrides --train-batches/--val-batches/"
                              "--test-batches and --split-fractions when given.")

    parser.add_argument("--ace-history-hours", type=float, default=9.0,
                         help="Depth (in output-cadence steps) of the causal window [T-h, T] NN2 "
                              "conditions on -- its own physics-track prediction history, inclusive "
                              "of T -- used to predict delta for T (same-timestep correction, no "
                              "real observations used)")
    parser.add_argument("--nn2-hidden", type=int, default=64, help="NN2 hidden width")
    parser.add_argument("--overfit-chunk", type=int, default=None,
                        help="--fold capacity test: train on this single chunk id only (val = next chunk, "
                             "test = the one after); combine with a large --patience to disable early stopping")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="Decoupled weight decay (AdamW) for all trained parameters; 0 = plain Adam")
    parser.add_argument("--nn1-dropout", type=float, default=0.0,
                        help="Feature NN1: dropout probability after each hidden layer (0 = none)")
    parser.add_argument("--nn1-drop-hmi", action="store_true",
                        help="--fold / feature NN1: use only the 6 AIA features (current and lagged), no HMI inputs")
    parser.add_argument("--nn2-rec-offsets", default=None,
                        help="NN2 recurrence inputs: comma-separated offsets in hours relative to T of the observed ACE "
                             "speed (measured hours only), e.g. -660,-654,-648,-642,-636 = T - 27 d +- 12 h at 6 h spacing")
    parser.add_argument("--nn1-only-epochs", type=int, default=0,
                        help="Two-phase training: NN1 alone (NN2 frozen at zero) for at most this many epochs or until "
                             "the validation plateau (--patience), whichever comes first; then the best NN1 state is "
                             "restored, NN2 is switched on and both are trained together. 0 = single phase")
    parser.add_argument("--phys-loss-weight", type=float, default=0.0,
                        help="Phase B (combined loss): total = S(final) + w * S(physics only), keeping NN1's own forecast "
                             "anchored while NN2 is trained")
    parser.add_argument("--nn1-extra", default=None,
                        help="--fold / feature NN1: comma-separated extra input channels from nn1_inputs.EXTRA_REGISTRY "
                             "(e.g. pfss_log10fs = daily PFSS expansion factor)")
    parser.add_argument("--nn1-lag-hours", default=None,
                        help="--fold / feature NN1: temporal context, comma-separated lags in hours (e.g. 12,24,48,72): "
                             "the same longitude's features at t - L are extra inputs")
    parser.add_argument("--nn1-fill", default="midpoint",
                        help="--fold / feature NN1: speed given to cells without valid AIA input: 'midpoint' "
                             "(450 km/s), 'train-median' (median observed speed of the selected spacecraft in "
                             "the fold's training batches), or a number in km/s")
    parser.add_argument("--disable-nn2", action="store_true",
                        help="Train NN1 + physics only: NN2 stays at its zero initialization (delta = 0, frozen, not in the optimizer) and lambda_boundary is pinned at 1 (forces --inner-floor 1).")
    parser.add_argument("--reg-weight", type=float, default=1e-4, help="Residual smoothness regularization weight")
    parser.add_argument("--sorted-weight", type=float, default=None,
                        help="Weight on the sorted/Wasserstein distribution term "
                             "mean((sort(pred)-sort(obs))^2), which penalizes amplitude "
                             "compression without biasing the mean. Defaults to 1.0 for "
                             "--loss-weighting mse+sorted, 0.0 otherwise.")
    parser.add_argument("--loss-weighting", choices=["timse", "uniform", "mse+sorted", "combined"], default="timse",
                         help="'timse' (default): average TIMSE, tail-weighted so rare high-speed points "
                              "count more. 'uniform': standard MSE, every point weighted equally. Applies "
                              "to both the boundary loss's per-track data misfit and the forecast loss. "
                              "'combined': S = sqrt(RMSE * sqrt(TIMSE)) with percentile TIMSE (101 quantiles of the "
                              "observations, velocity-width weights, / (vmax - vmin)), pooled over each optimizer "
                              "group on ONE fixed grid from all training observations; val/early stopping use S "
                              "pooled over the val chunks on the val-observation grid. Needs --batch-group-size > 1.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (numpy / torch)")
    parser.add_argument("--freeze-accel", action="store_true",
                        help="Keep the learned radial acceleration at its config values (L, Vref), not trained")
    parser.add_argument("--accel-vref", type=float, default=None,
                        help="Residual-acceleration amplitude Vref (km/s; default config.learned_radial_dv_vref = 40); "
                             "0 with --freeze-accel trains without residual acceleration (ablation N_no_accel, 2026-10-23)")
    parser.add_argument("--nn1-vmax", type=float, default=750.0, help="Feature NN1 output upper bound (km/s)")
    parser.add_argument("--no-speed-interp", action="store_true",
                        help="Do not interpolate gaps in the observed speed: loss and metrics use measured hours only "
                             "(spacecraft position is still interpolated)")
    parser.add_argument("--fold", type=int, choices=[1, 2, 3, 4, 5], default=None,
                        help="Use the set-2a feature dataset (data/nn1_dataset) and the date-defined development folds of folds.py: train/val/test batches = folds.FOLDS[fold]. Requires --nn1-arch feature. Overrides --train/--val/--test-batches.")
    parser.add_argument("--chunks-per-batch", type=int, default=1,
                        help="With --fold: split each development batch into this many contiguous chunks (shorter physics windows that can be trained as a group, see --batch-group-size). Each chunk keeps its batch's fold role.")
    parser.add_argument("--cv-split", choices=["b5", "il20"], default="b5",
                        help="With --fold: development split of folds.py. 'b5' (default, B0): five contiguous batches. "
                             "'il20': 20 interleaved chunks, round-robin roles, 54-d gaps around held-out chunks "
                             "(folds.fold_intervals).")
    parser.add_argument("--nn1-dataset-dir", default="data/nn1_dataset",
                        help="With --fold: NN1 input dataset (development). E2: data/nn1_dataset_deconv.")
    parser.add_argument("--nn1-heldout-dir", default="data/nn1_dataset_heldout",
                        help="Held-out dataset matching --nn1-dataset-dir; only recorded in the checkpoint, for "
                             "realtime_forecast.py --heldout. E2: data/nn1_dataset_heldout_deconv.")
    parser.add_argument("--box-days", type=float, default=None,
                        help="With --fold: cut every role interval into equal physics windows of at most this many "
                             "scored days (instead of --chunks-per-batch per b5 batch). Required with --cv-split il20.")
    parser.add_argument("--feature-mlp-radius", type=int, default=0,
                        help="Longitude stencil radius for --nn1-arch feature (default 0 = per-cell).")
    parser.add_argument("--nn1-arch", choices=["mlp", "threshold", "contextual", "multifeature", "polynomial", "shallow_relu", "merged_temporal", "feature"], default="mlp",
                        help="'mlp' = IntensityToVelocityMLP (10,077-param, 14-input pointwise MLP, "
                             "the original architecture). 'threshold' = ThresholdVelocityModel, a "
                             "4-parameter monotone coronal-hole-threshold mapping of a single input "
                             "(the pixel's own AIA intensity, no stencil, no HMI): "
                             "v = v_slow + (v_fast-v_slow)*sigmoid((I0-I)/w). Added after the "
                             "10,077-param MLP was found not to be identifiable from one training "
                             "batch (see SESSION_CHANGELOG.md); a monotone function of one variable "
                             "cannot reproduce that bang-bang failure mode. 'contextual' = "
                             "ContextualVelocityMLP, ~226 params, 4 inputs (central + wide-window "
                             "AIA, central HMI, wide-window magnetic unipolarity) through a small "
                             "unconstrained MLP -- adds non-local spatial context rather than more "
                             "shape flexibility on a single pixel.")
    parser.add_argument("--shallow-relu-radius", type=int, default=None,
                        help="Local stencil radius for ShallowReluVelocityModel (--nn1-arch "
                             "shallow_relu only): 1 (default, 6 inputs, ~803 params) or 2 "
                             "(10 inputs, ~1203 params). Overrides config.shallow_relu_radius.")
    parser.add_argument("--mlp-hidden-dims", default=None,
                        help="Comma-separated hidden-layer widths for IntensityToVelocityMLP's head "
                             "(--nn1-arch mlp only), e.g. '100,64,32' (default, 10,077 params) or "
                             "'27,16,8' (~998 params, same 14-input stencil and depth, much less "
                             "capacity). Overrides config.mlp_hidden_dims.")
    parser.add_argument("--group-lasso", type=float, default=0.0,
                        help="Group-lasso penalty on the linear boundary map (--mlp-hidden-dims none, 37 inputs; "
                             "2026-10-19): adds lambda * sqrt(p_g) * w_g / ||w_g|| to the gradient of each group of "
                             "weights (LINEAR_GROUPS: each AIA feature with its 4 lags, f_s, recurrence + flag; lag flags "
                             "and bias unpenalized). Units: training loss (S^2, km^2/s^2) per unit group norm.")
    parser.add_argument("--input-keep", default="",
                        help="Comma-separated LINEAR_GROUPS names to keep in the linear map; the weights of all other "
                             "groups are fixed at zero (relaxed-lasso refit). Empty: all.")
    parser.add_argument("--nn1-reduce", default="",
                        help="Comma-separated LINEAR_GROUPS names: reduce the 37 inputs to one per quantity (AIA features "
                             "averaged over the current and the available lagged images; boundary.ReducedInputsHead), "
                             "then a linear map, or a NAM with one curve per quantity with --nn1-head nam (2026-10-20).")
    parser.add_argument("--nam-smooth", type=float, default=0.0,
                        help="Curvature penalty on the NAM curves with --nn1-reduce: lambda * mean squared second "
                             "derivative over standardized inputs in [-3, 3], added to the combined-S training loss.")
    parser.add_argument("--hinge-spec", default="",
                        help="With --nn1-reduce and --nn1-head hinge: per quantity (order of --nn1-reduce) "
                             "'lo:init1:init2:hi' in standardized units (breakpoint range and starting points), "
                             "separated by commas (boundary.HingeHead; 2026-10-21).")
    parser.add_argument("--nn1-head", choices=["mlp", "nam", "hinge"], default="mlp",
                        help="Head of the per-cell feature model (--nn1-arch feature): 'mlp' (default, --mlp-hidden-dims) "
                             "or 'nam', a neural additive model with selected pairwise terms (boundary.NAMHead): "
                             "linear in all inputs + one small network curve per --nam-key input + one surface per "
                             "--nam-pairs pair (2026-10-18).")
    parser.add_argument("--nam-key", default="0,1,2,3,4,5,34,35",
                        help="Input indices (positions in the per-cell vector) that get a shape curve (--nn1-head nam).")
    parser.add_argument("--nam-pairs", default="0-35,0-34,4-35",
                        help="Pairs of input indices that get a 2-input surface, e.g. '0-35,0-34' ('' for none).")
    parser.add_argument("--nam-hidden", default="32,16", help="Hidden widths of every curve/surface network.")
    parser.add_argument("--propagation", choices=["solver", "fixed_lag", "ballistic"], default="solver",
                        help="Link from the inner boundary to the spacecraft: 'solver' (default, the transport "
                             "solver), or, for the factorial test of the solver, 'fixed_lag' (boundary value 96 h "
                             "earlier at the source longitude) or 'ballistic' (speed-dependent travel time, no "
                             "stream interaction). See kinematic_track.")
    parser.add_argument("--kinematic-lag-h", type=float, default=96.0)
    parser.add_argument("--ballistic-sigma-h", type=float, default=2.0)
    parser.add_argument("--contextual-hidden-dims", default=None,
                        help="Comma-separated hidden-layer widths for ContextualVelocityMLP's head "
                             "(--nn1-arch contextual only), e.g. '16,8' (default) or '16,16,8' for "
                             "the deeper ~2x-parameter variant. Overrides config.contextual_hidden_dims.")
    parser.add_argument("--merged-temporal-channels", choices=["both", "hmi_only"], default=None,
                        help="Input channels for MergedTemporalVelocityModel (--nn1-arch "
                             "merged_temporal only): 'both' (default, 28 inputs = AIA+HMI x2 "
                             "timesteps) or 'hmi_only' (14 inputs, drops all AIA-derived features "
                             "-- an ablation for how much skill comes from AIA vs. HMI alone). "
                             "Overrides config.merged_temporal_channels.")
    parser.add_argument("--aia-normalization", choices=["global", "asinh"], default=None,
                        help="How the AIA input is normalized before entering NN1: 'global' "
                             "(default, fixed dataset-wide mean/std, set once from the loaded "
                             "AIA data) or 'asinh' (clip at the dataset's 99th-percentile DN, "
                             "log-like asinh stretch, mapped to [-1,1]). Per-frame normalization "
                             "(each timestep centered on its own mean/std) was tried and reverted "
                             "-- it also destroys real cross-fold signal, not just the spurious "
                             "degradation trend -- and is not an available choice here. Overrides "
                             "config.aia_normalization_mode.")
    parser.add_argument("--sh-l-max", type=int, default=None, choices=[1, 2, 3, 5],
                        help="Milestone-6 global-SH ablation (--nn1-arch merged_temporal only): "
                             "adds (L_max+1)^2 low-order global-magnetic-state features (l=0 "
                             "retained) at t and t-1, on top of the unchanged 28-input local/"
                             "temporal model -- same hidden width, same head depth, same local "
                             "features. Default None reproduces the exact prior 28-input model. "
                             "Coefficients come from preprocess_hmi_sh.py (per-frame-normalized, "
                             "lambda=100 degree-damping ridge). Overrides config.merged_temporal_sh_l_max.")
    parser.add_argument("--exclusion-path", default=None,
                        help="CSV with a 'Time' column of ACE timestamps that are excluded from the "
                             "STANDARD dataset in ALL splits (train, val, test): ICME/CME (ICMECAT + "
                             "manual Gannon-storm extension) and despiked single-point spikes -- events "
                             "outside what this framework can predict. Applied after the pipeline's "
                             "short-gap interpolation (config.max_interp_gap_hours), which would otherwise "
                             "refill isolated NaN hours with interpolated pseudo-observations. "
                             "Typically data/preprocessed_window/ace_exclusion_log.csv.")
    parser.add_argument("--train-only-exclusion-path", default=None,
                        help="Path to a CSV with a 'Time' column of ACE timestamps to mask (NaN) in "
                             "the TRAINING loss only -- val/test batches always see the full, real "
                             "observed series. Recomputes the TIMSE lookup from the masked series. "
                             "See despike_ace_observations.py / cross_model_disagreement*.py for how "
                             "this file is built.")
    parser.add_argument("--nn1-init", default=None,
                         help="Optional path to a pretrained NN1 checkpoint (state_dict or "
                              "{'model_state_dict': ...}) used to warm-start the boundary network")

    parser.add_argument("--inner-floor", type=float, default=0.0, help="Minimum lambda_boundary")
    parser.add_argument("--schedule-transition-epochs", type=float, default=600.0,
                         help="Absolute number of epochs over which lambda_boundary decays from 1 to "
                              "inner_floor via a cosine ramp; lambda_boundary stays at inner_floor "
                              "afterward. This is an absolute epoch count, NOT a fraction of --epochs "
                              "-- e.g. NN1 gets a full 300 epochs before NN2 meaningfully switches on, "
                              "regardless of how long the overall run is or how --epochs is split "
                              "across --resume calls.")
    parser.add_argument("--restart-lambda-schedule", action="store_true",
                         help="On --resume, restart schedule_lambdas from epoch 0 (a fresh cosine ramp "
                              "over --schedule-transition-epochs) instead of continuing from the absolute "
                              "epoch counter. Without this flag, resuming past the original transition "
                              "point keeps lambda_boundary pinned at inner_floor for the whole resumed run.")
    parser.add_argument("--reset-lr-on-resume", action="store_true",
                        help="After loading a resume checkpoint, force the optimizer back to --lr "
                             "(and the accel group back to --lr * accel_lr_scale) and clear the "
                             "ReduceLROnPlateau state. Without this, --lr is silently ignored on "
                             "resume because the checkpoint's own (possibly decayed) lr is restored.")
    parser.add_argument("--reset-best-val-score", action="store_true",
                         help="On --resume, discard the checkpoint's best_val_score/best_epoch/"
                              "epochs_without_improvement (reset to inf/-1/0) instead of carrying them "
                              "forward. Model/optimizer/scheduler state and start_epoch are still loaded "
                              "normally. Use this when resuming under a different --loss-weighting (or any "
                              "other change to the val metric's scale), since the old best_val_score is on "
                              "the previous loss's scale and comparing new scores against it is meaningless "
                              "-- worse, it can silently prevent any new checkpoint from ever being saved "
                              "as best if the old value happens to be smaller.")

    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Training device selection")
    parser.add_argument("--cuda-index", type=int, default=None, help="Preferred CUDA GPU index (e.g., 1)")
    parser.add_argument("--multi-gpu", action="store_true",
                         help="DEPRECATED / not recommended: threads a forward+backward per batch across "
                              "multiple GPUs. Empirically measured SLOWER than single-GPU (0.71x) because "
                              "this workload is Python-GIL-bound (the rk2_tvd_split stepping loop issues "
                              "many small CUDA kernel launches per step, which is a CPU-side, GIL-holding "
                              "operation threading cannot parallelize). Kept for reference; use "
                              "--batch-group-size instead, which fixes the actual bottleneck.")
    parser.add_argument("--gpu-indices", default=None,
                         help="Comma-separated CUDA device indices to use with --multi-gpu (e.g. "
                              "'0,1,2,3'). Default: all visible CUDA devices.")
    parser.add_argument("--batch-group-size", type=int, default=1,
                         help="Process this many train batches per optimizer step as one shared batched "
                              "physics forward pass (see forward_model.py's forward_batched), instead of "
                              "one pass per batch. This fixes the real bottleneck (Python/kernel-launch "
                              "overhead in the 2880-ish-iteration rk2_tvd_split loop, not GPU compute) by "
                              "widening each iteration instead of running more of them -- unlike "
                              "--multi-gpu, this is genuinely faster, not just parallel. Default 1 "
                              "reproduces the original fully-alternating train_one_epoch exactly (one "
                              "batch per 'group', so this flag is a no-op then). Values >1 average the "
                              "group's losses into ONE combined gradient/optimizer step (standard "
                              "minibatch SGD) instead of one step per batch -- see "
                              "train_one_epoch_batched_group's docstring for why that's considered safe "
                              "now (gradient-conflict re-diagnosis on the 45-batch pool) after being "
                              "avoided earlier in this project for just 2 batches. Requires all batches to "
                              "have equal Nt (true whenever --batch-days is fixed, as it is for the whole "
                              "dataset here). Mutually exclusive with --multi-gpu.")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0,
                         help="Max gradient norm per training step (matches stage-1's optimization.py). "
                              "0 disables clipping.")
    parser.add_argument("--print-every", type=int, default=5, help="Epoch logging frequency")
    parser.add_argument("--patience", type=int, default=50, help="Early-stop patience (epochs without val improvement)")
    parser.add_argument("--eval-every", type=int, default=1,
                         help="Run val evaluation (and the LR scheduler step / best-checkpoint check that "
                              "depend on it) only every N epochs instead of every epoch, to skip its cost "
                              "on the skipped epochs. The logged val metrics on skipped epochs are just the "
                              "last computed ones (stale, not re-evaluated). --patience then counts "
                              "evaluations without improvement, not epochs -- e.g. patience=50 with "
                              "eval-every=5 tolerates up to 250 epochs without improvement, not 50. The "
                              "final epoch of the run always evaluates regardless of this setting. Default "
                              "1 evaluates every epoch (unchanged behavior).")

    parser.add_argument("--plot-every", type=int, default=10,
                         help="Save reconstructed-boundary + predicted-vs-observed track plots for every "
                              "train and val batch every N epochs. 0 disables plotting. Default is 1 "
                              "(every epoch) -- raise this for long runs once you've confirmed things work.")
    parser.add_argument("--plot-dir", default="joint_forecast_plots",
                         help="Directory for diagnostic plots (only used when --plot-every > 0)")

    return parser.parse_args()


# =============================================================================
# Device selection (mirrors train_final.py's convention)
# =============================================================================

def _query_gpu_memory_state():
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        return []

    rows = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            idx = int(parts[0])
            mem_used = float(parts[1])
            mem_total = float(parts[2])
        except ValueError:
            continue
        rows.append({"index": idx, "mem_used": mem_used, "mem_free": max(mem_total - mem_used, 0.0)})
    return rows


def choose_training_device(args):
    requested = str(args.device).strip().lower()
    if requested == "cpu":
        return torch.device("cpu"), None

    if not torch.cuda.is_available():
        if requested == "cuda":
            raise RuntimeError("--device=cuda requested but CUDA is not available")
        if args.cuda_index is not None:
            # An explicit GPU index is an unambiguous request for a GPU. Falling
            # back to CPU here is not benign: the CPU path has no GPU memory
            # gating and one such run held ~275 GB of host RAM, enough for 16
            # concurrent launches to OOM a shared server (CUDA error 802 after
            # a host reboot with nvidia-fabricmanager down).
            raise RuntimeError(
                f"--cuda-index={args.cuda_index} was requested but torch.cuda.is_available() is "
                f"False (check `python -c 'import torch; torch.cuda.device_count()'` for CUDA "
                f"error 802 / nvidia-fabricmanager). Refusing to silently fall back to CPU; "
                f"pass --device cpu if CPU is really intended."
            )
        return torch.device("cpu"), None

    n_cuda = torch.cuda.device_count()
    if args.cuda_index is not None:
        idx = int(args.cuda_index)
        if idx < 0 or idx >= n_cuda:
            raise RuntimeError(f"--cuda-index={idx} is out of range [0, {n_cuda - 1}]")
        torch.cuda.set_device(idx)
        return torch.device(f"cuda:{idx}"), idx

    if requested == "cuda":
        torch.cuda.set_device(0)
        return torch.device("cuda:0"), 0

    gpu_rows = _query_gpu_memory_state()
    if gpu_rows:
        best = max(gpu_rows, key=lambda row: (row["mem_free"], -row["mem_used"]))
        idx = int(best["index"])
        if 0 <= idx < n_cuda:
            torch.cuda.set_device(idx)
            return torch.device(f"cuda:{idx}"), idx

    torch.cuda.set_device(0)
    return torch.device("cuda:0"), 0


# =============================================================================
# Batch-index / split parsing (mirrors train_final.py's convention)
# =============================================================================

def parse_batch_list(text: str) -> List[int]:
    if text is None or text.strip() == "":
        return []
    return [int(tok.strip()) for tok in text.split(",") if tok.strip()]


def split_batches_by_fraction(available_batch_ids: List[int], fractions: List[float]):
    """Chronological block split: earliest frac[0] of batches -> train, next frac[1] -> val, rest -> test.

    Batch indices increase with t_start (see _build_batch_specs), so sorting
    available_batch_ids is sorting by time. This is a genuine forward-in-time
    extrapolation split, not an IID one -- train/val/test will generally
    differ in solar-cycle phase.
    """
    if len(fractions) != 3:
        raise RuntimeError(f"--split-fractions needs exactly 3 values (train,val,test), got {fractions}")
    total = sum(fractions)
    if total <= 0 or any(f < 0 for f in fractions):
        raise RuntimeError(f"--split-fractions must be non-negative and sum > 0, got {fractions}")

    ids = sorted(available_batch_ids)
    n = len(ids)
    if n < 3:
        raise RuntimeError(f"Need at least 3 batches for a train/val/test split, got {n}")

    norm = [f / total for f in fractions]
    n_train = min(max(1, round(norm[0] * n)), n - 2)
    n_val = min(max(1, round(norm[1] * n)), n - n_train - 1)
    train_batches = ids[:n_train]
    val_batches = ids[n_train:n_train + n_val]
    test_batches = ids[n_train + n_val:]
    return train_batches, val_batches, test_batches


def split_batches_by_rotating_months(batch_specs: List[Dict], train_months: int, val_months: int, test_months: int):
    """Repeating calendar-month block split: train_months consecutive months -> train,
    next val_months -> val, next test_months -> test, then repeat.

    Each batch's split is decided by which month its batch_start_time falls in,
    via month_index (months since the dataset's first batch, monotonically
    increasing) modulo the cycle length (train_months + val_months +
    test_months). Unlike split_batches_by_fraction (one single chronological
    train/val/test block each), this keeps every split represented across the
    WHOLE dataset span, at the cost of held-out blocks that are short and
    close to training data in time (more like standard cross-validation than
    a held-out future).

    When the cycle length does not evenly divide 12 (e.g. 8+3+2=13), which
    calendar month lands on which cycle position drifts by one month every
    time the cycle wraps -- so over several years, val/test see a broad mix
    of calendar months/seasons rather than being pinned to the same ones
    every year, which is the point: it avoids a systematic seasonal or
    solar-cycle-phase bias between splits.
    """
    cycle = int(train_months) + int(val_months) + int(test_months)
    if cycle <= 0 or train_months < 0 or val_months < 0 or test_months < 0:
        raise RuntimeError(
            f"--split-rotating-months values must be non-negative and sum > 0, "
            f"got train={train_months} val={val_months} test={test_months}"
        )

    specs_sorted = sorted(batch_specs, key=lambda s: s['batch_idx'])
    start_times = [pd.to_datetime(s['batch_start_time']) for s in specs_sorted]
    origin = min(start_times)

    train_batches, val_batches, test_batches = [], [], []
    for spec, start_time in zip(specs_sorted, start_times):
        month_index = (start_time.year - origin.year) * 12 + (start_time.month - origin.month)
        position = month_index % cycle
        if position < train_months:
            train_batches.append(spec['batch_idx'])
        elif position < train_months + val_months:
            val_batches.append(spec['batch_idx'])
        else:
            test_batches.append(spec['batch_idx'])
    return train_batches, val_batches, test_batches


def validate_split(train_batches: List[int], val_batches: List[int], test_batches: List[int],
                    available_batch_ids: List[int]):
    train_set, val_set, test_set = set(train_batches), set(val_batches), set(test_batches)

    if train_set & val_set:
        raise RuntimeError(f"Split overlap between train and val: {sorted(train_set & val_set)}")
    if train_set & test_set:
        raise RuntimeError(f"Split overlap between train and test: {sorted(train_set & test_set)}")
    if val_set & test_set:
        raise RuntimeError(f"Split overlap between val and test: {sorted(val_set & test_set)}")

    if len(train_batches) < 1:
        raise RuntimeError("Train split must contain at least 1 batch")
    if len(val_batches) < 1:
        raise RuntimeError("Val split must contain at least 1 batch")
    if len(test_batches) < 1:
        raise RuntimeError("Test split must contain at least 1 batch")

    selected = train_set | val_set | test_set
    available = set(available_batch_ids)
    missing = sorted(selected.difference(available))
    if missing:
        raise RuntimeError(f"Requested batch indices not produced by the current config: {missing}")


# =============================================================================
# Tail-weighted (TIMSE) forecast loss.
#
# This is PRECOMPUTE_TAIL_WEIGHTS / TAIL_LOSS from the spec:
#   thresholds = sorted unique finite training targets (minus the endpoints)
#   alpha_j    = (thr[j+1] - thr[j]) / max(1, count(target >= thr[j]))
#   weight(y)  = cumulative sum of alpha up to the largest threshold <= y
# so points in a sparsely-populated high-speed tail get a larger weight.
# =============================================================================

def build_timse_lookup(reference: torch.Tensor):
    """PRECOMPUTE_TAIL_WEIGHTS(training_targets)."""
    ref = torch.as_tensor(reference, dtype=torch.float32).flatten().cpu()
    ref = ref[torch.isfinite(ref)]
    if ref.numel() < 2:
        return None

    uniq = torch.unique(ref)
    uniq, _ = torch.sort(uniq)
    if uniq.numel() < 2:
        return None

    counts = torch.stack([(ref >= thr).sum() for thr in uniq[:-1]]).to(torch.float32).clamp(min=1)
    alpha = (uniq[1:] - uniq[:-1]) / counts
    return {
        "thresholds": uniq[:-1],
        "cum_alpha": torch.cumsum(alpha, dim=0),
    }


def timse_weights_from_lookup(obs: torch.Tensor, lookup) -> torch.Tensor:
    obs = torch.as_tensor(obs, dtype=torch.float32)
    if lookup is None:
        return torch.ones_like(obs)

    thr = lookup["thresholds"].to(obs.device)
    cum = lookup["cum_alpha"].to(obs.device)

    idx = torch.searchsorted(thr, obs, right=True) - 1
    w = torch.zeros_like(obs)
    valid = idx >= 0
    w[valid] = cum[idx[valid]]
    return w


def timse_loss_average(obs: torch.Tensor, pred: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Average TIMSE: TAIL_LOSS(y_true, y_pred, lookup) / n_valid, with pre-resolved weights.

    The original TAIL_LOSS pseudocode is sum(weights * (y - p)^2), which scales
    with both the number of points and the sum of weights -- on a track with
    ~15000 points that lands orders of magnitude above a plain per-point MSE,
    which silently breaks any lambda schedule combining it with another loss.
    Dividing by the valid point count keeps it on a per-point (km/s)^2 scale
    comparable to MSE while preserving the same tail-weighting *ranking*.
    """
    valid = torch.isfinite(obs) & torch.isfinite(pred)
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        return torch.tensor(0.0, device=pred.device)

    o = obs[valid]
    p = pred[valid]
    w = weights[valid] if weights is not None else torch.ones_like(o)
    return torch.sum(w * (o - p) ** 2) / n_valid


def finite_rmse(obs: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    valid = torch.isfinite(obs) & torch.isfinite(pred)
    if valid.sum() == 0:
        return torch.tensor(float("nan"), device=pred.device)
    return torch.sqrt(torch.mean((obs[valid] - pred[valid]) ** 2))


# =============================================================================
# Residual network
# =============================================================================

def shift_right(x: torch.Tensor, steps: int) -> torch.Tensor:
    if steps <= 0 or x.numel() == 0:
        return x
    if x.ndim != 1:
        x = x.reshape(-1)
    if x.shape[0] <= 1:
        return x.clone()
    steps = min(int(steps), x.shape[0] - 1)
    pad = x[:1].expand(steps)
    return torch.cat([pad, x[:-steps]], dim=0)


class ACEResidualHead(nn.Module):
    """Same-timestep residual head: ace_final(T) = ace_pred(T) + delta(T).

    delta(T) is predicted purely from the physics pipeline's own local
    trajectory shape: a causal window ace_pred[T-history_steps .. T]
    (inclusive of T itself). No real ACE observations and no explicit lead
    time are used anywhere in this module. Because ace_pred is itself
    derived only from images via NN1 -> inner boundary -> forward model ->
    corotation, this correction is causally valid and deployable at any
    forecast lead time -- it never depends on data from the window being
    forecast, and the correction function itself doesn't depend on issue
    time, so no lead-time input is needed. No image context is used.
    """

    REC_CENTER, REC_SCALE = 400.0, 100.0      # standardisation of the recurrence speeds (km/s)

    def __init__(self, history_steps: int, hidden: int = 64, rec_offsets_h=()):
        super().__init__()
        self.history_steps = int(history_steps)
        # optional recurrence inputs: observed ACE speed at T + offset (hours, e.g. -27 d -12 h ... +12 h), each with a
        # missing-value flag; values come from the full measured ACE record (record['ace_rec'], [Nt, n_rec])
        self.register_buffer('rec_offsets_h', torch.tensor([float(x) for x in rec_offsets_h], dtype=torch.float32))
        self.n_rec = len(rec_offsets_h)

        # window ace_pred[T-history_steps..T], inclusive of T
        in_dim = self.history_steps + 1 + 2 * self.n_rec

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        state_dict.setdefault(prefix + 'rec_offsets_h', torch.zeros(0))        # pre-recurrence checkpoints
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, ace_pred: torch.Tensor, rec: torch.Tensor = None) -> torch.Tensor:
        # ace_pred: [Nt], on the ACE track's time grid.
        # delta at output index t = T uses window [t, t-1, ..., t-history_steps]
        # (i.e. [T-history_steps, ..., T]), so it depends only on ace_pred at
        # indices <= T -- strictly causal, and uses no observation data at all.
        nt = int(ace_pred.shape[0])
        if nt <= 0:
            raise ValueError("NN2 received empty time horizon")

        ace_pred_ctx = torch.nan_to_num(ace_pred, nan=0.0, posinf=0.0, neginf=0.0)

        pred_window = [shift_right(ace_pred_ctx, k) for k in range(self.history_steps + 1)]
        feat = torch.stack(pred_window, dim=1)  # [Nt, history_steps + 1]
        if self.n_rec:
            if rec is None:
                raise ValueError('NN2 has recurrence inputs but the record carries no ace_rec')
            ok = torch.isfinite(rec)
            val = torch.where(ok, (rec - self.REC_CENTER) / self.REC_SCALE, torch.zeros_like(rec))
            feat = torch.cat([feat, val.to(feat.dtype), ok.to(feat.dtype)], dim=1)

        delta = 0.1 * self.net(feat).squeeze(-1)
        delta = torch.clamp(delta, -200.0, 200.0)
        return delta


# =============================================================================
# Lambda schedule
#
# Early epochs: lambda_boundary dominant. Later epochs: lambda_forecast
# dominant. The transition follows a cosine decay completing at an absolute
# epoch count (--schedule-transition-epochs, not a fraction of --epochs);
# lambda_boundary is held at inner_floor afterward.
# =============================================================================

def schedule_lambdas(epoch: int, transition_epochs: float, inner_floor: float = 0.0):
    """Cosine ramp of lambda_boundary from 1 -> inner_floor over `transition_epochs` epochs.

    `transition_epochs` is an ABSOLUTE epoch count, not a fraction of --epochs.
    A fraction-of---epochs design (the previous tau_frac parameterization) means
    the same tau_frac gives wildly different absolute timing depending on what
    --epochs happens to be passed on a given invocation -- fragile, and
    especially confusing across --resume calls with varying --epochs. This
    lets you directly say "NN1 should get N epochs before NN2 meaningfully
    switches on," independent of the total run length.
    """
    transition_epochs = max(1e-6, float(transition_epochs))
    p = min(max(float(epoch) / transition_epochs, 0.0), 1.0)

    lam_boundary = 0.5 * (1.0 + math.cos(math.pi * p))
    lam_boundary = max(float(inner_floor), lam_boundary)
    lam_forecast = 1.0 - lam_boundary
    return lam_boundary, lam_forecast


# =============================================================================
# Global-SH coefficient loading (milestone 6): mirrors _slice_image_batch's
# nearest-neighbor sampling exactly, but operates on preprocess_hmi_sh.py's
# [N_coef, T_hourly] array instead of an [Nphi, T_hourly] image array. Kept
# self-contained here rather than touching inverse_solver.py's shared
# _build_batch_specs/_slice_image_batch, since those are used by several
# other entry points (baseline_NN1.py, train_final.py, ...) unrelated to
# this ablation.
# =============================================================================

_sh_coeff_cache = {}


def _load_sh_coeff_data(config):
    """Load preprocess_hmi_sh.py's output once per process, filling short
    NaN gaps (missing HMI files) by linear interpolation per coefficient,
    same spirit as inverse_solver's _fill_short_nan_gaps. Cached since
    every fold in a sweep loads the same file."""
    path = str(getattr(config, 'sh_data_path'))
    if path in _sh_coeff_cache:
        return _sh_coeff_cache[path]

    data = np.load(path, allow_pickle=True)
    times = pd.to_datetime(np.asarray(data['times']))
    valid = np.asarray(data['valid'])
    print(f"Loaded global-SH coefficients: {path} ({valid.sum()}/{len(valid)} hours valid)")

    out = {'times': times}
    max_gap = 6  # hours; longer gaps are left as NaN (zeroed downstream, rare)
    for key in data.files:
        if not key.startswith('coeffs_L'):
            continue
        arr = np.asarray(data[key], dtype=np.float64).copy()  # [N_coef, T]
        for c in range(arr.shape[0]):
            s = pd.Series(arr[c])
            arr[c] = s.interpolate(method='linear', limit=max_gap, limit_direction='both').to_numpy()
        n_nan_left = int(np.isnan(arr).any(axis=0).sum())
        if n_nan_left > 0:
            print(f"  {key}: {n_nan_left}/{arr.shape[1]} hours still NaN after gap-fill (>{max_gap}h gaps) -- will be zeroed per-batch")
        out[key] = arr.astype(np.float32)

    _sh_coeff_cache[path] = out
    return out


def _slice_sh_coeffs_batch(sh_data, l_max, batch_start_time, nt, dt_seconds):
    """Same nearest-neighbor-within-tolerance sampling as
    inverse_solver._slice_image_batch, applied to the [N_coef, T_hourly]
    global-SH array instead of an [Nphi, T_hourly] image array."""
    coeffs = sh_data[f'coeffs_L{l_max}']  # [N_coef, T]
    src_times = sh_data['times']
    n_coef, total_time = coeffs.shape

    target_times = pd.date_range(
        start=pd.to_datetime(batch_start_time), periods=nt, freq=pd.Timedelta(seconds=dt_seconds),
    )
    old_t = src_times.view('int64')
    new_t = target_times.view('int64')
    right_idx = np.searchsorted(old_t, new_t, side='left')
    left_idx = np.clip(right_idx - 1, 0, total_time - 1)
    right_idx = np.clip(right_idx, 0, total_time - 1)
    left_dt = np.abs(new_t - old_t[left_idx])
    right_dt = np.abs(old_t[right_idx] - new_t)
    choose_right = right_dt < left_dt
    nearest_idx = np.where(choose_right, right_idx, left_idx)

    tolerance_ns = int(max(dt_seconds, 3600.0) * 1e9)
    nearest_dt = np.minimum(left_dt, right_dt)
    valid_target = nearest_dt <= tolerance_ns

    batch_coeffs = np.zeros((n_coef, nt), dtype=np.float32)
    if np.any(valid_target):
        sampled = coeffs[:, nearest_idx[valid_target]]
        batch_coeffs[:, valid_target] = np.nan_to_num(sampled, nan=0.0)
    return batch_coeffs


# =============================================================================
# Batch construction (whole-batch; no rolling windows)
#
# Reuses inverse_solver's own batch-geometry code so the batches here are
# identical to the ones main.py's inverse pipeline would process.
# =============================================================================

def build_joint_batch_specs(config: Config) -> List[Dict]:
    v_obs_df = load_observations(config).sort_values('Time').reset_index(drop=True)
    use_multispacecraft = (
        bool(getattr(config, 'enable_multispacecraft_loss', False))
        and ('Spacecraft_ID' in v_obs_df.columns)
    )

    if use_multispacecraft:
        v_obs_raw_df = v_obs_df.copy()
        unique_obs_times_df = (
            v_obs_raw_df[['Time']].drop_duplicates().sort_values('Time').reset_index(drop=True)
        )
        print(f"Using multi-spacecraft loss with {v_obs_raw_df['Spacecraft_ID'].nunique()} spacecraft")
    else:
        v_obs_raw_df = v_obs_df
        unique_obs_times_df = v_obs_df.copy()
        print("Using single-spacecraft loss (all observations treated as one track)")

    # Restrict which tracks enter the data misfit. Applied AFTER
    # unique_obs_times_df is taken, so batch geometry stays identical to an
    # unfiltered run and batch indices remain directly comparable.
    include_sc = getattr(config, 'spacecraft_include', None)
    if include_sc and 'Spacecraft_ID' in v_obs_raw_df.columns:
        present = set(v_obs_raw_df['Spacecraft_ID'].unique())
        missing = [s for s in include_sc if s not in present]
        if missing:
            raise RuntimeError(
                f"--spacecraft asked for {missing}, not present in the observation file. "
                f"Available: {sorted(present)}"
            )
        n_before = len(v_obs_raw_df)
        v_obs_raw_df = v_obs_raw_df[
            v_obs_raw_df['Spacecraft_ID'].isin(include_sc)
        ].reset_index(drop=True)
        print(
            f"Spacecraft filter: keeping {include_sc} "
            f"({sorted(present)} -> {n_before} obs reduced to {len(v_obs_raw_df)})"
        )

    all_image_slices = _load_all_image_data(config)

    batch_specs, batch_size_steps, overlap_steps, overlap_model_steps, stride_steps, total_steps, warmup_hours = (
        _prepare_batch_specs(config, unique_obs_times_df, all_image_slices)
    )

    cpu_device = torch.device('cpu')
    for spec in batch_specs:
        start_time = pd.to_datetime(spec['batch_start_time'])
        end_time = start_time + pd.Timedelta(hours=len(spec['speed_hourly']))
        spec['obs_tracks'] = _build_obs_tracks_for_batch(v_obs_raw_df, start_time, end_time, config, cpu_device)

    sh_l_max = getattr(config, 'merged_temporal_sh_l_max', None)
    if sh_l_max is not None:
        sh_data = _load_sh_coeff_data(config)
        for spec in batch_specs:
            spec['sh_coeffs_raw'] = _slice_sh_coeffs_batch(
                sh_data, int(sh_l_max), spec['batch_start_time'], config.Nt, config.dt,
            )

    print(f"Built {len(batch_specs)} whole-batch computational boxes (batch indices 0..{len(batch_specs) - 1})")
    return batch_specs


def make_batch_record(spec: Dict, config: Config, device: torch.device) -> Dict:
    """Convert one raw batch spec into a device-resident training record.

    TIMSE weights are precomputed once here, per track, from that track's own
    observations in this batch -- not shared across batches or splits, per the
    "weights for TIMSE can be precomputed on a given batch" design.
    """
    batch_idx = spec['batch_idx']

    obs_tracks = spec.get('obs_tracks')
    if not obs_tracks:
        raise ValueError(f"Batch {batch_idx} has no usable observation tracks; skipping is required upstream.")

    for track in obs_tracks:
        track['v_obs'] = track['v_obs'].to(device)
        track['phi_obs_t'] = track['phi_obs_t'].to(device)
        track['radius_obs_t'] = track['radius_obs_t'].to(device)
        track['time_obs_t'] = torch.tensor(track['time_obs_seconds'], dtype=torch.float32, device=device)
        # Build the tail-weight lookup from the same post-transient subset that
        # will actually be scored (see _transient_skip_steps), not the raw track.
        t0 = _transient_skip_steps(config, track['v_obs'].shape[0])
        track['timse_lookup'] = build_timse_lookup(track['v_obs'][t0:].detach().cpu())

    primary_track = _get_primary_track(obs_tracks)
    v_obs_batch = primary_track['v_obs']
    phi_obs_batch_t = primary_track['phi_obs_t']
    radius_obs_batch_t = primary_track['radius_obs_t']
    time_obs_batch_t = primary_track['time_obs_t']

    image_slice = torch.tensor(np.asarray(spec['image_slice'], dtype=np.float32), dtype=torch.float32, device=device)

    batch_start_time = pd.to_datetime(spec['batch_start_time'])

    record = {
        'batch_idx': batch_idx,
        'batch_start_time': batch_start_time,
        'image_slice': image_slice,
        'obs_tracks': obs_tracks,
        'phi_obs_t': phi_obs_batch_t,
        'radius_obs_t': radius_obs_batch_t,
        'time_obs_t': time_obs_batch_t,
        'ace_obs': v_obs_batch,
        'ace_timse_lookup': primary_track['timse_lookup'],
    }
    offs = list(getattr(config, 'nn2_rec_offsets_h', None) or [])
    if offs:
        series = config.nn2_rec_series                                          # measured ACE speed, hourly
        tq = pd.to_datetime(np.asarray(primary_track['time_hourly']))
        rec = np.stack([series.reindex(tq + pd.Timedelta(hours=float(h))).to_numpy(dtype=np.float64) for h in offs], 1)
        record['ace_rec'] = torch.tensor(rec, dtype=torch.float32, device=device)
    if 'sh_coeffs_raw' in spec:
        record['sh_coeffs_raw'] = torch.tensor(
            np.asarray(spec['sh_coeffs_raw'], dtype=np.float32), dtype=torch.float32, device=device,
        )
    return record


# =============================================================================
# Joint forward pipeline
# =============================================================================


def kinematic_track(v_inner_hat: torch.Tensor, record: Dict, config: Config, forward_model: nn.Module) -> torch.Tensor:
    """Speed at the spacecraft from the inner boundary WITHOUT the transport solver: the factorial test of what the
    solver adds (reviewer, 2026-10-16). Same boundary, same residual acceleration ΔV(r); only the link differs.

      config.propagation_mode = 'fixed_lag': v(t) = u(φ_src, t - lag) + ΔV(r_sc), lag = config.kinematic_lag_h (96 h),
                                the alignment of the linear comparator (no speed-dependent travel time).
      config.propagation_mode = 'ballistic': every boundary parcel travels at its own speed plus ΔV(r), with no stream
                                interaction; v(t) = average of the parcels arriving near t, softmax weights
                                exp(-(t_arr - t)^2 / 2σ^2), σ = config.ballistic_sigma_h (2 h), over departures 1-7 d
                                earlier (overtaking parcels are averaged).
    Source longitude of a departure at t_d for the spacecraft at t: φ_sc(t) + Ω (t - t_d), the Carrington longitude that
    faced the spacecraft's direction at t_d (Ω = config.Omega, the Carrington rate). Bilinear interpolation of u in
    (longitude, time) on the periodic grid. Differentiable in u (and, for 'ballistic', through the travel times)."""
    mode = config.propagation_mode
    v = v_inner_hat
    Nphi, Nt = v.shape
    dev = v.device
    dt = float(config.dt)
    Omega = float(config.Omega)
    t = record['time_obs_t'].to(dev, torch.float32)
    phi_t = record['phi_obs_t'].to(dev, torch.float32)
    r_t = record['radius_obs_t'].to(dev, torch.float32)
    if r_t.dim() == 0:
        r_t = r_t.expand_as(t)
    if phi_t.dim() == 0:
        phi_t = phi_t.expand_as(t)
    bad = ~(torch.isfinite(t) & torch.isfinite(phi_t) & torch.isfinite(r_t))   # hours without a spacecraft position:
    t = torch.where(bad, torch.zeros_like(t), t)                                 # safe values, so that no NaN enters the
    phi_t = torch.where(bad, torch.zeros_like(phi_t), phi_t)                     # interpolation weights (0 * NaN would
    r_t = torch.where(bad, torch.full_like(r_t, 215.0), r_t)                     # poison the gradient); output NaN there
    nan = torch.full_like(t, float('nan'))
    r0 = float(config.r0)
    acc = getattr(forward_model, 'learned_radial_accel', None)
    if acc is not None:
        L, Vref = acc.physical_parameters()
        L, Vref = L.to(dev, torch.float32), Vref.to(dev, torch.float32)
    else:
        L, Vref = torch.tensor(1.0, device=dev), torch.tensor(0.0, device=dev)
    dV = lambda rr: Vref * (1.0 - torch.exp(-torch.clamp(rr - r0, min=0.0) / L))
    dphi = 2.0 * math.pi / Nphi

    def sample(lon, tidx):
        x = torch.remainder(lon, 2.0 * math.pi) / dphi
        j0f = torch.floor(x)
        a = x - j0f
        j0 = j0f.long() % Nphi
        j1 = (j0 + 1) % Nphi
        tc = torch.clamp(tidx, 0.0, Nt - 1.0)
        n0f = torch.clamp(torch.floor(tc), max=Nt - 2.0)
        b = tc - n0f
        n0 = n0f.long()
        return ((1 - a) * ((1 - b) * v[j0, n0] + b * v[j0, n0 + 1]) + a * ((1 - b) * v[j1, n0] + b * v[j1, n0 + 1]))

    if mode == 'fixed_lag':
        lag = float(config.kinematic_lag_h) * 3600.0
        return torch.where(bad, nan, sample(phi_t + Omega * lag, (t - lag) / dt) + dV(r_t))
    if mode != 'ballistic':
        raise ValueError(f'unknown propagation_mode {mode!r}')
    k = torch.arange(int(round(86400.0 / dt)), int(round(7 * 86400.0 / dt)) + 1, device=dev, dtype=torch.float32)
    lag = k * dt                                                        # departure lags [K], s
    td = t[:, None] - lag[None, :]                                      # [Nobs, K]
    vb = sample(phi_t[:, None] + Omega * lag[None, :], td / dt)         # boundary speed of each candidate parcel
    # travel time with ΔV(x) = Vref (1 - exp(-x/L)), x = r - r0, exact: with a = v + Vref,
    # τ = ∫0^X dx / (a - Vref e^{-x/L}) = (1/a) [X + L ln((a - Vref e^{-X/L}) / v)]   (lengths in km)
    X = ((r_t - r0) * 695700.0)[:, None]
    Lk, a = L * 695700.0, vb + Vref
    tau = (X + Lk * torch.log((a - Vref * torch.exp(-X / Lk)) / vb)) / a
    z = -0.5 * ((td + tau - t[:, None]) / (float(config.ballistic_sigma_h) * 3600.0)) ** 2
    z = torch.where(td >= 0, z, torch.full_like(z, -1e9))               # departures before the run start: excluded
    w = torch.softmax(z, dim=1)
    return torch.where(bad, nan, (w * vb).sum(dim=1) + dV(r_t))


def forward_joint_pipeline(record: Dict, nn1: nn.Module, nn2: nn.Module, forward_model: nn.Module,
                            r: torch.Tensor, phi: torch.Tensor, config: Config) -> Dict:
    """images -> boundary net -> physics propagation -> residual net -> final forecast."""
    image_slice = record['image_slice']

    if getattr(nn1, 'use_global_sh', False):
        v_inner_hat = nn1(image_slice, record.get('sh_coeffs_raw'))  # [Nphi, Nt]
    else:
        v_inner_hat = nn1(image_slice)  # [Nphi, Nt]
    if not torch.isfinite(v_inner_hat).all():
        raise ValueError(f"Non-finite NN1 output for batch {record['batch_idx']}")

    v0 = steady_state_propagate(
        v_inner_hat[:, 0].detach(), r, direction='outward', Omega=float(config.Omega),
    )

    v_full = forward_model(v_inner_hat, v0)
    if not torch.isfinite(v_full).all():
        raise ValueError(f"Non-finite physics forward output for batch {record['batch_idx']}")

    ace_phys = sample_satellite_track(
        v_full, r, phi, record['phi_obs_t'], record['radius_obs_t'],
        time_obs=record['time_obs_t'], dt=config.dt,
    )
    if getattr(config, 'propagation_mode', 'solver') != 'solver':      # factorial test: replace the solver's link
        ace_phys = kinematic_track(v_inner_hat, record, config, forward_model)

    delta = nn2(ace_phys, record.get('ace_rec'))
    if not torch.isfinite(delta).all():
        raise ValueError(f"Non-finite NN2 delta for batch {record['batch_idx']}")

    ace_final = ace_phys + delta

    return {
        'v_inner_hat': v_inner_hat,
        'v0': v0,
        'v_full': v_full,
        'ace_phys': ace_phys,
        'ace_final': ace_final,
        'delta': delta,
    }


def _transient_skip_steps(config: Config, n: int) -> int:
    """First index to start scoring at.

    v0 (the domain-wide initial condition) is a numerical necessity for solving
    the PDE, not something the boundary condition controls -- observations
    before the boundary-driven signal has had time to reach the observation
    point (~94h transit time, hence overlap_hours=100) are still dominated by
    v0's arbitrary extrapolated value and shouldn't be scored against the
    boundary condition at all.

    skip_steps is exactly overlap_hours, one index per hourly-cadence track
    point -- the same overlap_hours that drives the batch-to-batch stride in
    _prepare_batch_specs, so the discarded prefix of each batch matches its
    shared/overlapping hours with the previous batch hour-for-hour. (loss.py's
    _compute_track_misfit uses int(overlap_hours * 3600 / dt) instead, which
    is a model-grid-steps conversion misapplied as an hourly-track index --
    2x too many hours skipped at overlap_hours=100, dt=1800. That's a
    pre-existing characteristic of the original inverse code, left untouched
    there; this version is fixed to actually match the batch overlap.)
    """
    skip_steps = int(getattr(config, 'overlap_hours', 0.0))
    return min(skip_steps, max(0, n - 1))


def _apply_train_only_exclusion(record: Dict, exclusion_times: set, config: Config) -> int:
    """Mask ace_obs (NaN) at the given timestamps, IN PLACE, for one record.

    Only meant to be called on records assigned to the TRAIN role -- val/test
    records must stay untouched so evaluation is always against the full,
    real observed series (see the despike/cross-model-disagreement design
    discussion: excluding "hard" points from training is defensible, but
    excluding them from evaluation too would just make the test set easier
    and hide whether anything real improved).

    NN2 (ACEResidualHead) never consumes ace_obs (see its docstring -- it's
    built purely from the physics-predicted trajectory), so this cannot
    propagate NaN into the forward computation; it only affects the loss
    target and the TIMSE lookup, which is recomputed here to stay
    consistent with the now-masked series.
    """
    primary_track = record['obs_tracks'][0]
    times = pd.to_datetime(np.asarray(primary_track['time_hourly']))
    to_exclude = torch.from_numpy(np.asarray(times.isin(exclusion_times))).to(record['ace_obs'].device)
    # Count only hours that were actually usable observations before masking
    # (finite after _build_obs_tracks_for_batch's short-gap interpolation).
    n_masked = int((to_exclude & torch.isfinite(record['ace_obs'])).sum().item())
    if not bool(to_exclude.any()):
        return 0

    record['ace_obs'][to_exclude] = float('nan')
    primary_track['v_obs'][to_exclude] = float('nan')

    t0 = _transient_skip_steps(config, record['ace_obs'].shape[0])
    new_lookup = build_timse_lookup(record['ace_obs'][t0:].detach().cpu())
    record['ace_timse_lookup'] = new_lookup
    primary_track['timse_lookup'] = new_lookup
    return n_masked


def compute_boundary_loss(record: Dict, v_inner_hat: torch.Tensor, v_full: torch.Tensor,
                           r: torch.Tensor, phi: torch.Tensor, config: Config, use_timse: bool = True) -> torch.Tensor:
    """Data misfit across every spacecraft track, plus phi/t curvature regularization.

    Same per-track-then-average structure and same phi/t regularization formula
    as loss.py's LossFunction, including skipping the v0-transient window at the
    start of each track (see _transient_skip_steps). When use_timse=True
    (default), the per-track data misfit uses average TIMSE (tail-weighted,
    using that track's own per-batch precomputed weights) instead of plain MSE
    -- so it's on the same scale as compute_tail_loss and
    lambda_boundary/lambda_forecast actually control which one dominates. When
    use_timse=False, every point is weighted equally (standard MSE); the
    average-TIMSE machinery already reduces exactly to plain MSE when
    weights=None, so this is a formula toggle, not a different code path.
    """
    spec = _as_loss_spec(use_timse)
    track_losses = []
    for track in record['obs_tracks']:
        pred = sample_satellite_track(
            v_full, r, phi, track['phi_obs_t'], track['radius_obs_t'],
            time_obs=track['time_obs_t'], dt=config.dt,
        )
        t0 = _transient_skip_steps(config, min(pred.shape[0], track['v_obs'].shape[0]))
        obs_t, pred_t = track['v_obs'][t0:], pred[t0:]
        weights = timse_weights_from_lookup(obs_t, track['timse_lookup']) if spec.use_timse else None
        l_track = timse_loss_average(obs_t, pred_t, weights=weights)
        if spec.sorted_weight > 0:
            l_track = l_track + spec.sorted_weight * sorted_distribution_loss(obs_t, pred_t)
        track_losses.append(l_track)
    data_loss = torch.stack(track_losses).mean()

    loss_phi = float(config.lambda_phi) * torch.mean(
        (v_inner_hat[2:, :] - 2 * v_inner_hat[1:-1, :] + v_inner_hat[:-2, :]) ** 2
    )
    loss_t = float(config.lambda_t) * torch.mean(
        (v_inner_hat[:, 2:] - 2 * v_inner_hat[:, 1:-1] + v_inner_hat[:, :-2]) ** 2
    )
    return data_loss + loss_phi + loss_t


# =============================================================================
# Loss specification
#
# `use_timse` used to be a bool threaded through every train/eval entry point.
# It now carries a LossSpec instead; every intermediate function still just
# forwards it opaquely, and only compute_boundary_loss / compute_tail_loss
# interpret it, so the plumbing is unchanged. Bools are still accepted so
# nothing that passes True/False breaks.
# =============================================================================

@dataclass(frozen=True)
class LossSpec:
    """How the per-track data misfit is formed.

    pointwise:
        'timse'   - tail-weighted MSE (weights depend on the OBSERVED value).
                    Note this biases the estimator: weighted-MSE converges to
                    E[w(Y)Y|x]/E[w(Y)|x], not E[Y|x]. Measured on ACE batch 0
                    the optimal constant moves from 408 to 568 km/s.
        'uniform' - plain MSE, unbiased but amplitude-compressing: with
                    correlation rho to truth, the MSE-optimal amplitude ratio
                    is exactly rho.

    sorted_weight:
        Weight on mean((sort(pred) - sort(obs))^2), the squared 2-Wasserstein
        distance between the predicted and observed MARGINAL distributions.
        Zero iff the marginals match, and (unlike a target-dependent weighting)
        it is minimized at amplitude ratio 1.0 at every rho without moving the
        optimum of the mean. This is the term that penalizes range compression
        directly, where compression actually lives. 0 disables it.
    """

    pointwise: str = "timse"
    sorted_weight: float = 0.0

    @property
    def use_timse(self) -> bool:
        return self.pointwise == "timse"


def _as_loss_spec(use_timse) -> LossSpec:
    if isinstance(use_timse, LossSpec):
        return use_timse
    return LossSpec(pointwise="timse" if use_timse else "uniform", sorted_weight=0.0)


def percentile_grid(obs, n_intervals: int = 100) -> Dict:
    """Percentile TIMSE grid from observations only: q_k = quantile(obs, k/K), k = 0..K (numpy 'linear'
    interpolation), plus vmin / vmax of the same observations (the TIMSE normalization)."""
    o = np.asarray(obs, dtype=np.float64)
    o = o[np.isfinite(o)]
    return {'q': torch.tensor(np.quantile(o, np.linspace(0.0, 1.0, n_intervals + 1)), dtype=torch.float64),
            'vmin': float(o.min()), 'vmax': float(o.max()), 'n': int(o.size)}


def combined_score(obs: torch.Tensor, pred: torch.Tensor, grid: Dict):
    """S = sqrt(RMSE * sqrt(TIMSE)) on finite pairs, TIMSE on the given percentile grid:
    TIMSE = sum_k (q_{k+1} - q_k) * mean_{obs >= q_k} err^2 / (vmax - vmin); a threshold with no
    observations at or above it, or a zero-width interval, contributes zero. Differentiable in pred.
    Returns (S, MSE, TIMSE) as tensors."""
    ok = torch.isfinite(obs) & torch.isfinite(pred)
    o, e2 = obs[ok].double(), (pred[ok].double() - obs[ok].double()) ** 2
    q = grid['q'].to(o.device)
    width = q[1:] - q[:-1]
    above = (o[None, :] >= q[:-1, None]).double()                      # [K, N]
    cnt = above.sum(1)
    use = (cnt > 0) & (width > 0)
    tail_mse = (above @ e2) / cnt.clamp(min=1.0)
    timse = (width * tail_mse * use.double()).sum() / (grid['vmax'] - grid['vmin'])
    mse = e2.mean()
    return (mse * timse) ** 0.25, mse, timse


def _scored_pairs(record: Dict, outputs: Dict, config: Config, key: str = 'ace_final'):
    """Primary-track (obs, prediction) over the scored window (warm-up excluded); key 'ace_final' or 'ace_phys'."""
    obs = record['ace_obs']
    t0 = _transient_skip_steps(config, obs.shape[0])
    return obs[t0:], outputs[key][t0:]


def _smoothness_reg(v_inner_hat: torch.Tensor, config: Config) -> torch.Tensor:
    """The phi / t second-difference penalties of compute_boundary_loss."""
    return (float(config.lambda_phi) * torch.mean((v_inner_hat[2:, :] - 2 * v_inner_hat[1:-1, :] + v_inner_hat[:-2, :]) ** 2)
            + float(config.lambda_t) * torch.mean((v_inner_hat[:, 2:] - 2 * v_inner_hat[:, 1:-1] + v_inner_hat[:, :-2]) ** 2))


def pooled_score_metrics(obs_list, pred_list, grid: Dict) -> Dict:
    """S, RMSE, TRMSE = sqrt(TIMSE), bias on the pooled finite pairs of several chunks."""
    o, p = torch.cat(obs_list), torch.cat(pred_list)
    S, mse, timse = combined_score(o, p, grid)
    ok = torch.isfinite(o) & torch.isfinite(p)
    return {'S': float(S), 'rmse': float(mse.sqrt()), 'trmse': float(timse.sqrt()),
            'bias': float((p[ok] - o[ok]).double().mean()), 'n': int(ok.sum())}


def sorted_distribution_loss(obs: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """mean((sort(pred) - sort(obs))^2) over jointly-finite pairs.

    Differentiable: torch.sort's backward is a scatter of the incoming
    gradient through the sort permutation.
    """
    valid = torch.isfinite(obs) & torch.isfinite(pred)
    if int(valid.sum().item()) == 0:
        return torch.zeros((), device=pred.device)
    o = torch.sort(obs[valid]).values
    p = torch.sort(pred[valid]).values
    return torch.mean((o - p) ** 2)


def amplitude_ratio(obs: torch.Tensor, pred: torch.Tensor) -> float:
    """std(pred)/std(obs) over jointly-finite pairs. 1.0 = calibrated spread."""
    valid = torch.isfinite(obs) & torch.isfinite(pred)
    if int(valid.sum().item()) < 2:
        return float('nan')
    so = torch.std(obs[valid])
    if float(so) <= 0:
        return float('nan')
    return float(torch.std(pred[valid]) / so)


def threshold_scores(obs: torch.Tensor, pred: torch.Tensor, thr: float) -> Dict[str, float]:
    """POD / FAR / TSS for the event `speed > thr`, the operational form of
    'did it capture the tail'."""
    valid = torch.isfinite(obs) & torch.isfinite(pred)
    if int(valid.sum().item()) == 0:
        return {'pod': float('nan'), 'far': float('nan'), 'tss': float('nan')}
    o = obs[valid] > thr
    p = pred[valid] > thr
    hits = float((o & p).sum())
    misses = float((o & ~p).sum())
    fa = float((~o & p).sum())
    cn = float((~o & ~p).sum())
    pod = hits / (hits + misses) if (hits + misses) > 0 else float('nan')
    far = fa / (hits + fa) if (hits + fa) > 0 else float('nan')
    pofd = fa / (fa + cn) if (fa + cn) > 0 else float('nan')
    tss = (pod - pofd) if (pod == pod and pofd == pofd) else float('nan')
    return {'pod': pod, 'far': far, 'tss': tss}


def compute_tail_loss(record: Dict, ace_final: torch.Tensor, config: Config, use_timse: bool = True) -> torch.Tensor:
    """Forecast loss for the primary track (skips the v0-transient window, see _transient_skip_steps).

    TIMSE-weighted (default) or plain MSE (use_timse=False).
    """
    spec = _as_loss_spec(use_timse)
    t0 = _transient_skip_steps(config, min(ace_final.shape[0], record['ace_obs'].shape[0]))
    obs_t, final_t = record['ace_obs'][t0:], ace_final[t0:]
    weights = timse_weights_from_lookup(obs_t, record['ace_timse_lookup']) if spec.use_timse else None
    loss = timse_loss_average(obs_t, final_t, weights=weights)
    if spec.sorted_weight > 0:
        loss = loss + spec.sorted_weight * sorted_distribution_loss(obs_t, final_t)
    return loss


def _delta_smoothness(delta: torch.Tensor) -> torch.Tensor:
    if delta.numel() > 1:
        return torch.mean((delta[1:] - delta[:-1]) ** 2)
    return torch.zeros((), device=delta.device)


def compute_total_loss(l_boundary: torch.Tensor, l_forecast: torch.Tensor, l_reg: torch.Tensor,
                        lam_boundary: float, lam_forecast: float) -> torch.Tensor:
    return lam_boundary * l_boundary + lam_forecast * l_forecast + l_reg


# =============================================================================
# Diagnostics / plotting
#
# Mirrors what main.py's inverse pipeline shows (optimization.py's
# _save_solution_plot / helpers.py's save_batch_solution_plot): the
# reconstructed inner boundary, and predicted-vs-observed speed for every
# spacecraft track. This pipeline additionally has a physics-only prediction
# AND a residual-corrected final forecast for the primary (ACE) track, so
# that panel shows all three curves instead of just prediction-vs-observed.
#
# =============================================================================

def _plot_series_with_gaps(ax, times, values, label, color, max_gap_hours=12.0, **kwargs):
    """Plot a 1D series against datetimes, breaking the line across NaN/large time gaps."""
    times = pd.to_datetime(np.asarray(times), errors='coerce')
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values) & ~pd.isna(times)
    if not np.any(finite):
        return
    t_num = mdates.date2num(times[finite])
    y = values[finite]

    order = np.argsort(t_num)
    t_num, y = t_num[order], y[order]

    gap_days = float(max_gap_hours) / 24.0
    gap_idx = np.where(np.diff(t_num) > gap_days)[0]
    starts = np.concatenate(([0], gap_idx + 1))
    ends = np.concatenate((gap_idx + 1, [len(t_num)]))

    first = True
    for s, e in zip(starts, ends):
        if e - s < 1:
            continue
        ax.plot(t_num[s:e], y[s:e], label=label if first else '_nolegend_', color=color, **kwargs)
        first = False


def save_batch_diagnostic_plots(record: Dict, outputs: Dict, r: torch.Tensor, phi: torch.Tensor,
                                 config: Config, plot_dir, split_name: str, epoch: int):
    """Save the reconstructed-boundary heatmap and a predicted-vs-observed track plot for one batch."""
    if plot_dir is None:
        return
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    batch_idx = int(record['batch_idx'])
    tag = f"{split_name}_batch{batch_idx:03d}_epoch{int(epoch):04d}"

    obs_tracks = record['obs_tracks']
    primary_id = str(obs_tracks[0]['id'])
    extra_tracks = [t for t in obs_tracks if str(t['id']) != primary_id]

    v_full = outputs['v_full']

    # 1. Reconstructed inner boundary heatmap.
    v_inner_np = outputs['v_inner_hat'].detach().cpu().numpy()
    save_boundary_condition_plot(
        v_inner_np, config, plot_dir / f"{tag}_boundary.png",
        title=f"{split_name} batch {batch_idx} | epoch {epoch} | reconstructed inner boundary",
    )

    # 2. Predicted-vs-observed speed, one panel per spacecraft track.
    n_panels = 1 + len(extra_tracks)
    fig, axes = plt.subplots(n_panels, 1, figsize=(11, 3.5 * n_panels), squeeze=False)
    axes = axes[:, 0]

    primary_time = obs_tracks[0]['time_hourly']
    ax0 = axes[0]
    _plot_series_with_gaps(ax0, primary_time, record['ace_obs'].detach().cpu().numpy(),
                            label=f'{primary_id} observed', color='black', linewidth=1.3)
    _plot_series_with_gaps(ax0, primary_time, outputs['ace_phys'].detach().cpu().numpy(),
                            label='physics only (ace_phys)', color='tab:blue', linewidth=1.1)
    _plot_series_with_gaps(ax0, primary_time, outputs['ace_final'].detach().cpu().numpy(),
                            label='physics + residual (ace_final)', color='tab:red', linewidth=1.1)
    ax0.set_ylabel('Speed [km/s]')
    ax0.set_title(f"{split_name} batch {batch_idx} | epoch {epoch} | {primary_id} (primary track)")
    ax0.legend(loc='upper right', fontsize=8)
    ax0.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    for ax, track in zip(axes[1:], extra_tracks):
        track_pred = sample_satellite_track(
            v_full, r, phi, track['phi_obs_t'], track['radius_obs_t'],
            time_obs=track['time_obs_t'], dt=config.dt,
        )
        track_id = str(track['id'])
        _plot_series_with_gaps(ax, track['time_hourly'], track['v_obs'].detach().cpu().numpy(),
                                label=f'{track_id} observed', color='black', linewidth=1.3)
        _plot_series_with_gaps(ax, track['time_hourly'], track_pred.detach().cpu().numpy(),
                                label='physics only (no NN2 correction)', color='tab:blue', linewidth=1.1)
        ax.set_ylabel('Speed [km/s]')
        ax.set_title(f"{track_id} (physics-only; NN2 only corrects the primary track)")
        ax.legend(loc='upper right', fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    fig.tight_layout()
    fig.savefig(str(plot_dir / f"{tag}_tracks.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)


def _run_record(record: Dict, nn1, nn2, forward_model, r, phi, config, lam_boundary, lam_forecast,
                 reg_weight, use_timse=True):
    outputs = forward_joint_pipeline(record, nn1, nn2, forward_model, r, phi, config)

    l_boundary = compute_boundary_loss(record, outputs['v_inner_hat'], outputs['v_full'], r, phi, config, use_timse=use_timse)
    l_forecast = compute_tail_loss(record, outputs['ace_final'], config, use_timse=use_timse)
    l_reg = float(reg_weight) * _delta_smoothness(outputs['delta'])

    for name, l in (('boundary', l_boundary), ('forecast', l_forecast), ('reg', l_reg)):
        if not torch.isfinite(l):
            raise ValueError(f"Non-finite {name} loss for batch {record['batch_idx']}")

    total = compute_total_loss(l_boundary, l_forecast, l_reg, lam_boundary, lam_forecast)
    if not torch.isfinite(total):
        raise ValueError(f"Non-finite total loss for batch {record['batch_idx']}")

    metrics = {
        'boundary_loss': float(l_boundary.detach().item()),
        'forecast_loss': float(l_forecast.detach().item()),
        'reg_loss': float(l_reg.detach().item()),
        'total_loss': float(total.detach().item()),
        'ace_rmse': float(finite_rmse(record['ace_obs'], outputs['ace_final']).detach().item()),
        'ace_phys_rmse': float(finite_rmse(record['ace_obs'], outputs['ace_phys']).detach().item()),
        'delta_max': float(outputs['delta'].detach().abs().max().item()),
    }

    # Distribution diagnostics on the same post-transient window the loss uses.
    # RMSE alone cannot tell a well-timed forecast from an amplitude-collapsed
    # one; amp_ratio and the threshold scores are what "captured the tail"
    # actually means.
    metrics.update(_extra_distribution_metrics(record, outputs, config))
    return total, metrics, outputs


def _extra_distribution_metrics(record: Dict, outputs: Dict, config: Config) -> Dict:
    """amp_ratio / amp_ratio_phys / sorted_mse / POD-FAR-TSS@500,600.

    Factored out so every train/eval code path (this one via _run_record,
    the batched-group train/eval functions, and the multi-GPU x
    batched-group path) computes these identically -- a bug where the
    batched-group paths built their OWN metrics dict by hand and silently
    omitted these keys (KeyError at the first --batch-group-size>1 run
    that also used the amp_ratio log line) is exactly what this
    consolidation is meant to prevent from recurring.
    """
    t0 = _transient_skip_steps(config, min(outputs['ace_final'].shape[0], record['ace_obs'].shape[0]))
    obs = record['ace_obs'][t0:].detach()
    fin = outputs['ace_final'][t0:].detach()
    phy = outputs['ace_phys'][t0:].detach()
    ok = torch.isfinite(obs) & torch.isfinite(fin)
    lookup = record.get('ace_timse_lookup')
    w = timse_weights_from_lookup(obs, lookup) if lookup is not None else None
    extra = {
        # forecast errors on the scored (post-warm-up) window, independent of the training loss choice;
        # sse/n allow pooled RMSE across chunks (see _pooled)
        'fc_mse': timse_loss_average(obs, fin).item(),
        'fc_timse': timse_loss_average(obs, fin, weights=w).item() if w is not None else float('nan'),
        'fc_sse': float(((obs[ok] - fin[ok]) ** 2).sum().item()),
        'fc_n': float(ok.sum().item()),
        'amp_ratio': amplitude_ratio(obs, fin),
        'amp_ratio_phys': amplitude_ratio(obs, phy),
        'sorted_mse': float(sorted_distribution_loss(obs, fin).item()),
    }
    for thr in (500.0, 600.0):
        for k, v in threshold_scores(obs, fin, thr).items():
            extra[f'{k}{int(thr)}'] = v
    return extra


# =============================================================================
# Training / evaluation
# =============================================================================

def train_one_epoch(train_records, nn1, nn2, forward_model, optimizer, r, phi, config,
                     lam_boundary, lam_forecast, reg_weight, use_timse=True,
                     plot_every=0, plot_dir=None, epoch=None, clip_norm=1.0):
    """One optimizer step PER train batch (alternating), not one averaged step across all of them.

    Averaging gradients from batches with conflicting optimal mappings for
    NN1's shared weights causes destructive cancellation (empirically
    confirmed: batch0/batch1 NN1 gradients had cosine similarity -0.93, i.e.
    ~2/3 of each batch's signal cancelled when averaged). Alternating gives
    each batch a clean, unmixed gradient step. Order is shuffled each epoch
    (deterministically, seeded by epoch) so neither batch systematically goes
    first/last. Gradients are clipped per step (clip_norm, matching stage-1's
    optimization.py, which is not touched by this joint pipeline).
    """
    nn1.train()
    nn2.train()
    forward_model.train()

    should_plot = bool(plot_every) and epoch is not None and (epoch % int(plot_every) == 0)

    clip_params = list(nn1.parameters()) + list(nn2.parameters())
    accel = getattr(forward_model, 'learned_radial_accel', None)
    if accel is not None:
        clip_params += [p for p in accel.parameters() if p.requires_grad]

    order = list(range(len(train_records)))
    rng = np.random.default_rng(int(epoch) if epoch is not None else 0)
    rng.shuffle(order)

    n = max(1, len(train_records))
    agg = defaultdict(float)
    for idx in order:
        record = train_records[idx]
        optimizer.zero_grad(set_to_none=True)
        total, metrics, outputs = _run_record(
            record, nn1, nn2, forward_model, r, phi, config,
            lam_boundary, lam_forecast, reg_weight, use_timse=use_timse,
        )
        total.backward()
        if clip_norm and clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(clip_params, max_norm=float(clip_norm))
        optimizer.step()
        for k, v in metrics.items():
            agg[k] += v / n
        if should_plot:
            save_batch_diagnostic_plots(record, outputs, r, phi, config, plot_dir, split_name='train', epoch=epoch)

    return dict(agg)


def evaluate_records(records, nn1, nn2, forward_model, r, phi, config, reg_weight, use_timse=True,
                      plot_every=0, plot_dir=None, epoch=None, split_name='val'):
    nn1.eval()
    nn2.eval()
    forward_model.eval()

    empty = {
        'boundary_loss': float('nan'), 'forecast_loss': float('nan'), 'reg_loss': float('nan'),
        'total_loss': float('nan'), 'ace_rmse': float('nan'), 'ace_phys_rmse': float('nan'),
        'delta_max': float('nan'),
        'amp_ratio': float('nan'), 'amp_ratio_phys': float('nan'), 'sorted_mse': float('nan'),
        'pod500': float('nan'), 'far500': float('nan'), 'tss500': float('nan'),
        'pod600': float('nan'), 'far600': float('nan'), 'tss600': float('nan'),
        }
    if not records:
        return empty

    should_plot = bool(plot_every) and epoch is not None and (epoch % int(plot_every) == 0)

    agg = defaultdict(float)
    with torch.no_grad():
        for record in records:
            _, metrics, outputs = _run_record(
                record, nn1, nn2, forward_model, r, phi, config,
                lam_boundary=1.0, lam_forecast=1.0, reg_weight=reg_weight, use_timse=use_timse,
            )
            if should_plot:
                save_batch_diagnostic_plots(record, outputs, r, phi, config, plot_dir, split_name=split_name, epoch=epoch)
            for k, v in metrics.items():
                agg[k] += v

    n = len(records)
    return {k: v / n for k, v in agg.items()}


# =============================================================================
# Multi-GPU (parallel forward+backward per batch; sequential, unaveraged
# optimizer steps -- see --multi-gpu help text and train_one_epoch_multi_gpu's
# docstring for why this is not the gradient-averaging scheme this pipeline
# deliberately moved away from).
# =============================================================================

def _build_device_replica(nn1: nn.Module, nn2: nn.Module, forward_model: nn.Module,
                           config: Config, device: torch.device):
    """Fresh nn1/nn2/forward_model + grid tensors on device, weights synced from the masters.

    Built fresh (not deepcopy'd) specifically to avoid the plain-tensor-attribute
    bug that bit an earlier version of this pipeline's multi-GPU path:
    PropagationForwardModel.dr/.dphi are plain tensor attributes, not registered
    buffers, so nn.Module.to(device) silently leaves them on the original device
    after a deepcopy+.to(). Constructing fresh via setup_grid(device_config)
    sidesteps that entirely -- dr/dphi are correct-device tensors from the start.
    """
    device_config = copy.copy(config)
    device_config.device = str(device)
    r_r, phi_r, _, dr_r, dphi_r = setup_grid(device_config, print_cfl=False)

    # type(nn1) rather than the hardcoded class so this stays correct if nn1
    # is a ThresholdVelocityModel (--nn1-arch threshold) instead of the
    # default IntensityToVelocityMLP.
    nn1_r = type(nn1)(device_config).to(device)
    nn1_r.load_state_dict(nn1.state_dict())

    hidden = nn2.net[0].out_features
    nn2_r = ACEResidualHead(history_steps=nn2.history_steps, hidden=hidden, rec_offsets_h=nn2.rec_offsets_h.tolist()).to(device)
    nn2_r.load_state_dict(nn2.state_dict())

    fm_r = PropagationForwardModel(device_config, r_r, phi_r, dr_r, dphi_r).to(device)
    fm_r.load_state_dict(forward_model.state_dict())

    return nn1_r, nn2_r, fm_r, r_r, phi_r


def _move_record_to_device(record: Dict, device: torch.device) -> Dict:
    """Shallow copy of a batch record with all tensors moved to device.

    Does not mutate the original -- records are shared and reused every epoch
    across whichever device a batch happens to land on after reshuffling.
    """
    def _move_lookup(lookup):
        if lookup is None:
            return None
        return {k: v.to(device) for k, v in lookup.items()}

    def _move_track(track):
        moved = dict(track)
        moved['v_obs'] = track['v_obs'].to(device)
        moved['phi_obs_t'] = track['phi_obs_t'].to(device)
        moved['radius_obs_t'] = track['radius_obs_t'].to(device)
        moved['time_obs_t'] = track['time_obs_t'].to(device)
        moved['timse_lookup'] = _move_lookup(track.get('timse_lookup'))
        return moved

    moved = dict(record)
    moved['image_slice'] = record['image_slice'].to(device)
    moved['obs_tracks'] = [_move_track(tr) for tr in record['obs_tracks']]
    moved['phi_obs_t'] = record['phi_obs_t'].to(device)
    moved['radius_obs_t'] = record['radius_obs_t'].to(device)
    moved['time_obs_t'] = record['time_obs_t'].to(device)
    moved['ace_obs'] = record['ace_obs'].to(device)
    moved['ace_timse_lookup'] = _move_lookup(record.get('ace_timse_lookup'))
    return moved


def train_one_epoch_multi_gpu(train_records, nn1, nn2, forward_model, optimizer, config,
                               lam_boundary, lam_forecast, reg_weight, devices, use_timse=True,
                               plot_every=0, plot_dir=None, epoch=None, clip_norm=1.0):
    """Same alternating-per-batch-step semantics as train_one_epoch, computed faster.

    Batches are processed in groups of len(devices): within a group, each
    batch's forward+backward runs concurrently on its own GPU replica (all
    replicas synced to the SAME starting weights at the top of the group).
    Each batch still gets its own individual, unmixed gradient and its own
    optimizer.step() -- gradients are never averaged across batches, exactly
    like the single-GPU path, because averaging destructively cancels here
    (see module docstring: -0.93 cosine similarity between conflicting
    batches). The only difference from strict sequential alternation is that
    batches after the first *within a group* have their gradient computed
    against the group's starting weights rather than the just-updated weights
    from the batch immediately before -- a bounded staleness of up to
    len(devices)-1 steps.
    """
    nn1.train()
    nn2.train()
    forward_model.train()

    should_plot = bool(plot_every) and epoch is not None and (epoch % int(plot_every) == 0)

    master_device = next(nn1.parameters()).device
    clip_params = list(nn1.parameters()) + list(nn2.parameters())
    accel = getattr(forward_model, 'learned_radial_accel', None)
    if accel is not None:
        clip_params += [p for p in accel.parameters() if p.requires_grad]

    order = list(range(len(train_records)))
    rng = np.random.default_rng(int(epoch) if epoch is not None else 0)
    rng.shuffle(order)

    n = max(1, len(train_records))
    agg = defaultdict(float)
    group_size = max(1, len(devices))

    for group_start in range(0, len(order), group_size):
        group_idx = order[group_start:group_start + group_size]
        group_devices = devices[:len(group_idx)]

        replicas = [_build_device_replica(nn1, nn2, forward_model, config, dev) for dev in group_devices]

        def _forward_backward_one(pos):
            i = group_idx[pos]
            dev = group_devices[pos]
            nn1_r, nn2_r, fm_r, r_r, phi_r = replicas[pos]
            record = _move_record_to_device(train_records[i], dev)
            total, metrics, outputs = _run_record(
                record, nn1_r, nn2_r, fm_r, r_r, phi_r, config,
                lam_boundary, lam_forecast, reg_weight, use_timse=use_timse,
            )
            total.backward()
            accel_r = getattr(fm_r, 'learned_radial_accel', None)
            grads = {
                'nn1': [p.grad.detach().to(master_device) if p.grad is not None else None
                        for p in nn1_r.parameters()],
                'nn2': [p.grad.detach().to(master_device) if p.grad is not None else None
                        for p in nn2_r.parameters()],
                'accel': (
                    [p.grad.detach().to(master_device) if p.grad is not None else None
                     for p in accel_r.parameters()]
                    if accel_r is not None else None
                ),
            }
            return i, grads, metrics, outputs, record, r_r, phi_r

        with ThreadPoolExecutor(max_workers=len(group_idx)) as ex:
            results = list(ex.map(_forward_backward_one, range(len(group_idx))))

        # Apply sequentially in the original alternation order: one real,
        # individually-clipped optimizer step per batch, same as single-GPU.
        for i, grads, metrics, outputs, record, r_r, phi_r in results:
            optimizer.zero_grad(set_to_none=True)
            for p, g in zip(nn1.parameters(), grads['nn1']):
                if g is not None:
                    p.grad = g.clone()
            for p, g in zip(nn2.parameters(), grads['nn2']):
                if g is not None:
                    p.grad = g.clone()
            if accel is not None and grads['accel'] is not None:
                for p, g in zip(accel.parameters(), grads['accel']):
                    if g is not None:
                        p.grad = g.clone()
            if clip_norm and clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(clip_params, max_norm=float(clip_norm))
            optimizer.step()
            for k, v in metrics.items():
                agg[k] += v / n
            if should_plot:
                save_batch_diagnostic_plots(record, outputs, r_r, phi_r, config, plot_dir, split_name='train', epoch=epoch)

    return dict(agg)


def evaluate_records_multi_gpu(records, nn1, nn2, forward_model, config, reg_weight, devices,
                                use_timse=True, plot_every=0, plot_dir=None, epoch=None, split_name='val'):
    """Exact parallel evaluation -- no gradients, so no averaging/staleness concern at all."""
    nn1.eval()
    nn2.eval()
    forward_model.eval()

    empty = {
        'boundary_loss': float('nan'), 'forecast_loss': float('nan'), 'reg_loss': float('nan'),
        'total_loss': float('nan'), 'ace_rmse': float('nan'), 'ace_phys_rmse': float('nan'),
        'delta_max': float('nan'),
        'amp_ratio': float('nan'), 'amp_ratio_phys': float('nan'), 'sorted_mse': float('nan'),
        'pod500': float('nan'), 'far500': float('nan'), 'tss500': float('nan'),
        'pod600': float('nan'), 'far600': float('nan'), 'tss600': float('nan'),
        }
    if not records:
        return empty

    should_plot = bool(plot_every) and epoch is not None and (epoch % int(plot_every) == 0)
    n_devices = max(1, len(devices))
    replicas = [_build_device_replica(nn1, nn2, forward_model, config, dev) for dev in devices[:n_devices]]

    def _eval_one(i):
        record_idx = i
        dev = devices[i % n_devices]
        nn1_r, nn2_r, fm_r, r_r, phi_r = replicas[i % n_devices]
        record = _move_record_to_device(records[record_idx], dev)
        with torch.no_grad():
            _, metrics, outputs = _run_record(
                record, nn1_r, nn2_r, fm_r, r_r, phi_r, config,
                lam_boundary=1.0, lam_forecast=1.0, reg_weight=reg_weight, use_timse=use_timse,
            )
        if should_plot:
            save_batch_diagnostic_plots(record, outputs, r_r, phi_r, config, plot_dir, split_name=split_name, epoch=epoch)
        return metrics

    with ThreadPoolExecutor(max_workers=n_devices) as ex:
        all_metrics = list(ex.map(_eval_one, range(len(records))))

    agg = defaultdict(float)
    for metrics in all_metrics:
        for k, v in metrics.items():
            agg[k] += v
    n = len(records)
    return {k: v / n for k, v in agg.items()}


# =============================================================================
# Batched-group training (real GPU-bound speedup via forward_model.py's
# forward_batched, not threading -- see module docstring addendum below).
# =============================================================================

_NAN_DEBUG = bool(int(os.environ.get('SWNN_NAN_DEBUG', '0')))   # stage-by-stage finiteness checks (diagnostics only)


def _nan_report(tag, **tensors):
    """SWNN_NAN_DEBUG: print range and non-finite count of each tensor; True if any is non-finite."""
    bad = False
    for k, t in tensors.items():
        t = t.detach()
        nf = int((~torch.isfinite(t)).sum())
        fin = t[torch.isfinite(t)]
        rng = f'[{float(fin.min()):.4g}, {float(fin.max()):.4g}]' if fin.numel() else '[]'
        print(f'  NAN_DEBUG {tag} {k}: non-finite {nf} of {t.numel()}, finite range {rng}', flush=True)
        bad |= nf > 0
    return bad


_GRAD_BAD = {}                                                    # SWNN_NAN_DEBUG: non-finite gradient counts per tensor


_NAN_DUMP = {}                                                    # SWNN_NAN_DEBUG: solver input and incoming gradient


def _grad_watch(t, name):
    """SWNN_NAN_DEBUG: record how many entries of d(loss)/d(t) are non-finite during backward."""
    if _NAN_DEBUG and t.requires_grad:
        def hook(g):
            _GRAD_BAD[name] = (int((~torch.isfinite(g)).sum()), g.numel())
            if name == 'v_full':
                _NAN_DUMP['grad_v_full'] = g.detach().cpu()
        t.register_hook(hook)


def _run_group_forward(group_records, nn1, nn2, forward_model, r, phi, config, use_timse=True):
    """Shared batched physics forward for a group, then per-batch NN2/loss/metrics
    using the existing unbatched functions (sample_satellite_track,
    compute_boundary_loss, compute_tail_loss) on slices of the batched output.

    NN1 is batched via time-axis concatenation (exact: NN1 has no cross-time
    coupling, only a local +/-radius window in longitude -- validated to give
    bitwise-identical output to per-batch calls). The physics propagation is
    batched via forward_model.forward_batched (validated to give bitwise-
    identical forward AND gradients vs. the unbatched path, per-batch, with
    no cross-batch leakage). Everything after that (NN2, track sampling,
    loss) is untouched, called once per batch on that batch's own slice.
    """
    G = len(group_records)

    image_slices = [rec['image_slice'] for rec in group_records]
    nts = [im.shape[2] for im in image_slices]
    concat = torch.cat(image_slices, dim=2)
    if getattr(nn1, 'use_global_sh', False):
        sh_concat = torch.cat([rec['sh_coeffs_raw'] for rec in group_records], dim=1)
        v_inner_hat_concat = nn1(concat, sh_concat)
    else:
        v_inner_hat_concat = nn1(concat)
    v_inner_hat_list = list(torch.split(v_inner_hat_concat, nts, dim=1))

    v0_list = [
        steady_state_propagate(v_inner_hat_list[g][:, 0].detach(), r, direction='outward',
                                Omega=float(config.Omega))
        for g in range(G)
    ]

    v_inner_hat_batched = torch.stack(v_inner_hat_list, dim=0)
    v0_batched = torch.stack(v0_list, dim=0)
    v_full_batched = forward_model.forward_batched(v_inner_hat_batched, v0_batched)
    _grad_watch(v_inner_hat_batched, 'v_inner_hat'); _grad_watch(v_full_batched, 'v_full')
    if _NAN_DEBUG:
        _NAN_DUMP.update(v_inner=v_inner_hat_batched.detach().cpu(), v0=v0_batched.detach().cpu())
    if _NAN_DEBUG and (not torch.isfinite(v_inner_hat_batched).all() or not torch.isfinite(v_full_batched).all()):
        _nan_report('forward', v_inner_hat=v_inner_hat_batched, v0=v0_batched, v_full=v_full_batched)
        raise ValueError(f"NAN_DEBUG: non-finite forward in group {[rec['batch_idx'] for rec in group_records]}")

    per_batch = []
    for g in range(G):
        record = group_records[g]
        v_inner_hat_g = v_inner_hat_batched[g]
        v_full_g = v_full_batched[g]

        ace_phys = sample_satellite_track(
            v_full_g, r, phi, record['phi_obs_t'], record['radius_obs_t'],
            time_obs=record['time_obs_t'], dt=config.dt,
        )
        if getattr(config, 'propagation_mode', 'solver') != 'solver':  # factorial test: replace the solver's link
            ace_phys = kinematic_track(v_inner_hat_g, record, config, forward_model)
        _grad_watch(ace_phys, f'ace_phys[{record["batch_idx"]}]')
        delta = nn2(ace_phys, record.get('ace_rec'))
        ace_final = ace_phys + delta

        outputs = {
            'v_inner_hat': v_inner_hat_g, 'v0': v0_list[g], 'v_full': v_full_g,
            'ace_phys': ace_phys, 'ace_final': ace_final, 'delta': delta,
        }
        per_batch.append((record, outputs))
    return per_batch


def train_one_epoch_batched_group(train_records, nn1, nn2, forward_model, optimizer, r, phi, config,
                                   lam_boundary, lam_forecast, reg_weight, group_size, use_timse=True,
                                   plot_every=0, plot_dir=None, epoch=None, clip_norm=1.0):
    """Groups shuffled batches into micro-batches of group_size; each group shares
    ONE batched physics forward pass (see forward_model.py's forward_batched)
    and gets ONE combined (mean) loss, ONE backward, ONE optimizer.step().

    This is standard minibatch SGD -- a real, if partial, departure from
    train_one_epoch's fully alternating (unaveraged) per-batch steps.
    Averaging was previously found catastrophic for just 2 batches (NN1
    gradient cosine similarity -0.93). Revisited here only after: (1) moving
    to a 45-batch pool spanning the full solar cycle instead of 2, and (2) a
    corrected diagnostic -- relating each batch's gradient to actual local
    AIA intensity (a quantity comparable across batches, unlike raw
    weight-space dot products or same-index grid position across
    non-overlapping calendar windows) -- showing the batches mostly agree in
    sign, contradicting the naive weight-space read. If a run under this
    function shows the same pathological collapse the averaged 2-batch
    experiment did, that would be evidence the earlier read was closer to
    right after all; train_one_epoch (fully alternating) remains available
    as a fallback.
    """
    nn1.train()
    nn2.train()
    forward_model.train()

    should_plot = bool(plot_every) and epoch is not None and (epoch % int(plot_every) == 0)

    clip_params = list(nn1.parameters()) + list(nn2.parameters())
    accel = getattr(forward_model, 'learned_radial_accel', None)
    if accel is not None:
        clip_params += [p for p in accel.parameters() if p.requires_grad]

    order = list(range(len(train_records)))
    rng = np.random.default_rng(int(epoch) if epoch is not None else 0)
    rng.shuffle(order)

    n = max(1, len(train_records))
    agg = defaultdict(float)
    group_size = max(1, int(group_size))

    for group_start in range(0, len(order), group_size):
        group_idx = order[group_start:group_start + group_size]
        group_records = [train_records[i] for i in group_idx]

        optimizer.zero_grad(set_to_none=True)
        per_batch = _run_group_forward(group_records, nn1, nn2, forward_model, r, phi, config, use_timse=use_timse)

        group_losses = []
        for record, outputs in per_batch:
            l_boundary = compute_boundary_loss(record, outputs['v_inner_hat'], outputs['v_full'], r, phi, config, use_timse=use_timse)
            l_forecast = compute_tail_loss(record, outputs['ace_final'], config, use_timse=use_timse)
            l_reg = float(reg_weight) * _delta_smoothness(outputs['delta'])

            for name, l in (('boundary', l_boundary), ('forecast', l_forecast), ('reg', l_reg)):
                if not torch.isfinite(l):
                    raise ValueError(f"Non-finite {name} loss for batch {record['batch_idx']}")

            total = compute_total_loss(l_boundary, l_forecast, l_reg, lam_boundary, lam_forecast)
            if not torch.isfinite(total):
                raise ValueError(f"Non-finite total loss for batch {record['batch_idx']}")
            group_losses.append(total)

            metrics = {
                'boundary_loss': float(l_boundary.detach().item()),
                'forecast_loss': float(l_forecast.detach().item()),
                'reg_loss': float(l_reg.detach().item()),
                'total_loss': float(total.detach().item()),
                'ace_rmse': float(finite_rmse(record['ace_obs'], outputs['ace_final']).detach().item()),
                'ace_phys_rmse': float(finite_rmse(record['ace_obs'], outputs['ace_phys']).detach().item()),
                'delta_max': float(outputs['delta'].detach().abs().max().item()),
            }
            metrics.update(_extra_distribution_metrics(record, outputs, config))
            for k, v in metrics.items():
                agg[k] += v / n
            if should_plot:
                save_batch_diagnostic_plots(record, outputs, r, phi, config, plot_dir, split_name='train', epoch=epoch)

        if _as_loss_spec(use_timse).pointwise == "combined":
            pairs = [_scored_pairs(record, outputs, config) for record, outputs in per_batch]
            S, _, _ = combined_score(torch.cat([o for o, _ in pairs]), torch.cat([p for _, p in pairs]),
                                     config.combined_grid)
            reg = torch.stack([_smoothness_reg(outputs['v_inner_hat'], config) for _, outputs in per_batch]).mean()
            group_total = S.float() + reg
            pen = nn1.head.penalty() if hasattr(getattr(nn1, 'head', None), 'penalty') else None
            if pen is not None:                                  # curvature penalty of a reduced NAM (--nam-smooth)
                group_total = group_total + pen
                agg['curvature_penalty'] += float(pen.detach()) * len(per_batch) / n
            w_phys = float(getattr(config, 'phys_loss_weight', 0.0))
            if w_phys > 0:
                pp = [_scored_pairs(record, outputs, config, key='ace_phys') for record, outputs in per_batch]
                S_p, _, _ = combined_score(torch.cat([o for o, _ in pp]), torch.cat([p for _, p in pp]), config.combined_grid)
                group_total = group_total + w_phys * S_p.float()
                agg['combined_S_phys'] += float(S_p.detach()) * len(per_batch) / n
            if not torch.isfinite(group_total):
                raise ValueError("Non-finite combined loss")
            agg['combined_S'] += float(S.detach()) * len(per_batch) / n
        else:
            group_total = torch.stack(group_losses).mean()
        _GRAD_BAD.clear()
        group_total.backward()
        if _NAN_DEBUG:
            bad = [nm for nm, p_ in nn1.named_parameters() if p_.grad is not None and not torch.isfinite(p_.grad).all()]
            if bad:
                print(f'  NAN_DEBUG non-finite gradient in {bad} (group {[rec["batch_idx"] for rec, _ in per_batch]}, '
                      f'loss {float(group_total.detach()):.4g})', flush=True)
                print(f'  NAN_DEBUG d(loss)/d(tensor), non-finite of total: {_GRAD_BAD}', flush=True)
                dump = os.environ.get('SWNN_NAN_DUMP')
                if dump:
                    torch.save(dict(_NAN_DUMP), dump)
                    print(f'  NAN_DEBUG solver input and incoming gradient saved to {dump}', flush=True)
                for rec, out in per_batch:
                    _nan_report(f'batch {rec["batch_idx"]}', v_inner_hat=out['v_inner_hat'], v_full=out['v_full'],
                                ace_phys=out['ace_phys'])
                raise ValueError('NAN_DEBUG: non-finite gradient')
        if clip_norm and clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(clip_params, max_norm=float(clip_norm))
        optimizer.step()

    return dict(agg)


def evaluate_records_batched_group(records, nn1, nn2, forward_model, r, phi, config, reg_weight, group_size,
                                    use_timse=True, plot_every=0, plot_dir=None, epoch=None, split_name='val'):
    """Exact batched-group evaluation -- no gradients, so no averaging concern at all."""
    nn1.eval()
    nn2.eval()
    forward_model.eval()

    empty = {
        'boundary_loss': float('nan'), 'forecast_loss': float('nan'), 'reg_loss': float('nan'),
        'total_loss': float('nan'), 'ace_rmse': float('nan'), 'ace_phys_rmse': float('nan'),
        'delta_max': float('nan'),
        'amp_ratio': float('nan'), 'amp_ratio_phys': float('nan'), 'sorted_mse': float('nan'),
        'pod500': float('nan'), 'far500': float('nan'), 'tss500': float('nan'),
        'pod600': float('nan'), 'far600': float('nan'), 'tss600': float('nan'),
        }
    if not records:
        return empty

    should_plot = bool(plot_every) and epoch is not None and (epoch % int(plot_every) == 0)
    group_size = max(1, int(group_size))
    agg = defaultdict(float)
    obs_pool, pred_pool = [], []

    with torch.no_grad():
        for group_start in range(0, len(records), group_size):
            group_records = records[group_start:group_start + group_size]
            per_batch = _run_group_forward(group_records, nn1, nn2, forward_model, r, phi, config, use_timse=use_timse)
            for record, outputs in per_batch:
                o_s, p_s = _scored_pairs(record, outputs, config)
                obs_pool.append(o_s.detach().cpu()); pred_pool.append(p_s.detach().cpu())
                l_boundary = compute_boundary_loss(record, outputs['v_inner_hat'], outputs['v_full'], r, phi, config, use_timse=use_timse)
                l_forecast = compute_tail_loss(record, outputs['ace_final'], config, use_timse=use_timse)
                l_reg = float(reg_weight) * _delta_smoothness(outputs['delta'])
                total = compute_total_loss(l_boundary, l_forecast, l_reg, 1.0, 1.0)
                metrics = {
                    'boundary_loss': float(l_boundary.detach().item()),
                    'forecast_loss': float(l_forecast.detach().item()),
                    'reg_loss': float(l_reg.detach().item()),
                    'total_loss': float(total.detach().item()),
                    'ace_rmse': float(finite_rmse(record['ace_obs'], outputs['ace_final']).detach().item()),
                    'ace_phys_rmse': float(finite_rmse(record['ace_obs'], outputs['ace_phys']).detach().item()),
                    'delta_max': float(outputs['delta'].detach().abs().max().item()),
                }
                metrics.update(_extra_distribution_metrics(record, outputs, config))
                if should_plot:
                    save_batch_diagnostic_plots(record, outputs, r, phi, config, plot_dir, split_name=split_name, epoch=epoch)
                for k, v in metrics.items():
                    agg[k] += v

    n = len(records)
    out = {k: v / n for k, v in agg.items()}
    out['_obs'], out['_pred'] = obs_pool, pred_pool          # pooled scored pairs (not averaged)
    return out


# =============================================================================
# Multi-GPU x batched-group: one batched group of `group_size` batches
# computed per device, concurrently across devices.
# =============================================================================

def _run_group_forward_and_losses(group_records, nn1_r, nn2_r, fm_r, r_r, phi_r, config,
                                   lam_boundary, lam_forecast, reg_weight, use_timse):
    """_run_group_forward plus the per-batch loss/metric computation, factored
    out so both the multi-GPU and single-GPU batched-group paths share it."""
    per_batch = _run_group_forward(group_records, nn1_r, nn2_r, fm_r, r_r, phi_r, config, use_timse=use_timse)

    losses, metrics_list, outputs_list = [], [], []
    for record, outputs in per_batch:
        l_boundary = compute_boundary_loss(record, outputs['v_inner_hat'], outputs['v_full'], r_r, phi_r, config, use_timse=use_timse)
        l_forecast = compute_tail_loss(record, outputs['ace_final'], config, use_timse=use_timse)
        l_reg = float(reg_weight) * _delta_smoothness(outputs['delta'])
        for name, l in (('boundary', l_boundary), ('forecast', l_forecast), ('reg', l_reg)):
            if not torch.isfinite(l):
                raise ValueError(f"Non-finite {name} loss for batch {record['batch_idx']}")
        total = compute_total_loss(l_boundary, l_forecast, l_reg, lam_boundary, lam_forecast)
        if not torch.isfinite(total):
            raise ValueError(f"Non-finite total loss for batch {record['batch_idx']}")
        losses.append(total)
        m = {
            'boundary_loss': float(l_boundary.detach().item()),
            'forecast_loss': float(l_forecast.detach().item()),
            'reg_loss': float(l_reg.detach().item()),
            'total_loss': float(total.detach().item()),
            'ace_rmse': float(finite_rmse(record['ace_obs'], outputs['ace_final']).detach().item()),
            'ace_phys_rmse': float(finite_rmse(record['ace_obs'], outputs['ace_phys']).detach().item()),
            'delta_max': float(outputs['delta'].detach().abs().max().item()),
        }
        m.update(_extra_distribution_metrics(record, outputs, config))
        metrics_list.append(m)
        outputs_list.append((record, outputs))
    return losses, metrics_list, outputs_list


def train_one_epoch_multi_gpu_batched_group(train_records, nn1, nn2, forward_model, optimizer, config,
                                             lam_boundary, lam_forecast, reg_weight, devices, group_size,
                                             use_timse=True, plot_every=0, plot_dir=None, epoch=None, clip_norm=1.0):
    """Combines batched-group training (real GPU-bound speedup per device --
    see train_one_epoch_batched_group) with genuine multi-GPU parallelism
    (one group processed concurrently per device, dispatched via threads --
    see train_one_epoch_multi_gpu).

    The earlier single-batch multi-GPU attempt measured SLOWER than
    single-GPU (0.71x) because dispatching many small, cheap forward passes
    via threads is Python-GIL-bound. Each device here instead runs a full
    group_size-wide BATCHED forward pass per dispatch -- much more real GPU
    work per Python-level call -- which may or may not be enough to make
    genuine cross-device concurrency pay off this time. That's an open
    empirical question this function exists to test, not an assumed win.

    Semantics: batches within one device's group are averaged into one
    combined loss/gradient/step (same as train_one_epoch_batched_group).
    Groups on DIFFERENT devices are NOT averaged with each other -- each
    device's group gets its own, individually-clipped optimizer.step(),
    applied sequentially in the master process after all devices finish
    their concurrent forward+backward. That means the 2nd+ device's step
    within a round is computed against the round's starting weights, not the
    just-updated ones -- the same bounded-staleness tradeoff already
    accepted in train_one_epoch_multi_gpu.
    """
    nn1.train()
    nn2.train()
    forward_model.train()

    should_plot = bool(plot_every) and epoch is not None and (epoch % int(plot_every) == 0)

    master_device = next(nn1.parameters()).device
    clip_params = list(nn1.parameters()) + list(nn2.parameters())
    accel = getattr(forward_model, 'learned_radial_accel', None)
    if accel is not None:
        clip_params += [p for p in accel.parameters() if p.requires_grad]

    order = list(range(len(train_records)))
    rng = np.random.default_rng(int(epoch) if epoch is not None else 0)
    rng.shuffle(order)

    n = max(1, len(train_records))
    agg = defaultdict(float)
    group_size = max(1, int(group_size))
    n_devices = max(1, len(devices))
    round_size = group_size * n_devices

    for round_start in range(0, len(order), round_size):
        round_idx = order[round_start:round_start + round_size]
        device_groups = [round_idx[i:i + group_size] for i in range(0, len(round_idx), group_size)]
        round_devices = devices[:len(device_groups)]

        replicas = [_build_device_replica(nn1, nn2, forward_model, config, dev) for dev in round_devices]

        def _forward_backward_one_group(pos):
            dev = round_devices[pos]
            nn1_r, nn2_r, fm_r, r_r, phi_r = replicas[pos]
            group_records = [_move_record_to_device(train_records[i], dev) for i in device_groups[pos]]

            losses, metrics_list, outputs_list = _run_group_forward_and_losses(
                group_records, nn1_r, nn2_r, fm_r, r_r, phi_r, config,
                lam_boundary, lam_forecast, reg_weight, use_timse,
            )
            group_total = torch.stack(losses).mean()
            group_total.backward()

            accel_r = getattr(fm_r, 'learned_radial_accel', None)
            grads = {
                'nn1': [p.grad.detach().to(master_device) if p.grad is not None else None for p in nn1_r.parameters()],
                'nn2': [p.grad.detach().to(master_device) if p.grad is not None else None for p in nn2_r.parameters()],
                'accel': (
                    [p.grad.detach().to(master_device) if p.grad is not None else None for p in accel_r.parameters()]
                    if accel_r is not None else None
                ),
            }
            return grads, metrics_list, outputs_list, r_r, phi_r

        with ThreadPoolExecutor(max_workers=len(device_groups)) as ex:
            results = list(ex.map(_forward_backward_one_group, range(len(device_groups))))

        for grads, metrics_list, outputs_list, r_r, phi_r in results:
            optimizer.zero_grad(set_to_none=True)
            for p, g in zip(nn1.parameters(), grads['nn1']):
                if g is not None:
                    p.grad = g.clone()
            for p, g in zip(nn2.parameters(), grads['nn2']):
                if g is not None:
                    p.grad = g.clone()
            if accel is not None and grads['accel'] is not None:
                for p, g in zip(accel.parameters(), grads['accel']):
                    if g is not None:
                        p.grad = g.clone()
            if clip_norm and clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(clip_params, max_norm=float(clip_norm))
            optimizer.step()

            for metrics in metrics_list:
                for k, v in metrics.items():
                    agg[k] += v / n
            if should_plot:
                for record, outputs in outputs_list:
                    save_batch_diagnostic_plots(record, outputs, r_r, phi_r, config, plot_dir, split_name='train', epoch=epoch)

    return dict(agg)


def evaluate_records_multi_gpu_batched_group(records, nn1, nn2, forward_model, config, reg_weight, devices,
                                              group_size, use_timse=True, plot_every=0, plot_dir=None,
                                              epoch=None, split_name='val'):
    """Exact parallel batched-group evaluation -- no gradients, so no averaging/staleness concern at all."""
    nn1.eval()
    nn2.eval()
    forward_model.eval()

    empty = {
        'boundary_loss': float('nan'), 'forecast_loss': float('nan'), 'reg_loss': float('nan'),
        'total_loss': float('nan'), 'ace_rmse': float('nan'), 'ace_phys_rmse': float('nan'),
        'delta_max': float('nan'),
        'amp_ratio': float('nan'), 'amp_ratio_phys': float('nan'), 'sorted_mse': float('nan'),
        'pod500': float('nan'), 'far500': float('nan'), 'tss500': float('nan'),
        'pod600': float('nan'), 'far600': float('nan'), 'tss600': float('nan'),
        }
    if not records:
        return empty

    should_plot = bool(plot_every) and epoch is not None and (epoch % int(plot_every) == 0)
    group_size = max(1, int(group_size))
    n_devices = max(1, len(devices))
    groups = [list(range(i, min(i + group_size, len(records)))) for i in range(0, len(records), group_size)]

    replicas = [_build_device_replica(nn1, nn2, forward_model, config, dev) for dev in devices[:n_devices]]

    def _eval_one_group(pos):
        dev = devices[pos % n_devices]
        nn1_r, nn2_r, fm_r, r_r, phi_r = replicas[pos % n_devices]
        group_records = [_move_record_to_device(records[i], dev) for i in groups[pos]]
        with torch.no_grad():
            losses, metrics_list, outputs_list = _run_group_forward_and_losses(
                group_records, nn1_r, nn2_r, fm_r, r_r, phi_r, config,
                1.0, 1.0, reg_weight, use_timse,
            )
        if should_plot:
            for record, outputs in outputs_list:
                save_batch_diagnostic_plots(record, outputs, r_r, phi_r, config, plot_dir, split_name=split_name, epoch=epoch)
        return metrics_list

    with ThreadPoolExecutor(max_workers=n_devices) as ex:
        all_metrics_lists = list(ex.map(_eval_one_group, range(len(groups))))

    agg = defaultdict(float)
    n = len(records)
    for metrics_list in all_metrics_lists:
        for metrics in metrics_list:
            for k, v in metrics.items():
                agg[k] += v
    return {k: v / n for k, v in agg.items()}


# =============================================================================
# Checkpointing
# =============================================================================

def _forward_extra_state(forward_model):
    accel = getattr(forward_model, 'learned_radial_accel', None)
    return accel.state_dict() if accel is not None else None


def _format_learned_radial_accel_status(forward_model):
    """Compact status string for the physics model's learned radial acceleration (for epoch logging).

    These two parameters (L, Vref for the active 'delta_v' source form) are
    jointly optimized along with NN1/NN2 but otherwise never appear in the
    per-epoch log, making it hard to tell whether they've converged.
    """
    accel = getattr(forward_model, 'learned_radial_accel', None)
    if accel is None:
        return "accel=off"
    try:
        params = accel.physical_parameters()
    except Exception:
        return "accel=error"

    if hasattr(accel, 'ddelta_v_dr_rsun') and len(params) == 2:
        L = float(params[0].detach().item())
        vref = float(params[1].detach().item())
        return f"accel_L={L:.3f}Rs accel_Vref={vref:.2f}km/s"

    if len(params) == 3:
        alpha = float(params[0].detach().item())
        L = float(params[1].detach().item())
        v_inf = float(params[2].detach().item())
        return f"accel_alpha={alpha:.3e} accel_L={L:.3f}Rs accel_v_inf={v_inf:.2f}km/s"

    return "accel=unavailable"


_RUN_CONFIG = None          # the run's config (set in main), so checkpoints record its grid and scoring settings


# Groups of the 37 per-cell inputs of the linear feature map (drop-HMI layout: 6 current AIA features; per lag
# 12/24/48/72 h the 6 features + an image-present flag; then log10 f_s, recurrence, recurrence flag), for the group
# lasso (2026-10-19). Lag flags (12, 19, 26, 33) and the bias are not in any group (never penalized or removed).
from boundary import INPUT_GROUPS as LINEAR_GROUPS          # defined in boundary.py since 2026-10-20 (shared)


def _attach_group_sparsity(nn1, lam, keep):
    """Group lasso and/or input selection on a linear feature map, by a gradient hook on its weight (one backward
    per optimizer step in the batched training path, so the penalty is added once per step)."""
    head = getattr(nn1, 'head', None)
    if not (isinstance(head, torch.nn.Sequential) and len(head) == 1 and head[0].in_features == 37):
        raise ValueError('--group-lasso / --input-keep need the linear 37-input feature map (--mlp-hidden-dims none)')
    W = head[0].weight
    mask = torch.ones_like(W)
    if keep:
        unknown = set(keep) - set(LINEAR_GROUPS)
        if unknown:
            raise ValueError(f'unknown groups {sorted(unknown)}; choose from {sorted(LINEAR_GROUPS)}')
        for g, idx in LINEAR_GROUPS.items():
            if g not in keep:
                mask[0, idx] = 0.0
        with torch.no_grad():
            W.mul_(mask)
    groups = [(torch.tensor(idx, device=W.device), len(idx) ** 0.5) for g, idx in LINEAR_GROUPS.items()
              if not keep or g in keep]

    calls = [0]

    def hook(grad):
        calls[0] += 1
        if os.environ.get('GROUP_LASSO_DEBUG') and calls[0] <= int(os.environ['GROUP_LASSO_DEBUG']):
            w = W.detach()
            print('group |grad| / |w|: ' + ', '.join(f'{n} {grad[0, i].norm():.3g}/{w[0, i].norm():.3g}'
                                                    for n, i in LINEAR_GROUPS.items()), flush=True)
        g = grad * mask
        if lam > 0:
            w = W.detach()
            pen = torch.zeros_like(w)
            for idx, sq in groups:
                wg = w[0, idx]
                pen[0, idx] = lam * sq * wg / wg.norm().clamp_min(1e-8)
            g = g + pen * mask
        return g
    W.register_hook(hook)
    print(f"Linear map: group lasso lambda={lam:g}; groups kept: {keep or 'all'}", flush=True)


def _grid_settings(config=None):
    c = config or _RUN_CONFIG or Config()
    return {"r_outer": float(getattr(c, 'r_outer', c.rAU)), "phi_grid_periodic": bool(getattr(c, 'phi_grid_periodic', False)),
            "max_interp_gap_hours_speed": getattr(c, 'max_interp_gap_hours_speed', None),
            "cv_split": getattr(c, 'cv_split', 'b5'), "box_days": getattr(c, 'box_days', None),
            "nn1_dataset_dir": getattr(c, 'nn1_dataset_dir', 'data/nn1_dataset'),
            "nn1_heldout_dir": getattr(c, 'nn1_heldout_dir', 'data/nn1_dataset_heldout'),
            "Omega": float(c.Omega), "propagation_mode": getattr(c, 'propagation_mode', 'solver'),
            "kinematic_lag_h": float(getattr(c, 'kinematic_lag_h', 96.0)),
            "ballistic_sigma_h": float(getattr(c, 'ballistic_sigma_h', 2.0))}


def save_checkpoint(path, nn1, nn2, forward_model, optimizer, epoch, best_val_score, best_epoch,
                     epochs_without_improvement, split, schedule_settings, history_hours, scheduler=None):
    torch.save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "epoch": epoch,
            "best_val_score": best_val_score,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "nn1_state_dict": nn1.state_dict(),
            "nn2_state_dict": nn2.state_dict(),
            "forward_extra_state_dict": _forward_extra_state(forward_model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "split": split,
            "schedule": schedule_settings,
            "history_hours": history_hours,
            "grid": _grid_settings(),
        },
        path,
    )


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"Seed: {args.seed}")

    device, cuda_index = choose_training_device(args)
    print(f"Device: {device}" + (f" (GPU index {cuda_index})" if cuda_index is not None else ""))

    multi_gpu_devices = None
    if args.multi_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("--multi-gpu requires CUDA")
        if args.gpu_indices:
            idxs = [int(tok.strip()) for tok in args.gpu_indices.split(",") if tok.strip()]
        else:
            idxs = list(range(torch.cuda.device_count()))
        if len(idxs) < 2:
            raise RuntimeError(f"--multi-gpu needs at least 2 devices, got {idxs}")
        multi_gpu_devices = [torch.device(f"cuda:{i}") for i in idxs]
        device = multi_gpu_devices[0]
        print(f"Multi-GPU: {len(multi_gpu_devices)} devices {multi_gpu_devices}, master={device}")

    batch_group_size = max(1, int(args.batch_group_size))
    if batch_group_size > 1 and multi_gpu_devices is not None:
        print(f"Batch group size: {batch_group_size} x {len(multi_gpu_devices)} devices "
              f"(multi-GPU batched-group training)")
    elif batch_group_size > 1:
        print(f"Batch group size: {batch_group_size} (batched-group minibatch training)")

    if args.plot_every > 0:
        print(f"Plots: every {args.plot_every} epochs -> {os.path.abspath(args.plot_dir)}")
    else:
        print("Plots: disabled (--plot-every 0)")

    config = Config()
    config.device = str(device)

    if args.batch_days is not None:
        config.Nt = int(round(float(args.batch_days) * 24.0 * 3600.0 / config.dt))
        config.t_end = config.Nt * config.dt
        print(
            f"Overriding batch length: {args.batch_days} days -> Nt={config.Nt} steps "
            f"({config.Nt * config.dt / 3600.0:.1f} hours). config.py's own Nt is untouched; "
            f"this only affects this process."
        )
    if args.batch_overlap_hours is not None:
        config.overlap_hours = float(args.batch_overlap_hours)
    if args.spacecraft:
        config.spacecraft_include = [tok.strip() for tok in args.spacecraft.split(",") if tok.strip()]
    if args.contextual_hidden_dims:
        config.contextual_hidden_dims = tuple(int(tok.strip()) for tok in args.contextual_hidden_dims.split(","))
    if args.mlp_hidden_dims:                     # 'none': no hidden layer (a linear map, sigmoid-bounded)
        config.mlp_hidden_dims = (() if args.mlp_hidden_dims.strip().lower() == 'none'
                                  else tuple(int(tok.strip()) for tok in args.mlp_hidden_dims.split(",")))
    config.propagation_mode = args.propagation
    config.nn1_head = args.nn1_head
    config.nn1_reduce = [g for g in args.nn1_reduce.split(',') if g.strip()]
    config.nam_smooth = float(args.nam_smooth)
    if config.nn1_reduce and args.nn1_head == 'hinge':
        q = [[float(v) for v in t.split(':')] for t in args.hinge_spec.split(',')]
        if len(q) != len(config.nn1_reduce):
            raise ValueError('--hinge-spec needs one lo:init1:init2:hi per --nn1-reduce quantity')
        config.hinge_spec = dict(lo=[a[0] for a in q], init1=[a[1] for a in q], init2=[a[2] for a in q], hi=[a[3] for a in q])
    if config.nn1_reduce and args.nn1_head == 'nam':
        config.nam_hidden = tuple(int(t) for t in args.nam_hidden.split(','))
    if args.nn1_head == 'nam':
        config.nam_key = [int(t) for t in args.nam_key.split(',') if t.strip()]
        config.nam_pairs = [tuple(int(u) for u in t.split('-')) for t in args.nam_pairs.split(',') if t.strip()]
        config.nam_hidden = tuple(int(t) for t in args.nam_hidden.split(','))
    config.kinematic_lag_h, config.ballistic_sigma_h = float(args.kinematic_lag_h), float(args.ballistic_sigma_h)
    if args.shallow_relu_radius is not None:
        config.shallow_relu_radius = int(args.shallow_relu_radius)
    if args.merged_temporal_channels is not None:
        config.merged_temporal_channels = args.merged_temporal_channels
    if args.sh_l_max is not None:
        config.merged_temporal_sh_l_max = args.sh_l_max
    if args.aia_normalization is not None:
        config.aia_normalization_mode = args.aia_normalization

    if args.fold is not None:
        import folds
        import nn1_inputs
        if args.nn1_arch != "feature":
            raise ValueError("--fold uses the feature dataset and requires --nn1-arch feature")
        config.feature_mlp_radius = int(args.feature_mlp_radius)
        train_batches, val_batches, test_batches = folds.FOLDS[args.fold]
        train_batches, val_batches, test_batches = list(train_batches), [val_batches], [test_batches]
        config.cv_split, config.box_days = args.cv_split, args.box_days
        config.nn1_dataset_dir, config.nn1_heldout_dir = args.nn1_dataset_dir, args.nn1_heldout_dir
        nn1_inputs.DATA = Path(args.nn1_dataset_dir)
        if args.nn1_dataset_dir != "data/nn1_dataset":
            print(f"NN1 dataset: {args.nn1_dataset_dir} (held-out: {args.nn1_heldout_dir})")
        roles = None
        if args.cv_split != "b5":
            if args.box_days is None:
                raise ValueError("--cv-split il20 needs --box-days")
            roles = folds.fold_intervals(args.fold, args.cv_split)
            train_batches, val_batches, test_batches = roles['train'], ['val'], ['test']
            print(f"Development split {args.cv_split}, fold {args.fold}: " + "; ".join(
                f"{k} {sum((e - s).days for s, e in v)} d in {len(v)} interval(s)" for k, v in roles.items()))
        obs_df = load_observations(config).sort_values('Time').reset_index(drop=True)
        include_sc = getattr(config, 'spacecraft_include', None)
        if include_sc and 'Spacecraft_ID' in obs_df.columns:
            obs_df = obs_df[obs_df['Spacecraft_ID'].isin(include_sc)].reset_index(drop=True)
            print(f"Spacecraft filter: keeping {include_sc} ({len(obs_df)} observations)")
        config.nn1_v_max = float(args.nn1_vmax)
        if args.no_speed_interp:
            config.max_interp_gap_hours_speed = 0
        if args.freeze_accel:
            config.learned_radial_accel_learnable = False
        if args.accel_vref is not None:
            config.learned_radial_dv_vref = float(args.accel_vref)
        print(f"NN1 output range 150-{config.nn1_v_max:.0f} km/s; speed gaps "
              f"{'not interpolated (measured hours only)' if args.no_speed_interp else 'interpolated up to 5 h'}; "
              f"acceleration {'frozen' if args.freeze_accel else 'learned'}, Vref {config.learned_radial_dv_vref:g} km/s")
        config.nn1_drop_hmi = bool(args.nn1_drop_hmi)
        config.nn1_dropout = float(args.nn1_dropout)
        if args.nn2_rec_offsets:
            config.nn2_rec_offsets_h = [float(x) for x in args.nn2_rec_offsets.split(',') if x.strip()]
            ace_df = obs_df[obs_df['Spacecraft_ID'] == 'ACE'] if 'Spacecraft_ID' in obs_df.columns else obs_df
            config.nn2_rec_series = ace_df.set_index('Time')['Speed'].sort_index().resample('1h').mean()
            print(f"NN2 recurrence inputs: observed ACE speed at T + {config.nn2_rec_offsets_h} h")
        if args.nn1_extra:
            config.nn1_extra_channels = [tok.strip() for tok in args.nn1_extra.split(",") if tok.strip()]
        if args.nn1_lag_hours:
            config.nn1_lag_hours = [float(tok) for tok in args.nn1_lag_hours.split(",") if tok.strip()]
        if args.nn1_fill == "train-median":
            in_tr = folds.in_intervals(pd.to_datetime(obs_df['Time']).to_numpy(), train_batches)
            config.nn1_fill_speed = float(np.nanmedian(obs_df.loc[in_tr, 'Speed'].to_numpy(dtype=float)))
        elif args.nn1_fill != "midpoint":
            config.nn1_fill_speed = float(args.nn1_fill)
        print(f"NN1 fill speed for cells without AIA input: "
              f"{getattr(config, 'nn1_fill_speed', None) or 450.0:.1f} km/s ({args.nn1_fill})")
        batch_specs = nn1_inputs.build_fold_batch_specs(config, obs_df, train_batches,
                                                        chunks_per_batch=int(args.chunks_per_batch),
                                                        roles=roles, box_days=args.box_days)
        if roles is not None:
            train_batches = ['train']
        role = lambda bs: [sp['batch_idx'] for sp in batch_specs if sp['dev_batch'] in bs]
        train_batches, val_batches, test_batches = role(train_batches), role(val_batches), role(test_batches)
        if args.overfit_chunk is not None:
            c = int(args.overfit_chunk)
            train_batches, val_batches, test_batches = [c], [c + 1], [c + 2]
            print(f"Overfit test: training on chunk {c} only (val chunk {c + 1}, test chunk {c + 2})")
        available_ids = [spec['batch_idx'] for spec in batch_specs]
    else:
        batch_specs = build_joint_batch_specs(config)
        available_ids = [spec['batch_idx'] for spec in batch_specs]

    if args.fold is not None:
        pass
    elif args.split_rotating_months:
        months = [int(tok.strip()) for tok in args.split_rotating_months.split(",")]
        if len(months) != 3:
            raise RuntimeError(f"--split-rotating-months needs exactly 3 values (train,val,test), got {months}")
        train_batches, val_batches, test_batches = split_batches_by_rotating_months(batch_specs, *months)
    elif args.split_fractions:
        fractions = [float(tok.strip()) for tok in args.split_fractions.split(",")]
        train_batches, val_batches, test_batches = split_batches_by_fraction(available_ids, fractions)
    else:
        train_batches = parse_batch_list(args.train_batches)
        val_batches = parse_batch_list(args.val_batches)
        test_batches = parse_batch_list(args.test_batches)
    validate_split(train_batches, val_batches, test_batches, available_ids)

    print(f"Train batches ({len(train_batches)}): {train_batches}")
    print(f"Val batches   ({len(val_batches)}): {val_batches}")
    print(f"Test batches  ({len(test_batches)}): {test_batches}")

    specs_by_idx = {spec['batch_idx']: spec for spec in batch_specs}

    r, phi, t, dr, dphi = setup_grid(config)
    global _RUN_CONFIG
    _RUN_CONFIG = config

    print("Building NN1 / physics / NN2 modules...")
    nn1_cls = {
        "mlp": IntensityToVelocityMLP,
        "threshold": ThresholdVelocityModel,
        "contextual": ContextualVelocityMLP,
        "multifeature": MultiFeatureThresholdModel,
        "polynomial": PolynomialVelocityModel,
        "shallow_relu": ShallowReluVelocityModel,
        "merged_temporal": MergedTemporalVelocityModel,
        "feature": FeatureVelocityMLP,
    }[args.nn1_arch]
    if args.sh_l_max is not None and args.nn1_arch != "merged_temporal":
        raise ValueError("--sh-l-max is only supported with --nn1-arch merged_temporal")
    nn1 = nn1_cls(config).to(device)
    print(f"NN1 architecture: {args.nn1_arch} ({nn1_cls.__name__}, "
          f"{sum(p.numel() for p in nn1.parameters() if p.requires_grad)} trainable params)")
    if args.nn1_init:
        ckpt = torch.load(args.nn1_init, map_location=device, weights_only=True)
        state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        nn1.load_state_dict(state_dict)
        print(f"Warm-started NN1 from {args.nn1_init}")

    forward_model = PropagationForwardModel(config, r, phi, dr, dphi).to(device)

    needed_ids = sorted(set(train_batches + val_batches + test_batches))
    records_by_idx = {
        bid: make_batch_record(specs_by_idx[bid], config, device)
        for bid in needed_ids
    }

    if args.exclusion_path:
        std_excl = set(pd.to_datetime(pd.read_csv(args.exclusion_path)['Time']))
        n_std = 0
        for rec in records_by_idx.values():
            n_std += _apply_train_only_exclusion(rec, std_excl, config)
        print(f"Standard-dataset exclusion: {len(std_excl)} timestamps from {args.exclusion_path}, "
              f"applied to ALL splits AFTER short-gap interpolation; {n_std} hours that were "
              f"usable observations are now masked across the {len(records_by_idx)} loaded batch(es).")

    if args.train_only_exclusion_path:
        exclusion_times = set(pd.to_datetime(
            pd.read_csv(args.train_only_exclusion_path)['Time']
        ))
        print(f"Train-only exclusion: {len(exclusion_times)} timestamps from "
              f"{args.train_only_exclusion_path}, applied ONLY to batches in the TRAIN role "
              f"({train_batches}) -- val/test batches ({val_batches}, {test_batches}) are untouched.")
        n_masked_total = 0
        for b in train_batches:
            n_masked_total += _apply_train_only_exclusion(records_by_idx[b], exclusion_times, config)
        print(f"  masked {n_masked_total} ACE hours across the {len(train_batches)} training batch(es)")

    train_records = [records_by_idx[b] for b in train_batches]
    val_records = [records_by_idx[b] for b in val_batches]
    test_records = [records_by_idx[b] for b in test_batches]

    def _scored_obs(recs):
        return np.concatenate([rec['ace_obs'][_transient_skip_steps(config, rec['ace_obs'].shape[0]):].detach().cpu().numpy()
                               for rec in recs])
    # Percentile-TIMSE grids (observations only): training grid for the combined loss, one grid per eval split
    config.combined_grid = percentile_grid(_scored_obs(train_records))
    split_grids = {'val': percentile_grid(_scored_obs(val_records)), 'test': percentile_grid(_scored_obs(test_records))}
    g = config.combined_grid
    print(f"Combined-score grids: train {g['n']} obs, v {g['vmin']:.0f}-{g['vmax']:.0f} km/s, "
          f"q50 {float(g['q'][50]):.0f}, q90 {float(g['q'][90]):.0f}; val {split_grids['val']['n']} obs; "
          f"test {split_grids['test']['n']} obs")

    history_steps = max(1, int(round(args.ace_history_hours)))
    nn2 = ACEResidualHead(history_steps=history_steps, hidden=args.nn2_hidden,
                          rec_offsets_h=list(getattr(config, 'nn2_rec_offsets_h', None) or [])).to(device)
    phase_two_pending = int(args.nn1_only_epochs) > 0
    config.phys_loss_weight = 0.0
    if phase_two_pending:
        if args.disable_nn2:
            raise ValueError('--nn1-only-epochs and --disable-nn2 are exclusive')
        for p_ in nn2.parameters():
            p_.requires_grad_(False)
        print(f"Phase A: NN1 only for at most {args.nn1_only_epochs} epochs or until the validation plateau "
              f"(patience {args.patience}); then NN2 on, physics-only loss weight {args.phys_loss_weight}")
    if args.disable_nn2:
        for p_ in nn2.parameters():
            p_.requires_grad_(False)
        args.inner_floor = 1.0
        print("NN2 disabled: frozen at zero output (ace_final = ace_phys); lambda_boundary pinned at 1")

    accel_lr_scale = (
        float(args.accel_lr_scale) if args.accel_lr_scale is not None
        else float(getattr(config, 'learned_radial_accel_lr_scale', 0.05))
    )
    param_groups = [{'params': list(nn1.parameters()) + [p_ for p_ in nn2.parameters() if p_.requires_grad], 'lr': args.lr}]
    accel = getattr(forward_model, 'learned_radial_accel', None)
    if accel is not None:
        accel_params = [p for p in accel.parameters() if p.requires_grad]
        if accel_params:
            param_groups.append({'params': accel_params, 'lr': args.lr * accel_lr_scale})
    if args.group_lasso > 0 or args.input_keep:
        _attach_group_sparsity(nn1, float(args.group_lasso), [g for g in args.input_keep.split(',') if g.strip()])
    optimizer = (torch.optim.AdamW(param_groups, weight_decay=float(args.weight_decay)) if args.weight_decay > 0
                 else torch.optim.Adam(param_groups))
    if args.weight_decay > 0:
        print(f"Optimizer: AdamW, decoupled weight decay {args.weight_decay:g}")

    scheduler = None
    if not args.disable_adaptive_lr:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min',
            factor=args.lr_scheduler_factor,
            patience=args.lr_scheduler_patience,
            min_lr=args.lr_scheduler_min_lr,
        )

    if args.loss_weighting == "mse+sorted":
        sorted_weight = 1.0 if args.sorted_weight is None else float(args.sorted_weight)
        pointwise = "uniform"
    else:
        sorted_weight = 0.0 if args.sorted_weight is None else float(args.sorted_weight)
        pointwise = args.loss_weighting
    use_timse = LossSpec(pointwise=pointwise, sorted_weight=sorted_weight)
    if pointwise == "combined" and (batch_group_size <= 1 or multi_gpu_devices is not None):
        raise ValueError("--loss-weighting combined is implemented for the single-GPU batched-group path "
                         "(--batch-group-size > 1, no --multi-gpu)")
    print(
        f"Loss weighting: {args.loss_weighting} "
        f"(pointwise={pointwise}, sorted_weight={sorted_weight:g})"
    )

    start_epoch = 0
    best_val_score = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        if not (isinstance(ckpt, dict) and ckpt.get("schema") == CHECKPOINT_SCHEMA):
            raise RuntimeError(f"Resume file is not {CHECKPOINT_SCHEMA} schema")
        nn1.load_state_dict(ckpt["nn1_state_dict"])
        nn2.load_state_dict(ckpt["nn2_state_dict"])
        if ckpt.get("forward_extra_state_dict") is not None and accel is not None:
            accel.load_state_dict(ckpt["forward_extra_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        if args.reset_lr_on_resume:
            restored_lr = optimizer.param_groups[0]['lr']
            optimizer.param_groups[0]['lr'] = args.lr
            if len(optimizer.param_groups) > 1:
                optimizer.param_groups[1]['lr'] = args.lr * accel_lr_scale
            if scheduler is not None:
                # ReduceLROnPlateau caches the lrs it will decay from, plus the
                # plateau counters; both are stale once we override the lr.
                scheduler.best = float('inf')
                scheduler.num_bad_epochs = 0
                scheduler.cooldown_counter = 0
                scheduler._last_lr = [g['lr'] for g in optimizer.param_groups]
            print(
                f"Resume lr override: {restored_lr:.2e} (from checkpoint) -> {args.lr:.2e} (--lr); "
                f"plateau scheduler state cleared"
            )
        if args.reset_best_val_score:
            best_val_score = float("inf")
            best_epoch = -1
            epochs_without_improvement = 0
            print(f"Resumed: epoch={start_epoch}, best_val_score reset (was on a different loss scale)")
        else:
            best_val_score = float(ckpt.get("best_val_score", float("inf")))
            best_epoch = int(ckpt.get("best_epoch", -1))
            epochs_without_improvement = int(ckpt.get("epochs_without_improvement", 0))
            print(f"Resumed: epoch={start_epoch}, best_val_score={best_val_score:.6e}")

    rolling_checkpoint_path = os.path.splitext(args.checkpoint)[0] + "_resume.pt"
    split = {"train": train_batches, "val": val_batches, "test": test_batches}
    schedule_settings = {
        "inner_floor": args.inner_floor, "transition_epochs": args.schedule_transition_epochs, "epochs": args.epochs,
        "loss_weighting": args.loss_weighting,
    }

    print_every = max(1, int(args.print_every))
    eval_every = max(1, int(args.eval_every))
    training_start = time.time()
    print(
        f"Starting training: epochs={args.epochs}, print_every={print_every}, patience={args.patience}, "
        f"eval_every={eval_every}, train/val/test batches={len(train_records)}/{len(val_records)}/{len(test_records)}"
    )

    val_metrics = {
        'boundary_loss': float('nan'), 'forecast_loss': float('nan'), 'reg_loss': float('nan'),
        'total_loss': float('nan'), 'ace_rmse': float('nan'), 'ace_phys_rmse': float('nan'),
        'delta_max': float('nan'),
        'amp_ratio': float('nan'), 'amp_ratio_phys': float('nan'), 'sorted_mse': float('nan'),
        'pod500': float('nan'), 'far500': float('nan'), 'tss500': float('nan'),
        'pod600': float('nan'), 'far600': float('nan'), 'tss600': float('nan'),
        }

    for epoch in range(start_epoch, start_epoch + args.epochs):
        schedule_epoch = (epoch - start_epoch) if args.restart_lambda_schedule else epoch
        lam_boundary, lam_forecast = schedule_lambdas(
            schedule_epoch, transition_epochs=args.schedule_transition_epochs,
            inner_floor=args.inner_floor,
        )

        is_last_epoch = (epoch == start_epoch + args.epochs - 1)
        # Aligned on absolute epoch (not epoch - start_epoch) so that eval epochs
        # coincide with print epochs, which use `epoch % print_every`. On a resume
        # with start_epoch not a multiple of eval_every the two schedules would
        # otherwise never line up and every logged val number would be stale.
        should_eval = (epoch % eval_every == 0) or is_last_epoch

        if multi_gpu_devices is not None and batch_group_size > 1:
            train_metrics = train_one_epoch_multi_gpu_batched_group(
                train_records, nn1, nn2, forward_model, optimizer, config,
                lam_boundary, lam_forecast, args.reg_weight, multi_gpu_devices, batch_group_size,
                use_timse=use_timse, plot_every=args.plot_every, plot_dir=args.plot_dir, epoch=epoch,
                clip_norm=args.grad_clip_norm,
            )
            if should_eval:
                val_metrics = evaluate_records_multi_gpu_batched_group(
                    val_records, nn1, nn2, forward_model, config, args.reg_weight, multi_gpu_devices, batch_group_size,
                    use_timse=use_timse, plot_every=args.plot_every, plot_dir=args.plot_dir, epoch=epoch, split_name='val',
                )
        elif multi_gpu_devices is not None:
            train_metrics = train_one_epoch_multi_gpu(
                train_records, nn1, nn2, forward_model, optimizer, config,
                lam_boundary, lam_forecast, args.reg_weight, multi_gpu_devices, use_timse=use_timse,
                plot_every=args.plot_every, plot_dir=args.plot_dir, epoch=epoch, clip_norm=args.grad_clip_norm,
            )
            if should_eval:
                val_metrics = evaluate_records_multi_gpu(
                    val_records, nn1, nn2, forward_model, config, args.reg_weight, multi_gpu_devices,
                    use_timse=use_timse, plot_every=args.plot_every, plot_dir=args.plot_dir, epoch=epoch, split_name='val',
                )
        elif batch_group_size > 1:
            train_metrics = train_one_epoch_batched_group(
                train_records, nn1, nn2, forward_model, optimizer, r, phi, config,
                lam_boundary, lam_forecast, args.reg_weight, batch_group_size, use_timse=use_timse,
                plot_every=args.plot_every, plot_dir=args.plot_dir, epoch=epoch, clip_norm=args.grad_clip_norm,
            )
            if should_eval:
                val_metrics = evaluate_records_batched_group(
                    val_records, nn1, nn2, forward_model, r, phi, config, args.reg_weight, batch_group_size,
                    use_timse=use_timse, plot_every=args.plot_every, plot_dir=args.plot_dir, epoch=epoch, split_name='val',
                )
        else:
            train_metrics = train_one_epoch(
                train_records, nn1, nn2, forward_model, optimizer, r, phi, config,
                lam_boundary, lam_forecast, args.reg_weight, use_timse=use_timse,
                plot_every=args.plot_every, plot_dir=args.plot_dir, epoch=epoch, clip_norm=args.grad_clip_norm,
            )
            if should_eval:
                val_metrics = evaluate_records(
                    val_records, nn1, nn2, forward_model, r, phi, config, args.reg_weight, use_timse=use_timse,
                    plot_every=args.plot_every, plot_dir=args.plot_dir, epoch=epoch, split_name='val',
                )
        if '_obs' in val_metrics:
            val_pool = pooled_score_metrics(val_metrics['_obs'], val_metrics['_pred'], split_grids['val'])
        else:
            val_pool = {'S': float('nan'), 'rmse': float('nan'), 'trmse': float('nan'), 'bias': float('nan')}
        val_score = val_pool['S'] if args.loss_weighting == "combined" else val_metrics['forecast_loss']

        if should_eval and scheduler is not None and np.isfinite(val_score):
            scheduler.step(val_score)

        improved = should_eval and np.isfinite(val_score) and (val_score < best_val_score)
        if improved:
            best_val_score = val_score
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                args.checkpoint, nn1, nn2, forward_model, optimizer, epoch, best_val_score, best_epoch,
                epochs_without_improvement, split, schedule_settings, args.ace_history_hours,
                scheduler=scheduler,
            )
        elif should_eval:
            epochs_without_improvement += 1

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                rolling_checkpoint_path, nn1, nn2, forward_model, optimizer, epoch, best_val_score, best_epoch,
                epochs_without_improvement, split, schedule_settings, args.ace_history_hours,
                scheduler=scheduler,
            )

        if epoch % print_every == 0:
            lr_parts = []
            if len(optimizer.param_groups) > 0:
                lr_parts.append(f"lr_boundary={optimizer.param_groups[0]['lr']:.2e}")
            if len(optimizer.param_groups) > 1:
                lr_parts.append(f"lr_accel={optimizer.param_groups[1]['lr']:.2e}")
            lr_str = " ".join(lr_parts)
            accel_str = _format_learned_radial_accel_status(forward_model)
            val_tag = "" if should_eval else " (stale, not evaluated this epoch)"
            print(
                f"Epoch {epoch:04d} {'A' if phase_two_pending else 'B'} | lam_boundary={lam_boundary:.3f} lam_forecast={lam_forecast:.3f} | "
                f"train_total={train_metrics['total_loss']:.4e} train_boundary={train_metrics['boundary_loss']:.4e} "
                f"train_forecast={train_metrics['forecast_loss']:.4e} train_reg={train_metrics['reg_loss']:.4e} | "
                f"val_forecast={val_metrics['forecast_loss']:.4e} val_mse={val_metrics.get('fc_mse', float('nan')):.4e} "
                f"val_timse={val_metrics.get('fc_timse', float('nan')):.4e} "
                f"val_rmse_scored={math.sqrt(val_metrics['fc_sse'] / max(val_metrics['fc_n'], 1)) if 'fc_sse' in val_metrics else float('nan'):.2f} "
                f"val_S={val_pool['S']:.2f} val_rmse_pooled={val_pool['rmse']:.2f} val_trmse={val_pool['trmse']:.2f} "
                f"val_bias={val_pool['bias']:+.1f} train_S={train_metrics.get('combined_S', float('nan')):.2f} "
                f"val_ace_rmse={val_metrics['ace_rmse']:.2f} "
                f"val_ace_phys_rmse={val_metrics['ace_phys_rmse']:.2f}{val_tag} dmax={train_metrics['delta_max']:.3e} | "
                f"amp_tr={train_metrics['amp_ratio']:.2f} amp_val={val_metrics['amp_ratio']:.2f} "
                f"tss500_val={val_metrics['tss500']:.2f} tss600_val={val_metrics['tss600']:.2f} | "
                f"{lr_str} | {accel_str}",
            )

        if phase_two_pending and ((epoch - start_epoch + 1) >= int(args.nn1_only_epochs)
                                  or epochs_without_improvement >= args.patience):
            if os.path.exists(args.checkpoint):
                best = torch.load(args.checkpoint, map_location=device, weights_only=True)
                nn1.load_state_dict(best["nn1_state_dict"])
                if best.get("forward_extra_state_dict") is not None and accel is not None:
                    accel.load_state_dict(best["forward_extra_state_dict"], strict=False)
            for p_ in nn2.parameters():
                p_.requires_grad_(True)
            optimizer.add_param_group({'params': list(nn2.parameters()), 'lr': args.lr})
            if scheduler is not None:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=args.lr_scheduler_factor,
                    patience=args.lr_scheduler_patience, min_lr=args.lr_scheduler_min_lr)
            config.phys_loss_weight = float(args.phys_loss_weight)
            epochs_without_improvement = 0
            phase_two_pending = False
            print(f"Phase B from epoch {epoch + 1}: NN1 restored to best phase-A epoch {best_epoch} "
                  f"(val S {best_val_score:.3f}); NN2 on (lr {args.lr:g}); loss S(final) + {config.phys_loss_weight:g} x S(physics)",
                  flush=True)
            continue

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}, val_score {best_val_score:.6e})")
            break

    print(f"Training finished in {(time.time() - training_start) / 60.0:.2f} min")

    if os.path.exists(args.checkpoint):
        best = torch.load(args.checkpoint, map_location=device, weights_only=True)
        nn1.load_state_dict(best["nn1_state_dict"])
        nn2.load_state_dict(best["nn2_state_dict"])
        if best.get("forward_extra_state_dict") is not None and accel is not None:
            accel.load_state_dict(best["forward_extra_state_dict"], strict=False)
        print(f"Loaded best checkpoint: epoch={best.get('best_epoch', -1)} val_score={best.get('best_val_score', float('nan')):.6e}")

    if multi_gpu_devices is not None and batch_group_size > 1:
        test_metrics = evaluate_records_multi_gpu_batched_group(
            test_records, nn1, nn2, forward_model, config, args.reg_weight, multi_gpu_devices, batch_group_size,
            use_timse=use_timse,
        )
    elif multi_gpu_devices is not None:
        test_metrics = evaluate_records_multi_gpu(
            test_records, nn1, nn2, forward_model, config, args.reg_weight, multi_gpu_devices, use_timse=use_timse,
        )
    elif batch_group_size > 1:
        test_metrics = evaluate_records_batched_group(
            test_records, nn1, nn2, forward_model, r, phi, config, args.reg_weight, batch_group_size, use_timse=use_timse,
        )
    else:
        test_metrics = evaluate_records(test_records, nn1, nn2, forward_model, r, phi, config, args.reg_weight, use_timse=use_timse)
    print("\nFinal test metrics")
    print("------------------")
    print(f"Forecast loss ({args.loss_weighting}): {test_metrics['forecast_loss']:.6e}")
    print(f"Boundary loss (inverse loss) : {test_metrics['boundary_loss']:.6e}")
    print(f"ACE RMSE (final)              : {test_metrics['ace_rmse']:.3f} km/s")
    print(f"ACE RMSE (physics only)       : {test_metrics['ace_phys_rmse']:.3f} km/s")
    if 'fc_sse' in test_metrics:
        print(f"ACE RMSE (scored window, pooled): {math.sqrt(test_metrics['fc_sse'] / max(test_metrics['fc_n'], 1)):.3f} km/s")
        print(f"Forecast MSE / TIMSE (scored)  : {test_metrics['fc_mse']:.4e} / {test_metrics['fc_timse']:.4e}")
    if '_obs' in test_metrics:
        tp = pooled_score_metrics(test_metrics['_obs'], test_metrics['_pred'], split_grids['test'])
        print(f"Combined score S (test grid)   : {tp['S']:.3f} km/s  (RMSE {tp['rmse']:.3f}, TRMSE {tp['trmse']:.3f}, "
              f"bias {tp['bias']:+.2f}, n={tp['n']})")
    print(f"Amplitude ratio std(pred)/std(obs): {test_metrics['amp_ratio']:.3f} (physics only {test_metrics['amp_ratio_phys']:.3f})")
    print(f"Sorted/Wasserstein MSE        : {test_metrics['sorted_mse']:.6e} (km/s)^2")
    print(f"v>500 km/s  POD={test_metrics['pod500']:.3f} FAR={test_metrics['far500']:.3f} TSS={test_metrics['tss500']:.3f}")
    print(f"v>600 km/s  POD={test_metrics['pod600']:.3f} FAR={test_metrics['far600']:.3f} TSS={test_metrics['tss600']:.3f}")

    torch.save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "nn1_state_dict": nn1.state_dict(),
            "nn2_state_dict": nn2.state_dict(),
            "forward_extra_state_dict": _forward_extra_state(forward_model),
            "split": split,
            "schedule": schedule_settings,
            "history_hours": args.ace_history_hours,
            "grid": _grid_settings(config),
            "best_epoch": best_epoch,
            "best_val_score": best_val_score,
            "test_metrics": test_metrics,
        },
        args.output_model,
    )
    print(f"Saved final model to {args.output_model}")


if __name__ == "__main__":
    main()
