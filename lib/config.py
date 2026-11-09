import torch


class Config:
    """Configuration parameters for the propagation model and inversion."""

    # Physical constants
    # Rotation rate of the co-rotating frame [rad/s]. The longitude grid and the spacecraft tracks are Carrington, so the
    # frame must rotate at the Carrington sidereal rate (period 25.38 d). Until 2026-10-16 this was 2.97e-6 (24.49 d),
    # 3.6 % too fast; checkpoints record the value they were trained with (grid['Omega']; absent = 2.97e-6).
    Omega = 2.0 * 3.141592653589793 / (25.38 * 86400.0)   # 2.8653e-6
    # Link from the inner boundary to the spacecraft: 'solver' (the transport solver), or for the factorial test of
    # the solver 'fixed_lag' / 'ballistic' (train_joint_forecast.kinematic_track; --propagation)
    propagation_mode = 'solver'
    kinematic_lag_h = 96.0
    ballistic_sigma_h = 2.0
    use_corotating_frame = True  # If input longitudes are inertial, rotate to corotating via +Omega*t
    observation_longitude_frame = "carrington"  # 'carrington' (default) or 'inertial'

    # Grid parameters
    r0 = 21.5                  # Inner boundary [Rs]
    rAU = 215.0                # 1 AU in solar radii
    r_outer = 217.0            # outer edge of the radial grid [Rs]: ACE reaches 216.5 Rs at aphelion (2026-10-08; was rAU)
    phi_grid_periodic = True   # longitude cells at j*2pi/Nphi (2026-10-08; was linspace(0, 2pi, Nphi) incl. 2pi,
                               # spacing 2pi/(Nphi-1), inconsistent with the periodic ring and the NN1 inputs)
    # Nr=64 (was 50): with the MUSCL radial scheme below, going past ~64 buys
    # almost nothing on real boundary data (1 AU span 349.7 at Nr=64 vs 351.0 at
    # Nr=80, both from a 600 km/s boundary span) while the explicit-CFL safety
    # margin drops from 1.32x to 1.05x at the v=900 km/s clamp. 64 keeps the
    # headroom and essentially all of the accuracy.
    Nr = 64                   # Number of radial points
    Nphi = 128                  # Number of longitude points
    # Batch length, chosen to divide the 2011-2021 observation window (95,913
    # unique hourly timestamps across ACE+STEREO_A, enable_multispacecraft_loss=
    # True) into exactly 5 batches for the standing 5-fold protocol -- an even
    # split (4 batches of 19,351 obs-hours / ~806 days, 1 of 18,909 / ~788 days,
    # only 2.3% shorter), not an arbitrary value; verified directly against
    # build_joint_batch_specs, not just computed by formula. The previous
    # Nt=30618 was similarly reverse-engineered to give exactly 4 batches on the
    # old 2018-2024 (7-year) window -- same approach, different target count,
    # since the window length changed. Changing the observation window again
    # will change the batch count again; recompute Nt (see the search in the
    # session log around the 2011-2021 window change) rather than accepting
    # whatever count falls out, if a specific fold count matters again.
    Nt = 38700                # Number of time steps
    # Time stepping
    dt = 1800.0                 # Time step [seconds] - CFL stable for Nr=100
    t_start = 0.0
    t_end = Nt * dt

    # Transit time from r0 to 1AU at ~400 km/s is ~94 hours.
    # The model window (Nt*dt) must exceed this so the BC can influence
    # observations at 1AU. With Nt=1296 and dt=1800 s, one batch spans
    # 648 hours = 27 days.

    # Regularization weights
    # Strongly increased per improvements.md (Phase 1):
    # Reduces inverse ill-conditioning and oscillatory null-space solutions.
    lambda_phi = 1e-3          # Longitude smoothing (increased from 1e-4 for stability)
    lambda_t = 1e-3            # Time smoothing (increased from 1e-5 for stability)

    # Loss weighting: 'uniform' = standard MSE, 'velocity' = (V_obs/V_mean)^2 weighted MSE
    loss_weighting = "uniform"

    # Optimization
    lr = 1e-2                  # Learning rate (conservative to avoid NaN)
    N_epochs = 2000            # Number of epochs (more for slower convergence)
    # Temporal mini-batch training for observation misfit.
    # 0 disables mini-batching (full-window loss each optimizer step).
    time_batch_size = 0
    # Optional cap on optimizer steps per epoch when mini-batching is enabled.
    # 0 means no cap.
    max_minibatches_per_epoch = 0
    # Shuffle temporal mini-batch order each epoch (deterministic by seed).
    temporal_shuffle_minibatches = True
    temporal_shuffle_seed = 42
    # When mini-batching is enabled, evaluate full-window loss each epoch for
    # scheduler/early-stop/best-model selection.
    full_window_validation = True
    adaptive_lr = True         # Enable ReduceLROnPlateau scheduler
    lr_scheduler_factor = 0.8  # LR decay factor when loss plateaus
    lr_scheduler_patience = 20 # Epochs with no improvement before reducing LR
    lr_scheduler_min_lr = 1e-5 # Floor for learning rate

    adaptive_epochs = True     # Enable dynamic stopping/extension of epochs
    min_epochs = 100           # Minimum epochs before adaptive stopping is allowed
    max_epochs = 2000         # Hard cap when adaptive epochs are enabled
    early_stop_patience = 120  # Stop if no meaningful improvement for this many epochs
    early_stop_min_delta = 1e-3  # Minimum loss improvement to reset patience
    early_stop_max_ae = 10.0    # Stop early once max absolute error falls below this threshold [km/s]
    target_mse = -1.0          # Disable MSE target early stop by default
    print_every = 20           # Epoch logging interval

    sensitivity_aggregation = "squared"  # Scalar aggregator for sensitivity target; 'squared' or 'abs'
 
    # Performance: disable NaN/Inf checks in production (each check is a GPU sync)
    debug_nan_checks = False
    compile_rk2_rhs = False         # Compile only rk2_tvd_split RHS kernel (lower overhead than full-forward compile)

    # Periodic checkpoint during optimization (saves .npz every N epochs for restart)
    checkpoint_every = 20      # 0 to disable

    # Optional two-stage optimization: fast preconditioner then full refinement
    use_fast_preconditioner = False
    preconditioner_scheme = "rk4_fast"   # Usually rk4_fast
    preconditioner_lr = 2e-2              # Learning rate for preconditioning stage
    preconditioner_min_epochs = 40
    preconditioner_max_epochs = 400
    preconditioner_target_mse = 300.0     # Loose target before switching to refinement

    # Optional resume/finetune initialization from saved results
    resume_from_saved_boundary = True  # If True, loads from saved_boundary_path or searches saved_boundary_dir
    saved_boundary_path = None            # Explicit path; overrides directory search when set
    saved_boundary_dir = "inverse_results_acc"  # Directory containing batch_XXX_boundary.npz files
    saved_boundary_pattern = "batch_*_boundary.npz"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    forward_scheme = "rk2_tvd_split"      # 'fv', 'rk4_fast', 'rk4_time_upwind', or 'rk2_tvd_split'
    # Radial advection discretization used by rk2_tvd_split:
    #   'muscl'       - 2nd-order MUSCL Godunov flux (default)
    #   'first_order' - the original 1st-order upwind flux (much more diffusive)
    #
    # The first-order scheme's O(dr) numerical diffusion was measured to destroy
    # the boundary condition's time structure before it reached 1 AU: for a
    # sinusoidal boundary, amplitude retention at 1 AU was 82% at a 120 h
    # period, 31% at 48 h and only 2% at 24 h (matching the analytic prediction
    # exp(-2*pi^2*dr*(1-nu)*L/lambda^2), so this was scheme diffusion, not a bug).
    # Solar wind stream structure lives in that 0.5-3 day band, so the forward
    # model was deleting almost exactly the signal the boundary network needed
    # gradient signal from. MUSCL at Nr=64 retains 100%/99%/80% at those same
    # periods for no meaningful extra cost (re-measured 2026-10-16 with
    # reporting/part1/figures/fig_verification.py, A = 10 km/s, residual acceleration
    # off; the first-order values there are 86%/39%/3%, and MUSCL follows the
    # exact steepening solution to 3 km/s rms; an earlier note gave 104/67/32).
    radial_advection_scheme = "muscl"
    # Gradient-checkpoint the rk2_tvd_split time loop in blocks of this many
    # steps (0 disables). Engages only when gradients are being taken AND the
    # estimated activation memory would exceed ~60% of free GPU memory, so runs
    # that fit comfortably never pay the recompute penalty. Needed because MUSCL
    # retains ~2x the autograd intermediates per step of the old first-order
    # scheme (2.91 vs 1.54 MB/step at Nr=64), which takes a full 638-day batch
    # (Nt=30618) from ~46 GB to ~87 GB and OOMs an 80 GB card. Trades one
    # extra forward pass for a large drop in peak memory.
    time_checkpoint_chunk = 256
    use_learned_radial_acceleration = True
    learned_radial_accel_learnable = True
    learned_radial_source_form = "delta_v"  # 'delta_v' (HUX-style) or 'relaxation'
    learned_radial_accel_lr_scale = 0.1
    learned_radial_dv_L = 50.0
    learned_radial_dv_vref = 40.0
    learned_radial_dv_learn_vref = True
#    learned_radial_accel_alpha = 5e-6
#    learned_radial_accel_L = 50.0
#    learned_radial_accel_v_inf = 450.0

    # Input data span. None uses the full file; set e.g. 24*90 for the first 90 days.
    input_hours_limit = None

    # Batch processing
    parallel_batch_devices = None         # Auto-detect free GPUs; or e.g. ['cuda:0', 'cuda:2']
    parallel_allow_shared_device = False  # False avoids multiple workers on one CUDA device
    overlap_hours = 100.0       # Overlap between batches (≥ transit time ~94h)
    max_interp_gap_hours = 6    # Fill only NaN gaps up to this many hourly steps; longer gaps remain NaN

    # Boundary state reduction
    boundary_basis = "fourier"   # "full" or "fourier"
    fourier_max_mode = 5        # Keep modes k=0..K in longitude

    # Warm-start using a constant boundary prior before fitting data.
    warmup_prior_epochs = 0    # Number of warm-start epochs using constant prior target
    warmup_prior_value = 450.0   # Constant prior target for v_inner during warm-up
    warmup_prior_weight = 1.0    # Weight of constant prior loss term during warm-up

    # Time-domain low-pass smoothing of reconstructed FNO output.
    # Keep only the first K Fourier modes in time after inverse-transforming
    # the FNO output to [Nphi, Nt]. Set to 0 to disable smoothing.
    fno_time_lowpass_modes = 0

    # Fourier Neural Operator mapping from AIA imagery to inner boundary velocity
    
    # Window radius (pixels, longitude axis) for ContextualVelocityMLP's
    # wide-window features (I_wide, U_wide). Nphi=128 over 360 deg, so 15px
    # ~= 42 deg each side (~84 deg total) -- comparable to typical coronal
    # hole angular scales, vs. IntensityToVelocityMLP's radius=3 (~8.4 deg
    # each side) stencil.
    # Hidden-layer widths for IntensityToVelocityMLP's head (--nn1-arch
    # mlp). Default (100,64,32) is the original architecture, 10,077
    # params. (27,16,8) is the "shallow, same 14-input stencil" variant,
    # ~998 params.
    mlp_hidden_dims = (100, 64, 32)
    contextual_window_radius = 15
    # Local stencil radius for ShallowReluVelocityModel (--nn1-arch
    # shallow_relu). Default 1 -> 6 inputs (~803 params); 2 -> 10 inputs
    # (~1203 params). Still a LOCAL stencil, unlike contextual_window_radius
    # above which is Contextual/MultiFeature's wide NON-LOCAL window.
    shallow_relu_radius = 1
    # Hidden width for MergedTemporalVelocityModel's single ReLU layer
    # (--nn1-arch merged_temporal). 28 inputs -> hidden=8 gives 243 params.
    merged_temporal_hidden = 8
    # "both" (default, 28 inputs) or "hmi_only" (14 inputs, drops the AIA
    # stencil/I_center/I_wide) -- an ablation for how much of the model's
    # skill comes from AIA vs. HMI alone.
    merged_temporal_channels = "both"
    # Milestone 6 global-SH ablation: None (default, exact prior 28-input
    # model) or 1/2/3/5 to add (L_max+1)**2 global-magnetic-state features
    # (l=0 retained) at t and t-1, rotated per evaluation longitude. See
    # MergedTemporalVelocityModel and preprocess_hmi_sh.py.
    merged_temporal_sh_l_max = None
    sh_data_path = "./data/preprocessed_window/hmi_sh_coeffs_20180101_20241231T235959.npz"
    # How MergedTemporalVelocityModel's local AIA stencil is normalized before
    # entering the network. Only two options exist -- "global" (fixed dataset-
    # wide mean/std, config.aia_global_mean/aia_global_std, set once at data
    # load time in inverse_solver.py's _load_all_image_data) or "asinh" (clip
    # at aia_asinh_clip_dn, an aiapy-style log-like stretch with shape
    # parameter aia_asinh_a, mapped to [-1,1]). A third option, per-frame
    # normalization (subtracting each timestep's own visible-cell mean/std),
    # was tried and reverted: it collapsed the domain-classifier AUC between
    # folds from ~0.88-1.00 to ~0.58-0.65, but per-fold PCA/severity checks
    # showed it also removes real cross-fold signal along with the spurious
    # degradation trend, rather than separating the two -- see
    # degradation_pca_check.py / asinh_stretch_check.py / the corresponding
    # session log. Its implementation has been deleted from
    # _compute_contextual_features, not merely defaulted away from, so it
    # cannot be selected even by a typo -- do not reintroduce it without
    # rerunning that severity check on whatever replaces it.
    aia_normalization_mode = "global"
    # Shape parameters for aia_normalization_mode="asinh". aia_asinh_clip_dn
    # is set at data-load time (inverse_solver.py) to the 99th percentile of
    # visible raw AIA-193 DN over the currently loaded dataset; aia_asinh_a
    # is a fixed shape constant (not a data statistic), per the paper-derived
    # transform: y = 2*asinh(clip(x,0,clip_dn)/clip_dn / a)/asinh(1/a) - 1.
    aia_asinh_clip_dn = None
    aia_asinh_a = 0.005
    # Hidden-layer widths for ContextualVelocityMLP's head. Default (16, 8)
    # matches the first (entry 10/12) attempt: 227 params total. (16, 16, 8)
    # is the "deeper, ~2x params" variant (499 total).
    contextual_hidden_dims = (16, 8)

    # 2011-2021 window (not 2018-2024): chosen so AIA-193's degradation-correction
    # table is trustworthy throughout (flat/extrapolated past 2021, per
    # aia_degradation_factor_check.py) and 2010 is excluded for being a partial,
    # early-mission stub. AIA/HMI share a common hourly grid (76,385 timestamps);
    # aia_global_mean/std and aia_asinh_clip_dn are recomputed from this data at
    # load time (inverse_solver.py), not hardcoded here, so no separate update is
    # needed for those. PSP is absent from observation_data_path by physical
    # necessity (launched 2018-08-12, after this window) -- not a bug.
    aia_data_path = "./data/preprocessed_window/aia_equatorial_slices_20110101_20211231T235959.npz"
    hmi_data_path = "./data/preprocessed_window/hmi_equatorial_slices_20110101_20211231T235959.npz"
    observation_data_path = "./data/preprocessed_window/multi_spacecraft_obs_20110101_20211231T235959.csv"
    fno_modes_phi = 5
    fno_width = 20

    # Multi-spacecraft loss
    enable_multispacecraft_loss = True

    # ========================================================================
    # IMPROVEMENTS PHASE 1 (IMPLEMENTED 2026-05-15)
    # ========================================================================
    # Key changes made per improvements.md recommendations to reduce inverse
    # ill-conditioning and oscillatory null-space solutions:
    #
    # 1. ✅ Increased regularization weights:
    #    - lambda_phi: 1e-4 → 1e-1 (100x increase)
    #    - lambda_t: 1e-5 → 1e-1 (10k increase)
    #    Rationale: Much stronger penalty on spatial/temporal oscillations.
    #
    # 2. ✅ Implemented second-order (curvature) regularization in loss.py:
    #    - Replaced (v[i+1] - v[i])^2 with (v[i+1] - 2*v[i] + v[i-1])^2
    #    - Much more effective at suppressing ringing and checkerboard modes.
    #
    # 3. ✅ Replaced hard clamping with soft sigmoid constraints in boundary.py:
    #    - Removes dead gradients and optimization kinks from clamp() operator.
    #    - Maintains smooth differentiability everywhere.
    #
    # ========================================================================
    # FUTURE IMPROVEMENTS (PHASE 2–3)
    # ========================================================================
    # Recommended long-term enhancements per improvements.md:
    #
    # PHASE 2:
    #  - Move to co-rotating frame: phi_corot = phi + Omega*t
    #    (Reduces temporal variability, improves interpretability, reduces redundancy)
    #  - Solve for residuals relative to ballistic prior: delta_v = v - v_ballistic
    #    (Much better conditioning, smaller null-space)
    #  - Reduce state dimensionality via Fourier basis:
    #    v(phi,t) = Σ a_k(t) exp(i k phi), k ≤ 8–16
    #    (Massive reduction in inverse instability)
    #
    # PHASE 3:
    #  - Add multiple spacecraft: STEREO-A, Solar Orbiter, Parker Solar Probe
    #    (Increases geometric diversity, observable subspace rank)
    #  - Add characteristic-aware regularization along Parker spirals
    #    (Aligns regularization with true physics)
    #  - Transition toward data assimilation:
    #    • Reduced-order 4DVar
    #    • Ensemble Kalman smoothing
    #    • Heliospheric tomography
