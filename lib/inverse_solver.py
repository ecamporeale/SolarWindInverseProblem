"""
Inverse problem solver for inferring solar wind boundary conditions
via backpropagation through a differentiable propagation numerical model.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path

from parallel import _run_parallel_batches, execute_batch_pipeline
from save_io import _save_batch_summary_and_loss_plot
from helpers import setup_grid, load_image_slices

# Module-level flag: when False, NaN/Inf checks in hot-path functions are
# skipped, eliminating GPU-CPU syncs that stall the pipeline.
# Set from config.debug_nan_checks at the start of each forward call.
_DEBUG_NAN = False


# =========================================================
# BATCH PROCESSING WITH SEQUENTIAL WINDOWS
# =========================================================


def _slice_image_batch(all_image_data, batch_start_time, nt, dt_seconds):
    if all_image_data is None:
        return None

    aia_slices = all_image_data.get("aia_slices")
    hmi_slices = all_image_data.get("hmi_slices")
    aia_times = all_image_data.get("times")
    visibility_mask = all_image_data.get("visibility_mask")

    if aia_slices is None:
        return None

    if visibility_mask is None:
        visibility_mask = np.isfinite(aia_slices).astype(np.float32)

    total_time = aia_slices.shape[1]
    if total_time == 0:
        return None

    if aia_times is None:
        raise ValueError("AIA time stamps are required for batch slicing")

    aia_times = pd.to_datetime(np.asarray(aia_times))
    target_times = pd.date_range(
        start=pd.to_datetime(batch_start_time),
        periods=nt,
        freq=pd.Timedelta(seconds=dt_seconds),
    )

    old_t = aia_times.view('int64')
    new_t = target_times.view('int64')
    right_idx = np.searchsorted(old_t, new_t, side='left')
    left_idx = np.clip(right_idx - 1, 0, total_time - 1)
    right_idx = np.clip(right_idx, 0, total_time - 1)

    left_dt = np.abs(new_t - old_t[left_idx])
    right_dt = np.abs(old_t[right_idx] - new_t)
    choose_right = right_dt < left_dt
    nearest_idx = np.where(choose_right, right_idx, left_idx)

    # Do not fabricate image frames far away in time.
    tolerance_ns = int(max(dt_seconds, 3600.0) * 1e9)
    nearest_dt = np.minimum(left_dt, right_dt)

    batch_aia = np.zeros((aia_slices.shape[0], nt), dtype=np.float32)
    batch_visibility = np.zeros((aia_slices.shape[0], nt), dtype=np.float32)

    valid_target = nearest_dt <= tolerance_ns
    if np.any(valid_target):
        sampled_visibility = (
            visibility_mask[:, nearest_idx[valid_target]] > 0.5
        ).astype(np.float32, copy=False)
        sampled_aia = aia_slices[:, nearest_idx[valid_target]].astype(np.float32, copy=False)
        sampled_aia = np.where(sampled_visibility > 0.5, np.nan_to_num(sampled_aia, nan=0.0), 0.0)
        batch_aia[:, valid_target] = sampled_aia
        batch_visibility[:, valid_target] = sampled_visibility

    batch_hmi = None

    if hmi_slices is not None:

        if hmi_slices.shape != aia_slices.shape:
            raise ValueError(
                f"AIA shape {aia_slices.shape} != HMI shape {hmi_slices.shape}"
            )

        batch_hmi = np.zeros(
            (hmi_slices.shape[0], nt),
            dtype=np.float32,
        )

        if np.any(valid_target):

            sampled_hmi = (
                hmi_slices[:, nearest_idx[valid_target]]
                .astype(np.float32, copy=False)
            )

            sampled_hmi = np.where(
                sampled_visibility > 0.5,
                np.nan_to_num(sampled_hmi, nan=0.0),
                0.0,
            )

            batch_hmi[:, valid_target] = sampled_hmi
            
    if batch_hmi is not None:
        return np.stack(
            [
                batch_aia,
                batch_visibility,
                batch_hmi,
            ],
            axis=0,
        )            
    
    return np.stack(
        [
            batch_aia,
            batch_visibility,
        ],
        axis=0,
    )


def _build_batch_specs(v_obs_df, batch_size_steps, stride_steps, warmup_hours, all_image_slices=None, model_nt=None, model_dt_seconds=None):
    """Create the actual batch windows used by both sequential and parallel modes."""
    specs = []
    total_steps = len(v_obs_df)
    history_hours = 27 * 24
    batch_idx = 0
    t_start = warmup_hours

    while t_start < total_steps - batch_size_steps:
        t_end_obs = min(t_start + batch_size_steps, total_steps)
        history_start = max(0, t_start - history_hours)
        n_steps = t_end_obs - t_start
        speed_hourly = (
            v_obs_df['Speed'].iloc[t_start:t_end_obs].values.copy()
            if 'Speed' in v_obs_df.columns
            else np.full(n_steps, np.nan, dtype=float)
        )
        lon_deg_hourly = (
            v_obs_df['HC_lon'].iloc[t_start:t_end_obs].values.copy()
            if 'HC_lon' in v_obs_df.columns
            else np.full(n_steps, np.nan, dtype=float)
        )
        lat_deg_hourly = (
            v_obs_df['HC_lat'].iloc[t_start:t_end_obs].values.copy()
            if 'HC_lat' in v_obs_df.columns
            else np.full(n_steps, np.nan, dtype=float)
        )
        radius_hourly = (
            v_obs_df['HC_radius'].iloc[t_start:t_end_obs].values.copy()
            if 'HC_radius' in v_obs_df.columns
            else np.full(n_steps, np.nan, dtype=float)
        )
        history_speed_hourly = (
            v_obs_df['Speed'].iloc[history_start:t_start].values.copy()
            if 'Speed' in v_obs_df.columns
            else np.full(max(0, t_start - history_start), np.nan, dtype=float)
        )
        specs.append(
            {
                'batch_idx': batch_idx,
                't_start': t_start,
                't_end_obs': t_end_obs,
                'is_final': False,
                'batch_start_time': str(v_obs_df['Time'].iloc[t_start]),
                'batch_end_time': str(v_obs_df['Time'].iloc[t_end_obs - 1]),
                'speed_hourly': speed_hourly,
                'lon_deg_hourly': lon_deg_hourly,
                'lat_deg_hourly': lat_deg_hourly,
                'radius_hourly': radius_hourly,
                'time_hourly': v_obs_df['Time'].iloc[t_start:t_end_obs].values.copy(),
                'history_speed_hourly': history_speed_hourly,
                'image_slice': _slice_image_batch(
                    all_image_slices,
                    v_obs_df['Time'].iloc[t_start],
                    model_nt,
                    model_dt_seconds,
                ) if all_image_slices is not None else None,
            }
        )
        t_start += stride_steps
        batch_idx += 1

    if t_start < total_steps:
        history_start = max(0, t_start - history_hours)
        n_steps = total_steps - t_start
        speed_hourly = (
            v_obs_df['Speed'].iloc[t_start:total_steps].values.copy()
            if 'Speed' in v_obs_df.columns
            else np.full(n_steps, np.nan, dtype=float)
        )
        lon_deg_hourly = (
            v_obs_df['HC_lon'].iloc[t_start:total_steps].values.copy()
            if 'HC_lon' in v_obs_df.columns
            else np.full(n_steps, np.nan, dtype=float)
        )
        lat_deg_hourly = (
            v_obs_df['HC_lat'].iloc[t_start:total_steps].values.copy()
            if 'HC_lat' in v_obs_df.columns
            else np.full(n_steps, np.nan, dtype=float)
        )
        radius_hourly = (
            v_obs_df['HC_radius'].iloc[t_start:total_steps].values.copy()
            if 'HC_radius' in v_obs_df.columns
            else np.full(n_steps, np.nan, dtype=float)
        )
        history_speed_hourly = (
            v_obs_df['Speed'].iloc[history_start:t_start].values.copy()
            if 'Speed' in v_obs_df.columns
            else np.full(max(0, t_start - history_start), np.nan, dtype=float)
        )
        specs.append(
            {
                'batch_idx': batch_idx,
                't_start': t_start,
                't_end_obs': total_steps,
                'is_final': True,
                'batch_start_time': str(v_obs_df['Time'].iloc[t_start]),
                'batch_end_time': str(v_obs_df['Time'].iloc[total_steps - 1]),
                'speed_hourly': speed_hourly,
                'lon_deg_hourly': lon_deg_hourly,
                'lat_deg_hourly': lat_deg_hourly,
                'radius_hourly': radius_hourly,
                'time_hourly': v_obs_df['Time'].iloc[t_start:total_steps].values.copy(),
                'history_speed_hourly': history_speed_hourly,
                'image_slice': _slice_image_batch(
                    all_image_slices,
                    v_obs_df['Time'].iloc[t_start],
                    model_nt,
                    model_dt_seconds,
                ) if all_image_slices is not None else None,
            }
        )

    return specs


def _build_obs_tracks_for_batch(v_obs_raw_df, batch_start_time, batch_end_time, config, device):
    """Build per-spacecraft observation tracks for one batch window (unified format)."""
    max_interp_gap_hours = int(max(0, getattr(config, 'max_interp_gap_hours', 5)))

    # ... (Keep your existing _fill_short_nan_gaps helper function here exactly as it is) ...
    def _fill_short_nan_gaps(values, max_gap, circular_degrees=False):
        out = np.asarray(values, dtype=float).copy()
        if max_gap <= 0: return out
        valid_idx = np.where(np.isfinite(out))[0]
        if valid_idx.size < 2: return out
        for left, right in zip(valid_idx[:-1], valid_idx[1:]):
            gap = right - left - 1
            if gap <= 0 or gap > max_gap: continue
            frac = np.arange(1, gap + 1, dtype=float) / float(gap + 1)
            left_val = out[left]
            right_val = out[right]
            if circular_degrees:
                delta = ((right_val - left_val + 180.0) % 360.0) - 180.0
                out[left + 1:right] = np.mod(left_val + frac * delta, 360.0)
            else:
                out[left + 1:right] = left_val + frac * (right_val - left_val)
        return out

    window = v_obs_raw_df[
        (v_obs_raw_df['Time'] >= batch_start_time)
        & (v_obs_raw_df['Time'] < batch_end_time)
    ].copy()
    if window.empty:
        return None

    # MAGIC TRICK: If no Spacecraft_ID exists, fake one so the loop still works natively
    if 'Spacecraft_ID' not in window.columns:
        window['Spacecraft_ID'] = 'PRIMARY'

    tracks = []
    grouped_tracks = list(window.groupby('Spacecraft_ID'))
    # Sort to ensure ACE (or PRIMARY) is always first
    grouped_tracks.sort(key=lambda item: (str(item[0]).strip().upper() not in ['ACE', 'PRIMARY'], str(item[0]).strip().upper()))

    for spacecraft_id, sc_df in grouped_tracks:
        sc_df = sc_df.sort_values('Time')
        # Apply standard single-spacecraft fallbacks if columns are missing
        if 'HC_lat' not in sc_df.columns:
            sc_df['HC_lat'] = 0.0
        if 'HC_radius' not in sc_df.columns:
            sc_df['HC_radius'] = float(getattr(config, 'rAU', 215.0))
            
        sc_cols = ['Time', 'Speed', 'HC_lon', 'HC_lat', 'HC_radius']
        sc_df = sc_df[sc_cols].dropna(subset=['Time', 'HC_lon'])
        if sc_df.empty:
            continue

        sc_hourly = sc_df.set_index('Time').resample('1h').mean(numeric_only=True)

        full_hour_index = pd.date_range(
            start=pd.to_datetime(batch_start_time),
            end=pd.to_datetime(batch_end_time) - pd.Timedelta(hours=1),
            freq='1h',
        )
        sc_hourly = sc_hourly.reindex(full_hour_index).rename_axis('Time').reset_index()
        if sc_hourly.empty: continue

        # Apply interpolations
        lon_deg = np.asarray(sc_hourly['HC_lon'].values, dtype=float)
        sc_hourly['HC_lon'] = _fill_short_nan_gaps(lon_deg, max_interp_gap_hours, circular_degrees=True)
        speed_vals = np.asarray(sc_hourly['Speed'].values, dtype=float)
        # speed may use a different limit (0 = score measured hours only; spacecraft position is still interpolated)
        sc_hourly['Speed'] = _fill_short_nan_gaps(
            speed_vals, int(getattr(config, 'max_interp_gap_hours_speed', max_interp_gap_hours)))
        lat_vals = np.asarray(sc_hourly['HC_lat'].values, dtype=float)
        sc_hourly['HC_lat'] = _fill_short_nan_gaps(lat_vals, max_interp_gap_hours)
        radius_vals = np.asarray(sc_hourly['HC_radius'].values, dtype=float)
        sc_hourly['HC_radius'] = _fill_short_nan_gaps(radius_vals, max_interp_gap_hours)

        speed_hourly = np.asarray(sc_hourly['Speed'].values, dtype=float)
        speed_hourly = np.where((speed_hourly == 9999.0) | (speed_hourly < 0.0), np.nan, speed_hourly)

        lon_deg_hourly = np.asarray(sc_hourly['HC_lon'].values, dtype=float)
        lon_rad_hourly = np.mod(np.deg2rad(lon_deg_hourly), 2.0 * np.pi)

        radius_hourly = np.asarray(sc_hourly['HC_radius'].values, dtype=float)
        finite_radius = np.isfinite(radius_hourly)
        rAU = float(getattr(config, 'rAU', 215.0))
        if np.any(finite_radius):
            if float(np.nanmedian(radius_hourly[finite_radius])) < 5.0:
                radius_hourly = radius_hourly * rAU
        else:
            radius_hourly = np.full(speed_hourly.shape, rAU, dtype=float)

        t_hourly = pd.to_datetime(sc_hourly['Time'].values)
        time_obs_seconds = (t_hourly - pd.to_datetime(batch_start_time)).total_seconds().astype(float)

        v_obs_track = torch.tensor(speed_hourly, dtype=torch.float32, device=device)
        phi_obs_track = torch.tensor(lon_rad_hourly, dtype=torch.float32, device=device)
        radius_obs_track = torch.tensor(radius_hourly, dtype=torch.float32, device=device)
        time_obs_track = torch.tensor(time_obs_seconds, dtype=torch.float32, device=device)

        track_entry = {
            'id': str(spacecraft_id),
            'v_obs': v_obs_track,
            'phi_obs_t': phi_obs_track,
            'radius_obs_t': radius_obs_track,
            'time_obs_seconds': time_obs_seconds.copy(),
            'time_hourly': t_hourly.to_numpy(copy=True),
            'lon_deg_hourly': lon_deg_hourly.copy(),
            'lat_deg_hourly': sc_hourly['HC_lat'].values.copy(),
            'speed_hourly': speed_hourly.copy(),
            'radius_hourly': radius_hourly.copy(),
        }
        tracks.append(track_entry)

    return tracks if tracks else None


def _load_all_image_data(config):
    """
    Load AIA/HMI image slices and compute global normalization statistics.
    """

    all_image_slices = {}

    aia_data = load_image_slices(
        config,
        instrument="aia",
    )

    if aia_data is None:
        raise RuntimeError(
            f"AIA data could not be loaded from {config.aia_data_path}"
        )

    all_image_slices.update(aia_data)

    hmi_data = load_image_slices(
        config,
        instrument="hmi",
    )

    if hmi_data is not None:
        if not np.array_equal(
            all_image_slices["times"],
            hmi_data["times"],
        ):
            raise RuntimeError(
                "AIA/HMI timelines are not identical"
            )
        all_image_slices.update(hmi_data)
        

    aia = all_image_slices["aia_slices"]
    mask = all_image_slices["visibility_mask"] > 0.5

    visible_vals = aia[mask]

    global_mean = np.nanmean(visible_vals)
    global_std = np.nanstd(visible_vals)
    asinh_clip_dn = np.nanpercentile(visible_vals, 99)

    print(f"AIA global mean={global_mean}")
    print(f"AIA global std={global_std}")
    print(f"AIA asinh clip (p99)={asinh_clip_dn}")

    config.aia_global_mean = float(global_mean)
    config.aia_global_std = float(global_std)
    config.aia_asinh_clip_dn = float(asinh_clip_dn)

    if "hmi_slices" in all_image_slices:

        hmi = all_image_slices["hmi_slices"]

        hmi_visible = hmi[mask]

        hmi_mean = np.nanmean(hmi_visible)
        hmi_std = np.nanstd(hmi_visible)

        print(f"HMI global mean={hmi_mean}")
        print(f"HMI global std={hmi_std}")

        config.hmi_global_mean = float(hmi_mean)
        config.hmi_global_std = float(max(hmi_std, 1e-6))

    else:

        # Allow AIA-only mode
        config.hmi_global_mean = 0.0
        config.hmi_global_std = 1.0

    return all_image_slices    


def _prepare_batch_specs(config, observations_df, all_image_slices):
    """
    Compute batch geometry and construct batch specifications.
    """

    # Convert model temporal window to observation steps (ACE is hourly)
    model_window_hours = config.Nt * config.dt / 3600.0
    model_window_obs_steps = int(np.ceil(model_window_hours)) + 1

    # Each batch covers one model-window worth of ACE observations
    batch_size_steps = model_window_obs_steps  # in hourly ACE indices

    # Overlap: consecutive batches share this many hours so that the
    # unconstrained BC tail (transit time ~94h) gets constrained next batch
    overlap_steps = int(getattr(config, 'overlap_hours', 0.0))
    stride_steps = max(1, batch_size_steps - overlap_steps)

    # Overlap converted to model time-steps (for warm-starting BC and IC)
    overlap_model_steps = int(overlap_steps * 3600.0 / config.dt)
    
    # Log actual batch configuration
    print(f"\nBatch Configuration:")
    print(f"  Model temporal window: {config.Nt} timesteps × {config.dt/3600:.2f} hours = {model_window_hours:.1f} hours")
    print(f"  Observation steps per batch: {batch_size_steps} (hourly ACE samples)")
    print(f"  Overlap: {overlap_steps} hours ({overlap_model_steps} model steps)")
    print(f"  Stride: {stride_steps} hours")
    
    # Ensure we have enough data for at least one batch
    total_steps = len(observations_df)
    if total_steps < batch_size_steps:
        print(f"Warning: Total data points ({total_steps}) < batch size ({batch_size_steps})")
        batch_size_steps = total_steps
        stride_steps = total_steps
    
    warmup_hours = 0
    batch_specs = _build_batch_specs(
        observations_df,
        batch_size_steps,
        stride_steps,
        warmup_hours,
        all_image_slices=all_image_slices,
        model_nt=config.Nt,
        model_dt_seconds=config.dt,
    )
    return (
        batch_specs,
        batch_size_steps,
        overlap_steps,
        overlap_model_steps,
        stride_steps,
        total_steps,
        warmup_hours,
    )


def batch_inverse_solver(config, v_obs_df, output_dir="./results"):
    """
    Solve inverse problem in sequential batches across the entire time series.
    
    Args:
        config: Configuration object
        v_obs_df: pandas DataFrame with columns ['Speed', 'HC_lon', 'HC_lat', 'HC_radius', 'Time']
        output_dir: directory to save results
    
    Returns:
        all_solutions: dict with batch results and metadata
    """
    device = config.device
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    v_obs_df = v_obs_df.sort_values('Time').reset_index(drop=True)
    use_multispacecraft = bool(getattr(config, 'enable_multispacecraft_loss', False)) and ('Spacecraft_ID' in v_obs_df.columns)

    if use_multispacecraft:
        v_obs_raw_df = v_obs_df.copy()
        # Build batch windows on the global observation timeline only.
        # Per-spacecraft speed/lon/lat are preserved in v_obs_raw_df and used via obs_tracks.
        unique_obs_times_df = (
            v_obs_raw_df[['Time']]
            .drop_duplicates()
            .sort_values('Time')
            .reset_index(drop=True)
        )
        print(f"Using multi-spacecraft loss with {v_obs_raw_df['Spacecraft_ID'].nunique()} spacecraft")
    else:
        v_obs_raw_df = v_obs_df
        unique_obs_times_df = v_obs_df.copy()
        print("Using single-spacecraft loss (all observations treated as one track)")
        
    

    all_image_slices = _load_all_image_data(config)
    

    (
    batch_specs,
    batch_size_steps,
    overlap_steps,
    overlap_model_steps,
    stride_steps,
    total_steps,
    warmup_hours,
    ) = _prepare_batch_specs(
        config,
        unique_obs_times_df,
        all_image_slices,
    )

    for spec in batch_specs:
        if "obs_tracks" not in spec:
            spec["obs_tracks"] = None
        
    # Setup grid
    r, phi, t, dr, dphi = setup_grid(config)
    
    # Extract spacecraft longitude from first observation
    if "HC_lon" in unique_obs_times_df.columns and len(unique_obs_times_df) > 0:
        phi_obs = np.deg2rad(unique_obs_times_df["HC_lon"].iloc[0])
    else:
        phi_obs = 0.0
        
    
    # Initialize results storage
    all_solutions = {
        'batches': [],
        'config': {
            'input_hours_limit': getattr(config, 'input_hours_limit', None),
            'batch_execution_mode': getattr(config, 'batch_execution_mode', 'sequential'),
            'overlap_hours': float(overlap_steps),
            'batch_size_steps': batch_size_steps,
            'overlap_steps': overlap_steps,
            'stride_steps': stride_steps,
            'total_steps': total_steps,
            'dt': config.dt,
        },
        'phi_obs': phi_obs,
    }

    # Build obs_tracks for every batch upfront (unified format)
    cpu_device = torch.device('cpu')
    for spec in batch_specs:
        start_time = pd.to_datetime(spec['batch_start_time'])
        # The length of speed_hourly represents the number of hourly observations in the batch
        end_time = start_time + pd.Timedelta(hours=len(spec['speed_hourly']))
        
        obs_tracks = _build_obs_tracks_for_batch(
            v_obs_raw_df, 
            start_time, 
            end_time, 
            config, 
            cpu_device
        )
        spec['obs_tracks'] = obs_tracks

    print(f"\n  Warmup: {warmup_hours} hours reserved before first batch")
    
    # Execution Router
    batch_execution_mode = getattr(config, 'batch_execution_mode', 'parallel').lower()
    
    if batch_execution_mode == 'parallel':
        return _run_parallel_batches(config, batch_specs, all_solutions, output_dir, phi_obs)
    else:
        print(f"\nRunning {len(batch_specs)} batches sequentially...")
        for spec in batch_specs:
            result = execute_batch_pipeline(spec, config, config.device, output_dir, phi_obs)
            if result is not None:
                all_solutions['batches'].append(result)
                
        # Ensure sequential results are ordered and summarize
        all_solutions['batches'].sort(key=lambda batch: batch['batch_idx'])
        _save_batch_summary_and_loss_plot(all_solutions, output_dir)
        return all_solutions
    