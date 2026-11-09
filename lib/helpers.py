from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
import pandas as pd

# =========================================================
# STEP 0: SETUP GRID AND PARAMETERS
# =========================================================

def setup_grid(config, print_cfl=True):
    """Initialize grids for the 2D model."""
    device = config.device
    
    # Radial grid [Rs]
    r = torch.linspace(config.r0, float(getattr(config, 'r_outer', config.rAU)), config.Nr, device=device)
    
    # Longitude grid [radians]
    if getattr(config, 'phi_grid_periodic', False):
        phi = torch.arange(config.Nphi, device=device, dtype=torch.float32) * (2 * np.pi / config.Nphi)
    else:   # legacy grid (checkpoints before 2026-10-08)
        phi = torch.linspace(0, 2 * np.pi, config.Nphi, device=device)
    
    # Time grid [seconds]
    t = torch.arange(0, config.Nt, dtype=torch.float32, device=device) * config.dt
    
    # Grid spacing
    dr = r[1] - r[0]
    dphi = phi[1] - phi[0]
    
    # Check CFL condition for stability
    # Convert grid spacing to km: 1 Rs ≈ 695,700 km
    Rs_to_km = 695700.0
    dr_km = dr.item() * Rs_to_km  # Grid spacing in km
    
    # For radial advection: dt <= dr_km / v_max
    # For azimuthal advection: dt <= dphi / Omega
    v_max = 900.0  # km/s (typical max solar wind speed)
    cfl_radial = dr_km / v_max  # Seconds
    cfl_azimuth = dphi / config.Omega  # Seconds
    cfl_limit = min(cfl_radial, cfl_azimuth)
    
    if print_cfl:
        print(f"\n  CFL Analysis:")
        print(f"    Grid spacing: dr={dr.item():.3f} Rs = {dr_km:.2e} km, dphi={dphi:.4f} rad")
        print(f"    Radial CFL:   dt_max = {cfl_radial:.2f} s (v_max={v_max} km/s)")
        print(f"    Azimuthal CFL: dt_max = {cfl_azimuth:.2f} s (Omega={config.Omega:.2e} rad/s)")
        print(f"    Overall limit: dt_max = {cfl_limit:.2f} s")
        print(f"    Current dt:    {config.dt} s")
        
        if config.dt > cfl_limit:
            violation_factor = config.dt / cfl_limit
            print(f"    WARNING: CFL VIOLATED by factor {violation_factor:.1f}x")
            print(f"    Recommend: dt <= {cfl_limit * 0.8:.2f} s (with safety margin 0.8)")
        else:
            safety_margin = cfl_limit / config.dt
            print(f"    Safe with margin {safety_margin:.2f}x")
    
    return r, phi, t, dr, dphi




def load_observations(config, data_path=None):
    """
    Load observations from ACE solar wind data file.
    
    Args:
        config: Configuration object
        data_path: path to ACE_speed_lon_lat_hr.txt file
    
    Returns:
        v_obs: (Nt,) solar wind speed observations
        phi_obs: spacecraft longitude in radians
    """
    device = config.device
    
    if data_path is None:
        # Default path relative to current working directory
        data_path = Path("../LiveWind/data/ACE_speed_lon_lat_hr.txt")
    
    data_path = Path(data_path)
    
    if data_path.exists():
        print(f"Loading observations from {data_path}")
        # Load CSV file
        df = pd.read_csv(data_path)
        
        # Extract solar wind speed (observations)
        v_obs_np = df['Speed'].values[:config.Nt]
        v_obs = torch.tensor(v_obs_np, dtype=torch.float32, device=device)
        
        # Extract spacecraft longitude (mean or first value)
        phi_obs_deg = df['HC_lon'].iloc[0]  # Use first longitude measurement
        phi_obs = np.deg2rad(phi_obs_deg)  # Convert to radians
        
        print(f"Loaded {len(v_obs)} observations")
        print(f"Speed range: {v_obs.min():.1f} - {v_obs.max():.1f} km/s")
        print(f"Spacecraft longitude: {np.rad2deg(phi_obs):.1f}°")
    else:
        print(f"Warning: Data file not found at {data_path}. Using synthetic data.")
        # Generate synthetic observations as fallback
        v_obs = 400.0 + 50.0 * torch.sin(
            torch.linspace(0, 4 * np.pi, config.Nt, device=device)
        )
        phi_obs = 0.0
    
    return v_obs, phi_obs


def load_image_slices(config, instrument="aia"):
    """
        Load full-length AIA equatorial slices and visibility metadata.
        Args:
            config: Configuration object with aia_data_path

        Returns a dict with:
            - 'image_slices': float32 array of shape (Nphi, Nt_full)
            - 'visibility_mask': float32 array of shape (Nphi, Nt_full)
            - 'times': pandas datetime index or None
        The preprocessed file may be saved as .npy or .npz.
    """
    image_data_path = getattr(config, f"{instrument}_data_path", None)

    if image_data_path is None:
        return None

    image_path_cfg = Path(image_data_path)
    candidate_paths = []
    if image_path_cfg.is_absolute():
        candidate_paths.append(image_path_cfg)
    else:
        candidate_paths.append(Path.cwd() / image_path_cfg)
        candidate_paths.append(Path(__file__).resolve().parents[1] / image_path_cfg)   # repository root

    image_path = None
    for candidate in candidate_paths:
        if candidate.exists():
            image_path = candidate.resolve()
            break

    if image_path is None:
        tried = ", ".join(str(p) for p in candidate_paths)
        print(f"WARNING: {instrument.upper()} data file not found. Config path='{image_path_cfg}', tried: {tried}")
        return None

    source_longitude = None
    visibility_mask = None
    times = None
    image_data = np.load(image_path)
    if isinstance(image_data, np.lib.npyio.NpzFile):
        npz_data = image_data
        if f'{instrument}_slices' in npz_data:
            image_data = npz_data[f'{instrument}_slices']
        else:
            key = npz_data.files[0]
            image_data = npz_data[key]
        if 'carrington_longitude' in npz_data:
            source_longitude = np.asarray(npz_data['carrington_longitude'])
        if 'visibility_mask' in npz_data:
            visibility_mask = np.asarray(npz_data['visibility_mask'], dtype=np.float32)
        if 'times' in npz_data:
            times = pd.to_datetime(np.asarray(npz_data['times']))

    image_data = np.asarray(image_data, dtype=np.float32)
    if image_data.ndim == 3 and image_data.shape[0] == 1:
        image_data = image_data[0]
    if image_data.ndim == 3:
        raise ValueError(
            f"Unsupported {instrument.upper()} data shape {image_data.shape}: expected 2D or 3D channel-first with one channel"
        )

    if image_data.ndim != 2:
        raise ValueError(f"{instrument.upper()} data must be 2D with shape (Nphi, Nt); got {image_data.shape}")

    if visibility_mask is None:
        print(
            f"WARNING: {instrument.upper()} file {image_path} has no visibility_mask; invisible HC_lon cannot be distinguished reliably. Re-run {instrument.upper()} preprocessing to generate frontside masking."
        )
        visibility_mask = np.isfinite(image_data).astype(np.float32)
    else:
        visibility_mask = np.asarray(visibility_mask, dtype=np.float32)

    if visibility_mask.shape != image_data.shape:
        raise ValueError(
            f"{instrument.upper()} visibility mask shape {visibility_mask.shape} does not match {instrument.upper()} data {image_data.shape}"
        )

    if image_data.shape[0] == config.Nt and image_data.shape[1] == config.Nphi:
        image_data = image_data.T
        visibility_mask = visibility_mask.T

    if image_data.shape[0] != config.Nphi:
        source_nphi = image_data.shape[0]

        # Periodic interpolation along longitude so preprocessed AIA grids (e.g. 360)
        # can be consumed by solver grids with different Nphi (e.g. 128).
        source_phi = None
        if source_longitude is not None and source_longitude.size == source_nphi:
            source_phi = np.asarray(source_longitude, dtype=np.float64).reshape(-1)
            if np.nanmax(np.abs(source_phi)) > 2.0 * np.pi + 1e-3:
                source_phi = np.deg2rad(source_phi)
            source_phi = np.mod(source_phi, 2.0 * np.pi)
            order = np.argsort(source_phi)
            source_phi = source_phi[order]
            image_data = image_data[order, :]
            visibility_mask = visibility_mask[order, :]

            source_phi, unique_idx = np.unique(source_phi, return_index=True)
            image_data = image_data[unique_idx, :]
            visibility_mask = visibility_mask[unique_idx, :]
        else:
            source_phi = np.linspace(0.0, 2.0 * np.pi, source_nphi, endpoint=False, dtype=np.float64)

        target_phi = np.linspace(0.0, 2.0 * np.pi, config.Nphi, endpoint=False, dtype=np.float64)

        # Close periodic seam for interpolation at 2pi.
        source_phi_ext = np.concatenate([source_phi, source_phi[:1] + 2.0 * np.pi])
        image_filled = np.nan_to_num(image_data, nan=0.0)
        image_ext = np.concatenate([image_filled, image_filled[:1, :]], axis=0)
        mask_ext = np.concatenate([visibility_mask, visibility_mask[:1, :]], axis=0)

        resampled = np.empty((config.Nphi, image_data.shape[1]), dtype=np.float32)
        resampled_mask = np.empty((config.Nphi, image_data.shape[1]), dtype=np.float32)
        for t_idx in range(image_data.shape[1]):
            interp_mask = np.interp(target_phi, source_phi_ext, mask_ext[:, t_idx]).astype(np.float32)
            interp_val = np.interp(target_phi, source_phi_ext, image_ext[:, t_idx]).astype(np.float32)
            resampled_mask[:, t_idx] = np.clip(interp_mask, 0.0, 1.0)

            out_col = np.full(config.Nphi, np.nan, dtype=np.float32)
            visible = interp_mask > 0.5
            out_col[visible] = interp_val[visible] / np.maximum(interp_mask[visible], 1e-6)
            resampled[:, t_idx] = out_col

        print(
            f"Resampled {instrument.upper()} longitude grid from Nphi={source_nphi} to Nphi={config.Nphi}"
        )
        image_data = resampled
        visibility_mask = resampled_mask

    image_data = np.where(visibility_mask > 0.5, image_data, np.nan)

    result = {
        f'{instrument.lower()}_slices': image_data,
        'times': times,
        'visibility_mask': visibility_mask,
    }
    
    return result


def sample_satellite_track(v_full, r, phi_grid, phi_obs, radius_obs, time_obs=None, dt=None):
    """
    Sample model speed along a satellite track using time-varying radius and longitude.

    Args:
        v_full: (Nr, Nphi, Nt) full propagated speed cube
        r: (Nr,) radial grid [Rs], increasing outward
        phi_grid: (Nphi,) longitude grid [rad]
        phi_obs: scalar or (Nt,) satellite longitude in Carrington frame [rad]
        radius_obs: scalar or (Nt,) satellite radius [Rs]

    Returns:
        v_pred: (Nobs,) predicted speed sampled along the satellite track
    """
    Nt_model = v_full.shape[2]
    Nr = v_full.shape[0]
    Nphi = v_full.shape[1]

    if time_obs is None:
        n_obs = Nt_model
        time_coord = torch.arange(Nt_model, dtype=torch.float32, device=v_full.device)
    else:
        if dt is None:
            raise ValueError("sample_satellite_track requires dt when time_obs is provided")
        if not isinstance(time_obs, torch.Tensor):
            time_obs = torch.tensor(time_obs, dtype=torch.float32, device=v_full.device)
        else:
            time_obs = time_obs.to(device=v_full.device, dtype=torch.float32)
        if time_obs.dim() == 0:
            time_obs = time_obs.unsqueeze(0)
        n_obs = int(time_obs.shape[0])
        time_coord = time_obs / float(dt)

    if not isinstance(phi_obs, torch.Tensor):
        phi_obs = torch.tensor(phi_obs, dtype=torch.float32, device=v_full.device)
    else:
        phi_obs = phi_obs.to(device=v_full.device, dtype=torch.float32)
    if phi_obs.dim() == 0:
        phi_obs = phi_obs.unsqueeze(0).expand(n_obs)

    if not isinstance(radius_obs, torch.Tensor):
        radius_obs = torch.tensor(radius_obs, dtype=torch.float32, device=v_full.device)
    else:
        radius_obs = radius_obs.to(device=v_full.device, dtype=torch.float32)
    if radius_obs.dim() == 0:
        radius_obs = radius_obs.unsqueeze(0).expand(n_obs)

    if phi_obs.shape[0] != n_obs or radius_obs.shape[0] != n_obs:
        raise ValueError(
            f"sample_satellite_track expects phi_obs and radius_obs length Nobs={n_obs}; "
            f"got {phi_obs.shape[0]} and {radius_obs.shape[0]}"
        )

    valid_mask = torch.isfinite(phi_obs) & torch.isfinite(radius_obs) & torch.isfinite(time_coord)
    valid_mask = valid_mask & (time_coord >= 0.0) & (time_coord <= float(Nt_model - 1))

    t_safe = torch.where(valid_mask, time_coord, torch.zeros_like(time_coord))
    t_idx_lo = torch.floor(t_safe).long()
    t_idx_hi = torch.clamp(t_idx_lo + 1, max=Nt_model - 1)
    t_w_hi = t_safe - t_idx_lo.to(dtype=torch.float32)
    t_w_lo = 1.0 - t_w_hi

    # Interpolate model cube in time to observation timestamps.
    v_time = v_full.permute(2, 0, 1)
    v_t_lo = v_time[t_idx_lo, :, :]
    v_t_hi = v_time[t_idx_hi, :, :]
    v_rt = (t_w_lo.view(n_obs, 1, 1) * v_t_lo) + (t_w_hi.view(n_obs, 1, 1) * v_t_hi)

    radius_safe = torch.where(valid_mask, radius_obs, r[-1])
    radius_safe = torch.clamp(radius_safe, min=float(r[0]), max=float(r[-1]))

    idx_hi = torch.searchsorted(r, radius_safe)
    idx_hi = torch.clamp(idx_hi, min=1, max=Nr - 1)
    idx_lo = idx_hi - 1

    r_lo = r[idx_lo]
    r_hi = r[idx_hi]
    denom = torch.clamp(r_hi - r_lo, min=1e-6)
    w_hi = (radius_safe - r_lo) / denom
    w_lo = 1.0 - w_hi

    lo_idx = idx_lo.view(n_obs, 1, 1).expand(-1, 1, Nphi)
    hi_idx = idx_hi.view(n_obs, 1, 1).expand(-1, 1, Nphi)
    v_lo = torch.gather(v_rt, 1, lo_idx).squeeze(1)
    v_hi = torch.gather(v_rt, 1, hi_idx).squeeze(1)

    v_radius = (w_lo.unsqueeze(1) * v_lo) + (w_hi.unsqueeze(1) * v_hi)
    v_phi_t = v_radius.transpose(0, 1)

    # IMPORTANT: sanitize invalid longitudes before interpolation.
    # Even when losses later mask invalid samples, NaN interpolation weights can
    # still poison gradients via 0*NaN in backprop.
    phi_safe = torch.where(valid_mask, phi_obs, torch.zeros_like(phi_obs))
    v_pred = interpolate_phi(v_phi_t, phi_grid, phi_safe)
    v_pred = torch.where(valid_mask, v_pred, torch.full_like(v_pred, float('nan')))
    return v_pred

def interpolate_phi(v_phi_t, phi_grid, phi_obs):
    """
    Interpolate velocity field to spacecraft longitude using linear interpolation (differentiable).
    
    In the Carrington (co-rotating) frame, the spacecraft drifts:
      φ_spacecraft(t) = φ_obs(0) + Ω·t  (mod 2π)
    
    So at each time step we interpolate at a DIFFERENT longitude.
    
    Args:
        v_phi_t: (Nphi, Nt) - velocity over longitude and time at a sampled radius
        phi_grid: (Nphi,) - longitude grid [0, 2π)
        phi_obs: scalar or (Nt,) tensor - observation longitude(s) in radians
    
    Returns:
        v_pred: (Nt,) - predicted velocity time series at spacecraft
    """
    Nphi = phi_grid.shape[0]
    Nt = v_phi_t.shape[1]
    dphi = phi_grid[1] - phi_grid[0]
    
    # If phi_obs is scalar, broadcast to (Nt,)
    if not isinstance(phi_obs, torch.Tensor):
        phi_obs = torch.tensor(phi_obs, dtype=torch.float32, device=phi_grid.device)
    if phi_obs.dim() == 0:
        phi_obs = phi_obs.unsqueeze(0).expand(Nt)
    
    # Normalize to [0, 2π)
    phi_obs_mod = torch.fmod(phi_obs, 2 * torch.pi)
    phi_obs_mod = torch.where(phi_obs_mod < 0, phi_obs_mod + 2 * torch.pi, phi_obs_mod)
    
    # Fractional index into phi_grid for each time step
    frac_idx = phi_obs_mod / dphi  # (Nt,)
    idx_lo = frac_idx.long() % Nphi        # lower grid index
    idx_hi = (idx_lo + 1) % Nphi            # upper grid index (periodic)
    weight_hi = frac_idx - frac_idx.floor()  # interpolation weight for upper
    weight_lo = 1.0 - weight_hi              # interpolation weight for lower
    
    # Gather values at both indices for all time steps
    # v_phi_t is (Nphi, Nt), need v_phi_t[idx_lo[t], t] and v_phi_t[idx_hi[t], t]
    t_idx = torch.arange(Nt, device=phi_grid.device)
    v_lo = v_phi_t[idx_lo, t_idx]  # (Nt,)
    v_hi = v_phi_t[idx_hi, t_idx]  # (Nt,)
    
    v_pred = weight_lo * v_lo + weight_hi * v_hi  # (Nt,)
    
    return v_pred




def save_boundary_condition_plot(v_inner_np, config, output_path, title):
    """Save v(phi, r=r0, t) as a color plot for diagnostics."""
    phi_deg = np.linspace(0.0, 360.0, config.Nphi, endpoint=False)
    t_hours = np.arange(v_inner_np.shape[1]) * config.dt / 3600.0

    fig, ax = plt.subplots(figsize=(10, 5))
    mesh = ax.pcolormesh(t_hours, phi_deg, v_inner_np, shading='auto', cmap='RdYlBu_r')
    ax.set_xlabel("Time [hours]")
    ax.set_ylabel("Carrington Longitude [deg]")
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, label='v [km/s]')
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _ballistic_backmap_track_times(t_track, speed, radius, config, use_utc):
    """Return track times backmapped from the observation radius to the inner boundary by ballistic travel."""
    if speed is None:
        return t_track

    speed_arr = np.asarray(speed, dtype=float)
    radius_arr = None
    if radius is not None:
        radius_arr = np.asarray(radius, dtype=float)
    if speed_arr.size == 0:
        return t_track

    if speed_arr.ndim == 0:
        speed_arr = np.full((len(t_track),), float(speed_arr), dtype=float)
    elif speed_arr.shape[0] == 1 and len(t_track) > 1:
        speed_arr = np.full((len(t_track),), float(speed_arr[0]), dtype=float)

    if speed_arr.shape[0] != len(t_track):
        return t_track

    if radius_arr is not None:
        if radius_arr.size == 0:
            radius_arr = None
        elif radius_arr.ndim == 0:
            radius_arr = np.full((len(t_track),), float(radius_arr), dtype=float)
        elif radius_arr.shape[0] == 1 and len(t_track) > 1:
            radius_arr = np.full((len(t_track),), float(radius_arr[0]), dtype=float)
        elif radius_arr.shape[0] != len(t_track):
            radius_arr = None

    valid = np.isfinite(speed_arr) & (speed_arr > 0.0)
    if not np.any(valid):
        return t_track

    Rs_to_km = 695700.0
    r0 = float(getattr(config, 'r0', 21.5))
    rAU = float(getattr(config, 'rAU', 215.0))
    r_obs = np.full(speed_arr.shape, rAU, dtype=float)
    if radius_arr is not None:
        radius_valid = np.isfinite(radius_arr) & (radius_arr > r0)
        r_obs[radius_valid] = radius_arr[radius_valid]
    dr = np.maximum(r_obs - r0, 0.0)
    travel_seconds = dr * Rs_to_km / speed_arr

    if use_utc:
        try:
            t_dt = pd.to_datetime(t_track)
            return t_dt - pd.to_timedelta(travel_seconds, unit='s')
        except (TypeError, ValueError):
            return t_track

    t_track = np.asarray(t_track, dtype=float)
    t_shifted = t_track.copy()
    t_shifted[valid] = t_track[valid] - (travel_seconds[valid] / 3600.0)
    t_shifted[~valid] = np.nan
    return t_shifted


def _fill_inner_nans_columnwise(arr):
    """Fill interior NaN gaps in each column (time slice) by linear interpolation.

    For each column, NaN values that sit between two finite values are
    interpolated.  Leading/trailing NaN runs are left untouched so the
    all-NaN tail is preserved.
    """
    out = arr.copy()
    for j in range(out.shape[1]):
        col = out[:, j]
        finite = np.isfinite(col)
        if finite.sum() < 2:
            continue
        idx = np.arange(len(col))
        first_valid = idx[finite][0]
        last_valid = idx[finite][-1]
        interior = (idx >= first_valid) & (idx <= last_valid) & (~finite)
        if interior.any():
            col[interior] = np.interp(idx[interior], idx[finite], col[finite])
    return out


def save_batch_aia_plot(aia_np, config, output_path, batch_start_time=None, title=None):
    """Save AIA input slice for one batch as Time vs HC_lon."""
    if aia_np is None:
        return

    aia_plot = np.asarray(aia_np, dtype=float)
    if aia_plot.ndim == 3:
        if aia_plot.shape[0] < 2:
            raise ValueError(
                f"AIA batch plot expects channel-first shape (C, Nphi, Nt) with C>=2; got {aia_plot.shape}"
            )
        intensity = aia_plot[0]
        visibility = aia_plot[1] > 0.5
        aia_plot = np.where(visibility, intensity, np.nan)
    elif aia_plot.ndim != 2:
        raise ValueError(f"AIA batch plot expects 2D or (C, Nphi, Nt) array; got {aia_plot.shape}")

    phi_deg = np.linspace(0.0, 360.0, aia_plot.shape[0], endpoint=False)

    if batch_start_time is not None:
        t_axis = pd.date_range(
            start=pd.to_datetime(batch_start_time),
            periods=aia_plot.shape[1],
            freq=pd.Timedelta(seconds=config.dt),
        )
        t_plot = mdates.date2num(t_axis)
        use_utc = True
    else:
        t_plot = np.arange(aia_plot.shape[1], dtype=float) * config.dt / 3600.0
        use_utc = False

    fig, ax = plt.subplots(figsize=(11, 4.2))
    mesh = ax.pcolormesh(t_plot, phi_deg, aia_plot, shading='auto', cmap='magma')
    ax.set_ylabel("Carrington Longitude [deg]")
    ax.set_title(title or "AIA input slice")
    fig.colorbar(mesh, ax=ax, label='AIA intensity [arb]')

    if use_utc:
        ax.set_xlabel("UTC")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        fig.autofmt_xdate(rotation=30, ha='right')
    else:
        ax.set_xlabel("Time [hours]")

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_batch_hmi_plot(hmi_np, config, output_path, batch_start_time=None, title=None):
    """Save HMI input slice for one batch as Time vs HC_lon."""
    if hmi_np is None:
        return


    hmi_plot = np.asarray(hmi_np, dtype=float)

    if hmi_plot.ndim == 3:

        if hmi_plot.shape[0] < 3:
            return

        visibility = hmi_plot[1] > 0.5
        hmi_plot = hmi_plot[2]

        hmi_plot = np.where(
            visibility,
            hmi_plot,
            np.nan,
        )

    elif hmi_plot.ndim != 2:

        raise ValueError(
            f"HMI batch plot expects 2D or (C,Nphi,Nt); got {hmi_plot.shape}"
        )

    phi_deg = np.linspace(0.0, 360.0, hmi_plot.shape[0], endpoint=False)

    if batch_start_time is not None:
        t_axis = pd.date_range(
            start=pd.to_datetime(batch_start_time),
            periods=hmi_plot.shape[1],
            freq=pd.Timedelta(seconds=config.dt),
        )
        t_plot = mdates.date2num(t_axis)
        use_utc = True
    else:
        t_plot = np.arange(hmi_plot.shape[1], dtype=float) * config.dt / 3600.0
        use_utc = False

    fig, ax = plt.subplots(figsize=(11, 4.2))
    mesh = ax.pcolormesh(t_plot, phi_deg, hmi_plot, shading='auto', cmap='RdBu_r')
    ax.set_ylabel("Carrington Longitude [deg]")
    ax.set_title(title or "HMI magnetogram slice")
    fig.colorbar(mesh, ax=ax, label='HMI B [Gauss]')

    if use_utc:
        ax.set_xlabel("UTC")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        fig.autofmt_xdate(rotation=30, ha='right')
    else:
        ax.set_xlabel("Time [hours]")

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_batch_solution_plot(v_inner_np, v_obs_np, v_pred_np, config, output_path, title, batch_start_time=None, sensitivity_np=None, corrected_mask=None, extra_track_series=None, track_lon_series=None):
    """Save combined batch solution figure with boundary heatmap and track observed vs predicted.

    sensitivity_np: optional sensitivity heatmap of the same shape as v_inner_np.
    corrected_mask: optional boolean mask of the same shape as v_inner_np marking corrected BC cells.
    """
    phi_deg = np.linspace(0.0, 360.0, config.Nphi, endpoint=False)

    # --- Pre-process BC: fill interior NaN gaps, then find informative time range ---
    v_inner_plot = _fill_inner_nans_columnwise(v_inner_np)

    # Trim trailing all-NaN columns (BC that never reaches 1AU)
    col_all_nan = np.all(np.isnan(v_inner_plot), axis=0)
    non_nan_cols = np.where(~col_all_nan)[0]
    if len(non_nan_cols) > 0:
        bc_last = non_nan_cols[-1] + 1
    else:
        bc_last = v_inner_plot.shape[1]
    v_inner_plot = v_inner_plot[:, :bc_last]

    if sensitivity_np is not None:
        sensitivity_plot = sensitivity_np[:, :bc_last]
        sensitivity_plot = np.nan_to_num(sensitivity_plot, nan=0.0)
    else:
        sensitivity_plot = None

    skip_hours = float(getattr(config, 'overlap_hours', 100.0))
    skip_steps = int(skip_hours * 3600.0 / config.dt)

    if batch_start_time is not None:
        import pandas as pd
        t_axis_full = pd.date_range(start=batch_start_time, periods=v_inner_np.shape[1], freq=pd.Timedelta(seconds=config.dt))
        t_axis_bc = t_axis_full[:bc_last]
        use_utc = True
    else:
        t_axis_full = np.arange(v_inner_np.shape[1]) * config.dt / 3600.0
        t_axis_bc = t_axis_full[:bc_last]
        use_utc = False

    t_axis_bc_plot = mdates.date2num(t_axis_bc) if use_utc else t_axis_bc
    overlap_start = t_axis_full[skip_steps] if skip_steps < len(t_axis_full) else None

    extra_track_series = extra_track_series or []
    track_lon_series = track_lon_series or []
    n_extra = len(extra_track_series)
    max_line_gap_hours = float(getattr(config, 'plot_max_line_gap_hours', 12.0))

    def _plot_with_gaps(ax, t_vals, y_vals, *, label, color, linewidth):
        t_vals = np.asarray(t_vals)
        y_vals = np.asarray(y_vals, dtype=float)
        finite = np.isfinite(y_vals)
        if t_vals.shape[0] != y_vals.shape[0] or not np.any(finite):
            return

        if use_utc:
            t_dt = pd.to_datetime(t_vals, errors='coerce')
            finite &= ~pd.isna(t_dt)
            if not np.any(finite):
                return
            t_dt = t_dt[finite]
            y_plot = y_vals[finite]
            t_num = mdates.date2num(t_dt)
            max_gap_days = max_line_gap_hours / 24.0
            gap_idx = np.where(np.diff(t_num) > max_gap_days)[0]
        else:
            t_num = np.asarray(t_vals, dtype=float)
            finite &= np.isfinite(t_num)
            if not np.any(finite):
                return
            t_num = t_num[finite]
            y_plot = y_vals[finite]
            gap_idx = np.where(np.diff(t_num) > max_line_gap_hours)[0]

        starts = np.concatenate(([0], gap_idx + 1))
        ends = np.concatenate((gap_idx + 1, [len(t_num)]))

        first = True
        for s, e in zip(starts, ends):
            if e - s <= 1:
                continue
            ax.plot(
                t_num[s:e],
                y_plot[s:e],
                label=label if first else '_nolegend_',
                linewidth=linewidth,
                color=color,
            )
            first = False

    n_rows = 2 + (1 if sensitivity_plot is not None else 0) + n_extra
    height_ratios = [2.0]
    if sensitivity_plot is not None:
        height_ratios.append(2.0)
    height_ratios.append(1.2)
    for _ in range(n_extra):
        height_ratios.append(1.1)

    fig_h = 8.0 + 2.0 * n_extra + (3.0 if sensitivity_plot is not None else 0.0)
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(11, fig_h),
        gridspec_kw={"height_ratios": height_ratios},
    )
    if n_rows == 1:
        axes = [axes]

    mesh = axes[0].pcolormesh(t_axis_bc_plot, phi_deg, v_inner_plot, shading='auto', cmap='RdYlBu_r')
    axes[0].set_ylabel("Carrington Longitude [deg]")
    axes[0].set_title(title)
    fig.colorbar(mesh, ax=axes[0], label='v [km/s]')

    # Panel 1 diagnostic: overlay spacecraft longitude tracks to verify sampling geometry.
    for track in track_lon_series:
        t_track = np.asarray(track.get('time', []))
        lon_track = np.asarray(track.get('lon_deg', []), dtype=float)
        speed_track = np.asarray(track.get('speed', []), dtype=float)
        if t_track.size == 0 or lon_track.size == 0:
            continue
        if speed_track.size > 0:
            n = min(t_track.shape[0], lon_track.shape[0], speed_track.shape[0], bc_last)
        else:
            n = min(t_track.shape[0], lon_track.shape[0], bc_last)
        t_track = t_track[:n]
        lon_track = lon_track[:n]
        if speed_track.size > 0:
            speed_track = speed_track[:n]
        if use_utc and np.issubdtype(t_track.dtype, np.number):
            t_track = pd.to_datetime(batch_start_time) + pd.to_timedelta(t_track, unit='h')
        if use_utc:
            t_plot = mdates.date2num(pd.to_datetime(t_track))
        else:
            t_plot = np.asarray(t_track, dtype=float)
        valid = np.isfinite(lon_track)
        if speed_track.size > 0:
            valid &= np.isfinite(speed_track)
        if np.any(valid):
            axes[0].plot(
                t_plot[valid],
                lon_track[valid],
                linewidth=1.2,
                alpha=0.85,
                linestyle='--',
                label=str(track.get('id', 'TRACK')),
            )
    if track_lon_series:
        axes[0].legend(loc='upper right', fontsize=8)

    if use_utc:
        axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        axes[0].xaxis.set_major_locator(mdates.DayLocator(interval=2))

    bottom_axes = axes[1:]

    if sensitivity_plot is not None:
        if use_utc:
            bottom_axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            bottom_axes[0].xaxis.set_major_locator(mdates.DayLocator(interval=2))

        abs_sens = np.abs(sensitivity_plot)
        if abs_sens.size == 0:
            clip = 1.0
        else:
            clip = float(np.nanpercentile(abs_sens, 90))
            clip = max(clip, 0.1 * float(np.nanmax(abs_sens)))
            if clip == 0:
                clip = 1.0

        mesh_sens = bottom_axes[0].pcolormesh(
            t_axis_bc_plot,
            phi_deg,
            sensitivity_plot,
            shading='auto',
            cmap='seismic',
            vmin=-clip,
            vmax=clip,
        )
        bottom_axes[0].set_ylabel("Carrington Longitude [deg]")
        bottom_axes[0].set_title("Boundary sensitivity")
        fig.colorbar(mesh_sens, ax=bottom_axes[0], label='Sensitivity')

        # Show where each spacecraft constrains the boundary in time/longitude.
        for track in track_lon_series:
            t_track = np.asarray(track.get('time', []))
            lon_track = np.asarray(track.get('lon_deg', []), dtype=float)
            speed_track = np.asarray(track.get('speed', []), dtype=float)
            if t_track.size == 0 or lon_track.size == 0:
                continue

            if speed_track.size > 0:
                n = min(t_track.shape[0], lon_track.shape[0], speed_track.shape[0], bc_last)
            else:
                n = min(t_track.shape[0], lon_track.shape[0], bc_last)

            t_track = t_track[:n]
            lon_track = lon_track[:n]
            if speed_track.size > 0:
                speed_track = speed_track[:n]

            if use_utc and np.issubdtype(t_track.dtype, np.number):
                t_track = pd.to_datetime(batch_start_time) + pd.to_timedelta(t_track, unit='h')

            if use_utc:
                t_plot = mdates.date2num(pd.to_datetime(t_track, errors='coerce'))
            else:
                t_plot = np.asarray(t_track, dtype=float)

            valid = np.isfinite(lon_track)
            if speed_track.size > 0:
                valid &= np.isfinite(speed_track)

            if np.any(valid):
                bottom_axes[0].plot(
                    t_plot[valid],
                    lon_track[valid],
                    linewidth=0.9,
                    alpha=0.8,
                    linestyle='--',
                    label=str(track.get('id', 'TRACK')),
                )

        if track_lon_series:
            bottom_axes[0].legend(loc='upper right', fontsize=7)

        if corrected_mask is not None:
            corr_rows, corr_cols = np.nonzero(corrected_mask[:, :bc_last])
            if corr_rows.size > 0:
                x_pts = t_axis_bc_plot[corr_cols]
                y_pts = phi_deg[corr_rows]
                bottom_axes[0].scatter(
                    x_pts,
                    y_pts,
                    s=12,
                    c='black',
                    edgecolors='white',
                    linewidths=0.3,
                    alpha=0.85,
                    marker='o',
                    rasterized=True,
                )
                bottom_axes[0].text(
                    0.02,
                    0.95,
                    "Corrected BC cells marked",
                    transform=bottom_axes[0].transAxes,
                    fontsize=8,
                    color='black',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                )

        plot_axis = bottom_axes[1]
    else:
        plot_axis = bottom_axes[0]
        if corrected_mask is not None:
            corr_rows, corr_cols = np.nonzero(corrected_mask[:, :bc_last])
            if corr_rows.size > 0:
                x_pts = t_axis_bc_plot[corr_cols]
                y_pts = phi_deg[corr_rows]
                axes[0].scatter(
                    x_pts,
                    y_pts,
                    s=12,
                    c='black',
                    edgecolors='white',
                    linewidths=0.3,
                    alpha=0.85,
                    marker='o',
                    rasterized=True,
                )
                axes[0].text(
                    0.02,
                    0.95,
                    "Corrected BC cells marked",
                    transform=axes[0].transAxes,
                    fontsize=8,
                    color='black',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                )
        plot_axis = bottom_axes[0]

    t_obs_full = t_axis_full[:len(v_pred_np)]
    t_obs = t_obs_full[skip_steps:]
    v_pred_trimmed = v_pred_np[skip_steps:]
    v_obs_trimmed = v_obs_np[skip_steps:]

    valid_obs_main = np.isfinite(v_obs_trimmed)
    valid_pred_main = np.isfinite(v_pred_trimmed)
    valid_main = valid_obs_main & valid_pred_main

    if np.any(valid_main):
        rmse_main = float(np.sqrt(np.mean((v_pred_trimmed[valid_main] - v_obs_trimmed[valid_main]) ** 2)))
        mae_main = float(np.mean(np.abs(v_pred_trimmed[valid_main] - v_obs_trimmed[valid_main])))
        main_title = f"ACE track | RMSE={rmse_main:.1f}  MAE={mae_main:.1f} km/s"
    else:
        main_title = "ACE track | RMSE=nan  MAE=nan km/s"

    # Plot predicted as a continuous line over timestamps where observations exist.
    _plot_with_gaps(
        plot_axis,
        t_obs,
        np.where(valid_main, v_pred_trimmed, np.nan),
        label='Predicted track',
        color='tab:blue',
        linewidth=1.8,
    )
    plot_axis.scatter(
        t_obs[valid_obs_main],
        v_obs_trimmed[valid_obs_main],
        label='Observed track',
        s=10,
        color='tab:orange',
        alpha=0.85,
        edgecolors='none',
        rasterized=True,
    )
    if use_utc:
        plot_axis.set_xlabel("UTC")
        plot_axis.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plot_axis.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        fig.autofmt_xdate(rotation=30, ha='right')
    else:
        plot_axis.set_xlabel("Time [hours]")
    plot_axis.set_ylabel("Speed [km/s]")
    plot_axis.set_title(main_title)
    plot_axis.grid(True, alpha=0.3)
    plot_axis.legend(loc='best')

    for idx, track in enumerate(extra_track_series):
        ax_sc = axes[len(axes) - n_extra + idx]
        t_sc = np.asarray(track['time'])
        v_sc_obs = np.asarray(track['v_obs'])
        v_sc_pred = np.asarray(track['v_pred'])
        sc_id = str(track.get('id', f'TRACK_{idx+1}'))

        n_sc = min(t_sc.shape[0], v_sc_obs.shape[0], v_sc_pred.shape[0])
        t_sc = t_sc[:n_sc]
        v_sc_obs = v_sc_obs[:n_sc]
        v_sc_pred = v_sc_pred[:n_sc]

        # Some callers pass hour offsets while the figure is in UTC mode.
        # Convert offsets to absolute UTC timestamps so date locators behave.
        if use_utc and np.issubdtype(t_sc.dtype, np.number):
            t_sc = pd.to_datetime(batch_start_time) + pd.to_timedelta(t_sc, unit='h')

        if overlap_start is not None:
            if use_utc:
                t_sc_dt = pd.to_datetime(t_sc, errors='coerce')
                keep_sc = t_sc_dt >= pd.to_datetime(overlap_start)
            else:
                t_sc_num = np.asarray(t_sc, dtype=float)
                keep_sc = np.isfinite(t_sc_num) & (t_sc_num >= float(overlap_start))
            if not np.any(keep_sc):
                continue
            t_sc = t_sc[keep_sc]
            v_sc_obs = v_sc_obs[keep_sc]
            v_sc_pred = v_sc_pred[keep_sc]

        valid_obs_sc = np.isfinite(v_sc_obs)
        valid_pred_sc = np.isfinite(v_sc_pred)
        valid_sc = valid_obs_sc & valid_pred_sc

        if np.any(valid_sc):
            rmse_sc = float(np.sqrt(np.mean((v_sc_pred[valid_sc] - v_sc_obs[valid_sc]) ** 2)))
            mae_sc = float(np.mean(np.abs(v_sc_pred[valid_sc] - v_sc_obs[valid_sc])))
            sc_title = f'{sc_id} track | RMSE={rmse_sc:.1f}  MAE={mae_sc:.1f} km/s'
        else:
            sc_title = f'{sc_id} track | RMSE=nan  MAE=nan km/s'

        # Plot predicted as a continuous line over valid observed timestamps for this spacecraft.
        _plot_with_gaps(
            ax_sc,
            t_sc,
            np.where(valid_sc, v_sc_pred, np.nan),
            label=f'{sc_id} predicted',
            color='tab:blue',
            linewidth=1.6,
        )
        ax_sc.scatter(
            t_sc[valid_obs_sc],
            v_sc_obs[valid_obs_sc],
            label=f'{sc_id} observed',
            s=10,
            color='tab:orange',
            alpha=0.85,
            edgecolors='none',
            rasterized=True,
        )
        if use_utc:
            ax_sc.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax_sc.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax_sc.set_ylabel('Speed [km/s]')
        ax_sc.set_title(sc_title)
        ax_sc.grid(True, alpha=0.3)
        ax_sc.legend(loc='best')

    if n_extra > 0:
        if use_utc:
            axes[-1].set_xlabel("UTC")
        else:
            axes[-1].set_xlabel("Time [hours]")

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


