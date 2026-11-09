import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# Default debug flag so low-level RHS helpers can be called safely
# before PropagationForwardModel.forward initializes runtime settings.
_DEBUG_NAN = False


def _inverse_softplus(x):
    x = torch.as_tensor(x, dtype=torch.float32)
    return torch.where(x > 20.0, x, torch.log(torch.expm1(x)))

# =========================================================
# STEP 1: BACKWARD PROPAGATION FOR INITIAL GUESS
# =========================================================

def steady_state_propagate(v_at_radius, r, direction='inward', Omega=None):
    """
    Steady-state propagation with longitude remapping.
    
    Physics: Each solar wind parcel travels radially at constant speed v.
    Travel time across one radial cell: dt = dr_km / v.
    During dt, the Sun rotates by Δφ = Ω·dt underneath the parcel,
    so its Carrington longitude shifts by -Ω·dr_km/v (westward drift
    in the co-rotating frame).
    
    Outward (r₀→1AU): parcel at φ at r_i arrives at φ - Ω·dr/v at r_{i+1}
    Inward  (1AU→r₀): parcel at φ at r_{i+1} came from φ + Ω·dr/v at r_i
    
    Different speeds produce different longitude shifts → stream interaction.
    
    Args:
        v_at_radius: (Nphi,) - velocity profile at reference radius [km/s]
        r: (Nr,) - radial grid [solar radii]
        direction: 'inward' (1AU→r0) or 'outward' (r0→1AU)
        Omega: solar rotation rate [rad/s]
    
    Returns:
        v_full: (Nr, Nphi) - velocity at all radii
    """
    if Omega is None:                     # default: the configured frame rate (callers in the pipeline pass it)
        from config import Config
        Omega = Config.Omega
    Rs_to_km = 695700.0
    device = r.device
    Nr = r.shape[0]
    
    # Convert to numpy for interpolation
    v_np = v_at_radius.detach().cpu().numpy().copy()
    r_np = r.detach().cpu().numpy()
    
    Nphi = v_np.shape[0]
    phi_grid = np.linspace(0, 2 * np.pi, Nphi, endpoint=False)
    
    v_full = np.zeros((Nr, Nphi))
    
    if direction == 'inward':
        # Propagate from 1AU (index -1) to r0 (index 0)
        v_full[-1, :] = v_np
        for i in range(Nr - 2, -1, -1):
            dr_km = (r_np[i + 1] - r_np[i]) * Rs_to_km
            v_prev = v_full[i + 1, :]  # velocity at r_{i+1}
            
            # Clamp velocity to avoid division by near-zero
            v_safe = np.clip(v_prev, 150.0, 900.0)
            
            # Each parcel at φ_j at r_{i+1} came from φ_j + Ω·dr_km/v_j at r_i
            # (reverse the westward drift)
            dphi_shift = Omega * dr_km / v_safe  # (Nphi,) radians
            phi_source = (phi_grid + dphi_shift) % (2 * np.pi)
            
            # We have irregular samples: v(phi_source[j], r_i) = v_prev[j]
            # Interpolate onto regular phi_grid at r_i
            sort_idx = np.argsort(phi_source)
            phi_s = phi_source[sort_idx]
            v_s = v_prev[sort_idx]
            # Periodic extension for interpolation across 0/2π boundary
            phi_ext = np.concatenate([phi_s - 2 * np.pi, phi_s, phi_s + 2 * np.pi])
            v_ext = np.concatenate([v_s, v_s, v_s])
            v_full[i, :] = np.interp(phi_grid, phi_ext, v_ext)
    
    elif direction == 'outward':
        # Propagate from r0 (index 0) to 1AU (index -1)
        v_full[0, :] = v_np
        for i in range(1, Nr):
            dr_km = (r_np[i] - r_np[i - 1]) * Rs_to_km
            v_prev = v_full[i - 1, :]  # velocity at r_{i-1}
            
            # Clamp velocity to avoid division by near-zero
            v_safe = np.clip(v_prev, 150.0, 900.0)
            
            # Each parcel at φ_j at r_{i-1} arrives at φ_j - Ω·dr_km/v_j at r_i
            # (westward drift in co-rotating frame)
            dphi_shift = Omega * dr_km / v_safe  # (Nphi,) radians
            phi_arrival = (phi_grid - dphi_shift) % (2 * np.pi)
            
            # We have irregular samples: v(phi_arrival[j], r_i) = v_prev[j]
            # Interpolate onto regular phi_grid at r_i
            sort_idx = np.argsort(phi_arrival)
            phi_s = phi_arrival[sort_idx]
            v_s = v_prev[sort_idx]
            # Periodic extension for interpolation across 0/2π boundary
            phi_ext = np.concatenate([phi_s - 2 * np.pi, phi_s, phi_s + 2 * np.pi])
            v_ext = np.concatenate([v_s, v_s, v_s])
            v_full[i, :] = np.interp(phi_grid, phi_ext, v_ext)
    
    else:
        raise ValueError(f"direction must be 'inward' or 'outward', got {direction}")
    
    return torch.tensor(v_full, dtype=torch.float32, device=device)

# =========================================================
# STEP 3: FINITE-VOLUME MUSCL SCHEME WITH MC LIMITER
# =========================================================

def MC_limiter(r):
    """
    Monotonized Central (MC) limiter: phi(r) = max(0, min(2r, (1+r)/2, 2))
    Provides good balance between diffusion and stability.
    r: slope ratio (can be > 1)
    
    CRITICAL: Clamp r to prevent unbounded growth in limiter output.
    If slope ratio is too large, limit it to max_ratio to prevent NaN.
    """
    # Clamp slope ratio to prevent numerical instability
    # If r > max_ratio, limiter is already saturated anyway
    max_ratio = 100.0  # Reasonable bound for slope ratios
    r_clamped = torch.clamp(r, min=-max_ratio, max=max_ratio)
    
    # MC limiter formula
    term1 = 2.0 * r_clamped
    term2 = 0.5 * (1.0 + r_clamped)
    term3 = 2.0 * torch.ones_like(r_clamped)
    
    # phi = max(0, min(2r, (1+r)/2, 2))
    phi = torch.clamp(torch.minimum(torch.minimum(term1, term2), term3), min=0.0)
    
    # Extra safety: clamp phi output
    phi = torch.clamp(phi, min=0.0, max=2.0)
    
    return phi


def MUSCL_reconstruction(v, axis=-1):
    """
    MUSCL (Monotone Upstream-centered Scheme for Conservation Laws) reconstruction.
    Computes left (L) and right (R) states at cell faces using MC limiter.
    
    For cell i with neighbors i-1, i, i+1:
    - Left face: v_L = v_i - 0.5 * phi(r_i) * (v_{i+1} - v_i)
    - Right face: v_R = v_i + 0.5 * phi(r_i) * (v_{i+1} - v_i)
    
    where r_i = (v_i - v_{i-1}) / (v_{i+1} - v_i + eps)
    
    STABILITY IMPROVEMENTS:
    1. Use relative epsilon: eps = 1e-6 * |v_{i+1} - v_i|
    2. Clamp slope ratio r_i before limiter
    3. Clamp reconstructed states v_L, v_R to reasonable range
    
    Returns: (v_L, v_R) both shape same as v
    """
    # Shift to get neighbors
    v_im = torch.roll(v, shifts=1, dims=axis)    # v_{i-1}
    v_ip = torch.roll(v, shifts=-1, dims=axis)   # v_{i+1}
    
    # Compute slope differences
    dv_forward = v_ip - v   # (v_{i+1} - v_i)
    dv_backward = v - v_im  # (v_i - v_{i-1})
    
    # Use relative epsilon: scale with local gradient magnitude
    dv_max = torch.abs(dv_forward)
    eps_relative = torch.clamp(1e-6 * dv_max, min=1e-8)  # Relative epsilon, but not too small
    
    # Compute slope ratio r_i with robust epsilon
    # Avoid division by zero: if dv_forward is tiny, r_i is clamped anyway
    denom = dv_forward + eps_relative
    r_i = dv_backward / denom
    
    # CRITICAL: Clamp slope ratio to prevent unbounded limiter values
    r_i = torch.clamp(r_i, min=-100.0, max=100.0)
    
    # Compute limiter
    phi = MC_limiter(r_i)
    
    # Compute slopes
    slope = phi * dv_forward
    
    # CRITICAL: Clamp slope to prevent unbounded reconstructed states
    max_slope = 50.0  # Max slope as fraction of velocity scale
    slope = torch.clamp(slope, min=-max_slope, max=max_slope)
    
    # Reconstruct states
    v_L = v - 0.5 * slope  # Left state at right face of cell i
    v_R = v + 0.5 * slope  # Right state at right face of cell i
    
    # CRITICAL: Clamp reconstructed states to prevent huge velocity values
    # Use 150-800 km/s for solar wind
    v_L = torch.clamp(v_L, min=100.0, max=900.0)
    v_R = torch.clamp(v_R, min=100.0, max=900.0)
    
    # Check for NaN/Inf (gated: each .any() is a GPU sync)
    if _DEBUG_NAN:
        if torch.isnan(v_L).any() or torch.isnan(v_R).any() or torch.isinf(v_L).any() or torch.isinf(v_R).any():
            print(f"    WARNING: NaN/Inf in MUSCL_reconstruction!")
            print(f"      v range: [{v.min():.3f}, {v.max():.3f}]")
            print(f"      dv_forward range: [{dv_forward.min():.3f}, {dv_forward.max():.3f}]")
            print(f"      r_i range: [{r_i.min():.3f}, {r_i.max():.3f}]")
            print(f"      phi range: [{phi.min():.3f}, {phi.max():.3f}]")
            print(f"      slope range: [{slope.min():.3f}, {slope.max():.3f}]")
            v_L = torch.where(torch.isnan(v_L) | torch.isinf(v_L), v, v_L)
            v_R = torch.where(torch.isnan(v_R) | torch.isinf(v_R), v, v_R)
    
    return v_L, v_R


def radial_flux(v_L, v_R):
    """
    Compute radial conservative flux: F = (1/2) v^2 (upwind)
    Uses Godunov's method: pick upwind state based on sign of velocity
    
    STABILITY: Clamp face velocity before computing flux to prevent huge fluxes.
    
    Args:
        v_L: left state (velocity from left)
        v_R: right state (velocity from right)
    
    Returns:
        flux at face using upwind Godunov scheme
    """
    # Clamp velocities to prevent unrealistic values before flux computation
    v_L_safe = torch.clamp(v_L, min=100.0, max=900.0)
    v_R_safe = torch.clamp(v_R, min=100.0, max=900.0)
    
    # Godunov upwind face state for Burgers-type conservative flux.
    v_face = torch.where(v_L_safe > 0, v_L_safe, v_R_safe)
    flux = 0.5 * v_face * v_face  # F = +(1/2) v^2 at face
    
    if _DEBUG_NAN:
        if torch.isnan(flux).any() or torch.isinf(flux).any():
            print(f"    WARNING: NaN/Inf in radial_flux!")
            print(f"      v_L range: [{v_L.min():.3f}, {v_L.max():.3f}]")
            print(f"      v_R range: [{v_R.min():.3f}, {v_R.max():.3f}]")
            print(f"      v_face range: [{v_face.min():.3f}, {v_face.max():.3f}]")
            flux = torch.where(torch.isnan(flux) | torch.isinf(flux), torch.zeros_like(flux), flux)
    
    return flux


def azimuthal_flux(v_L, v_R, Omega):
    """
    Upwind flux for azimuthal advection at wave speed -Omega.

    Wave speed is -Omega < 0 (always), so information flows in the
    -phi direction and the upwind state is always v_R (from higher phi).
    """
    v_R_safe = torch.clamp(v_R, min=100.0, max=900.0)

    flux = -Omega * v_R_safe

    if _DEBUG_NAN:
        if torch.isnan(flux).any() or torch.isinf(flux).any():
            print(f"    WARNING: NaN/Inf in azimuthal_flux!")
            flux = torch.where(torch.isnan(flux) | torch.isinf(flux), torch.zeros_like(flux), flux)

    return flux


def apply_radial_advection_FV(v, dr, dt, r_grid=None, learned_radial_accel=None):
    """
    Finite-volume radial advection: dv/dt = -d(v^2/2)/dr
    Using MUSCL reconstruction and Godunov flux splitting.
    
    v shape: (Nr, Nphi)
    Returns: dv/dt shape (Nr, Nphi)
    
    STABILITY: Added NaN detection and recovery throughout.
    """
    Rs_to_km = 695700.0
    dr_km = dr * Rs_to_km  # Convert grid spacing from Rs to km

    # MUSCL reconstruction along radial direction (axis=0)
    v_L, v_R = MUSCL_reconstruction(v, axis=0)
    
    if _DEBUG_NAN:
        if torch.isnan(v_L).any() or torch.isnan(v_R).any():
            print(f"    WARNING: NaN in radial MUSCL reconstruction")
            v_L = torch.clamp(v_L, min=100.0, max=900.0)
            v_R = torch.clamp(v_R, min=100.0, max=900.0)
    
    # Left (i-1/2) and right (i+1/2) face values
    v_L_ip = torch.roll(v_R, shifts=1, dims=0)   # v_R from cell i, evaluated at i-1/2
    v_R_im = v_L                                   # v_L from cell i, evaluated at i-1/2
    
    # Fluxes at faces
    f_ip = radial_flux(v_L, v_R)                  # Flux at i+1/2
    f_im = radial_flux(v_L_ip, v_R_im)            # Flux at i-1/2
    
    if _DEBUG_NAN:
        if torch.isnan(f_ip).any() or torch.isnan(f_im).any() or torch.isinf(f_ip).any() or torch.isinf(f_im).any():
            print(f"    WARNING: NaN/Inf in radial fluxes")
            f_ip = torch.where(torch.isnan(f_ip) | torch.isinf(f_ip), torch.zeros_like(f_ip), f_ip)
            f_im = torch.where(torch.isnan(f_im) | torch.isinf(f_im), torch.zeros_like(f_im), f_im)
    
    # Conservative update: dv/dt = -(f_{i+1/2} - f_{i-1/2}) / dr_km
    dv_dt = -(f_ip - f_im) / dr_km

    if learned_radial_accel is not None and r_grid is not None:
        dv_dt = dv_dt + learned_radial_accel.source_term(r_grid, v)
    
    # Clamp gradient
    dv_dt = torch.clamp(dv_dt, min=-50.0, max=50.0)
    
    if _DEBUG_NAN:
        if torch.isnan(dv_dt).any() or torch.isinf(dv_dt).any():
            print(f"    ERROR: NaN/Inf in dv_dt after radial advection!")
            dv_dt = torch.zeros_like(dv_dt)
    
    return dv_dt


def apply_radial_advection_first_order(v, dr, r_grid=None, learned_radial_accel=None):
    """
    Lower-cost conservative radial advection using first-order upwind/Godunov flux.

    Uses Burgers-consistent flux F = +(1/2) v^2 with outward upwinding.
    This is less accurate than MUSCL reconstruction but substantially cheaper.
    """
    Rs_to_km = 695700.0
    dr_km = dr * Rs_to_km  # Convert grid spacing from Rs to km

    v_safe = torch.clamp(v, min=100.0, max=900.0)

    # Face flux at i+1/2 for positive radial speeds.
    # dims=-2 (not dims=0): identical to dims=0 for the usual (Nr, Nphi)
    # shape, but also correct if v carries a leading batch dim (B, Nr, Nphi)
    # -- the radial axis is always second-from-last.
    f_ip = 0.5 * v_safe * v_safe
    f_im = torch.roll(f_ip, shifts=1, dims=-2)

    dv_dt = -(f_ip - f_im) / dr_km

    if learned_radial_accel is not None and r_grid is not None:
        dv_dt = dv_dt + learned_radial_accel.source_term(r_grid, v)

    dv_dt = torch.clamp(dv_dt, min=-50.0, max=50.0)

    if _DEBUG_NAN:
        if torch.isnan(dv_dt).any() or torch.isinf(dv_dt).any():
            print("    WARNING: NaN/Inf in apply_radial_advection_first_order")
            dv_dt = torch.where(torch.isnan(dv_dt) | torch.isinf(dv_dt), torch.zeros_like(dv_dt), dv_dt)

    return dv_dt


def apply_radial_advection_muscl(v, dr, r_grid=None, learned_radial_accel=None):
    """Second-order MUSCL Godunov radial advection: dv/dt = -d(v^2/2)/dr.

    Same conservative Burgers flux as apply_radial_advection_first_order, but
    with MUSCL (limited linear) reconstruction of the cell face value instead
    of piecewise-constant. Because v is clamped to [100, 900] the wave speed
    is strictly positive, so the Godunov flux at face i+1/2 is just f of the
    upwind (left) state, i.e. the right-face reconstruction of cell i.

    Why this exists: the first-order version's numerical diffusion is
    O(dr) and, on this grid, destroys the boundary's time structure before it
    reaches 1 AU -- measured amplitude retention at 1 AU for a sinusoidal
    boundary was 31% at a 48 h period and 2% at 24 h, matching the analytic
    first-order-upwind prediction exp(-2*pi^2*dr*(1-nu)*L/lambda^2). Solar
    wind stream structure lives in exactly that 0.5-3 day band, so almost
    none of it survived propagation, leaving no gradient signal to train the
    boundary network against. With MUSCL the same test retains 65% at 48 h
    and 30% at 24 h, at essentially no extra cost.
    """
    Rs_to_km = 695700.0
    dr_km = dr * Rs_to_km

    v_safe = torch.clamp(v, min=100.0, max=900.0)

    # Right-face reconstruction of each cell; upwind state for face i+1/2.
    _, v_face = MUSCL_reconstruction(v_safe, axis=-2)
    f_ip = 0.5 * v_face * v_face
    f_im = torch.roll(f_ip, shifts=1, dims=-2)

    dv_dt = -(f_ip - f_im) / dr_km

    if learned_radial_accel is not None and r_grid is not None:
        dv_dt = dv_dt + learned_radial_accel.source_term(r_grid, v)

    dv_dt = torch.clamp(dv_dt, min=-50.0, max=50.0)

    if _DEBUG_NAN:
        if torch.isnan(dv_dt).any() or torch.isinf(dv_dt).any():
            print("    WARNING: NaN/Inf in apply_radial_advection_muscl")
            dv_dt = torch.where(torch.isnan(dv_dt) | torch.isinf(dv_dt), torch.zeros_like(dv_dt), dv_dt)

    return dv_dt


# Autograd intermediates retained per timestep, per (Nr x Nphi) cell, in bytes.
# Calibrated by measurement: 2.91 MB/step at Nr=64, Nphi=128 with the MUSCL
# radial scheme (1.54 MB/step with the old first-order one).
_BYTES_PER_STEP_PER_CELL = {'muscl': 2.91 * 2**20 / (64 * 128),
                            'first_order': 1.54 * 2**20 / (64 * 128)}
# Only checkpoint if the estimate exceeds this fraction of free GPU memory.
_TIME_CHECKPOINT_MEM_FRACTION = 0.6


def _time_checkpoint_chunk(config, Nt, v_inner):
    """Chunk size for gradient checkpointing of the time loop, or 0 to disable.

    The MUSCL radial scheme retains roughly twice as many autograd
    intermediates per timestep as the old first-order one (measured 2.91 vs
    1.54 MB/step at Nr=64/Nphi=128), which pushes a full 638-day batch
    (Nt=30618) from ~46 GB to ~87 GB and OOMs an 80 GB card. Checkpointing
    recomputes each chunk's interior during backward instead of storing it,
    cutting peak memory to roughly (chunk_size x per-step) at the cost of one
    extra forward pass.

    Engages only when it is actually needed: gradients are being taken AND the
    estimated activation memory would take a large fraction of the card. That
    is decided from the estimate rather than a fixed step count, because what
    matters is memory, not Nt -- a run that fits comfortably should not pay
    the recompute penalty.
    """
    chunk = int(getattr(config, 'time_checkpoint_chunk', 256) or 0)
    if chunk <= 0 or not torch.is_grad_enabled() or not v_inner.requires_grad:
        return 0
    if not v_inner.is_cuda:
        return 0

    scheme = getattr(config, 'radial_advection_scheme', 'muscl')
    per_cell = _BYTES_PER_STEP_PER_CELL.get(scheme, _BYTES_PER_STEP_PER_CELL['muscl'])
    n_batch = v_inner.shape[0] if v_inner.dim() == 3 else 1
    est_bytes = per_cell * config.Nr * config.Nphi * Nt * n_batch

    # mem_get_info reports DRIVER-level free memory, which excludes the blocks
    # PyTorch's caching allocator is holding but not currently using. Those are
    # reusable by this process immediately. Counting only driver-free memory
    # makes the gate flip to checkpointing for every batch after the first --
    # the first full-graph forward leaves ~45 GB cached, the driver then
    # reports only ~34 GB free, and a 44 GB estimate fails the test even though
    # ~79 GB is actually usable. Measured cost of that mistake: 3 of every 4
    # training batches paying the recompute penalty for no reason.
    free_bytes, _total = torch.cuda.mem_get_info(v_inner.device)
    reserved = torch.cuda.memory_reserved(v_inner.device)
    allocated = torch.cuda.memory_allocated(v_inner.device)
    free_bytes += max(0, reserved - allocated)
    if est_bytes < _TIME_CHECKPOINT_MEM_FRACTION * free_bytes:
        return 0
    return min(chunk, Nt)


def apply_radial_advection(v, dr, r_grid=None, learned_radial_accel=None, scheme="muscl"):
    """Dispatches the radial advection term. 'muscl' (default, 2nd order) or
    'first_order' (the older, much more diffusive scheme -- kept selectable
    for reproducing earlier runs)."""
    if scheme == "muscl":
        fn = apply_radial_advection_muscl
    elif scheme in ("first_order", "first"):
        fn = apply_radial_advection_first_order
    else:
        raise ValueError(
            f"Unknown radial_advection_scheme {scheme!r}; expected 'muscl' or 'first_order'"
        )
    return fn(v, dr, r_grid=r_grid, learned_radial_accel=learned_radial_accel)


def apply_azimuthal_advection_FV(v, dphi, Omega):
    """
    Legacy MUSCL azimuthal advection path used by rk2_tvd_split.

    This intentionally keeps the pre-Godunov single-state upwind shortcut
    that produced the better empirical behavior in earlier runs.

    v shape: (Nr, Nphi)
    Returns: dv/dt shape (Nr, Nphi)
    """
    v_L, v_R = MUSCL_reconstruction(v, axis=-1)

    if _DEBUG_NAN:
        if torch.isnan(v_L).any() or torch.isnan(v_R).any():
            print(f"    WARNING: NaN in azimuthal MUSCL reconstruction")
            v_L = torch.clamp(v_L, min=100.0, max=900.0)
            v_R = torch.clamp(v_R, min=100.0, max=900.0)

    # Legacy face construction: use the state extrapolated from the cell on the
    # left side of each face, which is the empirically preferred rk2 behavior.
    f_ip = azimuthal_flux(v_R, torch.roll(v_L, -1, dims=-1), Omega)
    f_im = azimuthal_flux(torch.roll(v_R, 1, dims=-1), v_L, Omega)

    if _DEBUG_NAN:
        if torch.isnan(f_ip).any() or torch.isnan(f_im).any() or torch.isinf(f_ip).any() or torch.isinf(f_im).any():
            print(f"    WARNING: NaN/Inf in azimuthal fluxes")
            f_ip = torch.where(torch.isnan(f_ip) | torch.isinf(f_ip), torch.zeros_like(f_ip), f_ip)
            f_im = torch.where(torch.isnan(f_im) | torch.isinf(f_im), torch.zeros_like(f_im), f_im)

    # Conservative update: dv/dt = -(F_{i+1/2} - F_{i-1/2}) / dphi
    dv_dt = -(f_ip - f_im) / dphi

    # Clamp gradient
    dv_dt = torch.clamp(dv_dt, min=-50.0, max=50.0)

    if _DEBUG_NAN:
        if torch.isnan(dv_dt).any() or torch.isinf(dv_dt).any():
            print(f"    ERROR: NaN/Inf in dv_dt after azimuthal advection!")
            dv_dt = torch.zeros_like(dv_dt)

    return dv_dt


# =========================================================
# STEP 4: IMEX RK2 (SSP RK2 / Heun) TIME STEPPING
# =========================================================

def IMEX_RK2_radial(v, dr, dt, r_grid=None, learned_radial_accel=None):
    """
    IMEX RK2 (Implicit-Explicit Runge-Kutta) for radial advection.
    Explicit in space, can be made semi-implicit for stability.
    
    Standard SSP RK2:
    v1 = v + dt * L(v)
    v2 = v + 0.5 * dt * (L(v) + L(v1))
    
    v shape: (Nr, Nphi)
    
    STABILITY: Added clamping and early NaN detection with recovery.
    """
    # Stage 1: forward Euler
    L_v = apply_radial_advection_FV(v, dr, dt, r_grid=r_grid, learned_radial_accel=learned_radial_accel)
    
    if _DEBUG_NAN:
        if torch.isnan(L_v).any() or torch.isinf(L_v).any():
            print(f"    ERROR in IMEX_RK2_radial: NaN/Inf in L(v)")
            print(f"      v range: {v.min():.3f} to {v.max():.3f}")
            print(f"      L_v has NaN: {torch.isnan(L_v).sum().item()} cells")
            L_v = torch.where(torch.isnan(L_v) | torch.isinf(L_v), torch.zeros_like(L_v), L_v)
    
    # Clamp gradient magnitude to prevent blowup
    grad_max = 50.0  # Max gradient
    L_v = torch.clamp(L_v, min=-grad_max, max=grad_max)
    
    v1 = v + dt * L_v
    v1 = torch.clamp(v1, min=100.0, max=900.0)  # Strict clamping
    
    # Stage 2: RK2 correction
    L_v1 = apply_radial_advection_FV(v1, dr, dt, r_grid=r_grid, learned_radial_accel=learned_radial_accel)
    
    if _DEBUG_NAN:
        if torch.isnan(L_v1).any() or torch.isinf(L_v1).any():
            print(f"    ERROR in IMEX_RK2_radial: NaN/Inf in L(v1)")
            L_v1 = torch.where(torch.isnan(L_v1) | torch.isinf(L_v1), torch.zeros_like(L_v1), L_v1)
    
    L_v1 = torch.clamp(L_v1, min=-grad_max, max=grad_max)
    
    v2 = v + 0.5 * dt * (L_v + L_v1)
    v2 = torch.clamp(v2, min=100.0, max=900.0)  # Strict clamping
    
    if _DEBUG_NAN:
        if torch.isnan(v2).any() or torch.isinf(v2).any():
            print(f"    ERROR in IMEX_RK2_radial: NaN/Inf in output v2!")
            v2 = v1
    
    return v2


def IMEX_RK2_azimuthal(v, dphi, Omega, dt):
    """
    IMEX RK2 for azimuthal advection.
    
    STABILITY: Added clamping and early NaN detection with recovery.
    """
    # Stage 1
    L_v = apply_azimuthal_advection_FV(v, dphi, Omega)
    
    if _DEBUG_NAN:
        if torch.isnan(L_v).any() or torch.isinf(L_v).any():
            print(f"    ERROR in IMEX_RK2_azimuthal: NaN/Inf in L(v)")
            L_v = torch.where(torch.isnan(L_v) | torch.isinf(L_v), torch.zeros_like(L_v), L_v)
    
    # Clamp gradient
    grad_max = 50.0
    L_v = torch.clamp(L_v, min=-grad_max, max=grad_max)
    
    v1 = v + dt * L_v
    v1 = torch.clamp(v1, min=100.0, max=900.0)  # Strict clamping
    
    # Stage 2
    L_v1 = apply_azimuthal_advection_FV(v1, dphi, Omega)
    
    if _DEBUG_NAN:
        if torch.isnan(L_v1).any() or torch.isinf(L_v1).any():
            print(f"    ERROR in IMEX_RK2_azimuthal: NaN/Inf in L(v1)")
            L_v1 = torch.where(torch.isnan(L_v1) | torch.isinf(L_v1), torch.zeros_like(L_v1), L_v1)
    
    L_v1 = torch.clamp(L_v1, min=-grad_max, max=grad_max)
    
    v2 = v + 0.5 * dt * (L_v + L_v1)
    v2 = torch.clamp(v2, min=100.0, max=900.0)  # Strict clamping
    
    if _DEBUG_NAN:
        if torch.isnan(v2).any() or torch.isinf(v2).any():
            print(f"    ERROR in IMEX_RK2_azimuthal: NaN/Inf in output v2!")
            v2 = v1
    
    return v2


def dphi_hybrid(w, h):
    """Hybrid phi-derivative used by the fast RK4-log radial propagator."""
    ip1 = torch.roll(w, shifts=-1, dims=-1)
    im1 = torch.roll(w, shifts=1, dims=-1)
    d_c = (ip1 - im1) / (2.0 * h)
    d_f = (ip1 - w) / h
    switch = torch.sigmoid(5.0 * d_f)
    return switch * d_c + (1.0 - switch) * d_f


class LearnedRadialAcceleration(nn.Module):
    """Low-dimensional learnable correction for the radial update."""

    def __init__(self, config):
        super().__init__()
        self.r0 = float(getattr(config, 'r0', 21.5))
        self.learnable = bool(getattr(config, 'learned_radial_accel_learnable', True))

        alpha0 = float(getattr(config, 'learned_radial_accel_alpha', 0.0))
        L0 = float(getattr(config, 'learned_radial_accel_L', 50.0))
        v_inf0 = float(getattr(config, 'learned_radial_accel_v_inf', 450.0))

        if self.learnable:
            self.raw_alpha = nn.Parameter(_inverse_softplus(max(alpha0, 1e-6)))
            self.raw_L = nn.Parameter(_inverse_softplus(max(L0, 1e-6)))
            self.raw_v_inf = nn.Parameter(_inverse_softplus(max(v_inf0, 1e-6)))
        else:
            self.register_buffer('alpha', torch.tensor(alpha0, dtype=torch.float32))
            self.register_buffer('L', torch.tensor(max(L0, 1e-6), dtype=torch.float32))
            self.register_buffer('v_inf', torch.tensor(max(v_inf0, 1e-6), dtype=torch.float32))

    def physical_parameters(self):
        if self.learnable:
            alpha = F.softplus(self.raw_alpha)
            L = F.softplus(self.raw_L)
            v_inf = F.softplus(self.raw_v_inf)
        else:
            alpha = self.alpha
            L = self.L
            v_inf = self.v_inf
        return alpha, L, v_inf

    def forward(self, r_grid, v_state):
        if r_grid is None:
            raise ValueError('r_grid is required for LearnedRadialAcceleration')

        if not isinstance(r_grid, torch.Tensor):
            r_grid = torch.tensor(r_grid, dtype=v_state.dtype, device=v_state.device)
        else:
            r_grid = r_grid.to(device=v_state.device, dtype=v_state.dtype)

        if r_grid.dim() == 1:
            r_grid = r_grid.view(-1, 1)
        if r_grid.shape != v_state.shape:
            r_grid = r_grid.expand_as(v_state)

        alpha, L, v_inf = self.physical_parameters()
        alpha = alpha.to(device=v_state.device, dtype=v_state.dtype)
        L = torch.clamp(L.to(device=v_state.device, dtype=v_state.dtype), min=1e-6)
        v_inf = v_inf.to(device=v_state.device, dtype=v_state.dtype)

        decay = torch.exp(-torch.clamp(r_grid - self.r0, min=0.0) / L)
        return alpha * decay * (v_inf - v_state)

    def source_term(self, r_grid, v_state):
        return self.forward(r_grid, v_state)


class LearnedRadialDeltaV(nn.Module):
    """Low-dimensional HUX-style radial speed-correction source.

    DeltaV(r) = Vref * (1 - exp(-(r-r0)/L))
    S(r, v) = v * d(DeltaV)/dr, with per-Rs -> per-km conversion.
    """

    def __init__(self, config):
        super().__init__()
        self.r0 = float(getattr(config, 'r0', 21.5))
        self.learnable = bool(getattr(config, 'learned_radial_accel_learnable', True))
        self.Rs_to_km = 695700.0

        L0 = float(getattr(config, 'learned_radial_dv_L', 50.0))
        Vref0 = float(getattr(config, 'learned_radial_dv_vref', 400.0))
        self.learn_vref = bool(getattr(config, 'learned_radial_dv_learn_vref', True))

        if self.learnable:
            self.raw_L = nn.Parameter(_inverse_softplus(max(L0, 1e-6)))
            if self.learn_vref:
                self.raw_Vref = nn.Parameter(_inverse_softplus(max(Vref0, 1e-6)))
            else:
                self.register_buffer('Vref', torch.tensor(max(Vref0, 1e-6), dtype=torch.float32))
        else:
            self.register_buffer('L', torch.tensor(max(L0, 1e-6), dtype=torch.float32))
            self.register_buffer('Vref', torch.tensor(max(Vref0, 1e-6), dtype=torch.float32))

    def physical_parameters(self):
        if self.learnable:
            L = F.softplus(self.raw_L)
            if self.learn_vref:
                Vref = F.softplus(self.raw_Vref)
            else:
                Vref = self.Vref
        else:
            L = self.L
            Vref = self.Vref
        return L, Vref

    def ddelta_v_dr_rsun(self, r_grid, v_state):
        if r_grid is None:
            raise ValueError('r_grid is required for LearnedRadialDeltaV')

        if not isinstance(r_grid, torch.Tensor):
            r_grid = torch.tensor(r_grid, dtype=v_state.dtype, device=v_state.device)
        else:
            r_grid = r_grid.to(device=v_state.device, dtype=v_state.dtype)

        if r_grid.dim() == 1:
            r_grid = r_grid.view(-1, 1)
        if r_grid.shape != v_state.shape:
            r_grid = r_grid.expand_as(v_state)

        L, Vref = self.physical_parameters()
        L = torch.clamp(L.to(device=v_state.device, dtype=v_state.dtype), min=1e-6)
        Vref = Vref.to(device=v_state.device, dtype=v_state.dtype)

        decay = torch.exp(-torch.clamp(r_grid - self.r0, min=0.0) / L)
        # km/s per Rs
        return Vref * decay / L

    def source_term(self, r_grid, v_state):
        ddelta_dr_km = self.ddelta_v_dr_rsun(r_grid, v_state) / self.Rs_to_km
        v_safe = torch.clamp(v_state, min=100.0, max=900.0)
        return v_safe * ddelta_dr_km



# =========================================================
# STEP 5: FULL PROPAGATION FORWARD MODEL
# =========================================================

class PropagationForwardModel(nn.Module):
    """Differentiable forward propagation solver."""
    
    def __init__(self, config, r, phi, dr, dphi):
        super().__init__()
        self.config = config
        self.register_buffer('r', r)
        self.register_buffer('phi', phi)
        self.dr = dr
        self.dphi = dphi
        self.learned_radial_accel = None
        if bool(getattr(config, 'use_learned_radial_acceleration', False)):
            source_form = str(getattr(config, 'learned_radial_source_form', 'delta_v')).strip().lower()
            if source_form == 'relaxation':
                self.learned_radial_accel = LearnedRadialAcceleration(config)
            else:
                self.learned_radial_accel = LearnedRadialDeltaV(config)
        self._compiled_rhs = None
        if getattr(config, 'compile_rk2_rhs', False):
            try:
                # Compile only the hot RHS kernel used by rk2_tvd_split.
                # Compiling the full forward path has high startup overhead
                # due to long Python control flow in the time loop.
                self._compiled_rhs = torch.compile(self._rhs_rk2_tvd, mode='reduce-overhead')
            except Exception as e:
                print(f"Warning: torch.compile failed for PropagationForwardModel._rhs_rk2_tvd: {e}")
                self._compiled_rhs = None

    def _rhs_rk2_tvd(self, v_state):
        """RHS used by rk2_tvd_split."""
        Omega = self.config.Omega
        r_grid = self.r.view(-1, 1).expand_as(v_state)
        dv_r = apply_radial_advection(
            v_state,
            self.dr,
            r_grid=r_grid,
            learned_radial_accel=self.learned_radial_accel,
            scheme=getattr(self.config, 'radial_advection_scheme', 'muscl'),
        )
        dv_p = apply_azimuthal_advection_FV(v_state, self.dphi, Omega)
        return torch.clamp(dv_r + dv_p, min=-80.0, max=80.0)

    def _radial_rk4_log_profile(self, v_boundary_phi):
        """RK4 radial sweep for one boundary profile v(r0, phi)."""
        Nr = self.config.Nr
        Omega = self.config.Omega
        Rs_to_km = 695700.0

        v_prof = torch.zeros(Nr, self.config.Nphi, device=v_boundary_phi.device)
        v_prof[0, :] = torch.clamp(v_boundary_phi, min=100.0, max=900.0)

        for i in range(1, Nr):
            dr_km = (self.r[i] - self.r[i - 1]) * Rs_to_km
            vv_prev = v_prof[i - 1, :]

            vv_safe = 150.0 + torch.nn.functional.softplus(vv_prev - 150.0)
            vv_safe = torch.clamp(vv_safe, max=1200.0)

            lnv = torch.log(vv_safe)

            k1 = Omega * dphi_hybrid(lnv, self.dphi)
            a2 = torch.clamp(vv_safe + 0.5 * dr_km * k1, min=1.0)
            k2 = Omega * dphi_hybrid(torch.log(a2), self.dphi)
            a3 = torch.clamp(vv_safe + 0.5 * dr_km * k2, min=1.0)
            k3 = Omega * dphi_hybrid(torch.log(a3), self.dphi)
            a4 = torch.clamp(vv_safe + dr_km * k3, min=1.0)
            k4 = Omega * dphi_hybrid(torch.log(a4), self.dphi)

            vv_next = vv_safe + (dr_km / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            v_prof[i, :] = torch.clamp(vv_next, min=100.0, max=900.0)

        return v_prof

    def _forward_rk4_fast(self, v_inner):
        """
        Quick less-accurate forward propagator (debug mode).

        This propagates each time slice independently in radius using
        RK4 on d(v)/dr = Omega * d(log(v))/dphi.
        It is much cheaper than the full time-dependent FV solver and is
        intended for robustness/debugging checks.
        """
        Nr = self.config.Nr
        Omega = self.config.Omega
        Rs_to_km = 695700.0

        # Batch over time: vv has shape (Nt, Nphi)
        vv = v_inner.transpose(0, 1)
        vv_list = [vv]

        for i in range(1, Nr):
            dr_km = (self.r[i] - self.r[i - 1]) * Rs_to_km
            vv_prev = vv_list[-1]

            vv_safe = 150.0 + torch.nn.functional.softplus(vv_prev - 150.0)
            vv_safe = torch.clamp(vv_safe, max=1200.0)

            lnv = torch.log(vv_safe)

            k1 = Omega * dphi_hybrid(lnv, self.dphi)
            a2 = torch.clamp(vv_safe + 0.5 * dr_km * k1, min=1.0)
            k2 = Omega * dphi_hybrid(torch.log(a2), self.dphi)
            a3 = torch.clamp(vv_safe + 0.5 * dr_km * k2, min=1.0)
            k3 = Omega * dphi_hybrid(torch.log(a3), self.dphi)
            a4 = torch.clamp(vv_safe + dr_km * k3, min=1.0)
            k4 = Omega * dphi_hybrid(torch.log(a4), self.dphi)

            vv_next = vv_safe + (dr_km / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            vv_next = torch.clamp(vv_next, min=100.0, max=900.0)
            vv_list.append(vv_next)

        # Stack to (Nr, Nt, Nphi) then permute -> (Nr, Nphi, Nt)
        v_full = torch.stack(vv_list, dim=0).permute(0, 2, 1)
        return v_full

    def _forward_rk4_time_upwind(self, v_inner, v0):
        """
        Experimental time-dependent propagator.

        Combines:
        - explicit first-order upwind time update in phi,
        - RK4-in-r profile precomputed for all times (vectorized),
        - relaxation blend for stability.
        """
        device = v_inner.device
        Nr, Nphi = self.config.Nr, self.config.Nphi
        Nt = v_inner.shape[1]
        dt = self.config.dt
        Omega = self.config.Omega

        v_full = torch.zeros(Nr, Nphi, Nt, device=device)
        v_full[:, :, 0] = torch.clamp(v0, min=100.0, max=900.0)

        # Precompute RK4 radial targets for every boundary time slice in one pass.
        # This removes Nt repeated Python loops over radius and is much faster.
        v_rad_all = self._forward_rk4_fast(v_inner)

        # Upwind-in-time relaxation factor (smaller = more stable)
        tau = 3.0 * dt
        alpha = min(1.0, dt / max(tau, 1e-8))
        cfl_phi = Omega * dt / self.dphi

        for n in range(1, Nt):
            v_prev = v_full[:, :, n - 1]

            # Explicit first-order upwind phi advection (wave speed = -Omega,
            # so upwind is from the +phi side => forward difference)
            v_ip1 = torch.roll(v_prev, shifts=-1, dims=-1)
            v_adv = torch.clamp(v_prev + cfl_phi * (v_ip1 - v_prev), min=100.0, max=900.0)

            # RK4 radial target at this time (already precomputed)
            v_rad = v_rad_all[:, :, n]

            # Stable time relaxation toward radial target
            v_next = (1.0 - alpha) * v_adv + alpha * v_rad
            v_next[0, :] = torch.clamp(v_inner[:, n], min=100.0, max=900.0)
            v_next = torch.clamp(v_next, min=100.0, max=900.0)

            if _DEBUG_NAN:
                if torch.isnan(v_next).any() or torch.isinf(v_next).any():
                    print(f"  ERROR in rk4_time_upwind at time step {n}: NaN/Inf detected")
                    v_next = torch.where(torch.isnan(v_next) | torch.isinf(v_next), v_prev, v_next)

            v_full[:, :, n] = v_next

        return v_full

    def _forward_rk2_tvd_split(self, v_inner, v0):
        """
        Intermediate scheme between rk4_time_upwind and full fv.

        - Conservative radial Burgers flux with first-order upwind (cheap)
        - FV azimuthal operator (MUSCL) for longitudinal structure
        - Coupled SSPRK2 time integration for stability
        """
        device = v_inner.device
        Nr, Nphi = self.config.Nr, self.config.Nphi
        Nt = v_inner.shape[1]
        dt = self.config.dt
        Omega = self.config.Omega

        if _DEBUG_NAN:
            if torch.isnan(v0).any() or torch.isinf(v0).any():
                print("  ERROR: Initial condition v0 contains NaN/Inf!")
                return v0.unsqueeze(-1).expand(-1, -1, Nt)

        # Collect slices in a list to avoid in-place writes (autograd-safe)
        slices = [torch.clamp(v0, min=100.0, max=900.0)]

        use_compiled_rhs = self._compiled_rhs is not None
        rhs_fn = self._compiled_rhs if use_compiled_rhs else self._rhs_rk2_tvd

        chunk = _time_checkpoint_chunk(self.config, Nt, v_inner)
        if chunk:
            return self._forward_rk2_tvd_split_checkpointed(v_inner, v0, rhs_fn, chunk)

        for n in range(1, Nt):
            v_prev = slices[n - 1]
            if _DEBUG_NAN:
                if torch.isnan(v_prev).any() or torch.isinf(v_prev).any():
                    print(f"  ERROR in rk2_tvd_split at time step {n}: NaN/Inf in v_prev")
                    return torch.stack(slices + [slices[-1]] * (Nt - n), dim=2)

            boundary_row = torch.clamp(v_inner[:, n], min=100.0, max=900.0).unsqueeze(0)

            # SSPRK2 stage 1
            k1 = rhs_fn(v_prev)
            if use_compiled_rhs:
                # Compiled CUDA graph outputs are reused across invocations.
                # Clone to persist k1 for stage-2 use in this time step.
                k1 = k1.clone()
            v1_raw = torch.clamp(v_prev + dt * k1, min=100.0, max=900.0)
            v1 = torch.cat([boundary_row, v1_raw[1:, :]], dim=0)

            # SSPRK2 stage 2
            k2 = rhs_fn(v1)
            if use_compiled_rhs:
                k2 = k2.clone()
            v_next_raw = torch.clamp(v_prev + 0.5 * dt * (k1 + k2), min=100.0, max=900.0)
            v_next = torch.cat([boundary_row, v_next_raw[1:, :]], dim=0)

            if _DEBUG_NAN:
                if torch.isnan(v_next).any() or torch.isinf(v_next).any():
                    print(f"  ERROR in rk2_tvd_split at time step {n}: NaN/Inf detected")
                    v_next = torch.where(torch.isnan(v_next) | torch.isinf(v_next), v_prev, v_next)

            slices.append(v_next)

        return torch.stack(slices, dim=2)

    def _rk2_tvd_chunk(self, v_prev, v_inner_chunk, rhs_fn, batched):
        """Runs one contiguous block of SSPRK2 timesteps; wrapped in a
        gradient checkpoint by the callers, so its interior is recomputed
        during backward rather than retained."""
        dt = self.config.dt
        r_axis = 1 if batched else 0
        outs = []
        v = v_prev
        n_steps = v_inner_chunk.shape[-1]
        for j in range(n_steps):
            if batched:
                boundary_row = torch.clamp(v_inner_chunk[:, :, j], min=100.0, max=900.0).unsqueeze(1)
                interior = lambda x: x[:, 1:, :]
            else:
                boundary_row = torch.clamp(v_inner_chunk[:, j], min=100.0, max=900.0).unsqueeze(0)
                interior = lambda x: x[1:, :]

            k1 = rhs_fn(v)
            v1_raw = torch.clamp(v + dt * k1, min=100.0, max=900.0)
            v1 = torch.cat([boundary_row, interior(v1_raw)], dim=r_axis)

            k2 = rhs_fn(v1)
            v_next_raw = torch.clamp(v + 0.5 * dt * (k1 + k2), min=100.0, max=900.0)
            v = torch.cat([boundary_row, interior(v_next_raw)], dim=r_axis)
            outs.append(v)

        return torch.stack(outs, dim=-1)

    def _forward_rk2_tvd_split_checkpointed(self, v_inner, v0, rhs_fn, chunk, batched=False):
        """Memory-lean equivalent of the rk2_tvd_split time loop.

        Numerically identical to the unchunked version (same ops, same order);
        only the autograd storage strategy differs. See _time_checkpoint_chunk
        for why this is needed with the MUSCL radial scheme on long batches.
        """
        from torch.utils.checkpoint import checkpoint

        Nt = v_inner.shape[-1]
        blocks = [torch.clamp(v0, min=100.0, max=900.0).unsqueeze(-1)]
        v_cur = blocks[0][..., 0]

        n = 1
        while n < Nt:
            end = min(n + chunk, Nt)
            block = checkpoint(
                self._rk2_tvd_chunk,
                v_cur,
                v_inner[..., n:end],
                rhs_fn,
                batched,
                use_reentrant=False,
            )
            blocks.append(block)
            v_cur = block[..., -1]
            n = end

        return torch.cat(blocks, dim=-1)

    def _rhs_rk2_tvd_batched(self, v_state):
        """Batched RHS for rk2_tvd_split. v_state: (B, Nr, Nphi)."""
        Omega = self.config.Omega
        r_grid = self.r.view(1, -1, 1).expand_as(v_state)
        dv_r = apply_radial_advection(
            v_state, self.dr, r_grid=r_grid, learned_radial_accel=self.learned_radial_accel,
            scheme=getattr(self.config, 'radial_advection_scheme', 'muscl'),
        )
        dv_p = apply_azimuthal_advection_FV(v_state, self.dphi, Omega)
        return torch.clamp(dv_r + dv_p, min=-80.0, max=80.0)

    def _forward_rk2_tvd_split_batched(self, v_inner, v0):
        """Batched version of _forward_rk2_tvd_split.

        Added to let a group of independent batches share ONE sequence of
        2880-ish Python-loop iterations instead of one sequence per batch --
        the dominant cost is the loop's Python/kernel-launch overhead, not
        raw FLOPs on a (Nr=50, Nphi=128) grid, so batching the *width* of
        each iteration (not the loop length) is what actually saves wall
        time. This is a separate method (not a shape-branch inside the
        original _forward_rk2_tvd_split) specifically so the existing,
        already-validated single-batch path is never touched.

        Args:
            v_inner: (B, Nphi, Nt) - boundary condition at r0, per batch
            v0: (B, Nr, Nphi) - initial condition at t=0, per batch
        Returns:
            v_full: (B, Nr, Nphi, Nt)
        """
        Nt = v_inner.shape[2]

        slices = [torch.clamp(v0, min=100.0, max=900.0)]
        rhs_fn = self._rhs_rk2_tvd_batched

        chunk = _time_checkpoint_chunk(self.config, Nt, v_inner)
        if chunk:
            return self._forward_rk2_tvd_split_checkpointed(v_inner, v0, rhs_fn, chunk, batched=True)

        for n in range(1, Nt):
            v_prev = slices[n - 1]
            boundary_row = torch.clamp(v_inner[:, :, n], min=100.0, max=900.0).unsqueeze(1)  # (B, 1, Nphi)

            k1 = rhs_fn(v_prev)
            v1_raw = torch.clamp(v_prev + self.config.dt * k1, min=100.0, max=900.0)
            v1 = torch.cat([boundary_row, v1_raw[:, 1:, :]], dim=1)

            k2 = rhs_fn(v1)
            v_next_raw = torch.clamp(v_prev + 0.5 * self.config.dt * (k1 + k2), min=100.0, max=900.0)
            v_next = torch.cat([boundary_row, v_next_raw[:, 1:, :]], dim=1)

            slices.append(v_next)

        return torch.stack(slices, dim=3)

    def forward_batched(self, v_inner, v0):
        """Batched forward pass. Only supports forward_scheme='rk2_tvd_split'
        (the only scheme this pipeline uses) -- raises otherwise rather than
        silently falling back to an unbatched/incorrect path.

        Args:
            v_inner: (B, Nphi, Nt)
            v0: (B, Nr, Nphi)
        Returns:
            v_full: (B, Nr, Nphi, Nt)
        """
        scheme = getattr(self.config, 'forward_scheme', 'fv')
        if scheme != 'rk2_tvd_split':
            raise NotImplementedError(
                f"forward_batched only supports forward_scheme='rk2_tvd_split', got {scheme!r}"
            )
        return self._forward_rk2_tvd_split_batched(v_inner, v0)

    def forward(self, v_inner, v0):
        """
        Propagate velocity from inner boundary through full domain using Strang splitting (2nd order).
        
        Strang splitting: R(dt/2) * P(dt) * R(dt/2)
        where R = radial operator (IMEX RK2), P = phi operator (IMEX RK2)
        
        Args:
            v_inner: (Nphi, Nt) - boundary condition at r0
            v0: (Nr, Nphi) - initial condition at t=0
        
        Returns:
            v_full: (Nr, Nphi, Nt) - velocity field over full domain and time
        """
        global _DEBUG_NAN
        _DEBUG_NAN = bool(getattr(self.config, 'debug_nan_checks', False))

        scheme = getattr(self.config, 'forward_scheme', 'fv')
        if scheme == 'rk4_fast':
            return self._forward_rk4_fast(v_inner)
        if scheme == 'rk4_time_upwind':
            return self._forward_rk4_time_upwind(v_inner, v0)
        if scheme == 'rk2_tvd_split':
            return self._forward_rk2_tvd_split(v_inner, v0)

        device = v_inner.device
        Nr, Nphi = self.config.Nr, self.config.Nphi
        Nt = v_inner.shape[1]
        dt = self.config.dt
        Omega = self.config.Omega
        
        # Validate initial condition
        if _DEBUG_NAN:
            if torch.isnan(v0).any() or torch.isinf(v0).any():
                print(f"  ERROR: Initial condition v0 contains NaN/Inf!")
                return v0.unsqueeze(-1).expand(-1, -1, Nt)
        
            # Check v_inner boundary condition
            if torch.isnan(v_inner).any() or torch.isinf(v_inner).any():
                print(f"  ERROR: Inner boundary condition v_inner contains NaN/Inf!")
                return v0.unsqueeze(-1).expand(-1, -1, Nt)
        
        # Initialize output array
        v_full = torch.zeros(Nr, Nphi, Nt, device=device)
        v_full[:, :, 0] = v0
        
        # Time stepping with Strang splitting: R(dt/2) - P(dt) - R(dt/2)
        for n in range(1, Nt):
            v_prev = v_full[:, :, n - 1]
            
            if _DEBUG_NAN:
                if torch.isnan(v_prev).any() or torch.isinf(v_prev).any():
                    print(f"  ERROR at time step {n}: v_prev contains NaN/Inf")
                    return v_full
            
            # Step 1: Half radial step (IMEX RK2, dt/2)
            v_half_r = IMEX_RK2_radial(
                v_prev,
                self.dr,
                dt / 2.0,
                r_grid=self.r.view(-1, 1).expand_as(v_prev),
                learned_radial_accel=self.learned_radial_accel,
            )
            
            if _DEBUG_NAN:
                if torch.isnan(v_half_r).any() or torch.isinf(v_half_r).any():
                    print(f"  ERROR at step 1 (half radial): NaN at time {n}")
                    return v_full
            
            # Step 2: Full azimuthal step (IMEX RK2, dt)
            v_full_phi = IMEX_RK2_azimuthal(v_half_r, self.dphi, Omega, dt)
            
            if _DEBUG_NAN:
                if torch.isnan(v_full_phi).any() or torch.isinf(v_full_phi).any():
                    print(f"  ERROR at step 2 (azimuthal): NaN at time {n}")
                    return v_full
            
            # Step 3: Half radial step (IMEX RK2, dt/2)
            v_strang = IMEX_RK2_radial(
                v_full_phi,
                self.dr,
                dt / 2.0,
                r_grid=self.r.view(-1, 1).expand_as(v_full_phi),
                learned_radial_accel=self.learned_radial_accel,
            )
            
            if _DEBUG_NAN:
                if torch.isnan(v_strang).any() or torch.isinf(v_strang).any():
                    print(f"  ERROR at step 3 (half radial): NaN at time {n}")
                    return v_full
            
            # Apply inner boundary condition
            v_strang[0, :] = v_inner[:, n]
            
            # Store result
            v_full[:, :, n] = v_strang
        
        return v_full
