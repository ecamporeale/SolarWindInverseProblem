"""Low-order real-spherical-harmonic representation of the global photospheric
radial field B_r, fit from a SINGLE visible HMI hemisphere per timestamp.

Deliberately NOT a synoptic/full-sphere reconstruction: no far-side filling,
no PFSS. The fit is a masked/weighted least-squares projection of B_r onto
the first few real SH degrees (l=0..L_max, all m), using only genuinely
observed near-side pixels. The point is to ask how much LOW-ORDER global
magnetic state is actually constrained by one hemisphere's worth of data --
see verify_sh_fit.py for the synthetic-recovery / conditioning diagnostics
this module is built to support.

Conventions
-----------
theta : colatitude in [0, pi], 0 = north rotational pole.
phi   : longitude in [0, 2*pi), the SAME longitude convention as the rest of
        this codebase's Carrington-longitude grid (see preprocess_solar_image.py).
Coefficients are ordered by sh_index_list(l_max): l=0..L_max outer, m=-l..+l
inner, ascending. This ordering is the single source of truth other modules
must use to interpret a coefficient vector -- see sh_index_list().
"""
from __future__ import annotations

import numpy as np
from scipy.special import sph_harm_y


def sh_index_list(l_max: int) -> list[tuple[int, int]]:
    """The fixed (l, m) ordering used everywhere in this module."""
    return [(l, m) for l in range(l_max + 1) for m in range(-l, l + 1)]


def real_sh_basis(l_max: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Orthonormal real spherical harmonics, degrees 0..l_max, all orders.

    theta, phi: arrays of the same shape, any number of dimensions (colatitude,
    longitude respectively), in radians.

    Returns basis of shape (*theta.shape, n_coef) with n_coef = (l_max+1)**2,
    columns ordered per sh_index_list(l_max). Orthonormal in the sense that
    integral over the unit sphere of Y_a * Y_b dOmega = delta_ab (verified
    numerically in verify_sh_fit.py).

    Built from scipy's complex sph_harm_y (Condon-Shortley phase included):
        m=0:  Y_l0 = Re(Y_l^0)                      (already real)
        m>0:  sqrt(2) * (-1)^m * Re(Y_l^m)
        m<0:  sqrt(2) * (-1)^m * Im(Y_l^|m|)
    """
    theta = np.asarray(theta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    if theta.shape != phi.shape:
        raise ValueError(f"theta/phi shape mismatch: {theta.shape} vs {phi.shape}")

    idx = sh_index_list(l_max)
    out = np.empty(theta.shape + (len(idx),), dtype=np.float64)
    # Cache complex Y_l^|m| per (l, |m|) since m and -m share the same magnitude.
    cache: dict[tuple[int, int], np.ndarray] = {}
    for j, (l, m) in enumerate(idx):
        am = abs(m)
        key = (l, am)
        if key not in cache:
            cache[key] = sph_harm_y(l, am, theta, phi)
        y = cache[key]
        if m == 0:
            out[..., j] = y.real
        elif m > 0:
            out[..., j] = np.sqrt(2.0) * ((-1) ** m) * y.real
        else:
            out[..., j] = np.sqrt(2.0) * ((-1) ** am) * y.imag
    return out


def fit_sh_coefficients(
    values: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    l_max: int,
    weights: np.ndarray | None = None,
    ridge_lambda: float = 0.0,
    degree_damping: bool = False,
) -> dict:
    """Masked/weighted least-squares fit of values ~ sum a_lm Y_lm(theta,phi).

    values, theta, phi: 1-D arrays over OBSERVED pixels only (already masked
    -- this function does no masking of its own).
    weights: per-pixel weight w_p >= 0 (e.g. center-to-limb foreshortening
        factor); None -> uniform weights.
    ridge_lambda: Tikhonov regularization strength on the normal equations
        (A^T W A + ridge_lambda * D) a = A^T W b.
    degree_damping: if True, D = diag(l*(l+1)) per coefficient (damps
        high-l modes more, standard Tikhonov-on-roughness choice); if False,
        D = identity (plain ridge).

    Returns dict with:
        coeffs        : (n_coef,) fitted a_lm, ordered per sh_index_list(l_max)
        residual_rms  : RMS of (fit - values) over observed pixels
        cond_number   : condition number of the (unregularized) normal matrix
                         A^T W A -- the key conditioning diagnostic
        singular_values: singular values of the weighted design matrix
        n_obs         : number of observed pixels used
        n_coef        : number of fitted coefficients
        rank_deficient: True if n_obs < n_coef (fit is underdetermined even
                         before regularization)
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    theta = np.asarray(theta, dtype=np.float64).ravel()
    phi = np.asarray(phi, dtype=np.float64).ravel()
    n_obs = values.shape[0]
    if theta.shape[0] != n_obs or phi.shape[0] != n_obs:
        raise ValueError("values/theta/phi must have the same length")
    if weights is None:
        w = np.ones(n_obs, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.shape[0] != n_obs:
            raise ValueError("weights must match values length")

    idx = sh_index_list(l_max)
    n_coef = len(idx)
    A = real_sh_basis(l_max, theta, phi)  # (n_obs, n_coef)
    sqrt_w = np.sqrt(np.clip(w, 0.0, None))
    Aw = A * sqrt_w[:, None]
    bw = values * sqrt_w

    # Conditioning diagnostics on the UNREGULARIZED weighted normal matrix,
    # since that's what answers "is this actually constrained by the data".
    AtA = Aw.T @ Aw
    svals = np.linalg.svd(Aw, compute_uv=False)
    cond_number = float(svals[0] / svals[-1]) if svals[-1] > 0 else float('inf')

    if degree_damping:
        D = np.array([l * (l + 1) for l, m in idx], dtype=np.float64)
    else:
        D = np.ones(n_coef, dtype=np.float64)
    normal = AtA + ridge_lambda * np.diag(D)
    rhs = Aw.T @ bw
    coeffs = np.linalg.solve(normal, rhs)

    fit_vals = A @ coeffs
    residual_rms = float(np.sqrt(np.mean((fit_vals - values) ** 2)))

    return {
        'coeffs': coeffs,
        'idx': idx,
        'residual_rms': residual_rms,
        'cond_number': cond_number,
        'singular_values': svals,
        'n_obs': n_obs,
        'n_coef': n_coef,
        'rank_deficient': n_obs < n_coef,
    }


def rotate_coeffs_z(coeffs: np.ndarray, l_max: int, delta_phi: float) -> np.ndarray:
    """Re-express real-SH coefficients under phi -> phi - delta_phi, i.e. the
    coefficients such that sum a'_lm Y_lm(theta, phi') reproduces the SAME
    physical field evaluated at phi = phi' + delta_phi.

    Concretely: to make the field's value at absolute longitude phi0 become
    the new "phi'=0" reference, call with delta_phi = phi0.

    For each l, the (+m, -m) pair (m>0) rotates as a 2D rotation by angle
    m*delta_phi; m=0 (axisymmetric) is invariant. Derivation: the real basis
    at fixed (l, |m|=k) factors as K(theta) * [cos(k*phi), sin(k*phi)] with a
    SHARED radial factor K(theta) for the +k and -k slots (same associated-
    Legendre normalization). Substituting phi = phi' + delta_phi into
    a_+ cos(k*phi) + a_- sin(k*phi) and collecting cos(k*phi')/sin(k*phi')
    terms gives
        a'_+ =  a_+ * cos(k*delta_phi) + a_- * sin(k*delta_phi)
        a'_- = -a_+ * sin(k*delta_phi) + a_- * cos(k*delta_phi)
    Verified numerically in verify_sh_fit.py against direct basis-function
    evaluation (an earlier version of this function had a-/+ transposed and
    failed that check at every angle except 0/180 deg, where sin(k*delta)=0
    makes the bug invisible).
    """
    idx = sh_index_list(l_max)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    if coeffs.shape[0] != len(idx):
        raise ValueError(f"coeffs length {coeffs.shape[0]} != n_coef {len(idx)} for l_max={l_max}")
    pos = {(l, m): j for j, (l, m) in enumerate(idx)}
    out = coeffs.copy()
    for l in range(l_max + 1):
        for m in range(1, l + 1):
            jp, jn = pos[(l, m)], pos[(l, -m)]
            a, b = coeffs[jp], coeffs[jn]
            c, s = np.cos(m * delta_phi), np.sin(m * delta_phi)
            out[jp] = c * a + s * b
            out[jn] = -s * a + c * b
    return out


def evaluate_sh_field(coeffs: np.ndarray, l_max: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """sum_lm coeffs[lm] * Y_lm(theta, phi), for sanity-checking fits/rotations."""
    basis = real_sh_basis(l_max, theta, phi)
    return basis @ coeffs
