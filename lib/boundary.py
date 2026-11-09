import itertools
import math
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import warnings

from forward_model import steady_state_propagate
from global_field_sh import sh_index_list


# =========================================================
# STEP 2: INITIALIZE BOUNDARY CONDITION
# =========================================================

class SpectralLongitudeLayer(nn.Module):
    """
    Fourier operator acting ONLY along longitude dimension.

    Input:
        [Nt, Cin, Nphi]

    Output:
        [Nt, Cout, Nphi]
    """

    def __init__(self, in_channels, out_channels, n_modes):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes

        scale = 1.0 / (in_channels * out_channels)

        self.weight_real = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, n_modes)
        )

        self.weight_imag = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, n_modes)
        )

    def forward(self, x):
        """
        x:
            [Nt, Cin, Nphi]
        """

        Nt, Cin, Nphi = x.shape

        # -------------------------------------------------
        # FFT along longitude ONLY
        # -------------------------------------------------

        x_ft = torch.fft.rfft(x, dim=-1)

        Nk = x_ft.shape[-1]

        # -------------------------------------------------
        # Allocate output Fourier tensor
        # -------------------------------------------------

        out_ft = torch.zeros(
            Nt,
            self.out_channels,
            Nk,
            dtype=torch.cfloat,
            device=x.device,
        )

        # -------------------------------------------------
        # Learned complex weights
        # -------------------------------------------------

        nm = min(self.n_modes, Nk)

        weights = torch.complex(
            self.weight_real[:, :, :nm],
            self.weight_imag[:, :, :nm],
        )

        # Complex spectral mixing
        #
        # x_ft:
        #   [Nt, Cin, Nk]
        #
        # weights:
        #   [Cin, Cout, Nk]
        #
        out_ft[:, :, :nm] = torch.einsum(
            "bik,iok->bok",
            x_ft[:, :, :nm],
            weights,
        )

        # -------------------------------------------------
        # IFFT back to longitude space
        # -------------------------------------------------

        x_out = torch.fft.irfft(
            out_ft,
            n=Nphi,
            dim=-1,
        )

        return x_out


class BoundaryCondition(nn.Module):
    """Trainable inner boundary condition at r0."""

    def __init__(self, config, initial_guess=None):
        super().__init__()

        self.register_buffer(
            "aia_mean",
            torch.tensor(config.aia_global_mean, dtype=torch.float32)
        )

        self.register_buffer(
            "aia_std",
            torch.tensor(config.aia_global_std, dtype=torch.float32)
        )


        self.fno_input_channels = 2

        k1 = 5
        k2 = 5
        k3 = 3

        hidden_channels = 8

        # -------------------------------------------------
        # Local encoder
        # -------------------------------------------------

        self.spatial_encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=2,
                out_channels=4,
                kernel_size=k1,
                padding=k1 // 2,
                padding_mode='circular',
            ),
            nn.GELU(),

            nn.Conv1d(
                in_channels=4,
                out_channels=hidden_channels,
                kernel_size=k2,
                padding=k2 // 2,
                padding_mode='circular',
            ),
            nn.GELU(),
        )

        # -------------------------------------------------
        # TRUE spectral operator
        # -------------------------------------------------

        self.output_fourier = self.is_reduced_basis

        self.max_mode = int(
            getattr(config, 'fourier_max_mode', 12)
        )

        self.max_mode = max(
            1,
            min(self.max_mode, config.Nphi // 2)
        )

        self.spectral_layer = SpectralLongitudeLayer(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            n_modes=self.max_mode + 1,
        )

        # Local residual mixing
        self.local_mixing = nn.Conv1d(
            hidden_channels,
            hidden_channels,
            kernel_size=1,
        )

        self.activation = nn.GELU()

        # -------------------------------------------------
        # Final reconstruction
        # -------------------------------------------------

        self.final_conv = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=1,
            kernel_size=k3,
            padding=k3 // 2,
            padding_mode='circular',
        )

        self.register_buffer(
            'corrected_mask',
            torch.zeros(config.Nphi, config.Nt, dtype=torch.bool)
        )

        self.time_lowpass_modes = int(
            getattr(config, 'fno_time_lowpass_modes', 0)
        )

        self.warmup_prior = False


    def forward(self, aia_slice=None):

        v_min = 150.0
        v_max = 750.0

        v_mid = (v_min + v_max) / 2.0
        v_scale = (v_max - v_min) / 2.0


        if aia_slice is None:
            raise ValueError(
                "AIA slice tensor required when use_aia_nn_mapping=True"
            )

        if aia_slice.ndim == 3:

            if aia_slice.shape[0] < self.fno_input_channels:
                raise ValueError(
                    f"Expected at least {self.fno_input_channels} input channels; "
                    f"got {aia_slice.shape[0]}"
                )

            aia = aia_slice[0]
            valid_mask = aia_slice[1] > 0.5

        else:

            valid_mask = torch.isfinite(aia_slice)
            aia = aia_slice

        aia = torch.nan_to_num(
            aia,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        encoder_dtype = next(
            self.spatial_encoder.parameters()
        ).dtype

        aia = aia.to(dtype=encoder_dtype)

        valid_mask = valid_mask.to(dtype=encoder_dtype)

        if torch.any(valid_mask > 0):

            aia = torch.where(
                valid_mask > 0,
                (aia - self.aia_mean)
                / torch.clamp(self.aia_std, min=1e-6),
                torch.zeros_like(aia)
            )

        else:

            aia = torch.zeros_like(aia)

        aia = torch.clamp(aia, min=-6.0, max=6.0)

        # -------------------------------------------------
        # Build [Nt, 2, Nphi]
        # -------------------------------------------------

        x = torch.stack([aia, valid_mask], dim=0)

        x = x.permute(2, 0, 1)

        # [Nt, 2, Nphi]

        # -------------------------------------------------
        # Local encoder
        # -------------------------------------------------

        hidden = self.spatial_encoder(x)

        # [Nt, 32, Nphi]

        # -------------------------------------------------
        # TRUE Fourier operator
        # -------------------------------------------------

        spectral_out = self.spectral_layer(hidden)

        local_out = self.local_mixing(hidden)

        hidden = self.activation(
            spectral_out + local_out
        )

        # -------------------------------------------------
        # Reconstruct longitude structure
        # -------------------------------------------------

        v_logits = self.final_conv(hidden)

        # [Nt, 1, Nphi]

        v_raw = v_logits.squeeze(1).transpose(0, 1)

        # [Nphi, Nt]

        v_normalized = torch.sigmoid(v_raw)
        v_physical = (
            v_mid
            + v_scale * (2.0 * v_normalized - 1.0)
        )

        if self.time_lowpass_modes > 0:
            v_physical = self._apply_time_lowpass(v_physical, self.time_lowpass_modes)
        # Re-apply mask after lowpass: Fourier smoothing can bleed v_mid into
        # the very edge of valid cells and pull valid-edge cells slightly toward
        # v_mid, but the backside cells themselves must stay exactly at v_mid.
        v_physical = torch.where(
                valid_mask > 0.5,
                v_physical,
                torch.full_like(v_physical, v_mid),
            )

        return v_physical
    def _apply_time_lowpass(self, v_raw, keep_modes):

        if keep_modes <= 0:
            return v_raw

        n_time = v_raw.shape[1]

        spectrum = torch.fft.rfft(v_raw, dim=1)

        max_keep = min(
            keep_modes,
            spectrum.shape[1]
        )

        if max_keep < spectrum.shape[1]:

            spectrum = spectrum.clone()

            spectrum[:, max_keep:] = 0.0

        return torch.fft.irfft(
            spectrum,
            n=n_time,
            dim=1
        )

class IntensityToVelocityMLP(nn.Module):
    """
    Pointwise AIA -> solar wind speed mapping.

    Each visible pixel is processed independently:

        v(φ,t) = g(I(φ,t))

    No longitude coupling.
    No temporal coupling.
    No spectral layers.
    """

    def __init__(self, config):
        super().__init__()

        # Hidden-layer widths, configurable via config.mlp_hidden_dims (CLI
        # --mlp-hidden-dims). Default (100,64,32) is the original
        # architecture (10,077 params total) unchanged. A shallower list,
        # e.g. (27,16,8) (~998 params), keeps the same 14-input stencil and
        # depth (3 hidden layers) while cutting capacity ~10x, to test
        # capacity vs. input/stencil as separate axes (see
        # SESSION_CHANGELOG.md).
        hidden_dims = list(getattr(config, "mlp_hidden_dims", (100, 64, 32)))

        self.radius = 3

        input_dim = 2*(2 * self.radius + 1)

        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.GELU()]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.head = nn.Sequential(*layers)



        self.register_buffer(
            "aia_mean",
            torch.tensor(
                float(config.aia_global_mean),
                dtype=torch.float32,
            ),
        )

        self.register_buffer(
            "aia_std",
            torch.tensor(
                float(config.aia_global_std),
                dtype=torch.float32,
            ),
        )

        self.register_buffer(
            "hmi_mean",
            torch.tensor(
                float(config.hmi_global_mean),
                dtype=torch.float32,
            ),
        )

        self.register_buffer(
            "hmi_std",
            torch.tensor(
                float(config.hmi_global_std),
                dtype=torch.float32,
            ),
        )

        self.time_lowpass_modes = int(
            getattr(config, "fno_time_lowpass_modes", 0)
        )

    def forward(self, image_slice):

#        print('Using IntensityToVelocityMLP boundary condition with pointwise MLP mapping')
        if image_slice is None:
            raise ValueError(
                "Image slice tensor required"
            )

        # ---------------------------------------------
        # unpack channels
        # ---------------------------------------------
#        print('inside IntensityToVelocityMLP forward, image_slice.shape:', image_slice.shape)
        aia = image_slice[0]

        if image_slice.shape[0] >= 3:
            hmi = image_slice[2]
        else:
            hmi = torch.zeros_like(aia)

        if image_slice.ndim != 3:
            raise ValueError(
                f"Expected [C,Nphi,Nt], got {image_slice.shape}"
            )

     
#        print('image_slice.shape:', image_slice.shape)
        if image_slice.shape[0] >= 2:
            valid_mask = image_slice[1] > 0.5
        else:
            valid_mask = torch.isfinite(aia)

        # ---------------------------------------------
        # sanitize
        # ---------------------------------------------

        aia = torch.nan_to_num(
            aia,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # remove invisible pixels completely
        aia = torch.where(
            valid_mask,
            aia,
            torch.zeros_like(aia)
        )

        # HMI gets the same local sanitization as AIA. In the current pipeline
        # _slice_image_batch already sanitizes HMI upstream, so this is a
        # no-op on real data -- but without it, a NaN reaching this function
        # would survive `hmi * valid_mask.float()` (NaN * 0.0 == NaN in
        # IEEE754, not 0), unlike AIA which is already NaN-free by that point.
        hmi = torch.nan_to_num(
            hmi,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        hmi = torch.where(
            valid_mask,
            hmi,
            torch.zeros_like(hmi)
        )

        # ---------------------------------------------
        # global normalization
        #
        # replace with your stored global stats
        # ---------------------------------------------

        aia = (
            aia - self.aia_mean
        ) / torch.clamp(
            self.aia_std,
            min=1e-6
        )
        hmi = (
            hmi - self.hmi_mean
        ) / torch.clamp(
            self.hmi_std,min = 1e-6)
          
       
 #       print(
 #           "normalized AIA",
 #           aia.min().item(),
 #           aia.max().item(),
 #           aia.mean().item(),
 #           aia.std().item(),
 #       )

 #       print(
 #           "normalized HMI",
 #           hmi.min().item(),
 #           hmi.max().item(),
 #           hmi.mean().item(),
 #           hmi.std().item(),
 #       )

        # ---------------------------------------------
        # pointwise MLP
        # ---------------------------------------------


        masked_aia = aia * valid_mask.float()
        masked_hmi = hmi * valid_mask.float()

        aia_patch = []
        hmi_patch = []

        for shift in range(-self.radius, self.radius + 1):

            aia_patch.append(
                torch.roll(masked_aia,
                        shifts=shift,
                        dims=0)
            )

            hmi_patch.append(
                torch.roll(masked_hmi,
                        shifts=shift,
                        dims=0)
            )

        x = torch.stack(aia_patch + hmi_patch,dim=-1,)    # [Nphi, Nt, input_dim]
#        print('x.shape:', x.shape)
#        print(
#            "head input",
#            x.min().item(),
#            x.max().item(),
#            x.mean().item(),
#            x.std().item(),
#        )


        v_raw = self.head(x)
        v_raw = v_raw.squeeze(-1)
        

#        print(
#            "v_raw",
#            v_raw.min().item(),
#            v_raw.max().item(),
#            v_raw.mean().item(),
#            v_raw.std().item(),
#        )


        # ---------------------------------------------
        # convert to physical speed
        # ---------------------------------------------

        v_min = 150.0
        v_max = 750.0

        v_mid = 0.5 * (v_min + v_max)
        v_scale = 0.5 * (v_max - v_min)

        v_normalized = torch.sigmoid(v_raw)

#        print(
#            "sigmoid",
#            v_normalized.min().item(),
#            v_normalized.max().item(),
#            v_normalized.mean().item(),
#        )


        v_physical = (
            v_mid
            + v_scale
            * (2.0 * v_normalized - 1.0)
        )

#        print("v_min =", v_min)
#        print("v_max =", v_max)
#        print("v_mid =", v_mid)
#        print("v_scale =", v_scale)
        # enforce mask
        v_physical = torch.where(
            valid_mask,
            v_physical,
            torch.full_like(v_physical, v_mid),
        )
#        print(
#            "v_physical after mask",
#            v_physical.min().item(),
#            v_physical.max().item(),
#            v_physical.mean().item(),
#            v_physical.std().item(),
#        )

        return v_physical

    def _apply_time_lowpass(
        self,
        v_raw,
        keep_modes,
    ):

        if keep_modes <= 0:
            return v_raw

        spectrum = torch.fft.rfft(
            v_raw,
            dim=1,
        )

        max_keep = min(
            keep_modes,
            spectrum.shape[1],
        )

        if max_keep < spectrum.shape[1]:
            spectrum = spectrum.clone()
            spectrum[:, max_keep:] = 0.0

        return torch.fft.irfft(
            spectrum,
            n=v_raw.shape[1],
            dim=1,
        )


class ThresholdVelocityModel(nn.Module):
    """
    Pointwise AIA -> speed mapping with 4 free parameters instead of
    IntensityToVelocityMLP's 10,077.

        v(phi,t) = v_slow + (v_fast - v_slow) * sigmoid((I0 - I(phi,t)) / w)

    A single input (the pixel's own normalized AIA intensity, no stencil,
    no HMI) run through a fixed monotone shape -- the coronal-hole-threshold
    form used in WSA-family empirical models (dark/low-intensity -> fast,
    bright -> slow). Monotone by construction, so it cannot act as the
    switch/bang-bang pattern IntensityToVelocityMLP was found to converge to
    (see SESSION_CHANGELOG.md, "Root-cause analysis" entry): a monotone
    function of one variable has no spare capacity to carve the input space
    into disconnected regions sent to opposite rails.

    v_slow/v_fast are reparametrized through a sigmoid into [150, 750] (same
    physical range IntensityToVelocityMLP's output sigmoid targets) so the
    raw parameters are unconstrained and stay well-conditioned under Adam.
    w is passed through softplus to stay positive (a zero or negative width
    would make the sigmoid threshold non-differentiable / discontinuous).
    I0 is left free: it is a threshold in the same normalized-AIA units the
    aia_mean/aia_std buffers produce, not a physical intensity, so there is
    no natural bound to reparametrize it into.
    """

    def __init__(self, config):
        super().__init__()

        self.register_buffer(
            "aia_mean", torch.tensor(float(config.aia_global_mean), dtype=torch.float32),
        )
        self.register_buffer(
            "aia_std", torch.tensor(float(config.aia_global_std), dtype=torch.float32),
        )

        v_min, v_max = 150.0, 750.0
        self._v_min = v_min
        self._v_span = v_max - v_min

        # Init: slow~380, fast~650 (roughly the pre-fit ACE mean +/- 1 std),
        # I0 at the normalized-intensity origin (~global median AIA pixel),
        # w=1.0 (one normalized-intensity std of transition width).
        def _inv_sigmoid(frac):
            frac = min(max(frac, 1e-4), 1.0 - 1e-4)
            return float(np.log(frac / (1.0 - frac)))

        self.raw_v_slow = nn.Parameter(torch.tensor(_inv_sigmoid((380.0 - v_min) / (v_max - v_min))))
        self.raw_v_fast = nn.Parameter(torch.tensor(_inv_sigmoid((650.0 - v_min) / (v_max - v_min))))
        self.I0 = nn.Parameter(torch.tensor(0.0))
        self.raw_w = nn.Parameter(torch.tensor(float(np.log(np.expm1(1.0)))))  # softplus^-1(1.0)

    def _params(self):
        v_slow = self._v_min + self._v_span * torch.sigmoid(self.raw_v_slow)
        v_fast = self._v_min + self._v_span * torch.sigmoid(self.raw_v_fast)
        w = nn.functional.softplus(self.raw_w) + 1e-3
        return v_slow, v_fast, w

    def forward(self, image_slice):
        if image_slice is None:
            raise ValueError("Image slice tensor required")
        if image_slice.ndim != 3:
            raise ValueError(f"Expected [C,Nphi,Nt], got {image_slice.shape}")

        aia = image_slice[0]
        valid_mask = image_slice[1] > 0.5 if image_slice.shape[0] >= 2 else torch.isfinite(aia)

        aia = torch.nan_to_num(aia, nan=0.0, posinf=0.0, neginf=0.0)
        aia_norm = (aia - self.aia_mean) / torch.clamp(self.aia_std, min=1e-6)

        v_slow, v_fast, w = self._params()
        v_physical = v_slow + (v_fast - v_slow) * torch.sigmoid((self.I0 - aia_norm) / w)

        v_mid = 0.5 * (v_slow + v_fast)
        v_physical = torch.where(valid_mask, v_physical, v_mid.expand_as(v_physical))
        return v_physical


def _windowed_sum(x, radius):
    """Circular moving sum of x over +/-radius pixels along the longitude
    axis (dim 0). Nphi=128 spans the full 360 deg, so this wraps
    correctly at the 0/360 boundary via torch.roll. Shared by
    ContextualVelocityMLP and MultiFeatureThresholdModel."""
    out = torch.zeros_like(x)
    for shift in range(-radius, radius + 1):
        out = out + torch.roll(x, shifts=shift, dims=0)
    return out


def _build_aia_norm_spec(config):
    """Read config.aia_normalization_mode ("global" or "asinh" -- no other
    value is accepted, see config.py's comment on why per-frame normalization
    was removed) and return the (mode, mean, std, clip_dn, a) tuple that
    _normalize_aia/_compute_contextual_features need. Called once per model
    __init__; raises immediately (not lazily inside forward()) if the config
    is missing what the chosen mode requires.
    """
    mode = str(getattr(config, "aia_normalization_mode", None))
    if mode not in ("global", "asinh"):
        raise ValueError(
            f"config.aia_normalization_mode={mode!r} is not supported; must be "
            "'global' or 'asinh'. Per-frame normalization was removed entirely "
            "(see config.py) -- it collapsed cross-fold AUC but also destroyed "
            "real signal, so it is not a selectable option, not even by mistake."
        )
    if mode == "global":
        return (mode, float(config.aia_global_mean), float(config.aia_global_std), 0.0, 0.0)
    clip_dn = getattr(config, "aia_asinh_clip_dn", None)
    if clip_dn is None:
        raise ValueError(
            "aia_normalization_mode='asinh' but config.aia_asinh_clip_dn is None -- "
            "it's set at data-load time in inverse_solver.py's _load_all_image_data; "
            "build the model AFTER loading AIA/HMI data, not before."
        )
    return (mode, 0.0, 1.0, float(clip_dn), float(getattr(config, "aia_asinh_a", 0.005)))


def _normalize_aia(aia, mask_f, mode, mean, std, clip_dn, a):
    """AIA normalization, dispatched on config.aia_normalization_mode:

    "global": fixed dataset-wide (aia - mean) / std, mask-aware.

    "asinh": clip(aia, 0, clip_dn) / clip_dn, then a log-like stretch
    asinh(x/a) / asinh(1/a) that expands faint structure (coronal holes)
    and compresses bright active regions, mapped from [0,1] to [-1,1].

    Per-frame normalization (each timestep centered on its own visible-cell
    mean/std) was tried and reverted: it cut cross-fold domain-classifier
    AUC from ~0.88-1.00 to ~0.58-0.65, but re-running the SAME check with
    this fixed clip+asinh stretch (still global, not per-frame) landed at
    the same ~0.62 AUC as the fixed global z-score -- so the residual
    ~0.6 isn't a rescaling-curve problem either way, and per-frame
    normalization's extra removal of the multi-year AUC-0.88+ signal was
    coming out of real cross-fold signal, not only the spurious trend.
    See degradation_pca_check.py / asinh_stretch_check.py / apply_
    degradation_correction_check.py and the session log for the measurements.
    """
    if mode == "global":
        return ((aia - mean) / torch.clamp(std, min=1e-6)) * mask_f
    if mode == "asinh":
        z = torch.clamp(aia, min=0.0, max=clip_dn) / clip_dn
        w = torch.asinh(z / a) / math.asinh(1.0 / a)
        return (2.0 * w - 1.0) * mask_f
    raise ValueError(f"Unknown AIA normalization mode {mode!r}")


def _compute_contextual_features(image_slice, hmi_mean, hmi_std, window, aia_norm_spec):
    """The 4 non-local features shared by ContextualVelocityMLP,
    MultiFeatureThresholdModel and MergedTemporalVelocityModel -- see the
    relevant class's docstring for the physical motivation. Returns
    (I_center, I_wide, B_center, U_wide, valid_mask), each [Nphi, Nt]
    except valid_mask which is bool.

    aia_norm_spec is the (mode, mean, std, clip_dn, a) tuple from
    _build_aia_norm_spec -- see _normalize_aia for what each mode does.
    HMI is always on the fixed global normalization (a domain-classifier
    check found it essentially unshifted across batches, unlike raw AIA).
    """
    if image_slice is None:
        raise ValueError("Image slice tensor required")
    if image_slice.ndim != 3:
        raise ValueError(f"Expected [C,Nphi,Nt], got {image_slice.shape}")

    aia = image_slice[0]
    hmi = image_slice[2] if image_slice.shape[0] >= 3 else torch.zeros_like(aia)
    valid_mask = image_slice[1] > 0.5 if image_slice.shape[0] >= 2 else torch.isfinite(aia)
    mask_f = valid_mask.float()

    aia = torch.nan_to_num(aia, nan=0.0, posinf=0.0, neginf=0.0) * mask_f
    hmi = torch.nan_to_num(hmi, nan=0.0, posinf=0.0, neginf=0.0) * mask_f

    aia_norm = _normalize_aia(aia, mask_f, *aia_norm_spec)
    hmi_norm = (hmi - hmi_mean) / torch.clamp(hmi_std, min=1e-6)

    # Mask-aware windowed averages: divide by the windowed COUNT of
    # visible pixels, not by the window width, so a window that partially
    # crosses the far-side terminator isn't biased toward zero by
    # blending in invisible (zeroed) cells.
    den = torch.clamp(_windowed_sum(mask_f, window), min=1.0)
    I_wide = _windowed_sum(aia_norm * mask_f, window) / den
    # Unipolarity uses RAW (not z-scored) hmi -- it's a ratio of two
    # windowed means, so it's already scale-invariant.
    B_wide_signed = _windowed_sum(hmi * mask_f, window) / den
    B_wide_abs = _windowed_sum(hmi.abs() * mask_f, window) / den
    U_wide = torch.clamp(B_wide_signed / torch.clamp(B_wide_abs, min=1e-3), min=-1.0, max=1.0)

    return aia_norm, I_wide, hmi_norm, U_wide, valid_mask


class ContextualVelocityMLP(nn.Module):
    """
    Pointwise-in-time AIA/HMI -> speed mapping with a small set of
    physically-motivated, spatially NON-LOCAL features feeding a compact
    MLP, one notch up in complexity from ThresholdVelocityModel (4 params,
    single-pixel input) and ~45x down from IntensityToVelocityMLP (10,077
    params, 14-pixel stencil, found not to be identifiable from one
    training batch -- see SESSION_CHANGELOG.md).

    Motivation: coronal-hole wind-speed models (WSA/HUX-family) key off
    NON-LOCAL quantities -- flux-tube expansion factor, distance to the
    coronal-hole boundary -- that a single pixel's own brightness cannot
    encode: a small transient dark speck and a large polar coronal hole
    can look identical to one pixel but produce very different wind
    speeds. Without a full PFSS extrapolation (the real expansion-factor
    calculation), this adds the cheapest available proxies from data
    already in `image_slice`:

    4 input features, still pointwise IN TIME (no temporal coupling --
    that assumption is unchanged from every other NN1 variant this
    project uses):
        I_center - central AIA pixel intensity (normalized)
        I_wide   - AIA intensity mask-aware-averaged over a wide
                   (+/- config.contextual_window_radius px) longitude
                   window -- how large/dark the surrounding region is,
                   not just this one point
        B_center - central HMI signed magnetic field (normalized)
        U_wide   - mean(B)/mean(|B|) over the same wide window, in
                   [-1, 1] -- UNIPOLARITY, the standard coronal-hole
                   detection heuristic (near +/-1 = strongly one-signed
                   = coronal-hole-like; near 0 = mixed-polarity quiet Sun)

    fed through a small MLP (default 4 -> 16 -> 8 -> 1; width/depth
    configurable via config.contextual_hidden_dims), output mapped into
    [150, 750] km/s via an EXPLICIT learned center/span:

        v = clamp(v_mid + v_halfspan * tanh(head(x)), 150, 750)

    v_mid and v_halfspan are themselves learned scalars (reparametrized
    through sigmoid/softplus to stay bounded), the same trick
    ThresholdVelocityModel uses for v_slow/v_fast. This replaces an
    earlier version of this class that squashed the MLP's raw output
    directly through a fixed sigmoid(150 + 600*sigmoid(head(x))) -- with
    NO learned scale, that version required the MLP's own small-random-
    init weights to "discover" a wide output range from scratch, and on
    all 4 batches it never did: the pre-activation logit's std stayed at
    0.008-0.031 through 40 epochs (collapsed into the sigmoid's near-
    linear, low-slope region around v=450, output range never reaching
    beyond ~193-475 km/s, POD@500=0.000 on every batch). See
    SESSION_CHANGELOG.md entry 12. Giving v_mid/v_halfspan their own
    parameters means the optimizer only has to learn the mapping's SHAPE,
    not also discover the right output scale.

    Unlike ThresholdVelocityModel this is NOT constrained to be monotone
    -- deliberately relaxing that guardrail now that the working
    hypothesis is that the input needs more information, not that the
    mapping needs less shape freedom.
    """

    def __init__(self, config):
        super().__init__()
        self.window = int(getattr(config, "contextual_window_radius", 15))
        hidden_dims = list(getattr(config, "contextual_hidden_dims", (16, 8)))

        self.register_buffer(
            "hmi_mean", torch.tensor(float(config.hmi_global_mean), dtype=torch.float32),
        )
        self.register_buffer(
            "hmi_std", torch.tensor(float(config.hmi_global_std), dtype=torch.float32),
        )
        self.aia_mode, aia_mean, aia_std, aia_clip_dn, self.aia_a = _build_aia_norm_spec(config)
        self.register_buffer("aia_mean", torch.tensor(aia_mean, dtype=torch.float32))
        self.register_buffer("aia_std", torch.tensor(aia_std, dtype=torch.float32))
        self.register_buffer("aia_clip_dn", torch.tensor(aia_clip_dn, dtype=torch.float32))

        layers = []
        in_dim = 4
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.GELU()]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.head = nn.Sequential(*layers)

        self._v_min = 150.0
        self._v_span = 600.0
        # v_mid: learned scalar in [150,750], init at the midpoint (450).
        self.raw_v_mid = nn.Parameter(torch.tensor(0.0))
        # v_halfspan: learned scalar in [0,300] (so v_mid+-v_halfspan can
        # cover up to the full [150,750] range), init at 150 -- roughly
        # the same order of magnitude ThresholdVelocityModel's init span
        # (650-380=270, i.e. +-135) starts with, so this architecture
        # isn't disadvantaged relative to that one from the first step.
        self.raw_v_halfspan = nn.Parameter(torch.tensor(0.0))  # sigmoid(0)=0.5 -> 150

    def forward(self, image_slice):
        aia_norm_spec = (self.aia_mode, self.aia_mean, self.aia_std, self.aia_clip_dn, self.aia_a)
        aia_norm, I_wide, hmi_norm, U_wide, valid_mask = _compute_contextual_features(
            image_slice, self.hmi_mean, self.hmi_std, self.window, aia_norm_spec,
        )

        x = torch.stack([aia_norm, I_wide, hmi_norm, U_wide], dim=-1)  # [Nphi, Nt, 4]
        v_raw = self.head(x).squeeze(-1)

        v_mid = self._v_min + self._v_span * torch.sigmoid(self.raw_v_mid)
        v_halfspan = 300.0 * torch.sigmoid(self.raw_v_halfspan)
        v_physical = torch.clamp(v_mid + v_halfspan * torch.tanh(v_raw), min=self._v_min, max=self._v_min + self._v_span)

        # Far-side fallback reuses v_mid directly (the model's own learned
        # center) rather than a separate parameter -- v_mid already means
        # "the typical/central value", so a dedicated far-side scalar was
        # redundant.
        v_physical = torch.where(valid_mask, v_physical, v_mid.expand_as(v_physical))
        return v_physical


class MultiFeatureThresholdModel(nn.Module):
    """
    Logistic-regression-style extension of ThresholdVelocityModel: the
    SAME 4 non-local features as ContextualVelocityMLP (I_center, I_wide,
    B_center, U_wide -- see that class's docstring for what each means
    physically), combined through a single LEARNED LINEAR index z, fed
    through one sigmoid with an explicit learned center/span:

        z = w1*I_center + w2*I_wide + w3*B_center + w4*U_wide
        v = v_mid + v_halfspan * tanh((I0 - z) / w)

    8 trainable parameters total (4 feature weights + I0 + w + v_mid +
    v_halfspan). Isolates the "does more INPUT INFORMATION help" question
    from ContextualVelocityMLP's confound of also relaxing monotonicity
    and using an unconstrained multi-layer MLP: z is a single learned
    linear combination, so this is still monotone along its own learned
    direction and just as low-variance/well-conditioned as
    ThresholdVelocityModel, with no output-scale-collapse risk
    (v_mid/v_halfspan are explicit parameters from the start, same fix as
    entry 13 in SESSION_CHANGELOG.md).

    Weights init at [1, 0, 0, 0] -- i.e. this starts as an EXACT copy of
    ThresholdVelocityModel (pure I_center threshold, matching that
    model's own init and its solved-identifiability result), so training
    only has to discover whether the other 3 features earn nonzero
    weight, rather than searching from a random starting point.
    """

    def __init__(self, config):
        super().__init__()
        self.window = int(getattr(config, "contextual_window_radius", 15))

        self.register_buffer(
            "hmi_mean", torch.tensor(float(config.hmi_global_mean), dtype=torch.float32),
        )
        self.register_buffer(
            "hmi_std", torch.tensor(float(config.hmi_global_std), dtype=torch.float32),
        )
        self.aia_mode, aia_mean, aia_std, aia_clip_dn, self.aia_a = _build_aia_norm_spec(config)
        self.register_buffer("aia_mean", torch.tensor(aia_mean, dtype=torch.float32))
        self.register_buffer("aia_std", torch.tensor(aia_std, dtype=torch.float32))
        self.register_buffer("aia_clip_dn", torch.tensor(aia_clip_dn, dtype=torch.float32))

        # [I_center, I_wide, B_center, U_wide] weights, init = pure
        # I_center (matches ThresholdVelocityModel's starting point).
        self.weights = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        self.I0 = nn.Parameter(torch.tensor(0.0))
        self.raw_w = nn.Parameter(torch.tensor(float(np.log(np.expm1(1.0)))))  # softplus^-1(1.0)

        self._v_min = 150.0
        self._v_span = 600.0
        self.raw_v_mid = nn.Parameter(torch.tensor(0.0))       # sigmoid(0)=0.5 -> 450
        self.raw_v_halfspan = nn.Parameter(torch.tensor(0.0))  # sigmoid(0)=0.5 -> 150

    def forward(self, image_slice):
        aia_norm_spec = (self.aia_mode, self.aia_mean, self.aia_std, self.aia_clip_dn, self.aia_a)
        I_center, I_wide, B_center, U_wide, valid_mask = _compute_contextual_features(
            image_slice, self.hmi_mean, self.hmi_std, self.window, aia_norm_spec,
        )
        feats = torch.stack([I_center, I_wide, B_center, U_wide], dim=-1)  # [Nphi, Nt, 4]
        z = (feats * self.weights).sum(-1)
        w = nn.functional.softplus(self.raw_w) + 1e-3

        v_mid = self._v_min + self._v_span * torch.sigmoid(self.raw_v_mid)
        v_halfspan = 300.0 * torch.sigmoid(self.raw_v_halfspan)
        v_physical = torch.clamp(
            v_mid + v_halfspan * torch.tanh((self.I0 - z) / w),
            min=self._v_min, max=self._v_min + self._v_span,
        )
        v_physical = torch.where(valid_mask, v_physical, v_mid.expand_as(v_physical))
        return v_physical


class PolynomialVelocityModel(nn.Module):
    """
    Pointwise-in-time AIA/HMI -> speed mapping via polynomial regression:
    LINEAR in the learned parameters, polynomial (up to total degree 3,
    with all mixed/cross terms) in the inputs.

    6 raw inputs -- a narrow (+/-1 pixel) stencil, not the wide non-local
    windows ContextualVelocityMLP/MultiFeatureThresholdModel use:
        AIA[-1], AIA[0], AIA[+1]  (center pixel + 2 immediate neighbors)
        HMI[-1], HMI[0], HMI[+1]  (center pixel + 2 immediate neighbors)
    all z-scored with the same global aia/hmi mean+std buffers used
    throughout this project.

    Every monomial of total degree 0..3 in these 6 variables (including
    mixed products like AIA[0]*AIA[-1]*HMI[+1], not just pure powers of a
    single variable) is a feature; the output is a single linear
    combination of those 84 features:
        v = sum_k  w_k * monomial_k(x)
    84 trainable parameters (1 constant + 6 linear + 21 quadratic + 56
    cubic terms; count of degree-d monomials in n=6 variables is
    C(n+d-1, d), summed d=0..3 -> 1+6+21+56=84).

    Deliberately UNBOUNDED output (per explicit instruction, after
    reviewing entries 9/10/12/13): no sigmoid/tanh squashing wrapper, no
    separate learned v_mid/v_halfspan. The output is used as the boundary
    velocity directly. Two consequences, both accepted as part of that
    choice rather than guarded against here:
      - cubic cross-terms can in principle produce extreme values; the
        physics forward model's own boundary-injection clamp ([100,900],
        see forward_model.py's _forward_rk2_tvd_split) is the only
        safety net, and gradients vanish for boundary values outside that
        clamp.
      - there is no explicit far-side (invisible-cell) fallback constant.
        Invisible cells get their AIA/HMI values zeroed then normalized
        (the same masking convention as every other NN1 variant here),
        which is a FIXED input across all invisible cells regardless of
        batch/time -- so the polynomial evaluated there is deterministic
        and consistent, but it is whatever the learned weights say at
        that fixed point, not a separately-controlled "typical value"
        parameter the way v_mid is for the other architectures.

    Trained with the `mse+sorted` loss (LossSpec(sorted_weight=1.0)) by
    design choice, not `uniform` -- see SESSION_CHANGELOG.md: of 19
    model/batch combinations visually inspected, `ace_only` (this same
    loss) was the only one that produced a substantially non-constant
    prediction; every architecture tried under plain `uniform` MSE
    collapsed to a near-constant output regardless of parameter count.
    """

    def __init__(self, config):
        super().__init__()
        self.degree = 3
        self.n_raw_inputs = 6  # 3 AIA (center +/-1) + 3 HMI (center +/-1)

        self.register_buffer(
            "aia_mean", torch.tensor(float(config.aia_global_mean), dtype=torch.float32),
        )
        self.register_buffer(
            "aia_std", torch.tensor(float(config.aia_global_std), dtype=torch.float32),
        )
        self.register_buffer(
            "hmi_mean", torch.tensor(float(config.hmi_global_mean), dtype=torch.float32),
        )
        self.register_buffer(
            "hmi_std", torch.tensor(float(config.hmi_global_std), dtype=torch.float32),
        )

        # All monomial index-tuples of degree 0..3 over the 6 input
        # variables. A tuple's length is its degree; repeated indices are
        # powers (e.g. (0,0,1) -> x0^2 * x1). combinations_with_replacement
        # over range(6) enumerates every mixed/cross term, not just pure
        # powers of one variable.
        terms = [()]
        for d in range(1, self.degree + 1):
            terms += list(itertools.combinations_with_replacement(range(self.n_raw_inputs), d))
        self._terms = terms
        assert len(terms) == 84, f"expected 84 monomials, got {len(terms)}"

        self.weights = nn.Parameter(torch.zeros(len(terms)))
        # Init: only the constant term nonzero, at a typical solar wind
        # speed (400 km/s) -- starts as a flat prediction safely inside
        # the [100,900] boundary clamp, same "start safe" rationale as
        # every other model's init here, without adding a dedicated
        # squashing parameter (which was explicitly ruled out).
        with torch.no_grad():
            self.weights[terms.index(())] = 400.0

    def forward(self, image_slice):
        if image_slice is None:
            raise ValueError("Image slice tensor required")
        if image_slice.ndim != 3:
            raise ValueError(f"Expected [C,Nphi,Nt], got {image_slice.shape}")

        aia = image_slice[0]
        hmi = image_slice[2] if image_slice.shape[0] >= 3 else torch.zeros_like(aia)
        valid_mask = image_slice[1] > 0.5 if image_slice.shape[0] >= 2 else torch.isfinite(aia)
        mask_f = valid_mask.float()

        aia = torch.nan_to_num(aia, nan=0.0, posinf=0.0, neginf=0.0) * mask_f
        hmi = torch.nan_to_num(hmi, nan=0.0, posinf=0.0, neginf=0.0) * mask_f

        aia_norm = (aia - self.aia_mean) / torch.clamp(self.aia_std, min=1e-6)
        hmi_norm = (hmi - self.hmi_mean) / torch.clamp(self.hmi_std, min=1e-6)

        feats = [torch.roll(aia_norm, shifts=s, dims=0) for s in (-1, 0, 1)]
        feats += [torch.roll(hmi_norm, shifts=s, dims=0) for s in (-1, 0, 1)]
        x = torch.stack(feats, dim=-1)  # [Nphi, Nt, 6]

        v_physical = torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)
        for w, term in zip(self.weights, self._terms):
            if len(term) == 0:
                v_physical = v_physical + w
                continue
            val = x[..., term[0]]
            for idx in term[1:]:
                val = val * x[..., idx]
            v_physical = v_physical + w * val

        return v_physical


class ShallowReluVelocityModel(nn.Module):
    """
    Pointwise-in-time AIA/HMI -> speed mapping via a genuine (nonlinear)
    shallow network, as the direct counterpart to PolynomialVelocityModel:
    narrow-stencil raw inputs (AIA center +/-radius, HMI center +/-radius
    -- a local stencil, not the wide non-local windows Contextual/
    MultiFeature use), SAME intended loss (`mse+sorted`), but a single
    ReLU hidden layer instead of a fixed-degree polynomial:

        h = ReLU(Linear(n_inputs -> 100))
        raw = Linear(100 -> 1)(h)
        v = v_mid + v_halfspan * tanh(raw)

    n_inputs = 2*(2*radius+1): radius=1 (default) -> 6 inputs, ~803
    params; radius=2 -> 10 inputs, ~1203 params. Widening the radius
    (while staying a LOCAL stencil, not the wide non-local window) is a
    separate axis from Contextual/MultiFeature's non-local features --
    isolates "does a bit more local context help" from "does genuinely
    non-local information help". Unlike PolynomialVelocityModel this DOES
    use the bounded output rescale -- specifically the entry-13 FIXED form
    (v_mid/v_halfspan themselves learned) rather than the older
    fixed-range `v_min + v_span*sigmoid(raw)` form the original
    ContextualVelocityMLP (v1) used. That older form is exactly what
    caused v1's collapse (raw output stuck near 0, deep in the sigmoid's
    near-linear region, never growing) -- see SESSION_CHANGELOG.md entries
    12/13. Far-side (invisible) cells fall back to v_mid, same convention
    as Contextual/MultiFeature.
    """

    def __init__(self, config):
        super().__init__()
        hidden = int(getattr(config, "shallow_relu_hidden", 100))
        self.radius = int(getattr(config, "shallow_relu_radius", 1))
        n_inputs = 2 * (2 * self.radius + 1)

        self.register_buffer(
            "aia_mean", torch.tensor(float(config.aia_global_mean), dtype=torch.float32),
        )
        self.register_buffer(
            "aia_std", torch.tensor(float(config.aia_global_std), dtype=torch.float32),
        )
        self.register_buffer(
            "hmi_mean", torch.tensor(float(config.hmi_global_mean), dtype=torch.float32),
        )
        self.register_buffer(
            "hmi_std", torch.tensor(float(config.hmi_global_std), dtype=torch.float32),
        )

        self.head = nn.Sequential(
            nn.Linear(n_inputs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        self._v_min = 150.0
        self._v_span = 600.0
        self.raw_v_mid = nn.Parameter(torch.tensor(0.0))       # sigmoid(0)=0.5 -> 450
        self.raw_v_halfspan = nn.Parameter(torch.tensor(0.0))  # sigmoid(0)=0.5 -> 150

    def forward(self, image_slice):
        if image_slice is None:
            raise ValueError("Image slice tensor required")
        if image_slice.ndim != 3:
            raise ValueError(f"Expected [C,Nphi,Nt], got {image_slice.shape}")

        aia = image_slice[0]
        hmi = image_slice[2] if image_slice.shape[0] >= 3 else torch.zeros_like(aia)
        valid_mask = image_slice[1] > 0.5 if image_slice.shape[0] >= 2 else torch.isfinite(aia)
        mask_f = valid_mask.float()

        aia = torch.nan_to_num(aia, nan=0.0, posinf=0.0, neginf=0.0) * mask_f
        hmi = torch.nan_to_num(hmi, nan=0.0, posinf=0.0, neginf=0.0) * mask_f

        aia_norm = (aia - self.aia_mean) / torch.clamp(self.aia_std, min=1e-6)
        hmi_norm = (hmi - self.hmi_mean) / torch.clamp(self.hmi_std, min=1e-6)

        feats = [torch.roll(aia_norm, shifts=s, dims=0) for s in range(-self.radius, self.radius + 1)]
        feats += [torch.roll(hmi_norm, shifts=s, dims=0) for s in range(-self.radius, self.radius + 1)]
        x = torch.stack(feats, dim=-1)  # [Nphi, Nt, n_inputs]

        v_raw = self.head(x).squeeze(-1)

        v_mid = self._v_min + self._v_span * torch.sigmoid(self.raw_v_mid)
        v_halfspan = 300.0 * torch.sigmoid(self.raw_v_halfspan)
        v_physical = torch.clamp(
            v_mid + v_halfspan * torch.tanh(v_raw),
            min=self._v_min, max=self._v_min + self._v_span,
        )
        v_physical = torch.where(valid_mask, v_physical, v_mid.expand_as(v_physical))
        return v_physical


class MergedTemporalVelocityModel(nn.Module):
    """
    Merges relu100r2's local stencil (AIA/HMI at center +/-2, 10 raw
    inputs) with ContextualVelocityMLP's 4 features (I_center, I_wide,
    B_center, U_wide) -- 14 total -- THEN duplicates the whole feature
    vector with a one-timestep-earlier (t-1) copy, giving 28 inputs.

    This is the FIRST NN1 architecture in this project to use temporal
    context. Every prior architecture (IntensityToVelocityMLP,
    ThresholdVelocityModel, ContextualVelocityMLP, MultiFeatureThreshold-
    Model, PolynomialVelocityModel, ShallowReluVelocityModel) was strictly
    pointwise in time, per the user's own early-session instruction ("no
    temporal context is needed... the NN needs to learn how to map
    bright/dim to fast/slow"). Adding the t-1 copy is a deliberate,
    explicitly-confirmed reversal of that, to test whether frame-to-frame
    change carries information a single snapshot doesn't.

        x_t   = [AIA(-2..+2), HMI(-2..+2), I_center, I_wide, B_center, U_wide]  (14)
        x_tm1 = the SAME 14 features, evaluated one model timestep (dt)
                earlier (one column to the left in the image arrays)
        x = concat(x_t, x_tm1)                                              (28)
        h = ReLU(Linear(28 -> 8))
        v = v_mid + v_halfspan * tanh(Linear(8 -> 1)(h))

    241 params (232 in the first linear layer, 9 in the second; v_mid/
    v_halfspan are hardcoded constants, not learned -- see their
    definition below) -- single hidden layer, not two: with this many
    inputs at a small param budget, a second layer would force an even
    tighter first-layer bottleneck (28->~4) before any nonlinearity
    happens twice, likely worse at combining 28 inputs than one layer
    doing the compression once.

    Implementation note: x_tm1 is computed by rolling the FULL computed
    x_t feature stack by one step along the time axis, not by
    re-deriving each feature from a separately time-shifted image. This
    is exact (not an approximation): every feature here is built by
    combining `torch.roll`s along the LONGITUDE axis (and the wide-window
    mask-aware means) with the AIA/HMI arrays -- operations that commute
    with an independent shift along the TIME axis, since they act on
    different axes. Verified by direct comparison against recomputing
    from a manually time-shifted input (see unit test at introduction).
    One edge effect: `torch.roll` wraps circularly, so the very first
    timestep's x_tm1 spuriously pulls in the batch's LAST timestep rather
    than having no t-1 -- affects 1 of ~15,000-30,000 timesteps per
    batch, deemed negligible rather than special-cased.

    Far-side (invisible) cells fall back to v_mid, same convention as
    Contextual/MultiFeature/ShallowRelu.
    """

    def __init__(self, config):
        super().__init__()
        self.stencil_radius = 2
        self.window = int(getattr(config, "contextual_window_radius", 15))
        hidden = int(getattr(config, "merged_temporal_hidden", 8))
        self.channels = str(getattr(config, "merged_temporal_channels", "both"))
        if self.channels not in ("both", "hmi_only"):
            raise ValueError(f"Unknown merged_temporal_channels: {self.channels!r}")

        self.register_buffer(
            "hmi_mean", torch.tensor(float(config.hmi_global_mean), dtype=torch.float32),
        )
        self.register_buffer(
            "hmi_std", torch.tensor(float(config.hmi_global_std), dtype=torch.float32),
        )
        self.aia_mode, aia_mean, aia_std, aia_clip_dn, self.aia_a = _build_aia_norm_spec(config)
        self.register_buffer("aia_mean", torch.tensor(aia_mean, dtype=torch.float32))
        self.register_buffer("aia_std", torch.tensor(aia_std, dtype=torch.float32))
        self.register_buffer("aia_clip_dn", torch.tensor(aia_clip_dn, dtype=torch.float32))

        # "both": AIA stencil + HMI stencil + 4 wide/center features = 14/timestep.
        # "hmi_only": HMI stencil + B_center + U_wide = 7/timestep -- an AIA
        # ablation to gauge how much of the model's skill comes from AIA at all.
        n_per_t = 7 if self.channels == "hmi_only" else 14
        n_inputs = 2 * n_per_t  # 2 timesteps
        self.head = nn.Sequential(
            nn.Linear(n_inputs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        self._v_min = 150.0
        self._v_span = 600.0
        # v_mid/v_halfspan are HARDCODED from ACE speed statistics over the
        # entire 2018-2024 dataset (all 4 batches, N=47,561 hourly obs from
        # data/preprocessed_window/multi_spacecraft_obs_...csv): mean=409.8
        # km/s, min=249.4, max=1087.2 -> halfrange=(max-min)/2=418.9. NOT
        # learned: letting these start at the arbitrary midpoint (450/150)
        # and grow via gradient descent made amp_ratio (std(pred)/std(obs))
        # climb far too slowly -- at lr=1e-3 several folds were still stuck
        # at amp_ratio 0.2-0.4 after 100+ epochs. Fixing them at the true
        # data mean/range from step 0 removes that bottleneck entirely: the
        # optimizer only has to learn v_raw's SHAPE (which input maps to
        # which point in [150,750]), not also discover the right center/
        # scale from scratch. The final clamp to [_v_min, _v_min+_v_span]
        # still protects against the halfrange's mean-min/max-mean
        # asymmetry pushing outside the physical envelope.
        self.register_buffer("v_mid", torch.tensor(409.8))
        self.register_buffer("v_halfspan", torch.tensor(418.9))

        # Milestone 6 (global-SH ablation): OPTIONAL low-order global-
        # magnetic-state features, added on top of the above WITHOUT
        # changing anything else -- same hidden width, same head depth,
        # same local features, same t/t-1 temporal construction. Disabled
        # (use_global_sh=False) reproduces the exact prior model bit-for-
        # bit, since sh_l_max defaults to None and every code path below
        # that depends on it is skipped entirely.
        #
        # config.merged_temporal_sh_l_max in {None, 1, 2, 3, 5}. Coefficients
        # come from preprocess_hmi_sh.py: real SH, l=0..L_max, ALL m, l=0
        # (monopole) RETAINED -- N_coef = (L_max+1)**2 (4/9/16/36), fit
        # per-frame-amplitude-normalized with lambda=100 degree-damping
        # ridge regularization (see global_field_sh.py / milestones 1-5).
        # Raw (un-rotated) coefficients arrive via forward()'s sh_coeffs_raw
        # argument, shape [N_coef, Nt] -- rotated per NN-evaluation
        # longitude phi0 HERE (not at preprocessing time), via the exact
        # verified rotate_coeffs_z transform, reimplemented in torch and
        # vectorized over the whole (Nphi, Nt) grid at once.
        sh_l_max = getattr(config, "merged_temporal_sh_l_max", None)
        self.use_global_sh = sh_l_max is not None
        if self.use_global_sh:
            self.sh_l_max = int(sh_l_max)
            self.sh_idx = sh_index_list(self.sh_l_max)
            n_coef = len(self.sh_idx)
            self.n_sh_coef = n_coef
            Nphi = int(config.Nphi)
            phi_grid = torch.linspace(0.0, 2 * torch.pi, Nphi + 1)[:-1]
            self.register_buffer("phi_grid", phi_grid)
            n_inputs += 2 * n_coef  # t and t-1 global features, same convention as local x_t/x_tm1
            self.head = nn.Sequential(
                nn.Linear(n_inputs, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )
            print(f"MergedTemporalVelocityModel: global SH enabled, L_max={self.sh_l_max}, "
                  f"N_coef={n_coef} (l=0 retained), n_inputs={n_inputs} (was {n_inputs - 2*n_coef})")

    def _rotate_sh_to_phi0(self, coeffs_raw):
        """coeffs_raw: [N_coef, Nt] (raw, un-rotated). Returns [Nphi, Nt, N_coef],
        the coefficients re-expressed relative to each grid longitude phi0
        (phi0 -> 0), via the SAME formula verified in
        global_field_sh.rotate_coeffs_z / verify_sh_fit.py:
            a'_+m =  a_+m * cos(m*phi0) + a_-m * sin(m*phi0)
            a'_-m = -a_+m * sin(m*phi0) + a_+m... (see below, m=0 invariant)
        Vectorized over the whole (Nphi, Nt) grid at once."""
        Nt = coeffs_raw.shape[1]
        Nphi = self.phi_grid.shape[0]
        out = coeffs_raw.unsqueeze(0).expand(Nphi, -1, -1).clone()  # [Nphi, N_coef, Nt]
        pos = {(l, m): j for j, (l, m) in enumerate(self.sh_idx)}
        for l in range(self.sh_l_max + 1):
            for m in range(1, l + 1):
                jp, jn = pos[(l, m)], pos[(l, -m)]
                a, b = coeffs_raw[jp], coeffs_raw[jn]  # [Nt]
                c = torch.cos(m * self.phi_grid).unsqueeze(1)  # [Nphi,1]
                s = torch.sin(m * self.phi_grid).unsqueeze(1)  # [Nphi,1]
                out[:, jp, :] = c * a.unsqueeze(0) + s * b.unsqueeze(0)
                out[:, jn, :] = -s * a.unsqueeze(0) + c * b.unsqueeze(0)
        return out.permute(0, 2, 1)  # [Nphi, Nt, N_coef]

    def _compute_x_t(self, image_slice):
        """The base features at time t, stacked as [Nphi, Nt, 14] ("both")
        or [Nphi, Nt, 7] ("hmi_only")."""
        aia_norm_spec = (self.aia_mode, self.aia_mean, self.aia_std, self.aia_clip_dn, self.aia_a)
        I_center, I_wide, B_center, U_wide, valid_mask = _compute_contextual_features(
            image_slice, self.hmi_mean, self.hmi_std, self.window, aia_norm_spec,
        )
        hmi_norm = B_center
        hmi_stencil = [torch.roll(hmi_norm, shifts=s, dims=0) for s in range(-self.stencil_radius, self.stencil_radius + 1)]
        if self.channels == "hmi_only":
            feats = hmi_stencil + [B_center, U_wide]
        else:
            aia_norm = I_center  # _compute_contextual_features already returns the normalized center pixel
            aia_stencil = [torch.roll(aia_norm, shifts=s, dims=0) for s in range(-self.stencil_radius, self.stencil_radius + 1)]
            feats = aia_stencil + hmi_stencil + [I_center, I_wide, B_center, U_wide]
        x_t = torch.stack(feats, dim=-1)
        return x_t, valid_mask

    def forward(self, image_slice, sh_coeffs_raw=None):
        if image_slice is None:
            raise ValueError("Image slice tensor required")
        if image_slice.ndim != 3:
            raise ValueError(f"Expected [C,Nphi,Nt], got {image_slice.shape}")

        x_t, valid_mask = self._compute_x_t(image_slice)
        x_tm1 = torch.roll(x_t, shifts=1, dims=1)  # one timestep earlier, along the Nt axis
        x = torch.cat([x_t, x_tm1], dim=-1)  # [Nphi, Nt, 28]

        if self.use_global_sh:
            if sh_coeffs_raw is None:
                raise ValueError("use_global_sh=True but no sh_coeffs_raw was provided")
            if sh_coeffs_raw.shape[0] != self.n_sh_coef:
                raise ValueError(f"sh_coeffs_raw has {sh_coeffs_raw.shape[0]} coefficients, expected {self.n_sh_coef}")
            g_t = self._rotate_sh_to_phi0(sh_coeffs_raw)  # [Nphi, Nt, n_coef]
            g_tm1 = torch.roll(g_t, shifts=1, dims=1)  # same t-1 convention as the local features
            x = torch.cat([x, g_t, g_tm1], dim=-1)

        v_raw = self.head(x).squeeze(-1)

        v_physical = torch.clamp(
            self.v_mid + self.v_halfspan * torch.tanh(v_raw),
            min=self._v_min, max=self._v_min + self._v_span,
        )
        v_physical = torch.where(valid_mask, v_physical, self.v_mid.expand_as(v_physical))
        return v_physical


def save_boundary_weights(
    boundary,
    batch_path,
    optimizer_state_dict=None,
    scheduler_state_dict=None,
    forward_extra_state_dict=None,
):
    """Save model, optimizer, and scheduler checkpoints for a boundary batch."""
    import torch

    saved_paths = []

    weights_path = str(batch_path).replace('.npz', '_weights.pt')
    torch.save(boundary.state_dict(), weights_path)
    saved_paths.append(weights_path)

    if optimizer_state_dict is not None:
        optimizer_path = str(batch_path).replace('.npz', '_optimizer.pt')
        torch.save(optimizer_state_dict, optimizer_path)
        saved_paths.append(optimizer_path)

    if scheduler_state_dict is not None:
        scheduler_path = str(batch_path).replace('.npz', '_scheduler.pt')
        torch.save(scheduler_state_dict, scheduler_path)
        saved_paths.append(scheduler_path)

    if forward_extra_state_dict is not None:
        forward_extra_path = str(batch_path).replace('.npz', '_forward_extra.pt')
        torch.save(forward_extra_state_dict, forward_extra_path)
        saved_paths.append(forward_extra_path)

    print(f"Saved boundary checkpoint files: {', '.join(saved_paths)}")


def load_saved_boundary_guess(config, device, fallback_dir=None, batch_idx=None):
    """Load saved boundary guess from .npy for finetuning/resume workflows."""
    empty_result = (None, None, None, None, None, None, None, None)

    if not bool(getattr(config, 'resume_from_saved_boundary', False)):
        return empty_result

    def _weights_path_from_npz(npz_path: Path) -> Path:
        return npz_path.with_name(f"{npz_path.stem}_weights.pt")

    selected_path = None
    explicit_path = getattr(config, 'saved_boundary_path', None)
    if explicit_path:
        p = Path(explicit_path)
        if p.exists() and p.is_file():
            selected_path = p
        else:
            print(f"Resume requested, but saved_boundary_path not found: {p}")
            return empty_result
    else:
        search_dir_value = getattr(config, 'saved_boundary_dir', None)
        search_dir = Path(search_dir_value) if search_dir_value else (fallback_dir if fallback_dir is not None else Path("."))
        pattern = getattr(config, 'saved_boundary_pattern', 'batch_*_boundary.npz')
        if not search_dir.exists():
            print(f"Resume requested, but saved boundary directory not found: {search_dir}")
            return empty_result

        if batch_idx is not None:
            if "*" in pattern:
                expected_name = pattern.replace("*", f"{batch_idx:03d}")
            else:
                expected_name = f"batch_{batch_idx:03d}_boundary.npz"
            expected_path = search_dir / expected_name
            if expected_path.exists() and expected_path.is_file():
                selected_path = expected_path
            else:
                print(
                    f"Resume requested, but no saved boundary for batch {batch_idx} at {expected_path}"
                )
                return empty_result
        else:
            candidates = sorted(search_dir.glob(pattern))
            if not candidates:
                alt = search_dir / "inferred_boundary.npy"
                if alt.exists() and alt.is_file():
                    selected_path = alt
                else:
                    print(f"Resume requested, but no saved boundary files found in {search_dir} (pattern={pattern})")
                    return empty_result
            else:
                selected_path = candidates[-1]

    loaded = np.load(selected_path, allow_pickle=False)
    # Support both .npz (dict-like) and legacy .npy (plain array)
    saved_v0 = None
    saved_state_dict = None
    saved_optimizer_state = None
    saved_scheduler_state = None
    saved_forward_extra_state = None
    saved_meta = {}
    weights_path = _weights_path_from_npz(selected_path)
    if weights_path.exists() and weights_path.is_file():
        saved_state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        if isinstance(saved_state_dict, dict):
            lift_key = 'lifting_layer.weight'
            if lift_key in saved_state_dict:
                lift_weight = saved_state_dict[lift_key]
                if isinstance(lift_weight, torch.Tensor) and lift_weight.ndim == 4 and lift_weight.shape[1] == 1:
                    upgraded = torch.zeros(
                        lift_weight.shape[0],
                        2,
                        lift_weight.shape[2],
                        lift_weight.shape[3],
                        dtype=lift_weight.dtype,
                        device=lift_weight.device,
                    )
                    upgraded[:, 0, :, :] = lift_weight[:, 0, :, :]
                    saved_state_dict[lift_key] = upgraded
                    print(
                        f"Upgraded legacy FNO lifting layer weights from 1 to 2 input channels for resume from {weights_path}"
                    )
            mask_key = 'corrected_mask'
            if mask_key in saved_state_dict:
                saved_mask = saved_state_dict[mask_key]
                if isinstance(saved_mask, torch.Tensor) and tuple(saved_mask.shape) != (config.Nphi, config.Nt):
                    del saved_state_dict[mask_key]
                    print(
                        f"Dropped stale corrected_mask buffer with shape {tuple(saved_mask.shape)} when resuming {weights_path}; expected {(config.Nphi, config.Nt)}"
                    )

    optimizer_path = selected_path.with_name(f"{selected_path.stem}_optimizer.pt")
    scheduler_path = selected_path.with_name(f"{selected_path.stem}_scheduler.pt")
    forward_extra_path = selected_path.with_name(f"{selected_path.stem}_forward_extra.pt")
    if optimizer_path.exists() and optimizer_path.is_file():
        saved_optimizer_state = torch.load(optimizer_path, map_location=device, weights_only=True)
    if scheduler_path.exists() and scheduler_path.is_file():
        saved_scheduler_state = torch.load(scheduler_path, map_location=device, weights_only=True)
    if forward_extra_path.exists() and forward_extra_path.is_file():
        saved_forward_extra_state = torch.load(forward_extra_path, map_location=device, weights_only=True)

    if not isinstance(loaded, np.ndarray):
        for key in ['epoch', 'loss', 'max_ae', 'rmse']:
            if key in loaded:
                value = loaded[key]
                try:
                    saved_meta[key] = value.item() if hasattr(value, 'item') else value
                except Exception:
                    saved_meta[key] = value

    if saved_state_dict is None:
        print(
            f"Resume requested in FNO mode, but weights file not found: {weights_path}. "
            "Skipping resume."
        )
        return None, None, None, None, None, None, None, saved_meta

    if not isinstance(loaded, np.ndarray) and 'v0' in loaded:
        saved_v0 = torch.tensor(loaded['v0'], dtype=torch.float32, device=device)
    if isinstance(loaded, np.ndarray):
        arr = loaded
    else:
        arr = loaded['v_inner'] if 'v_inner' in loaded else loaded[loaded.files[0]]
        if 'v0' in loaded:
            saved_v0 = torch.tensor(loaded['v0'], dtype=torch.float32, device=device)
    if arr.ndim != 2:
        print(f"Saved boundary has invalid ndim={arr.ndim} in {selected_path}; expected 2D array")
        return None, None, None, None, None, None, None, saved_meta

    # Support accidental transpose if shape is (Nt, Nphi)
    if arr.shape == (config.Nt, config.Nphi):
        arr = arr.T

    if arr.shape[0] != config.Nphi:
        if arr.shape[0] > 1:
            saved_nphi = arr.shape[0]
            phi_saved = np.linspace(0.0, 360.0, saved_nphi, endpoint=False)
            phi_target = np.linspace(0.0, 360.0, config.Nphi, endpoint=False)
            arr_interp = np.full((config.Nphi, arr.shape[1]), np.nan, dtype=arr.dtype)

            for col in range(arr.shape[1]):
                col_vals = arr[:, col]
                valid = np.isfinite(col_vals)
                if np.sum(valid) == 0:
                    continue
                if np.sum(valid) == 1:
                    arr_interp[:, col] = float(col_vals[valid][0])
                    continue

                phi_valid = phi_saved[valid]
                vals_valid = col_vals[valid]
                phi_ext = np.concatenate([phi_valid - 360.0, phi_valid, phi_valid + 360.0])
                vals_ext = np.concatenate([vals_valid, vals_valid, vals_valid])
                arr_interp[:, col] = np.interp(phi_target, phi_ext, vals_ext)

            arr = arr_interp
            print(
                f"Resampled saved boundary from Nphi={saved_nphi} to Nphi={config.Nphi} "
                f"for resume from {selected_path}"
            )
        else:
            print(f"Saved boundary has incompatible phi dimension {arr.shape[0]} (expected {config.Nphi}) in {selected_path}")
            return None, None, None, None, None, None, None, saved_meta

    if arr.shape[1] < config.Nt:
        pad = np.repeat(arr[:, -1:], config.Nt - arr.shape[1], axis=1)
        arr = np.concatenate([arr, pad], axis=1)
    elif arr.shape[1] > config.Nt:
        arr = arr[:, :config.Nt]

    # Saved outputs may encode non-informative cells as NaN for visualization.
    # Replace NaNs so resume mode still has a valid numerical initial guess.
    if np.isnan(arr).any():
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            col_mean = np.nanmean(arr, axis=0, keepdims=True)
            global_mean = np.nanmean(arr)
        if not np.isfinite(global_mean):
            global_mean = 400.0
        col_mean = np.where(np.isfinite(col_mean), col_mean, global_mean)
        nan_idx = np.isnan(arr)
        if np.any(nan_idx):
            arr[nan_idx] = np.take(col_mean[0], np.where(nan_idx)[1])
        arr = np.clip(arr, 150.0, 750.0)

    guess = torch.tensor(arr, dtype=torch.float32, device=device)
    return guess, saved_v0, str(selected_path), saved_state_dict, saved_optimizer_state, saved_scheduler_state, saved_forward_extra_state, saved_meta


# Groups of the 37 per-cell inputs of FeatureVelocityMLP in the drop-HMI layout with lags 12/24/48/72 h (2026-10-19):
# each AIA feature with its four lagged copies, log10 f_s, recurrence + its validity flag. Lag-presence flags: LAG_FLAGS.
INPUT_GROUPS = {f'a{k}': [k, 6 + k, 13 + k, 20 + k, 27 + k] for k in range(6)}
INPUT_GROUPS.update({'fs': [34], 'rec': [35, 36]})
GROUP_NAMES = list(INPUT_GROUPS)
LAG_FLAGS = [12, 19, 26, 33]


class HingeHead(nn.Module):
    """Piecewise-linear map with two learnable breakpoints per quantity (2026-10-21; worklog INTERPRETABLE_BOUNDARY §10).

    logit = lin(x) + sum_k [ b_k1 h(x_k - c_k1) + b_k2 h(x_k - c_k2) ] * gate_k, with h(u) = softplus(beta u) / beta a
    smooth hinge (transition width ~1/beta standardized units). Breakpoints are kept ordered and inside [lo_k, hi_k]
    (input percentiles 2 and 98): c1 = lo + (hi - lo) sigmoid(t1), c2 = c1 + (hi - c1) sigmoid(t2); two breakpoints that
    meet act as one with the summed slope change (merged). Slope changes start at zero (the model starts linear);
    breakpoints start at init1, init2 (percentiles 25 and 75). gate_k: index of a validity column whose standardized
    value is > 0 where the quantity is valid (the hinge terms of quantity k act only there), or -1 for none.
    Specification in buffers, so checkpoints rebuild themselves."""

    def __init__(self, d_in, quant, lo, hi, init1, init2, gate=None, beta=10.0):
        super().__init__()
        n = len(quant)
        t = lambda v: torch.tensor(v, dtype=torch.float32)
        self.register_buffer('hinge_quant', torch.tensor(list(quant), dtype=torch.long))
        self.register_buffer('hinge_gate', torch.tensor(list(gate) if gate is not None else [-1] * n, dtype=torch.long))
        self.register_buffer('hinge_lo', t(lo)); self.register_buffer('hinge_hi', t(hi))
        self.register_buffer('hinge_beta', t(float(beta)))
        lo_, hi_, i1, i2 = t(lo), t(hi), t(init1), t(init2)
        f1 = ((i1 - lo_) / (hi_ - lo_)).clamp(0.02, 0.98)
        f2 = ((i2 - i1) / (hi_ - i1)).clamp(0.02, 0.98)
        self.t1 = nn.Parameter(torch.log(f1 / (1 - f1)))
        self.t2 = nn.Parameter(torch.log(f2 / (1 - f2)))
        self.b1 = nn.Parameter(torch.zeros(n)); self.b2 = nn.Parameter(torch.zeros(n))
        self.lin = nn.Linear(d_in, 1)

    def breakpoints(self):
        c1 = self.hinge_lo + (self.hinge_hi - self.hinge_lo) * torch.sigmoid(self.t1)
        c2 = c1 + (self.hinge_hi - c1) * torch.sigmoid(self.t2)
        return c1, c2

    def hinge_terms(self, x):
        """[..., n] hinge contribution of each quantity (logit scale), gated."""
        c1, c2 = self.breakpoints()
        xq = x[..., self.hinge_quant]
        beta = float(self.hinge_beta)
        h = lambda u: nn.functional.softplus(beta * u) / beta
        out = self.b1 * h(xq - c1) + self.b2 * h(xq - c2)
        g = self.hinge_gate
        if (g >= 0).any():
            gate = torch.where(g >= 0, (x[..., g.clamp(min=0)] > 0).to(out.dtype), torch.ones_like(out))
            out = out * gate
        return out

    def forward(self, x):
        return self.lin(x) + self.hinge_terms(x).sum(-1, keepdim=True)


class ReducedInputsHead(nn.Module):
    """One input per physical quantity (2026-10-20, after the group lasso): the 37-input cell vector is reduced to
    the selected groups, each AIA feature replaced by its mean over the current image and the lagged images that
    exist (lag copies are zero where missing), f_s as is, recurrence value and validity flag, plus the four
    lag-presence flags; then an inner head on the reduced vector: a linear map (inner.0) or a NAM with one curve per
    quantity and no pairs, whose curvature can be penalized (penalty(); boundary.NAMHead.curvature_penalty).
    Specification in buffers (red_groups = codes into GROUP_NAMES), so checkpoints rebuild themselves."""

    def __init__(self, groups, inner='linear', nam_hidden=(16, 8), smooth=0.0, hinge=None):
        super().__init__()
        self.register_buffer('red_groups', torch.tensor([GROUP_NAMES.index(g) for g in groups], dtype=torch.long))
        cols, quant = [], []
        for g in groups:
            if len(INPUT_GROUPS[g]) == 5:
                quant.append(len(cols)); cols.append(('mean', INPUT_GROUPS[g]))
            elif g == 'rec':
                quant.append(len(cols)); cols.append(('raw', [35])); cols.append(('raw', [36]))
            else:
                quant.append(len(cols)); cols.append(('raw', INPUT_GROUPS[g]))
        cols += [('raw', [f]) for f in LAG_FLAGS]
        self.cols = cols
        self.names = [f'{g}' for g in groups]
        d = len(cols)
        if inner == 'linear':
            self.inner = nn.Sequential(nn.Linear(d, 1))
        elif inner == 'hinge':                   # hinge = dict(lo, hi, init1, init2) per quantity (order of groups)
            gate = [quant[i] + 1 if g == 'rec' else -1 for i, g in enumerate(groups)]   # rec flag follows rec
            self.inner = HingeHead(d, quant, hinge['lo'], hinge['hi'], hinge['init1'], hinge['init2'], gate)
        else:
            self.inner = NAMHead(d, quant, [], tuple(nam_hidden))
        self.register_buffer('smooth', torch.tensor(float(smooth)))

    def reduce(self, x):
        n = 1.0 + sum(x[..., f] for f in LAG_FLAGS)
        out = []
        for kind, idx in self.cols:
            out.append(sum(x[..., i] for i in idx) / n if kind == 'mean' else x[..., idx[0]])
        return torch.stack(out, -1)

    def forward(self, x):
        return self.inner(self.reduce(x))

    def penalty(self):
        if float(self.smooth) <= 0 or not isinstance(self.inner, NAMHead):
            return None
        return float(self.smooth) * self.inner.curvature_penalty()


class NAMHead(nn.Module):
    """Neural additive model with selected pairwise terms (GA²M style) for FeatureVelocityMLP (2026-10-18).

    Maps the per-cell input vector x [..., D] to a logit: a linear term in all D inputs, plus a small network curve
    f_i(x_i) (1 -> h1 -> h2 -> 1, GELU) for each input index in `key`, plus a small network surface f_ij(x_i, x_j)
    (2 -> h1 -> h2 -> 1) for each pair in `pairs`. All subnetworks of one kind are evaluated together (batched einsum).
    The specification is stored as buffers (nam_key, nam_pairs, nam_hidden), so checkpoints rebuild themselves
    (plot_best_checkpoint.load_fold). Components can be read out with .components(x) for shape plots.
    """

    def __init__(self, d_in, key, pairs, hidden=(32, 16)):
        super().__init__()
        h1, h2 = hidden
        self.register_buffer('nam_key', torch.tensor(list(key), dtype=torch.long))
        self.register_buffer('nam_pairs', torch.tensor(list(pairs), dtype=torch.long).view(-1, 2))
        self.register_buffer('nam_hidden', torch.tensor([h1, h2], dtype=torch.long))
        self.lin = nn.Linear(d_in, 1)

        def subnet(k, d):                                    # k parallel networks with d inputs each
            w = lambda *shape: nn.Parameter(torch.randn(*shape) * (2.0 / shape[-2]) ** 0.5)
            return nn.ParameterDict({'w1': w(k, d, h1), 'b1': nn.Parameter(torch.zeros(k, h1)),
                                     'w2': w(k, h1, h2), 'b2': nn.Parameter(torch.zeros(k, h2)),
                                     'w3': nn.Parameter(torch.zeros(k, h2, 1)), 'b3': nn.Parameter(torch.zeros(k, 1))})
        self.shape_nets = subnet(len(key), 1)
        self.pair_nets = subnet(max(len(pairs), 1), 2)        # zero-initialised output: starts as the linear model

    @staticmethod
    def _run(P, z):                                           # z [..., k, d] -> [..., k]
        a = nn.functional.gelu(torch.einsum('...kd,kdh->...kh', z, P['w1']) + P['b1'])
        a = nn.functional.gelu(torch.einsum('...kh,khg->...kg', a, P['w2']) + P['b2'])
        return (torch.einsum('...kg,kgo->...ko', a, P['w3']) + P['b3']).squeeze(-1)

    def components(self, x):
        """(linear [..., 1], curves [..., n_key], surfaces [..., n_pairs]) on the logit scale."""
        cur = self._run(self.shape_nets, x[..., self.nam_key].unsqueeze(-1))
        if self.nam_pairs.numel():
            pr = self._run(self.pair_nets, torch.stack([x[..., self.nam_pairs[:, 0]], x[..., self.nam_pairs[:, 1]]], -1))
        else:
            pr = x[..., :0]
        return self.lin(x), cur, pr

    def forward(self, x):
        lin, cur, pr = self.components(x)
        return lin + cur.sum(-1, keepdim=True) + pr.sum(-1, keepdim=True)

    def curvature_penalty(self, lo=-3.0, hi=3.0, n=61):
        """Mean squared second derivative of the curves over [lo, hi] (standardized inputs; finite differences)."""
        z = torch.linspace(lo, hi, n, device=self.lin.weight.device)
        h = (hi - lo) / (n - 1)
        f = self._run(self.shape_nets, z.view(n, 1, 1).expand(n, len(self.nam_key), 1))      # [n, k]
        d2 = (f[2:] - 2 * f[1:-1] + f[:-2]) / h ** 2
        return (d2 ** 2).mean()


class FeatureVelocityMLP(nn.Module):
    """Per-cell MLP on the set-2a feature dataset (nn1_inputs.py, PAPER_2.md).

    image_slice: [11, Nphi, Nt] = 8 features (CM-equivalent), dlon_center/60, AIA-valid mask, HMI-available flag.
    dlon_center is NOT a model input by default (config.feature_use_dlon=False): one universal mapping.
    Features are standardized with training-batch statistics (config.nn1_feature_mean/std); HMI features
    are zero (= mean) where HMI is unavailable and the flag says so. Optional longitude stencil of radius
    config.feature_mlp_radius (default 0: features already integrate over longitude strips).
    Cells without valid AIA input get a constant fill speed (no gradient): config.nn1_fill_speed if set
    (e.g. the training-set median, train_joint_forecast --nn1-fill train-median), else the output midpoint
    450 km/s as in the other NN1 models. Stored as a buffer, so checkpoints carry their own fill.
    Extra channels (config.nn1_extra_channels, e.g. ['pfss_log10fs']): one input each, standardized with
    config.nn1_extra_mean / std (training batches); stored as buffers with their registry codes.
    config.nn1_drop_hmi: use only the 6 AIA features (current and lagged) and no HMI flags.
    Output range: 150 .. config.nn1_v_max (default 750) km/s via a sigmoid; the bounds are a buffer too.
    Temporal context (config.nn1_lag_hours, default none): image_slice then carries, per lag L, 10 more
    channels = the same longitude's 8 features at hour t - L (CM-equivalent) + its AIA-valid and HMI flags
    (nn1_inputs.build_fold_batch_specs); each lag adds 9 inputs per cell (features, HMI flag, valid flag).
    """

    N_FEAT, CH_DLON, CH_VALID, CH_HMI = 8, 8, 9, 10

    def __init__(self, config):
        super().__init__()
        self.radius = int(getattr(config, 'feature_mlp_radius', 0))
        self.register_buffer('feat_mean', torch.tensor(config.nn1_feature_mean, dtype=torch.float32).view(-1, 1, 1))
        self.register_buffer('feat_std', torch.tensor(config.nn1_feature_std, dtype=torch.float32).view(-1, 1, 1))
        self.use_dlon = bool(getattr(config, 'feature_use_dlon', False))   # default: universal mapping, no offset input
        lags = [float(h) for h in (getattr(config, 'nn1_lag_hours', None) or ())]
        self.register_buffer('lag_hours', torch.tensor(lags, dtype=torch.float32))
        self.n_lags = len(lags)
        import nn1_inputs
        extras = list(getattr(config, 'nn1_extra_channels', None) or ())
        self.n_extra = len(extras)
        self.register_buffer('extra_codes', torch.tensor([nn1_inputs.EXTRA_NAMES.index(e) for e in extras], dtype=torch.long))
        self.register_buffer('extra_mean', torch.tensor(getattr(config, 'nn1_extra_mean', []) or [], dtype=torch.float32).view(-1, 1, 1))
        self.register_buffer('extra_std', torch.tensor(getattr(config, 'nn1_extra_std', []) or [], dtype=torch.float32).view(-1, 1, 1))
        self.drop_hmi = bool(getattr(config, 'nn1_drop_hmi', False))           # use AIA channels 0-5 only
        self.register_buffer('drop_hmi_flag', torch.tensor(int(self.drop_hmi)))
        n_cur = 6 if self.drop_hmi else self.N_FEAT + 1
        n_lag = 7 if self.drop_hmi else 10
        per_cell = n_cur + int(self.use_dlon) + n_lag * self.n_lags + self.n_extra
        dims = tuple(getattr(config, 'mlp_hidden_dims', (100, 64, 32)))
        layers, d = [], per_cell * (2 * self.radius + 1)
        p_drop = float(getattr(config, 'nn1_dropout', 0.0) or 0.0)            # dropout after each hidden layer
        self.register_buffer('dropout_p', torch.tensor(p_drop))
        for h in dims:
            layers += [nn.Linear(d, h), nn.GELU()] + ([nn.Dropout(p_drop)] if p_drop > 0 else [])
            d = h
        layers += [nn.Linear(d, 1)]
        self.head = nn.Sequential(*layers)
        if getattr(config, 'nn1_reduce', None):                              # one input per quantity (lag means)
            kind = {'nam': 'nam', 'hinge': 'hinge'}.get(getattr(config, 'nn1_head', 'mlp'), 'linear')
            self.head = ReducedInputsHead(config.nn1_reduce, kind, tuple(getattr(config, 'nam_hidden', (16, 8))),
                                          float(getattr(config, 'nam_smooth', 0.0)), getattr(config, 'hinge_spec', None))
        elif getattr(config, 'nn1_head', 'mlp') == 'nam':                   # additive model + pairs instead of the MLP
            self.head = NAMHead(per_cell * (2 * self.radius + 1), config.nam_key, config.nam_pairs,
                                tuple(getattr(config, 'nam_hidden', (32, 16))))
        self.v_min, self.v_max = 150.0, float(getattr(config, 'nn1_v_max', 750.0))
        self.register_buffer('v_bounds', torch.tensor([self.v_min, self.v_max]))      # output range, saved with the model
        fill = getattr(config, 'nn1_fill_speed', None)
        self.register_buffer('fill_speed', torch.tensor(0.5 * (self.v_min + self.v_max) if fill is None else float(fill)))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        state_dict.setdefault(prefix + 'fill_speed', torch.tensor(0.5 * (self.v_min + self.v_max)))   # pre-fill checkpoints: 450
        state_dict.setdefault(prefix + 'lag_hours', torch.zeros(0))                                    # pre-lag checkpoints: none
        state_dict.setdefault(prefix + 'extra_codes', torch.zeros(0, dtype=torch.long))                 # pre-extra checkpoints: none
        state_dict.setdefault(prefix + 'drop_hmi_flag', torch.tensor(0))
        state_dict.setdefault(prefix + 'dropout_p', torch.tensor(0.0))
        state_dict.setdefault(prefix + 'extra_mean', torch.zeros(0, 1, 1))
        state_dict.setdefault(prefix + 'extra_std', torch.zeros(0, 1, 1))
        state_dict.setdefault(prefix + 'v_bounds', torch.tensor([150.0, 750.0]))                        # pre-bounds checkpoints
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, image_slice):
        n_ch = 11 + 10 * self.n_lags + self.n_extra
        if image_slice is None or image_slice.ndim != 3 or image_slice.shape[0] != n_ch:
            raise ValueError(f'Expected [{n_ch}, Nphi, Nt], got {None if image_slice is None else tuple(image_slice.shape)}')
        x = torch.nan_to_num(image_slice, nan=0.0, posinf=0.0, neginf=0.0)
        valid = x[self.CH_VALID] > 0.5
        hmi_ok = x[self.CH_HMI]
        feats = (x[:self.N_FEAT] - self.feat_mean) / self.feat_std
        if self.drop_hmi:
            parts = [feats[:6]] + ([x[self.CH_DLON:self.CH_DLON + 1]] if self.use_dlon else [])
        else:
            feats = torch.cat([feats[:6], feats[6:8] * hmi_ok], dim=0)
            parts = [feats, hmi_ok.unsqueeze(0)] + ([x[self.CH_DLON:self.CH_DLON + 1]] if self.use_dlon else [])
        for i in range(self.n_lags):
            lag = x[11 + 10 * i: 21 + 10 * i]                                  # 8 features, AIA valid, HMI flag at t - L
            v_l, h_l = lag[8:9], lag[9:10] * lag[8:9]
            f_l = (lag[:8] - self.feat_mean) / self.feat_std
            if self.drop_hmi:
                parts += [f_l[:6] * v_l, v_l]
            else:
                f_l = torch.cat([f_l[:6], f_l[6:8] * h_l], dim=0) * v_l
                parts += [f_l, h_l, v_l]
        if self.n_extra:
            e0 = 11 + 10 * self.n_lags
            parts.append((x[e0:e0 + self.n_extra] - self.extra_mean) / self.extra_std)
        cell = torch.cat(parts, dim=0)
        cell = cell * valid.float()
        if self.radius > 0:
            cell = torch.cat([torch.roll(cell, shifts=s, dims=1) for s in range(-self.radius, self.radius + 1)], dim=0)
        v_raw = self.head(cell.permute(1, 2, 0)).squeeze(-1)                 # [Nphi, Nt]
        lo, hi = self.v_bounds[0], self.v_bounds[1]
        v = lo + (hi - lo) * torch.sigmoid(v_raw)
        return torch.where(valid, v, self.fill_speed.to(v.dtype).expand_as(v))
