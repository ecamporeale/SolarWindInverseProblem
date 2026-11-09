"""AIA PSF deconvolution with the Basic Iterative Deconvolution (BID; Hofmeister et al. 2024, Sol. Phys. 299, 77),
as used by PROSWIN (Collin et al. 2026) for experiment E2 (EXPERIMENT_E2_psf_deconvolution.md).

Port of deconvolve_bid (whole image, pad=True, large_psf=False, k = 1, at most 25 iterations, stop when no pixel
changes by more than 0.1 DN) from github.com/DanielCollin96/solar_image_processing,
psf_deconvolution/deconvolve_image.py (MIT licence, (c) Stefan Hofmeister / Daniel Collin), with the padded PSF's
FFT computed once (prepare) instead of per call; numerics otherwise identical (checked against the original:
checks/check_bid_port.py). PSF: aiapy.psf.psf(193 A) at 4096 px, rebinned to 1024 px with their rebin_psf
(data/psf/psf_193_1024.npy, experiments/e2_data/make_psf_193.py). Their pipeline clips negative output pixels to 0; so does deconvolve.
"""
import numpy as np
from scipy import fft

PSF_FILE = 'data/psf/psf_193_1024.npy'


def _pad(a, value):
    p = (0.25 * np.array(a.shape)).astype(int)
    return np.pad(a, ((p[0], p[0]), (p[1], p[1])), constant_values=value)


def prepare(psf):
    """FFT of the zero-padded, centre-rolled PSF (their pad_img_psf + roll + rfft2)."""
    psf = _pad(np.asarray(psf, np.float64), np.nan)
    psf[np.isnan(psf)] = 0.0
    psf = np.roll(np.roll(psf, psf.shape[0] // 2, axis=0), psf.shape[1] // 2, axis=1)
    return fft.rfft2(psf)


def bid(img, psf_fft, iterations=25, tolerance=0.1, k=1.0, workers=1):
    img = _pad(np.asarray(img, np.float64), np.nan)
    pad_mask = np.isnan(img)
    img[pad_mask] = 0.0
    dec = img.copy()
    for _ in range(iterations):
        last = dec.copy()
        con = fft.irfft2(fft.rfft2(dec, workers=workers) * psf_fft, s=img.shape, workers=workers)
        con[pad_mask] = 0.0
        dec -= k * (con - img)
        if np.max(np.abs(dec - last)) <= tolerance:
            break
    u = (np.array(dec.shape) / 6.0).astype(int)                       # their unpad_img_psf
    return dec[u[0]:dec.shape[0] - u[0], u[1]:dec.shape[1] - u[1]]


_PSF_FFT = None


def deconvolve(data):
    """Raw DN image (1024 x 1024) -> deconvolved DN, negatives clipped to 0. NaN pixels are treated as 0 (their maps
    have no NaN; ours only off-disk corners, if any)."""
    global _PSF_FFT
    if _PSF_FFT is None:
        _PSF_FFT = prepare(np.load(PSF_FILE))
    out = bid(np.nan_to_num(np.asarray(data, np.float64)), _PSF_FFT)
    out[out < 0] = 0.0
    return out
