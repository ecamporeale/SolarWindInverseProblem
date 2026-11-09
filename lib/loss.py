import torch
import torch.nn as nn
import numpy as np
from helpers import sample_satellite_track

# =========================================================
# STEP 7: LOSS FUNCTION
# =========================================================

class LossFunction(nn.Module):
    """Computes loss for inverse problem."""
    
    def __init__(self, config, forward_model, observed_data, v0, phi_obs_t=None, radius_obs_t=None, time_obs_t=None, obs_tracks=None):
        super().__init__()
        self.config = config
        self.forward_model = forward_model
        self.v0 = v0
        self.track_ids = []
        self.num_tracks = 0
        self.track_lon_plot_series = []
        self.track_obs_plot_series = []
        self._nan_debug_reported = set()
        self._loss_data = float('nan')
        self._loss_phi = float('nan')
        self._loss_t = float('nan')
        self.use_multispacecraft = bool(getattr(config, 'enable_multispacecraft_loss', False))
        self.constant_prior_value = float(getattr(config, 'warmup_prior_value', 400.0))
        self.constant_prior_weight = float(getattr(config, 'warmup_prior_weight', 1.0))
        
        # phi_obs_t: (Nt,) Carrington longitude of the spacecraft at each timestep
        if phi_obs_t is None:
            # Fallback: assume fixed at 0
            phi_obs_t = torch.zeros(config.Nt, dtype=torch.float32)

        self.register_buffer('v_obs', observed_data)
        self.register_buffer('phi_obs_t', phi_obs_t)
        if radius_obs_t is None:
            radius_obs_t = torch.full_like(observed_data, float(getattr(config, 'rAU', 215.0)))
        self.register_buffer('radius_obs_t', radius_obs_t)
        if time_obs_t is None:
            time_obs_t = torch.arange(observed_data.shape[0], dtype=torch.float32, device=observed_data.device) * float(config.dt)
        self.register_buffer('time_obs_t', time_obs_t)

        if self.use_multispacecraft and obs_tracks:
            for idx, track in enumerate(obs_tracks):
                v_obs_track = track['v_obs']
                phi_obs_track = track['phi_obs_t']
                radius_obs_track = track.get('radius_obs_t', None)
                if radius_obs_track is None:
                    radius_obs_track = torch.full_like(v_obs_track, float(getattr(config, 'rAU', 215.0)))
                time_obs_track = track.get('time_obs_seconds', None)
                if time_obs_track is None:
                    time_obs_track = np.arange(v_obs_track.shape[0], dtype=float) * float(config.dt)
                time_obs_track = torch.tensor(time_obs_track, dtype=torch.float32, device=v_obs_track.device)
                self.register_buffer(f'v_obs_track_{idx}', v_obs_track)
                self.register_buffer(f'phi_obs_track_{idx}', phi_obs_track)
                self.register_buffer(f'radius_obs_track_{idx}', radius_obs_track)
                self.register_buffer(f'time_obs_track_{idx}', time_obs_track)
                self.track_ids.append(str(track.get('id', f'TRACK_{idx}')))
                t_hourly = track.get('time_hourly', None)
                t_ballistic_hourly = track.get('time_ballistic_hourly', None)
                lon_hourly = track.get('lon_deg_hourly', None)
                lat_hourly = track.get('lat_deg_hourly', None)
                speed_hourly = track.get('speed_hourly', None)
                radius_hourly = track.get('radius_hourly', None)
                if t_hourly is not None and lon_hourly is not None:
                    track_entry = {
                        'id': self.track_ids[-1],
                        'time': np.asarray(t_ballistic_hourly if t_ballistic_hourly is not None else t_hourly),
                        'time_src': np.asarray(t_hourly),
                        'lon_deg': np.asarray(lon_hourly, dtype=float),
                    }
                    self.track_lon_plot_series.append(track_entry)
                if t_hourly is not None and lon_hourly is not None and speed_hourly is not None:
                    track_obs_entry = {
                        'id': self.track_ids[-1],
                        'time': np.asarray(t_hourly),
                        'lon_deg': np.asarray(lon_hourly, dtype=float),
                        'lat_deg': np.asarray(lat_hourly, dtype=float) if lat_hourly is not None else None,
                        'speed': np.asarray(speed_hourly, dtype=float),
                        'radius': np.asarray(radius_hourly, dtype=float) if radius_hourly is not None else None,
                    }
                    self.track_obs_plot_series.append(track_obs_entry)
            self.num_tracks = len(obs_tracks)

            # For single-track diagnostics, use the first provided track.
            if self.num_tracks > 0:
                primary_idx = 0
                self.v_obs = getattr(self, f'v_obs_track_{primary_idx}')
                self.phi_obs_t = getattr(self, f'phi_obs_track_{primary_idx}')
                self.radius_obs_t = getattr(self, f'radius_obs_track_{primary_idx}')
                self.time_obs_t = getattr(self, f'time_obs_track_{primary_idx}')

        self.estimated_train_points = self._estimate_train_points()

    def _estimate_train_points(self):
        """Estimate the number of finite observation points used for data misfit."""
        skip_steps = int(getattr(self.config, 'overlap_hours', 0.0) * 3600.0 / self.config.dt)

        if self.use_multispacecraft and self.num_tracks > 0:
            total = 0
            for idx in range(self.num_tracks):
                v_obs_track = getattr(self, f'v_obs_track_{idx}')
                n = int(v_obs_track.numel())
                if n <= 0:
                    continue
                t0 = min(skip_steps, n - 1)
                valid = torch.isfinite(v_obs_track[t0:])
                total += int(valid.sum().item())
            return total

        n = int(self.v_obs.numel())
        if n <= 0:
            return 0
        t0 = min(skip_steps, n - 1)
        valid = torch.isfinite(self.v_obs[t0:])
        return int(valid.sum().item())

    def _compute_track_misfit(self, v_pred_track, v_obs_track, minibatch_idx=0, num_minibatches=1):
        """Compute misfit terms for one spacecraft track."""
        Nt_obs = min(v_pred_track.shape[0], v_obs_track.shape[0])
        if Nt_obs <= 0:
            return None

        skip_steps = int(getattr(self.config, 'overlap_hours', 0.0) * 3600.0 / self.config.dt)
        t0 = min(skip_steps, Nt_obs - 1)

        v_pred_valid = v_pred_track[t0:Nt_obs]
        v_obs_valid = v_obs_track[t0:Nt_obs]
        valid_mask = torch.isfinite(v_obs_valid) & torch.isfinite(v_pred_valid)

        if not torch.any(valid_mask):
            return None

        residuals = (v_pred_valid[valid_mask] - v_obs_valid[valid_mask]) ** 2
        abs_errors = (v_pred_valid[valid_mask] - v_obs_valid[valid_mask]).abs()

        num_minibatches = max(1, int(num_minibatches))
        minibatch_idx = int(minibatch_idx)
        if num_minibatches > 1:
            n_valid = int(residuals.shape[0])
            if n_valid <= 0:
                return None
            chunk = int(np.ceil(n_valid / float(num_minibatches)))
            start = minibatch_idx * chunk
            end = min(start + chunk, n_valid)
            if start >= n_valid:
                return None
            residuals = residuals[start:end]
            abs_errors = abs_errors[start:end]
            if residuals.numel() == 0:
                return None

        weighting = getattr(self.config, 'loss_weighting', 'uniform')
        if weighting == 'velocity':
            if num_minibatches > 1:
                v_obs_abs_all = v_obs_valid[valid_mask].detach()
                n_obs = int(v_obs_abs_all.shape[0])
                chunk = int(np.ceil(n_obs / float(num_minibatches)))
                start = minibatch_idx * chunk
                end = min(start + chunk, n_obs)
                if start >= n_obs:
                    return None
                v_obs_abs = v_obs_abs_all[start:end]
            else:
                v_obs_abs = v_obs_valid[valid_mask].detach()
            v_mean = torch.clamp(v_obs_abs.mean(), min=1e-6)
            weights = (v_obs_abs / v_mean) ** 2
            loss_data = (weights * residuals).mean()
        else:
            loss_data = residuals.mean()

#        valid = torch.isfinite(v_pred_track)

#        loss_data = torch.mean(
#            v_pred_track[valid]**2
#            )
        return {
            'loss_data': loss_data,
            'residuals': residuals,
            'abs_errors': abs_errors,
            'rmse': torch.sqrt(residuals.mean()),
            'max_ae': abs_errors.max(),
            'n_valid': int(valid_mask.sum().item()),
        }
    
    def forward(
        self,
        v_inner,
        loss_mode='standard',
        minibatch_idx=0,
        num_minibatches=1,
        regularization_scale=1.0,
    ):
        """
        Compute total loss for the inverse problem.

        Args:
            v_inner: (Nphi, Nt) - boundary condition (trainable)
            loss_mode: 'standard' or 'prior'

        Returns:
            loss: scalar
        """
        loss_mode = str(loss_mode).strip().lower()
        if loss_mode == 'prior':
            return self._compute_constant_prior_loss(v_inner)

        # Forward model
        v_full = self.forward_model(v_inner, self.v0)

        # Check for NaN after forward model
        if torch.isnan(v_full).any():
            print("WARNING: NaN detected in v_full after forward model")
            return torch.tensor(float('inf'), device=v_full.device)
        
        # Sample along each spacecraft track using actual radius and longitude.
        r = self.forward_model.r
        phi = self.forward_model.phi
        
        track_losses = []
        all_residuals = []
        all_abs_errors = []
        track_terms = []

        if self.use_multispacecraft and self.num_tracks > 0:
            for idx in range(self.num_tracks):
                v_obs_track = getattr(self, f'v_obs_track_{idx}')
                phi_obs_track = getattr(self, f'phi_obs_track_{idx}')
                radius_obs_track = getattr(self, f'radius_obs_track_{idx}')
                time_obs_track = getattr(self, f'time_obs_track_{idx}')
                v_pred_track = sample_satellite_track(
                    v_full,
                    r,
                    phi,
                    phi_obs_track,
                    radius_obs_track,
                    time_obs=time_obs_track,
                    dt=self.config.dt,
                )
                nan_mask = torch.isnan(v_pred_track)
                if nan_mask.any():
                    # NaN can be expected for invalid geometry/time samples;
                    # only warn when NaN appears on geometrically valid points.
                    tmin = 0.0
                    tmax = float(self.config.dt) * float(v_full.shape[2] - 1)
                    expected_nan_mask = (
                        ~torch.isfinite(phi_obs_track)
                        | ~torch.isfinite(radius_obs_track)
                        | ~torch.isfinite(time_obs_track)
                        | (time_obs_track < tmin)
                        | (time_obs_track > tmax)
                    )
                    unexpected_nan_mask = nan_mask & ~expected_nan_mask
                    if unexpected_nan_mask.any():
                        n_finite = int(torch.isfinite(v_pred_track).sum().item())
                        print(
                            f"WARNING: NaN detected in v_pred along satellite track {idx}; "
                            f"using finite subset only ({n_finite}/{v_pred_track.numel()})"
                        )
                        debug_key = f"track_{idx}"
                        if debug_key not in self._nan_debug_reported:
                            nan_idx = torch.where(unexpected_nan_mask)[0]
                            nan_idx_cpu = nan_idx.detach().cpu().numpy()
                            if nan_idx.numel() > 0:
                                phi_nan = phi_obs_track[nan_idx]
                                r_nan = radius_obs_track[nan_idx]
                                phi_finite_nan = int(torch.isfinite(phi_nan).sum().item())
                                r_finite_nan = int(torch.isfinite(r_nan).sum().item())
                                t_unique = torch.unique(nan_idx)
                                nonfinite_vfull = int((~torch.isfinite(v_full[:, :, t_unique])).sum().item())
                                print(
                                    f"  NaN forensic track {idx}: first_nan_t={nan_idx_cpu[:20].tolist()}, "
                                    f"phi_finite_on_nan={phi_finite_nan}/{nan_idx.numel()}, "
                                    f"r_finite_on_nan={r_finite_nan}/{nan_idx.numel()}, "
                                    f"nonfinite_v_full_at_nan_times={nonfinite_vfull}"
                                )
                            self._nan_debug_reported.add(debug_key)
                result = self._compute_track_misfit(
                    v_pred_track,
                    v_obs_track,
                    minibatch_idx=minibatch_idx,
                    num_minibatches=num_minibatches,
                )
                
                if result is None and int(num_minibatches) > 1:
                    continue

                if result is not None:
                    track_losses.append(result['loss_data'])
                    all_residuals.append(result['residuals'])
                    all_abs_errors.append(result['abs_errors'])
                    track_terms.append(
                        {
                            'id': self.track_ids[idx] if idx < len(self.track_ids) else f'TRACK_{idx}',
                            'loss': float(result['loss_data'].detach().item()),
                            'rmse': float(result['rmse'].detach().item()),
                            'max_ae': float(result['max_ae'].detach().item()),
                            'n_valid': int(result['n_valid']),
                        }
                    )
        else:
            v_pred_track = sample_satellite_track(
                v_full,
                r,
                phi,
                self.phi_obs_t,
                self.radius_obs_t,
                time_obs=self.time_obs_t,
                dt=self.config.dt,
            )
            nan_mask = torch.isnan(v_pred_track)
            if nan_mask.any():
                tmin = 0.0
                tmax = float(self.config.dt) * float(v_full.shape[2] - 1)
                expected_nan_mask = (
                    ~torch.isfinite(self.phi_obs_t)
                    | ~torch.isfinite(self.radius_obs_t)
                    | ~torch.isfinite(self.time_obs_t)
                    | (self.time_obs_t < tmin)
                    | (self.time_obs_t > tmax)
                )
                unexpected_nan_mask = nan_mask & ~expected_nan_mask
                if unexpected_nan_mask.any():
                    n_finite = int(torch.isfinite(v_pred_track).sum().item())
                    print(
                        "WARNING: NaN detected in v_pred along primary satellite track; "
                        f"using finite subset only ({n_finite}/{v_pred_track.numel()})"
                    )
                    debug_key = 'track_primary'
                    if debug_key not in self._nan_debug_reported:
                        nan_idx = torch.where(unexpected_nan_mask)[0]
                        nan_idx_cpu = nan_idx.detach().cpu().numpy()
                        if nan_idx.numel() > 0:
                            phi_nan = self.phi_obs_t[nan_idx]
                            r_nan = self.radius_obs_t[nan_idx]
                            phi_finite_nan = int(torch.isfinite(phi_nan).sum().item())
                            r_finite_nan = int(torch.isfinite(r_nan).sum().item())
                            t_unique = torch.unique(nan_idx)
                            nonfinite_vfull = int((~torch.isfinite(v_full[:, :, t_unique])).sum().item())
                            print(
                                f"  NaN forensic primary track: first_nan_t={nan_idx_cpu[:20].tolist()}, "
                                f"phi_finite_on_nan={phi_finite_nan}/{nan_idx.numel()}, "
                                f"r_finite_on_nan={r_finite_nan}/{nan_idx.numel()}, "
                                f"nonfinite_v_full_at_nan_times={nonfinite_vfull}"
                            )
                        self._nan_debug_reported.add(debug_key)
            result = self._compute_track_misfit(
                v_pred_track,
                self.v_obs,
                minibatch_idx=minibatch_idx,
                num_minibatches=num_minibatches,
            )
            if result is not None:
                track_losses.append(result['loss_data'])
                all_residuals.append(result['residuals'])
                all_abs_errors.append(result['abs_errors'])
                track_terms.append(
                    {
                        'id': 'TRACK_0',
                        'loss': float(result['loss_data'].detach().item()),
                        'rmse': float(result['rmse'].detach().item()),
                        'max_ae': float(result['max_ae'].detach().item()),
                        'n_valid': int(result['n_valid']),
                    }
                )

        self._track_loss_terms = track_terms

        if loss_mode == 'prior':
            self._max_abs_error = float('nan')
            self._rmse = float('nan')
            track_terms = []

        if track_losses:
            loss_data = torch.stack(track_losses).mean()
            residuals_all = torch.cat(all_residuals)
            abs_errors_all = torch.cat(all_abs_errors)
            self._max_abs_error = abs_errors_all.max().item()
            self._rmse = torch.sqrt(residuals_all.mean()).item()
        else:
            loss_data = torch.tensor(0.0, device=v_full.device)
            self._max_abs_error = float('nan')
            self._rmse = float('nan')
        
        # Check for NaN in data loss and print diagnostics
        if torch.isnan(loss_data) or loss_data.item() > 1e6:
            if torch.isnan(loss_data):
                print("WARNING: NaN in loss_data")
            else:
                print(f"WARNING: Very large loss_data = {loss_data.item():.3e}")
            print(f"  active tracks: {self.num_tracks if self.use_multispacecraft else 1}")
            if torch.isnan(loss_data):
                return torch.tensor(float('inf'), device=v_full.device)
        
        # Regularization: second-order smoothness in space (phi)
        # Penalizes curvature: v[i+1] - 2*v[i] + v[i-1]
        # Much more effective at suppressing ringing and oscillatory modes than first-order.
        # Per improvements.md recommendation #6.
        reg_scale = float(regularization_scale)
        loss_phi = reg_scale * self.config.lambda_phi * torch.mean(
            (v_inner[2:, :] - 2*v_inner[1:-1, :] + v_inner[:-2, :]) ** 2
        )
        
        # Regularization: second-order smoothness in time
        # Penalizes curvature: v[:, i+1] - 2*v[:, i] + v[:, i-1]
        loss_t = reg_scale * self.config.lambda_t * torch.mean(
            (v_inner[:, 2:] - 2*v_inner[:, 1:-1] + v_inner[:, :-2]) ** 2
        )
        
        # Check for NaN in regularization
        if torch.isnan(loss_phi) or torch.isnan(loss_t):
            print("WARNING: NaN in regularization terms")
            return torch.tensor(float('inf'), device=v_full.device)

        # Expose objective breakdown for epoch logging and diagnostics.
        self._loss_data = float(loss_data.detach().item())
        self._loss_phi = float(loss_phi.detach().item())
        self._loss_t = float(loss_t.detach().item())
        
        total_loss = loss_data + loss_phi + loss_t
        
        # Check for NaN in total loss
        if torch.isnan(total_loss):
            print("WARNING: NaN in total_loss")
            print(f"  loss_data: {loss_data.item():.6f}")
            print(f"  loss_phi: {loss_phi.item():.6f}")
            print(f"  loss_t: {loss_t.item():.6f}")
            return torch.tensor(float('inf'), device=v_full.device)
        
        return total_loss

    def _compute_constant_prior_loss(self, v_inner):
        prior_loss = self.constant_prior_weight * torch.mean((v_inner - self.constant_prior_value) ** 2)
        loss_phi = self.config.lambda_phi * torch.mean(
            (v_inner[2:, :] - 2*v_inner[1:-1, :] + v_inner[:-2, :]) ** 2
        )
        loss_t = self.config.lambda_t * torch.mean(
            (v_inner[:, 2:] - 2*v_inner[:, 1:-1] + v_inner[:, :-2]) ** 2
        )
        if torch.isnan(loss_phi) or torch.isnan(loss_t):
            print("WARNING: NaN in prior regularization terms")
            return torch.tensor(float('inf'), device=v_inner.device)
        self._max_abs_error = float('nan')
        self._rmse = float('nan')
        self._track_loss_terms = []
        self._loss_data = float(prior_loss.detach().item())
        self._loss_phi = float(loss_phi.detach().item())
        self._loss_t = float(loss_t.detach().item())
        return prior_loss + loss_phi + loss_t


def build_training_stages(config, skip_warmup_prior=False):
    """Build optimization stages: optional fast preconditioner then refinement."""
    refine_scheme = getattr(config, 'forward_scheme', 'fv')
    stages = []

    warmup_prior_epochs = int(getattr(config, 'warmup_prior_epochs', 0))
    if warmup_prior_epochs > 0 and not skip_warmup_prior:
        stages.append({
            'name': 'warmup_prior',
            'scheme': refine_scheme,
            'lr': float(getattr(config, 'warmup_prior_lr', config.lr)),
            'min_epochs': warmup_prior_epochs,
            'max_epochs': warmup_prior_epochs,
            'target_mse': -1.0,
            'loss_mode': 'prior',
        })

    if getattr(config, 'use_fast_preconditioner', False):
        pre_scheme = getattr(config, 'preconditioner_scheme', 'rk4_fast')
        if pre_scheme != refine_scheme:
            stages.append({
                'name': 'precondition',
                'scheme': pre_scheme,
                'lr': float(getattr(config, 'preconditioner_lr', config.lr)),
                'min_epochs': int(getattr(config, 'preconditioner_min_epochs', 0)),
                'max_epochs': int(getattr(config, 'preconditioner_max_epochs', config.N_epochs)),
                'target_mse': float(getattr(config, 'preconditioner_target_mse', -1.0)),
            })

    stages.append({
        'name': 'refine',
        'scheme': refine_scheme,
        'lr': float(config.lr),
        'min_epochs': int(getattr(config, 'min_epochs', 0)),
        'max_epochs': int(getattr(config, 'max_epochs', config.N_epochs))
            if bool(getattr(config, 'adaptive_epochs', False)) else int(config.N_epochs),
        'target_mse': float(getattr(config, 'target_mse', -1.0)),
    })
    return stages

