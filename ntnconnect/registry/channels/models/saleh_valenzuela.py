# Saleh-Valenzuela (1987) cluster/ray multipath (Rayleigh rays, exp power decay).
# sv_mode: simplified (flat Rayleigh) | full (cluster/ray PDP). sv_fading: iid|ar1|sos.

from __future__ import annotations
import numpy as np
from ntnconnect.registry.channels.base import AbstractChannelModel, ChannelData
from ntnconnect.registry.channels.pipeline import register_model

C = 299_792_458.0


@register_model("SV")
class SalehValenzuela(AbstractChannelModel):
    """Saleh-Valenzuela cluster/ray model with a simplified narrowband collapse
    (default) and a full cluster/ray-realisation mode."""

    has_intrinsic_spatial_structure = False
    consumes_external_k_factor    = True
    produces_per_antenna_h = True

    def __init__(self, cfg):
        super().__init__(cfg)
        self.lam_c = float(cfg.sv_lambda_cluster_ns)
        self.lam_r = float(cfg.sv_lambda_ray_ns)
        self.gam_c = float(cfg.sv_gamma_cluster_ns)
        self.gam_r = float(cfg.sv_gamma_ray_ns)
        self.n_clu = int(cfg.sv_num_clusters)
        self.n_ray = int(cfg.sv_rays_per_cluster)

    def _realisation(self, rng):
        """One cluster/ray PDP realisation -> (delays_s, powers_lin) normalized
        to unit total power. Exponential inter-arrivals and power decays."""
        delays, powers = [], []
        T = 0.0
        for _ in range(self.n_clu):
            T += rng.exponential(self.lam_c)          # cluster arrival [ns]
            tau = 0.0
            pc = np.exp(-T / self.gam_c)
            for _ in range(self.n_ray):
                tau += rng.exponential(self.lam_r)    # ray arrival within cluster
                delays.append((T + tau) * 1e-9)       # [s]
                powers.append(pc * np.exp(-tau / self.gam_r))
        powers = np.asarray(powers, float)
        s = powers.sum()
        if s > 0:
            powers = powers / s                       # unit total power
        return np.asarray(delays, float), powers

    def compute_channel(self, dist_m, elev_deg, dt_sec, state=None,
                        pos_m=None, azimuth_deg=None):
        d = np.maximum(np.asarray(dist_m, dtype=float), 1.0)
        n = len(d)
        n_x = int(getattr(self.cfg, "n_tx_x", 8))
        n_y = int(getattr(self.cfg, "n_tx_y", 8))
        n_tx = n_x * n_y

        if str(getattr(self.cfg, "sv_mode", "simplified")).lower() != "full":
            # Simplified: fixed-amplitude narrowband collapse (historical).
            from ntnconnect.registry.channels._steering import per_antenna_rayleigh
            total_amp_sq = 0.0
            for k in range(self.n_clu):
                c_amp = np.exp(-(k * self.lam_c) / self.gam_c)
                for l in range(self.n_ray):
                    r_amp = np.exp(-(k * self.lam_c + l * self.lam_r) / self.gam_r)
                    total_amp_sq += c_amp * r_amp
            norm = max(np.sqrt(self.n_clu * self.n_ray), 1.0)
            eff_amp = float(np.sqrt(total_amp_sq) / norm)   # legacy arbitrary scale
            # Package convention E[||h||^2]=n_tx (per-element unit power) so the
            # scale matches every other model. The simplified SV is a flat Rayleigh
            # proxy, so use unit per-element power directly (the legacy eff_amp is an
            # arbitrary SV-param-dependent scale). flag=False keeps legacy eff_amp.
            if bool(getattr(self.cfg, "rician_unit_mean_fading", True)):
                eff_amp = float(np.sqrt(n_tx))
            rng = getattr(state, "rng", None) or np.random
            h = np.zeros((n, n_tx), dtype=complex)
            for i in range(n):
                h[i] = per_antenna_rayleigh(0.0, 0.0, n_x, n_y, amp=eff_amp, rng=rng)
            if state is not None:
                state.data["model_did_temporal_fading"] = False
            return ChannelData(loss_db=np.zeros(n), h=h, doppler_hz=np.zeros(n),
                               extras={"sv_mode": "simplified"})

        # Full mode: cluster/ray realisation -> unit-power composite
        from ntnconnect.registry.channels._fading import diffuse, doppler_spread_hz
        rng = getattr(state, "rng", None) or np.random.default_rng()
        delays, powers = self._realisation(rng)
        carrier_hz = float(getattr(self.cfg, "carrier_freq_hz", 20.0e9))
        f_d = doppler_spread_hz(self.cfg, carrier_hz)
        # Pure-NLOS SV: the narrowband composite is unit-power Rayleigh; use the
        # shared diffuse helper so the composite has the requested Doppler
        # correlation (per-ray gains sum to the same statistics).
        g = diffuse(str(getattr(self.cfg, "sv_fading", "iid")), n, n_tx, f_d,
                    float(dt_sec), state, "sv", rng,
                    sos_M=int(getattr(self.cfg, "threegpp_sos_M", 16)))
        if bool(getattr(self.cfg, "rician_unit_mean_fading", True)):
            g = g * np.sqrt(n_tx)          # per-element unit power (E[||h||^2]=n_tx)
        if state is not None:
            state.data["model_did_temporal_fading"] = \
                str(getattr(self.cfg, "sv_fading", "iid")).lower() in ("ar1", "sos")
        return ChannelData(loss_db=np.zeros(n), h=g, doppler_hz=np.zeros(n),
                           extras={"sv_mode": "full", "ray_delays_s": delays,
                                   "ray_powers_lin": powers,
                                   "rms_delay_spread_s": float(
                                       np.sqrt(np.sum(powers * delays ** 2)
                                               - np.sum(powers * delays) ** 2))
                                       if powers.size else 0.0})
