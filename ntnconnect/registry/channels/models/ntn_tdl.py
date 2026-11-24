# NTN TDL (TR 38.811 §6.9.2, profiles A-D). Narrowband collapse of the tap PDP.
# tdl_mode: simplified (flat Rayleigh) | full (tabulated profile). tdl_fading: iid|ar1|sos.

from __future__ import annotations
import numpy as np
from ntnconnect.registry.channels.base import AbstractChannelModel, ChannelData
from ntnconnect.registry.channels.pipeline import register_model

C = 299_792_458.0

# TR 38.811 Table 6.9.2-1..4. rayleigh_taps: (normalized delay, power dB).
# los: the first-tap Rician split (LoS-path dB, tap-1 Rayleigh dB, K1 dB) at
# normalized delay 0, present only for the LOS profiles C and D.
_TDL = {
    "A": dict(rayleigh=[(0.0, 0.0), (1.0811, -4.675), (2.8416, -6.482)], los=None),
    "B": dict(rayleigh=[(0.0, 0.0), (0.7249, -1.973), (0.7410, -4.332),
                        (5.7392, -11.914)], los=None),
    "C": dict(rayleigh=[(0.0, -10.618), (14.8124, -23.373)],
              los=dict(los_db=-0.394, k1_db=10.224)),
    "D": dict(rayleigh=[(0.0, -11.991), (0.5596, -9.887), (7.3340, -16.771)],
              los=dict(los_db=-0.284, k1_db=11.707)),
}


@register_model("NTN_TDL")
class NTN_TDL(AbstractChannelModel):
    """NTN Tapped-Delay-Line (TR 38.811 §6.9.2) with a simplified narrowband
    collapse (default) and a full tabulated-profile mode."""

    has_intrinsic_spatial_structure = False
    # NTN-TDL does not consume an external K: simplified mode is pure Rayleigh (no
    # K), full mode derives its Rician split from the tabulated profile powers
    # (k_eff), not from state["k_factor_db_pending"]. Declaring False makes the
    # pipeline warn if a K-factor component is enabled alongside it.
    consumes_external_k_factor    = False
    produces_per_antenna_h = True

    def compute_channel(self, dist_m, elev_deg, dt_sec, state=None,
                        pos_m=None, azimuth_deg=None):
        d = np.maximum(np.asarray(dist_m, dtype=float), 1.0)
        n = len(d)
        n_x = int(getattr(self.cfg, "n_tx_x", 8))
        n_y = int(getattr(self.cfg, "n_tx_y", 8))
        n_tx = n_x * n_y

        if str(getattr(self.cfg, "tdl_mode", "simplified")).lower() != "full":
            # Simplified narrowband collapse: flat-fading Rayleigh proxy (not the
            # wideband TDL). Normalised to the package convention E[||h||^2]=n_tx
            # (per-element unit power) so switching models does not rescale SINR;
            # rician_unit_mean_fading=False restores the legacy E[||h||^2]=1.
            from ntnconnect.registry.channels._steering import per_antenna_rayleigh
            rng = getattr(state, "rng", None) or np.random
            amp = (np.sqrt(n_tx)
                   if bool(getattr(self.cfg, "rician_unit_mean_fading", True)) else 1.0)
            h = np.zeros((n, n_tx), dtype=complex)
            for i in range(n):
                h[i] = per_antenna_rayleigh(0.0, 0.0, n_x, n_y, amp=amp, rng=rng)
            if state is not None:
                state.data["model_did_temporal_fading"] = False
            return ChannelData(loss_db=np.zeros(n), h=h, doppler_hz=np.zeros(n),
                               extras={"tdl_mode": "simplified"})

        # Full mode: tabulated NTN-TDL profile
        from ntnconnect.registry.channels._steering import steering_matrix_batch, off_nadir_rad
        from ntnconnect.registry.channels._fading import diffuse, doppler_spread_hz
        prof = str(getattr(self.cfg, "tdl_profile", "A")).upper()
        tbl = _TDL.get(prof, _TDL["A"])
        rms_ds = float(getattr(self.cfg, "tdl_rms_delay_spread_ns", 50.0)) * 1e-9

        p_ray = np.array([10.0 ** (pw / 10.0) for _, pw in tbl["rayleigh"]])
        delays = np.array([dl * rms_ds for dl, _ in tbl["rayleigh"]])
        p_los = (10.0 ** (tbl["los"]["los_db"] / 10.0)) if tbl["los"] else 0.0
        p_tot = p_los + float(p_ray.sum())
        los_amp = float(np.sqrt(p_los / p_tot))
        diff_amp = float(np.sqrt(p_ray.sum() / p_tot))
        k_eff_db = (10.0 * np.log10(p_los / p_ray.sum())) if p_los > 0 else -np.inf

        elev = np.clip(np.asarray(elev_deg, dtype=float), 0.0, 90.0)
        az = (np.asarray(azimuth_deg, dtype=float)
              if azimuth_deg is not None else np.zeros(n))
        sat_alt_m = float(getattr(self.cfg, "satellite_altitude_m", 600_000.0))
        carrier_hz = float(getattr(self.cfg, "carrier_freq_hz", 20.0e9))
        d_lambda = float(getattr(self.cfg, "cvx_antenna_spacing_d_over_lambda", 0.5))
        strategy = str(getattr(self.cfg, "steering_strategy", "upa_kron")).lower()
        A = steering_matrix_batch(off_nadir_rad(elev, sat_alt_m), np.radians(az),
                                  n_x, n_y, strategy=strategy, dist_m=d,
                                  carrier_hz=carrier_hz, d_over_lambda=d_lambda)

        rng = getattr(state, "rng", None) or np.random
        f_d = doppler_spread_hz(self.cfg, carrier_hz)
        g = diffuse(str(getattr(self.cfg, "tdl_fading", "iid")), n, n_tx, f_d,
                    float(dt_sec), state, "ntn_tdl", rng,
                    sos_M=int(getattr(self.cfg, "threegpp_sos_M", 16)))
        if bool(getattr(self.cfg, "rician_unit_mean_fading", True)):
            g = g * np.sqrt(n_tx)          # per-element unit variance (E[||g||^2]=n_tx)
        h = los_amp * A + diff_amp * g
        if state is not None:
            state.data["model_did_temporal_fading"] = \
                str(getattr(self.cfg, "tdl_fading", "iid")).lower() in ("ar1", "sos")
        return ChannelData(loss_db=np.zeros(n), h=h, doppler_hz=np.zeros(n),
                           extras={"tdl_mode": "full", "tdl_profile": prof,
                                   "tap_delays_s": delays, "tap_powers_lin": p_ray,
                                   "k_eff_db": k_eff_db,
                                   "k1_db": (tbl["los"]["k1_db"] if tbl["los"] else None)})
