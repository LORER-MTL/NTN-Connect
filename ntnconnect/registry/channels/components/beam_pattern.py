# Steered-array scan loss: G=G_max*cos^q(theta) -> -10q*log10(cos theta) dB.
# Optional pointing-error term 12*(delta/theta_3dB)^2. Default off. Knobs:
# beam_scan_loss_exp_q (q; tie to antenna spec), beam_pointing_error_deg, beam_max_loss_db.
from __future__ import annotations
import numpy as np
from ntnconnect.registry.channels.base import AbstractChannelComponent
from ntnconnect.registry.channels._steering import off_nadir_rad


class BeamPatternComponent(AbstractChannelComponent):
    name  = "beam_pattern"
    phase = "post_model"

    def apply(self, data, cfg, dt_sec, state=None):
        if not bool(getattr(cfg, "enable_beam_pattern", False)):
            return
        if data.loss_db is None:
            return
        elev = np.asarray(data.extras.get("elev_deg",
                                          np.zeros(data.n_links())), dtype=float)
        sat_alt_m = float(getattr(cfg, "satellite_altitude_m", 600_000.0))
        theta_n     = np.asarray(off_nadir_rad(elev, sat_alt_m), dtype=float)  # scan angle
        theta_n_deg = np.degrees(theta_n)

        # Scan loss: G = G_max cos^q(theta_scan) -> loss = -10 q log10(cos theta).
        q      = float(getattr(cfg, "beam_scan_loss_exp_q", 1.2))
        cos_s  = np.clip(np.cos(theta_n), 1e-3, 1.0)
        loss   = -10.0 * q * np.log10(cos_s)

        # Optional residual pointing-error roll-off about the steered beam centre.
        dtheta = float(getattr(cfg, "beam_pointing_error_deg", 0.0))
        if dtheta > 0.0:
            theta_3db = float(getattr(cfg, "beam_3db_deg", 4.0))
            loss = loss + 12.0 * (dtheta / max(theta_3db, 1e-3)) ** 2

        a_max = float(getattr(cfg, "beam_max_loss_db", 30.0))
        loss  = np.minimum(loss, a_max)
        data.loss_db = np.asarray(data.loss_db, dtype=float) + loss
        data.extras["beam_db"] = loss
        data.extras["beam_off_nadir_deg"] = theta_n_deg
