"""WISEE weather (rain+cloud): P.838-HV rain + P.840 cloud, mu_wx and
sigma_wx=WEATHER_SIG*mu_wx. Shared by the channel stage and the weather risk margin."""
from __future__ import annotations
import numpy as np

WEATHER_SIG   = 0.5      # weather (rain+cloud) std as a fraction of its mean
P618_RAIN_MMPH = 25.0    # rain rate [mm/h] (WISEE default; heavy)
P618_RAIN_HKM  = 3.0     # rain height [km]
P840_CLOUD_L   = 0.5     # cloud liquid water [kg/m^2]
# ITU-R P.838 k, alpha tables (H, V) vs frequency [GHz]
_RF = np.array([4., 6., 10., 12., 15., 20., 25., 30., 40., 50.])
_KH = np.array([0.000650, 0.00175, 0.0101, 0.0188, 0.0367, 0.0751, 0.124, 0.187, 0.350, 0.536])
_AH = np.array([1.121, 1.308, 1.276, 1.217, 1.154, 1.099, 1.061, 1.021, 0.939, 0.873])
_KV = np.array([0.000591, 0.00155, 0.00887, 0.0168, 0.0335, 0.0691, 0.113, 0.167, 0.310, 0.479])
_AV = np.array([1.075, 1.265, 1.264, 1.200, 1.128, 1.065, 1.030, 1.000, 0.929, 0.868])


def _rain_db(elev_deg, f_ghz):
    """ITU-R P.838-HV rain attenuation [dB], circular pol (mean over H,V)."""
    f = float(np.clip(f_ghz, _RF[0], _RF[-1]))
    kh = 10.0 ** np.interp(np.log10(f), np.log10(_RF), np.log10(_KH)); ah = np.interp(f, _RF, _AH)
    kv = 10.0 ** np.interp(np.log10(f), np.log10(_RF), np.log10(_KV)); av = np.interp(f, _RF, _AV)
    # ITU-R P.838 circular (tilt 45°): k = (kH+kV)/2; alpha = (kH*aH+kV*aV)/(kH+kV)
    k = 0.5 * (kh + kv); a = (kh * ah + kv * av) / (kh + kv)
    s = np.sin(np.radians(np.clip(elev_deg, 1e-3, 90.0)))
    L_s = P618_RAIN_HKM / s
    L_0 = 35.0 * np.exp(-0.015 * P618_RAIN_MMPH)
    nu = 1.0 / (1.0 + L_s / L_0)
    return k * (P618_RAIN_MMPH ** a) * L_s * nu


def _cloud_db(elev_deg, f_ghz):
    """ITU-R P.840 cloud attenuation [dB]."""
    s = np.sin(np.radians(np.clip(elev_deg, 1e-3, 90.0)))
    return 0.001 * (f_ghz ** 1.9) * P840_CLOUD_L / s


def weather_mean_db(elev_deg, f_ghz):
    """Mean rain+cloud attenuation mu_wx [dB]."""
    return _rain_db(elev_deg, f_ghz) + _cloud_db(elev_deg, f_ghz)


def weather_sigma_db(elev_deg, f_ghz):
    """Weather std sigma_wx = WEATHER_SIG * mu_wx [dB] (the weather risk term)."""
    return WEATHER_SIG * weather_mean_db(elev_deg, f_ghz)
