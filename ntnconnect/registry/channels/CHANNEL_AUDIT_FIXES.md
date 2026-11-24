# Channel package — audit fixes changelog

Trace of the correctness audit of `ntnconnect/registry/channels/`. Confirmed bugs B1–B11,
config-dependent items 12–24, architecture changes, new knobs, and reproducibility.

Behavior-changing fixes are gated by flags; set the flag to restore legacy output.

---

## Confirmed bugs (B1–B11)

### B1 — Refraction not seen by the model — FIXED
- Symptom: `RefractionP834Component` wrote `extras["elev_deg"]` but the model got the raw `elev_deg`.
- Fix (`pipeline.py`): model is fed `extras["elev_deg"]`; refraction ordered first in pre-model.
- Effect: only when `enable_refraction_p834=True` (off by default → no change).

### B2 — Rician K inflated by n_tx — FIXED
- Symptom: diffuse term normalized to total power 1 while LoS `‖A‖²=n_tx` → realized K = K·n_tx (~+18 dB at 8×8).
- Fix: per-element unit-variance diffuse (`E[‖g‖²]=n_tx`) in all Rician models via `_steering.per_antenna_rician_batch(unit_mean=True)` and in-model diffuse scaling.
- Flag: `rician_unit_mean_fading` (default True). False = legacy inflated K.
- Verify: realized K = configured K; `E[‖h‖²]=n_tx`.

### B3 — `h` scale differed per model — FIXED
- Symptom: SV/TDL simplified gave `E[‖h‖²]≈1` while Rician models gave `n_tx` → ~18 dB SINR jump on model switch.
- Fix: all models/modes normalized to `E[‖h‖²]=n_tx` (same flag).

### B4 — Jakes temporal correlation dead + wrong Doppler — FIXED
- Symptom: ran before Doppler (f_d=0 → no-op); if run, used LEO Doppler shift not fading spread.
- Fix (`jakes.py`, `pipeline.py`): uses fading spread `v·f_c/c` (from `extras["fading_doppler_hz"]` or `ue_speed_mps`); scheduled first post-model; skips models that did in-model temporal fading.

### B5 — Distance phase double-applied — FIXED
- Symptom: `upa_geometric_los` LoS carried `e^{+j2πfd/c}` and `PropagationDelayComponent` applied `e^{-j2πfd/c}` → cancelled.
- Fix (`_steering.py`, `steering_upa.py`): removed the bulk phase; `PropagationDelayComponent` is the single owner (correct `-j` sign). `upa_geometric_los` == `upa_kron`.

### B6 — DT sampler used conj(Σ) — FIXED
- Symptom: `dt_channel.sample` used `z @ L.conj().T` → covariance = conj(Σ).
- Fix: `z @ L.T` → covariance = Σ. Diagonal/power unaffected; off-diagonal phase sign corrected.

### B7 — Double shadowing (Loo/Lutz/Perez-Fontan) — FIXED
- Symptom: LMS models shadow inside `h`; external shadowing component added it again.
- Fix: those models declare `handles_internally={shadowing, shadowing_38811, shadowing_gudmundson}` → pipeline skips external shadowing. Enforced by `channel_class` contract check.

### B8 — GeometricNominal aged geometry not implemented — FIXED
- Symptom: `geo_nom_aged_elev_std_deg` / `_az_std_deg` read but never used.
- Fix: AR(1) Gauss-Markov elevation/azimuth error, `ρ=exp(-dt/τ)`. New knob `geo_nom_aged_corr_time_s` (τ). Defaults 0 → off (deterministic).

### B9 — RNG not reproducible / pipelines coupled — FIXED
- Symptom: models/components drew from global `np.random`; physical & DT pipelines shared it.
- Fix: each pipeline owns an independent `state.rng` seeded per role from `channel_rng_seed`; all draws routed through it (helpers receive `rng`).

### B10 — FSPL docstring default wrong — FIXED (doc)
- Code default `32.45` (MHz) is correct; docstring said `92.45`. Docstring corrected + warning against 92.45.

### B11 — P.838 circular α wrong — FIXED
- Was `α=(αH+αV)/2`; now k-weighted `α=(kHαH+kVαV)/(kH+kV)` (`rain_p838_hv.py`, `wisee_weather.py`). `k` was already correct.

---

## Config-dependent items (12–24)

| # | Item | Status |
|---|---|---|
| 12 | spatial-corr colors LoS (should be diffuse-only) | guarded + warn; correct fix (in-model) = FUTURE WORK |
| 13 | beam-pattern (fixed-nadir) vs steered array | replaced with scan loss `-10q·log10(cosθ)` + optional pointing error; default off |
| 14 | NTN-TDL `consumes_external_k_factor` inaccurate | set False |
| 15 | model K knobs overridden by pipeline pending K | by design (K centralized); documented |
| 16 | geo_nom aged knobs unused | FIXED (=B8) |
| 17 | legacy shadowing ρ hardcoded 0.8, corr-dist unused | ρ from corr-dist + speed, else configurable `shadowing_corr_rho` |
| 18 | Jakes unreachable | FIXED (=B4) |
| 19 | SteeringUPAComponent inert | kept (documented no-op) |
| 20 | predictor registry empty | reported: DT prediction needs concrete predictors registered |
| 21 | dead term in `terrestrial_uma` | removed |
| 22 | `_HAS_LEGACY` unused | removed |
| 23 | rain_p618 rain-height doc mismatch | doc fixed (3 km) |
| 24 | atmospheric/K tables approximate | NEEDS REFERENCE VERIFICATION (documented in-file) |

---

## Architecture changes

- **Pipeline role** (`pipeline.py`): `ChannelPipeline(role="physical"|"dt")`. DT-only components (e.g. `dt_obs_noise`) added only on `role="dt"`. `ChannelModelFactory.build_dt()` builds the twin pipeline. Physical pipeline stays pure.
- **RNG isolation**: each pipeline has an independent Generator (seeded per role from `channel_rng_seed`); physical and DT never share a stream.
- **`channel_class` contract** (`base.py`): `"fast_fading"` (default) or `"complete_lms"` (Loo/Lutz/Perez-Fontan); pipeline warns on mis-declared shadowing ownership.

---

## New config knobs

| knob | default | meaning |
|---|---|---|
| `rician_unit_mean_fading` | True | per-element unit-variance diffuse (correct K). False = legacy |
| `channel_rng_seed` | None | seeds per-pipeline RNG (reproducible) |
| `geo_nom_aged_corr_time_s` | 0.0 | GeometricNominal aging correlation time τ (0 = iid) |
| `beam_scan_loss_exp_q` | 1.2 | scan-loss exponent q (NEEDS REFERENCE) |
| `beam_pointing_error_deg` | 0.0 | optional residual mis-pointing |
| `shadowing_corr_rho` | 0.8 | legacy-shadowing AR(1) ρ fallback |
| `shadowing_user_speed_m_s` | 0.0 | legacy-shadowing drift for ρ=exp(-v·dt/dcorr) |

---

## Reproducing legacy output

- Rician K: `rician_unit_mean_fading=False`
- Everything else: off-by-default fixes (refraction, beam, spatial-corr, jakes, dt_obs_noise) have no effect unless enabled.

---

## Cosmetic

- Removed content-free banner/rule comment lines (`# ====`, `# ----`, `# ────`) repo-wide.
- Comments minimized to terse phrases.
