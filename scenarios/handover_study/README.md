# Handover study

A LEO handover scenario for NTN-Connect. It compares association strategies such as
max-RSRP and max-service-time (longest-visibility) over OneWeb and Starlink, under a
minimum-power allocation. The engine lives in `ntnconnect/`; this folder only
configures and drives it.

## Configuration

Every physical value is in `config.py`, the link budget and study settings. The main ones:

- Constellations: OneWeb and Starlink, from the TLEs in `tles/`
- Carrier 12 GHz, 500 MHz shared over 8 beams (62.5 MHz per user)
- Satellite antenna gain 38.5 dBi (OneWeb) / 32 dBi (Starlink), 20 dBi receive, noise figure 1.2 dB
- 50 users in a Quebec hotspot (52 N, 72 W, 400 km radius)
- User elevation mask 25 degrees, gateway mask 10 degrees
- Per-satellite power cap 100 W, outage target 5 percent
- Run length 45 minutes, t_pass 180 s (Starlink) / 300 s (OneWeb)

The available handover strategies are `farthest_approaching` (MST), `max_sinr`
(max-RSRP), `max_rx_power`, `gpp_a3`, `load_aware`, `utility_based`, `nearest_sat`,
`persistent_nearest`, `centralized`, and their `_cluster` variants.

## Pipeline

The three phases are separate scripts so each can be run and inspected on its own:

1. `scripts/phase1/generate_dataset.py` builds the per-constellation geometry dataset once.
2. `scripts/phase2/simulate.py` runs one case (constellation, strategy, operating point, gamma)
   over that dataset through the engine stage pipeline.
3. `scripts/phase3/metrics.py` computes the KPIs (admission, outage, transmit power, IHI) from
   the per-tick snapshot.

See `examples/example.py` at the repository root for an end-to-end run of all three phases.
