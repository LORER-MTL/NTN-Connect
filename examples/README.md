# Examples

A single, self-contained walkthrough of the NTN-Connect pipeline.

## `example.py`

Runs the three phases of the simulator on a short 5-minute case so it finishes
in a couple of minutes:

```bash
python examples/example.py
```

| Phase | Script it calls | What it does |
|-------|-----------------|--------------|
| 1 | `scenarios/handover_study/scripts/phase1/generate_dataset.py` | Propagate the constellation, compute geometry and visibility, write the per-batch dataset |
| 2 | `scenarios/handover_study/scripts/phase2/simulate.py` | Channel, Doppler, user–satellite association / handover, minimum-power allocation |
| 3 | `scenarios/handover_study/scripts/phase3/metrics.py` | Reduce the run to the paper KPIs: admission, outage, tx power, inter-handover interval, handovers |

Edit the constants at the top of the file (`CONSTELLATION`, `STRATEGY`,
`SE_LEVEL`, `MINUTES`, `STEP_S`) to try other cases. The available handover
strategies are `farthest_approaching` (MST), `max_sinr` (max-RSRP),
`max_rx_power`, `gpp_a3`, `load_aware`, `utility_based`, `nearest_sat`,
`persistent_nearest`, `centralized`, and their `_cluster` variants.

A longer study is the same three phases run at a finer step and over more
strategies and spectral-efficiency levels; edit the constants at the top of
`example.py` to explore.

## Testing

A fast smoke test exercises every pipeline stage on a 5-minute case:

```bash
pytest tests/            # or: python tests/test_smoke.py
```

## Advanced: network-architecture variants

`configs/examples/S1_*.yaml … S6_*.yaml` are example scenario configs that show
the engine's range of control-plane architectures, from fully centralized to
fully distributed:

| Config | Architecture |
|--------|--------------|
| `S1_fully_centralized.yaml` | One central controller for all operators |
| `S2_per_operator.yaml` | One controller per operator |
| `S3_per_network_central_core.yaml` | Per-network controllers with a shared core |
| `S4_per_type_federation.yaml` | Federated by node type |
| `S5_fully_distributed.yaml` | No central coordination |
| `S6_hybrid.yaml` | Mixed centralized / distributed |

These are not used by the handover paper (which is the `handover_study`
scenario); they are reference configurations for building new studies.
