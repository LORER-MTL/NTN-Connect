# ntnconnect.analysis — results analysis and figures

Scenario-independent analysis stage. It reads the engine's standard per-run
Parquet outputs (via a run `manifest.csv`), computes metrics, and renders the
figures and summary tables. Nothing here is tied to a particular scenario,
constellation, or handover/routing strategy: every figure is driven by whatever
the run actually produced, so the same command works for any experiment.

## Command-line usage

    python -m ntnconnect.analysis <out_dir> [--runs-dir DIR] [options]

`<out_dir>` receives the figures, tables, and `outputs_index.json`. It is also
where the run `manifest.csv` is read from, unless `--runs-dir` points elsewhere
(e.g. when runs live in `simulation/` and analysis output goes to `analysis/`).

List everything selectable and exit:

    python -m ntnconnect.analysis --list

### Selecting which figures to render

Figures are activated by selection, no code edits. A selection is any mix of
explicit figure names and category selectors.

| Option | Effect |
|---|---|
| *(none)* | the curated default set (13 figures) |
| `--figures a,b,c` | activate only these (names and/or category selectors) |
| `--all` | activate every registered figure |
| `--exclude a,b` | drop these from the current selection |
| `--auto` | drop selected figures whose data this experiment did not produce |

Category selectors expand to the matching figure names:

- `kpi:<family>` where family is one of
  `coverage, capacity, latency, reliability, mobility, energy, connection, fairness, doppler`
- `scope:<level>` where level is one of `link, user, network`
- `stage:<phase>` where phase is `geometry` (dataset-only, no simulation) or `sim`

Examples:

    # everything the run supports, chosen automatically:
    python -m ntnconnect.analysis <out_dir> --all --auto

    # a coverage + capacity study, no handover:
    python -m ntnconnect.analysis <out_dir> --figures kpi:coverage,kpi:capacity

    # dataset-only geometry figures (no simulation stage):
    python -m ntnconnect.analysis <out_dir> --figures stage:geometry

    # mix explicit names and a category, minus one figure:
    python -m ntnconnect.analysis <out_dir> --figures sinr_ccdf,rate_ccdf,kpi:latency --exclude ho_geo_map

    # the Doppler CCDF only, at a 20 GHz carrier:
    python -m ntnconnect.analysis <out_dir> --figures kpi:doppler --carrier-hz 20e9

### Auto mode

`--auto` inspects the run after loading it and removes any selected figure with
no backing data, before rendering. Built-in figures are tested against the
aggregated metrics; the extra-KPI figures are tested against the columns present
in `sinr_snapshot.parquet`. It prints what it removed, for example:

    analysis[auto]: dropped 19 figure(s) with no data in this experiment:
      ['availability_bar', 'doppler_ccdf', 'e2e_rtt_ccdf', 'ho_geo_map', 'ihi_cdf', ...]

So a run without handover drops the handover figures, a run without routing
drops the RTT figure, and so on. Without `--auto` a figure with no data is
simply skipped at render time (same result, but the selection list is not
trimmed up front).

### Rendering options

| Option | Default | Meaning |
|---|---|---|
| `--no-titles` | titles on | render without titles (publication mode) |
| `--dpi N` | 140 | output resolution |
| `--format {png,pdf,svg}` | png | figure format |
| `--tex` | off | use a real LaTeX engine; falls back to mathtext if none is installed |
| `--bandwidth-hz HZ` | 62.5e6 | bandwidth for the extra-KPI rate/energy figures |
| `--carrier-hz HZ` | 12e9 | carrier frequency for the extra-KPI Doppler figure |

## Figure families

`--list` prints the full set grouped by KPI family. There are two kinds:

- Aggregate figures (31): one figure per metric, overlaying every constellation
  and strategy the run contains. Written to `<out_dir>/fig*.png`.
- Extra-KPI figures (6): energy, connection, fairness, and Doppler, computed from
  each run's `sinr_snapshot.parquet`. Written per run to
  `<out_dir>/<constellation>__<strategy>/<name>.png`.

| KPI family | Figures |
|---|---|
| coverage | visibility_ccdf, coverage_bar, mean_nvis_vs_lat, visibility_gap_cdf, availability_bar, visgap_per_user_overlay |
| capacity | sinr_ccdf, rate_ccdf, avg_rate_bar, avg_rate_bar_norm, rate_per_user_overlay, sinr_vs_elev, sinr_per_user_overlay, sinr_per_user_grid |
| latency | access_delay_ccdf, e2e_rtt_ccdf |
| reliability | ion_ccdf, ion_per_user_overlay, outage_ccdf, service_outage_ccdf, service_gap_cdf |
| mobility | contact_cdf, contact_vs_peakelev, intercontact_gap_cdf, ho_rate_bar, ihi_cdf, ihi_per_user_overlay, pingpong_bar, ho_geo_map, sinr_drop_around_ho |
| energy | power_bar, energy_power_ccdf, energy_efficiency_bar |
| connection | connection_admission_bar, connection_blocking_ccdf |
| fairness | fairness_jain_bar |
| doppler | doppler_ccdf |

Some KPI families a system-level simulator does not measure are documented as
out of scope (no figure) in `figure_catalog.py`: handover interruption time,
handover success/failure rate, control-plane latency, and beam/gateway
utilisation. These need lower-layer or protocol modelling.

## Python API

The CLI is a thin front-end over `run_analysis`, which any scenario can call
directly:

    from ntnconnect.analysis import run_analysis, list_figure_names

    run_analysis(
        out_dir,
        runs_dir=None,            # where manifest.csv lives (defaults to out_dir)
        figures=["kpi:coverage", "sinr_ccdf"],   # names and/or category selectors
        all_figures=False,        # or True for everything
        exclude=["ho_geo_map"],   # drop from the selection
        auto=False,               # drop figures with no data in this run
        show_titles=True, dpi=140, fig_format="png", use_tex=False,
        bandwidth_hz=None, carrier_hz=None,       # extra-KPI parameters
    )

    default_set, all_names = list_figure_names()   # inspect the registry

## Inputs and outputs

Reads (per run, whatever is present): `sinr_snapshot.parquet`,
`handover_events.parquet`, `contact_events.parquet`, `link_visibility.parquet`,
`visibility_per_tick.parquet`.

Writes to `<out_dir>`: `fig*.<format>` (aggregate figures),
`<constellation>__<strategy>/<name>.png` (extra-KPI figures),
`strategy_summary.csv` (unless `write_tables=False`), and `outputs_index.json`
listing what was produced.
