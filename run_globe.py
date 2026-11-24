"""3D globe launcher: regenerate globe from outputs/, serve over http, open browser.
Needs pandas + pyarrow + numpy (Anaconda base)."""
import os
import sys
import time
import functools
import threading
import webbrowser
import http.server
import socketserver
import subprocess
from pathlib import Path


for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# Paths
ROOT = Path(__file__).resolve().parent
VIZ  = ROOT / "ntnconnect" / "analysis" / "visualization"
PORT = 8542
STEP_S = "1"            # [s] must match the sim's step_s for link/sat sync
REGENERATE = True       # False = serve the existing cesium_mc.html
BUILD_INTERACTIVE = "auto"  # auto = build interactive.json only if missing; True/False force
PICK_FOLDER = True      # pop a folder-picker dialog each run

# DATASET (manifest.json + ephemeris) and RUNS_ROOT (simulation runs) live in
# separate trees; the globe is served from SERVE_DIR (ROOT/outputs). Keep a
# dataset and its runs as one matched pair.
def _autodetect_dataset() -> Path:
    """Find a dataset folder Cesium can read: prefer conference_paper, then
    demo_dataset, else the first dataset with a manifest under outputs/datasets."""
    base = ROOT / "outputs" / "datasets"

    def _has_manifest(d: Path) -> Path | None:
        # Prefer the STUDY layer so the globe clock matches the phase-2 runs; the
        # coarser coverage layer must not drive the runs.
        for sub in (d, d / "study", d / "coverage"):
            if (sub / "manifest.json").is_file():
                return sub
        return None

    # nested conference_paper layout: <const>/run_XXX/{coverage,study}
    conf = base / "conference_paper"
    if conf.is_dir():
        for const in sorted(c for c in conf.iterdir() if c.is_dir()):
            for run in sorted(const.glob("run_*")):
                hit = _has_manifest(run)
                if hit:
                    return hit
    # flat layouts: demo_dataset, then anything else with a manifest.
    cands = [base / "demo_dataset"]
    if base.is_dir():
        cands += sorted(d for d in base.iterdir() if d.is_dir())
    for d in cands:
        hit = _has_manifest(d)
        if hit:
            return hit
    return base / "demo_dataset"


DEFAULT_DATASET = _autodetect_dataset()


def _autodetect_runs() -> Path:
    """Prefer the latest conference_paper experiment, then demo_results, then the
    whole results tree."""
    base = ROOT / "outputs" / "results"
    conf = base / "conference_paper"
    if conf.is_dir():
        exps = sorted(d for d in conf.glob("run_*") if d.is_dir())
        if exps:
            return exps[-1]                      # latest experiment
    demo = base / "demo_results"
    return demo if demo.exists() else base


DEFAULT_RUNS    = _autodetect_runs()
SERVE_DIR       = ROOT / "outputs"


def _pick_folder(default: Path, title: str) -> Path:
    """Native folder dialog; returns the chosen folder (or `default`)."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
        d = filedialog.askdirectory(title=title, initialdir=str(default))
        root.destroy()
        return Path(d) if d else default
    except Exception as e:
        print("Folder picker unavailable (%s) — using default." % e)
        return default


def _scan_roots(runs_root: Path) -> list[Path]:
    """cesium_view discovers runs only two levels deep, so pass the folder itself
    AND each immediate sub-folder to cover both a single campaign and a parent of
    several campaigns."""
    roots = [runs_root]
    if runs_root.is_dir():
        roots += [d for d in sorted(runs_root.iterdir()) if d.is_dir()]
    return roots


DATASET   = (_pick_folder(DEFAULT_DATASET,
                          "Select the DATASET folder (manifest.json + oneweb/ starlink/)")
             if PICK_FOLDER else DEFAULT_DATASET)
RUNS_ROOT = (_pick_folder(DEFAULT_RUNS,
                          "Select the RUNS folder (a campaign, or the parent of several)")
             if PICK_FOLDER else DEFAULT_RUNS)
EPH  = DATASET
OUT  = SERVE_DIR
HTML = OUT / "cesium_mc.html"
SCAN = _scan_roots(RUNS_ROOT)
print("Dataset (ephemeris) →", EPH)
print("Runs root           →", RUNS_ROOT, f"({len(SCAN)} scan root(s))")
if not (EPH / "manifest.json").is_file():
    print(f"⚠ no manifest.json in {EPH} — pick the DATASET folder, "
          "not outputs/ or a runs folder.")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONIOENCODING"] = "utf-8"


def main():
    if REGENERATE:
        print("⏳ Generating globe from", RUNS_ROOT, "(this takes ~1–2 min)…")
        scan_args = []
        for r_ in SCAN:
            scan_args += ["--scan-root", str(r_)]
        # The subprocess is a FRESH interpreter that does NOT inherit the launcher's
        # module path (e.g. when run from Spyder/an IDE), so put ROOT on PYTHONPATH
        # for it - otherwise cesium_view.py cannot `import ntnconnect`.
        _env = dict(os.environ)
        _env["PYTHONPATH"] = str(ROOT) + os.pathsep + _env.get("PYTHONPATH", "")
        r = subprocess.run(
            [sys.executable, str(VIZ / "cesium_view.py"),
             "--eph-dir", str(EPH), *scan_args,
             "--out", str(HTML), "--step-s", STEP_S, "--port", "0"],
            cwd=str(ROOT), env=_env)
        if r.returncode != 0:
            print("Generation failed (cesium_view.py exited with code "
                  f"{r.returncode}). Re-run it directly to see the error:\n"
                  f"  python ntnconnect/analysis/visualization/cesium_view.py "
                  f"--eph-dir <dataset> --scan-root <runs> --out outputs/cesium_mc.html")
            return

    # Offline KPI data for the rail panels; built only when missing (or forced).
    interj = OUT / "analysis" / "interactive.json"
    if BUILD_INTERACTIVE is True or (BUILD_INTERACTIVE == "auto" and not interj.exists()):
        try:
            from ntnconnect.analysis.visualization.build_interactive import build_interactive
            print(" Building analysis/interactive.json OFFLINE (one-time, slow)…")
            build_interactive(SCAN, EPH, OUT / "analysis")
        except Exception as e:
            print("  interactive.json build skipped:", e)
    elif not interj.exists():
        print(" Analytics/Compare/Reports panels are empty until you build\n"
              "  analysis/interactive.json (set BUILD_INTERACTIVE='auto' above, or run\n"
              "  python -m ntnconnect.analysis.visualization.build_interactive "
              "--scan-root <runs> --eph-dir <dataset> --out outputs/analysis).")
    if not HTML.exists():
        print("No globe HTML found. Set REGENERATE=True and run again.")
        return

    # Serve outputs/ over http; fall back to a free port if PORT is busy.
    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=str(OUT))
    socketserver.TCPServer.allow_reuse_address = True
    port = PORT
    try:
        httpd = socketserver.TCPServer(("0.0.0.0", PORT), handler)
    except OSError:
        print(f" port {PORT} busy — using a free port instead.")
        httpd = socketserver.TCPServer(("0.0.0.0", 0), handler)
        port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    url = f"http://localhost:{port}/cesium_mc.html"
    print("\n Globe live at:", url)
    print("   (leave this script running; stop it to shut the server down)")
    webbrowser.open(url)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
