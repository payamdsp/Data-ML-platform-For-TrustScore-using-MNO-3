"""Runner for the Trust Score 0.5 Standalone Track (Notebooks 00 to 04).

Run this directly from terminal:
    & "$env:LOCALAPPDATA\Programs\Python\Python314\python.exe" run_pipeline.py
"""

import json
import os
import pathlib
import sys
import time
import traceback
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend to save plots without blocking

def display_shim(*args, **kwargs):
    for arg in args:
        if hasattr(arg, "to_string"):
            print(arg.to_string())
        else:
            print(arg)

NOTEBOOKS = [
    "00-setup-and-data.ipynb",
    "01-explore-and-validate.ipynb",
    "02-select-features.ipynb",
    "03-train-and-score.ipynb",
    "04-compare-and-report.ipynb",
]

def run_notebook(nb_path: pathlib.Path, base_dir: pathlib.Path):
    print("\n" + "=" * 70)
    print(f"  EXECUTING: {nb_path.name}")
    print("=" * 70)

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    global_env = {
        "__name__": "__main__",
        "__file__": str(nb_path),
        "display": display_shim,
    }

    os.chdir(base_dir)
    start_time = time.time()
    code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]

    for i, cell in enumerate(code_cells):
        src_lines = cell.get("source", [])
        src = "".join(src_lines) if isinstance(src_lines, list) else src_lines

        if not src.strip():
            continue

        try:
            exec(src, global_env)
        except Exception as e:
            print(f"\n[ERROR] Cell {i+1} failed in {nb_path.name}:")
            print("-" * 50)
            print(src[:300] + ("..." if len(src) > 300 else ""))
            print("-" * 50)
            traceback.print_exc()
            raise e

    elapsed = time.time() - start_time
    print(f"\n[DONE] {nb_path.name} finished in {elapsed:.1f}s")

def main():
    root = pathlib.Path(__file__).resolve().parent
    nb_dir = root / "notebooks" / "0.0-standalone"

    print("=" * 70)
    print(" TRUST SCORE 0.5 - LOCAL STANDALONE PIPELINE RUNNER")
    print("=" * 70)
    print(f"Workspace Directory : {root}")
    print(f"Notebooks Directory : {nb_dir}")

    total_start = time.time()
    for nb_name in NOTEBOOKS:
        nb_path = nb_dir / nb_name
        if not nb_path.exists():
            print(f"Error: {nb_path} not found!")
            sys.exit(1)
        run_notebook(nb_path, root)

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"  ALL 5 STAGES COMPLETED SUCCESSFULLY IN {total_elapsed:.1f}s!")
    print("=" * 70)
    print("\nGenerated Artifacts in ./_run/:")
    run_dir = root / "_run"
    if run_dir.exists():
        for p in sorted(run_dir.iterdir()):
            if p.is_file():
                print(f"  - {p.name:<30} ({p.stat().st_size:,} bytes)")
            elif p.is_dir():
                print(f"  - [{p.name}/] directory ({len(list(p.iterdir()))} files)")
    print("\nTo view the full report, check: _run/report.md")

if __name__ == "__main__":
    main()
