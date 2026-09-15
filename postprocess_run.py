"""Materialize GP snapshots once, then apply the shared filter independently."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "4"
import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
_EXECUTOR = _DATES = _STOCKS = _ASSET_STOCKS = _VALUES = None


def _materialize_one(row):
    name, stack = row
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("Invalid factor filename")
    value = np.asarray(_EXECUTOR.execute_formulation(stack), dtype=float)
    if value.shape != (len(_DATES), len(_STOCKS)):
        raise ValueError(f"Unexpected factor shape: {name}: {value.shape}")
    # Shared filter expects stock rows, quarter columns, and the asset stock axis.
    frame = pd.DataFrame(value.T, index=_STOCKS, columns=_DATES).reindex(_ASSET_STOCKS)
    frame.replace([np.inf, -np.inf], np.nan).to_csv(_VALUES / f"{name}.csv")
    return name


def materialize(catalog, values, assets, workers):
    global _EXECUTOR, _DATES, _STOCKS, _ASSET_STOCKS, _VALUES
    sys.path.insert(0, str(ROOT / "core_func"))
    from step.factor_replicate import FactorReplicate
    from data_reader.data_reader_csv import load_feature
    from constant.params import feature_names, DATES_quar, STOCKS
    _EXECUTOR = FactorReplicate.__new__(FactorReplicate)
    _EXECUTOR.X = load_feature(feature_names)
    _DATES, _STOCKS, _VALUES = DATES_quar, STOCKS, values
    _ASSET_STOCKS = pd.read_csv(assets, usecols=[0], dtype=str).iloc[:, 0].tolist()
    if len(set(_ASSET_STOCKS)) != len(_ASSET_STOCKS):
        raise ValueError("Duplicate asset stock IDs")
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as pool:
        list(pool.map(_materialize_one, catalog[["factor_name", "formulation_stack"]].itertuples(index=False, name=None)))


def process(run_dir, workers=40, ic_cut=0.04, corr_cut=0.6,
            assets=Path("/data/fund_agent/data/gate1_features/T_ASSETS.csv"),
            filter_script=Path("/data/fund_agent/factor_filter/filter.py"), materializer=materialize):
    if not 2 <= workers <= 80:
        raise ValueError("workers must be 2..80")
    run_dir, assets, filter_script = Path(run_dir), Path(assets), Path(filter_script)
    output = run_dir / "postprocess"
    output.mkdir(exist_ok=False)
    report = {"status": "running", "stage": "read_catalogs", "workers": workers,
              "ic_cut": ic_cut, "corr_cut": corr_cut, "score_column": "base_fitness", "snapshots": {}}
    def save():
        (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    save()
    try:
        catalogs = {}
        for cutoff in (4, 8):
            path = run_dir / f"result_production_through_gen{cutoff}.csv"
            frame = pd.read_csv(path)
            if frame.factor_name.isna().any() or frame.factor_name.duplicated().any():
                raise ValueError("Snapshot factor IDs must be unique")
            if not frame.empty and not frame.dependency_kind.eq("contains_step1").all():
                raise ValueError("Snapshot contains base-only or constant-only factors")
            catalogs[cutoff] = (path, frame)
        merged = pd.concat([frame for _, frame in catalogs.values()], ignore_index=True)
        if (merged.groupby("factor_name").formulation_stack.nunique() > 1).any():
            raise ValueError("Conflicting expressions for one factor ID")
        union = merged.drop_duplicates("factor_name")
        scores = pd.to_numeric(union["base_fitness"], errors="raise")
        union = union[np.isfinite(scores) & (scores >= ic_cut) & (scores <= 1)].copy()
        values = output / "quarterly_values"
        values.mkdir()
        report["stage"] = "materialize"; save()
        if not union.empty:
            materializer(union, values, assets, workers)
        report["materialized_count"] = len(union)
        for cutoff, (catalog_path, frame) in catalogs.items():
            report["stage"] = f"filter_gen{cutoff}"; save()
            target = output / f"through_gen{cutoff}"
            if frame.empty:
                # Shared filter rejects an empty catalog; zero candidates are a valid result.
                target.mkdir()
                frame.to_csv(target / "selected.csv", index=False)
                frame.assign(filter_status=pd.Series(dtype=str)).to_csv(target / "decisions.csv", index=False)
                summary = {"status": "completed", "total": 0, "selected": 0,
                           "reason": "empty_catalog", "ic_cut": ic_cut, "corr_cut": corr_cut,
                           "score_column": "base_fitness", "quarters": 32, "min_quarters": 22}
                (target / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            else:
                command = [sys.executable, str(filter_script), "--catalog", str(catalog_path),
                           "--factor-dir", str(values), "--output-dir", str(target),
                           "--score-column", "base_fitness", "--assets", str(assets),
                           "--start", "2017-03-31", "--end", "2024-12-31",
                           "--ic-cut", str(ic_cut), "--corr-cut", str(corr_cut), "--workers", str(workers)]
                subprocess.run(command, check=True)
                summary = json.loads((target / "summary.json").read_text(encoding="utf-8"))
                if summary.get("status") != "completed": raise RuntimeError("Filter incomplete")
            report["snapshots"][str(cutoff)] = {"before_filter": len(frame),
                "after_filter": summary["selected"], "selected_csv": str(target / "selected.csv")}
            save()
        report.update(status="completed", stage="completed"); save()
        return report
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}"); save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--ic-cut", type=float, default=0.04)
    parser.add_argument("--corr-cut", type=float, default=0.6)
    args = parser.parse_args()
    print(json.dumps(process(**vars(args)), indent=2))


if __name__ == "__main__": main()
