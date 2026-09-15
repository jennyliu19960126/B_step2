"""Print the fixed 24-run plan by default. Training requires --execute."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    plan = json.loads((ROOT / "configs/experiment_24.json").read_text(encoding="utf-8"))
    sys.path.insert(0, str(ROOT / "core_func"))
    from utility.evolution_policy import feature_origins
    assert len(plan["runs"]) == len({r["id"] for r in plan["runs"]}) == 24
    for row in plan["runs"]:
        path = ROOT / row["input_tensor"]
        if not path.is_file(): raise FileNotFoundError(path)
        meta = json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
        origins = feature_origins(meta)
        assert len(origins) == row["feature_count"]
        assert origins.count("base") == row["n_base"]
        assert origins.count("step1") == row["p"]
        assert row["population_size"] == len(origins) * row["population_multiplier"]
    if not args.execute:
        print(json.dumps(plan, indent=2))
        print("PLAN ONLY: no training started. Use --execute --output-root runs/<new_batch_name> to start.")
        return 0
    if args.output_root is None:
        parser.error("--execute requires a new --output-root")
    output = args.output_root.resolve()
    if not output.is_relative_to((ROOT / "runs").resolve()):
        parser.error("--output-root must be under this project's runs directory")
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    def run(row):
        target = output / row["id"]
        target.mkdir()
        env = os.environ.copy()
        env["GP_FEATURE_TENSOR"] = str(ROOT / row["input_tensor"])
        env["GP_OUTPUT_ROOT"] = str(target)
        command = [sys.executable, str(ROOT / "run_production.py"),
                   "--population-multiplier", str(row["population_multiplier"]),
                   "--min-raw-ic", str(row["min_raw_ic"]),
                   "--penalty-weight", str(plan["penalty_weight"]),
                   "--n-jobs", str(plan["workers_per_run"])]
        with open(target / "launcher.log", "w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        manifest = target / "backtest/production/test_manifest.json"
        completed = False
        if result.returncode == 0 and manifest.exists():
            completed = json.loads(manifest.read_text(encoding="utf-8")).get("status") == "completed"
        return {"id": row["id"], "returncode": result.returncode,
                "status": "completed" if completed else "failed", "output": str(target)}
    statuses = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=plan["concurrent_runs"]) as pool:
        futures = {pool.submit(run, row): row for row in plan["runs"]}
        for future in concurrent.futures.as_completed(futures):
            row = futures[future]
            try: status = future.result()
            except Exception as exc: status = {"id": row["id"], "status": "failed", "error": str(exc)}
            statuses.append(status)
            (output / "batch_status.json").write_text(json.dumps(statuses, indent=2), encoding="utf-8")
            print(json.dumps(status), flush=True)
    return int(any(row["status"] != "completed" for row in statuses))


if __name__ == "__main__":
    raise SystemExit(main())
