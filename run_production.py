"""Reproducible three-layer smoke test for the financial-factor GP framework.

The production configuration remains untouched.  Run this script with:

    python run_production.py
"""

from __future__ import annotations

import json
import os
import platform
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

for thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[thread_var] = "4"
if "GP_OUTPUT_ROOT" not in os.environ:
    os.environ["GP_OUTPUT_ROOT"] = str(Path(__file__).resolve().parent / "runs" / ("output_production_" + time.strftime("%Y%m%d_%H%M%S")))
os.environ["GP_LLM_USAGE_DIR"] = str(Path(os.environ["GP_OUTPUT_ROOT"]) / "api_usage")
import numpy as np

from core_func import GpLearnStock


TEST_PARAMETERS = {
    "population_size": 10000,
    "hall_of_fame": 1000,
    "n_components": 10,
    # Warm start evaluates 5x population, then eight evolutionary generations.
    "generations": 8,
    "tournament_size": 1000,
    "init_depth": [1, 4],
    "n_jobs": 80,
    "random_state": 20260908,
    "low_memory": False,
    # The framework's detailed per-program file logger remains enabled.
    "verbose": 0,
    # Every generated candidate is scored before its numerical IC evaluation.
    "llm_interpretability_enabled": True,
    "llm_interpretability_weight": 0.03,
}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    started_at = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    start = time.time()

    gp = GpLearnStock(**TEST_PARAMETERS)
    gp.set_params_backtest(need_parallel=True)
    # Distinguish this test from production even if GP_OUTPUT_ROOT is omitted.
    gp.dir_kw = "production"
    gp._prep_save_dir()
    gp._prep_logger()
    gp.llm_interpretability_cache_path = os.path.join(
        gp.save_dir, "llm_interpretability.sqlite"
    )

    output_dir = Path(gp.save_dir).resolve()
    manifest_path = output_dir / "test_manifest.json"
    manifest = {
        "status": "running",
        "started_at": started_at,
        "python": sys.version,
        "platform": platform.platform(),
        "parameters": TEST_PARAMETERS,
        "fitness_metric": gp.metric,
        "generation_metrics": gp.gen_metric_list,
        "feature_count": len(gp.feature_names),
        "output_dir": str(output_dir),
    }
    _write_json(manifest_path, manifest)

    gp.prep_data_backtest()
    manifest["data"] = {
        "X_shape": list(gp.X.shape),
        "X_dtype": str(gp.X.dtype),
        "y_shape": list(gp.y.shape),
        "evaluated_quarters": int(gp.sample_weight.sum()),
        "X_finite_rate": float(np.isfinite(gp.X).mean()),
        "y_finite_rate": float(np.isfinite(gp.y).mean()),
    }
    _write_json(manifest_path, manifest)

    assert gp.X.shape[0] == 32 and gp.X.shape[1] == 249
    assert gp.y.shape == (32, gp.X.shape[2])
    from core_func.constant.params import DATES_quar, feature_tensor_path
    assert DATES_quar[0] == "2017-03-31" and DATES_quar[-1] == "2024-12-31"
    assert np.isfinite(gp.X).sum() == np.count_nonzero(~np.isnan(gp.X))
    allowed = gp.fitness_cache[1] == 0
    coverage = (np.isfinite(gp.X) & allowed[:, None, :]).sum(axis=2) / allowed.sum(axis=1)[:, None]
    assert np.isfinite(coverage).all() and (coverage >= .9).all(), "Input coverage preflight failed"
    manifest["minimum_input_coverage"] = float(coverage.min())
    manifest["input_tensor"] = str(feature_tensor_path)
    manifest["training_start"] = DATES_quar[0]
    manifest["training_end"] = DATES_quar[-1]
    manifest["possible_ic_quarters"] = 24
    manifest["fitness_formula"] = "abs(mean(signed quarterly IC)) - 0.03 * (1 - LLM_score / 10)"
    _write_json(manifest_path, manifest)
    gp.learn_formulation()

    raw_result = output_dir / "raw_result_production.csv"
    original_result = output_dir / "original_result_production.csv"
    if not raw_result.exists() or not original_result.exists():
        manifest["status"] = "failed"
        manifest["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
        manifest["elapsed_seconds"] = time.time() - start
        _write_json(manifest_path, manifest)
        raise RuntimeError("GP run did not create its expected result files")

    programs = [
        program
        for generation in gp._programs
        if generation is not None
        for program in generation
        if program is not None
    ]
    depth_counts = Counter(program.depth_ for program in programs)
    llm_audit = None
    llm_cache = Path(gp.llm_interpretability_cache_path)
    if TEST_PARAMETERS["llm_interpretability_enabled"] and llm_cache.exists():
        with sqlite3.connect(llm_cache) as connection:
            count, minimum, average, maximum = connection.execute(
                "SELECT COUNT(*), MIN(score), AVG(score), MAX(score) "
                "FROM llm_interpretability"
            ).fetchone()
            outcome_counts = dict(connection.execute(
                "SELECT outcome, COUNT(*) FROM llm_interpretability_audit "
                "GROUP BY outcome"
            ).fetchall())
            candidate_scoring_attempts = connection.execute(
                "SELECT COUNT(*) FROM llm_interpretability_audit"
            ).fetchone()[0]
        expected_candidate_evaluations = (
            TEST_PARAMETERS["population_size"] * 5
            + TEST_PARAMETERS["population_size"] * TEST_PARAMETERS["generations"]
        )
        llm_audit = {
            "expected_candidate_evaluations": expected_candidate_evaluations,
            "candidate_scoring_attempts": candidate_scoring_attempts,
            "all_candidates_audited": candidate_scoring_attempts == expected_candidate_evaluations,
            "unique_scored_expressions": count,
            "minimum_score": minimum,
            "average_score": average,
            "maximum_score": maximum,
            "outcome_counts": outcome_counts,
        }
    manifest.update(
        {
            "status": "completed",
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "elapsed_seconds": time.time() - start,
            "depth_audit": {
                "retained_program_count": len(programs),
                "maximum_observed_depth": max(depth_counts, default=None),
                "depth_histogram": {
                    str(depth): count for depth, count in sorted(depth_counts.items())
                },
                "passed": bool(programs)
                and max(depth_counts, default=999) <= TEST_PARAMETERS["init_depth"][1],
            },
            "llm_audit": llm_audit,
        }
    )
    usage_records = [json.loads(line) for f in Path(os.environ["GP_LLM_USAGE_DIR"]).glob("*.jsonl") for line in f.read_text().splitlines()]
    token_fields = ["prompt_tokens", "completion_tokens", "total_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens"]
    manifest["token_usage"] = {"responses": len(usage_records), "missing_usage": sum(r["usage"] is None for r in usage_records),
                               **{key: sum((r["usage"] or {}).get(key, 0) for r in usage_records) for key in token_fields}}
    _write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
