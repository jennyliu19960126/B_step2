import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import postprocess_run as pp


class PostprocessTests(unittest.TestCase):
    @unittest.skipUnless(Path("/data/fund_agent/factor_filter/filter.py").exists(), "server filter required")
    def test_real_filter_two_snapshots_and_orientation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stocks = [f"s{i:03}" for i in range(120)]
            dates = pd.date_range("2017-03-31", periods=32, freq="QE").strftime("%Y-%m-%d").tolist()
            assets = root / "assets.csv"
            pd.DataFrame(1., index=stocks, columns=dates).to_csv(assets)
            rng = np.random.RandomState(42)
            first = rng.normal(size=(32, 120))
            values = {"a": first, "b": first * 2, "c": rng.normal(size=(32, 120))}
            frame = pd.DataFrame({"factor_name": ["a", "b", "c"], "formulation_stack": ["[0]", "[1]", "[2]"],
                "base_fitness": [.08, .07, .06], "dependency_kind": ["contains_step1"] * 3})
            frame.iloc[:2].to_csv(root / "result_production_through_gen4.csv", index=False)
            frame.to_csv(root / "result_production_through_gen8.csv", index=False)
            calls = []
            def materialize(catalog, output, asset_path, workers):
                calls.append(len(catalog))
                pp._DATES, pp._STOCKS, pp._ASSET_STOCKS, pp._VALUES = dates, stocks, stocks, output
                for row in catalog.itertuples():
                    pp._EXECUTOR = SimpleNamespace(execute_formulation=lambda stack, n=row.factor_name: values[n])
                    pp._materialize_one((row.factor_name, row.formulation_stack))
            report = pp.process(root, workers=2, assets=assets, materializer=materialize)
            self.assertEqual(calls, [3])
            self.assertEqual(report["snapshots"]["4"]["after_filter"], 1)
            self.assertEqual(report["snapshots"]["8"]["after_filter"], 2)
            self.assertEqual(report["status"], "completed")
            with self.assertRaises(FileExistsError): pp.process(root, assets=assets)

    @unittest.skipUnless(Path("/data/fund_agent/factor_filter/filter.py").exists(), "server filter required")
    def test_below_final_ic_skips_materialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dates = pd.date_range("2017-03-31", periods=32, freq="QE").strftime("%Y-%m-%d")
            assets = root / "assets.csv"
            pd.DataFrame(1., index=[f"s{i}" for i in range(100)], columns=dates).to_csv(assets)
            frame = pd.DataFrame({"factor_name": ["low"], "formulation_stack": ["[0]"],
                "base_fitness": [.039], "dependency_kind": ["contains_step1"]})
            for gen in (4, 8): frame.to_csv(root / f"result_production_through_gen{gen}.csv", index=False)
            def forbidden(*args): raise AssertionError("Below-cut factor should not materialize")
            report = pp.process(root, workers=2, assets=assets, materializer=forbidden)
            self.assertEqual(report["materialized_count"], 0)
            self.assertEqual(report["snapshots"]["4"]["after_filter"], 0)

    def test_empty_snapshots_do_not_materialize(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = pd.DataFrame(columns=["factor_name", "formulation_stack", "base_fitness", "dependency_kind"])
            for gen in (4, 8): frame.to_csv(root / f"result_production_through_gen{gen}.csv", index=False)
            def forbidden(*args): raise AssertionError("Empty input should not materialize")
            report = pp.process(root, workers=2, materializer=forbidden)
            self.assertEqual(report["snapshots"]["8"]["after_filter"], 0)

    def test_materialization_failure_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = pd.DataFrame({"factor_name": ["a"], "formulation_stack": ["[0]"],
                "base_fitness": [.05], "dependency_kind": ["contains_step1"]})
            for gen in (4, 8): frame.to_csv(root / f"result_production_through_gen{gen}.csv", index=False)
            def broken(*args): raise RuntimeError("test materialization failure")
            with self.assertRaises(RuntimeError): pp.process(root, workers=2, materializer=broken)
            import json
            report = json.loads((root / "postprocess/summary.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["stage"], "materialize")


if __name__ == "__main__": unittest.main()
