import ast
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core_func"))
from utility.evolution_policy import (population_target, feature_origins,
    dependency_kind, passes_raw_ic, select_final_candidates)


def source_method(file, cls, method, namespace):
    tree = ast.parse((ROOT / file).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    func = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == method)
    exec(compile(ast.Module(body=[func], type_ignores=[]), str(file), "exec"), namespace)
    return namespace[method]


class PolicyTests(unittest.TestCase):
    def test_population_size(self):
        self.assertEqual(population_target(249, 50), 12450)
        for value in (0, -1, 1.5, True):
            with self.assertRaises(ValueError): population_target(249, value)

    def test_provenance_requires_known_sources(self):
        meta = {"feature_names": ["a", "b"], "source_mapping": [
            {"source_group": "gate1"}, {"source_group": "x20_ic0.02_corr0.8"}]}
        self.assertEqual(feature_origins(meta), ["base", "step1"])
        meta["source_mapping"][1]["source_group"] = "unknown"
        with self.assertRaises(ValueError): feature_origins(meta)

    def test_dependency_classification(self):
        origins = ["base", "step1"]
        self.assertEqual(dependency_kind("['cs_add', 0, 1.0]", origins), "base_only")
        self.assertEqual(dependency_kind(["cs_add", 0, 1], origins), "contains_step1")
        self.assertEqual(dependency_kind([1], origins), "contains_step1")
        self.assertEqual(dependency_kind([1.0], origins), "constant_only")
        with self.assertRaises(ValueError): dependency_kind([2], origins)

    def test_gate_boundary_and_empty_pool(self):
        programs = [SimpleNamespace(base_fitness_=v) for v in (0.009, 0.01, float("nan"), None)]
        method = source_method("core_func/core/genetic.py", "BaseSymbolic", "_filter_population",
                               {"passes_raw_ic": passes_raw_ic})
        gp = SimpleNamespace(min_raw_ic=0.01, logger=SimpleNamespace(info=lambda x: None))
        self.assertEqual(method(gp, programs, 1), [programs[1]])
        with self.assertRaises(RuntimeError): method(gp, [programs[0]], 1)

    def test_low_ic_does_not_call_llm(self):
        calls = []
        class Function: pass
        namespace = {"np": np, "_Function": Function,
                     "interpretability_penalty": lambda *a, **k: (calls.append(a) or (8, 0.002))}
        method = source_method("core_func/core/_program.py", "_Program", "raw_fitness_3D", namespace)
        for raw, expected_calls in ((0.009, 0), (0.01, 1)):
            calls.clear()
            metric = lambda *a: (raw, np.array([raw]))
            program = SimpleNamespace(execute_3D=lambda x: x, transformer=None, metric=metric,
                       gen_metric_list=[metric], program=[0], feature_names=["base"],
                       min_raw_ic=0.01, llm_interpretability={"cache_path": "unused"})
            fitness, _, _ = method(program, np.ones((1, 1)), None, None, None, None)
            self.assertEqual(len(calls), expected_calls)
            self.assertAlmostEqual(fitness, raw - (0.002 if expected_calls else 0))

    def test_four_generation_snapshot_excludes_future_and_base_only(self):
        frame = pd.DataFrame([
            [1, "base", [0], 0.08], [2, "mixed", ["add", 0, 1], 0.04],
            [3, "constant", [1.0], 0.05], [4, "weak", [1], 0.005],
            [5, "future", [1], 0.07], [8, "mixed", ["add", 0, 1], 0.04]],
            columns=["generation", "formulation", "formulation_stack", "IC"])
        selected, counts = select_final_candidates(frame, ["base", "step1"], "IC", 0.01, 4)
        independent, _ = select_final_candidates(frame[frame.generation <= 4], ["base", "step1"], "IC", 0.01)
        pd.testing.assert_frame_equal(selected, independent)
        self.assertEqual(selected.formulation.tolist(), ["mixed"])
        self.assertEqual(counts["base_only"], 1)
        all_selected, _ = select_final_candidates(frame, ["base", "step1"], "IC", 0.01, 8)
        self.assertEqual(all_selected.formulation.tolist(), ["mixed", "future"])


if __name__ == "__main__": unittest.main()
