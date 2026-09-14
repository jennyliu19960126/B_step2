"""Server integration check with synthetic candidates; no API or model training."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace, MethodType
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core_func"))
from core.genetic import SymbolicTransformer

class Candidate:
    def __init__(self, value, name):
        self.base_fitness_ = value
        self.raw_fitness_ = value
        self.length_ = 1
        self.parents = None
        self.formulation_stack = [1]
        self.feature_stack = ["step1"]
        self.gen_performance_ = [value]
        self.llm_interpretability_score_ = None
        self.llm_interpretability_penalty_ = 0.0
        self.name = name
    def fitness(self, coefficient=None): return self.raw_fitness_
    def execute_3D(self, X): return X[:, 1, :]
    def __str__(self): return self.name


def run(generations, root):
    gp = SymbolicTransformer(population_size=4, hall_of_fame=4, n_components=2,
        generations=generations, tournament_size=2, function_set=["cs_add"],
        metric="IC", gen_metric_list=["IC"], feature_names=["base", "step1"],
        init_depth=[1, 2], warm_start=True, random_state=123, n_jobs=1,
        stopping_criteria=2.0, low_memory=False)
    gp.min_raw_ic = 0.01
    gp.sample_weight = np.ones(3, dtype=int)
    gp.logger = SimpleNamespace(info=lambda *a: None)
    gp.save_dir = str(root)
    gp.save_gen_dir = str(root)
    gp.dir_kw = "smoke"
    calls = []
    def execute(self, parents, X, y, sample_weight, population_size, params,
                random_state, need_parallel, generation, index_range):
        calls.append(population_size)
        draw = int(random_state.randint(100000))
        # Exactly one survivor in every generation exercises shrinking pools
        # and final hall_of_fame larger than the retained population.
        return [Candidate(0.03 if i == 0 else 0.001, f"g{generation}_{draw}_{i}")
                for i in range(population_size)]
    gp._execute_generation = MethodType(execute, gp)
    X = np.arange(24, dtype=float).reshape(3, 2, 4)
    result = gp.fit_3D(X, np.zeros((3, 4)), need_parallel=False, fitness_threshold=0)
    assert calls == [20] + [4] * generations, calls
    assert len(gp._best_programs) == 1
    assert gp.run_details_["generation"] == list(range(1, generations + 1))
    return result

with tempfile.TemporaryDirectory(prefix="gp_policy_smoke_") as tmp:
    root = Path(tmp)
    (root / "four").mkdir()
    (root / "eight").mkdir()
    four = run(4, root / "four")
    eight = run(8, root / "eight")
    pd.testing.assert_frame_equal(four.reset_index(drop=True),
        eight[eight.generation <= 4].reset_index(drop=True))
print("PASS: shrinking population, single-survivor final selection, 5x initialization, 4/8 prefix equivalence")
