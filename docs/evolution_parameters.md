# Population and final-output policy

## Production parameters

- `population_multiplier` (default 50): formal population target is `population_multiplier * (p + n_base)`, where the total is the actual number of input tensor features. Warm initialization evaluates five times this target and keeps the best eligible candidates. An explicit constructor `population_size` remains available for small tests and overrides the formula.
- `min_raw_ic` (default 0.01): discard candidates with nonfinite raw IC or raw IC strictly below this threshold. Equality passes. This uses `base_fitness = abs(mean(signed quarterly Pearson IC))`, before the interpretability penalty. `None` in the constructor/config disables the numeric cutoff but still rejects invalid IC.
- `--penalty-weight`: 0.01 or 0.03, fixed for the entire run.
- Production runs eight evolutionary generations after Gen0 initialization. The stopping threshold is above the attainable IC range so quality-based early stopping does not truncate the comparison.

Example (set a fresh GP_OUTPUT_ROOT and provide the local config/data first):

```bash
/data/miniconda3/envs/py310/bin/python run_production.py --population-multiplier 50 --min-raw-ic 0.01 --penalty-weight 0.03
```

The hard IC gate runs before LLM scoring. Rejected candidates never enter the next breeding pool. Each generation still generates the formal target number of offspring; if fewer candidates qualify, the retained pool is smaller. No unbounded refill is performed. An empty eligible pool raises a clear error. Initialization preserves fivefold exploration. API audit counts need not equal numerical candidate evaluation counts.

## Input provenance and base-only exclusion

Input metadata must supply `feature_origins`, an aligned list of `base`/`step1`, or an aligned `source_mapping`. Mapping entries can explicitly specify `origin`. Existing source groups `gate1`/`base` map to base; `step1` and the existing `x20_`, `x40_`, `x60_` groups map to Step1. Unknown provenance raises an error instead of silently counting factors.

Expressions are classified using integer terminal indices from `formulation_stack`. Float constants are not feature indices. A derived Step1 terminal remains Step1 even if its own underlying formula uses base financial fields.

`raw_result_*` and `original_result_*` retain base-only expressions for audit and breeding is unchanged by this output policy. Final `result_*` CSVs exclude base-only and constant-only expressions, so downstream replication, counting and external correlation filtering consume only expressions containing Step1 inputs. Existing final penalized-fitness thresholds remain unchanged. These CSV counts are before the separate external correlation filter.

## Four/eight-generation comparison

A single eight-generation trajectory supplies both cumulative candidate sets:

- `result_<run>_through_gen4.csv`: candidates recorded in evolutionary generations 1-4, deduplicated and filtered.
- `result_<run>_through_gen8.csv`: the same policy through generation 8.
- `final_counts.json`: unique candidate, base-only, constant-only and final counts for each cutoff.

Gen0 is the initialization pool and is not directly included in the historical cumulative result format. Cutoffs use recorded generation numbers, not the final surviving population. The four-generation file excludes information from generations 5-8. Snapshot CSVs are written after a successful fit; per-generation raw CSVs are persisted while fitting. Failed/incomplete runs must not be reported as complete eight-generation experiments.

## Checks

```bash
/data/miniconda3/envs/py310/bin/python -m unittest discover -s tests -v
/data/miniconda3/envs/py310/bin/python tests/smoke_evolution_loop.py
```

The integration smoke test runs the real evolutionary loop with synthetic candidates, including a one-survivor pool, and verifies independent four-generation versus eight-generation-prefix equality. It does not call an LLM or perform a production data training run.
