# Thinking-mode scoring ablation

Run `run_cot_scoring_experiment.py --samples 100 --repeats 2 --workers 4` from B_step2 using the server Python environment. Outputs are isolated under runs/cot_scoring_TIMESTAMP. This experiment does not change atomic hypothesis generation, GP scoring defaults, or shared score caches.

The production scorer already sends thinking.type=disabled, with max_tokens=32. Historical production_249_20260911_202918 took 9653.54 seconds, with 60836 API responses and 366626 completion tokens (6.03 per response). The penalty-zero run took 1934.83 seconds. This observational comparison does not isolate the causal contribution of request latency, expression population, caching, or numerical evaluation.

## Design

Select a deterministic sample balanced across historical integer-score strata from the shared cache, then freeze expressions. Randomize request order. Score every expression twice per mode, with the same prompt, model, temperature setting, token budget (8192), and timeout (180 seconds). Use fresh API calls in both arms. This isolates thinking at a common generous budget; it is not an exact latency benchmark of production max_tokens=32. Raw responses retain usage, model, elapsed time, completion status, and reasoning character count, but not reasoning text or credentials.

Compare expression-level mean scores using a paired Wilcoxon test and a bootstrap confidence interval for the mean difference. Report absolute differences, Spearman rank correlation, and implied penalty changes at weight 0.03. The expression is the statistical unit; repetitions are averaged, not treated as independent expressions. Report failed and truncated calls separately, excluding incomplete pairs from score comparisons. A nonsignificant p-value does not establish equivalence. Differences do not establish which mode is more accurate without expert labels or downstream validation. Balanced sampling is exploratory, not representative of production frequencies.

## Pilot

2026-09-15: 40 expressions, 2 repetitions per mode, 160 requests, 4 worker processes. Output: runs/cot_scoring_pilot_20260915. Read manifest.json for completion state, responses.jsonl for incremental results, and summary.json for final statistics. No automatic retries; errors remain visible.

Pilot result: both arms had 0/80 successful scores. A separate diagnostic request returned HTTP 401 (invalid configured API key). No CoT comparison is available. Update the existing credential or provide DEEPSEEK_API_KEY in the launch environment, then rerun to a new output directory. The runner now performs a two-mode preflight before submitting the batch.

## Successful retry (2026-09-15)

Updated the step2 credential and reran the pilot: runs/cot_scoring_retry_20260915_072837. All 160 measured requests succeeded. Disabled/enabled mean latency: 0.424/3.524 seconds (8.30x); completion tokens per request: 6.175/638.925 (103.47x, including reasoning). Across 40 expression-level pairs, enabled minus disabled mean score = -0.1625, bootstrap 95% CI [-0.54375, 0.23125], Wilcoxon p=0.282. Mean absolute score difference = 0.9125; 42.5% of expressions differ by at least 1 point; Spearman = 0.8783. Thus no significant overall directional score shift was detected, but individual scores are not equivalent. At penalty weight 0.03 the mean absolute fitness adjustment difference is 0.0027375. These exploratory results favor retaining disabled thinking for speed, without establishing relative score accuracy.
