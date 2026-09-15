# 财务因子遗传规划框架

Current population sizing, hard IC gate and base-only output policy: [evolution parameters](docs/evolution_parameters.md). These settings supersede the historical fixed-population examples below.

本项目使用季度财务特征构造遗传规划（Genetic Programming，GP）表达式，并以“季频原始 Pearson IC + DeepSeek 可解释性惩罚”选择候选因子。

当前阶段的首要目标是先跑通以下最小闭环：

```text
季度财务特征 X
  -> 生成候选表达式
  -> DeepSeek 对每个候选表达式评分
  -> 计算季度横截面 Pearson IC
  -> fitness = IC - 可解释性惩罚
  -> 遗传选择、交叉、变异
  -> CSV / SQLite / 日志 / manifest
```

当前迭代过程不计算 `neu_IC`、`long_only_neu_IC` 或 `hedge_sharpe`，因此不会加载 Barra 风格暴露，也不会让这些辅助指标参与候选评价或选择。

## 1. 主要入口

| 入口 | 用途 | 当前建议 |
|---|---|---|
| `step1_learn.py` | 使用生产配置运行完整 GP | 生产规模较大，三层验证通过后再运行 |
| `run_three_layer_test.py` | 独立的三层端到端验证，并生成审计 manifest | 当前首选入口 |
| `step2_factor_final.py` | 读取 Step 1 的入选表达式，打印数量和前几行，并复算季度因子 CSV | Step 1 成功后使用 |
| `step3_backtest.py` | 日频化因子、计算原始 Rank IC/组合收益并生成 PNG 图 | 当前可直接运行；默认不加载风格暴露 |
| `core_func/step/ic_analysis.py` | 汇总回测结果并按中性化 IC、相关性筛选 | 依赖完整中性化回测链路，当前最小闭环不启用 |

## 2. 运行环境和数据

当前验证环境：

```text
Python 3.10.20
工作目录 /data/fund_agent/B_step2
```

主要 Python 依赖包括 `numpy`、`pandas`、`scikit-learn`、`joblib`、`bottleneck`、`statsmodels`、`scipy`、`tqdm`、`matplotlib`、`seaborn`、`xlsxwriter` 和 `openpyxl`。

仓库内的 `.venv` 不是当前可执行环境，请使用已经安装上述依赖的 Python。当前 shell 中的 `python` 指向可用的 Conda Python。

### 2.1 财务特征

默认输入：

```text
artifacts/financial_feature_tensor.npy
artifacts/financial_feature_tensor.metadata.json
artifacts/financial_feature_tensor.coverage.csv
```

张量结构为：

```text
X[quarter, feature, stock] = X[36, 503, 5392]
dtype = float32
```

metadata 文件定义季度、特征和股票三个轴的实际顺序。不要只替换 `.npy` 而不同时替换对应 metadata。

如需指定另一份兼容 tensor：

```bash
export GP_FEATURE_TENSOR=/absolute/path/to/financial_feature_tensor.npy
```

重新构建默认 tensor：

```bash
cd /data/fund_agent/B_step2
python build_financial_feature_tensor.py
```

### 2.2 收益、股票限制和行业数据

默认数据根目录为：

```text
/data/research
```

代码从其 `vars/Base` 等子目录读取收益和股票限制文件。若在其他机器使用兼容数据包：

```bash
export GP_BACKTEST_ROOT=/absolute/path/to/backtest_root
```

若数据根目录没有 `basic_info/calendar.csv`，交易日历会从 `vars/Base/S_RESTRICT.csv` 的日期轴推导。

## 3. 配置文件地址和配置方式

### 3.1 主配置

主配置文件：

```text
core_func/constant/config.json
```

修改 JSON 后重新启动 Python 进程才会生效；配置在模块导入时读取，不支持在同一个进程中热更新。

当前核心配置：

| 配置项 | 当前值 | 含义 |
|---|---:|---|
| `fitness_metric` | `IC` | 优化季频、原始值、带方向的 Pearson IC |
| `gen_metric_list` | `["IC"]` | 每个候选只计算当前 IC fitness，不计算风格暴露相关辅助指标 |
| `fitness_threshold` | `0` | 只有最终 fitness 大于 0 的候选进入代际结果表 |
| `ic_cut_dict_quar.IC` | `0.02` | 最终结果文件的 IC 门槛 |
| `population_size` | `10000` | 生产正式种群大小 |
| `generations` | `8` | 框架循环采用包含式上界；非 warm start 时会运行 Gen 0–8 |
| `warm_start` | `1` | 先生成正式种群 5 倍的候选，再筛选出初始种群 |
| `tournament_size` | `1000` | 锦标赛选择规模 |
| `hall_of_fame` | `1000` | 最终去相关前的候选池 |
| `init_depth` | `[1,4]` | 生产表达式深度范围及严格最大深度 4 |
| `parsimony_coefficient` | `0.0` | 不额外按表达式长度扣分 |
| `corr_penalty` | `false` | 不在遗传选择中增加 IC 相关性惩罚 |
| `n_jobs` | `80` | 生产 GP worker 数量 |
| `random_state` | `null` | 生产默认不固定随机种子 |

`p_crossover`、`p_subtree_mutation`、`p_hoist_mutation`、`p_point_mutation` 和 `p_point_replace` 等遗传参数目前与原版保持一致。

### 3.2 DeepSeek 配置

同一配置文件中的相关字段：

| 配置项 | 作用 |
|---|---|
| `llm_interpretability_enabled` | 是否启用可解释性评分；当前为 `true` |
| `llm_interpretability_model` | DeepSeek 模型名 |
| `llm_interpretability_base_url` | API 根地址 |
| `llm_interpretability_timeout` | 单次请求超时秒数 |
| `llm_interpretability_weight` | 最大惩罚；当前为 `0.05` |
| `llm_interpretability_min_ic` | 兼容旧配置而保留，当前不作为调用门槛 |
| `llm_interpretability_api_key` | API 凭据；不要复制到日志或文档 |

推荐通过环境变量提供凭据，并从配置文件中移除明文值：

```bash
export DEEPSEEK_API_KEY='你的密钥'
```

每个候选表达式都会进入评分函数。相同的“表达式 + 模型”会读取当前输出目录中的 SQLite 缓存；缓存未命中时才发起外部 API 请求。多 worker 首次同时遇到同一个新表达式时，当前实现仍可能发生极少量并发重复请求，审计时应同时比较 `api_success` 和唯一缓存行数。请求失败采用 fail-open：该候选的 LLM 分数为空、惩罚为 0，GP 任务不会因此整体退出。

适应度计算公式：

```text
base_fitness = abs(mean_q(corr(factor[q, :], forward_return[q, :])))
penalty = 0.03 * (1 - score / 10)
IC = base_fitness - penalty
```

这里不做 rank、不做行业或风格中性化、对有符号逐季 IC 先求均值再取绝对值。

### 3.3 日期、特征和算子配置

文件：

```text
core_func/constant/params.py
```

其中包含：

- `factor_start_str`、`factor_end_str`、`eval_start_str` 等日期范围；
- 当前 90 个启用算子的 `mutation_functions` 和 `crossover_functions`；
- tensor metadata 驱动的 503 个 `feature_names`；
- 股票、日频日期和季度日期轴。

算子的逐项定义、真实窗口和注意事项见 [docs/operators.md](docs/operators.md)。

### 3.4 路径配置

文件：

```text
core_func/constant/path.py
```

建议不要直接修改源码路径，优先使用：

```bash
export GP_BACKTEST_ROOT=/data/research
export GP_OUTPUT_ROOT=/absolute/path/to/output
export GP_FEATURE_TENSOR=/absolute/path/to/financial_feature_tensor.npy
```

## 4. 启动方式

### 4.1 三层端到端验证（推荐）

测试脚本覆盖生产规模，但不会修改生产配置：

```text
population_size = 300
generations = 2
warm_start = 1（继承主配置）
init_depth = [1, 3]
n_jobs = 8
random_state = 20260908
```

由于 warm start 会先生成 5 倍候选，预期评价次数为：

```text
300 * 5 + 300 * 2 = 2100
```

使用全新的输出目录运行：

```bash
cd /data/fund_agent/B_step2
export DEEPSEEK_API_KEY='你的密钥'   # 若配置文件没有凭据
GP_OUTPUT_ROOT=/data/fund_agent/B_step2/runs/output_three_layer_ic_llm_v2 \
python run_three_layer_test.py
```

不要复用旧输出目录，否则 `llm_interpretability.sqlite` 会命中旧缓存，无法审计本次真实 API 请求。

### 4.2 生产运行

生产配置为 10,000 个正式种群、最大深度 4、warm start 5 倍初始化，API 调用规模和耗时都远高于三层测试。确认预算后运行：

```bash
cd /data/fund_agent/B_step2
GP_OUTPUT_ROOT=/data/fund_agent/B_step2/runs/output_genetic \
python step1_learn.py
```

### 4.3 复算入选因子

Step 1 成功并产生 `result_<dir_kw>.csv` 后：

```bash
GP_OUTPUT_ROOT=/data/fund_agent/B_step2/runs/output_genetic \
python step2_factor_final.py
```

该脚本会在终端显示入选因子数、有效表达式数和结果表前几行，然后把每个表达式复算为季度因子 CSV。

三层测试的运行目录名是 `three_layer_smoke`，继续 Step 2 时需要显式指定：

```bash
GP_OUTPUT_ROOT=/data/fund_agent/B_step2/runs/output_full_pipeline_v1 \
GP_RUN_DIR_KW=three_layer_smoke \
python step2_factor_final.py
```

### 4.4 原始 IC 与组合收益回测

```bash
GP_OUTPUT_ROOT=/data/fund_agent/B_step2/runs/output_full_pipeline_v1 \
GP_RUN_DIR_KW=three_layer_smoke \
GP_BACKTEST_JOBS=8 \
python step3_backtest.py
```

`GP_RUN_DIR_KW` 让下游读取指定的一次 GP 结果；`GP_BACKTEST_JOBS` 只控制展示/回测并行度，不改变遗传算法配置。默认 `include_neutralized=False`，所以不会加载 Barra 风格暴露。

## 5. 输出路径和结构

输出根目录由 `GP_OUTPUT_ROOT` 决定，单次 GP 结果位于：

```text
<GP_OUTPUT_ROOT>/backtest/<dir_kw>/
```

三层测试固定把 `<dir_kw>` 改为 `three_layer_smoke`。典型目录：

```text
<GP_OUTPUT_ROOT>/
├── 2015-09-30-2024-07-01-True-vwap5.csv
├── 2015-10-29-2024-10-31-vwap.csv
└── backtest/
    └── three_layer_smoke/
        ├── raw_result_three_layer_smoke.csv
        ├── original_result_three_layer_smoke.csv
        ├── result_three_layer_smoke.csv
        ├── gen_summary_all.csv
        ├── feature_summary_all.csv
        ├── function_summary_all.csv
        ├── time_three_layer_smoke.csv
        ├── test_manifest.json
        ├── llm_interpretability.sqlite
        ├── three_layer_smoke.log
        └── gen/
            ├── raw_result_three_layer_smoke_1.csv
            └── raw_result_three_layer_smoke_2.csv
```

warm start 的初始候选用于筛选正式 Gen 0，不会作为普通代际 CSV 单独完整导出；其 LLM 评分入口会记录在 SQLite 审计表中。

### 5.1 三类结果表

| 文件 | 内容 |
|---|---|
| `raw_result_<dir_kw>.csv` | 各代 fitness 大于 0、代内表达式去重后的累计记录 |
| `original_result_<dir_kw>.csv` | 跨代按表达式全局去重，并分配 `factor_name` |
| `result_<dir_kw>.csv` | 从全局去重结果中筛选 `abs(IC) > 0.02` 的最终候选；由于上游只保留正 fitness，实际为 `IC > 0.02` |

当前结果列：

| 字段 | 含义 |
|---|---|
| `factor_name` | 去重后生成的因子名；raw 表中没有该字段 |
| `generation` | 候选所属正式代 |
| `formulation` | 可读表达式 |
| `formulation_stack` | 用于复算的前缀表达式栈 |
| `feature_stack` | 表达式引用的中文财务特征 |
| `IC` | 扣除 LLM 惩罚后的最终 fitness |
| `base_fitness` | 惩罚前季频 Pearson IC |
| `llm_interpretability_score` | DeepSeek 解释性评分，范围 0–10 |
| `llm_interpretability_penalty` | 从 base fitness 扣除的惩罚 |

### 5.2 汇总、时间和审计文件

| 文件 | 内容 |
|---|---|
| `gen_summary_all.csv` | 各代平均/最佳 IC、表达式长度、样本数和算子统计 |
| `feature_summary_all.csv` | 每个财务特征出现次数与占比 |
| `function_summary_all.csv` | 每类算子出现次数与占比 |
| `time_<dir_kw>.csv` | 每个正式代耗时；后续 Step 2/3 也会追加耗时 |
| `test_manifest.json` | 三层测试参数、数据形状、深度审计、候选评分审计和完成状态 |
| `three_layer_smoke.log` | 运行参数、每个候选表达式和 fitness 日志 |
| `llm_interpretability.sqlite` | 唯一表达式评分缓存和逐候选审计记录 |

SQLite 包含：

- `llm_interpretability`：按“表达式 + 模型”保存唯一评分；
- `llm_interpretability_audit`：每个候选一次记录，`outcome` 区分 `api_success`、`cache_hit`、`api_failure` 和 `no_api_key`。

查看调用统计：

```bash
sqlite3 <输出目录>/llm_interpretability.sqlite \
  "SELECT outcome, COUNT(*) FROM llm_interpretability_audit GROUP BY outcome;"
```

查看最高 fitness：

```bash
python - <<'PY'
import pandas as pd
p = 'runs/output_full_pipeline_v1/backtest/three_layer_smoke/result_three_layer_smoke.csv'
df = pd.read_csv(p)
print(df.sort_values('IC', ascending=False)[
    ['factor_name', 'formulation', 'base_fitness',
     'llm_interpretability_score', 'llm_interpretability_penalty', 'IC']
].head(20).to_string(index=False))
PY
```

## 6. 结果展示和回测图

框架已有绘图代码，并已增加顶层 `step3_backtest.py`。绘图实现在 `core_func/step/backtest.py`，能够生成：

- `stock_num_<factor>.png`：有效股票数和覆盖率；
- `group_return_<factor>.png`：十分组累计收益；
- `long_short_return_<factor>.png`：多空及对冲累计收益；
- 对应 IC、分组收益和多空收益 CSV；
- `combine_summary/*.xls` 汇总工作簿。

设置 `include_neutralized=True` 时仍可生成带 `neu_`、`long_only_neu_` 前缀的中性化版本，但这不是当前默认流程。

完整链路原本是：

```text
step1_learn.py
  -> step2_factor_final.py
  -> FactorReplicate 生成 factor_csv
  -> Backtest 生成 factor_summary 和 PNG
  -> IcAnalysis 汇总和相关性筛选
```

也可以直接调用 Backtest 类：

```bash
python - <<'PY'
from core_func import Backtest
Backtest(backtest_hori=5, mp_mode=True,
         include_neutralized=False,
         dir_kw_override='three_layer_smoke',
         max_workers_override=8).backtest()
PY
```

Backtest 已改为按需加载：默认只加载收益、股票限制和指数基准；风格暴露只在显式启用中性化报告时读取。因此不需要为了画图修改 `gen_metric_list`。

## 7. 结果解读边界

- `base_fitness` 和 `IC` 都是样本内结果，不代表样本外有效性。
- LLM 只评价经济直觉和表达式结构，不评价历史收益。
- 高 LLM 分数只能减少惩罚，不能让低原始 IC 自动成为高质量因子。
- 当前最终门槛作用于惩罚后的 IC，因此原始 IC 略高于 0.02 的表达式可能因解释性不足而落选。
- 三层测试用于验证工程闭环、调用审计和深度约束，不用于形成投资结论。
- 正式使用前仍需时间切分、样本外 IC、换手率、容量、行业暴露和稳健性检验。

## Run directory convention

All training outputs, logs, API usage records, caches and reports belong under `runs/<run_name>/`. Keep source code and input `artifacts/` outside `runs/`. Set `GP_OUTPUT_ROOT` to a directory under this project's `runs/` when naming a run explicitly.

## Current training and IC dates

Training and IC evaluation both use 2017Q1 through 2024Q4 (2017-03-31 to 2024-12-31): 32 quarter rows and at most 24 non-Q4 IC observations. Forward-return and availability windows span 2017-05-01 through 2025-04-30. Q4 remains coverage-only. Date configuration is in `core_func/constant/params.py`; historical runs retain their original dates.

## Shared correlation filter

After materializing GP candidate values, call `/data/miniconda3/envs/py310/bin/python /data/fund_agent/factor_filter/filter.py --catalog <result_production.csv> --score-column base_fitness --factor-dir <quarterly_values> --output-dir <runs/run_name/filter> --ic-cut 0.01 --corr-cut 0.8`. CSV values must have stock rows and report-quarter columns. Use unpenalized base_fitness, not IC. This separate filter does not alter training or the internal 10-component selection.

### LLM token savings

Numerical IC is evaluated before LLM scoring. Invalid/nonfinite IC skips the API; zero penalty skips all LLM/cache activity. Valid candidates keep the original penalty formula. Scores are shared across runs in `runs/llm_cache/scores.sqlite` (override: `GP_LLM_SHARED_CACHE`), keyed by expression and model/API/prompt/settings fingerprint. Per-expression process locks coalesce concurrent requests; each run retains its own audit. The 60,797 scores from production_249_20260911_202918 were imported under the unchanged prompt/settings. Low valid IC alone does not bypass scoring because it can affect tournament selection.

## Git checkout setup

Copy `core_func/constant/config.example.json` to `core_func/constant/config.json` and set `DEEPSEEK_API_KEY` in the environment. The local config, input artifacts and run outputs are excluded from Git. Provide compatible input data and metadata before running.

Prepared 24-run batch: `run_experiment_24.py` prints and validates the plan; `--execute --output-root runs/<new_batch_name>` explicitly starts training. See [batch settings](docs/evolution_parameters.md#prepared-24-run-batch).
