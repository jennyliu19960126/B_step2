# 外部算子库迁移前盘点（历史记录）

> **状态说明（2026-09-10）：** 本文记录升级前的差异基线。外部算子实现现已合并到当前 GP；下文“当前框架尚无”等表述均指合并前状态，不再代表现状。最新注册及生产启用口径请以 [GP 算子手册](operators.md) 为准。

> 更新日期：2026-09-10  
> 外部代码：[`functions.py`](../../functions.py)、[`ts_functions.py`](../../ts_functions.py)  
> 当前 GP 对照：[`core_func/core/functions.py`](../core_func/core/functions.py)、[`core_func/core/ts_functions.py`](../core_func/core/ts_functions.py)、[`core_func/constant/params.py`](../core_func/constant/params.py)

本文盘点 `/data/fund_agent/functions.py` 与 `/data/fund_agent/ts_functions.py` 中已经实现并注册的算子，并与当前 `gp_factor_demo` 做静态代码比对。重点回答两个不同问题：

1. 哪些算子当前 GP **已经实现但生产配置未启用**；
2. 哪些算子在外部文件中已经实现，但当前 GP **连注册实现都没有**。

这两类状态不能混用。“未启用”只需评估后加入 `function_set`；“框架尚无”则必须先迁移实现、注册元数、补测试，再考虑启用。

统计范围以外部 `_function_map` 和 `ts_func_kw_list` 为准。`rolling_*`、`_cs_mean`、`_cs_std`、`_cs_quantile` 等只被其他函数调用的底层 helper 不单独计作 GP 算子；已经注释掉的 `index_mass_quantile_q50/q75`、`quantile_q75` 也不计入。

## 1. 盘点结论

| 统计口径 | 数量 |
|---|---:|
| 外部横截面算子 | 42 个注册名 |
| 外部时序算子族 | 71 个 |
| 每个外部时序族的注册形式 | 1 个动态窗口名 + 6 个固定窗口名 |
| 外部时序注册名 | `71 × 7 = 497` 个 |
| 外部文件合计注册名 | 539 个 |
| 与当前生产 `function_set` 完全同名 | 37 个 |
| 当前 GP 已注册、但生产未启用的外部横截面算子 | 11 个 |
| 当前 GP 完全没有的外部算子族 | **57 个：横截面 16 个 + 时序 41 个** |
| 对应当前 GP 完全没有的外部注册名 | **319 个：16 + 41×7** |

合并前的生产空间以 44 个算子为准。当时仅读取到外部文件并不代表这些算子已经能被 GP 使用：运行时只从项目内 `_function_map`、`_extra_function_map` 建立 `all_cal_dictionary`，再按 `function_set` 选择候选函数。合并后的最新口径见 [GP 算子手册](operators.md)。

## 2. 状态定义

| 状态 | 判定标准 | 要进入生产空间还需要什么 |
|---|---|---|
| 当前已启用 | 名称存在于当前 `function_set` | 无；仍需按研究流程做样本外验证 |
| 已实现、未启用 | 当前项目注册表中有同名算子，但 `function_set` 没有该名称 | 选择固定窗口并加入 `mutation_functions` 或 `crossover_functions` |
| 当前框架尚无 | 外部注册表中存在，但当前项目注册表中没有该算子族 | 迁移实现、依赖和 arity，加入本地注册表，测试后再加入 `function_set` |

本文按“算子族”比较时序函数。例如 `ts_rms_d` 代表 `rms` 这一族，不代表代码中真的注册了字面名称 `ts_rms_d`。

## 3. 外部 `functions.py`：42 个横截面算子

### 3.1 与当前生产空间同名的 15 个

```text
cs_abs, cs_add, cs_div, cs_indneutral, cs_inv,
cs_log, cs_max, cs_min, cs_mul, cs_neg,
cs_power_third, cs_rank, cs_sign, cs_sqrt, cs_sub
```

“同名”不保证实现逐行相同。例如外部 `cs_indneutral` 使用 DataFrame stack/merge 对齐行业；当前项目版本直接使用已经与季度张量对齐的行业数组，并额外要求行业码大于 0。生产口径仍应以当前项目代码为准。

### 3.2 当前 GP 已实现但生产未启用的 11 个

| 算子 | 外部实现含义 | 不启用时需要注意的原因 |
|---|---|---|
| `cs_sin(x)` | `sin(x)` | 对财务量纲和周期没有天然经济解释 |
| `cs_cos(x)` | `cos(x)` | 同上 |
| `cs_tan(x)` | `tan(x)` | 存在奇点，容易产生极端值 |
| `cs_power_2(x)` | `x²` | 丢失正负方向且放大极端值 |
| `cs_power_3(x)` | `x³` | 保留方向但强烈放大极端值 |
| `cs_sigmoid(x)` | `1/(1+exp(-x))` | 未标准化的大尺度输入容易饱和 |
| `cs_fold(x)` | `abs(2×cs_rank(x)-1)` | `cs_rank` 是原始名次而非百分位，因此当前公式不是通常意义上的 `[0,1]` 两端折叠 |
| `cs_ortho(y,x)` | 横截面回归残差 | 当前生产已用语义更清楚的别名 `cs_residual` |
| `cs_condition_and(x,y,z)` | 仅在 `y>0 and z>0` 时保留 `x` | 大量制造缺失值，容易触发 90% 覆盖率门槛 |
| `cs_condition_or(x,y,z)` | 仅在 `y>0 or z>0` 时保留 `x` | 同上 |
| `cs_mask_ifelse(x,y,z)` | `x>0` 取 `y`，否则取 `z` | 三元条件树搜索空间大，解释和稳定性要求更高 |

### 3.3 当前 GP 完全没有的 16 个横截面算子

以下算子已在外部 `_function_map` 注册，但当前项目的 `_function_map` 中不存在。

| 算子 | 元数 | 外部代码的实际定义 | 迁移前重点检查 |
|---|---:|---|---|
| `cs_signed_square(x)` | 1 | `sign(x) × x²` | 强烈放大尾部；无穷值转缺失 |
| `cs_demean(x)` | 1 | `x - 当期横截面均值` | 保留原量纲；容易被极端值影响 |
| `cs_zscore(x)` | 1 | 横截面去均值后除以样本标准差（`ddof=1`） | 横截面标准差 `<1e-10` 时外部代码返回 **1**，不是 0 或缺失，建议迁移前修正 |
| `cs_switch_packet_flexible(x)` | 1 | `-abs(x - 当期横截面中位数)` | 越接近中位数得分越高；名称不能直接表达该语义 |
| `cs_signmul(x,y)` | 2 | `sign((x-cs_mean(x)) × y)` | 只保留符号，输出高度离散；不是普通乘法 |
| `cs_zscore_diff(x,y)` | 2 | `zscore(x)-zscore(y)` | 两侧先标准化，适合不同量纲信号 |
| `cs_rank_diff(x,y)` | 2 | `rank(x)-rank(y)` | 使用原始名次差，尺度随有效股票数变化 |
| `cs_zscore_add(x,y)` | 2 | `zscore(x)+zscore(y)` | 两信号等权合成 |
| `cs_zscore_min(x,y)` | 2 | `min(zscore(x),zscore(y))` | 表达短板效应；缺失传播 |
| `cs_zscore_max(x,y)` | 2 | `max(zscore(x),zscore(y))` | 表达优势效应；缺失传播 |
| `cs_zscore_mul(x,y)` | 2 | `zscore(x)×zscore(y)` | 形成尺度受控的交互项，但方向解释仍需审查 |
| `cs_zscore_ratio(x,y)` | 2 | `zscore(x)/zscore(y)-1` | 分母绝对值 `≤1e-10` 时缺失；零附近仍可能不稳定 |
| `cs_imbalance_coef(x,y)` | 2 | `(x-y)/(x+y)` | 分母绝对值 `≤1e-10` 时缺失；适合成对科目的不平衡度 |
| `cs_zscore_imbalance_coef(x,y)` | 2 | `(zx-zy)/(zx+zy)` | 先做横截面 z-score；继承 `cs_zscore` 的常数截面返回 1 问题 |
| `cs_zscore_harmonic_mean(x,y)` | 2 | `zx×zy/(zx+zy)` | 名称叫 harmonic mean，但实现少了常见公式中的系数 2 |
| `cs_double_scored(x,y)` | 2 | `x` 做百分位并分 5 组；组内再排 `y`，最后两次排名等权平均 | 至少 5 个共同样本、每个有效组至少 2 个；循环和 `qcut` 成本较高 |

当前项目额外拥有、但外部 `functions.py` 没有的生产算子是 `cs_indrank`、`cs_residual` 和 `cs_regressor`。

## 4. 外部 `ts_functions.py`：71 个时序算子族

### 4.1 注册方式

外部文件对每个时序族生成：

- 动态窗口：`dynamic_ts_<name>`，配置范围写作 `(1,12)`；由于 `randint` 上界不含 12，实际随机窗口是 **1–11**；
- 固定窗口：`ts_<name>_1/2/4/6/8/12`，每族 6 个；
- 合计：每族 7 个注册名，71 族共 497 个。

当前生产 GP 只启用固定窗口函数，不启用任何 `dynamic_ts_*`。迁移时不建议一次性把 497 个名称全部加入搜索空间，否则会显著扩大等价表达式和窗口冗余。

### 4.2 与当前 GP 注册表共有的 30 个时序族

| 类别 | 共有算子族 | 当前生产中已启用的固定名称 |
|---|---|---|
| 基础滚动 | `min`, `max`, `mean`, `median`, `product` | `ts_mean_4/8`, `ts_median_4` |
| 变化与滞后 | `delay`, `delta`, `pct` | `ts_delay_1/2/4`, `ts_delta_1/2/4` |
| 标准化与稳定性 | `rank`, `std`, `grstable`, `zscore`, `demean` | `ts_rank_2/4/8`, `ts_std_4`, `ts_grstable_4`, `ts_zscore_4/8`, `ts_demean_4/8` |
| 双变量关系 | `cov`, `corr`, `cut_up_q1`, `cut_down_q1` | `ts_cov_4`, `ts_corr_4` |
| 高阶矩 | `skew`, `kurt` | 无 |
| 自相关 | `autocorr_lag1/2/3/4/5` | 无 |
| 位置与趋势 | `autoslope`, `argmin`, `argmax`, `decay_linear` | `ts_autoslope_4`, `ts_decay_linear_4` |
| 条件计数 | `count_and`, `count_or` | 无 |

这里的“共有”仍只代表同族同名。合并前至少有一处明确实现差异：外部 `_ts_cut_up_q1/_ts_cut_down_q1` 传入 `q=10`，表示 10% 分位；旧项目版本传入 `q=0.1`，下层又除以 100，实际使用 0.1% 分位。当前实现已采用外部新版的 `q=10`，但这两个算子仍未进入生产空间。

当前项目另有外部文件没有的 `ttm`、`growth` 两族；对应生产名称为 `ts_ttm_4`、`ts_growth_1/2/4`。

## 5. 当前 GP 完全没有的 41 个时序算子族

除非另行说明，外部新增滚动统计通常要求窗口内至少 `floor(0.75×window)` 个有效值，且不少实现只从完整日历窗口结束位置开始输出。表中的 `d` 是注册名后缀传入的参数，不一定等于内部最终窗口。

### 5.1 财务变化、滞后与趋势（11 个）

| 算子族 | 元数 | 外部代码的实际定义 | 关键边界/风险 |
|---|---:|---|---|
| `ts_last_d(x)` | 1 | `x[t-d]` | 与当前 `ts_delay_d` 完全重复，不建议重复迁移 |
| `ts_avg_d(x)` | 1 | `(x[t]+x[t-d])/2` | 不是长度为 `d` 的滚动均值 |
| `ts_qoq_d(x)` | 1 | `(x[t]-x[t-d])/abs(x[t-d])` | 与当前 `ts_growth_d` 近似重复，仅保护阈值由零变为 `1e-10` |
| `ts_yoy_d(x)` | 1 | 与 `ts_qoq_d` 完全相同 | 名称不强制同比；例如 `ts_yoy_1` 实际仍是 1 期变化 |
| `ts_diff1q_d(x)` | 1 | `x[t]-x[t-d]` | 与当前 `ts_delta_d` 重复；名称不强制 1 季 |
| `ts_diff4q_d(x)` | 1 | `x[t]-x[t-d]` | 与当前 `ts_delta_d` 重复；名称不强制 4 季 |
| `ts_sue0_score_d(x)` | 1 | 用此前窗口估计无截距 AR(1)，以当前预测误差除以历史残差标准差 | `d<3` 时全缺失；代码强制前 `d+1` 期缺失 |
| `ts_sue1_score_d(x)` | 1 | 同上，但 AR(1) 含截距 | 同上；需确认财务序列上 AR(1) 与 SUE 的研究定义是否一致 |
| `ts_first_slope_d(x)` | 1 | 窗口内回归 `signal ~ time` 的一次项斜率 | 至少 `max(3,floor(0.75d))` 个有效点，并等待完整日历窗口；比当前反向回归的 `autoslope` 更直观 |
| `ts_second_slope_d(x)` | 1 | 窗口内二次回归的 `time²` 系数 | 至少 `max(5,floor(0.75d))` 个点；固定窗口 1、2、4 永远全缺失 |
| `ts_roll_resid_d(y,x)` | 2 | 每只股票在时序窗口内回归 `y~1+x`，返回当前期残差 | `d<3` 时全缺失；与横截面 `cs_residual` 的方向完全不同 |

### 5.2 尺度、分布与归一化（10 个）

| 算子族 | 元数 | 外部代码的实际定义 | 关键边界/风险 |
|---|---:|---|---|
| `ts_median_abs_deviation_d(x)` | 1 | `median(abs(x-窗口均值))` | 名称像 MAD，但中心是**均值**而非通常定义的中位数；内部窗口为 `d+1` |
| `ts_rms_d(x)` | 1 | `sqrt(mean(x²))` | 内部窗口为 `d+1`；大值权重较高 |
| `ts_norm_mean_d(x)` | 1 | `mean(x)/RMS(x)` | 内部窗口为 `d+1`，输出有界性较好 |
| `ts_norm_max_d(x)` | 1 | `max(x)/RMS(x)` | 内部窗口为 `d+1`；对单个极值敏感 |
| `ts_norm_min_d(x)` | 1 | `min(x)/RMS(x)` | 内部窗口为 `d+1`；对单个极值敏感 |
| `ts_norm_min_max_d(x)` | 1 | `(max(x)-min(x))/RMS(x)` | 内部窗口为 `d+1`；RMS 接近零时缺失 |
| `ts_ratio_beyond_sigma_2_d(x)` | 1 | 窗口内落在 `mean±2σ` 之外的观测占比 | 内部窗口为 `d+1`；标准差用 `ddof=1` |
| `ts_ratio_beyond_sigma_3_d(x)` | 1 | 同上，阈值为 `3σ` | 季频短窗口下通常接近 0，辨识度可能不足 |
| `ts_mean_over_1norm_d(x)` | 1 | `mean(x)/mean(abs(x))` | 内部窗口为 `d+1`；可表达窗口内符号一致性 |
| `ts_quantile_q25_d(x)` | 1 | 窗口 25% 分位数 | 内部窗口至少 3，即 `max(d+1,3)` |

### 5.3 路径形态与持续性（6 个）

| 算子族 | 元数 | 外部代码的实际定义 | 关键边界/风险 |
|---|---:|---|---|
| `ts_index_mass_quantile_q25_d(x)` | 1 | 从窗口最早端累计 `abs(x)`，首次达到总绝对量 25% 时的零基位置 | 返回位置 `0..window-1`，不是数值分位数；内部窗口为 `d+1` |
| `ts_number_cross_mean_d(x)` | 1 | 信号穿越自身窗口均值的次数 | 只统计严格正负号翻转；缺失间隔会连接前后两个有效点 |
| `ts_time_asymmetry_stats_d(x)` | 1 | `mean((x[t]-x[t-1])² × (x[t-1]-x[t-2]))` | 内部窗口至少 3；量纲为原信号三次方，极端值敏感 |
| `ts_longest_strike_above_mean_d(x)` | 1 | 连续高于窗口均值的最长长度 | 缺失值会打断连续段；内部窗口为 `d+1` |
| `ts_longest_strike_below_mean_d(x)` | 1 | 连续低于窗口均值的最长长度 | 同上 |
| `ts_fw_fracdiff_d(x)` | 1 | 固定宽度、阶数 0.5 的分数差分加权和 | 内部窗口至少 3；缺失项按零贡献处理，需验证统计性质 |

### 5.4 加权、长短窗与位置指标（11 个）

| 算子族 | 元数 | 外部代码的实际定义 | 关键边界/风险 |
|---|---:|---|---|
| `ts_decay_exp_d(x)` | 1 | 指数权重滚动均值，越近期权重越高 | 内部窗口至少 3；缺失后重新归一化权重 |
| `ts_pctchg_lms_d(x)` | 1 | 长滞后变化率减去半窗口短滞后变化率 | 内部先令 `window=max(d+1,4)`；是变化率差，不是滚动统计 |
| `ts_stddev_lms_d(x)` | 1 | 长窗口标准差减短窗口标准差 | 外层传入 `d+2`，内部窗口至少 4；衡量波动尺度变化 |
| `ts_sharp_d(x)` | 1 | 窗口均值/窗口标准差 | 名称是 `sharp` 而非 `sharpe`；未年化，也不是收益专用指标 |
| `ts_sharp_lms_d(x)` | 1 | 长窗口 `mean/std` 减短窗口 `mean/std` | 外层传入 `d+2`，内部窗口至少 4 |
| `ts_skew_lms_d(x)` | 1 | 长窗口偏度减短窗口偏度 | 外层传入 `d+2`，内部强制长窗至少 6 |
| `ts_kurtosis_lms_d(x)` | 1 | 长窗口峰度减短窗口峰度 | 外层传入 `d+2`，内部强制长窗至少 8 |
| `ts_decay_linear_pctchg_d(x)` | 1 | 长、短窗口线性外推值之比减 1 | 内部窗口至少 4；分母接近零时缺失 |
| `ts_smooth_pos_gp_d(x)` | 1 | `(短窗均值-长窗最小值)/(长窗最大值-长窗最小值)` | 内部窗口至少 4，近似平滑后的位置指标 |
| `ts_count_over_cs_mean_d(x)` | 1 | 过去窗口中该股票高于当期全市场均值的次数 | 混合横截面与时序语义；不是高于自身均值的次数 |
| `ts_rsi_d(x)` | 1 | 基于一阶差分的 RSI：`100-100/(1+avg_gain/avg_loss)` | 外层窗口为 `d+1`；若平均损失接近零，保护除法返回缺失而不是常见 RSI=100 |

### 5.5 复合和双变量算子（3 个）

| 算子族 | 元数 | 外部代码的实际定义 | 关键边界/风险 |
|---|---:|---|---|
| `ts_rsi_of_pctchg_d(x)` | 1 | 先算 1 期变化率，再对其计算 RSI | 外层窗口为 `d+1`；叠加差分会损失更多初期覆盖 |
| `ts_signmul_d(x,y)` | 2 | `sign(x-rolling_mean(x)) × y` | 外层窗口为 `d+1`；只用 `x` 决定 `y` 的方向 |
| `ts_rolling_ic_d(x,y)` | 2 | 先算 `beta=cov(x,y)/var(x)`，滞后一期，再输出 `x_t×beta[t-1]` | 名称虽含 IC，实际不是相关系数或 IC；rolling covariance 的样本门槛还会再多要求 1 个共同有效点，建议重命名后再迁移 |

41 个新时序族的元数合计为：38 个一元、3 个二元。外部文件没有新增三元时序族；已有的 `count_and/count_or` 是与当前框架共有但未启用的三元族。

## 6. 不应直接迁移或需要先修正的项目

### 6.1 明确重复或名称误导

- `ts_last` 与当前 `ts_delay` 重复；
- `ts_qoq` 与当前 `ts_growth` 基本重复；
- `ts_diff1q/ts_diff4q` 与当前 `ts_delta` 重复，而且名称并不锁定 1/4 季；
- `ts_yoy` 与 `ts_qoq` 使用完全同一个函数，窗口取决于后缀而不是名称；
- `ts_rolling_ic` 实际是滞后滚动回归斜率乘当前 `x`，不是 IC；
- `ts_median_abs_deviation` 使用均值作为中心，不是标准 MAD；
- `cs_zscore_harmonic_mean` 少了传统调和平均公式中的系数 2；
- `ts_sharp` 建议改名为 `ts_mean_over_std`，避免被理解为收益 Sharpe。

### 6.2 会产生全缺失或异常常数的组合

- `ts_second_slope_1/2/4` 因最少样本数为 5，必然全缺失；
- `ts_sue0_score_1/2`、`ts_sue1_score_1/2`、`ts_roll_resid_1/2` 因最少样本数大于窗口，必然全缺失；
- `cs_zscore` 在横截面标准差接近零时返回常数 1，建议改为缺失或 0；
- `ts_rsi` 在窗口只有上涨、平均损失为零时返回缺失，不符合常见 RSI 边界定义；
- 多个算子内部强制 `window≥3/4/6/8`，因此不同注册后缀可能得到相同实际窗口，增加重复表达式。

### 6.3 性能与覆盖率

- `cs_double_scored`、`median_abs_deviation`、`ratio_beyond_sigma`、路径连续性、二次斜率等实现包含按日期、股票或窗口循环；在 10,000 种群和多进程环境下应先做基准测试。
- 多数新增时序算子要求 75% 窗口覆盖，且部分必须等完整日历窗口才输出。与当前 90% 因子覆盖率门槛叠加后，长窗口和嵌套表达式很容易整条失效。
- 外部动态窗口实际是 1–11，而固定窗口包含 12；动态和固定口径不完全对称。

## 7. 建议的接入优先级

### 第一批：语义清楚、与财务因子较匹配

建议先小范围迁移并只开放少量固定窗口：

```text
cs_demean
cs_zscore                 # 先修正常数截面行为
cs_imbalance_coef
cs_zscore_diff
ts_rms_4 / ts_rms_8
ts_norm_mean_4 / ts_norm_mean_8
ts_mean_over_1norm_4 / ts_mean_over_1norm_8
ts_decay_exp_4 / ts_decay_exp_8
ts_first_slope_4 / ts_first_slope_8
ts_roll_resid_4 / ts_roll_resid_8
```

这些算子分别补充横截面尺度统一、成对科目不平衡度、财务量级稳定性、方向一致性、近期加权趋势和个股时序正交化。

### 第二批：有研究价值但需更强验证

```text
cs_double_scored
ts_sue0_score / ts_sue1_score
ts_second_slope
ts_ratio_beyond_sigma_2/3
ts_number_cross_mean
ts_longest_strike_above_mean / below_mean
ts_smooth_pos_gp
ts_rsi / ts_rsi_of_pctchg
```

这批算子需要重点验证最小窗口、覆盖率、计算耗时和样本外稳定性。

### 暂不建议接入

- 与现有算子重复的 `last/qoq/yoy/diff1q/diff4q`；
- 未重命名的 `rolling_ic`、`sharp`、`median_abs_deviation`；
- 未修正常数截面行为的 `cs_zscore` 及所有依赖它的复合算子；
- 在季度短样本中经济含义较弱或过度复杂的高阶路径统计，除非有明确研究假设。

## 8. 正确接入当前 GP 的步骤

仅把外部文件复制进项目不会生效。每个候选算子至少需要完成：

1. 把底层计算函数与 `_ts_*`/`_cs_*` 包装迁入当前项目实现文件，并把外部相对导入和 `helper_data` 行业数据依赖改成当前项目路径及对齐口径；
2. 在 `_function_map` 或 `_extra_function_map` 中注册，并核对一元、二元、三元 `arity`；
3. 统一窗口语义，明确后缀表示滞后还是实际窗口，删除必然全缺失的固定窗口；
4. 为全零、常数、含缺失、极小分母、窗口不足和正常输入编写单元测试；
5. 验证输出形状始终为 `[季度, 股票]`，且无穷值被转成缺失；
6. 在小种群 smoke test 中检查覆盖率、耗时、表达式深度和多进程可序列化；
7. 最后才把通过测试的固定名称加入 `mutation_functions` 或 `crossover_functions`。

建议不要直接开放动态窗口；先用 4、8 季等少量、经济含义明确的固定窗口做消融测试，可显著降低搜索空间膨胀和同质表达式数量。
