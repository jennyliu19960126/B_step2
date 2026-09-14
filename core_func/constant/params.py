import os
import sys
import json
import datetime
import pandas as pd
from pathlib import Path
from itertools import product

sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(".")
from constant.path import daily_data_root_dict
from utility.symbol import add_symbol_posfix
from utility.date import int_to_date_str

# dates
today_str = datetime.datetime.now().strftime("%Y-%m-%d")
factor_start_str = "2017-03-31"
factor_end_str = "2024-12-31"

factor_start_str_forward = "2017-05-01"
factor_end_str_forward = "2025-04-30"

eval_start_str = "2017-03-31"
eval_end_str = "2024-12-31"

eval_start_str_forward = "2017-05-01"
eval_end_str_forward = "2025-04-30"

eval_start_step3 = "2017-03-31"
eval_end_step3 = "2025-04-30"


parent_path = str(Path(__file__).resolve().parent)
with open(os.path.join(parent_path,"config.json")) as f:
    my_config = json.load(f)
k_cs_func=my_config["k_cs_func"]
for k,v in my_config.items():
    locals()[k] = v

summary_df_cols = [
    "factor_name",
    "IC",
    "IR",
    "ret",
    "sharpe",
    "mdd",
    "hedge_ret",
    "hedge_sharpe",
    "hedge_mdd",
    "neu_IC",
    "neu_hedge_sharpe",
    "long_only_neu_IC",
    "start_date",
    "end_date",
    "date_size",
    "coverage",
    "formulation",
]


# Expression operators are deliberately split by economic role.  Genetic
# crossover/subtree mutation still operate on whole trees; these labels control
# which functions may appear inside those trees, not the genetic event itself.
mutation_functions = [
    # Cross-sectional normalization / shaping
    "cs_rank", "cs_indrank", "cs_indneutral", "cs_abs", "cs_neg",
    "cs_sign", "cs_sqrt", "cs_log", "cs_inv", "cs_power_third",
    # Trailing statement transforms and dynamics
    "ts_ttm_4", "ts_growth_1", "ts_growth_2", "ts_growth_4",
    "ts_delta_1", "ts_delta_2", "ts_delta_4",
    "ts_delay_1", "ts_delay_2", "ts_delay_4",
    "ts_rank_2", "ts_rank_4", "ts_rank_8",
    "ts_zscore_4", "ts_zscore_8", "ts_demean_4", "ts_demean_8",
    "ts_mean_4", "ts_mean_8", "ts_median_4", "ts_std_4",
    "ts_grstable_4", "ts_autoslope_4", "ts_decay_linear_4",
    # Latest external cross-sectional transforms (unary)
    "cs_signed_square", "cs_demean", "cs_zscore",
    "cs_switch_packet_flexible",
    # Latest external time-series transforms.  A compact set of representative
    # fixed windows is enabled; the complete 1/2/4/6/8/12 catalog remains
    # registered in core.ts_functions for controlled experiments.
    "ts_avg_2", "ts_first_slope_4", "ts_median_abs_deviation_2",
    "ts_rms_4", "ts_norm_mean_4", "ts_norm_max_4", "ts_norm_min_4",
    "ts_norm_min_max_4", "ts_ratio_beyond_sigma_2_2",
    "ts_ratio_beyond_sigma_3_2", "ts_index_mass_quantile_q25_2",
    "ts_number_cross_mean_2", "ts_time_asymmetry_stats_2",
    "ts_longest_strike_above_mean_2", "ts_longest_strike_below_mean_2",
    "ts_mean_over_1norm_4", "ts_quantile_q25_4", "ts_decay_exp_2",
    "ts_fw_fracdiff_2", "ts_stddev_lms_4", "ts_sharp_4",
    "ts_sharp_lms_4", "ts_skew_lms_4", "ts_decay_linear_pctchg_4",
    "ts_smooth_pos_gp_4", "ts_count_over_cs_mean_4", "ts_rsi_4",
    "ts_rsi_of_pctchg_2",
]

crossover_functions = [
    # Arithmetic combination
    "cs_add", "cs_sub", "cs_mul", "cs_div", "cs_max", "cs_min",
    # Cross-sectional relationship construction
    "cs_residual", "cs_regressor",
    # Two-series trailing relationship features
    "ts_cov_4", "ts_corr_4",
    # Latest external two-signal cross-sectional combinations
    "cs_signmul", "cs_zscore_diff", "cs_rank_diff", "cs_zscore_add",
    "cs_zscore_min", "cs_zscore_max", "cs_zscore_mul",
    "cs_zscore_ratio", "cs_imbalance_coef", "cs_zscore_imbalance_coef",
    "cs_zscore_harmonic_mean", "cs_double_scored",
    # Latest external two-series time-series functions
    "ts_roll_resid_4", "ts_signmul_4",
]

function_set = mutation_functions + crossover_functions
# Backwards-compatible lists consumed by the generation-summary report.
cs_functions = [name for name in function_set if name.startswith("cs_")]
ts_func_kw = sorted({name.split('_', 1)[1].rsplit('_', 1)[0]
                     for name in function_set if name.startswith("ts_")})
fix_ts_functions = [name for name in mutation_functions if name.startswith("ts_")]
dynamic_ts_functions = []


# feature

# The portable feature builder writes the GP input as one aligned tensor rather
# than 503 individual CSVs.  Its metadata is the single source of truth for
# feature order.  GP_FEATURE_TENSOR may point at another compatible artifact.
_default_tensor = Path(__file__).resolve().parents[2] / "artifacts" / "gate1_229_plus_x20_ic0.02_corr0.8" / "financial_feature_tensor.npy"
feature_tensor_path = os.environ.get("GP_FEATURE_TENSOR", str(_default_tensor))
with open(Path(feature_tensor_path).with_suffix(".metadata.json"), encoding="utf-8") as _feature_meta_file:
    feature_tensor_metadata = json.load(_feature_meta_file)
feature_names = feature_tensor_metadata["feature_names"]


# stocks and dates constant
_restrcit_df_raw = pd.read_csv(os.path.join(daily_data_root_dict["Base"],"S_RESTRICT.csv"),index_col=0).T
_restrict_df = _restrcit_df_raw.loc[
    factor_start_str:factor_end_str, _restrcit_df_raw.notnull().any().values
]
STOCKS = _restrict_df.columns.to_list()
# Keep the GP universe identical to the tensor stock axis.  This removes
# instruments for which no financial statement feature can ever be observed
# (for example the BSE-only names present in S_RESTRICT).
_tensor_stock_set = set(feature_tensor_metadata["stocks"])
STOCKS = [stock for stock in STOCKS if stock in _tensor_stock_set]
DATES = _restrict_df.index.to_list()
n_dates, n_stocks = len(DATES), len(STOCKS)


# Loading the data 
import pandas as pd
import numpy as np
import os

# 设置数据路径 - fundamental input data
data_root = '/data/research'

# 录入数据 - fundamental variables 
base_sheet_total_quarter = pd.read_csv(f'{data_root}/results_temp/base_sheet_total_quarter.csv',
                                       parse_dates=['REPORT_PERIOD', 'ANN_DT_MERGE'])
base_df = base_sheet_total_quarter

#单独生成时期
DATES_quar_raw = base_df['REPORT_PERIOD'].drop_duplicates().sort_values().tolist()
DATES_quar_raw = [d.strftime('%Y-%m-%d') for d in DATES_quar_raw]

# 用restrict 的时间卡季度数据的时间跨度
date = DATES
stock = STOCKS
date_start = min(date)
date_end = max(date)

DATES_quar = [d for d in DATES_quar_raw if date_start <= d <= date_end]

n_periods = len(DATES_quar)
    
