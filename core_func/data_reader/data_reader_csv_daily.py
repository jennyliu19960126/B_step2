import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

#sys.path.append(str(Path(__file__).resolve().parent.parent))
from constant.path import daily_data_root_dict,output_root
from constant.params import STOCKS, DATES, n_stocks, n_dates, factor_start_str, factor_end_str

def load_y():
    ret_df = load_rolling_ret_daily()
    ret_arr = ret_df.T.values
    return ret_arr

def load_rolling_ret_daily(factor_start_str=factor_start_str,factor_end_str=factor_end_str,h_hori=5,quantile=True):
    os.makedirs(output_root, exist_ok=True)
    temp_path = os.path.join(output_root,f"{factor_start_str}-{factor_end_str}-{quantile}-vwap{h_hori}.csv")
    if os.path.exists(temp_path):
        ret_df = pd.read_csv(temp_path,index_col=0)
    else:
        data_type = "Base"
        data_name = "S_DQ_RET_VWAP_0930_1000"
        data_dir = daily_data_root_dict[data_type]
        data_path = os.path.join(data_dir, f"{data_name}.csv")

        ret_df = pd.read_csv(data_path,index_col=0)
        factor_start_idx = ret_df.columns.tolist().index(factor_start_str)
        factor_end_index = ret_df.columns.tolist().index(factor_end_str)
        ret_df = ret_df.iloc[:,factor_start_idx-h_hori-1:factor_end_index+h_hori+1]
        # Vectorized rolling is substantially faster than rolling.apply over
        # the 5k-stock matrix and has the desired NaN-skipping sum semantics.
        ret_df = ret_df.T.rolling(window=h_hori, min_periods=1).sum().T
        # 参考榷浩的代码，rolling sum之后左移才是当天对应的收益
        
        ret_df = ret_df.shift(-h_hori, axis=1)
        ret_df = ret_df.loc[:,factor_start_str:factor_end_str] 
        if quantile:
            winsor_ret_df = ret_df.copy()
            daily_min_ret = ret_df.quantile(0.025, axis=0)
            daily_max_ret = ret_df.quantile(0.975, axis=0)
            min_group_ret = pd.DataFrame(0, index=ret_df.index, columns=ret_df.columns)
            min_group_ret = min_group_ret + daily_min_ret
            max_group_ret = pd.DataFrame(0, index=ret_df.index, columns=ret_df.columns)
            max_group_ret = max_group_ret + daily_max_ret
            winsor_ret_df[winsor_ret_df > max_group_ret] = max_group_ret
            winsor_ret_df[winsor_ret_df < min_group_ret] = min_group_ret
            ret_df = winsor_ret_df
        ret_df = ret_df.reindex(index=STOCKS)
        ret_df.to_csv(temp_path)
    return ret_df / h_hori

def load_csv_daily(data_type, data_name, start=None, end=None):
    data_dir = daily_data_root_dict[data_type]
    data_path = os.path.join(data_dir, f"{data_name}.csv")
    # The portable bundle keeps the original Wind return matrix name, without
    # the historical `_fill` suffix used by this project.
    if not os.path.exists(data_path) and data_name.endswith("_fill"):
        data_path = os.path.join(data_dir, f"{data_name[:-5]}.csv")
    if data_type == "DailyFeature":
        df = pd.read_csv(data_path, index_col=0).T
        df = df.reindex(columns=STOCKS)
    elif data_type == "index_data":
        df = pd.read_csv(data_path, index_col=0).T
    else:
        df = pd.read_csv(data_path, index_col=0).T
        df = df.reindex(columns=STOCKS)
    if start and end:
        df = df.loc[start:end]
    return df
