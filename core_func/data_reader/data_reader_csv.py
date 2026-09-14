import os
import sys
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from constant.path import daily_data_root_dict,output_root
from constant.params import n_periods, DATES_quar, STOCKS, DATES, n_stocks, n_dates,factor_start_str_forward,factor_end_str_forward, factor_start_str, factor_end_str, feature_tensor_path
#from constant.params_quarter import n_periods, DATES_quar

def load_rolling_ret_quarterly(factor_start_str=factor_start_str_forward,factor_end_str=factor_end_str_forward):
    os.makedirs(output_root, exist_ok=True)
    temp_path = os.path.join(output_root,f"{factor_start_str}-{factor_end_str}-vwap.csv")
    if os.path.exists(temp_path):
        custom_quarter_df = pd.read_csv(temp_path,index_col=0)
    else:
        data_type = "Base"
        data_name = "S_DQ_RET_VWAP_0930_1000"
        data_dir = daily_data_root_dict[data_type]
        data_path = os.path.join(data_dir, f"{data_name}.csv")
        ret_df = pd.read_csv(data_path,index_col=0)
        ret_df = ret_df.loc[:, factor_start_str:factor_end_str]

        ret_df.columns = pd.to_datetime(ret_df.columns)

        custom_quarter_df = pd.DataFrame(index=ret_df.index)

        all_dates = ret_df.columns.sort_values()

        years = sorted(set([d.year for d in all_dates]))

        for year in years:
            # === Q1: 当前年 5月1日 到 8月31日，年化×3，对应季度结束标记为 03-31 ===
            start_q1 = pd.Timestamp(f"{year}-05-01")
            end_q1 = pd.Timestamp(f"{year}-08-31")
            q1_dates = [d for d in all_dates if start_q1 <= d <= end_q1]
    
            if q1_dates:
                q1_ret = ret_df[q1_dates].apply(np.nansum, axis=1) * 3
                custom_quarter_df[pd.Timestamp(f"{year}-03-31")] = q1_ret
    
            # === Q2: 当前年 9月1日 到 10月31日，年化×6，对应季度结束标记为 06-30 ===
            start_q2 = pd.Timestamp(f"{year}-09-01")
            end_q2 = pd.Timestamp(f"{year}-10-31")
            q2_dates = [d for d in all_dates if start_q2 <= d <= end_q2]
    
            if q2_dates:
                q2_ret = ret_df[q2_dates].apply(np.nansum, axis=1) * 6
                custom_quarter_df[pd.Timestamp(f"{year}-06-30")] = q2_ret
    
            # === Q3: 当前年11月1日 到 次年4月30日，年化×2，记为次年09-30 ===
            start_q3 = pd.Timestamp(f"{year}-11-01")
            end_q3 = pd.Timestamp(f"{year+1}-04-30")
            q3_dates = [d for d in all_dates if start_q3 <= d <= end_q3]
    
            if q3_dates:
                q3_ret = ret_df[q3_dates].apply(np.nansum, axis=1) * 2
                custom_quarter_df[pd.Timestamp(f"{year}-09-30")] = q3_ret
            
            # === Q4: 每年12-31 记为NAN ===
            custom_quarter_df[pd.Timestamp(f"{year}-12-31")] = np.nan

        custom_quarter_df = custom_quarter_df.sort_index(axis=1)
        custom_quarter_df = custom_quarter_df.reindex(columns=pd.to_datetime(DATES_quar))
        custom_quarter_df.columns = custom_quarter_df.columns.strftime('%Y-%m-%d')
        custom_quarter_df = custom_quarter_df.reindex(index=STOCKS)
        custom_quarter_df.to_csv(temp_path)
    return custom_quarter_df

def load_y_quarter():
    ret_df = load_rolling_ret_quarterly()
    ret_arr = ret_df.T.values 
    return ret_arr

def load_csv_daily(data_type, data_name, start=None, end=None):
    data_dir = daily_data_root_dict[data_type]
    data_path = os.path.join(data_dir, f"{data_name}.csv")
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


def load_feature(feature_name_list):
    """Load GP features, preferring the portable X[quarter, feature, stock]."""
    if os.path.exists(feature_tensor_path):
        meta_path = os.path.splitext(feature_tensor_path)[0] + ".metadata.json"
        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)
        tensor = np.load(feature_tensor_path, mmap_mode="r")
        feature_index = {name: i for i, name in enumerate(metadata["feature_names"])}
        missing = [name for name in feature_name_list if name not in feature_index]
        if missing:
            raise KeyError(f"feature tensor does not contain: {missing[:5]}")
        date_index = {date: i for i, date in enumerate(metadata["quarters"])}
        requested_dates = [str(pd.Timestamp(date).date()) for date in DATES_quar]
        missing_dates = [date for date in requested_dates if date not in date_index]
        if missing_dates:
            raise KeyError(f"feature tensor does not contain report dates: {missing_dates[:5]}")
        stock_index = {stock: i for i, stock in enumerate(metadata["stocks"])}
        X = np.full((len(requested_dates), len(feature_name_list), len(STOCKS)), np.nan, dtype=np.float32)
        common_target = [(target_i, stock_index[stock]) for target_i, stock in enumerate(STOCKS) if stock in stock_index]
        if common_target:
            target_i, source_i = map(np.asarray, zip(*common_target))
            X[:, :, target_i] = tensor[np.asarray([date_index[d] for d in requested_dates])[:, None, None], np.asarray([feature_index[name] for name in feature_name_list])[None, :, None], source_i[None, None, :]]
        return X
    feature_df_list = []
    for feature_name in feature_name_list:
        _data = load_csv_daily("FinancialFeature", feature_name, factor_start_str, factor_end_str)
        feature_df_list.append(_data)
    n_features = len(feature_name_list)
    X = np.full((n_periods, n_features, n_stocks), np.nan)
    # i: date
    for i in range(n_periods):
        date = DATES_quar[i]
        # j: feature
        for j in range(n_features):
            X[i][j] = feature_df_list[j].loc[date].values
    return X
