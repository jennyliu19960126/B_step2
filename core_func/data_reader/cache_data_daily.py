import os
import sys
import traceback
import numpy as np
import pandas as pd
from pathlib import Path

#sys.path.append(str(Path(__file__).resolve().parent.parent))
from data_reader.data_reader_csv_daily import load_csv_daily,load_y
from utility.calc_func import cal_residual
from constant.params import fitness_metric, gen_metric_list, n_stocks, eval_start_str, eval_end_str, factor_start_str, factor_end_str,h_hori, index_name
from constant.path import output_root
from utility.tradedate import calendar


class CacheData_daily:

    def __init__(self):
        super(CacheData_daily, self).__init__()
        self.eval_start_str = eval_start_str 
        self.eval_end_str = eval_end_str
        self.factor_start_str = factor_start_str
        self.factor_end_str = factor_end_str
        self.prep_eval_mask()
        self.prep_restrict_arr()        
        self.ntr_ic_x = None
        self.index_ret_arr = None
        # Metric-specific preparation is conditional.  Keep the cache schema
        # stable when a run does not request neutralized or hedged metrics.
        self.excess_return = None
        self.return_residual = None

    def prep_eval_mask(self):
        _df = calendar.loc[self.factor_start_str:self.factor_end_str]
        _df['wt'] = 0
        _df.loc[self.eval_start_str:self.eval_end_str,'wt'] = 1
        self.eval_mask = _df['wt'].values

    def prep_restrict_arr(self):
        restrict_df = load_csv_daily('Base', 'S_RESTRICT', self.eval_start_str, self.eval_end_str)
        self.restrict = restrict_df.values

    def prep_cache_data_fitness_neutralized_ic(self, start=None, end=None):
        start = self.eval_start_str
        end = self.eval_end_str
       
        sz_list = [
            "size",
            "beta",
            "momentum",
            "residual_volatility",
            "book_to_price",
            "liquidity",
            "non_linear_size",
            "leverage",
            "earnings_yield",
            "growth",
        ]
        ntr_ic_x = None
        for x in sz_list:
            data = load_csv_daily("SzBa", x, start, end)
            data_arr = data.values
            if ntr_ic_x is None:
                ntr_ic_x = data_arr[:,:,None]
            else:
                ntr_ic_x = np.dstack((ntr_ic_x, data_arr))

        self.ntr_ic_x = ntr_ic_x
        y = load_y()
        y=y[self.eval_mask==1]
        self.return_residual = cal_residual(y,ntr_ic_x)
        return ntr_ic_x

    def prep_cache_data_fitness_hedged_sharpe(self,h_hori=h_hori):
        index_ret = load_csv_daily("index_data","S_DQ_RET")
        index_ret = index_ret.rolling(window=h_hori,min_periods=1).sum() / h_hori
        index_ret = index_ret.shift(-h_hori)
        index_ret = index_ret.loc[self.eval_start_str:self.eval_end_str,:]
        index_ret = index_ret[index_name]
        self.index_ret_arr = index_ret.values

        origin_index_ret = load_csv_daily("index_data","S_DQ_RET")
        origin_index_ret = origin_index_ret[[index_name]]
        vwap_ret = load_csv_daily('Base','S_DQ_RET_VWAP_0930_1000_fill')
        agg_vwap_exret = vwap_ret.sub(origin_index_ret.values)

        eval_factor_start_idx = agg_vwap_exret.index.tolist().index(self.eval_start_str)
        eval_factor_end_index = agg_vwap_exret.index.tolist().index(self.factor_end_str)
        origin_index_ret = origin_index_ret.iloc[eval_factor_start_idx-h_hori-1:eval_factor_end_index+h_hori+1,:]
        agg_vwap_exret = agg_vwap_exret.iloc[eval_factor_start_idx-h_hori-1:eval_factor_end_index+h_hori+1,:]
        exret_post_5d = agg_vwap_exret.rolling(h_hori, min_periods=1).sum() / h_hori
        exret_post_5d = exret_post_5d.shift(-h_hori)
        exret_post_5d  = exret_post_5d.loc[self.eval_start_str:self.eval_end_str].values
        self.excess_return = exret_post_5d

try:
    metric_set = set([fitness_metric]+gen_metric_list)
    cache_data = CacheData_daily()
    if "hedge_sharpe" in metric_set:
        cache_data.prep_cache_data_fitness_hedged_sharpe()
    if ("neu_IC" in metric_set) or ("long_only_neu_IC" in metric_set):
        cache_data.prep_cache_data_fitness_neutralized_ic()
    eval_mask_daily = cache_data.eval_mask
    restrict_daily = cache_data.restrict      
    ntr_ic_x_daily = cache_data.ntr_ic_x
    index_ret_arr_daily = cache_data.index_ret_arr
    excess_return_daily = cache_data.excess_return
    return_residual_daily = cache_data.return_residual
    fitness_cache_daily = [eval_mask_daily, restrict_daily, ntr_ic_x_daily, index_ret_arr_daily, excess_return_daily,return_residual_daily]
    print("cache data loaded")
except Exception as e:
    traceback.print_exc()
    print(e)
