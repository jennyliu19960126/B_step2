import os
import sys
import traceback
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from data_reader.data_reader_csv import load_csv_daily,load_y_quarter
from utility.calc_func import cal_residual
from constant.params import DATES_quar, fitness_metric, gen_metric_list, n_stocks,factor_start_str_forward,factor_end_str_forward,eval_start_str_forward,eval_end_str_forward, eval_start_str, eval_end_str, factor_start_str, factor_end_str,h_hori, index_name
from constant.path import output_root
from utility.tradedate import calendar, calendar_quarterly


class CacheData:

    def __init__(self):
        super(CacheData, self).__init__()
        self.eval_start_str = eval_start_str 
        self.eval_start_str_forward = eval_start_str_forward 
        self.eval_end_str = eval_end_str
        self.factor_start_str = factor_start_str
        self.factor_start_str_forward = factor_start_str_forward
        self.factor_end_str_forward = factor_end_str_forward
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
        _df = calendar_quarterly.loc[self.factor_start_str:self.factor_end_str]
        _df['wt'] = 0
        _df.loc[self.eval_start_str:self.eval_end_str,'wt'] = 1
        self.eval_mask = _df['wt'].values
    
    def prep_restrict_arr(self):
        restrict_df = load_csv_daily('Base', 'S_RESTRICT', factor_start_str_forward , factor_end_str_forward)
        restrict_df.index = pd.to_datetime(restrict_df.index)

        is_available = (restrict_df == 0)

        custom_quarter_result = pd.DataFrame(index=[], columns=is_available.columns)

        all_dates = is_available.index
        years = sorted(set(d.year for d in all_dates))
    
        for year in years:
            # Q1: 05-01 ~ 08-31 → 03-31
            start_q1 = pd.Timestamp(f"{year}-05-01")
            end_q1 = pd.Timestamp(f"{year}-08-31")
            q1_data = is_available.loc[(is_available.index >= start_q1) & (is_available.index <= end_q1)]
            if not q1_data.empty:
                q1_ratio = q1_data.sum() / q1_data.count()
                custom_quarter_result.loc[pd.Timestamp(f"{year}-03-31")] = (q1_ratio <= 0.5).astype(int)
    
            # Q2: 09-01 ~ 10-31 → 06-30
            start_q2 = pd.Timestamp(f"{year}-09-01")
            end_q2 = pd.Timestamp(f"{year}-10-31")
            q2_data = is_available.loc[(is_available.index >= start_q2) & (is_available.index <= end_q2)]
            if not q2_data.empty:
                q2_ratio = q2_data.sum() / q2_data.count()
                custom_quarter_result.loc[pd.Timestamp(f"{year}-06-30")] = (q2_ratio <= 0.5).astype(int)
    
            # Q3: 11-01 ~ 次年 04-30 → 年 09-30
            start_q3 = pd.Timestamp(f"{year}-11-01")
            end_q3 = pd.Timestamp(f"{year+1}-04-30")
            q3_data = is_available.loc[(is_available.index >= start_q3) & (is_available.index <= end_q3)]
            if not q3_data.empty:
                q3_ratio = q3_data.sum() / q3_data.count()
                custom_quarter_result.loc[pd.Timestamp(f"{year}-09-30")] = (q3_ratio <= 0.5).astype(int)
                
            # Q4: 11-01 ~ 次年 04-30 → 年 12-31
            start_q4 = pd.Timestamp(f"{year}-11-01")
            end_q4 = pd.Timestamp(f"{year+1}-04-30")
            q4_data = is_available.loc[(is_available.index >= start_q4) & (is_available.index <= end_q4)]
            if not q4_data.empty:
                q4_ratio = q4_data.sum() / q4_data.count()
                custom_quarter_result.loc[pd.Timestamp(f"{year}-12-31")] = (q4_ratio <= 0.5).astype(int)

        custom_quarter_result = custom_quarter_result.sort_index()

        DATES_quar1 = pd.to_datetime(DATES_quar)
        restrict_df_quar = custom_quarter_result[custom_quarter_result.index.isin(DATES_quar1)]

        self.restrict = restrict_df_quar.values
        
    
    def prep_cache_data_fitness_neutralized_ic(self, start=None, end=None):
        start = self.factor_start_str_forward
        end = self.factor_end_str_forward

        # ------------------------- 1. BARRA 原始日频处理 -------------------------
        sz_list = [
            "size", "beta", "momentum", "residual_volatility", "book_to_price",
            "liquidity", "non_linear_size", "leverage", "earnings_yield", "growth",
        ]
        barra_list = []

        for x in sz_list:
            data = load_csv_daily("SzBa", x, start, end)
            data.index = pd.to_datetime(data.index)

            # 按季度取值
            all_dates = data.index.sort_values()
            years = sorted(set([d.year for d in all_dates]))
            quarterly_df = pd.DataFrame(index=[], columns=data.columns)

            for year in years:
                # Q1：取4月最后一个交易日 → 映射为 03-31
                april_data = data[(data.index >= f"{year}-04-01") & (data.index <= f"{year}-04-30")]
                if not april_data.empty:
                    quarterly_df.loc[pd.Timestamp(f"{year}-03-31")] = data.loc[april_data.index.max()]

                # Q2：取8月最后一个交易日 → 映射为 06-30
                aug_data = data[(data.index >= f"{year}-08-01") & (data.index <= f"{year}-08-31")]
                if not aug_data.empty:
                    quarterly_df.loc[pd.Timestamp(f"{year}-06-30")] = data.loc[aug_data.index.max()]

                # Q3：取10月最后一个交易日 → 映射为 09-30
                oct_data = data[(data.index >= f"{year}-10-01") & (data.index <= f"{year}-10-31")]
                if not oct_data.empty:
                    quarterly_df.loc[pd.Timestamp(f"{year}-09-30")] = data.loc[oct_data.index.max()]

                # Q4：设为 NaN
                quarterly_df.loc[pd.Timestamp(f"{year}-12-31")] = np.nan

            quarterly_df = quarterly_df.sort_index()
            quarterly_df = quarterly_df[quarterly_df.index >= pd.Timestamp(factor_start_str)] 
            quarterly_df = quarterly_df[quarterly_df.index <= pd.Timestamp(factor_end_str)]

            barra_list.append(quarterly_df)

        # ------------------------- 2. 行业 Dummy（日频转季频） -------------------------
        ind = load_csv_daily('Base', 'SW_IND_CODE', start, end)
        ind.index = pd.to_datetime(ind.index)
        all_dates = ind.index.sort_values()
        years = sorted(set([d.year for d in all_dates]))

        quarterly_ind = pd.DataFrame(index=[], columns=ind.columns)

        for year in years:
            # Q1：4月末 → 映射为 03-31
            april_data = ind[(ind.index >= f"{year}-04-01") & (ind.index <= f"{year}-04-30")]
            if not april_data.empty:
                quarterly_ind.loc[pd.Timestamp(f"{year}-03-31")] = ind.loc[april_data.index.max()]

            # Q2：8月末 → 映射为 06-30
            aug_data = ind[(ind.index >= f"{year}-08-01") & (ind.index <= f"{year}-08-31")]
            if not aug_data.empty:
                quarterly_ind.loc[pd.Timestamp(f"{year}-06-30")] = ind.loc[aug_data.index.max()]

            # Q3：10月末 → 映射为 09-30
            oct_data = ind[(ind.index >= f"{year}-10-01") & (ind.index <= f"{year}-10-31")]
            if not oct_data.empty:
                quarterly_ind.loc[pd.Timestamp(f"{year}-09-30")] = ind.loc[oct_data.index.max()]

            quarterly_ind.loc[pd.Timestamp(f"{year}-12-31")] = np.nan

        quarterly_ind = quarterly_ind.sort_index()
        quarterly_ind = quarterly_ind[quarterly_ind.index >= pd.Timestamp(factor_start_str)]
        quarterly_ind = quarterly_ind[quarterly_ind.index <= pd.Timestamp(factor_end_str)]

        quarterly_ind = quarterly_ind.astype(float).fillna(0).astype(int)

        ind_list = []
        for ind_i in range(1, 32):
            ind_dummy = (quarterly_ind // 10000 == ind_i).astype(int)
            ind_list.append(ind_dummy.values)

        # ------------------------- 3. 拼接 BARRA + 行业 -------------------------
        # BARRA: list of DataFrames → 转为 numpy
        barra_arr = [df.values[:, :, None] for df in barra_list]  # 每个因子是 (T×N×1)
        barra_x = np.concatenate(barra_arr, axis=2)  # (T, N, 10)

        # 行业 dummy：
        ind_x = np.stack(ind_list, axis=2)  # (T, N, 31)

        # 合并得到最终 ntr_ic_x：T × N × (10+31)
        ntr_ic_x = np.concatenate([barra_x, ind_x], axis=2)

        # ------------------------- 4. 计算 residual -------------------------
        self.ntr_ic_x = ntr_ic_x
        y = load_y_quarter()
        y = y[self.eval_mask == 1]
        self.return_residual = cal_residual(y, ntr_ic_x)


    def prep_cache_data_fitness_hedged_sharpe(self):
        # === 加载指数日收益 ===
        index_ret = load_csv_daily("index_data", "S_DQ_RET")
        index_ret = index_ret[[index_name]]
        index_ret.index = pd.to_datetime(index_ret.index)
    
        index_ret = index_ret.loc[self.factor_start_str_forward:self.factor_end_str_forward, :]
    
        # === 初始化自定义季度结果 ===
        all_dates = index_ret.index
        years = sorted(set(d.year for d in all_dates))
    
        custom_index_ret = pd.DataFrame(index=[], columns=[index_name])
        
        for year in years:
            # Q1: 05-01 ~ 08-31 → 03-31
            start_q1 = pd.Timestamp(f"{year}-05-01")
            end_q1 = pd.Timestamp(f"{year}-08-31")
            q1_data = index_ret[(index_ret.index >= start_q1) & (index_ret.index <= end_q1)]
            if not q1_data.empty:
                custom_index_ret.loc[pd.Timestamp(f"{year}-03-31")] = q1_data.sum(skipna=True)
    
            # Q2: 09-01 ~ 10-31 → 06-30
            start_q2 = pd.Timestamp(f"{year}-09-01")
            end_q2 = pd.Timestamp(f"{year}-10-31")
            q2_data = index_ret[(index_ret.index >= start_q2) & (index_ret.index <= end_q2)]
            if not q2_data.empty:
                custom_index_ret.loc[pd.Timestamp(f"{year}-06-30")] = q2_data.sum(skipna=True)
    
            # Q3: 11-01 ~ 次年 04-30 → 次年 09-30
            start_q3 = pd.Timestamp(f"{year}-11-01")
            end_q3 = pd.Timestamp(f"{year+1}-04-30")
            q3_data = index_ret[(index_ret.index >= start_q3) & (index_ret.index <= end_q3)]
            if not q3_data.empty:
                custom_index_ret.loc[pd.Timestamp(f"{year}-09-30")] = q3_data.sum(skipna=True)
            
            # Q4: 11-01 ~ 次年 04-30 → 年 12-31
            custom_index_ret.loc[pd.Timestamp(f"{year}-12-31")] = np.nan

        custom_index_ret = custom_index_ret.sort_index()
        custom_index_ret = custom_index_ret[custom_index_ret.index >= pd.Timestamp(factor_start_str)]
        custom_index_ret = custom_index_ret[custom_index_ret.index <= pd.Timestamp(factor_end_str)]
        
        # === 年化指数收益 ===
        for dt in custom_index_ret.index:
            if dt.month == 6:
                custom_index_ret.loc[dt] *= 6
            elif dt.month == 9:
                custom_index_ret.loc[dt] *= 2
            elif dt.month == 3:
                custom_index_ret.loc[dt] *= 3
            
        self.index_ret_arr = custom_index_ret.values
    
        # === 计算个股超额收益 ===
        origin_index_ret = load_csv_daily("index_data", "S_DQ_RET")[[index_name]]
        vwap_ret = load_csv_daily('Base', 'S_DQ_RET_VWAP_0930_1000_fill')
    
        # 注意对齐 index
        vwap_ret.index = pd.to_datetime(vwap_ret.index)
        origin_index_ret.index = pd.to_datetime(origin_index_ret.index)
    
        # 对齐评估区间
        vwap_ret = vwap_ret.loc[self.factor_start_str_forward:self.factor_end_str_forward]
        origin_index_ret = origin_index_ret.loc[self.factor_start_str_forward:self.factor_end_str_forward]
    
        # === 计算超额收益 ===
        agg_vwap_exret = vwap_ret.sub(origin_index_ret.values)
    
        # === 构建自定义季度超额收益 ===
        all_dates = agg_vwap_exret.index
        years = sorted(set(d.year for d in all_dates))
        custom_exret = pd.DataFrame(index=[], columns=vwap_ret.columns)
    
        for year in years:
            # Q1: 05-01 ~ 08-31 → 03-31
            q1_start = pd.Timestamp(f"{year}-05-01")
            q1_end = pd.Timestamp(f"{year}-08-31")
            q1_data = agg_vwap_exret[(agg_vwap_exret.index >= q1_start) & (agg_vwap_exret.index <= q1_end)]
            if not q1_data.empty:
                custom_exret.loc[pd.Timestamp(f"{year}-03-31")] = q1_data.sum(skipna=True)
    
            # Q2: 09-01 ~ 10-31 → 06-30
            q2_start = pd.Timestamp(f"{year}-09-01")
            q2_end = pd.Timestamp(f"{year}-10-31")
            q2_data = agg_vwap_exret[(agg_vwap_exret.index >= q2_start) & (agg_vwap_exret.index <= q2_end)]
            if not q2_data.empty:
                custom_exret.loc[pd.Timestamp(f"{year}-06-30")] = q2_data.sum(skipna=True)
    
            # Q3: 11-01 ~ 次年 04-30 → 次年 09-30
            q3_start = pd.Timestamp(f"{year}-11-01")
            q3_end = pd.Timestamp(f"{year+1}-04-30")
            q3_data = agg_vwap_exret[(agg_vwap_exret.index >= q3_start) & (agg_vwap_exret.index <= q3_end)]
            if not q3_data.empty:
                custom_exret.loc[pd.Timestamp(f"{year}-09-30")] = q3_data.sum(skipna=True)
    
            # Q4: 11-01 ~ 次年 04-30 → 年 12-31
            custom_exret.loc[pd.Timestamp(f"{year}-12-31")] = np.nan

        custom_exret = custom_exret.sort_index()
        custom_exret = custom_exret[custom_exret.index >= pd.Timestamp(factor_start_str)]
        custom_exret = custom_exret[custom_exret.index <= pd.Timestamp(factor_end_str)]
        
        # === 年化超额收益 ===
        for dt in custom_exret.index:
            if dt.month == 6:
                custom_exret.loc[dt] *= 6
            elif dt.month == 9:
                custom_exret.loc[dt] *= 2
            elif dt.month == 3:
                custom_exret.loc[dt] *= 3
        
        self.excess_return = custom_exret.values


# for step3 - 区别在最后的数据保存形式
def prep_restrict_arr_dataframe():
    # 加载日频限制数据
    restrict_df = load_csv_daily('Base', 'S_RESTRICT', factor_start_str_forward , factor_end_str_forward)
    restrict_df.index = pd.to_datetime(restrict_df.index)

    is_available = (restrict_df == 0)

    custom_quarter_result = pd.DataFrame(index=[], columns=is_available.columns)

    all_dates = is_available.index
    years = sorted(set(d.year for d in all_dates))

    for year in years:
        # Q1: 05-01 ~ 08-31 → 03-31
        start_q1 = pd.Timestamp(f"{year}-05-01")
        end_q1 = pd.Timestamp(f"{year}-08-31")
        q1_data = is_available.loc[(is_available.index >= start_q1) & (is_available.index <= end_q1)]
        if not q1_data.empty:
            q1_ratio = q1_data.sum() / q1_data.count()
            custom_quarter_result.loc[pd.Timestamp(f"{year}-03-31")] = (q1_ratio <= 0.5).astype(int)

        # Q2: 09-01 ~ 10-31 → 06-30
        start_q2 = pd.Timestamp(f"{year}-09-01")
        end_q2 = pd.Timestamp(f"{year}-10-31")
        q2_data = is_available.loc[(is_available.index >= start_q2) & (is_available.index <= end_q2)]
        if not q2_data.empty:
            q2_ratio = q2_data.sum() / q2_data.count()
            custom_quarter_result.loc[pd.Timestamp(f"{year}-06-30")] = (q2_ratio <= 0.5).astype(int)

        # Q3: 11-01 ~ 次年 04-30 → 年 09-30
        start_q3 = pd.Timestamp(f"{year}-11-01")
        end_q3 = pd.Timestamp(f"{year+1}-04-30")
        q3_data = is_available.loc[(is_available.index >= start_q3) & (is_available.index <= end_q3)]
        if not q3_data.empty:
            q3_ratio = q3_data.sum() / q3_data.count()
            custom_quarter_result.loc[pd.Timestamp(f"{year}-09-30")] = (q3_ratio <= 0.5).astype(int)
            
        # Q4: 11-01 ~ 次年 04-30 → 年 12-31
        start_q4 = pd.Timestamp(f"{year}-11-01")
        end_q4 = pd.Timestamp(f"{year+1}-04-30")
        q4_data = is_available.loc[(is_available.index >= start_q4) & (is_available.index <= end_q4)]
        if not q4_data.empty:
            q4_ratio = q4_data.sum() / q4_data.count()
            custom_quarter_result.loc[pd.Timestamp(f"{year}-12-31")] = (q4_ratio <= 0.5).astype(int)

    # 按时间排序
    custom_quarter_result = custom_quarter_result.sort_index()

    # 与目标日期对齐
    DATES_quar1 = pd.to_datetime(DATES_quar)
    restrict_df_quar = custom_quarter_result[custom_quarter_result.index.isin(DATES_quar1)]

    # 保存为 numpy array
    restrict = restrict_df_quar
    return restrict 


def prep_index_ret_dataframe():
    # === 加载指数日收益 ===
    index_ret = load_csv_daily("index_data", "S_DQ_RET")
    index_ret = index_ret[[index_name]]
    index_ret.index = pd.to_datetime(index_ret.index)

    index_ret = index_ret.loc[factor_start_str_forward:factor_end_str_forward, :]

    # === 初始化自定义季度结果 ===
    all_dates = index_ret.index
    years = sorted(set(d.year for d in all_dates))

    custom_index_ret = pd.DataFrame(index=[], columns=[index_name])
    
    for year in years:
        # Q1: 05-01 ~ 08-31 → 03-31
        start_q1 = pd.Timestamp(f"{year}-05-01")
        end_q1 = pd.Timestamp(f"{year}-08-31")
        q1_data = index_ret[(index_ret.index >= start_q1) & (index_ret.index <= end_q1)]
        if not q1_data.empty:
            custom_index_ret.loc[pd.Timestamp(f"{year}-03-31")] = q1_data.sum(skipna=True)

        # Q2: 09-01 ~ 10-31 → 06-30
        start_q2 = pd.Timestamp(f"{year}-09-01")
        end_q2 = pd.Timestamp(f"{year}-10-31")
        q2_data = index_ret[(index_ret.index >= start_q2) & (index_ret.index <= end_q2)]
        if not q2_data.empty:
            custom_index_ret.loc[pd.Timestamp(f"{year}-06-30")] = q2_data.sum(skipna=True)

        # Q3: 11-01 ~ 次年 04-30 → 次年 09-30
        start_q3 = pd.Timestamp(f"{year}-11-01")
        end_q3 = pd.Timestamp(f"{year+1}-04-30")
        q3_data = index_ret[(index_ret.index >= start_q3) & (index_ret.index <= end_q3)]
        if not q3_data.empty:
            custom_index_ret.loc[pd.Timestamp(f"{year}-09-30")] = q3_data.sum(skipna=True)
        
        # Q4: 11-01 ~ 次年 04-30 → 年 12-31
        custom_index_ret.loc[pd.Timestamp(f"{year}-12-31")] = np.nan

    custom_index_ret = custom_index_ret.sort_index()
    custom_index_ret = custom_index_ret[custom_index_ret.index >= pd.Timestamp(factor_start_str)]
    custom_index_ret = custom_index_ret[custom_index_ret.index <= pd.Timestamp(factor_end_str)]
    
    # === 年化指数收益 ===
    for dt in custom_index_ret.index:
        if dt.month == 6:
            custom_index_ret.loc[dt] *= 6
        elif dt.month == 9:
            custom_index_ret.loc[dt] *= 2
        elif dt.month == 3:
            custom_index_ret.loc[dt] *= 3
        
    index_ret_arr = custom_index_ret
    return index_ret_arr


eval_mask = None
restrict = None
ntr_ic_x = None
index_ret_arr = None
excess_return = None
return_residual = None
fitness_cache = None

try:
    metric_set = set([fitness_metric] + gen_metric_list)
    cache_data = CacheData()
    
    if "hedge_sharpe" in metric_set:
        cache_data.prep_cache_data_fitness_hedged_sharpe()
    if ("neu_IC" in metric_set) or ("long_only_neu_IC" in metric_set):
        cache_data.prep_cache_data_fitness_neutralized_ic()

    eval_mask = cache_data.eval_mask
    restrict = cache_data.restrict      
    ntr_ic_x = cache_data.ntr_ic_x
    index_ret_arr = cache_data.index_ret_arr
    excess_return = cache_data.excess_return
    return_residual = cache_data.return_residual

    fitness_cache = [eval_mask, restrict, ntr_ic_x, index_ret_arr, excess_return, return_residual]
    print("✅ cache data loaded")
except Exception as e:
    traceback.print_exc()
    print("❌ error loading cache data:", e)
