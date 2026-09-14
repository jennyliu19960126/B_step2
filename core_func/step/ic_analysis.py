import os
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
from joblib import Parallel, delayed
import multiprocessing as mp
from multiprocessing import Pool
import gc
import psutil

sys.path.append(str(Path(__file__).resolve().parent.parent))
from constant.path import output_root
from constant.params import (dir_kw, eval_start_str, factor_end_str, summary_df_cols, ic_corr_cut, ic_cut_dict,
                             ic_sort_metric, h_hori, signal_filter_max_num, n_jobs_step4, portf_num, use_mono, factor_rate, version_for_batch_merge)
#from data_reader.cache_data import CacheData, ntr_ic_x, fitness_cache
#from data_reader.data_reader_csv import load_csv_daily, load_rolling_ret
from utility.update_worksheet import update_worksheet
from utility.logger import get_logger
from utility.calc_func import cal_residual, cal_ic_icir, cal_two_ic, cal_port_cumrets
from tqdm import tqdm
from concurrent.futures import as_completed,ProcessPoolExecutor

from data_reader.data_reader_csv import load_csv_daily
from data_reader.data_reader_csv_daily import load_rolling_ret_daily
from data_reader.cache_data_daily import CacheData_daily, ntr_ic_x_daily, fitness_cache_daily



class IcAnalysis:
    def __init__(self,backtest_ori=h_hori):
        # for backtest
        self.start_date = eval_start_str
        self.end_date = factor_end_str
        self.output_root = output_root
        self.dir_kw = dir_kw
        self.max_workers = n_jobs_step4
        self.backtest_hori = backtest_ori
        self.portf_num = portf_num
        self.use_mono = use_mono  
        self.ic_cut_dict = ic_cut_dict
        self.ic_corr_cut = ic_corr_cut
        self.ic_sort_metric = ic_sort_metric
        self.summary_df_cols = summary_df_cols
        self._prep_path()
        self._prep_logger()
        self._prep_data()
    
    def _prep_path(self):
        self.save_dir = os.path.join(self.output_root, "backtest", self.dir_kw)
        #self.save_dir = os.path.join(self.output_root, "backtest_"+version_for_batch_merge, self.dir_kw)
        self.factor_dir = os.path.join(self.save_dir, "factor_daily_pkl")
        self.combine_dir = os.path.join(self.save_dir, "combine_summary")
        self.backtest_dir = os.path.join(self.save_dir, "factor_summary")
        self.summary_file = os.path.join(self.combine_dir, f"factor_{self.start_date}_{self.end_date}_gap{self.backtest_hori}.xls")
        return

    def _prep_data(self):
        self.ret_arr = load_csv_daily("Base", "S_DQ_RET_VWAP_0930_1000_fill", self.start_date, self.end_date).values
        self.restrict_df = load_csv_daily("Base", "S_RESTRICT", self.start_date, self.end_date)
        self.restrict_df.index = pd.to_datetime(self.restrict_df.index)
        cd = CacheData_daily()
        self.ntr_ic_x = cd.prep_cache_data_fitness_neutralized_ic(self.start_date, self.end_date)
        return

    def _prep_logger(self):
        log_path = os.path.join(self.save_dir, f"{self.dir_kw}.log")
        self.logger = get_logger(self.dir_kw, log_path=log_path)
        return

    def ic_analysis(self,research=False):
        self.logger.info(f"IC analysis hori={self.backtest_hori} START \nfrom {self.start_date} to {self.end_date}")
        
        t1 = time.time()
        time_df_path = os.path.join(self.save_dir, f"time_{self.dir_kw}.csv")
        time_df = pd.read_csv(time_df_path)
        
        # select factor by ic&ic_corr
        self.select_factor_in_sequence(research=research)
        
        # pack sub ic results
        self.pack_ic_result()

        t2 = time.time()
        time_df.loc[len(time_df)] = [f'S4_ic_h{self.backtest_hori}(hr)', (t2-t1)/3600]
        time_df.to_csv(time_df_path, index=False)
        
        self.logger.info(f"IC analysis hori={self.backtest_hori} END\n")
        return
    
    def correlation_selection(self, sorted_factor_list, neu_ic_th, long_only_neu_ic_th, prev_num=False):
        signal_names = np.array(sorted_factor_list)

        # select factor by sequences
        winsor_ret_df = load_rolling_ret_daily(quantile=True)

        # T,N,D -> D,N,T
        bench_factor_arr = ntr_ic_x_daily.transpose(2, 1, 0)

        signal_filter1 = self.gen_signal_filter(
            winsor_ret_df, signal_names, bench_factor_arr,
            max_num=signal_filter_max_num,
            ic_cut=neu_ic_th,
            longonly_ic_cut=long_only_neu_ic_th,
            corr_cut=self.ic_corr_cut,
            longonly_corr_cut=self.ic_corr_cut,
            prev_num=prev_num,
            use_mono=False
        )
        signal_filter2 = self.gen_signal_filter(
            winsor_ret_df, signal_names, bench_factor_arr,
            max_num=signal_filter_max_num,
            ic_cut=neu_ic_th,
            longonly_ic_cut=long_only_neu_ic_th,
            corr_cut=self.ic_corr_cut,
            longonly_corr_cut=self.ic_corr_cut,
            prev_num=prev_num,
            use_mono=True
        )

        # selected_factor_list = signal_names[signal_filter].tolist()
        selected_factor_list = signal_names[signal_filter1 | signal_filter2].tolist()
        return selected_factor_list

    def select_factor_in_sequence(self,research=True):
        neu_ic_th = self.ic_cut_dict['neu_IC']
        long_only_neu_ic_th = self.ic_cut_dict['long_only_neu_IC']
        
        summary_df = pd.read_excel(self.summary_file)
        self.logger.info(f"Total {len(summary_df)} factors to be filtered in {self.summary_file}")
        
        quality_cols = [
            "ret",
            "sharpe",
            "mdd",
            "hedge_ret",
            "hedge_sharpe",
            "hedge_mdd",
            "neu_IC",
            "neu_hedge_sharpe",
            "long_only_neu_IC",
        ]
        
        for col in quality_cols:
            if col in summary_df.columns:
                summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce")
        
        summary_df = summary_df.replace([np.inf, -np.inf], np.nan)
        
        before_quality = len(summary_df)
        
        summary_df = summary_df.dropna(subset=quality_cols).copy()
        
        after_quality = len(summary_df)

        summary_df_sub = summary_df[(summary_df['neu_IC'].abs()>neu_ic_th) & (summary_df['long_only_neu_IC']>long_only_neu_ic_th)]
        
        
        
        sorted_factor_list = list(summary_df_sub.sort_values(by=self.ic_sort_metric, ascending=False)['factor_name'].values)
        n_factors = len(sorted_factor_list)
        if n_factors == 0: return
        self.logger.info(
            f"First Filter: neu_IC:{neu_ic_th}, long_only_neu_IC:{long_only_neu_ic_th},len:{n_factors}")
        sub_summary_research_path = os.path.join(self.combine_dir,
                                                 f"sub_summary_research_{self.start_date}_{self.end_date}_gap{self.backtest_hori}_{neu_ic_th:.3f}_{long_only_neu_ic_th:.3f}.xls")
        update_worksheet(sub_summary_research_path, summary_df_sub["factor_name"].tolist(), self.summary_df_cols,
                         self.backtest_hori, self.backtest_dir)
        if research:
            return

        self.logger.info(f"Filter in all {n_factors} factors...")
        selected_factor_list = self.correlation_selection(sorted_factor_list, neu_ic_th, long_only_neu_ic_th)

        self.logger.info(f"\nCombining w/ & w/o mono filtering, finally select {len(selected_factor_list)} factors from total {len(summary_df)} factors")
        self.logger.info(','.join(selected_factor_list))
        # save results
        sub_file = os.path.join(self.combine_dir, f"sub_factor_{self.start_date}_{self.end_date}_gap{self.backtest_hori}_{neu_ic_th:.3f}_{long_only_neu_ic_th:.3f}.xls")
        update_worksheet(sub_file, selected_factor_list, self.summary_df_cols, self.backtest_hori, self.backtest_dir)
        return

    def pack_ic_result(self):
        neu_ic_th = self.ic_cut_dict['neu_IC']
        long_only_neu_ic_th = self.ic_cut_dict['long_only_neu_IC']
        sub_cols = self.summary_df_cols[1:11]
        df = pd.DataFrame(columns=sub_cols)
        sub_file = os.path.join(self.combine_dir, f"sub_factor_{self.start_date}_{self.end_date}_gap{self.backtest_hori}_{neu_ic_th:.3f}_{long_only_neu_ic_th:.3f}.xls")
        sub_df = pd.read_excel(sub_file, index_col=0)
        avg_df = sub_df[sub_cols].abs().mean().to_frame('mean').T
        df = pd.concat([df, avg_df])
        df.to_csv(os.path.join(self.combine_dir, f"sub_summary_{self.start_date}_{self.end_date}_gap{self.backtest_hori}_{neu_ic_th:.3f}_{long_only_neu_ic_th:.3f}.csv"))
        return

    def gen_signal_filter(self,
                        # signal_arrs: List[np.array],
                        winsor_ret_df: pd.DataFrame,
                        signal_names: np.ndarray,
                        bench_factor_arr: np.ndarray,
                        max_num=100,
                        ic_cut=0.01,
                        longonly_ic_cut=0.005,
                        corr_cut=0.8,
                        longonly_corr_cut=0.8,
                        prev_num=False,
                        use_mono=False
                        ) -> np.array:
        
        """计算每个Signal在因子筛选的过程中应该被保留还是剔除

        参数:
            signal_arrs (List[np.array]): 包含所有Signals最近m年数据；
                                        List中每个元素都是一个（N， T=m years）的Signal;
            winsor_ret_df (pd.DataFrame)：代表未来k天的未来收益，
                            （已经进行过shift的操作， 不用再进行shift）;
            signal_names (np.array): 包含所有Signals的名称，与上面Signal_arrs相对应；
                        
            max_num (int): 选择因子目标最大数量（可能达不到）；
            ic_cut (float): 因子在窗口期内的rank ic平均值的绝对值需大于ic_cut；
            longonly_ic_cut (float): 因子在窗口期内的longonly_ic平均值需大于ic_cut；
            corr_cut (float): 待选因子和现有因子的daily ic低于corr_cut才入选；
            longonly_corr_cut(float): 待选因子和现有因子的daily longonly longonly_corr_cut才入选；
            pre_num (int): 之前已经选中的因子数量；
            use_mono (bool): 是否使用mono指标筛选因子；
        
        返回:
            np.array: array[i] = 0, 第i个因子被移除；array[i]=1, 第i个因子被选中；
        """
        
        K = len(signal_names)
        N, T = winsor_ret_df.shape
        
        signal_names_df = pd.DataFrame(signal_names, columns=['factor'])
        selected_index = []
        
        self.logger.info(f'total {K} factors')
        set_x_select = []
        dropped_flag = pd.Series(1, index=signal_names_df.index)
        
        # 多了一条条件
        if prev_num: # 直接保留之前步骤筛选得到的因子
            selected_index = list(range(prev_num))
            set_x_select = list(range(prev_num))
            dropped_flag.loc[set_x_select] = 0
            self.logger.info(f'Number of factors selected previously: {prev_num}')
        while dropped_flag.sum() > 0 and len(set_x_select) < max_num:
            self.logger.info(set_x_select)
            _bench_factor_arr = bench_factor_arr

            if len(selected_index)!=0:
                _temp_factor_arr = np.array([read_single_factor(signal_names[k], self.factor_dir, self.start_date, self.end_date, self.restrict_df.columns).rank(axis=0,pct=True).values for k in selected_index])
                _bench_factor_arr = np.concatenate([_bench_factor_arr,_temp_factor_arr],axis=0)
                self.logger.info(f'bench_factor_arr shape: {_bench_factor_arr.shape}')

            # D,N,T->T,N,D
            ret_resid = cal_residual(
                winsor_ret_df.values.T,
                _bench_factor_arr.transpose(2,1,0)).T

            ret_resid_df = pd.DataFrame(ret_resid)
            ret_resid_rank_df = ret_resid_df.rank(axis=0,pct=True)
            
            ic_array = np.empty((K, winsor_ret_df.shape[1]))
            longonly_ic_array = np.empty((K, winsor_ret_df.shape[1]))
            mono = np.empty(K)
            ic_array[:, :] = np.nan
            longonly_ic_array[:, :] = np.nan
            mono[:] = np.nan
        
            this_set_index = dropped_flag[dropped_flag > 0].index
            eval_mask, restrict, _, _, _, _ = fitness_cache_daily

            self.logger.info('Backtesting IC...')
            # t1 = time.time()
            with Parallel(n_jobs=min(self.max_workers, len(this_set_index)), backend='multiprocessing') as parallel:
                results = parallel(
                    delayed(backtest_factor)(k, ret_resid_rank_df, ret_resid_df, eval_mask, restrict, factor_rate,
                                             signal_names[k], self.factor_dir, self.start_date, self.end_date,
                                             use_mono, self.restrict_df, self.portf_num, self.backtest_hori)
                    for k in tqdm(this_set_index, desc='Backtesting factors', total=len(this_set_index)))

            for j, k in enumerate(this_set_index):
                ic_array_one, longonly_ic_array_one, mono_one = results[j]
                ic_array[k, :] = ic_array_one
                longonly_ic_array[k, :] = longonly_ic_array_one
                mono[k] = mono_one

            mean_ic = pd.DataFrame(ic_array.T).mean()
            mean_ic = mean_ic.abs()
            mean_longonly_ic = pd.DataFrame(longonly_ic_array.T).mean()
            if use_mono:
                mono = pd.Series(mono)
                combined_ic_df = pd.concat([mean_ic, mean_longonly_ic, mono], axis=1)
                combined_ic_df.columns = ['abs_mean_ic', 'mean_longonly_ic', 'mono']
            else:
                combined_ic_df = pd.concat([mean_ic, mean_longonly_ic], axis=1)
                combined_ic_df.columns = ['abs_mean_ic', 'mean_longonly_ic']

            combined_ic_df = combined_ic_df.loc[ (combined_ic_df['abs_mean_ic'] > ic_cut) & (combined_ic_df['mean_longonly_ic'] > longonly_ic_cut)]
            combined_ic_df = combined_ic_df.sort_values('abs_mean_ic', ascending=False)
            
            dropped_flag.loc[~dropped_flag.index.isin(combined_ic_df.index)] = 0
            
            corr_mat = pd.DataFrame(ic_array.T).corr()
            longonly_corr_mat = pd.DataFrame(longonly_ic_array.T).corr()
            
            if len(combined_ic_df) > 0:

                if use_mono:
                    min_mono = combined_ic_df['mono'].min()
                    print(f'min mono: {min_mono}')
                    k = combined_ic_df[combined_ic_df['mono'] == min_mono].index[0]
                else:
                    k = combined_ic_df.index[0]
                set_x_select.append(k)
                selected_index.append(k)
                
                this_corr_s = corr_mat[k]
                this_longonly_corr_s = longonly_corr_mat[k]
                
                to_drop = (
                    list(this_corr_s[this_corr_s.abs() > corr_cut].index) +
                    list(this_longonly_corr_s[this_longonly_corr_s.abs() > longonly_corr_cut].index)
                    )

                if len(to_drop) == 0:
                    to_drop = [k]
                    print(f'Directly drop {k} due to NaN correlation matrix')
                for idx in to_drop:
                    dropped_flag.loc[idx] = 0
                
        self.logger.info(f'select {len(set_x_select)}/{K} factors')
        
        signal_filter = np.zeros((K),dtype=bool)
        signal_filter[selected_index] = True
        
        return signal_filter


def backtest_factor(k, ret_resid_rank_df, ret_resid_df, eval_mask, restrict, factor_rate,
                    factor_name, factor_dir, start_date, end_date,
                    use_mono, restrict_df, portf_num, backtest_hori, mode='full'):

    # read factor value
    #factor_path = os.path.join(factor_dir, f"{factor_name}.pkl")
    factor_path = os.path.join(factor_dir, f"{factor_name}.pkl")
    #with open(factor_path, 'rb') as f:
        #factor_df = pd.read_pickle(f)
    factor_df = pd.read_pickle(factor_path)
    factor_df = factor_df.loc[start_date:end_date]
    factor_df = factor_df[restrict_df.columns]  # T x N
    
    factor_df = factor_df.apply(pd.to_numeric, errors="coerce")
    factor_arr = factor_df.T.values.copy()  # N x T

    # calculate ic
    ic_array_one, longonly_ic_array_one = cal_two_ic(k, factor_arr, ret_resid_rank_df, ret_resid_df, eval_mask, restrict, factor_rate)
    ic = ic_array_one.mean()

    # calculate mono
    if use_mono:
        factor_df.index = pd.to_datetime(factor_df.index)
        output_list = cal_port_cumrets(pd.DataFrame(ret_resid_df.T.values, index=restrict_df.index, columns=restrict_df.columns), factor_df, restrict=restrict_df, portf_num=portf_num,
                                       hori=backtest_hori)
        perf_df = pd.concat(output_list, axis=1)

        if mode == 'full':  # 全10组的mono
            cumret_list = [perf_df[f'group{i}_cumret'].iloc[-1] for i in range(1, 11)]
            sorted_list = sorted(cumret_list) if ic > 0 else sorted(cumret_list, reverse=True)
            cumret_rank_list = [sorted_list.index(cumret) + 1 for cumret in cumret_list]
            mono = sum([abs(x - y) for x, y in zip(range(1, 11), cumret_rank_list)]) / 50
        elif mode == 'longonly':  # longonly版本mono
            if ic > 0:
                top5 = [6, 7, 8, 9, 10]
                cumret_list = [perf_df[f'group{i}_cumret'].iloc[-1] for i in top5]
                sorted_list = sorted(cumret_list)
                cumret_rank_list = [sorted_list.index(cumret) + 6 for cumret in cumret_list]
                mono = sum([abs(x - y) for x, y in zip(top5, cumret_rank_list)]) / 12
            else:
                bottom5 = [1, 2, 3, 4, 5]
                cumret_list = [perf_df[f'group{i}_cumret'].iloc[-1] for i in bottom5]
                sorted_list = sorted(cumret_list, reverse=True)
                cumret_rank_list = [sorted_list.index(cumret) + 1 for cumret in cumret_list]
                mono = sum([abs(x - y) for x, y in zip(bottom5, cumret_rank_list)]) / 12
        else:
            raise ValueError(f"mode must be 'full' or 'longonly', but got {mode}")
        return ic_array_one, longonly_ic_array_one, mono

    else:
        return ic_array_one, longonly_ic_array_one, np.nan


def read_single_factor(factor_name, factor_dir, start_date, end_date, col):
    _factor_path = os.path.join(factor_dir, f"{factor_name}.pkl")

    _factor_df = pd.read_pickle(_factor_path)

    _factor_df = _factor_df.apply(pd.to_numeric, errors="coerce")
    
    _factor_df.index = pd.to_datetime(_factor_df.index)

    _factor_df = _factor_df.loc[start_date:end_date]

    _factor_df = _factor_df[col]  # T x N

    return _factor_df.T  # N x T





