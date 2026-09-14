import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['MKL_DYNAMIC'] = 'TRUE'


import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
#from concurrent.futures import ProcessPoolExecutor, wait

sys.path.append(str(Path(__file__).resolve().parent.parent))
from constant.path import output_root
from constant.params import n_jobs, n_jobs_step3, dir_kw, eval_start_str, eval_end_str, summary_df_cols,h_hori,mp_mode,ic_corr_cut,portf_num,index_name,ic_cut_list
from data_reader.cache_data import CacheData, fitness_cache, ntr_ic_x
#from data_reader.data_reader_csv import load_csv_daily,  load_y
from utility.calc_func import max_drawdown_cal, rankdata_nonmiss, cal_residual, cal_ic_icir,cal_port_ret_arr,mask_long_only,cal_port_rets
from utility.update_worksheet import update_worksheet
from utility.logger import get_logger
from tqdm import tqdm

import matplotlib.dates as mdates
import seaborn as sns
from data_reader.data_reader_csv import load_csv_daily
from data_reader.data_reader_csv_daily import load_rolling_ret_daily
from data_reader.cache_data_daily import CacheData_daily, fitness_cache_daily

from joblib import Parallel, delayed
import multiprocessing as mp
from multiprocessing import Pool

class Backtest:
    def __init__(self, backtest_hori=h_hori, mp_mode=mp_mode,
                 include_neutralized=False, dir_kw_override=None,
                 max_workers_override=None):
        # for backtest
        self.mp_mode = mp_mode
        self.include_neutralized = include_neutralized
        self.backtest_hori = backtest_hori
        self.start_date = eval_start_str
        self.end_date = eval_end_str
        # cap worker count to avoid oversubscription when many factors exist
        configured_workers = max_workers_override or n_jobs_step3
        self.max_workers = max(1, min(configured_workers, mp.cpu_count()))
        self.output_root = output_root
        self.dir_kw = dir_kw_override or dir_kw
        self.ic_cut_list = ic_cut_list
        self.ic_corr_cut = ic_corr_cut
        self.portf_num = portf_num
        self.index_name = index_name
        self.summary_df_cols = summary_df_cols
        self._prep_path()
        self._prep_logger()
        self._prep_data()

    def _prep_path(self):
        self.save_dir = os.path.join(self.output_root, "backtest", self.dir_kw)
        self.factor_dir = os.path.join(self.save_dir, "factor_csv")
        #self.factor_daily_dir = os.path.join(self.save_dir, "factor_daily_csv")
        self.combine_dir = os.path.join(self.save_dir, "combine_summary")
        self.backtest_dir = os.path.join(self.save_dir, "factor_summary")
        os.makedirs(self.combine_dir, exist_ok=True)
        os.makedirs(self.backtest_dir, exist_ok=True)
        return

    def _prep_logger(self):
        log_path = os.path.join(self.save_dir, f"{self.dir_kw}.log")
        self.logger = get_logger(self.dir_kw, log_path=log_path)
        return

    def _prep_data(self):
        result_df = pd.read_csv(os.path.join(self.save_dir, f"result_{self.dir_kw}.csv"))
        self.formulation_dict = result_df[["factor_name", "formulation"]].set_index(
            "factor_name").to_dict()['formulation']
        self.restrict_df = load_csv_daily("Base", "S_RESTRICT", self.start_date, self.end_date)

        self.ret_df = load_rolling_ret_daily(self.start_date,self.end_date,quantile=True).T
        self.ori_ret_df = load_rolling_ret_daily(self.start_date,self.end_date,quantile=False).T
        # Post-GP quality analysis is independent from gen_metric_list.  The
        # default raw-IC path loads only benchmark returns; style exposures are
        # prepared lazily when neutralized reporting is explicitly requested.
        cache = CacheData_daily()
        cache.prep_cache_data_fitness_hedged_sharpe(self.backtest_hori)
        self.index_ret_df = pd.Series(
            cache.index_ret_arr, index=self.restrict_df.index
        )
        self.ntr_ic_x = None
        if self.include_neutralized:
            self.ntr_ic_x = cache.prep_cache_data_fitness_neutralized_ic(
                self.start_date, self.end_date
            )
        return

    def backtest(self):
        self.logger.info(f"Backtest hori={self.backtest_hori} START \nfrom {self.start_date} to {self.end_date}")
        
        t1 = time.time()
        time_df_path = os.path.join(self.save_dir, f"time_{self.dir_kw}.csv")
        time_df = pd.read_csv(time_df_path)
        
        # individual factor backtest
        self.logger.info(f"hori={self.backtest_hori} individual factor backtest | START")
        #factor_list = list(self.formulation_dict.keys())# 回测前三个 [:3] 
        all_factors = list(self.formulation_dict.keys())   # 原始全部因子
        
        def summary_exists(factor_name):
            summary_file = os.path.join(
                self.backtest_dir, factor_name, str(self.backtest_hori), "summary.csv"
            )
            if not os.path.exists(summary_file):
                return False
            try:
                df = pd.read_csv(summary_file)
                return len(df) > 0
            except Exception:
                return False
        
        done_factors = [f for f in all_factors if summary_exists(f)]
        factor_list = [f for f in all_factors if not summary_exists(f)]

        self.logger.info(f"already finished factors: {len(done_factors)}")
        self.logger.info(f"remaining factors to run: {len(factor_list)}")

        if self.mp_mode:
            failed_factors = Parallel(
                n_jobs=self.max_workers,
                backend='loky',  
                batch_size=1,
            )(
                delayed(self._run_one_factor_safe)(factor_name)
                for factor_name in tqdm(factor_list, desc="Parallel Backtesting")
            )

            failed_factors = [f for f in failed_factors if f is not None]
            if failed_factors:
                self.logger.warning(
                    f"{len(failed_factors)} factor(s) failed in parallel, retry sequentially: {failed_factors}"
                )
                for factor_name in failed_factors:
                    self.logger.info(f"retry {factor_name} sequentially...")
                    self.backtest_one_factor(factor_name=factor_name)
        else:
            for i, factor_name in enumerate(factor_list):
                self.logger.info(f"[{i}]/[{len(factor_list)}] {factor_name}: backtesting....")
                self.backtest_one_factor(factor_name=factor_name)
                        
                
            
        self.logger.info(f"hori={self.backtest_hori} individual factor backtest | END") 
        
        summary_file = os.path.join(self.combine_dir, f"factor_{self.start_date}_{self.end_date}_gap{self.backtest_hori}.xls")
        summary_path = lambda f: os.path.join(self.backtest_dir, f, str(self.backtest_hori), "summary.csv")
        available_factors = [f for f in all_factors if os.path.exists(summary_path(f))]
        missing_summary = [f for f in all_factors if f not in available_factors]
        if missing_summary:
            self.logger.warning(
                f"Missing summary files for {len(missing_summary)} factors; they will be skipped in the combined workbook"
            )
        update_worksheet(summary_file, available_factors, self.summary_df_cols, self.backtest_hori, self.backtest_dir)

        t2 = time.time()
        time_df.loc[len(time_df)] = [f'S3_backtest_h{self.backtest_hori}(hr)', (t2-t1)/3600]
        time_df.to_csv(time_df_path, index=False)
        
        self.logger.info(f"Backtest hori={self.backtest_hori} END\n")
        return

    def _run_one_factor_safe(self, factor_name):
        """Wrap backtest_one_factor so joblib failures get logged and surfaced."""
        try:
            self.backtest_one_factor(factor_name)
            return None
        except Exception as exc:  # noqa: BLE001
            self.logger.exception(f"[{factor_name}] backtest failed: {exc}")
            return factor_name
    
    # 只回测前三个
    #def backtest_factors(self, factor_names):
        #for i,factor_name in enumerate(factor_names): 
            #self.logger.info(f"[{i}]/[{len(factor_names)}] {factor_name}: backtesting...")
            #self.backtest_one_factor(factor_name)
 
        
    def backtest_one_factor(self, factor_name):
        # output
        save_root = os.path.join(self.backtest_dir, factor_name, str(self.backtest_hori))
        os.makedirs(save_root, exist_ok=True)

        # input
        factor_path = os.path.join(self.factor_dir, f"{factor_name}.csv")
        factor_df_quar = pd.read_csv(factor_path, index_col=0).loc[self.start_date:self.end_date]
        
        #日频的因子处理
        self.restrict_df.index = pd.to_datetime(self.restrict_df.index)
        factor_df_quar.index = pd.to_datetime(factor_df_quar.index)
        
        all_dates = self.restrict_df.index
        years = sorted(set([d.year for d in all_dates]))
        
        # 构建 date → 季度标记 的映射字典
        date_to_quarter = {}

        for year in years:
            # Q1: 5月1日 - 8月31日 → 对应 {year}-03-31
            start_q1 = pd.Timestamp(f"{year}-05-01")
            end_q1 = pd.Timestamp(f"{year}-08-31")
            for d in all_dates:
                if start_q1 <= d <= end_q1:
                    date_to_quarter[d] = pd.Timestamp(f"{year}-03-31")

            # Q2: 9月1日 - 10月31日 → 对应 {year}-06-30
            start_q2 = pd.Timestamp(f"{year}-09-01")
            end_q2 = pd.Timestamp(f"{year}-10-31")
            for d in all_dates:
                if start_q2 <= d <= end_q2:
                    date_to_quarter[d] = pd.Timestamp(f"{year}-06-30")

            # Q3: 11月1日 - 次年4月30日 → 对应 {year}-09-30
            start_q3 = pd.Timestamp(f"{year}-11-01")
            end_q3 = pd.Timestamp(f"{year+1}-04-30")
            for d in all_dates:
                if start_q3 <= d <= end_q3:
                    date_to_quarter[d] = pd.Timestamp(f"{year}-09-30")
                    
        # 构造新表，结构与 b 一致
        factor_df = pd.DataFrame(index=self.restrict_df.index, columns=self.restrict_df.columns)
        
        # 遍历所有日期，填充季度因子值
        for date in factor_df.index:
            qdate = date_to_quarter.get(date, None)
            if qdate in factor_df_quar.index:
                factor_df.loc[date] = factor_df_quar.loc[qdate]
            else:
                factor_df.loc[date] = np.nan  
        
        
        # ========== 保存日频化后的因子到 factor_daily_pkl 文件夹 ==========
        # 构造输出路径
        factor_daily_dir = os.path.join(self.output_root, "backtest", self.dir_kw, "factor_daily_pkl")
        os.makedirs(factor_daily_dir, exist_ok=True)
        
        # 保存 pkl，保留日期索引；降为 float32 以减半体积
        factor_daily_path = os.path.join(factor_daily_dir, f"{factor_name}.pkl")
        factor_df = factor_df.astype("float32")
        factor_df.to_pickle(factor_daily_path)
        
        self.logger.info(f"[{factor_name}] 日频因子保存至: {factor_daily_path}, shape={factor_df.shape}")

        # basic
        formulation = self.formulation_dict[factor_name]
        factor_start = factor_df.first_valid_index()
        factor_end = factor_df.last_valid_index()
        data_size = factor_df.iloc[:-self.backtest_hori].notnull().any(axis=1).sum()
        _stock_num_df = self.cal_stock_num(factor_df, self.restrict_df)
        _coverage_df = self.cal_coverage(_stock_num_df, self.restrict_df)
        #coverage_ratio = _coverage_df.mean()
        
        # 原来 _coverage_df.mean() 返回的是 Series；
        # 这里改成真正的 float 数字
        coverage_ratio = float(_coverage_df["coverage"].mean())
        self.plot_stock_num_coverage(factor_name, _stock_num_df, _coverage_df, save_root)

        # stats
        factor_df[self.restrict_df != 0] = np.nan
        ic, icir, ann_ret, sharpe, mdd, hedge_ret, hedge_sharpe, hedge_mdd = self.backtest_one_mode(
            factor_name, factor_df.copy(), save_root, 
            if_neutralized=False, long_only=False)
        neu_ic = np.nan
        neu_hedge_sharpe = np.nan
        long_only_neu_ic = np.nan
        if self.include_neutralized:
            neu_ic, neu_hedge_sharpe = self.backtest_one_mode(
                factor_name, factor_df.copy(), save_root,
                if_neutralized=True, long_only=False)
            long_only_neu_ic = self.backtest_one_mode(
                factor_name, factor_df.copy(), save_root,
                if_neutralized=True, long_only=True)

        # save results
        summary_list = [
            factor_name, ic, icir, ann_ret, sharpe, mdd, hedge_ret, hedge_sharpe, hedge_mdd,
            neu_ic, neu_hedge_sharpe, long_only_neu_ic, factor_start, factor_end, data_size, coverage_ratio, formulation]
        summary_file = os.path.join(save_root,  "summary.csv")
         
        summary = pd.DataFrame([summary_list], columns=self.summary_df_cols)
        summary.to_csv(summary_file, index=False)
        return

    def backtest_one_mode(self, factor_name, factor_df, save_root, if_neutralized=False, long_only=False):
        ret_df = self.ret_df.copy()

        ret_df.index = pd.to_datetime(ret_df.index).normalize()
        factor_df.index = factor_df.index.normalize()
        self.restrict_df.index = self.restrict_df.index.normalize()
        
        common_index = factor_df.index.intersection(ret_df.index).intersection(self.restrict_df.index)
        factor_df = factor_df.loc[common_index]
        ret_df = ret_df.loc[common_index]
        self.restrict_df = self.restrict_df.loc[common_index]
        
        if if_neutralized == False and long_only==False:
            file_prefix = ""
            excess_return = self.ori_ret_df.values - self.index_ret_df.values.reshape(-1,1)
        elif if_neutralized==True and long_only==False:
            file_prefix = "neu_"
            y_residual = cal_residual(ret_df.values,self.ntr_ic_x)
            ret_df[:] = y_residual
            excess_return = ret_df.values
        elif if_neutralized==True and long_only==True:
            file_prefix = "long_only_neu_"
            y_residual = cal_residual(ret_df.values,self.ntr_ic_x)
            y_pred_long = mask_long_only(y_residual,factor_df.values,self.restrict_df.values)
            factor_df.loc[:, :] = y_pred_long
            ret_df[:] = y_residual
            ret_df = ret_df[factor_df.notnull()]
            
        ic_file = os.path.join(save_root, f'{file_prefix}IC_{factor_name}.csv')
        ls_file = os.path.join(save_root, f"{file_prefix}long_short_return_{factor_name}.csv")
        port_ret_file = os.path.join(save_root, f"{file_prefix}group_return_{factor_name}.csv")

        
        # IC
        factor_rank = factor_df.rank(axis=1, pct=True)
        ret_rank = ret_df.rank(axis=1, pct=True)
        valid_counts = ((~factor_rank.isna()) & (~ret_rank.isna())).sum(axis=1)
               
        ic_mean, icir, ic_series = cal_ic_icir(factor_df.rank(axis=1,pct=True), ret_df.rank(axis=1,pct=True),method="pearson")
        ic_series.to_csv(ic_file)
        if if_neutralized==True and long_only==True:
            return ic_mean
        
        # portf_ret_df
        portf_ret_arr = cal_port_rets(excess_return, y_pred=factor_df.values,restrict=self.restrict_df.values,portf_num=portf_num)
        if isinstance(portf_ret_arr,int) and portf_ret_arr == -1:
            if if_neutralized==False:
                return (ic_mean, icir, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
            else:
                return (ic_mean, np.nan)

        portf_ret_arr_fill = portf_ret_arr[~np.isnan(portf_ret_arr).any(axis=1)]
        if portf_ret_arr_fill[:, 0].sum() > portf_ret_arr_fill[:, -1].sum():
            sign = 'long small, short large'
            ls_ret = portf_ret_arr[:, 0] - portf_ret_arr[:, -1]
            hedge_ret = portf_ret_arr[:, 0] 
        else:
            sign = 'long large, short small'
            ls_ret = portf_ret_arr[:, -1] - portf_ret_arr[:, 0]
            hedge_ret = portf_ret_arr[:, -1]

        portf_ret_df = pd.DataFrame(portf_ret_arr, index=factor_df.index[1:], columns=range(1, self.portf_num+1))
        portf_ret_df.to_csv(port_ret_file)
        self.plot_portf_group_ret(portf_ret_df, self.index_ret_df.iloc[1:], factor_name, self.index_name, file_prefix, save_root)
        
        ann_ret, ann_sharpe, mdd = self.stat_port_ret(ls_ret, file_prefix=file_prefix)
        hedge_ann_ret, hedge_sharpe, hedge_mdd = self.stat_port_ret(hedge_ret, file_prefix=f"{file_prefix}hedged")

        ls_ret_df = pd.DataFrame(ls_ret, index=factor_df.index[1:])
        hedge_ret_df = pd.DataFrame(hedge_ret, index=factor_df.index[1:])
        ls_ret_df.to_csv(ls_file)
        ls_cum_ret_df = ls_ret_df.cumsum()
        hedge_cum_ret_df  = hedge_ret_df.cumsum()

        self.plot_long_short_portf_ret(ls_cum_ret_df, hedge_cum_ret_df, sign, factor_name, self.index_name, file_prefix, save_root)
        
        if if_neutralized==False:
            return (ic_mean, icir, ann_ret, ann_sharpe, mdd, hedge_ann_ret, hedge_sharpe, hedge_mdd)
        else:
            return (ic_mean, hedge_sharpe)
    
    @staticmethod
    def stat_port_ret(ls_ret, file_prefix):
        ls_cum_ret = np.nancumsum(ls_ret)
        mean_annualized_return = np.nanmean(ls_ret)*250
        mean_annualized_sharpe_ratio = np.nanmean(ls_ret)/np.nanstd(ls_ret) * (250**0.5)
        max_drawdown = max_drawdown_cal(ls_cum_ret)
        return mean_annualized_return, mean_annualized_sharpe_ratio, max_drawdown

    @staticmethod
    def plot_portf_group_ret(portf_ret_df, index_ret_df, factor_name, index_name, file_prefix, save_root):

        png_file = os.path.join(save_root, f"{file_prefix}group_return_{factor_name}.png")

        portf_cum_ret_df = portf_ret_df.cumsum()
    
        plt.figure(figsize=(12, 6), dpi=300)
        ax = plt.gca()
        ax.set_facecolor("white")
    
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=30, ha='right', fontsize=10)
    
        plt.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.6)
    
        palette = sns.color_palette("tab10", n_colors=portf_cum_ret_df.shape[1])
    
        for idx, col in enumerate(portf_cum_ret_df.columns):
            ax.plot(
                portf_cum_ret_df.index,
                portf_cum_ret_df[col],
                label=f'portf_{col}',
                color=palette[idx % len(palette)],
                linewidth=1.2,
                alpha=0.9
            )
    
        plt.title(f"{factor_name}\nPortfolio Cumulative Return by Group", fontsize=14, weight='bold')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.ylabel("Cumulative Return", fontsize=12)
        plt.xlabel("Date", fontsize=12)
    
        plt.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.15),
            ncol=5,
            fontsize=9,
            frameon=False
        )
    
        plt.tight_layout()
        plt.savefig(png_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod

    def plot_long_short_portf_ret(ls_cum_ret, hedge_cum_ret, sign, factor_name, index_name, file_prefix, save_root):

        png_file = os.path.join(save_root, f"{file_prefix}long_short_return_{factor_name}.png")
    
        plt.figure(figsize=(12, 6), dpi=300)
        ax = plt.gca()
        ax.set_facecolor("white")

        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=30, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
    
        ls_cum_ret_interp = ls_cum_ret.interpolate(method='time', limit_direction='both')
        hedge_cum_ret_interp = hedge_cum_ret.interpolate(method='time', limit_direction='both')
    
        palette = sns.color_palette("tab10")
    
        ax.plot(
            ls_cum_ret_interp.index,
            ls_cum_ret_interp.iloc[:, 0],
            label=f'{sign}',
            color=palette[0],
            linestyle='-',
            linewidth=1.5,
            alpha=0.9,
        )
    
        ax.plot(
            hedge_cum_ret_interp.index,
            hedge_cum_ret_interp.iloc[:, 0],
            label=f"{file_prefix}portfolio-{index_name}",
            color=palette[1],
            linestyle='-',
            linewidth=1.5,
            alpha=0.9,
        )
    
        plt.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        plt.grid(True, linestyle='--', alpha=0.3)
    
        plt.title(factor_name, fontsize=14, weight='bold')
        plt.ylabel("Cumulative Return", fontsize=12)
        plt.xlabel("Date", fontsize=12)
    
        plt.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.15),
            ncol=2,
            fontsize=10,
            frameon=False
        )
    
        plt.tight_layout()
        plt.savefig(png_file, dpi=300, bbox_inches='tight')
        plt.close()


    @staticmethod
    def cal_stock_num(factor_df, restrict_df):
        effect_factor_df = factor_df.copy()
        effect_factor_df[restrict_df!=0] = np.nan
        stock_num_df = effect_factor_df.notnull().sum(axis=1).to_frame("stock_num")
        return stock_num_df

    @staticmethod
    def cal_coverage(stock_num_df, restrict_df):
        coverage_ratio_df = stock_num_df['stock_num'] / (restrict_df == 0).sum(axis=1)
        return coverage_ratio_df.to_frame('coverage')

    @staticmethod
    def plot_stock_num_coverage(factor_name, stock_num_df, coverage_ratio_df, save_root=None):
    
        fig, ax1 = plt.subplots(figsize=(12, 6), dpi=150)
        ax2 = ax1.twinx()
    
        ax1.plot(stock_num_df.index, stock_num_df['stock_num'], color='tab:blue',
                 label='Stock Count', linewidth=2)
        ax1.set_ylabel('Stock Count', fontsize=12, color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
    
        ax2.plot(coverage_ratio_df.index, coverage_ratio_df.values, color='tab:orange',
                 label='Coverage Ratio', linewidth=2, linestyle='--')
        ax2.set_ylabel('Coverage Ratio', fontsize=12, color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')
    
        ax1.xaxis.set_major_locator(mdates.YearLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45, fontsize=10)
    
        plt.title(f"Stock Coverage of Factor: {factor_name}", fontsize=14, fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.5)
    
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=10)
    
        plt.tight_layout()
    
        if save_root:
            os.makedirs(save_root, exist_ok=True)
            file_path = os.path.join(save_root, f"stock_num_{factor_name}.png")
            plt.savefig(file_path, bbox_inches='tight')
    
        plt.close()
        return fig
