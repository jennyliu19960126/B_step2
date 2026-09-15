import os
import ast
import sys
import time
import traceback
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ALL_COMPLETED, ProcessPoolExecutor, as_completed, wait
import multiprocessing as mp

sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.genetic import all_cal_dictionary
from constant.params import feature_names, factor_start_str, factor_end_str, STOCKS, dir_kw, n_jobs, my_config
from constant.path import output_root
from data_reader.data_reader_csv import load_feature
from utility.logger import get_logger
from utility.tradedate import calendar, calendar_quarterly

from concurrent.futures import ThreadPoolExecutor, wait


class FactorReplicate:
    def __init__(self, dir_kw_override=None):
        # for backtest
        self.output_root = output_root
        self.dir_kw = dir_kw_override or dir_kw
        self.feature_names = feature_names
        self.max_workers = n_jobs
        self.factor_dates = calendar_quarterly.loc[factor_start_str:factor_end_str].index.to_list()
        self._prep_save_dir()
        self._prep_logger()
        self._prep_data()

    def _prep_save_dir(self):
        self.save_dir = os.path.join(self.output_root, "backtest", self.dir_kw)
        self.factor_root = os.path.join(self.save_dir, "factor_csv")
        os.makedirs(self.save_dir, exist_ok=True)    
        os.makedirs(self.factor_root, exist_ok=True)    
    
    def _prep_logger(self):
        log_path = os.path.join(self.save_dir, f"{self.dir_kw}.log")
        self.logger = get_logger(self.dir_kw, log_path=log_path)
    
    def _prep_data(self):
        # load inputs
        self.X = load_feature(self.feature_names)
        self.logger.info(f"X.shape={self.X.shape}")
    
    def execute_formulation(self, formulation_str):
        formulation_list = ast.literal_eval(formulation_str)
        # Check for single-node programs
        node = formulation_list[0]

        if isinstance(node, float):
            return np.tile(node, (self.X.shape[0], self.X.shape[2]))
        if isinstance(node, int):
            return self.X[:, node, :]

        apply_stack = []
        for node in formulation_list:
            if isinstance(node, str):
                apply_stack.append([all_cal_dictionary[node]])
            else:
                # Lazily evaluate later
                apply_stack[-1].append(node)

            while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
                # Apply functions that have sufficient arguments
                function = apply_stack[-1][0]
                terminals = [np.tile(t, (self.X.shape[0],self.X.shape[2])) if isinstance(t, float) else self.X[:,t,:] if isinstance(t, int)
                else t for t in apply_stack[-1][1:]]
                intermediate_result = function(*terminals)
                if len(apply_stack) != 1:
                    apply_stack.pop()
                    apply_stack[-1].append(intermediate_result)
                else:
                    return intermediate_result  

    def cal_one_factor(self, formulation_stack, factor_name):
        """
        trainX: called by eval()
        """
        try:
            factor_df = pd.DataFrame(index=self.factor_dates, columns=STOCKS)
            factor_df.iloc[:,:] = self.execute_formulation(formulation_stack)
            factor_df.to_csv(os.path.join(self.factor_root, f'{factor_name}.csv'))
            # 保存为 PKL 格式
            #pkl_path = os.path.join(self.factor_root, f'{factor_name}.pkl')
            #factor_df.to_pickle(pkl_path)
        except Exception as e:
            self.logger.info(e)
            traceback.print_exc()

    def cal_factors(self,formulations):
        # bar = tqdm(enumerate(formulations),position=pos,ncols=20,total=len(formulation_stack))
        bar = enumerate(formulations)
        for i,[factor_name, formulation_stack] in bar:
            # bar.set_description(f"[{i+1}/{len(formulations)}] 因子计算 {factor_name}")
            self.logger.info(f"[{i+1}/{len(formulations)}] 因子计算 {factor_name}")
            self.cal_one_factor(formulation_stack,factor_name)

 
    def replicate_factor(self, mp_mode=True, generation=None):
        self.logger.info("replicate_factor START")
        t1 = time.time()
        time_df_path = os.path.join(self.save_dir, f"time_{self.dir_kw}.csv")
        time_df = pd.read_csv(time_df_path)
        summary_df = pd.read_csv(os.path.join(self.save_dir, f"result_{self.dir_kw}.csv"))

        if generation is not None:
            summary_df = summary_df[summary_df["generation"] == generation]
    
        formulation_dict = summary_df[["factor_name", "formulation_stack"]].set_index("factor_name").to_dict()['formulation_stack']
        factor_list = list(set(formulation_dict.keys()) - set(
            [x.replace(".csv", "") for x in os.listdir(self.factor_root) if x.endswith(".csv")]
        ))

        self.logger.info(f'{len(factor_list)} factors to calculate') 

        formulations = [[factor_name, formulation_dict[factor_name]] for factor_name in factor_list]
        size = len(formulations) // self.max_workers
        if len(formulations) % self.max_workers != 0:
            size += 1

        if mp_mode:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                fs = [
                    pool.submit(
                        self.cal_factors,
                        formulations[pos:pos+size]
                    )
                    for pos in range(0, len(formulations), size)
                ]
                wait(fs)
        else:
            for factor_name, formulation_stack in formulation_dict.items():
                self.cal_one_factor(formulation_stack, factor_name)

        t2 = time.time()
        time_df.loc[len(time_df)] = ['S2_factor(hr)', (t2 - t1) / 3600]
        time_df.to_csv(time_df_path, index=False)
        self.logger.info("replicate_factor END\n")
