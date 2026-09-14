import os
import json
import sys
import traceback
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.genetic import SymbolicTransformer
from utility.summary_by_gen import summary_by_generation
from data_reader.cache_data import eval_mask, fitness_cache
#from data_reader.data_reader_csv import load_feature, load_y
from data_reader.data_reader_csv import load_feature, load_y_quarter

from utility.evolution_policy import population_target, feature_origins, select_final_candidates
from constant.params import my_config, feature_tensor_metadata
from constant.path import output_root
from constant.params import (factor_name_prefix, dir_kw, fitness_metric, fitness_threshold, gen_metric_list, feature_names, function_set, eval_start_str, factor_start_str, factor_end_str, 
                             h_hori, n_jobs, random_state, generations, population_size, tournament_size, hall_of_fame, init_depth, init_method, stopping_criteria, 
                             const_range, parsimony_coefficient, p_crossover, p_subtree_mutation, p_hoist_mutation, p_point_mutation, p_point_replace,n_components,max_samples,warm_start,low_memory,verbose,
                             corr_penalty, gamma, replacement, save_ic_array, ic_cut_dict_quar,
                             llm_interpretability_enabled, llm_interpretability_min_ic, llm_interpretability_weight,
                             llm_interpretability_model, llm_interpretability_base_url, llm_interpretability_timeout,
                             llm_interpretability_api_key)


if isinstance(random_state, int):
    np.random.seed(10)
elif random_state is None:
    random_state = np.random.randint(0, 2 ** 32 - 1)
else:
    raise ValueError("random_state must be an integer or None")

pd.set_option('display.max_columns', None)
pd.set_option('expand_frame_repr', True)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.float_format', lambda x: '%.18f' % x)
warnings.filterwarnings('ignore')

class GpLearnStock(SymbolicTransformer):
    def __init__(self,
                 *,
                 population_size=None,
                 population_multiplier=my_config.get("population_multiplier", 50),
                 min_raw_ic=my_config.get("min_raw_ic", 0.01),
                 hall_of_fame=hall_of_fame,
                 n_components=n_components,
                 generations=generations,
                 tournament_size=tournament_size,
                 stopping_criteria=stopping_criteria,
                 const_range=const_range,
                 init_depth=init_depth,
                 init_method=init_method,
                 function_set=function_set,
                 metric=fitness_metric,
                 gen_metric_list=gen_metric_list,
                 parsimony_coefficient=parsimony_coefficient,
                 p_crossover=p_crossover,
                 p_subtree_mutation=p_subtree_mutation,
                 p_hoist_mutation=p_hoist_mutation,
                 p_point_mutation=p_point_mutation,
                 p_point_replace=p_point_replace,
                 max_samples=max_samples,
                 feature_names=feature_names,
                 warm_start=warm_start,
                 low_memory=low_memory,
                 n_jobs=n_jobs,
                 verbose=verbose,
                 random_state=random_state,
                 corr_penalty=corr_penalty,
                 gamma=gamma,
                 replacement=replacement,
                 save_ic_array=save_ic_array,
                 ic_cut_dict_quar = ic_cut_dict_quar,
                 llm_interpretability_enabled=llm_interpretability_enabled,
                 llm_interpretability_min_ic=llm_interpretability_min_ic,
                 llm_interpretability_weight=llm_interpretability_weight,
                 llm_interpretability_model=llm_interpretability_model,
                 llm_interpretability_base_url=llm_interpretability_base_url,
                 llm_interpretability_timeout=llm_interpretability_timeout,
                 llm_interpretability_api_key=llm_interpretability_api_key):
        self.ic_cut_dict_quar = ic_cut_dict_quar
        self.population_multiplier = population_multiplier
        self.min_raw_ic = min_raw_ic
        if min_raw_ic is not None and (not np.isfinite(min_raw_ic) or min_raw_ic < 0):
            raise ValueError("min_raw_ic must be nonnegative and finite, or None")
        self.feature_origins_ = feature_origins(feature_tensor_metadata)
        if list(feature_names) != feature_tensor_metadata["feature_names"]:
            raise ValueError("Feature order must match input metadata")
        if population_size is None:
            population_size = population_target(len(feature_names), population_multiplier)
        hall_of_fame = min(hall_of_fame, population_size)
        n_components = min(n_components, hall_of_fame)
        super(GpLearnStock, self).__init__(
            population_size=population_size,
            hall_of_fame=hall_of_fame,
            n_components=n_components,
            generations=generations,
            tournament_size=tournament_size,
            stopping_criteria=stopping_criteria,
            const_range=const_range,
            init_depth=init_depth,
            init_method=init_method,
            function_set=function_set,
            metric=metric,
            gen_metric_list=gen_metric_list,
            parsimony_coefficient=parsimony_coefficient,
            p_crossover=p_crossover,
            p_subtree_mutation=p_subtree_mutation,
            p_hoist_mutation=p_hoist_mutation,
            p_point_mutation=p_point_mutation,
            p_point_replace=p_point_replace,
            max_samples=max_samples,
            feature_names=feature_names,
            warm_start=warm_start,
            low_memory=low_memory,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
            corr_penalty=corr_penalty,
            gamma=gamma,
            replacement=replacement,
            save_ic_array=save_ic_array,
            llm_interpretability_enabled=llm_interpretability_enabled,
            llm_interpretability_min_ic=llm_interpretability_min_ic,
            llm_interpretability_weight=llm_interpretability_weight,
            llm_interpretability_model=llm_interpretability_model,
            llm_interpretability_base_url=llm_interpretability_base_url,
            llm_interpretability_timeout=llm_interpretability_timeout,
            llm_interpretability_api_key=llm_interpretability_api_key)

    def set_params_backtest(self, need_parallel=True):
        # set params
        self.factor_name_prefix = factor_name_prefix
        self.need_parallel = need_parallel
        self.eval_start = eval_start_str
        self.factor_start = factor_start_str
        self.factor_end = factor_end_str
        self.output_root = output_root
        self.h_hori = h_hori
        self.dir_kw = dir_kw
        # prep save directory & logger
        self._prep_save_dir()
        if self.llm_interpretability_cache_path is None:
            self.llm_interpretability_cache_path = os.path.join(self.save_dir, "llm_interpretability.sqlite")
        self._prep_logger()
        # load cache data
        self.fitness_cache = fitness_cache
    
    def prep_data_backtest(self):
        # load inputs
        #self.X = load_feature(self.feature_names, self.save_dir) 
        self.X = load_feature(self.feature_names)
        # np.save(os.path.join(self.save_dir, "step1_x.npy"), self.X)
        if self.X.shape[1] != len(self.feature_origins_):
            raise ValueError("Tensor feature count differs from provenance metadata")
        self.y = load_y_quarter()
        # np.save(os.path.join(self.save_dir, "step1_y.npy"), self.y)
        self.logger.info(f"X.shape={self.X.shape}")
        self.logger.info(f"y.shape={self.y.shape}")
        self.sample_weight = np.array([1]*eval_mask.sum())
        self.logger.info(f"sample_weight.sum()={self.sample_weight.sum()}")
        return
    
    def learn_formulation(self):
        self.logger.info(f"\nfactor_range from {self.factor_start} to {self.factor_end} \neval_range from {self.eval_start} to {self.factor_end}")
        self.logger.info(f"\nmetric:{self.metric} \nfeature_names={self.feature_names} \nfunction_set={self.function_set} \nh_hori={self.h_hori} \nsave_dir={self.save_dir}")
        try:
            result = self.fit_3D(
                self.X,
                self.y,
                need_parallel=self.need_parallel,
                fitness_threshold=fitness_threshold
                )
            result.to_csv(os.path.join(self.save_dir, f"raw_result_{self.dir_kw}.csv"))
            result_dropdup = result.drop_duplicates(subset=['formulation'], keep='first').reset_index().rename(columns={"index":"factor_name"})
            result_dropdup['factor_name'] = result_dropdup['factor_name'].apply(lambda x: f"{self.factor_name_prefix}{x}")
            result_dropdup.to_csv(os.path.join(self.save_dir, f"original_result_{self.dir_kw}.csv"), index=False)
            # The active fitness may be raw IC or a neutralized metric.  Do
            # not hard-code neu_IC here: it would silently reintroduce a
            # neutralized selection criterion after raw-IC GP optimisation.
            ic_threshold = self.ic_cut_dict_quar.get(self.metric, 0)
            result_dropdup_sub, counts = select_final_candidates(
                result_dropdup, self.feature_origins_, self.metric, ic_threshold)
            result_dropdup_sub.to_csv(os.path.join(self.save_dir, f"result_{self.dir_kw}.csv"), index=False)
            statistics = {"all": counts}
            completed_generation = max(self.run_details_["generation"], default=0)
            for cutoff in (4, 8):
                if completed_generation < cutoff:
                    continue
                selected, counts = select_final_candidates(
                    result_dropdup, self.feature_origins_, self.metric, ic_threshold, cutoff)
                selected.to_csv(os.path.join(self.save_dir, f"result_{self.dir_kw}_through_gen{cutoff}.csv"), index=False)
                statistics[f"through_gen{cutoff}"] = counts
            with open(os.path.join(self.save_dir, "final_counts.json"), "w", encoding="utf-8") as handle:
                json.dump(statistics, handle, indent=2)
            result_dropdup = pd.read_csv(os.path.join(self.save_dir, f"original_result_{self.dir_kw}.csv"))
            # self.logger.info(f"\n{result_dropdup}")
            self.logger.info("performance statistics by geneartion | START")
            gen_summary, feature_summary, function_summary = summary_by_generation(result_df=result_dropdup, raw_result_df=result)
            gen_summary.to_csv(os.path.join(self.save_dir, "gen_summary_all.csv"))
            feature_summary.to_csv(os.path.join(self.save_dir, "feature_summary_all.csv"))
            function_summary.to_csv(os.path.join(self.save_dir, "function_summary_all.csv"))
            self.logger.info("performance statistics by geneartion | END")
        except Exception as e:
            traceback.print_exc()
            self.logger.info(e)
            raise
