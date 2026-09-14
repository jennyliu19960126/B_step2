import os
import re
import json
import traceback
import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod
from time import time
from warnings import warn
from concurrent.futures import ProcessPoolExecutor, wait, ThreadPoolExecutor
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin, TransformerMixin, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_array, _check_sample_weight
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from multiprocessing import shared_memory

from ._program import _Program
from .gp_utils import check_random_state
from .fitness import fitness_map, fitness_map_3d, _Fitness
from .functions import _function_map, _Function, cs_sigmoid as sigmoid
from .ts_functions import _extra_function_map

from utility.logger import get_logger

all_cal_dictionary = {**_function_map, **_extra_function_map}
__all__ = ['SymbolicRegressor', 'SymbolicClassifier', 'SymbolicTransformer']

MAX_INT = np.iinfo(np.int32).max


class BaseSymbolic(BaseEstimator, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self,
                 *,
                 population_size=1000,
                 hall_of_fame=None,
                 n_components=None,
                 generations=20,
                 tournament_size=20,
                 stopping_criteria=0.0,
                 const_range=[-1., 1.],
                 init_depth=[2, 6],
                 init_method='half and half',
                 function_set=('add', 'sub', 'mul', 'div'),
                 transformer=None,
                 metric='mean absolute error',
                 gen_metric_list=['hedged_sharpe'],
                 parsimony_coefficient=0.001,
                 p_crossover=0.9,
                 p_subtree_mutation=0.01,
                 p_hoist_mutation=0.01,
                 p_point_mutation=0.01,
                 p_point_replace=0.05,
                 max_samples=1.0,
                 class_weight=None,
                 feature_names=None,
                 warm_start=False,
                 low_memory=False,
                 n_jobs=1,
                 verbose=0,
                 random_state=None,
                 corr_penalty=False,
                 gamma=0.5,
                 replacement=False,
                 save_ic_array=False,
                 llm_interpretability_enabled=False,
                 llm_interpretability_min_ic=0.02,
                 llm_interpretability_weight=0.01,
                 llm_interpretability_model='deepseek-v4-flash',
                 llm_interpretability_base_url='https://api.deepseek.com',
                 llm_interpretability_timeout=15,
                 llm_interpretability_api_key='',
                 llm_interpretability_cache_path=None):

        self.population_size = population_size
        self.hall_of_fame = hall_of_fame
        self.n_components = n_components
        self.generations = generations
        self.tournament_size = tournament_size
        self.stopping_criteria = stopping_criteria
        self.const_range = (const_range[0], const_range[1])
        self.init_depth = init_depth
        self.init_method = init_method
        self.function_set = function_set
        self.transformer = transformer
        self.metric = metric
        self.gen_metric_list = gen_metric_list
        self.parsimony_coefficient = parsimony_coefficient
        self.p_crossover = p_crossover
        self.p_subtree_mutation = p_subtree_mutation
        self.p_hoist_mutation = p_hoist_mutation
        self.p_point_mutation = p_point_mutation
        self.p_point_replace = p_point_replace
        self.max_samples = max_samples
        self.class_weight = class_weight
        self.feature_names = feature_names
        self.warm_start = warm_start
        self.low_memory = low_memory
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        self.corr_penalty = corr_penalty
        self.gamma = gamma
        self.replacement = replacement
        self.save_ic_array = save_ic_array
        self.llm_interpretability_enabled = llm_interpretability_enabled
        self.llm_interpretability_min_ic = llm_interpretability_min_ic
        self.llm_interpretability_weight = llm_interpretability_weight
        self.llm_interpretability_model = llm_interpretability_model
        self.llm_interpretability_base_url = llm_interpretability_base_url
        self.llm_interpretability_timeout = llm_interpretability_timeout
        self.llm_interpretability_api_key = llm_interpretability_api_key
        self.llm_interpretability_cache_path = llm_interpretability_cache_path
        # Normal (non-warm-start) runs append the generation-0 PCA factors in
        # fit_3D.  Initialise the collection here so correlation penalisation
        # works regardless of the warm_start mode.
        self.ic_5pc = [] if corr_penalty else None

        # prep map dict
        self.feature_dictionary = self.prep_feature_dict(self.feature_names)
        # placeholder
        self.output_root = "./"
        self.fitness_cache = []
        self.function_cache = []

    @staticmethod
    def prep_feature_dict(feature_names):
        # feature dictionary
        feature_dict = {}
        for index, feature in enumerate(feature_names):
            feature_dict[feature] = "trainX[:,{},:]".format(index)
        return feature_dict

    def _verbose_reporter(self, run_details=None):
        if run_details is None:
            self.logger.info('    |{:^25}|{:^42}|'.format('Population Average',
                                                          'Best Individual'))
            self.logger.info('-' * 4 + ' ' + '-' * 25 +
                             ' ' + '-' * 42 + ' ' + '-' * 10)
            line_format = '{:>4} {:>8} {:>16} {:>8} {:>16} {:>16} {:>10}'
            self.logger.info(line_format.format('Gen', 'Length', 'Fitness', 'Length',
                                                'Fitness', 'OOB Fitness', 'Time Left'))

        else:
            # Estimate remaining time for run
            gen = run_details['generation'][-1]
            generation_time = run_details['generation_time'][-1]
            remaining_time = (self.generations - gen - 1) * generation_time
            if remaining_time > 60:
                remaining_time = '{0:.2f}m'.format(remaining_time / 60.0)
            else:
                remaining_time = '{0:.2f}s'.format(remaining_time)

            oob_fitness = 'N/A'
            line_format = '{:4d} {:8.2f} {:16g} {:8d} {:16g} {:>16} {:>10}'
            if self.max_samples < 1.0:
                oob_fitness = run_details['best_oob_fitness'][-1]
                line_format = '{:4d} {:8.2f} {:16g} {:8d} {:16g} {:16g} {:>10}'

            self.logger.info(line_format.format(run_details['generation'][-1],
                                                run_details['average_length'][-1],
                                                run_details['average_fitness'][-1],
                                                run_details['best_length'][-1],
                                                run_details['best_fitness'][-1],
                                                oob_fitness,
                                                remaining_time))

    def _prep_save_dir(self):
        self.save_dir = os.path.join(self.output_root, "backtest", self.dir_kw)
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_gen_dir = os.path.join(self.output_root, "backtest",self.dir_kw,"gen")
        os.makedirs(self.save_gen_dir, exist_ok=True)
        # self.step1_factor_dir = os.path.join(self.output_root, "backtest", self.dir_kw, "step1_factor_value")
        # os.makedirs(self.step1_factor_dir, exist_ok=True)
        if self.save_ic_array:
            self.save_ic_dir = os.path.join(self.output_root, "backtest",self.dir_kw,"ic_array")
            os.makedirs(self.save_ic_dir, exist_ok=True)
        return

    def _prep_logger(self):
        log_path = os.path.join(self.save_dir, f"{self.dir_kw}.log")
        self.logger = get_logger(self.dir_kw, log_path=log_path)
        return

    def _check_params(self):
        if self.hall_of_fame is None:
            self.hall_of_fame = self.population_size
        if self.hall_of_fame > self.population_size or self.hall_of_fame < 1:
            raise ValueError('hall_of_fame (%d) must be less than or equal to '
                             'population_size (%d).' % (self.hall_of_fame, self.population_size))

        if self.n_components is None:
            self.n_components = self.hall_of_fame
        if self.n_components > self.hall_of_fame or self.n_components < 1:
            raise ValueError('n_components (%d) must be less than or equal to '
                             'hall_of_fame (%d).' % (self.n_components, self.hall_of_fame))

        self._function_set = []
        for function in self.function_set:
            if isinstance(function, str):
                if function not in all_cal_dictionary:
                    raise ValueError(
                        'invalid function name %s found in `function_set`.' % function)
                self._function_set.append(all_cal_dictionary[function])
            elif isinstance(function, _Function):
                self._function_set.append(function)
            else:
                raise ValueError(
                    'invalid type %s found in `function_set`.' % type(function))
        if not self._function_set:
            raise ValueError('No valid functions found in `function_set`.')

        # For point-mutation to find a compatible replacement node
        self._arities = {}
        for function in self._function_set:
            arity = function.arity
            self._arities[arity] = self._arities.get(arity, [])
            self._arities[arity].append(function)

        if isinstance(self.metric, _Fitness):
            self._metric = self.metric
        elif isinstance(self, RegressorMixin):
            base_method = ('mean absolute error', 'mse',
                           'rmse', 'pearson', 'spearman')
            extra_method = tuple(fitness_map_3d.keys())
            total_method = base_method + extra_method
            if self.metric not in total_method:
                raise ValueError('Unsupported metric: %s' % self.metric)
            self._metric = fitness_map[self.metric]
        elif isinstance(self, ClassifierMixin):
            if self.metric != 'log loss':
                raise ValueError('Unsupported metric: %s' % self.metric)
            self._metric = fitness_map[self.metric]
        elif isinstance(self, TransformerMixin):
            base_method = ('pearson', 'spearman')
            extra_method = tuple(fitness_map_3d.keys())
            total_method = base_method + extra_method
            if self.metric not in total_method:
                raise ValueError('Unsupported metric: %s' % self.metric)
            self._metric = fitness_map[self.metric]

        self._method_probs = np.array([self.p_crossover,
                                       self.p_subtree_mutation,
                                       self.p_hoist_mutation,
                                       self.p_point_mutation])
        self._method_probs = np.cumsum(self._method_probs)

        if self._method_probs[-1] > 1:
            raise ValueError('The sum of p_crossover, p_subtree_mutation, '
                             'p_hoist_mutation and p_point_mutation should '
                             'total to 1.0 or less.')

        if self.init_method not in ('half and half', 'grow', 'full'):
            raise ValueError('Valid program initializations methods include '
                             '"grow", "full" and "half and half". Given %s.'
                             % self.init_method)

        if not ((isinstance(self.const_range, tuple) and len(self.const_range) == 2) or self.const_range is None):
            raise ValueError('const_range should be a tuple with length two, or None.')

        if (not isinstance(self.init_depth, list) or len(self.init_depth) != 2):
            raise ValueError('init_depth should be a tuple with length two.')
        if self.init_depth[0] > self.init_depth[1]:
            raise ValueError('init_depth should be in increasing numerical order: (min_depth, max_depth).')

        if self.transformer is not None:
            if isinstance(self.transformer, _Function):
                self._transformer = self.transformer
            elif self.transformer == 'sigmoid':
                self._transformer = sigmoid
            else:
                raise ValueError('Invalid `transformer`. Expected either "sigmoid" or _Function object, got %s' % type(self.transformer))
            if self._transformer.arity != 1:
                raise ValueError('Invalid arity for `transformer`. Expected 1, got %d.' % (self._transformer.arity))
        return

    def _update_params(self, params):
        # update params
        params['_metric'] = self._metric
        params['_gen_metric_list'] = [fitness_map[x] for x in self.gen_metric_list]
        # BaseEstimator.get_params() only sees parameters declared by the
        # concrete estimator.  The cache path is owned by BaseSymbolic, so add
        # it explicitly before parameters are sent to worker processes.
        params['llm_interpretability_cache_path'] = self.llm_interpretability_cache_path
        if hasattr(self, '_transformer'):
            params['_transformer'] = self._transformer
        else:
            params['_transformer'] = None
        params['function_set'] = self._function_set
        params['arities'] = self._arities
        params['method_probs'] = self._method_probs
        return params

    def _parallel_evolve_3D(self, parents, X, y, sample_weight, seed, params, generation, index, steps, index_in_population):
        """Private function used to build a batch of programs within a job."""
        n_dates, n_features, n_stocks = X.shape

        # Unpack parameters
        tournament_size = params['tournament_size']
        function_set = params['function_set']
        arities = params['arities']
        init_depth = params['init_depth']
        init_method = params['init_method']
        const_range = params['const_range']
        metric = params['_metric']
        gen_metric_list = params['_gen_metric_list']
        transformer = params['_transformer']
        parsimony_coefficient = params['parsimony_coefficient']
        method_probs = params['method_probs']
        p_point_replace = params['p_point_replace']
        feature_names = params['feature_names']
        llm_params = {key: params[key] for key in (
            'llm_interpretability_enabled', 'llm_interpretability_min_ic',
            'llm_interpretability_weight', 'llm_interpretability_model',
            'llm_interpretability_base_url', 'llm_interpretability_timeout',
            'llm_interpretability_api_key',
            'llm_interpretability_cache_path')}

        def _tournament():
            """Find the fittest individual from a sub-population."""
            contenders = random_state.randint(0, len(parents), tournament_size)
            fitness = [parents[p].fitness_ for p in contenders]
            if metric.greater_is_better:
                parent_index = contenders[np.argmax(fitness)]
            else:
                parent_index = contenders[np.argmin(fitness)]
            return parents[parent_index], parent_index

        # Build programs
        random_state = check_random_state(seed)
        if parents is None:
            program = None
            genome = None
        else:
            method = random_state.uniform()
            parent, parent_index = _tournament()
            if method < method_probs[0]:
                # crossover
                donor, donor_index = _tournament()
                program, removed, remains = parent.crossover(donor.program, random_state)
                genome = {'method': 'Crossover',
                            'parent_idx': parent_index,
                            'parent_nodes': removed,
                            'donor_idx': donor_index,
                            'donor_nodes': remains}
            elif method < method_probs[1]:
                # subtree_mutation
                program, removed, _ = parent.subtree_mutation(random_state)
                genome = {'method': 'Subtree Mutation',
                            'parent_idx': parent_index,
                            'parent_nodes': removed}
            elif method < method_probs[2]:
                # hoist_mutation
                program, removed = parent.hoist_mutation(random_state)
                genome = {'method': 'Hoist Mutation',
                            'parent_idx': parent_index,
                            'parent_nodes': removed}
            elif method < method_probs[3]:
                # point_mutation
                program, mutated = parent.point_mutation(random_state)
                genome = {'method': 'Point Mutation',
                            'parent_idx': parent_index,
                            'parent_nodes': mutated}
            elif self.replacement == 0:
                # reproduction
                program = parent.reproduce()
                genome = {'method': 'Reproduction',
                            'parent_idx': parent_index,
                            'parent_nodes': []}

        program = _Program(function_set=function_set,
                            arities=arities,
                            init_depth=init_depth,
                            init_method=init_method,
                            n_features=n_features,
                            metric=metric,
                            gen_metric_list=gen_metric_list,
                            transformer=transformer,
                            const_range=const_range,
                            p_point_replace=p_point_replace,
                            parsimony_coefficient=parsimony_coefficient,
                            feature_names=feature_names,
                            random_state=random_state,
                            **llm_params,
                            program=program)

        # init_depth controls only newly generated trees in vanilla GP.  The
        # research configuration treats its upper bound as a real expression
        # cap, so an over-deep crossover/mutation is safely replaced by its
        # selected parent instead of entering the population.
        if parents is not None and program.depth_ > init_depth[1]:
            program.program = parent.reproduce()
            genome = {'method': 'Depth-capped Reproduction',
                      'parent_idx': parent_index,
                      'parent_nodes': []}
        program.parents = genome
        # Draw samples, using sample weights, and then fit
        if sample_weight is None:
            curr_sample_weight = np.ones(n_dates)
        else:
            curr_sample_weight = sample_weight.copy()
        try:
            program.raw_fitness_, program.gen_performance_, program.ic_arr = program.raw_fitness_3D(
                X, y, curr_sample_weight, self.fitness_cache, all_cal_dictionary)

            # ReplacementV2
            if self.replacement == 2:
                if program.parents is not None:
                    if program.raw_fitness_ < parents[program.parents['parent_idx']].raw_fitness_ and program.raw_fitness_ < parents[index_in_population].raw_fitness_:
                        program = parents[index_in_population]
                        genome = {'method': 'Reproduction',
                                  'parent_idx': index_in_population,
                                  'parent_nodes': []}
                        program.parents = genome

            self.logger.info(f"Gen{generation} Index{index+1}/{steps} formulation: {program.__str__()} \nraw_fitness: {program.raw_fitness_:.20f}")
        except Exception as e:
            program.raw_fitness_ = -1
            program.gen_performance_ = [-1]*len(self.gen_metric_list)
            program.ic_arr = pd.Series([np.nan]*y.shape[0])
            self.logger.info(e)
            traceback.print_exc()
        return program

    def _execute_generation(self, parents, X, y, sample_weight, population_size, params, random_state, need_parallel, generation, index_range):
        if isinstance(random_state,(list,np.ndarray,)):
            seeds = random_state
        else:
            random_state = check_random_state(random_state)
            seeds = random_state.randint(MAX_INT, size=population_size).tolist()
        population = []
        if need_parallel:
            pool = ProcessPoolExecutor(max_workers=self.n_jobs)
            fs = []
            size_idx = 0
            for i in range(self.n_jobs):
                cur_size = population_size // self.n_jobs
                if i < population_size % self.n_jobs:
                    cur_size += 1
                fs.append(
                    pool.submit(
                        self._execute_generation,
                        parents=parents,
                        X=X,
                        y=y,
                        sample_weight=sample_weight,
                        population_size=cur_size,
                        params=params,
                        random_state=seeds[size_idx:size_idx+cur_size],
                        need_parallel=False,
                        generation=generation,
                        index_range=index_range[size_idx:size_idx+cur_size]
                    )
                )
                size_idx += cur_size
            for f in fs:
                try:
                    population += f.result()
                except Exception as e:
                    self.logger.info(e)
                    traceback.print_exc()
                    with open(os.path.join(self.save_dir, "error_log.txt"), "a") as f:
                        traceback.print_exc(file=f)
                    raise
        else:
            for i in range(population_size):
                program = self._parallel_evolve_3D(
                    parents=parents, 
                    X=X, 
                    y=y, 
                    sample_weight=sample_weight,
                    seed=seeds[i], 
                    params=params,
                    generation=generation,
                    index=i,
                    steps=population_size,
                    index_in_population=index_range[i]
                )
                population.append(program)
        return population

    def _init_population(self, X, y, sample_weight, params, random_state, need_parallel):
        warm_start_generation = 0
        if self.warm_start == 0 and not hasattr(self, '_programs'):
            self._programs = []   
        elif self.warm_start == 1 and not hasattr(self, '_programs'):
            size = self.population_size * 5
            index_range = list(range(size))
            self.logger.info(f"Warming Start: From {size} to {self.population_size}")
            raw_population = ThreadPoolExecutor(max_workers=1).submit(
                self._execute_generation,
                parents=None, 
                X=X, 
                y=y, 
                sample_weight=sample_weight, 
                population_size=size, 
                params=params, 
                random_state=random_state, 
                need_parallel=need_parallel, 
                generation=warm_start_generation,
                index_range=index_range
            ).result()

            fitness = [program.raw_fitness_ for program in raw_population]
            length = [program.length_ for program in raw_population]

            parsimony_coefficient = None
            if self.parsimony_coefficient == 'auto':
                parsimony_coefficient = (np.cov(length, fitness)[1, 0]/np.var(length))
            for program in raw_population:
                program.fitness_ = program.fitness(parsimony_coefficient)
            
            fitness_list = [p.fitness_ for p in raw_population]
            idx_list = list(range(size))
            idx_by_fitness = [i for _,i in sorted(zip(fitness_list, idx_list), reverse=True)]
            slt_idx = idx_by_fitness[:self.population_size]
            if self.corr_penalty:
                # 用PCA对种群中10000个因子的IC降维得到5个主成分
                ic_list = [raw_population[i].ic_arr for i in slt_idx]
                ic_array = np.transpose(ic_list)  # T x N
                ic_array[np.isnan(ic_array)] = 0
                ic_array[np.isinf(ic_array)] = 0
                pca = PCA(n_components=5)
                pca.fit(ic_array)
                ic_5pc = pca.transform(ic_array)
                self.ic_5pc = [ic_5pc]

            if self.save_ic_array:
                ic_list = [raw_population[i].ic_arr for i in slt_idx]
                ic_pd = pd.DataFrame(ic_list)  # N x T
                ic_pd.to_csv(os.path.join(self.save_ic_dir, f"ic_array_gen0.csv"), index=False)

            self._programs = [[raw_population[i] for i in slt_idx]]

        elif isinstance(self.warm_start, str):
            # place holder
            for i in range(len(self._programs)):
                _ = random_state.randint(MAX_INT, size=self.population_size)            
        return


    def fit_3D(self, X, y, need_parallel=True, fitness_threshold=None):
        self.logger.info('checking inputs...')
        random_state = check_random_state(self.random_state)
        self.logger.info(f"random_state: {random_state}")
        # Check arrays
        sample_check_arr = np.array(self.sample_weight.sum()*[1])
        if self.sample_weight is not None:
            sample_weight = _check_sample_weight(
                self.sample_weight, sample_check_arr)
        self._check_params()

        # update params
        params_raw = self.get_params()
        params = self._update_params(params_raw)
        sensitive_markers = ('api_key', 'token', 'secret', 'password')
        params_print = {
            k: ('<REDACTED>' if any(marker in k.lower() for marker in sensitive_markers) else str(v))
            for k, v in params.items()
            if k not in ['function_set', 'arities', 'feature_names']
        }
        self.logger.info(f"params:\n {json.dumps(params_print, indent=4)}")

        self._init_population(X, y, sample_weight, params, random_state, need_parallel)
        prior_generations = len(self._programs)
        n_more_generations = self.generations - prior_generations

        if n_more_generations < 0:
            raise ValueError('generations=%d must be larger or equal to len(_programs)=%d when warm_start==True'
                             % (self.generations, len(self._programs)))
        elif n_more_generations == 0:
            fitness = [program.raw_fitness_ for program in self._programs[-1]]
            warn('Warm-start fitting without increasing n_estimators does not fit new programs.')

        if self.verbose:
            # Print header fields
            self._verbose_reporter()

        # record while learning
        summary = pd.DataFrame(columns=["generation", "formulation", "formulation_stack", "feature_stack"]+self.gen_metric_list+["base_fitness", "llm_interpretability_score", "llm_interpretability_penalty"])
        summary_i = 0
        time_df_col = ['generation', 'generation_time']
        time_df = pd.DataFrame(columns=time_df_col)

        self.run_details_ = {
            'generation': [],
            'average_length': [],
            'average_fitness': [],
            'best_length': [],
            'best_fitness': [],
            'best_oob_fitness': [],
            'generation_time': [],
        }

        self.logger.info("backtest START")
        for gen in range(prior_generations, self.generations+1):
            self.logger.info(f"\n\n############################ Generation {gen} ############################\n")
            start_time = time()
            if gen == 0:
                parents = None
            else:
                parents = self._programs[gen - 1]
                if self.corr_penalty:
                    # penalize fitness with R2
                    if self.ic_5pc is not None:
                        fitness_series = pd.Series(0, index=range(self.population_size))
                        r2_series = pd.Series(0, index=range(self.population_size))
                        for i, program in enumerate(parents):
                            ic_arr = program.ic_arr.copy()
                            ic_arr = ic_arr.fillna(0)
                            ic_arr = ic_arr.replace([np.inf, -np.inf], 0)
                            model = LinearRegression()
                            model.fit(self.ic_5pc[gen - 1], ic_arr)
                            r2_score = model.score(self.ic_5pc[gen - 1], ic_arr)
                            r2_series[i] = r2_score
                            fitness_series[i] = program.fitness_
                        fitness_rank = fitness_series.rank(pct=True)
                        r2_rank = r2_series.rank(pct=True)
                        for i, program in enumerate(parents):
                            program.fitness_ = fitness_rank[i] - self.gamma * r2_rank[i]

            index_range = list(range(self.population_size))
            population = ThreadPoolExecutor(max_workers=1).submit(
                    self._execute_generation,
                    parents=parents, 
                    X=X,
                    y=y,
                    sample_weight=sample_weight, 
                    population_size=self.population_size, 
                    params=params, 
                    random_state=random_state, 
                    need_parallel=need_parallel,
                    generation=gen,
                    index_range=index_range
                ).result()

            # ReplacementV3
            fitness = []
            if self.replacement == 3 and parents is not None:
                combined_population = parents + population
                fitness = [program.raw_fitness_ for program in combined_population]
                length = [program.length_ for program in combined_population]

                parsimony_coefficient = None
                if self.parsimony_coefficient == 'auto':
                    parsimony_coefficient = (np.cov(length, fitness)[1, 0] / np.var(length))
                for program in combined_population:
                    program.fitness_ = program.fitness(parsimony_coefficient)

                fitness_list = [program.fitness_ for program in combined_population]
                idx_list = list(range(len(combined_population)))
                idx_by_fitness = [i for _, i in sorted(zip(fitness_list, idx_list), reverse=True)]
                slt_idx = idx_by_fitness[:self.population_size]
                population = [combined_population[i] for i in slt_idx]
                fitness = [program.raw_fitness_ for program in population]
            else:
                fitness = [program.raw_fitness_ for program in population]
                length = [program.length_ for program in population]

                parsimony_coefficient = None
                if self.parsimony_coefficient == 'auto':
                    parsimony_coefficient = (np.cov(length, fitness)[1, 0] / np.var(length))
                for program in population:
                    program.fitness_ = program.fitness(parsimony_coefficient)

            if self.save_ic_array:
                ic_list = [i.ic_arr for i in population]
                ic_pd = pd.DataFrame(ic_list)  # N x T
                ic_pd.to_csv(os.path.join(self.save_ic_dir, f"ic_array_gen{gen}.csv"), index=False)

            if self.corr_penalty:
                # 用PCA对种群中10000个因子的IC降维得到5个主成分
                ic_list = [i.ic_arr for i in population]
                ic_array = np.transpose(ic_list)  # T x N
                ic_array[np.isnan(ic_array)] = 0
                ic_array[np.isinf(ic_array)] = 0
                pca = PCA(n_components=5)
                pca.fit(ic_array)
                ic_5pc = pca.transform(ic_array)
                self.ic_5pc.append(ic_5pc)

            formulation_set = set()
            for program in population:
                if not program.__str__() in formulation_set and program.raw_fitness_>fitness_threshold:
                    summary.loc[summary_i] = ([gen, program.__str__(), program.formulation_stack, program.feature_stack]
                                              + program.gen_performance_ + [program.base_fitness_, program.llm_interpretability_score_, program.llm_interpretability_penalty_])
                    formulation_set.add(program.__str__())
                    summary_i += 1
            gen_summary = summary[summary["generation"]==gen]
            gen_summary.to_csv(os.path.join(self.save_gen_dir, f"raw_result_{self.dir_kw}_{gen}.csv"))
            summary.to_csv(os.path.join(self.save_dir, f"raw_result_{self.dir_kw}.csv"))


            self._programs.append(population)

            # Remove old programs that didn't make it into the new population.
            if not self.low_memory:
                for old_gen in np.arange(gen, 0, -1):
                    indices = []
                    for program in self._programs[old_gen]:
                        if program is not None and program.parents is not None:
                            for idx in program.parents:
                                if 'idx' in idx:
                                    indices.append(program.parents[idx])
                    indices = set(indices)
                    for idx in range(self.population_size):
                        if idx not in indices:
                            self._programs[old_gen - 1][idx] = None
            elif gen > 0:
                # Remove old generations
                self._programs[gen-1] = None

            # Record run details
            if self._metric.greater_is_better:
                best_index = np.argmax(fitness)
            else:
                best_index = np.argmin(fitness)
            best_program = population[best_index]
            best_fitness = fitness[best_index]

            oob_fitness = np.nan
            if self.max_samples < 1.0:
                oob_fitness = best_program.oob_fitness_
            self.run_details_['generation'].append(gen)    
            self.run_details_['average_length'].append(float(np.mean(length)))
            self.run_details_['average_fitness'].append(float(np.mean(fitness)))
            self.run_details_['best_length'].append(best_program.length_)
            self.run_details_['best_fitness'].append(best_fitness)
            self.run_details_['best_oob_fitness'].append(oob_fitness)
            generation_time = time() - start_time
            self.run_details_['generation_time'].append(generation_time)

            if self.verbose:
                self._verbose_reporter(self.run_details_)

            if self._metric.greater_is_better:
                if best_fitness >= self.stopping_criteria:
                    break
            else:
                if best_fitness <= self.stopping_criteria:
                    break

            # record run_details in generation table 
            time_df.loc[gen] = [self.run_details_[k][-1] for k in time_df_col]
            time_df.to_csv(os.path.join(self.save_dir, f"time_{self.dir_kw}.csv"), index=False)
        
        if isinstance(self, TransformerMixin):
            # Find the best individuals in the final generation
            fitness = np.array(fitness)
            if self._metric.greater_is_better:
                hall_of_fame = fitness.argsort()[::-1][:self.hall_of_fame]
            else:
                hall_of_fame = fitness.argsort()[:self.hall_of_fame]
            evaluation = np.array([gp.execute_3D(X).flatten() for gp in [
                                  self._programs[-1][i] for i in hall_of_fame]])
            with np.errstate(divide='ignore', invalid='ignore'):
                correlations = evaluation - \
                    np.nanmean(evaluation, axis=1).reshape(
                        (evaluation.shape[0], 1))
                correlations = np.abs(np.corrcoef(
                    np.nan_to_num(correlations, nan=0.)))
            np.fill_diagonal(correlations, 0.)
            components = list(range(self.hall_of_fame))
            indices = list(range(self.hall_of_fame))
            # Iteratively remove least fit individual of most correlated pair
            while len(components) > self.n_components:
                most_correlated = np.unravel_index(
                    np.argmax(correlations), correlations.shape)
                # The correlation matrix is sorted by fitness, so identifying
                # the least fit of the pair is simply getting the higher index
                worst = max(most_correlated)
                components.pop(worst)
                indices.remove(worst)
                correlations = correlations[:, indices][indices, :]
                indices = list(range(len(components)))
            self._best_programs = [self._programs[-1][i]
                                   for i in hall_of_fame[components]]
        else:
            # Find the best individual in the final generation
            if self._metric.greater_is_better:
                self._program = self._programs[-1][np.argmax(fitness)]
            else:
                self._program = self._programs[-1][np.argmin(fitness)]
        time_df.loc[gen+1] = ["S1_learn(hr)",time_df.loc[0:gen, "generation_time"].sum()/3600]
        time_df.to_csv(os.path.join(self.save_dir, f"time_{self.dir_kw}.csv"), index=False)
        self.logger.info("backtest END")
        return summary


class SymbolicTransformer(BaseSymbolic, TransformerMixin):

    def __init__(self,
                 *,
                 population_size=1000,
                 hall_of_fame=100,
                 n_components=10,
                 generations=20,
                 tournament_size=20,
                 stopping_criteria=1.0,
                 const_range=(-1., 1.),
                 init_depth=(2, 6),
                 init_method='half and half',
                 function_set=('add', 'sub', 'mul', 'div'),
                 metric='pearson',
                 gen_metric_list=['hedged_sharpe'],
                 parsimony_coefficient=0.001,
                 p_crossover=0.9,
                 p_subtree_mutation=0.01,
                 p_hoist_mutation=0.01,
                 p_point_mutation=0.01,
                 p_point_replace=0.05,
                 max_samples=1.0,
                 feature_names=None,
                 warm_start=False,
                 low_memory=False,
                 n_jobs=1,
                 verbose=0,
                 random_state=None,
                 corr_penalty=False,
                 gamma=0.7,
                 replacement=False,
                 save_ic_array=False,
                 llm_interpretability_enabled=False,
                 llm_interpretability_min_ic=0.02,
                 llm_interpretability_weight=0.01,
                 llm_interpretability_model='deepseek-v4-flash',
                 llm_interpretability_base_url='https://api.deepseek.com',
                 llm_interpretability_timeout=15,
                 llm_interpretability_api_key='',
                 llm_interpretability_cache_path=None):
        super(SymbolicTransformer, self).__init__(
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
            llm_interpretability_api_key=llm_interpretability_api_key,
            llm_interpretability_cache_path=llm_interpretability_cache_path)

    def __len__(self):
        """Overloads `len` output to be the number of fitted components."""
        if not hasattr(self, '_best_programs'):
            return 0
        return self.n_components

    def __getitem__(self, item):
        """Return the ith item of the fitted components."""
        if item >= len(self):
            raise IndexError
        return self._best_programs[item]

    def __str__(self):
        """Overloads `print` output of the object to resemble LISP trees."""
        if not hasattr(self, '_best_programs'):
            return self.__repr__()
        output = str([gp.__str__() for gp in self])
        return output.replace("',", ",\n").replace("'", "")

    def _more_tags(self):
        return {
            "_xfail_checks": {
                "check_sample_weights_invariance": (
                    "zero sample_weight is not equivalent to removing samples"
                ),
            }
        }

    def transform(self, X):
        if not hasattr(self, '_best_programs'):
            raise NotFittedError('SymbolicTransformer not fitted.')

        X = check_array(X)
        _, n_features = X.shape
        if self.n_features_in_ != n_features:
            raise ValueError('Number of features of the model must match the '
                             'input. Model n_features is %s and input '
                             'n_features is %s.'
                             % (self.n_features_in_, n_features))

        X_new = np.array([gp.execute(X) for gp in self._best_programs]).T
        return X_new

    def transform_3D(self, X):
        if not hasattr(self, '_best_programs'):
            raise NotFittedError('SymbolicTransformer not fitted.')
        X_new = np.array([gp.execute_3D(X)
                         for gp in self._best_programs]).transpose([1, 0, 2])
        return X_new

    def fit_transform(self, X, y, sample_weight=None):
        return self.fit(X, y, sample_weight).transform(X)
