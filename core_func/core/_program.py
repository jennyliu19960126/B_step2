import warnings
import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.utils.random import sample_without_replacement

from .functions import _Function
#from .gp_utils import check_random_state, first_principal_component
from .gp_utils import check_random_state
from utility.llm_interpretability import interpretability_penalty

warnings.filterwarnings("ignore")

class _Program(object):
    def __init__(self,
                 function_set,
                 arities,
                 init_depth,
                 init_method,
                 n_features,
                 const_range,
                 metric,
                 gen_metric_list,
                 p_point_replace,
                 parsimony_coefficient,
                 random_state,
                 transformer=None,
                 feature_names=None,
                 llm_interpretability_enabled=False,
                 llm_interpretability_min_ic=0.02,
                 llm_interpretability_weight=0.01,
                 llm_interpretability_model='deepseek-v4-flash',
                 llm_interpretability_base_url='https://api.deepseek.com',
                 llm_interpretability_timeout=15,
                 llm_interpretability_api_key='',
                 llm_interpretability_cache_path=None,
                 program=None):

        self.function_set = function_set
        self.arities = arities
        self.init_depth = (init_depth[0], init_depth[1] + 1)
        self.init_method = init_method
        self.n_features = n_features
        self.const_range = const_range
        self.metric = metric
        self.gen_metric_list = gen_metric_list
        self.p_point_replace = p_point_replace
        self.parsimony_coefficient = parsimony_coefficient
        self.transformer = transformer
        self.feature_names = feature_names
        self.program = program
        self.llm_interpretability = {
            'enabled': llm_interpretability_enabled,
            'min_fitness': llm_interpretability_min_ic,
            'penalty_weight': llm_interpretability_weight,
            'model': llm_interpretability_model,
            'base_url': llm_interpretability_base_url,
            'timeout': llm_interpretability_timeout,
            'api_key': llm_interpretability_api_key,
            'cache_path': llm_interpretability_cache_path,
        }

        if self.program is not None:
            if not self.validate_program():
                raise ValueError('The supplied program is incomplete.')
        else:
            self.program = self.build_program(random_state)

        self.raw_fitness_ = None
        self.fitness_ = None
        self.gen_performance_ = None
        self.oob_fitness_ = None
        self.fpc = None
        self.ic_arr = None
        self.parents = None
        self.formulation_stack = None
        self.feature_stack = None
        self.base_fitness_ = None
        self.llm_interpretability_score_ = None
        self.llm_interpretability_penalty_ = 0.0
        self._n_samples = None
        self._max_samples = None
        self._indices_state = None

    def build_program(self, random_state):
        if self.init_method == 'half and half':
            method = ('full' if random_state.randint(2) else 'grow')
        else:
            method = self.init_method
        # 确定了每次增加的最大深度
        max_depth = random_state.randint(*self.init_depth)

        function = random_state.randint(len(self.function_set))
        function = deepcopy(self.function_set[function])
        if function.isRandom:
            current_window = random_state.randint(function.RandRange[0], function.RandRange[1])
            function.baseConst = current_window
        program = [function]
        terminal_stack = [function.arity]
        terminal_value_stack = []

        while terminal_stack:

            depth = len(terminal_stack)
            choice = self.n_features + len(self.function_set)

            choice = random_state.randint(choice)

            if (depth<max_depth) and (method=='full' or choice<=len(self.function_set)):
                function = random_state.randint(len(self.function_set))
                function = deepcopy(self.function_set[function])
                if function.isRandom:
                    current_window = random_state.randint(function.RandRange[0],function.RandRange[1])
                    function.baseConst = current_window
                    program.append(function)
                else:
                    program.append(function)
                terminal_stack.append(function.arity)
            else:

                if self.const_range is not None:

                    terminal = random_state.randint(self.n_features + 1)

                    while True:
                        if (terminal in terminal_value_stack and terminal != self.n_features):
                            terminal = random_state.randint(self.n_features + 1)
                        else:
                            break
                else:
                    terminal = random_state.randint(self.n_features)

                if terminal == self.n_features and function.name in ['cs_add', 'cs_sub', 'cs_mul', 'cs_div', 'cs_max', 'cs_min']:
                    terminal = round(random_state.uniform(*self.const_range),3)
                else:
                    terminal = random_state.randint(self.n_features)
                    
                program.append(terminal)
                terminal_stack[-1] -= 1
                if terminal_stack[-1]>0:
                    terminal_value_stack.append(terminal)
                while terminal_stack[-1] == 0:
                    terminal_value_stack = []
                    terminal_stack.pop()
                    if not terminal_stack:
                        return program
                    terminal_stack[-1] -= 1

        return None

    def validate_program(self):
        terminals = [0]
        for node in self.program:
            if isinstance(node, _Function):
                terminals.append(node.arity)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return terminals == [-1]

    def __str__(self):
        terminals = [0]
        output = ''
        isRandomFunction = 0
        RandomFunctionStack = []
        # RandomFunctionStack[0].arity = 0
        for i, node in enumerate(self.program):
            if isinstance(node, _Function):
                # if node.isRandom:
                RandomFunctionStack.append(deepcopy(node))
                terminals.append(node.arity)
                output += node.name + '('
            else:
                if isinstance(node, int):
                    if self.feature_names is None:
                        output += 'X%s' % node
                    else:
                        output += self.feature_names[node]
                else:
                    output += '%.3f' % node
                terminals[-1] -= 1

                if len(RandomFunctionStack)>0:
                    RandomFunctionStack[-1].arity -= 1
                    if RandomFunctionStack[-1].isRandom and RandomFunctionStack[-1].arity==0:
                        output += ',' + str(RandomFunctionStack[-1].baseConst)
                while terminals[-1] == 0:
                    RandomFunctionStack.pop()
                    terminals.pop()

                    terminals[-1] -= 1
                    if len(RandomFunctionStack)>0:
                        RandomFunctionStack[-1].arity -= 1

                        output += ')'
                        if len(RandomFunctionStack)>0 and RandomFunctionStack[-1].isRandom:
                            output += ',' + str(RandomFunctionStack[-1].baseConst)
                    else:
                        output += ')'
                if i != len(self.program) - 1:
                    output += ', '

        return output

    def _depth(self):
        terminals = [0]
        depth = 1
        for node in self.program:
            if isinstance(node, _Function):
                terminals.append(node.arity)
                depth = max(len(terminals), depth)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return depth - 1

    def _length(self):
        return len(self.program)

    def _get_name_map(self, X):
        name_map = {}
        all_cols = X.columns
        all_cols = [col for col in all_cols if col != "交易日期"]
        col_dictionary = {}
        for pos, col in enumerate(all_cols):
            col_dictionary[pos] = col
        return col_dictionary

    def execute_3D(self, X):
        node = self.program[0]

        if isinstance(node, float):
            return np.tile(node, (X.shape[0], X.shape[2]))
        if isinstance(node, int):
            return X[:, node, :]

        apply_stack = []
        for node in self.program:
            if isinstance(node, _Function):
                apply_stack.append([node])
            else:
                # Lazily evaluate later
                apply_stack[-1].append(node)

            while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
                # Apply functions that have sufficient arguments
                function = apply_stack[-1][0]
                terminals = [np.tile(t, (X.shape[0],X.shape[2])) 
                             if isinstance(t, float)
                             else X[:,t,:] if isinstance(t, int) 
                             else t for t in apply_stack[-1][1:]]
                intermediate_result = function(*terminals)
                if len(apply_stack) != 1:
                    apply_stack.pop()
                    apply_stack[-1].append(intermediate_result)
                else:
                    return intermediate_result

        return

    def get_all_indices(self, n_samples=None, max_samples=None, random_state=None):
        if self._indices_state is None and random_state is None:
            raise ValueError('The program has not been evaluated for fitness '
                             'yet, indices not available.')

        if n_samples is not None and self._n_samples is None:
            self._n_samples = n_samples
        if max_samples is not None and self._max_samples is None:
            self._max_samples = max_samples
        if random_state is not None and self._indices_state is None:
            self._indices_state = random_state.get_state()

        indices_state = check_random_state(None)
        indices_state.set_state(self._indices_state)

        not_indices = sample_without_replacement(
            self._n_samples,
            self._n_samples - self._max_samples,
            random_state=indices_state)
        sample_counts = np.bincount(not_indices, minlength=self._n_samples)
        indices = np.where(sample_counts == 0)[0]

        return indices, not_indices

    def _indices(self):
        return self.get_all_indices()[0]

    def raw_fitness_3D(self, X, y, sample_weight, fitness_cache, function_cache):
        y_pred = self.execute_3D(X)
        if self.transformer:
            y_pred = self.transformer(y_pred)
            
        raw_fitness, ic_arr = self.metric(y, y_pred, sample_weight, fitness_cache)
        self.base_fitness_ = raw_fitness
        self.llm_interpretability_score_ = None
        self.llm_interpretability_penalty_ = 0.0
        penalty = 0.0
        if self.llm_interpretability['cache_path']:
            score, penalty = interpretability_penalty(
                str(self), raw_fitness, **self.llm_interpretability)
            self.llm_interpretability_score_ = score
            self.llm_interpretability_penalty_ = penalty
        raw_fitness -= penalty
            
            
        gen_performance = []
        for m in self.gen_metric_list:
            if m is self.metric:
                gen_performance.append(raw_fitness)
            else:
                perf = m(y, y_pred, sample_weight, fitness_cache)
                gen_performance.append(perf)
        formulation_stack = [x.name if isinstance(x, _Function) else x for x in self.program]
        self.formulation_stack = [s.replace('dynamic_', '')+f"_{f.baseConst}" if isinstance(s, str) and s.startswith('dynamic_') else s for s,f in zip(formulation_stack, self.program)]
        self.feature_stack = [self.feature_names[x] for x in self.program if isinstance(x, int)]
        # calculate first principal component
        # fpc = first_principal_component(y_pred)
        return raw_fitness, gen_performance, ic_arr

    def fitness(self, parsimony_coefficient=None):
        if parsimony_coefficient is None:
            parsimony_coefficient = self.parsimony_coefficient
        penalty = parsimony_coefficient * len(self.program) * self.metric.sign
        return self.raw_fitness_ - penalty

    def get_subtree(self, random_state, program=None):
        if program is None:
            program = self.program
        # Choice of crossover points follows Koza's (1992) widely used approach
        # of choosing functions 90% of the time and leaves 10% of the time.
        probs = np.array([0.9 if isinstance(node, _Function) else 0.1
                          for node in program])
        probs = np.cumsum(probs / probs.sum())
        start = np.searchsorted(probs, random_state.uniform())

        stack = 1
        end = start
        while stack > end - start:
            node = program[end]
            if isinstance(node, _Function):
                stack += node.arity
            end += 1

        return start, end

    def reproduce(self):
        return deepcopy(self.program)

    def crossover(self, donor, random_state):
        start, end = self.get_subtree(random_state)
        removed = range(start, end)
        
        donor_start, donor_end = self.get_subtree(random_state, donor)
        donor_removed = list(set(range(len(donor))) -
                             set(range(donor_start, donor_end)))
        
        return (self.program[:start] +
                donor[donor_start:donor_end] +
                self.program[end:]), removed, donor_removed

    def subtree_mutation(self, random_state):
        chicken = self.build_program(random_state)
        return self.crossover(chicken, random_state)

    def hoist_mutation(self, random_state):
        start, end = self.get_subtree(random_state)
        subtree = self.program[start:end]
        sub_start, sub_end = self.get_subtree(random_state, subtree)
        hoist = subtree[sub_start:sub_end]
        removed = list(set(range(start, end)) -
                       set(range(start + sub_start, start + sub_end)))
        return self.program[:start] + hoist + self.program[end:], removed

    def point_mutation(self, random_state):
        program = deepcopy(self.program)
        
        mutate = np.where(random_state.uniform(size=len(program))<self.p_point_replace)[0]
        for node in mutate:
            if isinstance(program[node], _Function):
                arity = program[node].arity
                
                replacement = len(self.arities[arity])
                replacement = random_state.randint(replacement)
                replacement = self.arities[arity][replacement]
                
                replacement.baseConst = program[node].baseConst
                program[node] = replacement
            else:
                if self.const_range is not None:
                    terminal = random_state.randint(self.n_features + 1)
                else:
                    terminal = random_state.randint(self.n_features)
                if terminal == self.n_features:
                    terminal = random_state.uniform(*self.const_range)
                    if self.const_range is None:
                        
                        raise ValueError('A constant was produced with const_range=None.')
                program[node] = terminal

        return program, list(mutate)

    depth_ = property(_depth)
    length_ = property(_length)
    indices_ = property(_indices)
