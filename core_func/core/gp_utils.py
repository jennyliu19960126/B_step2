import numbers
import bottleneck as bd
import numpy as np


def check_random_state(seed):
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState instance' % seed)


def _syntax_adapter(formulation:str):
    '''
    Args:
        formulation: 待解析的语法字符串
    Returns: 字典{'TOT':["condition a","condition b"]}
    '''
    elements = formulation.split(" ")
    iter_combination = ""
    adapted_dictionary={}
    register_flag = ""
    for pos,element in enumerate(elements):
        if element == "TOT" or element == "TRA" or element == "OOB":
            if len(iter_combination)>0:
                adapted_dictionary[register_flag].append(iter_combination)
            register_flag = element
            iter_combination = ""
            if element not in adapted_dictionary.keys():
                adapted_dictionary[element] = []
            continue
        iter_combination = iter_combination+element
        if pos==len(elements)-1:
            adapted_dictionary[register_flag].append(iter_combination)
    return adapted_dictionary


def check_floats(input_list,data):
    for item in input_list:
        if isinstance(item, float) and round(item, 3) == data:
            return True
    return False
# print(_syntax_adapter("TOT ((IC>=0.02) and (IR>=0.05)) TRA (IC>0.02)"))
# print(_syntax_adapter("TRA (IC>0.02) TOT (IC>=0.02) OOB (IR<0.05)"))


def np_shift(x, shift):
    x_shift = np.roll(x, shift=shift, axis=0)
    if shift>0:
        x_shift[:shift, :] = np.nan
    elif shift<0:
        x_shift[shift:, :] = np.nan
    return x_shift


def not_const(x, window, min_periods, atol=1e-08):
    mv_max = bd.move_max(x, window, min_periods, axis=0)
    mv_min = bd.move_min(x, window, min_periods, axis=0)
    mv_diff = mv_max-mv_min
    res = np.full(x.shape, np.nan)
    res[mv_diff>atol] = 1
    return res


