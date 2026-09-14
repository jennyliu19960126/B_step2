import sys
import pandas as pd
import numpy as np
import bottleneck as bd
from pathlib import Path
from joblib import wrap_non_picklable_objects

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utility.calc_func import cal_residual
from data_reader.helper_data import ind_df

__all__ = ['make_function']


class _Function(object):
    def __init__(self, function, name, arity, isRandom=(False,(1,100)), isParam=(False, 1)):
        self.function = function
        self.name = name
        self.arity = arity
        self.isRandom = isRandom[0]
        self.RandRange = isRandom[1]
        if (not isinstance(self.RandRange,tuple)) or (not isinstance(self.RandRange[0],int)) or (not isinstance(self.RandRange[1],int)) or len(self.RandRange)!=2:
            raise TypeError("RandRange 格式错误, 应该是类似(1,100)的tuple")
        self.isParam = isParam[0]
        self.paramValue = isParam[1]     
        if (not isinstance(self.paramValue,int)):
            raise TypeError("paramValue 格式错误, 应该是类似1的integer")   
        self.baseConst = -1

    def __call__(self, *args):
        if self.isRandom and self.baseConst>0:
            if len(args)>1 and isinstance(args[-1],int):
                return self.function(*args)
            return self.function(*args, self.baseConst)
        elif self.isParam:
            return self.function(*args, self.paramValue)
        else:
            return self.function(*args)


def make_function(*, function, name, arity, wrap=True):
    if not isinstance(arity, int):
        raise ValueError('arity must be an int, got %s' % type(arity))
    if not isinstance(function, np.ufunc):
        if function.__code__.co_argcount != arity:
            raise ValueError('arity %d does not match required number of '
                             'function arguments of %d.'
                             % (arity, function.__code__.co_argcount))
    if not isinstance(name, str):
        raise ValueError('name must be a string, got %s' % type(name))
    if not isinstance(wrap, bool):
        raise ValueError('wrap must be an bool, got %s' % type(wrap))

    args = [np.ones(10) for _ in range(arity)]
    try:
        function(*args)
    except (ValueError, TypeError):
        raise ValueError('supplied function %s does not support arity of %d.'
                         % (name, arity))
    if not hasattr(function(*args), 'shape'):
        raise ValueError('supplied function %s does not return a numpy array.'
                         % name)
    if function(*args).shape != (10,):
        raise ValueError('supplied function %s does not return same shape as '
                         'input vectors.' % name)

    args = [np.zeros(10) for _ in range(arity)]
    if not np.all(np.isfinite(function(*args))):
        raise ValueError('supplied function %s does not have closure against '
                         'zeros in argument vectors.' % name)
    args = [-1 * np.ones(10) for _ in range(arity)]
    if not np.all(np.isfinite(function(*args))):
        raise ValueError('supplied function %s does not have closure against '
                         'negatives in argument vectors.' % name)

    if wrap:
        return _Function(function=wrap_non_picklable_objects(function),
                         name=name,
                         arity=arity)
    return _Function(function=function,
                     name=name,
                     arity=arity)


def _protected_division(x1, x2, value=1., atol=0.001):
    with np.errstate(divide='ignore', invalid='ignore'):
        res = np.where(np.abs(x2)>atol, np.divide(x1, x2), value)
        res[np.isnan(x1)|np.isnan(x2)] = np.nan
        return res


def _sqrt(x1):
    return np.sqrt(np.abs(x1))


def _protected_inverse(x1, value=0., atol=0.001):
    with np.errstate(divide='ignore', invalid='ignore'):
        res = np.where(np.abs(x1)>atol, 1./x1, value)
        res[np.isnan(x1)] = np.nan
        return res


def _protected_log(x1):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log(np.abs(x1)+1)


def _cs_rank(x1):
    rank_x = bd.nanrankdata(x1, axis=1)
    return rank_x


def _cs_ortho(y, X):
    n_dates, n_stocks = X.shape
    X1 = X.reshape(n_dates, n_stocks, 1)
    return cal_residual(y, X1)


def _cs_regressor(y, x):
    """Cross-sectional OLS slope of y on x, broadcast to valid stocks."""
    result = np.full(y.shape, np.nan, dtype=float)
    for date_idx in range(y.shape[0]):
        valid = np.isfinite(y[date_idx]) & np.isfinite(x[date_idx])
        if valid.sum() < 2:
            continue
        x_valid = x[date_idx, valid]
        y_valid = y[date_idx, valid]
        x_centered = x_valid - x_valid.mean()
        denominator = np.dot(x_centered, x_centered)
        if denominator <= 1e-12:
            continue
        beta = np.dot(x_centered, y_valid - y_valid.mean()) / denominator
        result[date_idx, valid] = beta
    return result


def _signedpower(x1, a):
    return x1**a


def _signedpower_third(x1):
    with np.errstate(over='ignore', under='ignore'):
        return np.sign(x1)*(abs(x1)**(1/3))


def _signedpower_2(x1):
    with np.errstate(over='ignore', under='ignore'):
        return _signedpower(x1, 2)


def _signedpower_3(x1):
    with np.errstate(over='ignore', under='ignore'):
        return _signedpower(x1, 3)


def _cs_indneutralize(x):
    """Industry z-score on the quarterly tensor's already aligned axes."""
    industry = ind_df.values[:x.shape[0], :x.shape[1]]
    result = np.full(x.shape, np.nan, dtype=float)
    for date_idx in range(x.shape[0]):
        valid = np.isfinite(x[date_idx]) & (industry[date_idx] > 0)
        if not valid.any():
            continue
        values = pd.Series(x[date_idx, valid])
        groups = pd.Series(industry[date_idx, valid])
        std = values.groupby(groups).transform('std')
        result[date_idx, valid] = (
            (values - values.groupby(groups).transform('mean')) / std
        ).values
    return np.clip(result, -3, 3)


def _cs_indrank(x):
    """Percentile rank within industry at each date (1 is highest)."""
    industry = ind_df.values[:x.shape[0], :x.shape[1]]
    result = np.full(x.shape, np.nan, dtype=float)
    for date_idx in range(x.shape[0]):
        valid = np.isfinite(x[date_idx]) & (industry[date_idx] > 0)
        if valid.any():
            values = pd.Series(x[date_idx, valid])
            groups = pd.Series(industry[date_idx, valid])
            result[date_idx, valid] = values.groupby(groups).rank(pct=True).values
    return result


def _cs_fold(x):
    x_ranked = _cs_rank(x)
    x_ranked_adj = x_ranked * 2 - 1
    x_fold = np.abs(x_ranked_adj)
    return x_fold


def _sigmoid(x1):
    """Special case of logistic function to transform to probabilities."""
    with np.errstate(over='ignore', under='ignore'):
        return 1/(1 + np.exp(-x1))


def _condition_and(x, y, z):
    res = x.copy()
    mask = (y>0)&(z>0)
    res[~mask] = np.nan
    return res


def _condition_or(x, y, z):
    res = x.copy()
    mask = (y>0)|(z>0)
    res[~mask] = np.nan
    return res


def _mask_ifelse(x, y, z):
    res = np.full(x.shape, np.nan)
    res[x>0] = y[x>0]
    res[x<=0] = z[x<=0]
    return res


def _cs_mean(x):
    with np.errstate(invalid='ignore'):
        mean_x = np.nanmean(x, axis=1, keepdims=True)
        return np.repeat(mean_x, x.shape[1], axis=1)


def _cs_std(x):
    with np.errstate(invalid='ignore'):
        std_x = np.nanstd(x, axis=1, keepdims=True, ddof=1)
        return np.repeat(std_x, x.shape[1], axis=1)


def _cs_demean(x):
    return x - _cs_mean(x)


def _cs_zscore(x):
    x_demeaned = _cs_demean(x)
    x_std = _cs_std(x)

    with np.errstate(divide='ignore', invalid='ignore'):
        # A constant cross-section has no standardized information.  Returning
        # NaN avoids turning it into the artificial all-ones signal used by the
        # upstream draft implementation.
        res = np.where(np.abs(x_std) < 1e-10, np.nan, np.divide(x_demeaned, x_std))
        res[np.isnan(x)] = np.nan
        res[np.isinf(res)] = np.nan
        return res


def _cs_quantile(x, q=50):
    with np.errstate(invalid='ignore'):
        q_value = np.nanpercentile(x, q, axis=1, keepdims=True)
        return np.repeat(q_value, x.shape[1], axis=1)


def _cs_signed_square(x):
    with np.errstate(over='ignore', invalid='ignore'):
        res = np.sign(x) * (x ** 2)
        res[np.isinf(res)] = np.nan
        return res


def _cs_signmul(x, y):
    with np.errstate(over='ignore', invalid='ignore'):
        res = np.sign(_cs_demean(x) * y)
        res[np.isnan(x) | np.isnan(y)] = np.nan
        return res


def _cs_switch_packet_flexible(x):
    with np.errstate(over='ignore', invalid='ignore'):
        res = -np.abs(x - _cs_quantile(x, q=50))
        res[np.isnan(x)] = np.nan
        return res


def _cs_zscore_diff(x, y):
    return np.subtract(_cs_zscore(x), _cs_zscore(y))


def _cs_rank_diff(x, y):
    return np.subtract(_cs_rank(x), _cs_rank(y))


def _cs_zscore_add(x, y):
    return np.add(_cs_zscore(x), _cs_zscore(y))


def _cs_zscore_min(x, y):
    return np.minimum(_cs_zscore(x), _cs_zscore(y))


def _cs_zscore_max(x, y):
    return np.maximum(_cs_zscore(x), _cs_zscore(y))


def _cs_zscore_mul(x, y):
    return np.multiply(_cs_zscore(x), _cs_zscore(y))


def _cs_zscore_ratio(x, y):
    zx = _cs_zscore(x)
    zy = _cs_zscore(y)

    with np.errstate(divide='ignore', invalid='ignore'):
        res = _protected_division(zx, zy, value=np.nan, atol=1e-10) - 1
        res[np.isnan(zx) | np.isnan(zy)] = np.nan
        res[np.isinf(res)] = np.nan
        return res


def _cs_imbalance_coef(x, y):
    numerator = x - y
    denominator = x + y

    with np.errstate(divide='ignore', invalid='ignore'):
        res = _protected_division(numerator, denominator, value=np.nan, atol=1e-10)
        res[np.isnan(x) | np.isnan(y)] = np.nan
        res[np.isinf(res)] = np.nan
        return res


def _cs_zscore_imbalance_coef(x, y):
    zx = _cs_zscore(x)
    zy = _cs_zscore(y)

    numerator = zx - zy
    denominator = zx + zy

    with np.errstate(divide='ignore', invalid='ignore'):
        res = _protected_division(numerator, denominator, value=np.nan, atol=1e-10)
        res[np.isnan(zx) | np.isnan(zy)] = np.nan
        res[np.isinf(res)] = np.nan
        return res


def _cs_zscore_harmonic_mean(x, y):
    zx = _cs_zscore(x)
    zy = _cs_zscore(y)

    numerator = zx * zy
    denominator = zx + zy

    with np.errstate(divide='ignore', invalid='ignore'):
        res = _protected_division(numerator, denominator, value=np.nan, atol=1e-10)
        res[np.isnan(zx) | np.isnan(zy)] = np.nan
        res[np.isinf(res)] = np.nan
        return res


def _cs_double_scored(x, y):
    n_dates, n_stocks = x.shape
    result = np.full(x.shape, np.nan)

    n_group = 5

    for t in range(n_dates):
        x_t = x[t, :]
        y_t = y[t, :]

        valid = (~np.isnan(x_t)) & (~np.isnan(y_t))

        if valid.sum() < n_group:
            continue

        x_valid = x_t[valid]
        y_valid = y_t[valid]

        x_rank = pd.Series(x_valid).rank(pct=True).values

        try:
            group = pd.qcut(x_rank, q=n_group, labels=False, duplicates='drop')
        except ValueError:
            continue

        group = np.asarray(group, dtype=float)

        y_group_rank = np.full(len(y_valid), np.nan)

        for g in np.unique(group[~np.isnan(group)]):
            idx = group == g

            if idx.sum() < 2:
                continue

            y_group_rank[idx] = pd.Series(y_valid[idx]).rank(pct=True).values

        score = 0.5 * x_rank + 0.5 * y_group_rank

        tmp = np.full(n_stocks, np.nan)
        tmp[valid] = score
        result[t, :] = tmp

    result[np.isinf(result)] = np.nan
    return result


cs_sin = _Function(function=np.sin, name='cs_sin', arity=1)
cs_cos = _Function(function=np.cos, name='cs_cos', arity=1)
cs_tan = _Function(function=np.tan, name='cs_tan', arity=1)
cs_add = _Function(function=np.add, name='cs_add', arity=2)
cs_sub = _Function(function=np.subtract, name='cs_sub', arity=2)
cs_mul = _Function(function=np.multiply, name='cs_mul', arity=2)
cs_div = _Function(function=_protected_division, name='cs_div', arity=2)
cs_max = _Function(function=np.maximum, name='cs_max', arity=2)
cs_min = _Function(function=np.minimum, name='cs_min', arity=2)
cs_abs = _Function(function=np.abs, name='cs_abs', arity=1)
cs_neg = _Function(function=np.negative, name='cs_neg', arity=1)
cs_sqrt= _Function(function=_sqrt, name='cs_sqrt', arity=1)
cs_inv = _Function(function=_protected_inverse, name='cs_inv', arity=1)
cs_log = _Function(function=_protected_log, name='cs_log', arity=1)
cs_rank = _Function(function=_cs_rank, name='cs_rank', arity=1)
cs_ortho = _Function(function=_cs_ortho, name='cs_ortho', arity=2)
cs_residual = _Function(function=_cs_ortho, name='cs_residual', arity=2)
cs_regressor = _Function(function=_cs_regressor, name='cs_regressor', arity=2)
cs_power_third = _Function(function=_signedpower_third, name='cs_power_third', arity=1)
cs_power_2 = _Function(function=_signedpower_2, name='cs_power_2', arity=1)
cs_power_3 = _Function(function=_signedpower_3, name='cs_power_3', arity=1)
cs_sigmoid = _Function(function=_sigmoid, name="cs_sigmoid", arity=1)
cs_indneutral = _Function(function=_cs_indneutralize, name='cs_indneutral', arity=1)
cs_indrank = _Function(function=_cs_indrank, name='cs_indrank', arity=1)
cs_fold = _Function(function=_cs_fold, name='cs_fold', arity=1)
cs_condition_and = _Function(function=_condition_and, name='cs_condition_and', arity=3)
cs_condition_or = _Function(function=_condition_or, name='cs_condition_or', arity=3)
cs_mask_ifelse = _Function(function=_mask_ifelse, name='cs_mask_ifelse', arity=3)
cs_sign = _Function(function=np.sign, name='cs_sign', arity=1)


cs_signed_square = _Function(function=_cs_signed_square, name='cs_signed_square', arity=1)
cs_demean = _Function(function=_cs_demean, name='cs_demean', arity=1)
cs_signmul = _Function(function=_cs_signmul, name='cs_signmul', arity=2)
cs_zscore = _Function(function=_cs_zscore, name='cs_zscore', arity=1)
cs_switch_packet_flexible = _Function(function=_cs_switch_packet_flexible, name='cs_switch_packet_flexible', arity=1)

cs_zscore_diff = _Function(function=_cs_zscore_diff, name='cs_zscore_diff', arity=2)
cs_rank_diff = _Function(function=_cs_rank_diff, name='cs_rank_diff', arity=2)
cs_zscore_add = _Function(function=_cs_zscore_add, name='cs_zscore_add', arity=2)
cs_zscore_min = _Function(function=_cs_zscore_min, name='cs_zscore_min', arity=2)
cs_zscore_max = _Function(function=_cs_zscore_max, name='cs_zscore_max', arity=2)
cs_zscore_mul = _Function(function=_cs_zscore_mul, name='cs_zscore_mul', arity=2)
cs_zscore_ratio = _Function(function=_cs_zscore_ratio, name='cs_zscore_ratio', arity=2)

cs_imbalance_coef = _Function(function=_cs_imbalance_coef, name='cs_imbalance_coef', arity=2)
cs_zscore_imbalance_coef = _Function(function=_cs_zscore_imbalance_coef, name='cs_zscore_imbalance_coef', arity=2)
cs_zscore_harmonic_mean = _Function(function=_cs_zscore_harmonic_mean, name='cs_zscore_harmonic_mean', arity=2)

cs_double_scored = _Function(function=_cs_double_scored, name='cs_double_scored', arity=2)


_function_map = {
    'cs_sin': cs_sin,
    'cs_cos': cs_cos,
    'cs_tan': cs_tan,    
    'cs_add': cs_add,
    'cs_sub': cs_sub,
    'cs_mul': cs_mul,
    'cs_div': cs_div,
    'cs_max': cs_max,
    'cs_min': cs_min, 
    'cs_abs': cs_abs,
    'cs_neg': cs_neg,       
    'cs_sqrt': cs_sqrt,
    'cs_inv': cs_inv,    
    'cs_log': cs_log,
    'cs_rank': cs_rank,
    'cs_ortho': cs_ortho,
    'cs_residual': cs_residual,
    'cs_regressor': cs_regressor,
    'cs_power_third': cs_power_third,
    'cs_power_2': cs_power_2,
    'cs_power_3': cs_power_3,
    'cs_indneutral': cs_indneutral,
    'cs_indrank': cs_indrank,
    'cs_fold': cs_fold,
    'cs_sigmoid': cs_sigmoid,       
    'cs_condition_and': cs_condition_and,
    'cs_condition_or': cs_condition_or,
    'cs_mask_ifelse': cs_mask_ifelse,
    'cs_sign': cs_sign,
    
    'cs_signed_square': cs_signed_square,
    'cs_demean': cs_demean,
    'cs_signmul': cs_signmul,
    'cs_zscore': cs_zscore,
    'cs_switch_packet_flexible': cs_switch_packet_flexible,

    'cs_zscore_diff': cs_zscore_diff,
    'cs_rank_diff': cs_rank_diff,
    'cs_zscore_add': cs_zscore_add,
    'cs_zscore_min': cs_zscore_min,
    'cs_zscore_max': cs_zscore_max,
    'cs_zscore_mul': cs_zscore_mul,
    'cs_zscore_ratio': cs_zscore_ratio,

    'cs_imbalance_coef': cs_imbalance_coef,
    'cs_zscore_imbalance_coef': cs_zscore_imbalance_coef,
    'cs_zscore_harmonic_mean': cs_zscore_harmonic_mean,

    'cs_double_scored': cs_double_scored
  
}
