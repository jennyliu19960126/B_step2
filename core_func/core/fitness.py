import sys
import numbers
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import wrap_non_picklable_objects

sys.path.append(str(Path(__file__).resolve().parent.parent))
from constant.params import factor_rate, h_hori, portf_num, points_of_year
from utility.calc_func import rankdata_nonmiss, cal_residual, cal_ic_icir

__all__ = ['make_fitness']


class _Fitness(object):
    def __init__(self, function, greater_is_better):
        self.function = function
        self.greater_is_better = greater_is_better
        self.sign = 1 if greater_is_better else -1

    def __call__(self, *args):
        return self.function(*args)


def make_fitness(*, function, greater_is_better, wrap=True):
    if not isinstance(greater_is_better, bool):
        raise ValueError('greater_is_better must be bool, got %s'% type(greater_is_better))
    if not isinstance(wrap, bool):
        raise ValueError('wrap must be an bool, got %s' % type(wrap))
    if function.__code__.co_argcount != 3:
        raise ValueError('function requires 3 arguments (y, y_pred, w), got %d.' % function.__code__.co_argcount)
    if not isinstance(function(np.array([1, 1]),np.array([2, 2]),np.array([1, 1])), numbers.Number):
        raise ValueError('function must return a numeric.')
    if wrap:
        return _Fitness(function=wrap_non_picklable_objects(function),greater_is_better=greater_is_better)
    return _Fitness(function=function,greater_is_better=greater_is_better)


def _mask_eval_data(y_raw, y_pred_raw, sample_weight, eval_mask, restrict):

    y_masked = y_raw[np.where(sample_weight == 1)].copy()
    y_pred_masked = y_pred_raw[np.where(sample_weight == 1)].copy()
    y_masked = y_masked[np.where(eval_mask == 1)]
    y_pred_masked = y_pred_masked[np.where(eval_mask == 1)]

    y_pred_masked[restrict!=0] = np.nan

    factor_coverage_arr = (~np.isnan(y_pred_masked)).sum(
        axis=1)/(restrict == 0).sum(axis=1)
    factor_coverage_mean = factor_coverage_arr.mean()
    if factor_coverage_mean < factor_rate:
        y_pred_masked.fill(np.nan)
    else:
        y_pred_masked[np.where(factor_coverage_arr < factor_rate)] = np.nan

    return y_masked,y_pred_masked


def _cal_total_IC(y, y_pred, method):
    with np.errstate(divide='ignore', invalid='ignore'):
        factor_df = pd.DataFrame(y_pred)
        ret_df = pd.DataFrame(y)
        ic_mean, _, ic_series = cal_ic_icir(factor_df, ret_df, method=method)
        return ic_mean, ic_series


def _weighted_pearson(y_raw, y_pred_raw, sample_weight, fitness_cache):
    """Calculate the weighted Pearson correlation coefficient."""
    # y: array - like, shape = [n_samples] -> [n_dates, n_stocks]
    eval_mask, restrict, ntr_ic_x, index_ret_arr,exret,y_residual = fitness_cache
    y, y_pred = _mask_eval_data(y_raw, y_pred_raw, sample_weight, eval_mask, restrict)
    total_IC, _ = _cal_total_IC(y, y_pred, method="pearson")
    if np.isfinite(total_IC):
        return np.abs(total_IC)
    return 0.


def _alert_weighted_pearson(y_raw, y_pred_raw, sample_weight, fitness_cache):
    """Calculate the weighted Pearson correlation coefficient."""
    # y: array - like, shape = [n_samples] -> [n_dates, n_stocks]
    eval_mask, restrict, ntr_ic_x, index_ret_arr,exret,y_residual = fitness_cache
    y, y_pred = _mask_eval_data(y_raw, y_pred_raw, sample_weight, eval_mask, restrict)
    total_IC, _ = _cal_total_IC(y, y_pred, method="pearson")
    if np.isfinite(total_IC):
        return total_IC
    return -1


def _raw_pearson_ic(y_raw, y_pred_raw, sample_weight, fitness_cache):
    """Absolute mean of signed quarterly Pearson IC for 3-D GP fitness.

    This deliberately uses factor values and forward returns as they are:
    neither side is ranked and returns are not residualised.
    """
    eval_mask, restrict, ntr_ic_x, index_ret_arr, exret, y_residual = fitness_cache
    y, y_pred = _mask_eval_data(y_raw, y_pred_raw, sample_weight, eval_mask, restrict)
    # The denominator depends only on available return labels, never the formula.
    min_quarters = int(np.ceil(np.isfinite(y).any(axis=1).sum() * 2 / 3))
    paired = np.isfinite(y) & np.isfinite(y_pred)
    good = paired.sum(axis=1) >= 100
    for q in np.flatnonzero(good):
        take = paired[q]
        good[q] = np.ptp(y[q, take]) > 0 and np.ptp(y_pred[q, take]) > 0
    y = np.where(paired & good[:, None], y, np.nan)
    y_pred = np.where(paired & good[:, None], y_pred, np.nan)
    total_ic, ic_series = _cal_total_IC(y, y_pred, method="pearson")
    if min_quarters > 0 and np.isfinite(ic_series).sum() >= min_quarters and np.isfinite(total_ic):
        return abs(total_ic), ic_series
    return -1., ic_series


def _weighted_spearman(y_raw, y_pred_raw, sample_weight, fitness_cache):
    """Calculate the weighted Pearson correlation coefficient."""
    # y: array - like, shape = [n_samples] -> [n_dates, n_stocks]
    eval_mask, restrict, ntr_ic_x, index_ret_arr,exret,y_redisual = fitness_cache
    y, y_pred = _mask_eval_data(y_raw, y_pred_raw, sample_weight, eval_mask, restrict)
    y_pred = pd.DataFrame(y_pred).rank(axis=1,pct=True).values
    y = pd.DataFrame(y).rank(axis=1,pct=True).values
    total_IC, _ = _cal_total_IC(y, y_pred, method="pearson")
    if np.isfinite(total_IC):
        return np.abs(total_IC)
    return 0.


def _alert_weighted_spearman(y_raw, y_pred_raw, sample_weight, fitness_cache):
    """Calculate the weighted Pearson correlation coefficient."""
    eval_mask, restrict, ntr_ic_x, index_ret_arr,exret,y_residual = fitness_cache
    y, y_pred = _mask_eval_data(y_raw, y_pred_raw, sample_weight, eval_mask, restrict)
    y_pred = pd.DataFrame(y_pred).rank(axis=1,pct=True).values
    y = pd.DataFrame(y).rank(axis=1,pct=True).values
    total_IC, _ = _cal_total_IC(y, y_pred, method="pearson")
    if np.isfinite(total_IC):
        return total_IC
    return -1


def _weighted_neu_spearman(y_raw, y_pred_raw, sample_weight, fitness_cache):
    eval_mask, restrict, ntr_ic_x, index_ret_arr, exret, y_residual = fitness_cache
    y, y_pred = _mask_eval_data(y_raw, y_pred_raw, sample_weight, eval_mask, restrict)

    y_residual = pd.DataFrame(y_residual).rank(axis=1, pct=True).values
    y_pred_rank = pd.DataFrame(y_pred).rank(axis=1, pct=True).values

    total_IC, ic_series = _cal_total_IC(y_residual, y_pred_rank, method="pearson")

    if np.isfinite(total_IC):
        return np.abs(total_IC), ic_series

    return 0., ic_series


#中性化多头-only IC
def _weighted_long_only_neu_spearman(y_raw, y_pred_raw, sample_weight, fitness_cache):
    eval_mask, restrict, ntr_ic_x, index_ret_arr, exret, y_residual = fitness_cache
    y, y_pred = _mask_eval_data(y_raw, y_pred_raw, sample_weight, eval_mask, restrict)
    y_pred_rank = pd.DataFrame(y_pred).rank(pct=True, axis=1)
    ret = pd.DataFrame(y_residual,index=y_pred_rank.index,columns=y_pred_rank.columns)
    ret_rank = ret.rank(axis=1,pct=True)
    top_half_f_rank = y_pred_rank[y_pred_rank>0.5].copy()
    bot_half_f_rank = y_pred_rank[y_pred_rank<0.5].copy()
    top_half_ret_rank = ret[top_half_f_rank.notnull()].rank(axis=1,pct=True)
    bot_half_ret_rank = ret[bot_half_f_rank.notnull()].rank(axis=1,pct=True)
    _,_,top_ics = cal_ic_icir(top_half_f_rank,top_half_ret_rank,method="pearson")
    _,_,bot_ics = cal_ic_icir(bot_half_f_rank,bot_half_ret_rank,method="pearson")
    mean_ic,_,_ = cal_ic_icir(ret_rank,y_pred_rank,method="pearson")
    if mean_ic>0:
        total_IC = top_ics.mean()
    else:
        total_IC = bot_ics.mean()
    if np.isfinite(total_IC):
        return np.abs(total_IC)
    return 0


def _cal_hedge_sharpe(y, y_pred, restrict,num_groups=10):
    restrict_df = pd.DataFrame(restrict)
    y_pred = pd.DataFrame(y_pred)
    y = pd.DataFrame(y)
    y_pred[(restrict_df == 1) | restrict_df.isna()] = np.nan
    percentile = y_pred.rank(axis=1, method='average', pct=True)
    group_number = num_groups - ((1 - percentile) * num_groups) // 1

    percentile.loc[percentile.count(1) <= 10] = np.nan

    daily_group_number = group_number.fillna(-9999)

    all_i = [1, num_groups]

    ret_list = []
    for i in all_i:
        this_group_weight = pd.DataFrame(0, index=restrict_df.index, columns=restrict_df.columns)
        this_group_weight[daily_group_number == i] = 1
        this_group_weight = (this_group_weight.T / this_group_weight.sum(1)).T

        use_weight = this_group_weight
        this_ret = (use_weight * y).sum(axis=1, min_count=1)

        this_df = pd.concat([this_ret], axis=1)
        this_df.columns = [['ret']]
        ret_list.append(this_df)
    [ret1,ret10] = ret_list
    if ret1["ret"].sum().item() > ret10["ret"].sum().item():
        hedge_ret = ret1["ret"]
    else:
        hedge_ret = ret10["ret"]
    hedge_annualized_sharpe_ratio = hedge_ret.mean() / hedge_ret.std() * (points_of_year**0.5)
    return hedge_annualized_sharpe_ratio.item()

def _weighted_hedge_sharpe(y_raw, y_pred_raw, sample_weight, fitness_cache):
    eval_mask, restrict, ntr_ic_x, index_ret_arr,exret,y_redisual = fitness_cache
    y, y_pred = _mask_eval_data(y_raw, y_pred_raw, sample_weight, eval_mask, restrict)
    hedge_annualized_sharpe_ratio = _cal_hedge_sharpe(exret, y_pred, restrict)
    if np.isfinite(hedge_annualized_sharpe_ratio):
        return hedge_annualized_sharpe_ratio
    return -100

def _weighted_neu_hedge_sharpe(y_raw, y_pred_raw, sample_weight, fitness_cache):
    eval_mask, restrict, ntr_ic_x, index_ret_arr,exret,y_residual = fitness_cache
    y, y_pred = _mask_eval_data(y_raw, y_pred_raw, sample_weight, eval_mask, restrict)
    # y_residual = cal_residual(y, ntr_ic_x)
    hedge_annualized_sharpe_ratio = _cal_hedge_sharpe(y_residual, y_pred, restrict)
    if np.isfinite(hedge_annualized_sharpe_ratio):
        return hedge_annualized_sharpe_ratio
    return -100


def _weighted_IR(y_raw, y_pred_raw, sample_weight, fitness_cache):
    eval_mask, restrict, ntr_ic_x, index_ret_arr,exret,y_residual = fitness_cache
    y, y_pred = _mask_eval_data(y_raw, y_pred_raw, sample_weight, eval_mask, restrict)
    with np.errstate(divide='ignore', invalid='ignore'):
        factor_df = pd.DataFrame(y_pred).rank(axis=1,pct=True)
        ret_df = pd.DataFrame(y).rank(axis=1,pct=True)
        _, IR, _ = cal_ic_icir(factor_df, ret_df, method="spearman")
        if np.isfinite(IR):
            return IR
        return -100

def _alert_weighted_IR(y, y_pred, sample_weight, fitness_cache):
    return _weighted_IR(y, y_pred, sample_weight, fitness_cache)  


weighted_pearson = _Fitness(function=_weighted_pearson, greater_is_better=True)
alert_weighted_pearson = _Fitness(function=_alert_weighted_pearson, greater_is_better=True)

weighted_spearman = _Fitness(function=_weighted_spearman, greater_is_better=True)
alert_weighted_spearman = _Fitness(function=_alert_weighted_spearman, greater_is_better=True)

weighted_information_ratio = _Fitness(function=_weighted_IR, greater_is_better=True)
alert_weighted_information_ratio = _Fitness(function=_alert_weighted_IR, greater_is_better=True)

weighted_neutralized_spearman = _Fitness(function=_weighted_neu_spearman, greater_is_better=True)
weighted_long_only_neutralized_spearman = _Fitness(function=_weighted_long_only_neu_spearman, greater_is_better=True)

weighted_hedged_sharpe = _Fitness(function=_weighted_hedge_sharpe, greater_is_better=True)
weighted_neutralized_hedged_sharpe = _Fitness(function=_weighted_neu_hedge_sharpe, greater_is_better=True)
raw_pearson_ic = _Fitness(function=_raw_pearson_ic, greater_is_better=True)

fitness_map = {}
fitness_map_3d = {
    # "pearson": weighted_pearson,
    # Absolute mean of signed quarterly Pearson IC; retain signed ic_series.
    # No ranking or return residualisation.
    "IC": raw_pearson_ic,
    # "ICIR": weighted_information_ratio,
    "neu_IC": weighted_neutralized_spearman,
    "hedge_sharpe": weighted_hedged_sharpe,
    "neu_hedge_sharpe": weighted_neutralized_hedged_sharpe,
    "long_only_neu_IC": weighted_long_only_neutralized_spearman,
    "alert_IC": alert_weighted_spearman,
    "alert_pearson": alert_weighted_pearson,
    "alert_ICIR": alert_weighted_information_ratio
}

fitness_map = {**fitness_map, **fitness_map_3d}
