import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import rankdata


def mean_normalization_df(df):

    std = df.std(axis=1)
    re = df.sub(df.mean(axis=1), axis=0)
    re = re.div(std, axis=0)
    return re


def rolling_weighted_sum(a, b):
    b_sum = np.nansum(b)
    if b_sum == 0:
        b_sum = 1
    b_new = b/ b_sum
    res = np.nansum(a * b_new, axis=0)
    return res


def fraction_of_positive(a_raw):
    a = a_raw.copy()
    a[a <= 0] = np.nan
    a[a > 0] = 1.
    return np.nansum(a, axis=0) / a.shape[0]


def winsorize(arr, q_thres=0.975):
    max_thres = np.quantile(arr[~np.isnan(arr)], q_thres)
    min_thres = np.quantile(arr[~np.isnan(arr)], 1 - q_thres)
    new_data = arr.copy()
    new_data[new_data>max_thres] = max_thres
    new_data[new_data<min_thres] = min_thres
    return new_data


def calc_pearson_correlation(x, y):

    if len(x) < 30:
        return np.nan
    expt = np.nanmean(x*y) - np.nanmean(x) * np.nanmean(y)
    return expt / np.nanstd(x) / np.nanstd(y)


def max_drawdown_cal(cum_ret):
    max_ret = 0
    max_drawdown = 0
    for i in range(len(cum_ret)):
        if cum_ret[i] > max_ret:
            max_ret = cum_ret[i]
        if max_drawdown < max_ret - cum_ret[i]:
            max_drawdown = max_ret - cum_ret[i]
    return max_drawdown


def rankdata_nonmiss(arr):
   
    if len(arr) == 0:
        return np.array([])
    
    rank_arr = arr.copy()
    rank_arr[:] = np.nan
    cond = ~np.isnan(arr) & ~np.isinf(arr)
    rank_arr[cond] = rankdata(arr[cond])
    return rank_arr


def _pinv_extended(x, rcond=1e-15):
    """
    Return the pinv of an array X as well as the singular values
    used in computation.

    Code adapted from numpy.
    """
    x = np.asarray(x)
    x = x.conjugate()
    u, s, vt = np.linalg.svd(x, False)
    m = u.shape[0]
    n = vt.shape[1]
    cutoff = rcond * np.maximum.reduce(s)
    for i in range(min(n, m)):
        if s[i] > cutoff:
            s[i] = 1./s[i]
        else:
            s[i] = 0.
    res = np.dot(np.transpose(vt), np.multiply(s[:, np.core.newaxis],
                                               np.transpose(u)))
    return res

def _cal_beta(X, y):
    pinv_wexog = _pinv_extended(X)
    beta = np.dot(pinv_wexog, y)
    return beta


def cal_residual(y, X):
    """
    y.shape = (n_dates, n_stocks)
    X.shape = (n_dates, n_stocks, n_features) 
    """
    n_dates, n_stocks = y.shape
    residual_arr = np.full(y.shape, np.nan)
    for t in range(n_dates):
        y_t = y[t, :]
        X_t = X[t, :, :]
        mask = (~np.isnan(y_t)) & (~np.isnan(X_t).any(axis=1)) 
        if mask.sum()<2: continue
        # select qualified samples to regresss
        y_arr = y_t[mask]
        X_arr = X_t[mask, :]
        try:
            x_add_constant = sm.add_constant(X_arr)
            # beta=(X_T @ X)' @ X_T @ Y
            beta = _cal_beta(x_add_constant, y_arr)
            # y = X @ beta
            y_pred = x_add_constant @ beta 
            res_t = y_arr - y_pred
            residual_arr[t, mask] = res_t
        except:
            pass
    return residual_arr


def cal_ic_icir(factor_df_raw, ret_df_raw, method="spearman", min_obs=2):
    valid_mask = ((~factor_df_raw.isna()) & (~ret_df_raw.isna())).sum(axis=1) >= min_obs

    ic_series = factor_df_raw.corrwith(ret_df_raw, axis=1, method=method)

    ic_series[~valid_mask] = np.nan

    ic_mean = ic_series.mean()
    ic_std = ic_series.std()
    icir = ic_mean / ic_std * (len(ic_series.dropna()) ** 0.5) if ic_std > 0 else np.nan

    return ic_mean, icir, ic_series



def cal_port_ret_arr(y_true, y_pred,restrict_arr,portf_num):

    factor_arr = y_pred[1:, :].copy()
    y = y_true[1:, :].copy()
    restrict = restrict_arr[1:,:].copy()

    n_dates, n_stocks = factor_arr.shape
    unique_values = np.unique(factor_arr)
    if len(unique_values) < portf_num:
        return -1  

    portf_ret_list = []
    for t in range(n_dates):
        factor_t_raw = factor_arr[t,:]
        ret_t_raw = y[t, :]
        restrict_t = restrict[t,:]

        b1 = ~np.isnan(factor_t_raw)
        b2 = ~np.isnan(ret_t_raw)
        b3 = restrict_t == 0
        ret_t = ret_t_raw[b1&b2&b3]
        factor_t = factor_t_raw[b1&b2&b3]

        if len(factor_t) < portf_num:
            portf_ret_list.append([0]*portf_num)
            continue

        portf_ret_t = []
        for k in range(portf_num):

            if len(unique_values) < portf_num:
                condition = (factor_t==unique_values[k])
            else:

                rank_factor_t = rankdata_nonmiss(factor_t) / sum(~np.isnan(factor_t))

                if k ==0 :
                    condition = (rank_factor_t >= k/portf_num) & (rank_factor_t <= (k+1)/portf_num)
                else:
                    condition = (rank_factor_t > k/portf_num) & (rank_factor_t <= (k+1)/portf_num)

            portf_ret_t.append(np.nanmean(ret_t[condition]))
        portf_ret_list.append(portf_ret_t)

    portf_ret_arr = np.array(portf_ret_list)
    portf_ret_arr[np.isnan(portf_ret_arr)] = 0  
    return portf_ret_arr

def cal_port_rets(y,y_pred,restrict,portf_num):
    ret = pd.DataFrame(y[1:].copy())
    factor_value = pd.DataFrame(y_pred[1:].copy())
    agg_restrict = pd.DataFrame(restrict[1:].copy())
    factor_value[(agg_restrict == 1) | agg_restrict.isna() | ret.isna()] = np.nan
    percentile = factor_value.rank(axis=1, method='average', pct=True)
    group_number = portf_num - ((1 - percentile) * portf_num) // 1

    unique_values = np.unique(factor_value.values)
    if len(unique_values) < portf_num:
        return -1  

    percentile.loc[percentile.count(1) < portf_num] = np.nan

    daily_group_number = group_number.fillna(-1)

    all_i = [i for i in range(1,portf_num+1)]
    output_list = []
    for i in all_i:
        this_group_weight = pd.DataFrame(0, index=agg_restrict.index, columns=agg_restrict.columns)
        this_group_weight[daily_group_number == i] = 1
        this_group_weight = (this_group_weight.T / this_group_weight.sum(1)).T

        use_weight = this_group_weight
        use_exret = ret
        this_ret = (use_weight * use_exret).sum(axis=1, min_count=1)
       
        output_list.append(this_ret)

    perf_df = pd.concat(output_list, axis=1)
    return perf_df.values

def mask_long_only(y, y_pred,restrict):
    y_pred_rank = pd.DataFrame(y_pred).rank(pct=True, axis=1)
    ret = pd.DataFrame(y,index=y_pred_rank.index,columns=y_pred_rank.columns)
    ret_rank = ret.rank(axis=1,pct=True)
    y_masked = y_pred.copy()
    y_masked[restrict!=0] = np.nan
   
    mean_ic,_,_ = cal_ic_icir(ret_rank,y_pred_rank,method="pearson")
    if mean_ic > 0:
        y_masked[y_pred_rank<0.5] = np.nan
    else:
        y_masked[y_pred_rank>0.5] = np.nan
        y_masked *= -1
    return y_masked


def cal_two_ic(k, signal_arr_temp, ret_resid_rank_df, ret_resid_df, eval_mask, restrict, factor_rate):
    
    factor_array_masked = signal_arr_temp.T # (N,T) -> (T,N)
    factor_array_masked = factor_array_masked[np.where(eval_mask == 1)].copy()
    factor_array_masked[restrict != 0] = np.nan
    factor_coverage_arr = (~np.isnan(factor_array_masked)).sum(
        axis=1) / (restrict == 0).sum(axis=1)
    factor_array_masked[np.where(factor_coverage_arr < factor_rate)] = np.nan
    signal_arr_temp = factor_array_masked.T # (T,N) -> (N,T)

    f_rank = pd.DataFrame(signal_arr_temp).rank(axis=0,pct=True)
    ic_array_one = f_rank.corrwith(ret_resid_rank_df, axis=0, method="pearson")
    ic_mean = ic_array_one.mean()

    top_f_rank = f_rank[f_rank > 0.5].copy()
    top_half_ret_rank = ret_resid_df[top_f_rank.notnull()].rank(axis=0, pct=True)
    top_ic = top_f_rank.corrwith(top_half_ret_rank, axis=0, method="pearson")

    bot_f_rank = f_rank[f_rank < 0.5].copy()
    bot_half_ret_rank = ret_resid_df[bot_f_rank.notnull()].rank(axis=0, pct=True)
    bot_ic = bot_f_rank.corrwith(bot_half_ret_rank, axis=0, method="pearson")

    if ic_mean > 0:
        longonly_ic_array_one = top_ic.values
    else:
        longonly_ic_array_one = - bot_ic.values

    return ic_array_one, longonly_ic_array_one

def cal_port_cumrets(y,y_pred,restrict,portf_num=10,hori=5):
    ret = y
    factor_value = y_pred
    agg_restrict = restrict
    
    factor_value.loc[factor_value.std(1) == 0] = np.nan
    factor_value[(agg_restrict == 1) | agg_restrict.isna() | ret.isna()] = np.nan

    percentile = factor_value.rank(axis=1, method='average', pct=True)
    
    percentile.loc[percentile.count(1) < portf_num] = np.nan
    group_number = portf_num - ((1 - percentile) * portf_num) // 1

    daily_group_number = group_number.fillna(-9999).shift(1)

    assert hori == 5
    weekly_group_number = daily_group_number.resample('W').first().resample('D') \
        .first().shift(-6).ffill().reindex(daily_group_number.index)


    all_i = [i for i in range(1,portf_num+1)]
    output_list = []
    for i in all_i:
        this_group_weight = pd.DataFrame(0, index=agg_restrict.index, columns=agg_restrict.columns)
        this_group_weight[weekly_group_number == i] = 1
        this_group_weight = (this_group_weight.T / this_group_weight.sum(1)).T

        use_weight = this_group_weight
        use_exret = ret
        this_ret = (use_weight * use_exret).sum(axis=1, min_count=1)
        this_cumret = this_ret.expanding().sum()
        
        this_df = pd.concat([this_cumret], axis=1)
        this_df.columns = [f'group{i}_' + s for s in ['cumret']]
       
        output_list.append(this_df)
    
    return output_list