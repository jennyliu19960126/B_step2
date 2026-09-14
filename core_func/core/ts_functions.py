import warnings
import pandas as pd
import numpy as np
import bottleneck as bd
from functools import reduce

from .functions import _Function, _protected_division, _condition_and, _condition_or
from .gp_utils import np_shift, not_const


ts_func_info_dict = {
    "_ts_cov": {
        "arity": 2},
    "_ts_corr": {
        "arity": 2},         
    "_ts_count_and": {
        "arity": 3},       
    "_ts_count_or": {
        "arity": 3},          
    
    "_ts_cut_up_q1": {"arity": 2},
    "_ts_cut_down_q1": {"arity": 2},
    
    "_ts_roll_resid": {
        "arity": 2},
    "_ts_signmul": {
        "arity": 2},
    "_ts_rolling_ic": {
        "arity": 2},
}


def rolling_min(x, window, min_count=1):
    return bd.move_min(x, window=window, min_count=min_count, axis=0)


def rolling_max(x, window, min_count=1):
    return bd.move_max(x, window=window, min_count=min_count, axis=0)


def rolling_mean(x, window, min_count=1):
    result = bd.move_mean(x, window=window, min_count=1, axis=0)
    return result

def rolling_median(x, window, min_count=1):
    result = bd.move_median(x, window=window, min_count=1, axis=0)
    return result


def rolling_delay(x, window):
    return np_shift(x, window)


def rolling_delta(x, window):
    x_shift = np_shift(x, window)
    result = x - x_shift
    return result


def rolling_product(x, window):
    n_dates, n_stocks = x.shape
    prod_arr = np.full((n_dates, n_stocks), np.nan)
    for t in range(window, n_dates+1):
        prod_arr[t-1, :] = reduce(lambda x, y: x*y, x[t-window:t, :])
    result = np.clip(prod_arr, -10000, 10000)
    return result


def rolling_sum(x, window, min_count=1):
    return bd.move_sum(x, window=window, min_count=min_count, axis=0)


def rolling_pct(x, window):
    x_shift = np_shift(x, window)
    x_shift[x_shift==0] = np.nan
    result = (x - x_shift) / np.abs(x_shift) 
    return result


def rolling_rank(x, window, min_periods=2):
    x_rank = bd.move_rank(x, window=window, min_count=min_periods, axis=0)
    x_rank_adj = (x_rank+1)/2
    return x_rank_adj


def rolling_std(x, window, min_periods=2):
    result = bd.move_std(x, window, axis=0, min_count=min_periods, ddof=1)
    result[np.isinf(result)] = np.nan
    return result


def rolling_cov(x1_raw, x2_raw, window, min_periods=2):
    mask = ~np.isnan(x1_raw+x2_raw)
    x1 = x1_raw.copy()
    x1[~mask] = np.nan
    x2 = x2_raw.copy()
    x2[~mask] = np.nan
    mean_xy = bd.move_mean(x1*x2, window=window, min_count=min_periods, axis=0)
    mean_x = bd.move_mean(x1, window=window, min_count=min_periods, axis=0)
    mean_y = bd.move_mean(x2, window=window, min_count=min_periods, axis=0)
    cnt = bd.move_sum(~np.isnan(x1+x2), window=window, min_count=2, axis=0)
    cnt[cnt<=min_periods] = np.nan
    cov = mean_xy - mean_x*mean_y
    result = cov*cnt/(cnt-1)
    return result


def rolling_corr(x1_raw, x2_raw, window, min_periods=2):
    mask = ~np.isnan(x1_raw+x2_raw)
    x1 = x1_raw.copy()
    x1[~mask] = np.nan
    x2 = x2_raw.copy()
    x2[~mask] = np.nan
    mean_x1_x2 = bd.move_mean(x1*x2, window=window, min_count=min_periods, axis=0)
    mean_x1 = bd.move_mean(x1, window=window, min_count=min_periods, axis=0)
    mean_x2 = bd.move_mean(x2, window=window, min_count=min_periods, axis=0)
    cnt = bd.move_sum(mask, window=window, min_count=2, axis=0)
    cnt[cnt<min_periods] = np.nan
    x1_var = bd.move_var(x1, window=window, min_count=min_periods, ddof=1, axis=0)
    x2_var = bd.move_var(x2, window=window, min_count=min_periods, ddof=1, axis=0)
    nominator = (mean_x1_x2 - mean_x1*mean_x2) *cnt/(cnt-1)
    denominator = np.sqrt(x1_var*x2_var)
    result = _protected_division(nominator, denominator, atol=1e-6)
    result[np.isinf(result)] = np.nan
    result = result*not_const(x1, window=window, min_periods=min_periods)*not_const(x2, window=window, min_periods=min_periods)  
    return result


def rolling_skew(x, window, min_periods=2):
    result = pd.DataFrame(x).rolling(window, min_periods=min_periods).skew().values
    result = result*not_const(x, window=window, min_periods=min_periods)
    return result


def rolling_kurt(x, window, min_periods=2):
    result = pd.DataFrame(x).rolling(window, min_periods=min_periods).kurt().values
    result = result*not_const(x, window=window, min_periods=min_periods)
    return result


def rolling_autocorr(x1_raw, window, lag=1, min_periods=2):
    x1 = x1_raw.copy()
    x2 = np_shift(x1, lag)
    result = rolling_corr(x1, x2, window, min_periods=min_periods)
    return result


def rolling_autoslope(x, window, min_periods=2):
    n_dates, n_stocks = x.shape
    mask = ~np.isnan(x)
    y = np.repeat(np.array(list(range(n_dates))), n_stocks).reshape(n_dates, n_stocks).astype(np.float64)
    y[~mask] = np.nan
    cnt = bd.move_sum(mask, window, min_count=min_periods, axis=0) 
    x_sum = bd.move_sum(x, window, min_count=min_periods, axis=0)
    y_sum = bd.move_sum(y, window, min_count=min_periods, axis=0)
    x_sqr_sum = bd.move_sum(x**2, window, min_count=min_periods, axis=0)
    xy_sum = bd.move_sum(x*y, window, min_count=min_periods, axis=0)
    nominator = xy_sum - x_sum*y_sum/cnt
    denominator = x_sqr_sum - (x_sum**2)/cnt
    result = _protected_division(nominator, denominator, np.nan, atol=1e-4)
    result = np_shift(result, 1) * not_const(x, window=window, min_periods=min_periods)
    result[np.isinf(result)] = np.nan
    return result


def rolling_argmax(x1_raw, window, min_periods=1):
    mask = ~np.isnan(x1_raw)
    x1 = x1_raw.copy()
    x1[~mask] = np.nan
    x1_reverse = x1[::-1]
    # add window rows of NaN
    x1_reverse = np.concatenate((x1_reverse, np.full((window-1, x1.shape[1]), np.nan)), axis=0)
    result_r = bd.move_argmax(x1_reverse, window=window, min_count=min_periods, axis=0)
    result = result_r[::-1][:-(window-1)]
    # first window rows should be modified
    window_len = np.repeat(np.array(range(window)), repeats=x1.shape[1], axis=0).reshape((window, x1.shape[1])) + 1
    result[:window] = result[:window] + window_len - window
    return result


def rolling_argmin(x1_raw, window, min_periods=1):
    mask = ~np.isnan(x1_raw)
    x1 = x1_raw.copy()
    x1[~mask] = np.nan
    x1_reverse = x1[::-1]
    # add window rows of NaN
    x1_reverse = np.concatenate((x1_reverse, np.full((window-1, x1.shape[1]), np.nan)), axis=0)
    result_r = bd.move_argmin(x1_reverse, window=window, min_count=min_periods, axis=0)
    result = result_r[::-1][:-(window-1)]
    # first window rows should be modified
    window_len = np.repeat(np.array(range(window)), repeats=x1.shape[1], axis=0).reshape((window, x1.shape[1])) + 1
    result[:window] = result[:window] + window_len - window
    return result


def rolling_decay_linear(x, window):
    min_periods = max(2, int(2/3*window))
    n_dates, n_stocks = x.shape
    mask = ~np.isnan(x)
    # _x is independent variable, x is dependent variable, SLR: x ~ a + b * _x
    _x = np.repeat(np.array(list(range(n_dates))), n_stocks).reshape(n_dates, n_stocks).astype(np.float64)
    _x[~mask] = np.nan
    cnt = bd.move_sum(mask, window, min_count=min_periods, axis=0) 
    x_sum = bd.move_sum(_x, window, min_count=min_periods, axis=0)
    y_sum = bd.move_sum(x, window, min_count=min_periods, axis=0)
    x_sqr_sum = bd.move_sum(_x**2, window, min_count=min_periods, axis=0)
    xy_sum = bd.move_sum(x*_x, window, min_count=min_periods, axis=0)
    nominator = xy_sum - x_sum*y_sum/cnt
    denominator = x_sqr_sum - (x_sum**2)/cnt
    b = _protected_division(nominator, denominator, np.nan)
    x_mean = bd.move_mean(_x, window, min_count=min_periods, axis=0)
    y_mean = bd.move_mean(x, window, min_count=min_periods, axis=0)
    a = y_mean - b * x_mean
    result = a + b * (np.repeat(np.array(list(range(n_dates))), n_stocks).reshape(n_dates, n_stocks).astype(np.float64)+1)
    result[np.isinf(result)] = np.nan
    return result


def rolling_cut_up(x,y,q,window,min_periods=2):

    x=np.asarray(x,dtype=float)
    y=np.asarray(y,dtype=float)

    n_dates,n_stocks=x.shape
    result=np.full_like(x,np.nan)


    joint=np.isfinite(x)&np.isfinite(y)

    cnt=bd.move_sum(
        joint,
        window=window,
        min_count=min_periods,
        axis=0
    )


    y_q=pd.DataFrame(y).rolling(
        window=window,
        min_periods=min_periods
    ).quantile(
        q/100,
        interpolation="higher"
    ).to_numpy()


    for t in range(min_periods,n_dates):

        sub_x=x[t-window:t]
        sub_y=y[t-window:t]

        threshold=y_q[t-1]


        selected=(
            sub_y>threshold.reshape(1,-1)
        )

        selected &= np.isfinite(sub_x)


        num=np.where(
            selected,
            sub_x,
            0.0
        ).sum(axis=0)


        den=selected.sum(axis=0)


        usable=(
            (cnt[t-1]>min_periods)
            &
            (den>0)
        )


        result[t,usable]=(
            num[usable]/den[usable]
        )


    result[~np.isfinite(result)]=np.nan

    return result


def rolling_cut_down(x,y,q,window,min_periods=2):

    x=np.asarray(x,dtype=float)
    y=np.asarray(y,dtype=float)

    n_dates,n_stocks=x.shape
    result=np.full_like(x,np.nan)

    valid_xy=np.isfinite(x)&np.isfinite(y)

    cnt=bd.move_sum(
        valid_xy,
        window,
        min_count=min_periods,
        axis=0
    )

    y_q=pd.DataFrame(y).rolling(
        window,
        min_periods=min_periods
    ).quantile(
        q/100,
        interpolation="lower"
    ).to_numpy()


    for t in range(min_periods,n_dates):

        sub_x=x[max(0,t-window):t]
        sub_y=y[max(0,t-window):t]

        selected=(
            sub_y <
            y_q[t-1][None,:]
        )

        selected_valid=(
            selected&
            np.isfinite(sub_x)
        )

        cnt_x=selected_valid.sum(axis=0)

        sum_x=np.where(
            selected_valid,
            sub_x,
            0.0
        ).sum(axis=0)

        ok=(
            (cnt[t-1]>min_periods)
            &
            (cnt_x>0)
        )

        result[t,ok]=(
            sum_x[ok]/
            cnt_x[ok]
        )

    result[~np.isfinite(result)]=np.nan

    return result



def rolling_count_and(x, y, z, window, min_periods=1):
    cond_x = _condition_and(x, y, z)
    cond_x[~np.isnan(cond_x)] = 1
    result = bd.move_sum(cond_x, window=window, min_count=min_periods, axis=0)
    return result


def rolling_count_or(x, y, z, window, min_periods=1):
    cond_x = _condition_or(x, y, z)
    cond_x[~np.isnan(cond_x)] = 1
    result = bd.move_sum(cond_x, window=window, min_count=min_periods, axis=0)
    return result

def rolling_grstable(x, window, min_count=1):
   
    x_shift = np_shift(x, 1)
    qoq = (x - x_shift) / np.abs(x_shift)
    qoq[0, :] = np.nan 

    min_periods = max(min_count, int(np.floor(window * 0.75)))
    qoq_mean = bd.move_mean(qoq, window=window, min_count=min_periods, axis=0)
    qoq_std = bd.move_std(qoq, window=window, min_count=min_periods, axis=0)

    result = qoq_mean / qoq_std
    result[np.isinf(result)] = np.nan
    return result


def rolling_zscore(x, window, min_count=1):

    min_periods = max(min_count, int(np.floor(window * 0.75)))

    x_mean = bd.move_mean(x, window=window, min_count=min_periods, axis=0)
    x_std = bd.move_std(x, window=window, min_count=min_periods, axis=0)

    result = _protected_division((x - x_mean), x_std, np.nan, atol=1e-6)
    result[np.isinf(result)] = np.nan

    return result

def rolling_demean(x, window, min_count=1):
    
    min_periods = max(min_count, int(np.floor(window * 0.75)))

    x_mean = bd.move_mean(x, window=window, min_count=min_periods, axis=0)
    
    result = x - x_mean
    result[np.isinf(result)] = np.nan
    return result


def rolling_median_abs_deviation(x, window, min_count=1):
    
    min_periods = max(min_count, int(np.floor(window * 0.75)))
    n_dates, n_stocks = x.shape
    result = np.full((n_dates, n_stocks), np.nan)

    for t in range(window - 1, n_dates):
        start = t - window + 1
        sub_x = x[start:t + 1, :]

        valid_cnt = (~np.isnan(sub_x)).sum(axis=0)
        ok = valid_cnt >= min_periods

        x_mean = np.nanmean(sub_x, axis=0)
        abs_dev = np.abs(sub_x - x_mean)

        result[t, ok] = np.nanmedian(abs_dev[:, ok], axis=0)

    result[np.isinf(result)] = np.nan
    return result


def rolling_rms(x, window, min_count=1):
    
    min_periods = max(min_count, int(np.floor(window * 0.75)))

    result = np.sqrt(
        bd.move_mean(x ** 2, window=window, min_count=min_periods, axis=0)
    )

    result[np.isinf(result)] = np.nan
    return result


def rolling_norm_mean(x, window, min_count=1):
    
    min_periods = max(min_count, int(np.floor(window * 0.75)))

    x_mean = bd.move_mean(x, window=window, min_count=min_periods, axis=0)
    x_rms = np.sqrt(
        bd.move_mean(x ** 2, window=window, min_count=min_periods, axis=0)
    )

    result = _protected_division(x_mean, x_rms, np.nan, atol=1e-10)
    result[np.isinf(result)] = np.nan
    return result


def rolling_norm_max(x, window, min_count=1):
    
    min_periods = max(min_count, int(np.floor(window * 0.75)))

    x_max = bd.move_max(x, window=window, min_count=min_periods, axis=0)
    x_rms = np.sqrt(
        bd.move_mean(x ** 2, window=window, min_count=min_periods, axis=0)
    )

    result = _protected_division(x_max, x_rms, np.nan, atol=1e-10)
    result[np.isinf(result)] = np.nan
    return result


def rolling_norm_min(x, window, min_count=1):
    
    min_periods = max(min_count, int(np.floor(window * 0.75)))

    x_min = bd.move_min(x, window=window, min_count=min_periods, axis=0)
    x_rms = np.sqrt(
        bd.move_mean(x ** 2, window=window, min_count=min_periods, axis=0)
    )

    result = _protected_division(x_min, x_rms, np.nan, atol=1e-10)
    result[np.isinf(result)] = np.nan
    return result


def rolling_norm_min_max(x, window, min_count=1):
    
    min_periods = max(min_count, int(np.floor(window * 0.75)))

    x_max = bd.move_max(x, window=window, min_count=min_periods, axis=0)
    x_min = bd.move_min(x, window=window, min_count=min_periods, axis=0)
    x_rms = np.sqrt(
        bd.move_mean(x ** 2, window=window, min_count=min_periods, axis=0)
    )

    result = _protected_division(x_max - x_min, x_rms, np.nan, atol=1e-10)
    result[np.isinf(result)] = np.nan
    return result


def rolling_ratio_beyond_sigma(x, window, r=2, min_count=1):
    
    min_periods = max(min_count, int(np.floor(window * 0.75)))
    n_dates, n_stocks = x.shape
    result = np.full((n_dates, n_stocks), np.nan)

    for t in range(window - 1, n_dates):
        start = t - window + 1
        sub_x = x[start:t + 1, :]

        valid_cnt = (~np.isnan(sub_x)).sum(axis=0)
        ok = valid_cnt >= min_periods

        x_mean = np.nanmean(sub_x, axis=0)
        x_std = np.nanstd(sub_x, axis=0, ddof=1)

        cond = np.abs(sub_x - x_mean) > r * x_std
        ratio = np.nanmean(cond.astype(float), axis=0)

        ratio[x_std <= 1e-10] = np.nan
        result[t, ok] = ratio[ok]

    result[np.isinf(result)] = np.nan
    return result



def rolling_index_mass_quantile(x,window,q=0.5,min_count=1):
    x=np.asarray(x,float)
    n_dates,n_stocks=x.shape
    min_periods=max(min_count,int(np.floor(window*0.75)))
    result=np.full_like(x,np.nan)

    for t in range(window-1,n_dates):
        sub_x=x[t-window+1:t+1]
        valid=np.isfinite(sub_x)
        valid_cnt=valid.sum(axis=0)

        mass=np.where(valid,np.abs(sub_x),0.0)
        total_mass=mass.sum(axis=0)
        cum_mass=np.cumsum(mass,axis=0)

        ok=(valid_cnt>=min_periods)&(total_mass>1e-10)
        reached=cum_mass>=q*total_mass[None,:]
        first_index=np.argmax(reached,axis=0)

        result[t,ok]=first_index[ok]

    result[~np.isfinite(result)]=np.nan
    return result



def rolling_number_cross_mean(x,window,min_count=1):

    x=np.asarray(x,dtype=np.float64)

    n_dates,n_stocks=x.shape
    min_periods=max(min_count,int(np.floor(window*0.75)))

    result=np.full_like(x,np.nan)


    for t in range(window-1,n_dates):

        sub_x=x[t-window+1:t+1]

        valid=np.isfinite(sub_x)
        valid_cnt=valid.sum(axis=0)

        ok=valid_cnt>=max(min_periods,2)

        total=np.where(valid,sub_x,0).sum(axis=0)

        mean_x=np.divide(
            total,
            valid_cnt,
            out=np.full(n_stocks,np.nan),
            where=valid_cnt>0
        )

        signs=np.sign(
            sub_x-mean_x.reshape(1,-1)
        )


        cross_cnt=np.zeros(n_stocks)
        last_sign=np.zeros(n_stocks)
        has_last=np.zeros(n_stocks,dtype=bool)


        for k in range(window):

            cur_valid=valid[k]
            cur_sign=signs[k]

            compare=cur_valid&has_last

            cross_cnt[compare]+=(
                cur_sign[compare]*last_sign[compare]<0
            )

            last_sign[cur_valid]=cur_sign[cur_valid]
            has_last[cur_valid]=True


        result[t,ok]=cross_cnt[ok]


    result[~np.isfinite(result)]=np.nan

    return result



def rolling_time_asymmetry_stats(x, window, min_count=1):
    window=max(window,3)
    
    min_periods = max(min_count, int(np.floor(window * 0.75)))
    n_dates, n_stocks = x.shape
    result = np.full((n_dates, n_stocks), np.nan)

    for t in range(window - 1, n_dates):
        start = t - window + 1
        sub_x = x[start:t + 1, :]

        valid_cnt = (~np.isnan(sub_x)).sum(axis=0)
        ok = valid_cnt >= min_periods

        if window < 3:
            continue

        temp = (sub_x[2:, :] - sub_x[1:-1, :]) ** 2 * (sub_x[1:-1, :] - sub_x[:-2, :])
        result[t, ok] = np.nanmean(temp[:, ok], axis=0)

    result[np.isinf(result)] = np.nan
    return result


def rolling_longest_strike_above_mean(x,window,min_count=1):

    x=np.asarray(x,dtype=np.float64)

    n_dates,n_stocks=x.shape

    min_p=max(min_count,int(np.floor(window*0.75)))

    result=np.full_like(x,np.nan)


    for t in range(window-1,n_dates):

        sub=x[t-window+1:t+1]

        valid=np.isfinite(sub)

        cnt=valid.sum(0)

        ok=cnt>=min_p


        total=np.where(valid,sub,0).sum(0)

        mean=np.divide(
            total,
            cnt,
            out=np.full(n_stocks,np.nan),
            where=cnt>0
        )


        above=valid&(sub>mean.reshape(1,-1))


        cur=np.zeros(n_stocks,dtype=np.int16)
        mx=np.zeros(n_stocks,dtype=np.int16)


        for k in range(window):

            cur=np.where(above[k],cur+1,0)

            mx=np.maximum(mx,cur)


        result[t,ok]=mx[ok]


    result[~np.isfinite(result)]=np.nan

    return result


def rolling_longest_strike_below_mean(x,window,min_count=1):

    x=np.asarray(x,dtype=np.float64)

    n_dates,n_stocks=x.shape

    min_p=max(min_count,int(np.floor(window*0.75)))

    result=np.full_like(x,np.nan)


    for t in range(window-1,n_dates):

        sub=x[t-window+1:t+1]

        valid=np.isfinite(sub)

        cnt=valid.sum(0)

        ok=cnt>=min_p


        total=np.where(valid,sub,0).sum(0)

        mean=np.divide(
            total,
            cnt,
            out=np.full(n_stocks,np.nan),
            where=cnt>0
        )


        below=valid&(sub<mean.reshape(1,-1))


        cur=np.zeros(n_stocks,dtype=np.int16)

        mx=np.zeros(n_stocks,dtype=np.int16)


        for k in range(window):

            cur=np.where(below[k],cur+1,0)

            mx=np.maximum(mx,cur)


        result[t,ok]=mx[ok]


    result[~np.isfinite(result)]=np.nan

    return result


def rolling_mean_over_1norm(x, window, min_count=1):
    
    min_periods = max(min_count, int(np.floor(window * 0.75)))

    x_mean = bd.move_mean(x, window=window, min_count=min_periods, axis=0)
    x_abs_mean = bd.move_mean(np.abs(x), window=window, min_count=min_periods, axis=0)

    result = _protected_division(x_mean, x_abs_mean, np.nan, atol=1e-10)
    result[np.isinf(result)] = np.nan
    return result


def rolling_quantile(x, window, q=0.25, min_count=1):
    
    window=max(window,3)
    
    min_periods = max(min_count, int(np.floor(window * 0.75)))

    result = pd.DataFrame(x).rolling(
        window=window,
        min_periods=min_periods
    ).quantile(q).values

    result[np.isinf(result)] = np.nan
    return result


def rolling_decay_exp(x, window, min_count=1):
    
    window=max(window,3)
    
    min_periods = max(min_count, int(np.floor(window * 0.75)))
    n_dates, n_stocks = x.shape
    result = np.full((n_dates, n_stocks), np.nan)

    weight = np.exp(np.linspace(-1, 0, window))
    weight = weight / weight.sum()
    weight = weight.reshape(-1, 1)

    for t in range(window - 1, n_dates):
        start = t - window + 1
        sub_x = x[start:t + 1, :]

        mask = ~np.isnan(sub_x)
        valid_cnt = mask.sum(axis=0)

        weight_adj = weight * mask
        weight_sum = weight_adj.sum(axis=0)

        ok = (valid_cnt >= min_periods) & (weight_sum > 1e-10)

        temp = np.nansum(sub_x * weight_adj, axis=0)
        result[t, ok] = temp[ok] / weight_sum[ok]

    result[np.isinf(result)] = np.nan
    return result


def rolling_fw_fracdiff(x, window, diff_order=0.5, min_count=1):
    
    window=max(window,3)
    
    min_periods = max(min_count, int(np.floor(window * 0.75)))
    n_dates, n_stocks = x.shape
    result = np.full((n_dates, n_stocks), np.nan)

    weight = [1.0]
    for k in range(1, window):
        weight.append(-weight[-1] * (diff_order - k + 1) / k)

    weight = np.array(weight, dtype=float)
    weight = weight[::-1].reshape(-1, 1)

    for t in range(window - 1, n_dates):
        start = t - window + 1
        sub_x = x[start:t + 1, :]

        valid_cnt = (~np.isnan(sub_x)).sum(axis=0)
        ok = valid_cnt >= min_periods

        temp = np.nansum(sub_x * weight, axis=0)
        result[t, ok] = temp[ok]

    result[np.isinf(result)] = np.nan
    return result


def rolling_rolling_ic(x, y, window, min_count=1):
    min_periods = max(min_count, int(np.floor(window * 0.75)))

    x_var = bd.move_var(x, window=window, min_count=min_periods, ddof=1, axis=0)
    xy_cov = rolling_cov(x, y, window=window, min_periods=min_periods)

    beta = _protected_division(xy_cov, x_var, np.nan, atol=1e-10)
    shift_beta = np_shift(beta, 1)

    result = x * shift_beta
    result[np.isinf(result)] = np.nan
    return result


def rolling_pctchg_lms(x, window, min_count=1):
    
    window=max(window,4)
    
    d_short = max(1, int(np.floor(window / 2)))
    d_long = window

    pct_long = rolling_pct(x, window=d_long)
    pct_short = rolling_pct(x, window=d_short)

    result = pct_long - pct_short
    result[np.isinf(result)] = np.nan
    return result


def rolling_stddev_lms(x, window, min_count=1):
    
    window=max(window,4)
    
    min_periods_short = max(min_count, int(np.floor(max(2, window / 2) * 0.75)))
    min_periods_long = max(min_count, int(np.floor(window * 0.75)))

    d_short = max(2, int(np.floor(window / 2)))
    d_long = window

    std_long = rolling_std(x, window=d_long, min_periods=min_periods_long)
    std_short = rolling_std(x, window=d_short, min_periods=min_periods_short)

    result = std_long - std_short
    result[np.isinf(result)] = np.nan
    return result


def rolling_sharp(x, window, min_count=1):
    
    min_periods = max(min_count, int(np.floor(window * 0.75)))

    x_mean = bd.move_mean(x, window=window, min_count=min_periods, axis=0)
    x_std = bd.move_std(x, window=window, min_count=min_periods, axis=0)

    result = _protected_division(x_mean, x_std, np.nan, atol=1e-10)
    result[np.isinf(result)] = np.nan
    return result


def rolling_sharp_lms(x, window, min_count=1):
    
    window=max(window,4)
    
    d_short = max(2, int(np.floor(window / 2)))
    d_long = window

    sharp_long = rolling_sharp(x, window=d_long, min_count=min_count)
    sharp_short = rolling_sharp(x, window=d_short, min_count=min_count)

    result = sharp_long - sharp_short
    result[np.isinf(result)] = np.nan
    return result


def rolling_skew_lms(x, window, min_count=1):
    
    # skew至少需要3个有效点
    window = max(window, 6)
    
    d_short = max(3, int(np.floor(window / 2)))
    d_long = window

    min_periods_short = max(3, int(np.floor(d_short * 0.75)))
    min_periods_long = max(3, int(np.floor(d_long * 0.75)))

    skew_long = rolling_skew(x, window=d_long, min_periods=min_periods_long)
    skew_short = rolling_skew(x, window=d_short, min_periods=min_periods_short)

    result = skew_long - skew_short
    result[np.isinf(result)] = np.nan
    return result


def rolling_kurtosis_lms(x, window, min_count=1):

    window = max(window, 8)
    
    d_short = max(4, int(np.floor(window / 2)))
    d_long = window

    min_periods_short = max(4, int(np.floor(d_short * 0.75)))
    min_periods_long = max(4, int(np.floor(d_long * 0.75)))

    kurt_long = rolling_kurt(x, window=d_long, min_periods=min_periods_long)
    kurt_short = rolling_kurt(x, window=d_short, min_periods=min_periods_short)

    result = kurt_long - kurt_short
    result[np.isinf(result)] = np.nan
    return result


def rolling_decay_linear_pctchg(x, window, min_count=1):
    
    window=max(window,4)
    
    d_short = max(2, int(np.floor(window / 2)))
    d_long = window

    decay_long = rolling_decay_linear(x, window=d_long)
    decay_short = rolling_decay_linear(x, window=d_short)

    result = _protected_division(decay_long, decay_short, np.nan, atol=1e-10) - 1
    result[np.isinf(result)] = np.nan
    return result


def rolling_smooth_pos_gp(x, window, min_count=1):
    
    window=max(window,4)
    
    min_periods_short = max(min_count, int(np.floor(max(2, window / 2) * 0.75)))
    min_periods_long = max(min_count, int(np.floor(window * 0.75)))

    d_short = max(2, int(np.floor(window / 2)))
    d_long = window

    x_mean = bd.move_mean(x, window=d_short, min_count=min_periods_short, axis=0)
    x_max = bd.move_max(x, window=d_long, min_count=min_periods_long, axis=0)
    x_min = bd.move_min(x, window=d_long, min_count=min_periods_long, axis=0)

    result = _protected_division(x_mean - x_min, x_max - x_min, np.nan, atol=1e-10)
    result[np.isinf(result)] = np.nan
    return result


def rolling_count_over_cs_mean(x, window, min_count=1):

    min_periods = max(min_count, int(np.floor(window * 0.75)))

    cs_mean = np.nanmean(x, axis=1).reshape(-1, 1)
    cs_std = np.nanstd(x, axis=1).reshape(-1, 1)

    cs_zscore = _protected_division(x - cs_mean, cs_std, np.nan, atol=1e-10)

    cond = (cs_zscore > 0).astype(float)
    cond[np.isnan(cs_zscore)] = np.nan

    result = bd.move_sum(cond, window=window, min_count=min_periods, axis=0)
    result[np.isinf(result)] = np.nan
    return result


def rolling_rsi(x, window, min_count=1):
    
    min_periods = max(min_count, int(np.floor(window * 0.75)))

    x_diff = x - np_shift(x, 1)

    gain = np.where(x_diff > 0, x_diff, 0.0)
    loss = np.where(x_diff < 0, -x_diff, 0.0)

    gain[np.isnan(x_diff)] = np.nan
    loss[np.isnan(x_diff)] = np.nan

    avg_gain = bd.move_mean(gain, window=window, min_count=min_periods, axis=0)
    avg_loss = bd.move_mean(loss, window=window, min_count=min_periods, axis=0)

    rs = _protected_division(avg_gain, avg_loss, np.nan, atol=1e-10)
    result = 100 - 100 / (1 + rs)

    result[np.isinf(result)] = np.nan
    return result


def rolling_rsi_of_pctchg(x, window, min_count=1):
    
    pct = rolling_pct(x, window=1)
    result = rolling_rsi(pct, window=window, min_count=min_count)

    result[np.isinf(result)] = np.nan
    return result


def rolling_signmul(x, y, window, min_count=1):
    
    min_periods = max(min_count, int(np.floor(window * 0.75)))

    x_mean = bd.move_mean(x, window=window, min_count=min_periods, axis=0)
    result = np.sign(x - x_mean) * y

    result[np.isnan(x) | np.isnan(y) | np.isnan(x_mean)] = np.nan
    result[np.isinf(result)] = np.nan
    return result


def rolling_last(x, window):
    """
    获取 window 期前的字段值。
    如果 window=1，就是上一期。
    """
    return np_shift(x, window)


def rolling_avg(x, window):
    """
    当前期与 window 期前字段值的均值。
    """
    x_shift = np_shift(x, window)
    result = (x + x_shift) / 2
    result[np.isinf(result)] = np.nan
    return result


def rolling_qoq(x, window):
    """
    环比/相对变化率：
    (x_t - x_{t-window}) / abs(x_{t-window})
    """
    x_shift = np_shift(x, window)

    with np.errstate(divide='ignore', invalid='ignore'):
        result = _protected_division(x - x_shift, np.abs(x_shift), np.nan, atol=1e-10)
        result[np.isnan(x) | np.isnan(x_shift)] = np.nan
        result[np.isinf(result)] = np.nan
        return result


def rolling_diff(x, window):
    """
    差分：
    x_t - x_{t-window}
    """
    x_shift = np_shift(x, window)
    result = x - x_shift
    result[np.isinf(result)] = np.nan
    return result


def rolling_sue_score(x, window, intercept=False, min_periods=None):
    """
    向量化滚动SUE。
    """
    x = np.asarray(x, dtype=np.float64)
    n_dates, _ = x.shape
    result = np.full_like(x, np.nan, dtype=np.float64)

    if min_periods is None:
        min_periods = max(3, int(np.floor(window * 0.75)))

    if min_periods > window:
        return result

    y = x
    z = np_shift(x, 1)

    valid = np.isfinite(y) & np.isfinite(z)
    valid_float = valid.astype(np.float64)

    y0 = np.where(valid, y, 0.0)
    z0 = np.where(valid, z, 0.0)

    cnt = bd.move_sum(
        valid_float, window=window, min_count=1, axis=0
    )
    sum_y = bd.move_sum(
        y0, window=window, min_count=1, axis=0
    )
    sum_z = bd.move_sum(
        z0, window=window, min_count=1, axis=0
    )
    sum_y2 = bd.move_sum(
        y0 * y0, window=window, min_count=1, axis=0
    )
    sum_z2 = bd.move_sum(
        z0 * z0, window=window, min_count=1, axis=0
    )
    sum_zy = bd.move_sum(
        z0 * y0, window=window, min_count=1, axis=0
    )

    cnt = np_shift(cnt, 1)
    sum_y = np_shift(sum_y, 1)
    sum_z = np_shift(sum_z, 1)
    sum_y2 = np_shift(sum_y2, 1)
    sum_z2 = np_shift(sum_z2, 1)
    sum_zy = np_shift(sum_zy, 1)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):

        if intercept:
            # 含截距OLS
            sxx = sum_z2 - sum_z * sum_z / cnt
            sxy = sum_zy - sum_z * sum_y / cnt
            syy = sum_y2 - sum_y * sum_y / cnt

            beta = np.full_like(sxy, np.nan)
            valid_sxx = np.isfinite(sxx) & (np.abs(sxx) > 1e-10)
            beta[valid_sxx] = sxy[valid_sxx] / sxx[valid_sxx]

            alpha = (sum_y - beta * sum_z) / cnt
            pred = alpha + beta * z

            # OLS含截距残差平方和
            centered_sse = syy - beta * sxy

        else:
            # 无截距OLS
            beta = np.full_like(sum_zy, np.nan)
            valid_denom = np.isfinite(sum_z2) & (np.abs(sum_z2) > 1e-10)
            beta[valid_denom] = (
                sum_zy[valid_denom] / sum_z2[valid_denom]
            )

            pred = beta * z

            # 原始残差平方和：Σ(y-beta*z)^2
            raw_sse = (
                sum_y2
                - 2.0 * beta * sum_zy
                + beta * beta * sum_z2
            )

            sum_err = sum_y - beta * sum_z
            centered_sse = raw_sse - sum_err * sum_err / cnt

        centered_sse = np.maximum(centered_sse, 0.0)

        err_var = centered_sse / (cnt - 1)
        err_std = np.sqrt(err_var)

    ok = (
        (cnt >= min_periods)
        & np.isfinite(y)
        & np.isfinite(z)
        & np.isfinite(pred)
        & np.isfinite(err_std)
        & (err_std > 1e-10)
    )

    result[ok] = (y[ok] - pred[ok]) / err_std[ok]

    result[:window + 1, :] = np.nan
    result[~np.isfinite(result)] = np.nan

    return result


def rolling_first_slope(x, window, min_periods=None, block_size=512, atol=1e-10):

    x = np.asarray(x, dtype=np.float64)
    n_dates, n_stocks = x.shape
    result = np.full_like(x, np.nan)

    if min_periods is None:
        min_periods = max(3, int(np.floor(window * 0.75)))

    if min_periods > window:
        return result

    time_index = np.arange(n_dates, dtype=np.float64).reshape(-1,1)

    for left in range(0, n_stocks, block_size):
        right = min(left + block_size, n_stocks)
        xb = x[:, left:right]

        valid = np.isfinite(xb)
        y0 = np.where(valid, xb, 0.0)
        t0 = np.where(valid, time_index, 0.0)
        v = valid.astype(float)

        cnt = bd.move_sum(v, window=window, min_count=1, axis=0)
        sum_y = bd.move_sum(y0, window=window, min_count=1, axis=0)
        sum_t = bd.move_sum(t0, window=window, min_count=1, axis=0)
        sum_t2 = bd.move_sum(t0*t0, window=window, min_count=1, axis=0)
        sum_ty = bd.move_sum(t0*y0, window=window, min_count=1, axis=0)

        numerator = sum_ty - sum_t * sum_y / cnt
        denominator = sum_t2 - sum_t * sum_t / cnt

        beta = _protected_division(
            numerator, denominator, np.nan, atol=atol
        )

        beta[cnt < min_periods] = np.nan
        beta[~np.isfinite(beta)] = np.nan

        if window > 1:
            beta[:window-1, :] = np.nan

        result[:, left:right] = beta

    return result



def rolling_second_slope(
        x,
        window,
        min_periods=None,
        block_size=512):


    x=np.asarray(
        x,
        dtype=np.float64
    )


    n_dates,n_stocks=x.shape


    result=np.full_like(
        x,
        np.nan
    )


    if min_periods is None:
        min_periods=max(
            5,
            int(np.floor(window*0.75))
        )

    tt=np.arange(
        window,
        dtype=float
    )


    A=np.column_stack([
        np.ones(window),
        tt,
        tt**2
    ])


    pinv=np.linalg.pinv(A)


    for left in range(
        0,
        n_stocks,
        block_size
    ):

        right=min(
            left+block_size,
            n_stocks
        )


        xb=x[
            :,
            left:right
        ]


        for t in range(
            window-1,
            n_dates
        ):


            y=xb[
                t-window+1:t+1,
                :
            ]


            valid=np.isfinite(y)


            cnt=valid.sum(axis=0)


            ok=cnt>=min_periods


            if not np.any(ok):
                continue


            cols=np.where(ok)[0]


            for c in cols:

                yc=y[:,c]

                m=np.isfinite(yc)


                if m.all():

                    coef=pinv @ yc


                else:

                    A_m=A[m]

                    coef=np.linalg.lstsq(
                        A_m,
                        yc[m],
                        rcond=None
                    )[0]


                result[
                    t,
                    left+c
                ]=coef[2]


    result[
        ~np.isfinite(result)
    ]=np.nan


    return result




def rolling_roll_resid(y,x,window,min_periods=None,block_size=512,atol=1e-10):

    y=np.asarray(y,dtype=np.float64)
    x=np.asarray(x,dtype=np.float64)

    n_dates,n_stocks=y.shape
    result=np.full_like(y,np.nan)

    if min_periods is None:
        min_periods=max(3,int(np.floor(window*0.75)))

    if min_periods>window:
        return result

    for left in range(0,n_stocks,block_size):

        right=min(left+block_size,n_stocks)

        yb,xb=y[:,left:right],x[:,left:right]

        valid=np.isfinite(yb)&np.isfinite(xb)

        y0=np.where(valid,yb,0.0)
        x0=np.where(valid,xb,0.0)
        v=valid.astype(float)

        cnt=bd.move_sum(v,window=window,min_count=1,axis=0)
        sum_y=bd.move_sum(y0,window=window,min_count=1,axis=0)
        sum_x=bd.move_sum(x0,window=window,min_count=1,axis=0)
        sum_x2=bd.move_sum(x0*x0,window=window,min_count=1,axis=0)
        sum_xy=bd.move_sum(x0*y0,window=window,min_count=1,axis=0)

        with np.errstate(divide="ignore",invalid="ignore"):
            mean_x=sum_x/cnt
            mean_y=sum_y/cnt
            sxx=sum_x2-sum_x*sum_x/cnt
            sxy=sum_xy-sum_x*sum_y/cnt

            beta=_protected_division(sxy,sxx,np.nan,atol=atol)
            alpha=mean_y-beta*mean_x
            resid=yb-alpha-beta*xb

        usable=(cnt>=min_periods)&valid&np.isfinite(resid)

        resid[~usable]=np.nan

        if window>1:
            resid[:window-1,:]=np.nan

        result[:,left:right]=resid

    result[~np.isfinite(result)]=np.nan
    return result



def _ts_min(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_min(x1, window=d+1)


def _ts_max(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_max(x1, window=d+1)


def _ts_mean(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_mean(x1, window=d+1)

def _ts_delay(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_delay(x1, window=d)


def _ts_delta(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_delta(x1, window=d)


def _ts_product(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_product(x1, window=d+1)


def _ts_ttm(x1, d):
    """Trailing-period sum; with d=4 this is quarterly TTM."""
    with np.errstate(over='ignore', under='ignore'):
        return rolling_sum(x1, window=d, min_count=1)


def _ts_pct(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_pct(x1, window=d)


def _ts_growth(x1, d):
    """Financial-statement alias for period-over-period growth."""
    return _ts_pct(x1, d)


def _ts_rank(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_rank(x1, window=d+1)


def _ts_std(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_std(x1, window=d+2)


def _ts_cov(x1, x2, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_cov(x1, x2, window=d+2)


def _ts_corr(x1, x2, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_corr(x1, x2, window=d+2)


def _ts_skew(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
        return rolling_skew(x1, window=d+2)


def _ts_kurt(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
        return rolling_kurt(x1, window=d+2)


def _ts_autocorr_lag1(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_autocorr(x1, window=d+2, lag=1)


def _ts_autocorr_lag2(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_autocorr(x1, window=d+2, lag=2)


def _ts_autocorr_lag3(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_autocorr(x1, window=d+2, lag=3)


def _ts_autocorr_lag4(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_autocorr(x1, window=d+2, lag=4)


def _ts_autocorr_lag5(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_autocorr(x1, window=d+2, lag=5)


def _ts_autoslope(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_autoslope(x1, window=d+2)


def _ts_argmax(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_argmax(x1, window=d+1)    


def _ts_argmin(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_argmin(x1, window=d+1)   


def _ts_decay_linear(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_decay_linear(x1, window=d+1)   


def _ts_cut_up_q1(x, y, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_cut_up(x, y, q=10, window=d+1)   


def _ts_cut_down_q1(x, y, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_cut_down(x, y, q=10, window=d+1)   


def _ts_count_and(x1, y1, z1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_count_and(x1, y1, z1, window=d)   


def _ts_count_or(x1, y1, z1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_count_or(x1, y1, z1, window=d)   
    
    

def _ts_median(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_median(x1, window=d+1)

def _ts_grstable(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_grstable(x1, window=d+1)


def _ts_zscore(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_zscore(x1, window=d+1)


def _ts_demean(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_demean(x1, window=d+1)


def _ts_median_abs_deviation(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_median_abs_deviation(x1, window=d+1)


def _ts_rms(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_rms(x1, window=d+1)


def _ts_norm_mean(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_norm_mean(x1, window=d+1)


def _ts_norm_max(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_norm_max(x1, window=d+1)


def _ts_norm_min(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_norm_min(x1, window=d+1)


def _ts_norm_min_max(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_norm_min_max(x1, window=d+1)


def _ts_ratio_beyond_sigma_2(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_ratio_beyond_sigma(x1, window=d+1, r=2)


def _ts_ratio_beyond_sigma_3(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_ratio_beyond_sigma(x1, window=d+1, r=3)


def _ts_index_mass_quantile_q25(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_index_mass_quantile(x1, window=d+1, q=0.25)


# def _ts_index_mass_quantile_q50(x1, d):
#     with np.errstate(over='ignore', under='ignore'):
#         return rolling_index_mass_quantile(x1, window=d+1, q=0.50)


# def _ts_index_mass_quantile_q75(x1, d):
#     with np.errstate(over='ignore', under='ignore'):
#         return rolling_index_mass_quantile(x1, window=d+1, q=0.75)


def _ts_number_cross_mean(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_number_cross_mean(x1, window=d+1)


def _ts_time_asymmetry_stats(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_time_asymmetry_stats(x1, window=d+1)


def _ts_longest_strike_above_mean(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_longest_strike_above_mean(x1, window=d+1)


def _ts_longest_strike_below_mean(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_longest_strike_below_mean(x1, window=d+1)


def _ts_mean_over_1norm(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_mean_over_1norm(x1, window=d+1)


def _ts_quantile_q25(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_quantile(x1, window=d+1, q=0.25)


# def _ts_quantile_q75(x1, d):
#     with np.errstate(over='ignore', under='ignore'):
#         return rolling_quantile(x1, window=d+1, q=0.75)


def _ts_decay_exp(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_decay_exp(x1, window=d+1)


def _ts_fw_fracdiff(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_fw_fracdiff(x1, window=d+1, diff_order=0.5)


def _ts_rolling_ic(x1, x2, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_rolling_ic(x1, x2, window=d+1)


def _ts_pctchg_lms(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_pctchg_lms(x1, window=d+1)


def _ts_stddev_lms(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_stddev_lms(x1, window=d+2)


def _ts_sharp(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_sharp(x1, window=d+2)


def _ts_sharp_lms(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_sharp_lms(x1, window=d+2)


def _ts_skew_lms(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
        return rolling_skew_lms(x1, window=d+2)


def _ts_kurtosis_lms(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
        return rolling_kurtosis_lms(x1, window=d+2)


def _ts_decay_linear_pctchg(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_decay_linear_pctchg(x1, window=d+1)


def _ts_smooth_pos_gp(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_smooth_pos_gp(x1, window=d+1)


def _ts_count_over_cs_mean(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_count_over_cs_mean(x1, window=d+1)


def _ts_rsi(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_rsi(x1, window=d+1)


def _ts_rsi_of_pctchg(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_rsi_of_pctchg(x1, window=d+1)


def _ts_signmul(x1, x2, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_signmul(x1, x2, window=d+1)


def _ts_last(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_last(x1, window=d)


def _ts_avg(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_avg(x1, window=d)


def _ts_qoq(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_qoq(x1, window=d)


def _ts_yoy(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_qoq(x1, window=d)


def _ts_diff1q(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_diff(x1, window=d)


def _ts_diff4q(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_diff(x1, window=d)


def _ts_sue0_score(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_sue_score(x1, window=d, intercept=False)


def _ts_sue1_score(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_sue_score(x1, window=d, intercept=True)


def _ts_first_slope(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_first_slope(x1, window=d)


def _ts_second_slope(x1, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_second_slope(x1, window=d)


def _ts_roll_resid(y, x, d):
    with np.errstate(over='ignore', under='ignore'):
        return rolling_roll_resid(y, x, window=d)


ts_func_kw_list = ["min", "max", "mean", "median", "grstable", "zscore", "demean", "delay", "delta", "product", "ttm", "pct", "growth", "rank", "std", "cov", "corr", "skew", "kurt",
                   "autocorr_lag1", "autocorr_lag2", "autocorr_lag3", "autocorr_lag4", "autocorr_lag5", "autoslope", 
                   "argmin", "argmax", "decay_linear", "cut_up_q1", "cut_down_q1", "count_and", "count_or",
                   # 新增 TS 算子
                   "last","avg","qoq","yoy","diff1q","diff4q","sue0_score","sue1_score","first_slope","second_slope", "roll_resid",
                   # 新增 TS 统计类算子
                   "median_abs_deviation", "rms", "norm_mean", "norm_max", "norm_min", "norm_min_max",
                   "ratio_beyond_sigma_2", "ratio_beyond_sigma_3",
                   "index_mass_quantile_q25",
                   "number_cross_mean", "time_asymmetry_stats",
                   "longest_strike_above_mean", "longest_strike_below_mean",
                   "mean_over_1norm", "quantile_q25", 
                   "decay_exp", "fw_fracdiff", "rolling_ic",
                   "pctchg_lms", "stddev_lms", "sharp", "sharp_lms", "skew_lms", "kurtosis_lms",
                   "decay_linear_pctchg", "smooth_pos_gp", "count_over_cs_mean",
                   "rsi", "rsi_of_pctchg", "signmul",
                   ]


_ts_dynamic_function_map = {}
for func_kw in ts_func_kw_list:
    raw_func_name = f"_ts_{func_kw}"
    func_name = f"dynamic_ts_{func_kw}"
    if raw_func_name in ts_func_info_dict.keys(): 
        arity = ts_func_info_dict[raw_func_name]['arity']
    else:
        arity = 1
    locals()[func_name] = _Function(function=locals()[raw_func_name], name=func_name, arity=arity, isRandom=(True, (1, 12)))
    _ts_dynamic_function_map[func_name] = locals()[func_name]        


fix_ts_func_win_list = [1, 2, 4, 6, 8, 12]

_ts_fix_function_map = {}
for func_kw in ts_func_kw_list:
    raw_func_name = f"_ts_{func_kw}"
    if raw_func_name in ts_func_info_dict.keys(): 
        arity = ts_func_info_dict[raw_func_name]['arity']
    else:
        arity = 1
    for i in fix_ts_func_win_list:
        func_name = f"ts_{func_kw}_{i}"
        locals()[func_name] = _Function(function=locals()[raw_func_name], name=func_name, arity=arity, isParam=(True, i))
        _ts_fix_function_map[func_name] = locals()[func_name]

_extra_function_map = {**_ts_dynamic_function_map, **_ts_fix_function_map}
