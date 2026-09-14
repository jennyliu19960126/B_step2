import pandas as pd
from constant.params import gen_metric_list, fitness_metric, cs_functions, ts_func_kw, feature_names

def summary_by_generation(result_df,  raw_result_df, greater_is_better=True):
    result_df['length'] = result_df['formulation_stack'].apply(lambda x: len(eval(x)))
    gen_list = sorted(set(result_df['generation'].values))
    gen_summary = pd.DataFrame(index=gen_list)

    # for counting the number of occurrences of features
    feature0 = {}
    for i in feature_names:
        feature0[i] = 0

    # for counting the number of occurrences of functions
    ts_old, cs_old = {}, {}
    for i in cs_functions:
        cs_old[i] = 0
    for i in ts_func_kw:
        ts_old[i] = 0

    for gen in gen_list:
        gen_df = result_df[result_df['generation']==gen]
        raw_gen_df = raw_result_df[raw_result_df['generation']==gen]
        gen_summary.loc[gen, "average_length"] = gen_df['length'].mean()
        for m in gen_metric_list:
            gen_summary.loc[gen, f"average_{m}"] = gen_df[m].mean()
        if greater_is_better:
            best_idx = gen_df[fitness_metric].idxmax()
        else:
            best_idx = gen_df[fitness_metric].idxmin()
        gen_summary.loc[gen, 'best_length'] = gen_df.loc[best_idx, "length"]
        for m in gen_metric_list:
            gen_summary.loc[gen, f"best_{m}"] = gen_df.loc[best_idx, m]
        # gen_summary.loc[gen, 'n_sample'] = len(gen_df)
        gen_summary.loc[gen, 'n_sample'] = len(raw_gen_df) 
        gen_summary.loc[gen, 'n_unique_formula'] = len(set(gen_df['formulation'].values))
        gen_summary.loc[gen, 'n_unique_fitness'] = len(set(gen_df[fitness_metric].values))
        _n_cs = 0
        _n_ts = 0
        for formu in gen_df['formulation_stack'].values:
            for _func in eval(formu):
                if not isinstance(_func, str):
                    continue
                if _func.startswith("cs_"):
                    _n_cs += 1
                    if _func in cs_functions:
                        cs_old[_func] += 1
                elif _func.startswith("ts_"):
                    _n_ts += 1
                    _func_name = _func.split('_', 1)[1].rsplit('_', 1)[0]
                    if _func_name in ts_func_kw:
                        ts_old[_func.split('_', 1)[1].rsplit('_', 1)[0]] += 1

        gen_summary.loc[gen, 'n_cs_func'] = _n_cs
        gen_summary.loc[gen, 'n_ts_func'] = _n_ts

        _n_feature_batch0 = 0
        for features in gen_df['feature_stack'].values:
            for _feature in eval(features):
                if _feature in feature_names:
                    _n_feature_batch0 += 1
                    feature0[_feature] += 1

        gen_summary.loc[gen, 'n_feature_batch0'] = _n_feature_batch0

    # total
    if greater_is_better:
        best_idx = result_df[fitness_metric].idxmax()
    else:
        best_idx = result_df[fitness_metric].idxmin()
    gen_summary.loc['total', 'average_length'] = result_df['length'].mean()
    gen_summary.loc['total', f'best_length'] = result_df.loc[best_idx, 'length']
    for m in gen_metric_list:
        gen_summary.loc['total', f'average_{m}'] = result_df[m].mean()
        gen_summary.loc['total', f'best_{m}'] = result_df.loc[best_idx, m]
    # gen_summary.loc['total', 'n_sample'] = len(result_df)
    gen_summary.loc['total', 'n_sample'] = len(raw_result_df)  
    gen_summary.loc['total', 'n_unique_formula'] = len(set(result_df['formulation'].values))
    gen_summary.loc['total', 'n_unique_fitness'] = len(set(result_df[fitness_metric].values))
    gen_summary.loc['total', 'n_cs_func'] = gen_summary.loc[gen_list, "n_cs_func"].sum()
    gen_summary.loc['total', 'n_ts_func'] = gen_summary.loc[gen_list, "n_ts_func"].sum()
    gen_summary.loc['total', 'n_feature_batch0'] = gen_summary.loc[gen_list, "n_feature_batch0"].sum()
    gen_summary = gen_summary.reset_index().rename(columns={'index':'generation'})

    feature_all = {**feature0}
    feature_summary = pd.Series(feature_all)
    feature_summary_rate = feature_summary / feature_summary.sum()
    feature_summary_df = pd.DataFrame({"count": feature_summary, "rate": feature_summary_rate})
    function_all = {**cs_old, **ts_old}
    function_summary = pd.Series(function_all)
    function_summary_rate = function_summary / function_summary.sum()
    function_summary_df = pd.DataFrame({"count": function_summary, "rate": function_summary_rate})
    return gen_summary, feature_summary_df, function_summary_df
