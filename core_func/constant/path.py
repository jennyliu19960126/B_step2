import os

# Self-contained backtest data bundle.  Set GP_BACKTEST_ROOT to switch to a
# compatible copy without touching code.
# Prefer the original research-data root.  Portable deployments may still
# override it with GP_BACKTEST_ROOT without changing source code.
backtest_root = os.environ.get("GP_BACKTEST_ROOT", "/data/research")
basic_info_dir = os.path.join(backtest_root, "basic_info")

# The bundle does not ship a separate calendar/symbol map: both are derived
# from the date and Wind-code axes of vars/Base/S_RESTRICT.csv by tradedate.py.
calendar_path = os.path.join(basic_info_dir,"calendar.csv")
symbol_map_path = os.path.join(basic_info_dir,"symbol_map.csv")

# output root
output_root = os.environ.get("GP_OUTPUT_ROOT", "/data/fund_agent/B_step2/runs/output_genetic")

#research_data, vars for backtest, csv for Jason (sorted symbol, xxxxxx.SH)
daily_data_root = os.path.join(backtest_root, "vars")
dailyprice_root = os.path.join(daily_data_root, "DailyPrice")
dailyderived_root = os.path.join(daily_data_root, 'DailyDerived')
base_root = os.path.join(daily_data_root, "Base")
index_root = os.path.join(daily_data_root, "index_data")
szba_root = os.path.join(daily_data_root, "SzBa")
#daily_feature_root = "/data/research/gp_base_var/"
financial_variable_root = os.path.join(output_root, "financial_variables")

daily_data_root_dict = {
    'DailyPrice': dailyprice_root,
    'DailyDerived': dailyderived_root,
    'Base': base_root,
    "index_data": index_root,
    "FinancialFeature": financial_variable_root,
    "SzBa": szba_root
}
