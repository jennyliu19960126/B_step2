import sys
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from constant.params import factor_start_str, factor_end_str, DATES_quar, STOCKS
from data_reader.data_reader_csv import load_csv_daily

class HelperData:

    def __init__(self):
        super(HelperData, self).__init__()
        self.factor_start_str = factor_start_str
        self.factor_end_str = factor_end_str
        self.prep_cache_data_function()
    
    def prep_cache_data_function(self):
        ind_df_raw = load_csv_daily('Base','SW_IND_CODE', self.factor_start_str, self.factor_end_str)
        ind_df_raw.index = pd.to_datetime(ind_df_raw.index)
        # Fundamental X is quarterly.  Carry the last available industry code
        # to each report period, then retain exactly X's stock axis/order.
        quarterly_index = pd.to_datetime(DATES_quar)
        ind_df = ind_df_raw.reindex(index=quarterly_index, columns=STOCKS, method='ffill')
        ind_df = (ind_df.fillna(0) // 10000).astype(int)
        self.ind_df = pd.DataFrame(ind_df.values)
        return

try:
    helper_data = HelperData()
    ind_df = helper_data.ind_df
    print("function data loaded")
except Exception as e:
    print(e)
