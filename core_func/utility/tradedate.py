import os
import sys
import pandas as pd
from pathlib import Path
from functools import partial

sys.path.append(str(Path(__file__).resolve().parent.parent))
from constant.path import calendar_path, daily_data_root_dict

def get_calendar():
    # Legacy deployments may have calendar.csv.  The portable bundle instead
    # derives the trading calendar from the canonical restriction matrix.
    if os.path.exists(calendar_path):
        return pd.read_csv(calendar_path, index_col=0, parse_dates=True)
    restrict_path = os.path.join(daily_data_root_dict["Base"], "S_RESTRICT.csv")
    columns = pd.read_csv(restrict_path, nrows=0, index_col=0).columns
    dates = pd.to_datetime(columns, errors="coerce")
    dates = pd.DatetimeIndex(dates[dates.notna()]).sort_values().unique()
    return pd.DataFrame(index=dates)

calendar = get_calendar()


def get_calendar_quarterly():
    return get_calendar().resample("Q").last()

calendar_quarterly = get_calendar_quarterly()


def _offset_trading_day(start_day, offset, calendar):
    assert offset != 0
    if start_day in calendar.index:
        return calendar.index[calendar.index.get_loc(start_day)+offset]
    else:
        union = list(sorted(set(calendar.index.tolist() + [start_day])))
        ix = union.index(start_day)
        if ix + offset < 0:
            raise ValueError('Exceed TRADEDATE start day')
        if ix + offset >= len(union):
            raise ValueError('Exceed TRADEDATE end day')
        return union[ix + offset]
