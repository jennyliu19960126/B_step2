def get_factor_start_end_date(calendar, start_date, end_date):
    datelist = calendar.get_range(start_date, end_date)
    factor_start_date, factor_end_date = datelist[0], datelist[-1]
    return factor_start_date, factor_end_date

def int_to_date_str(date_int):
    date = str(date_int)
    return "-".join([date[:4], date[4:6], date[6:8]])

def date_str_to_int(date_str):
    return int("".join(date_str.split("-")))
