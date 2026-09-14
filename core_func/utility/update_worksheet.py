import os
import xlsxwriter
import pandas as pd

def update_worksheet(file_path, factor_list, summary_df_cols, backtest_hori, backtest_dir):
    workbook = xlsxwriter.Workbook(file_path, {'nan_inf_to_errors': True})
    worksheet = workbook.add_worksheet('factor')
    # write header
    for num, col in enumerate(summary_df_cols):
        worksheet.write(0, num, col)
    # set format
    stat_format = workbook.add_format({'num_format': '0.000'})
    int_format = workbook.add_format({'num_format': '0'})
    perc_format = workbook.add_format({'num_format': '0.0%'})
    # combine all summary
    count = 1
    for factor in factor_list:
        factor_dir = os.path.join(backtest_dir, factor, str(backtest_hori))
        url = factor_dir
        summary_path = os.path.join(factor_dir, "summary.csv")
        if os.path.exists(summary_path):
            summary = pd.read_csv(summary_path)
            for num, col in enumerate(summary_df_cols):
                if col not in summary.columns:
                    worksheet.write(count, num, "")
                    continue
                val = summary.at[0, col]
                if num == 0:
                    worksheet.write_url(
                        count, num, f"external:{os.path.abspath(url)}",
                        string=str(val)
                    )
                elif col in ['start_date', 'end_date', 'formulation']:
                    worksheet.write(count, num, str(val))
                elif col == 'date_size':
                    worksheet.write(count, num, val, int_format)
                elif col in ['IC', 'IR','sharpe', 'hedge_sharpe', 'neu_IC', 'neu_hedge_sharpe', 'long_only_neu_IC']:
                    worksheet.write(count, num, val, stat_format)
                elif col in ['ret', 'mdd', 'hedge_ret', 'hedge_mdd', 'coverage']:
                    worksheet.write(count, num, val, perc_format)
        else:
            worksheet.write_url(
                count, 0, f"external:{os.path.abspath(url)}", string=factor
            )
            print(f"Error_file: {summary_path}")
        count += 1
    # conditional_format
    for cl in ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']:
        worksheet.conditional_format(f'{cl}2:{cl}{count}', {'type': 'data_bar', 'data_bar_2010': True})
    workbook.close()        
    return
