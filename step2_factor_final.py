import multiprocessing as mp
from core_func import FactorReplicate
import os
import pandas as pd

if __name__ == "__main__":
    mp_mode = True
    fr = FactorReplicate(dir_kw_override=os.getenv("GP_RUN_DIR_KW"))
    
    result_path = os.path.join(fr.save_dir, f"result_{fr.dir_kw}.csv")
    if os.path.exists(result_path):
        summary_df = pd.read_csv(result_path)
        print("读取到的因子个数：", len(summary_df))
        print("有效表达式个数：", summary_df["formulation_stack"].notna().sum())
        print("头几行：\n", summary_df.head())
    else:
        print("没找到 result 文件，请确认是否已跑完 step1")
        
    fr.replicate_factor(mp_mode=mp_mode)
