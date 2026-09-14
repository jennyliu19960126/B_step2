"""Run post-GP raw-IC and portfolio-quality analysis.

Style-neutralized statistics are intentionally disabled by default so the
display/backtest stage does not change the IC+LLM genetic fitness pipeline.
"""

import os

from core_func import Backtest


if __name__ == "__main__":
    Backtest(backtest_hori=5, mp_mode=True,
             include_neutralized=False,
             dir_kw_override=os.getenv("GP_RUN_DIR_KW"),
             max_workers_override=int(os.getenv("GP_BACKTEST_JOBS", "8"))).backtest()
