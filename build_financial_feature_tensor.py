#!/usr/bin/env python3
"""Build the 503 financial features listed in the factor workbook.

The output is a float32 ``.npy`` array named ``X`` with shape
``(quarter, feature, stock)``.  Its adjacent JSON metadata records the ISO
report-period dates, Wind-style stock identifiers (e.g. ``000001.SZ``), and
Chinese feature names in the corresponding axis order.

The input income-statement and cash-flow fields are year-to-date numbers.  They
are first converted to standalone quarters; ``TTM`` then means the rolling sum
of the last four standalone quarters.  ``diff1q`` and ``diff4q`` are arithmetic
differences, not percentage changes.  Division by zero is represented by NaN.

The tensor is indexed by report period.  ``PUBLISH_DATE`` is retained in the
source-selection logic (earliest disclosure for each statement key), but this
file is not a daily point-in-time panel.  A daily production backtest should
forward-fill a report only from its publication date onwards.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


# Chinese labels in agg_perf_info_with_chinese_formula.xlsx -> source column.
BS_FIELDS = {
    "货币资金": "CASH_C_EQUIV", "应收票据及应收账款": "NR_AR",
    "预付款项": "PREPAYMENT", "其他应收款(合计)": "OTH_RECEIV_TOTAL",
    "存货": "INVENTORIES", "其他流动资产": "OTH_CA",
    "长期股权投资": "LT_EQUITY_INVEST", "固定资产(合计)": "FIXED_ASSETS_TOTAL",
    "在建工程(合计)": "CIP_TOTAL", "无形资产": "INTAN_ASSETS",
    "商誉": "GOODWILL", "长期待摊费用": "LT_AMOR_EXP",
    "递延所得税资产": "DEFER_TAX_ASSETS", "流动资产合计": "T_CA",
    "非流动资产合计": "T_NCA", "资产总计": "T_ASSETS",
    "短期借款": "ST_BORR", "应付票据及应付账款": "NP_AP",
    "预收款项": "ADVANCE_RECEIPTS", "应付职工薪酬": "PAYROLL_PAYABLE",
    "应交税费": "TAXES_PAYABLE", "其他应付款(合计)": "OTH_PAYABLE_TOTAL",
    "其他流动负债": "OTH_CL", "长期借款": "LT_BORR",
    "递延收益": "DEFER_REVENUE", "递延所得税负债": "DEFER_TAX_LIAB",
    "流动负债合计": "T_CL", "非流动负债合计": "T_NCL", "负债合计": "T_LIAB",
    "实收资本(或股本)": "PAID_IN_CAPITAL", "资本公积": "CAPITAL_RESER",
    "盈余公积": "SURPLUS_RESER", "未分配利润": "RETAINED_EARNINGS",
    "归属于母公司所有者权益合计": "T_EQUITY_ATTR_P",
    "所有者权益(或股东权益)合计": "T_SH_EQUITY",
}
IS_FIELDS = {
    "营业总收入": "T_REVENUE", "营业成本": "COGS", "税金及附加": "BIZ_TAX_SURCHG",
    "销售费用": "SELL_EXP", "管理费用": "ADMIN_EXP", "研发费用": "R_D_EXP",
    "财务费用": "FINAN_EXP", "投资收益(损失以“-”号填列)": "INVEST_INCOME",
    "营业利润(亏损以“－”号填列)": "OPERATE_PROFIT", "营业外收入": "NOPERATE_INCOME",
    "营业外支出": "NOPERATE_EXP", "利润总额(亏损总额以“－”号填列)": "T_PROFIT",
    "所得税费用": "INCOME_TAX", "净利润(净亏损以“-”号填列)": "N_INCOME",
}
CF_FIELDS = {
    "销售商品、提供劳务收到的现金": "C_FR_SALE_G_S", "收到的税费返还": "REFUND_OF_TAX",
    "购买商品、接受劳务支付的现金": "C_PAID_G_S", "支付给职工以及为职工支付的现金": "C_PAID_TO_FOR_EMPL",
    "支付的各项税费": "C_PAID_FOR_TAXES", "支付其他与经营活动有关的现金": "C_PAID_FOR_OTH_OP_A",
    "收回投资收到的现金": "PROC_SELL_INVEST", "取得投资收益收到的现金": "GAIN_INVEST",
    "处置固定资产、无形资产和其他长期资产收回的现金净额": "DISP_FIX_ASSETS_OTH",
    "购建固定资产、无形资产和其他长期资产支付的现金": "PUR_FIX_ASSETS_OTH",
    "投资支付的现金": "C_PAID_INVEST", "取得借款收到的现金": "C_FR_BORR",
    "偿还债务支付的现金": "C_PAID_FOR_DEBTS",
    "分配股利、利润或偿付利息支付的现金": "C_PAID_DIV_PROF_INT",
    "支付其他与筹资活动有关的现金": "C_PAID_OTH_FINAN_A",
    "经营活动现金流入小计": "C_INF_FR_OPERATE_A", "经营活动现金流出小计": "C_OUTF_OPERATE_A",
    "经营活动产生的现金流量净额": "N_CF_OPERATE_A", "投资活动现金流入小计": "C_INF_FR_INVEST_A",
    "投资活动现金流出小计": "C_OUTF_FR_INVEST_A", "投资活动产生的现金流量净额": "N_CF_FR_INVEST_A",
    "筹资活动现金流入小计": "C_INF_FR_FINAN_A", "筹资活动现金流出小计": "C_OUTF_FR_FINAN_A",
    "筹资活动产生的现金流量净额": "N_CF_FR_FINAN_A", "现金及现金等价物净增加额": "N_CHANGE_IN_CASH",
}


def stock_id(df: pd.DataFrame) -> pd.Series:
    suffix = df["EXCHANGE_CD"].map({"XSHG": ".SH", "XSHE": ".SZ", "XBSE": ".BJ"}).fillna("")
    return df["TICKER_SYMBOL"].astype(str).str.zfill(6) + suffix


def read_statement(path: Path, fields: dict[str, str]) -> pd.DataFrame:
    columns = ["ID", "TICKER_SYMBOL", "EXCHANGE_CD", "END_DATE", "END_DATE_REP", "REPORT_TYPE", "MERGED_FLAG", "INDUSTRY_CATEGORY", "PUBLISH_DATE"] + list(fields.values())
    raw = pd.read_csv(path, usecols=lambda c: c in columns, low_memory=False)
    raw = raw[(raw["INDUSTRY_CATEGORY"] == "一般工商业") & (raw["MERGED_FLAG"] == 1)].copy()
    # The GP universe in this repository is mainland Shanghai/Shenzhen A shares.
    # NEEQ/BSE records and non-numeric Shanghai legacy symbols are excluded.
    raw = raw[raw["EXCHANGE_CD"].isin(["XSHG", "XSHE"])]
    raw = raw[raw["TICKER_SYMBOL"].astype(str).str.fullmatch(r"\d{6}")]
    # A few source rows use the preceding calendar day (for example 2019-03-30)
    # for an otherwise standard quarter close.  Canonicalise to calendar quarter
    # end before statement-version selection and all rolling calculations.
    raw["quarter"] = pd.to_datetime(raw["END_DATE"], errors="coerce").dt.to_period("Q").dt.end_time.dt.normalize()
    raw["published_at"] = pd.to_datetime(raw["PUBLISH_DATE"], errors="coerce")
    raw["stock"] = stock_id(raw)
    raw = raw[raw["quarter"].dt.month.isin([3, 6, 9, 12])].copy()
    # A statement can have several physical versions.  Earliest publication is
    # the version that was genuinely available first; ID makes ties deterministic.
    keys = ["stock", "quarter", "END_DATE_REP", "REPORT_TYPE", "MERGED_FLAG"]
    raw = raw.sort_values(["published_at", "ID"], na_position="last").drop_duplicates(keys, keep="first")
    # There can still be report-type variants for a stock-quarter. Prefer the
    # first disclosed one, which prevents silently averaging incompatible forms.
    raw = raw.sort_values(["stock", "quarter", "published_at", "ID"], na_position="last").drop_duplicates(["stock", "quarter"], keep="first")
    return raw.set_index(["quarter", "stock"]).sort_index()[list(fields.values())]


def standalone_quarter(cumulative: pd.DataFrame) -> pd.DataFrame:
    """Convert YTD income/cash-flow values into standalone quarter values."""
    out = cumulative.copy()
    frame = out.reset_index()
    frame["year"] = frame["quarter"].dt.year
    frame["month"] = frame["quarter"].dt.month
    values = list(cumulative.columns)
    frame = frame.sort_values(["stock", "year", "quarter"])
    previous = frame.groupby(["stock", "year"], sort=False)[values].shift(1)
    # Q1 is already standalone. For later quarters missing prior cumulative
    # observations should remain missing instead of producing an invalid value.
    later = frame["month"] != 3
    frame.loc[later, values] = frame.loc[later, values] - previous.loc[later, values]
    return frame.set_index(["quarter", "stock"])[values].sort_index()


def top_level_divide(expression: str) -> tuple[str, str] | None:
    depth = 0
    for i in range(len(expression) - 2):
        char = expression[i]
        if char == "(": depth += 1
        elif char == ")": depth -= 1
        elif depth == 0 and expression[i:i + 3] == " / ":
            return expression[:i], expression[i + 3:]
    return None


def unwrap(expression: str, prefix: str) -> str | None:
    start = f"{prefix}("
    if expression.startswith(start) and expression.endswith(")"):
        return expression[len(start):-1]
    return None


def safe_divide(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    return left.divide(right.where(right != 0)).replace([np.inf, -np.inf], np.nan)


def build_tensor(data_dir: Path, workbook: Path, output: Path) -> None:
    factors = pd.read_excel(workbook, sheet_name="factor")["factor_name"].dropna().astype(str).tolist()
    if len(factors) != len(set(factors)):
        raise ValueError("factor_name must be unique")

    bs = read_statement(data_dir / "vw_fdmt_bs_new.csv", BS_FIELDS)
    income = standalone_quarter(read_statement(data_dir / "vw_fdmt_is_new.csv", IS_FIELDS))
    cashflow = standalone_quarter(read_statement(data_dir / "vw_fdmt_cf_new.csv", CF_FIELDS))
    quarters = pd.DatetimeIndex(sorted(set(bs.index.get_level_values("quarter")) | set(income.index.get_level_values("quarter")) | set(cashflow.index.get_level_values("quarter"))))
    stocks = pd.Index(sorted(set(bs.index.get_level_values("stock")) | set(income.index.get_level_values("stock")) | set(cashflow.index.get_level_values("stock"))))

    def align(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.unstack("stock").reindex(index=quarters, columns=pd.MultiIndex.from_product([frame.columns, stocks])).reindex(columns=pd.MultiIndex.from_product([frame.columns, stocks]))

    # Store every raw series in a common [quarter, stock] layout.
    series: dict[str, pd.DataFrame] = {}
    for fields, frame in ((BS_FIELDS, bs), (IS_FIELDS, income), (CF_FIELDS, cashflow)):
        wide = align(frame)
        for chinese, column in fields.items():
            series[chinese] = wide[column].reindex(columns=stocks)
    for chinese in IS_FIELDS:
        series[f"QTR({chinese})"] = series[chinese]
        series[f"TTM({chinese})"] = series[chinese].rolling(4, min_periods=4).sum()
    for chinese in CF_FIELDS:
        series[f"QTR({chinese})"] = series[chinese]
        series[f"TTM({chinese})"] = series[chinese].rolling(4, min_periods=4).sum()

    cache: dict[str, pd.DataFrame] = {}
    def evaluate(expression: str) -> pd.DataFrame:
        if expression in cache:
            return cache[expression]
        inner = unwrap(expression, "diff1q")
        if inner is not None:
            value = evaluate(inner).diff(1)
        else:
            inner = unwrap(expression, "diff4q")
            if inner is not None:
                value = evaluate(inner).diff(4)
            else:
                pair = top_level_divide(expression)
                value = safe_divide(evaluate(pair[0]), evaluate(pair[1])) if pair else series[expression]
        cache[expression] = value
        return value

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix != ".npy":
        raise ValueError("output must use the .npy extension")
    # Write feature-by-feature so the complete tensor does not need an extra
    # in-memory copy while being constructed.
    X = np.lib.format.open_memmap(output, mode="w+", dtype=np.float32, shape=(len(quarters), len(factors), len(stocks)))
    coverage_values = []
    for index, name in enumerate(factors):
        values = evaluate(name).to_numpy(dtype=np.float32, copy=False)
        X[:, index, :] = values
        coverage_values.append(float(np.isfinite(values).mean()))
    X.flush()
    coverage = pd.DataFrame({"feature_name": factors, "coverage": coverage_values})
    coverage.to_csv(output.with_suffix(".coverage.csv"), index=False)
    metadata = {"shape": list(X.shape), "dtype": str(X.dtype), "quarter_start": str(quarters.min().date()), "quarter_end": str(quarters.max().date()), "quarters": quarters.strftime("%Y-%m-%d").tolist(), "stocks": stocks.tolist(), "feature_names": factors}
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: metadata[k] for k in ("shape", "dtype", "quarter_start", "quarter_end")}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/data/fund_agent/data"))
    parser.add_argument("--workbook", type=Path, default=Path("/data/fund_agent/agg_perf_info_with_chinese_formula.xlsx"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/financial_feature_tensor.npy"))
    args = parser.parse_args()
    build_tensor(args.data_dir, args.workbook, args.output)


if __name__ == "__main__":
    main()
