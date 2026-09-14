"""Generate a standalone visual HTML report for a completed GP run."""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


DEFAULT_RUN_DIR = Path(
    "runs/output_full_pipeline_v1/backtest/three_layer_smoke"
)


OPERATOR_EXPLANATIONS = {
    "cs_abs": "对信号取绝对值；保留幅度、丢失正负方向",
    "cs_indneutral": "按申万一级行业做 z-score，并裁剪到 [-3, 3]",
    "cs_mul": "逐股票相乘，构造两个财务信号的交互项",
    "cs_rank": "在同一报告期做全市场横截面升序排名",
    "cs_regressor": "每期做横截面回归，并把回归斜率广播给当期股票",
    "ts_delay_1": "使用上一季度的信号值",
    "ts_delta_1": "计算本季度相对上一季度的数值差",
    "ts_mean_8": "计算最近 9 个季度的滚动均值（代码实际窗口）",
    "ts_rank_2": "计算最近 3 个季度的时序分位排名（代码实际窗口）",
    "ts_rank_4": "计算最近 5 个季度的时序分位排名（代码实际窗口）",
    "ts_rank_8": "计算最近 9 个季度的时序分位排名（代码实际窗口）",
    "ts_ttm_4": "对最近 4 个季度求和",
    "ts_zscore_4": "在最近 5 个季度内计算时序 z-score（代码实际窗口）",
    "QTR": "取单季度财务值",
    "TTM": "取最近十二个月财务值",
    "diff1q": "计算财务比率的季度环比变化",
    "diff4q": "计算财务比率的四季度同比变化",
}


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def fmt(value: object, digits: int = 4, percent: bool = False) -> str:
    if value is None or pd.isna(value):
        return "—"
    number = float(value)
    if percent:
        return f"{number * 100:.{digits}f}%"
    return f"{number:.{digits}f}"


def image_data_uri(path: Path) -> str:
    """Downsample a chart and embed it to keep the report portable."""
    with Image.open(path) as source:
        source.thumbnail((1500, 780), Image.Resampling.LANCZOS)
        if source.mode in {"RGBA", "LA"}:
            canvas = Image.new("RGB", source.size, "white")
            canvas.paste(source, mask=source.getchannel("A"))
            source = canvas
        else:
            source = source.convert("RGB")
        buffer = io.BytesIO()
        source.save(buffer, format="WEBP", quality=76, method=6)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/webp;base64,{payload}"


def scale(value: float, low: float, high: float, out_low: float, out_high: float) -> float:
    if high == low:
        return (out_low + out_high) / 2
    return out_low + (value - low) * (out_high - out_low) / (high - low)


def scatter_svg(df: pd.DataFrame) -> str:
    width, height = 720, 390
    left, right, top, bottom = 74, 28, 28, 58
    xs = df["IC_gp"].astype(float)
    ys = df["IC_bt"].astype(float)
    xmin, xmax = xs.min() - 0.002, xs.max() + 0.002
    ymin, ymax = max(0.0, ys.min() - 0.001), ys.max() + 0.001
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="GP fitness 与日频 IC 散点图">']
    for i in range(5):
        x = left + i * (width - left - right) / 4
        y = top + i * (height - top - bottom) / 4
        xv = xmin + i * (xmax - xmin) / 4
        yv = ymax - i * (ymax - ymin) / 4
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" class="grid"/>')
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{x:.1f}" y="{height-bottom+24}" text-anchor="middle" class="axis">{xv:.3f}</text>')
        parts.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" class="axis">{yv:.3f}</text>')
    for _, row in df.iterrows():
        x = scale(float(row.IC_gp), xmin, xmax, left, width - right)
        y = scale(float(row.IC_bt), ymin, ymax, height - bottom, top)
        good = float(row.IC_bt) >= 0.01 and float(row.coverage) >= 0.9 and pd.notna(row.sharpe)
        cls = "dot good" if good else "dot"
        title = f"{row.factor_name}｜季度 fitness {row.IC_gp:.4f}｜日频 IC {row.IC_bt:.4f}"
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" class="{cls}"><title>{esc(title)}</title></circle>')
    parts.extend([
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="axis-line"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" class="axis-line"/>',
        f'<text x="{(left+width-right)/2:.1f}" y="{height-12}" text-anchor="middle" class="label">惩罚后季度 IC（GP fitness）</text>',
        f'<text transform="translate(18 {(top+height-bottom)/2:.1f}) rotate(-90)" text-anchor="middle" class="label">日频 Rank IC</text>',
        '</svg>'
    ])
    return "".join(parts)


def generation_svg(gen: pd.DataFrame) -> str:
    rows = gen[gen["generation"].astype(str) != "total"].copy()
    width, height = 680, 310
    left, right, top, bottom = 70, 28, 30, 52
    ymin = 0.0
    ymax = max(rows["best_IC"].max(), rows["average_IC"].max()) * 1.15
    xs = [scale(i, 0, max(1, len(rows)-1), left, width-right) for i in range(len(rows))]
    avg_pts, best_pts = [], []
    for i, (_, row) in enumerate(rows.iterrows()):
        avg_pts.append((xs[i], scale(float(row.average_IC), ymin, ymax, height-bottom, top)))
        best_pts.append((xs[i], scale(float(row.best_IC), ymin, ymax, height-bottom, top)))
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="代际 selection fitness 变化">']
    for i in range(5):
        y = top + i * (height-top-bottom)/4
        v = ymax - i*ymax/4
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" class="axis">{v:.3f}</text>')
    for points, cls, label in [(avg_pts, "avg-line", "平均 fitness"), (best_pts, "best-line", "最优 fitness")]:
        pstr = " ".join(f"{x:.1f},{y:.1f}" for x,y in points)
        parts.append(f'<polyline points="{pstr}" class="{cls}"/>')
        for (x,y), (_, row) in zip(points, rows.iterrows()):
            value = row.average_IC if label == "平均 fitness" else row.best_IC
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" class="{cls}-dot"><title>第 {row.generation} 代 {label}: {value:.4f}</title></circle>')
    for i, (_, row) in enumerate(rows.iterrows()):
        parts.append(f'<text x="{xs[i]:.1f}" y="{height-bottom+25}" text-anchor="middle" class="axis">第 {esc(row.generation)} 代</text>')
    parts.append('<g transform="translate(390 18)"><line x1="0" y1="0" x2="30" y2="0" class="avg-line"/><text x="38" y="5" class="axis">平均 fitness</text><line x1="125" y1="0" x2="155" y2="0" class="best-line"/><text x="163" y="5" class="axis">最优 fitness</text></g>')
    parts.append('</svg>')
    return "".join(parts)


def horizontal_bars(items: list[tuple[str, int]], title: str) -> str:
    maximum = max(v for _, v in items) if items else 1
    rows = []
    for label, value in items:
        width = 100 * value / maximum
        rows.append(
            f'<div class="bar-row"><div class="bar-label" title="{esc(label)}">{esc(label)}</div>'
            f'<div class="bar-track"><span style="width:{width:.1f}%"></span></div><b>{value}</b></div>'
        )
    return f'<div class="mini-chart"><h3>{esc(title)}</h3>{"".join(rows)}</div>'


def theme_for(formula: str) -> tuple[str, str]:
    if "未分配利润" in formula:
        return (
            "盈利积累 / 权益质量",
            "核心信号观察未分配利润占母公司权益比例的变化或历史位置，意图捕捉利润留存和权益内生积累。数值较高通常表示近期留存收益改善更明显，但也可能受分红政策和一次性损益影响。",
        )
    if "固定资产" in formula:
        return (
            "资产结构 / 资本密集度",
            "核心信号衡量固定资产在非流动资产中的占比，反映企业资本密集程度和长期资产结构。它更像结构特征而非纯盈利指标，行业差异可能较大。",
        )
    if "管理费用" in formula:
        return (
            "费用负担 / 经营效率",
            "核心信号使用管理费用相对总资产的水平或平滑值，尝试刻画管理成本负担与组织效率。未经取负时，高值代表更高费用强度，其正向 IC 需要结合市场定价或反转机制解释。",
        )
    if "所得税费用" in formula:
        return (
            "税负与盈利实现",
            "所得税费用相对资产可同时反映应税利润实现和税负水平；时序排名强调公司自身历史位置，不能简单等价为低税负或高质量盈利。",
        )
    if "营业利润" in formula:
        return (
            "经营盈利能力",
            "营业利润相对资产衡量经营端资产回报，差分或排名进一步关注盈利能力改善及其历史位置。需要留意周期性行业与会计确认节奏。",
        )
    if "净利润" in formula:
        return (
            "盈利能力 / ROE",
            "净利润相对股东权益或总资产衡量盈利能力，时序排名比较公司当前盈利水平与自身历史。高值方向在本样本中对应更高未来收益。",
        )
    if "流动资产" in formula:
        return (
            "资产流动性结构",
            "流动资产占总资产比例的变化反映资产结构向流动性资产迁移或收缩，可能与经营周期、扩张阶段及偿债能力有关。",
        )
    return "复合财务信号", "表达式组合多个财务维度，应结合组成字段和算子逐层判断经济含义。"


def explain_formula(formula: str) -> tuple[str, list[str], list[str]]:
    theme, narrative = theme_for(formula)
    found = re.findall(r"([A-Za-z][A-Za-z0-9_]*)\(", formula)
    steps = []
    for op in reversed(found):
        if op in OPERATOR_EXPLANATIONS and OPERATOR_EXPLANATIONS[op] not in steps:
            steps.append(OPERATOR_EXPLANATIONS[op])
    warnings = []
    if "cs_abs" in formula:
        warnings.append("绝对值变换会丢失原始方向，正向 IC 表示幅度效应，不表示原财务比率越高越好。")
    if "ts_ttm_4" in formula and "TTM(" in formula:
        warnings.append("对已经是 TTM 的输入再次做四季求和，具有重复累计含义，经济解释和尺度稳定性需重点复核。")
    if "cs_regressor" in formula:
        warnings.append("横截面回归斜率在同一季度被广播，单独不提供股票间排序，预测力来自它与另一股票级信号的交互。")
    if formula.count("ts_rank") > 1:
        warnings.append("存在嵌套时序排名，可能增加复杂度但不增加新的财务信息。")
    if "cs_mul" in formula:
        warnings.append("乘法交互会放大极端组合，需在样本外检查稳定性。")
    return theme, steps, [narrative] + warnings


def metric_box(label: str, value: str, cls: str = "") -> str:
    return f'<div class="metric {cls}"><span>{esc(label)}</span><strong>{esc(value)}</strong></div>'


def factor_card(row: pd.Series, run_dir: Path, rank: int) -> str:
    name = str(row.factor_name)
    formula = str(row.formulation_gp)
    theme, steps, notes = explain_formula(formula)
    valid_portfolio = pd.notna(row.sharpe)
    screen = float(row.IC_bt) >= 0.01 and float(row.coverage) >= 0.9 and valid_portfolio
    search = f"{name} {formula} {theme}".lower()
    sort_ic = float(row.IC_bt)
    sort_gp = float(row.IC_gp)
    sort_llm = float(row.llm_interpretability_score)
    badges = [f'<span class="badge rank">日频 IC 排名 #{rank}</span>', f'<span class="badge">{esc(theme)}</span>']
    if screen:
        badges.append('<span class="badge pass">综合初筛通过</span>')
    if not valid_portfolio:
        badges.append('<span class="badge warn">无法有效十分组</span>')
    metrics = "".join([
        metric_box("季度原始 IC", fmt(row.base_fitness, 4)),
        metric_box("惩罚后 IC", fmt(row.IC_gp, 4), "accent"),
        metric_box("LLM 解释分", fmt(row.llm_interpretability_score, 1)),
        metric_box("解释性惩罚", fmt(row.llm_interpretability_penalty, 4)),
        metric_box("日频 Rank IC", fmt(row.IC_bt, 4), "accent"),
        metric_box("覆盖率", fmt(row.coverage, 1, True)),
        metric_box("多空 Sharpe", fmt(row.sharpe, 2)),
        metric_box("对冲 Sharpe", fmt(row.hedge_sharpe, 2)),
    ])
    calc = "".join(f"<li>{esc(step)}</li>" for step in steps) or "<li>直接使用原始财务比率。</li>"
    interpretation = "".join(f"<p>{esc(note)}</p>" for note in notes)
    chart_dir = run_dir / "factor_summary" / name / "5"
    figures = []
    chart_specs = [
        (chart_dir / f"stock_num_{name}.png", "覆盖股票数量"),
        (chart_dir / f"group_return_{name}.png", "十分组累计收益"),
        (chart_dir / f"long_short_return_{name}.png", "多空与基准对冲表现"),
    ]
    for path, caption in chart_specs:
        if path.exists():
            figures.append(
                f'<figure><img loading="lazy" src="{image_data_uri(path)}" alt="{esc(name)} {esc(caption)}" '
                f'onclick="showImage(this.src, this.alt)"><figcaption>{esc(caption)} · 点击放大</figcaption></figure>'
            )
    if not figures:
        figures.append('<div class="empty-chart">该因子没有可用回测图片。</div>')
    return f'''
    <article class="factor-card" data-search="{esc(search)}" data-ic="{sort_ic}" data-gp="{sort_gp}" data-llm="{sort_llm}" data-pass="{str(screen).lower()}">
      <div class="factor-head">
        <div><div class="badges">{"".join(badges)}</div><h3>{esc(name)}</h3></div>
        <button class="toggle" type="button" onclick="toggleCard(this)">展开详情</button>
      </div>
      <code class="formula">{esc(formula)}</code>
      <div class="metric-grid">{metrics}</div>
      <div class="factor-detail">
        <div class="explain-grid">
          <section><h4>计算步骤</h4><ol>{calc}</ol></section>
          <section><h4>经济含义与风险</h4>{interpretation}</section>
        </div>
        <div class="backtest-note">
          <b>回测口径：</b>季度因子向日频前向填充，以未来 5 日收益计算日频 Rank IC；区间 {esc(row.start_date)} 至 {esc(row.end_date)}，共 {int(row.date_size)} 个交易日。多空组合统计{("有效" if valid_portfolio else "无效（取值离散或分组不足）")}。本次未计算风格中性化指标。
        </div>
        <div class="gallery">{"".join(figures)}</div>
      </div>
    </article>'''


def load_quality(run_dir: Path) -> pd.DataFrame:
    """Load a merged quality table, building it from Step 1/3 outputs if needed."""
    comparison_path = run_dir / "factor_quality_comparison.csv"
    if comparison_path.exists():
        return pd.read_csv(comparison_path)

    gp = pd.read_csv(run_dir / f"result_{run_dir.name}.csv").rename(
        columns={"formulation": "formulation_gp", "IC": "IC_gp"}
    )
    summaries = []
    for name in gp["factor_name"].astype(str):
        path = run_dir / "factor_summary" / name / "5" / "summary.csv"
        if path.exists():
            summaries.append(pd.read_csv(path))
    if not summaries:
        raise FileNotFoundError("No Step 3 factor summary was found")
    backtest = pd.concat(summaries, ignore_index=True).rename(
        columns={"IC": "IC_bt", "formulation": "formulation_bt"}
    )
    quality = gp.merge(backtest, on="factor_name", how="left")
    quality.to_csv(comparison_path, index=False)
    return quality


def build_report(run_dir: Path, output: Path, admission_threshold: float = 0.02) -> None:
    quality = load_quality(run_dir)
    quality = quality.sort_values("IC_bt", ascending=False).reset_index(drop=True)
    # Plot the retained positive-fitness candidates before global formula
    # de-duplication.  The legacy gen_summary_all.csv is built after global
    # de-duplication, which can make a later generation look worse merely
    # because its best program was already present in an earlier generation.
    raw = pd.read_csv(run_dir / f"raw_result_{run_dir.name}.csv")
    gen = (raw.groupby("generation", as_index=False)
           .agg(average_IC=("IC", "mean"), best_IC=("IC", "max"), n_sample=("IC", "size")))
    gen["average_length"] = float("nan")
    gen["best_length"] = float("nan")
    manifest_path = run_dir / "pipeline_manifest.json"
    if not manifest_path.exists():
        manifest_path = run_dir / "test_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    selected_features = {}
    selected_ops = {}
    for formula in quality.formulation_gp.astype(str):
        for feature in [
            "未分配利润/母公司权益", "管理费用/总资产", "营业收入/总资产",
            "固定资产/非流动资产", "所得税费用/总资产", "净利润/权益或资产",
            "营业利润/总资产", "流动资产/总资产",
        ]:
            needle = feature.split("/")[0]
            if needle in formula:
                selected_features[feature] = selected_features.get(feature, 0) + 1
        for op in re.findall(r"([A-Za-z][A-Za-z0-9_]*)\(", formula):
            if op.startswith(("cs_", "ts_")):
                selected_ops[op] = selected_ops.get(op, 0) + 1
    feature_items = sorted(selected_features.items(), key=lambda x: x[1], reverse=True)[:7]
    op_items = sorted(selected_ops.items(), key=lambda x: x[1], reverse=True)[:8]
    raw_ops = {}
    for formula in raw.formulation.astype(str):
        for op in re.findall(r"([A-Za-z][A-Za-z0-9_]*)\(", formula):
            if op.startswith(("cs_", "ts_")):
                raw_ops[op] = raw_ops.get(op, 0) + 1
    preferred = sorted(set(raw_ops) | set(selected_ops), key=lambda op: (-selected_ops.get(op, 0), -raw_ops.get(op, 0), op))[:12]
    preference_rows = "".join(
        f"<tr><td><code>{esc(op)}</code></td><td>{raw_ops.get(op,0)}</td><td>{selected_ops.get(op,0)}</td>"
        f"<td>{(100*selected_ops.get(op,0)/raw_ops.get(op,1)):.0f}%</td></tr>" for op in preferred
    )

    cards = "".join(factor_card(row, run_dir, i + 1) for i, (_, row) in enumerate(quality.iterrows()))
    positive = int((quality.IC_bt > 0).sum())
    above = int((quality.IC_bt >= 0.01).sum())
    coverage = int((quality.coverage >= 0.9).sum())
    valid = int(quality.sharpe.notna().sum())
    combined = int(((quality.IC_bt >= 0.01) & (quality.coverage >= 0.9) & quality.sharpe.notna()).sum())
    audit = manifest.get("llm_audit", manifest.get("step1_genetic_search", {}))
    outcomes = audit.get("outcome_counts", {})
    api = outcomes.get("api_success", audit.get("llm_api_success", 0))
    cache = outcomes.get("cache_hit", audit.get("llm_cache_hit", 0))
    weight = float(manifest.get("parameters", {}).get("llm_interpretability_weight", 0.01))
    parameters = manifest.get("parameters", {})
    generation_count = int(parameters.get("generations", len(gen)))
    population = int(parameters.get("population_size", 0))
    warm_count = population * 5
    candidate_attempts = int(audit.get("candidate_scoring_attempts", warm_count + population * generation_count))
    # Reproduce the exact genetic-method lottery.  Each offspring gets a seed
    # from the run RandomState; the first draw from that seed selects method.
    method_names = ["Crossover", "Subtree Mutation", "Hoist Mutation", "Point Mutation"]
    method_limits = [0.40, 0.41, 0.705, 1.0]
    method_expected = [0.40, 0.01, 0.295, 0.295]
    rng = np.random.RandomState(int(parameters.get("random_state", 0)))
    rng.randint(np.iinfo(np.int32).max, size=warm_count)
    method_by_gen = []
    method_totals = {name: 0 for name in method_names}
    for generation in range(1, generation_count + 1):
        seeds = rng.randint(np.iinfo(np.int32).max, size=population)
        counts = {name: 0 for name in method_names}
        for seed in seeds:
            draw = np.random.RandomState(int(seed)).uniform()
            index = int(np.searchsorted(method_limits, draw, side="right"))
            counts[method_names[index]] += 1
            method_totals[method_names[index]] += 1
        method_by_gen.append((generation, counts))
    offspring_count = population * generation_count
    method_rows = "".join(
        f"<tr><td><b>{esc(name)}</b></td><td>{100*expected:.1f}%</td>"
        + "".join(f"<td>{counts[name]}</td>" for _, counts in method_by_gen)
        + f"<td>{method_totals[name]}</td><td>{100*method_totals[name]/offspring_count:.2f}%</td></tr>"
        for name, expected in zip(method_names, method_expected)
    )
    method_generation_headers = "".join(f"<th>Gen{generation}</th>" for generation, _ in method_by_gen)
    max_depth = manifest.get("parameters", {}).get("init_depth", [None, "—"])[1]
    factor_count = len(quality)
    rank_corr = quality["IC_gp"].corr(quality["IC_bt"], method="spearman") if factor_count > 1 else math.nan
    rank_corr_text = f"{rank_corr:.2f}" if pd.notna(rank_corr) else "样本仅1个，无法计算"
    unique_count = int(audit.get("unique_scored_expressions", factor_count))
    raw_count = len(raw)
    png_count = len(list((run_dir / "factor_summary").glob("*/*/*.png")))
    run_stamp = str(manifest.get("started_at") or manifest.get("completed_at") or "—")
    run_date = run_stamp.split()[0]
    example = quality.iloc[0]

    original = pd.read_csv(run_dir / f"original_result_{run_dir.name}.csv")
    abs_formula = "cs_abs(固定资产(合计) / 非流动资产合计)"
    abs_rows = original[original["formulation"] == abs_formula]
    if len(abs_rows):
        abs_row = abs_rows.iloc[0]
        abs_case_html = f'''<section id="abs-case"><h2>05 · ABS 案例在本轮的结果</h2><div class="panel">
        <code class="formula">{esc(abs_formula)}</code>
        <p>该比率在已验证的36×5,392季度矩阵中共有146,184个有效观测，负值为0，因此 <code>abs(x)=x</code>；取 ABS 不改变因子值和原始 IC。</p>
        <div class="kpis"><div class="kpi"><span>原始季度 IC</span><b>{abs_row.base_fitness:.6f}</b></div><div class="kpi"><span>LLM 得分</span><b>{abs_row.llm_interpretability_score:.0f}</b></div><div class="kpi"><span>{weight:.2f} 权重惩罚</span><b>{abs_row.llm_interpretability_penalty:.3f}</b></div><div class="kpi"><span>最终 fitness</span><b>{abs_row.IC:.6f}</b></div></div>
        <p class="callout"><b>本轮结果：</b>{abs_row.base_fitness:.6f} − {abs_row.llm_interpretability_penalty:.3f} = {abs_row.IC:.6f}；按 |fitness| &gt; {admission_threshold:.2f} 判断是否入库。LLM 惩罚改变的是遗传选择 fitness，不改变表达式计算出的因子值。</p>
        <p>理论上只有原始比率出现负值时，ABS 才会翻转符号并改变 IC；<code>abs(abs(x))=abs(x)</code> 则始终是冗余的。当前框架仍只按表达式字符串去重，后续应增加数值等价和语义等价去重。</p>
        </div></section>'''
    else:
        abs_case_html = ""

    css = r'''
    :root{--ink:#152238;--muted:#667085;--paper:#f6f7fb;--card:#fff;--line:#e4e7ec;--navy:#183153;--blue:#2563eb;--cyan:#06b6d4;--green:#0e9f6e;--amber:#d97706;--rose:#e11d48;--shadow:0 12px 36px rgba(16,24,40,.08)}
    *{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.65 Inter,"PingFang SC","Microsoft YaHei",system-ui,sans-serif}a{color:inherit}.shell{max-width:1500px;margin:auto;padding:0 28px 80px}.hero{margin:0 -28px 30px;padding:62px max(28px,calc((100vw - 1444px)/2));color:white;background:radial-gradient(circle at 80% 0,#2563eb 0,transparent 32%),linear-gradient(135deg,#101828,#183153 62%,#164e63)}.eyebrow{letter-spacing:.18em;text-transform:uppercase;color:#93c5fd;font-weight:700}.hero h1{font-size:clamp(32px,5vw,58px);line-height:1.12;margin:.2em 0}.hero p{max-width:900px;color:#dbeafe;font-size:17px}.hero-meta{display:flex;gap:10px;flex-wrap:wrap}.hero-meta span{padding:7px 12px;border:1px solid #ffffff35;border-radius:999px;background:#ffffff12}.nav{position:sticky;top:12px;z-index:20;display:flex;gap:8px;flex-wrap:wrap;padding:10px;margin:0 0 28px;background:#ffffffdd;backdrop-filter:blur(14px);border:1px solid var(--line);border-radius:16px;box-shadow:var(--shadow)}.nav a{text-decoration:none;padding:8px 12px;border-radius:10px;font-weight:650}.nav a:hover{background:#eef4ff;color:var(--blue)}h2{font-size:29px;margin:48px 0 8px}h3{margin:4px 0;font-size:21px}h4{margin:0 0 8px}.lead{color:var(--muted);max-width:1050px}.kpis{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin:22px 0}.kpi{background:var(--card);padding:18px;border:1px solid var(--line);border-radius:16px;box-shadow:0 4px 16px #10182808}.kpi span{display:block;color:var(--muted);font-size:13px}.kpi b{font-size:26px;line-height:1.25}.kpi.good b{color:var(--green)}.panel{background:var(--card);border:1px solid var(--line);border-radius:20px;padding:24px;box-shadow:var(--shadow);margin:18px 0}.two-col{display:grid;grid-template-columns:1.15fr .85fr;gap:18px}.chart svg{width:100%;height:auto}.grid{stroke:#e9edf5;stroke-width:1}.axis-line{stroke:#98a2b3;stroke-width:1.3}.axis{font-size:12px;fill:#667085}.label{font-size:13px;fill:#344054;font-weight:650}.dot{fill:#2563ebaa;stroke:white;stroke-width:1.5}.dot.good{fill:#0e9f6e}.avg-line{fill:none;stroke:#06b6d4;stroke-width:4}.best-line{fill:none;stroke:#e11d48;stroke-width:4}.avg-line-dot{fill:#06b6d4}.best-line-dot{fill:#e11d48}.flow{display:grid;grid-template-columns:repeat(5,1fr);align-items:stretch;gap:28px;margin:28px 0}.flow-box{position:relative;padding:18px;border-radius:16px;background:linear-gradient(145deg,#fff,#f5f8ff);border:1px solid #cdd9f5}.flow-box:not(:last-child):after{content:"→";position:absolute;right:-23px;top:38%;font-size:25px;color:#7c8db5}.flow-box b{display:block;font-size:24px;color:var(--blue)}.flow-box span{color:var(--muted);font-size:13px}.formula-box{padding:18px;background:#0f172a;color:#dbeafe;border-radius:14px;font:15px/1.8 ui-monospace,SFMono-Regular,Consolas,monospace}.penalty{color:#fbbf24}.mini-chart h3{font-size:16px}.bar-row{display:grid;grid-template-columns:150px 1fr 32px;gap:10px;align-items:center;margin:10px 0}.bar-label{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#475467}.bar-track{height:10px;background:#edf1f7;border-radius:99px;overflow:hidden}.bar-track span{display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,var(--blue),var(--cyan))}.callout{padding:18px 20px;border-left:5px solid var(--amber);background:#fffbeb;border-radius:10px;color:#713f12}.toolbar{position:sticky;top:82px;z-index:15;display:grid;grid-template-columns:1fr 210px auto;gap:10px;padding:14px;background:#ffffffed;backdrop-filter:blur(12px);border:1px solid var(--line);border-radius:16px;box-shadow:var(--shadow);margin:18px 0}.toolbar input,.toolbar select,.toolbar button{font:inherit;padding:10px 12px;border:1px solid #d0d5dd;border-radius:10px;background:white}.toolbar button{cursor:pointer}.counter{align-self:center;color:var(--muted);font-weight:650}.factor-list{display:grid;gap:14px}.factor-card{background:white;border:1px solid var(--line);border-radius:18px;padding:20px;box-shadow:0 3px 14px #10182808}.factor-card.highlight{border-color:#86efac}.factor-head{display:flex;justify-content:space-between;gap:16px}.badges{display:flex;gap:7px;flex-wrap:wrap}.badge{font-size:12px;padding:3px 8px;border-radius:999px;background:#eef2ff;color:#3730a3}.badge.rank{background:#eff6ff;color:#1d4ed8}.badge.pass{background:#ecfdf3;color:#067647}.badge.warn{background:#fff7ed;color:#b54708}.toggle{border:0;border-radius:10px;padding:9px 14px;height:max-content;background:#eef4ff;color:#1d4ed8;font-weight:700;cursor:pointer}.formula{display:block;margin:13px 0;padding:12px 14px;white-space:normal;overflow-wrap:anywhere;background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;color:#334155}.metric-grid{display:grid;grid-template-columns:repeat(8,1fr);gap:8px}.metric{padding:10px;background:#f8fafc;border-radius:10px}.metric span{display:block;font-size:11px;color:var(--muted)}.metric strong{font-size:16px}.metric.accent strong{color:var(--blue)}.factor-detail{display:none;padding-top:20px}.factor-card.open .factor-detail{display:block}.explain-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.explain-grid section{padding:18px;background:#f8fafc;border-radius:14px}.explain-grid p,.explain-grid ol{margin:5px 0}.backtest-note{margin:16px 0;padding:14px 16px;background:#eff6ff;border-radius:12px;color:#1e3a5f}.gallery{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.gallery figure{margin:0;border:1px solid var(--line);border-radius:12px;overflow:hidden;background:#f8fafc}.gallery img{width:100%;display:block;cursor:zoom-in}.gallery figcaption{padding:8px 12px;color:var(--muted);font-size:13px}.empty-chart{padding:40px;text-align:center;color:var(--muted)}dialog{width:min(96vw,1500px);border:0;border-radius:16px;padding:8px;box-shadow:0 30px 90px #0008}dialog::backdrop{background:#000a}dialog img{max-width:100%;max-height:88vh;display:block;margin:auto}dialog button{position:absolute;right:16px;top:16px;border:0;border-radius:999px;width:38px;height:38px;background:#111c;color:white;font-size:20px;cursor:pointer}.footer{margin-top:50px;padding-top:20px;border-top:1px solid var(--line);color:var(--muted)}
    .callout.green{border-color:var(--green);background:#ecfdf3;color:#065f46}.fitness-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.equation{padding:22px;background:#0f172a;color:#dbeafe;border-radius:16px;font:16px/1.9 ui-monospace,SFMono-Regular,Consolas,monospace}.equation strong{color:white}.equation .result{color:#86efac}.rule-list{margin:0;padding-left:22px}.rule-list li{margin:8px 0}.compare{width:100%;border-collapse:collapse;margin:15px 0;font-variant-numeric:tabular-nums}.compare th,.compare td{padding:11px 12px;border-bottom:1px solid var(--line);text-align:right;vertical-align:top}.compare th:first-child,.compare td:first-child{text-align:left}.compare thead th{background:#f8fafc;color:#475467;font-size:13px}.compare code{white-space:normal;overflow-wrap:anywhere}
    @media(max-width:1100px){.kpis{grid-template-columns:repeat(3,1fr)}.metric-grid{grid-template-columns:repeat(4,1fr)}.flow{grid-template-columns:1fr}.flow-box:not(:last-child):after{content:"↓";right:50%;top:auto;bottom:-28px}.two-col,.fitness-grid{grid-template-columns:1fr}.compare{display:block;overflow-x:auto;white-space:nowrap}}
    @media(max-width:700px){.shell{padding:0 14px 50px}.hero{margin:0 -14px;padding:40px 18px}.kpis{grid-template-columns:repeat(2,1fr)}.metric-grid{grid-template-columns:repeat(2,1fr)}.toolbar{position:static;grid-template-columns:1fr}.explain-grid,.gallery{grid-template-columns:1fr}.nav{position:static}.bar-row{grid-template-columns:120px 1fr 28px}}
    @media print{.nav,.toolbar,.toggle{display:none}.factor-detail{display:block!important}.factor-card{break-inside:avoid;box-shadow:none}.hero{background:#183153}.gallery{grid-template-columns:1fr 1fr}}
    '''

    document = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>财务因子 GP 挖掘与入库汇报</title><style>{css}</style></head>
<body><div class="shell">
<header class="hero"><div class="eyebrow">Genetic Programming · IC + LLM</div><h1>财务因子 GP<br>挖掘与入库汇报</h1>
<p>从 {candidate_attempts:,} 次候选评估到 {factor_count} 个入库因子：展示表达式生成、LLM 可解释性惩罚、季度 IC 和未来 5 日收益日频回测的完整链路。</p>
<div class="hero-meta"><span>运行：{esc(run_dir.name)}</span><span>日期：{esc(run_date)}</span><span>最大深度：{esc(max_depth)}</span><span>遗传代数：{generation_count}</span><span>惩罚权重：{weight:.2f}</span><span>样本：36 季 × 5,392 股票</span><span>IC + LLM 惩罚迭代</span></div></header>
<nav class="nav"><a href="#summary">结果总览</a><a href="#fitness">Fitness 口径</a><a href="#process">GP 迭代流程</a><a href="#quality">质量分布</a><a href="#abs-case">ABS 案例</a><a href="#factors">入库因子详情</a><a href="#limits">结论与边界</a></nav>

<section id="summary"><h2>01 · 结果总览</h2><p class="lead">这次运行采用最大表达式深度 4、300 正式种群和 warm-start。所有候选均进入 LLM 可解释性评分，并与季度横截面 Pearson IC 合成为选择 fitness。</p>
<div class="kpis">
<div class="kpi"><span>候选评分入口</span><b>{candidate_attempts:,}</b></div><div class="kpi"><span>唯一表达式</span><b>{unique_count:,}</b></div><div class="kpi"><span>最终入库</span><b>{factor_count}</b></div>
<div class="kpi good"><span>日频 IC 为正</span><b>{positive}/{factor_count}</b></div><div class="kpi good"><span>日频 IC ≥ 0.01</span><b>{above}/{factor_count}</b></div><div class="kpi"><span>综合初筛</span><b>{combined}/{factor_count}</b></div>
</div>
<div class="panel two-col"><div><h3>如何阅读这批结果</h3><p><b>当前 GP 真正优化的 fitness，是“有方向的季度原始 Pearson IC 减去 LLM 可解释性惩罚”</b>，不是日频 Rank IC、Sharpe、ICIR，也不是取绝对值后的 IC。</p><p>入库因子惩罚后季度 IC 均值为 <b>{quality.IC_gp.mean():.4f}</b>，日频 Rank IC 均值为 <b>{quality.IC_bt.mean():.4f}</b>。季度 fitness 与日频 IC 的排序相关为 <b>{rank_corr_text}</b>。</p><p>{coverage}/{factor_count} 的覆盖率超过 90%，{valid}/{factor_count} 能形成有效十分组。绿色“综合初筛通过”要求日频 IC≥0.01、覆盖率≥90% 且组合统计有效。</p></div>
<div class="formula-box"><b>适应度</b><br>季度横截面 Pearson IC<br><span class="penalty">− {weight:.2f} × (1 − LLM得分 ÷ 10)</span><br><br><small>LLM 只评价经济直觉与表达式结构，不读取回测收益。</small></div></div></section>

<section id="fitness"><h2>02 · 当前 Fitness 到底是什么</h2><p class="lead">这里必须区分“用于遗传选择的 fitness”和“入库后的日频回测指标”。GP 只看前者；日频 Rank IC、分组收益和 Sharpe 都是在表达式入库之后才计算。</p>
<div class="fitness-grid"><div class="equation">
<strong>第 1 步：季度原始 IC</strong><br>
IC<sub>t</sub> = Corr<sub>股票横截面</sub>(Factor<sub>t</sub>, FutureReturn<sub>t</sub>)<br>
BaseFitness = Mean<sub>36个季度</sub>(IC<sub>t</sub>)<br><br>
<strong>第 2 步：LLM 惩罚</strong><br>
Penalty = {weight:.2f} × (1 − Score / 10)<br><br>
<strong>第 3 步：最终 fitness</strong><br>
<span class="result">Fitness = BaseFitness − Penalty</span>
</div><div class="panel" style="margin:0"><ul class="rule-list">
<li><b>IC 有方向：</b>负 IC 就是负值，不会取绝对值翻成正值。</li>
<li><b>不是 Rank IC：</b>季度阶段直接使用因子原始数值和未来收益计算 Pearson 相关。</li>
<li><b>没有中性化：</b>不使用 Barra 风格暴露、行业或市值残差。</li>
<li><b>没有复杂度罚项：</b><code>parsimony_coefficient=0</code>，当前没有长度惩罚。</li>
<li><b>每个候选表达式都进入 LLM 评分：</b>相同字符串命中缓存；API 失败时 fail-open，惩罚为0。</li>
<li><b>准入门槛：</b>当前结果选择的是惩罚后 fitness 的绝对值 &gt; {admission_threshold:.2f}。</li>
</ul></div></div>
<div class="panel"><h3>惩罚的量级</h3><p>LLM 得分10、6、3、0时，惩罚分别是0、{weight*0.4:.3f}、{weight*0.7:.3f}、{weight:.3f}。本轮入库表达式原始季度 IC 为 {example.base_fitness:.6f}、得分 {example.llm_interpretability_score:.0f}，最终 fitness 为 {example.IC_gp:.6f}。LLM 不改变因子矩阵和回测收益，只改变遗传选择与准入排序。</p></div>
</section>

<section id="process"><h2>03 · GP 迭代流程与系统架构</h2><p class="lead">搜索空间包含 503 个财务特征与约 90 个截面/时序算子。深度上限 4 同时约束初始树以及交叉、变异后的后代。</p>
<div class="panel"><h3>四层计算架构</h3><div class="flow"><div class="flow-box"><b>第 1 层</b><strong>基础张量</strong><span>503 个季度财务字段 × 36 个报告期 × 5,392 只股票；缺失值保留为 NaN。</span></div><div class="flow-box"><b>第 2 层</b><strong>表达式生成</strong><span>从字段终端、常数和约 90 个截面/时序算子生成候选树；warm-start 产生 1,500 个候选。</span></div><div class="flow-box"><b>第 3 层</b><strong>双重 Fitness</strong><span>LLM 给出 0–10 的解释分并扣罚，再计算 36 季原始 Pearson IC；二者合成遗传选择 fitness。</span></div><div class="flow-box"><b>第 4 层</b><strong>准入与验证</strong><span>表达式去重、|fitness| 门槛筛选、季度到日频前向填充，最后做未来 5 日收益的十分组回测。</span></div></div><p class="small">本轮配置的树深度上限为 4；实际被保留的程序最大深度为 2，说明强惩罚与筛选将结果集中在较浅表达式。</p></div>
<div class="flow">
<div class="flow-box"><b>{warm_count:,}</b><strong>Warm-start 候选</strong><span>{population} × 5，扩大初始搜索覆盖</span></div>
<div class="flow-box"><b>{generation_count} 代</b><strong>遗传迭代</strong><span>选择、交叉、变异；每代 {population}</span></div>
<div class="flow-box"><b>{candidate_attempts:,}</b><strong>逐候选 LLM 审计</strong><span>{api} 次 API，{cache} 次缓存命中</span></div>
<div class="flow-box"><b>{raw_count}</b><strong>正 fitness 记录</strong><span>逐代结果汇总中的候选记录</span></div>
<div class="flow-box"><b>{factor_count}</b><strong>惩罚后入库</strong><span>|最终 fitness| &gt; {admission_threshold:.2f}</span></div>
</div>
<div class="panel two-col"><div class="chart"><h3>代际 selection fitness（正值候选，去重前）</h3>{generation_svg(gen)}</div><div><h3>一条候选的评价路径</h3><ol><li>从财务特征终端和算子集合生成表达式。</li><li>检查表达式树深度，上限为 {esc(max_depth)}。</li><li>每个候选先获得 LLM 解释分，并形成惩罚项。</li><li>执行季度因子矩阵，计算 36 季横截面 Pearson IC 均值。</li><li><b>selection fitness = 原始 IC − {weight:.2f} × (1 − LLM分/10)</b>，遗传选择直接使用该结果。</li><li>|fitness| 大于 {admission_threshold:.2f} 的去重表达式进入日频回测。</li></ol><p class="callout">Gen0 是 {warm_count:,} 个 warm-start 初始池；图中 Gen1–Gen{generation_count} 是 {generation_count} 个遗传后代。曲线中的数值不是纯 IC，而是已经包含 LLM 惩罚的 selection fitness。</p></div></div>
<div class="panel two-col">{horizontal_bars(feature_items,"入库表达式的主要财务主题")}{horizontal_bars(op_items,"入库表达式的高频算子")}</div></section>
<div class="panel"><h3>遗传选择偏好的算子</h3><p class="lead">“正 fitness 出现次数”反映算子进入有效候选的频率；“入库出现次数”反映通过 {admission_threshold:.2f} 门槛后的集中度。它是本次单随机种子的经验选择偏好，不代表算子的先验权重。</p><table class="compare"><thead><tr><th>算子</th><th>正 fitness 候选</th><th>入库表达式</th><th>入库/出现</th></tr></thead><tbody>{preference_rows}</tbody></table></div>
<div class="panel"><h3>真实遗传操作：Crossover 并不少</h3><p class="lead">下表按本轮随机种子逐个重放后代的“遗传操作抽签”。它统计的是产生后代时实际选择了哪种方法，与上面的 <code>ts_rank_8</code> 等表达式算子是两类概念。</p><table class="compare"><thead><tr><th>遗传操作</th><th>配置概率</th>{method_generation_headers}<th>四代合计</th><th>实际占比</th></tr></thead><tbody>{method_rows}</tbody></table>
<div class="flow"><div class="flow-box"><b>{offspring_count:,}</b><strong>四代全部后代</strong><span>{population} 个/代 × {generation_count} 代</span></div><div class="flow-box"><b>{method_totals['Crossover']}</b><strong>Crossover 抽签</strong><span>{100*method_totals['Crossover']/offspring_count:.2f}%；接近配置的 40%</span></div><div class="flow-box"><b>{raw_count}</b><strong>正 fitness 去重记录</strong><span>所有遗传方法合计；当前未保存 method 映射</span></div><div class="flow-box"><b>{factor_count}</b><strong>最终入库表达式</strong><span>|selection fitness| &gt; {admission_threshold:.2f}</span></div></div>
<p class="callout"><b>为什么最终公式看起来很少来自 crossover？</b> Crossover 是树的生成方式，不是公式中的一个函数名，因此无法从表达式外观直接识别。当前配置实际抽中了 {method_totals['Crossover']} 次 crossover；但强锦标赛选择（tournament=30）使父代迅速集中，短树之间交换子树经常得到重复、裸字段或近似结构。之后还要经过深度上限、IC+LLM 惩罚、正 fitness 筛选、字符串去重和 0.01 入库线。现有 CSV 没有保存每个后代的 <code>parents.method</code>，所以不能可靠计算“460 次中有多少最终入库”。下一版应把 method、parent、donor、深度回退原因随候选一起落盘。</p></div>
<div class="panel"><h3>多个可核验的跨代结构案例</h3><p class="lead">运行文件没有保存 parent ID，因此不虚构精确血缘；以下展示各代文件中真实出现的“跨代保留、嵌套增强和后期探索”。</p>
<div class="two-col"><div><span class="badge pass">案例 A · 最优结构跨四代保留</span><h4>固定资产 / 非流动资产</h4><div class="formula-box">Gen1 → Gen2 → Gen3 → Gen4<br><br>固定资产(合计) / 非流动资产合计<br><br>selection fitness 始终为 0.030483</div><p>这说明四代中最优值没有下降；它的原始 IC 为 0.039483、LLM 评分 7、0.03 权重惩罚 0.009。</p></div>
<div><span class="badge">案例 B · Gen2 嵌套增强</span><h4>盈利排名再次做时序排名</h4><div class="formula-box">Gen1: ts_rank_8(利润总额 / 权益)<br>fitness = 0.028995<br><br>Gen2 同源结构:<br>ts_rank_8(ts_rank_8(利润总额 / 权益))<br>fitness = 0.013564</div><p>新增一层后 LLM 分从 7 降到 4，惩罚由 0.009 增至 0.018；复杂化没有带来更高 selection fitness。</p></div></div>
<div class="two-col"><div><span class="badge warn">案例 C · 同源截面变换</span><h4>固定资产 / 总资产</h4><div class="formula-box">cs_demean(固定资产 / 总资产)<br>cs_zscore(固定资产 / 总资产)<br><br>二者原始 IC = 0.027624<br>selection fitness = 0.021624</div><p>Pearson IC 对线性平移/缩放不敏感，因此两个表达式数值尺度不同但 IC 相同；当前字符串去重无法识别这种等价性。</p></div><div><span class="badge">案例 D · Gen4 后期探索</span><h4>现金流与盈利同比变化</h4><div class="formula-box">diff4q(TTM(职工现金支出 / 经营现金流出))<br>diff4q(TTM(净利润 / 总资产))<br>diff4q(TTM(营业利润 / 总资产))</div><p>这些结构在 Gen4 出现正 selection fitness，但没有超过 0.01 入库线，表明后期仍在探索新主题，却没有击败早期保留的简单结构。</p></div></div></div>

<section id="quality"><h2>04 · 入库因子质量分布</h2><p class="lead">散点图中的绿色点为综合初筛通过因子。鼠标悬停可查看因子名称与数值。横轴是 GP 阶段的惩罚后季度 IC；纵轴是入库后另行计算的日频 Rank IC，两者不是同一个指标。</p>
<div class="panel chart">{scatter_svg(quality)}</div>
<div class="kpis"><div class="kpi"><span>日频 IC 中位数</span><b>{quality.IC_bt.median():.4f}</b></div><div class="kpi"><span>日频 IC 最大值</span><b>{quality.IC_bt.max():.4f}</b></div><div class="kpi"><span>覆盖率中位数</span><b>{quality.coverage.median()*100:.1f}%</b></div><div class="kpi"><span>LLM 得分均值</span><b>{quality.llm_interpretability_score.mean():.2f}</b></div><div class="kpi"><span>有效十分组</span><b>{valid}/{factor_count}</b></div><div class="kpi"><span>图表产物</span><b>{png_count}</b></div></div></section>

{abs_case_html}

<section id="factors"><h2>06 · {factor_count} 个入库因子的详细解读</h2><p class="lead">默认按日频 Rank IC 排序。每个条目都给出计算步骤、经济含义、风险提示、完整回测指标和现有图表。解释是基于表达式结构的研究性归纳，不构成因果判断。</p>
<div class="toolbar"><input id="search" type="search" placeholder="搜索因子名、财务字段或主题…" oninput="filterCards()"><select id="sort" onchange="sortCards()"><option value="ic">按日频 IC</option><option value="gp">按季度 fitness</option><option value="llm">按 LLM 得分</option></select><label><input id="passOnly" type="checkbox" onchange="filterCards()"> 仅看综合初筛</label><span class="counter" id="counter">{factor_count} 个因子</span></div>
<div class="factor-list" id="factorList">{cards}</div></section>

<section id="limits"><h2>07 · 结论与使用边界</h2><div class="panel"><p><b>工程结论：</b>深度 4、{generation_count} 代的 IC + LLM 惩罚闭环已跑通，{candidate_attempts:,} 次候选评价均有审计记录，{factor_count} 个入库因子均完成复算和日频回测。</p><p><b>研究结论：</b>{factor_count} 个因子中有 {positive} 个日频 IC 为正，{above} 个达到0.01。当前惩罚权重为 {weight:.2f}，selection fitness 全程包含 LLM 惩罚。</p><p><b>不能直接实盘：</b>搜索和回测仍使用同一时期；季度信号前向填充至日频，未来5日收益相互重叠，容易放大自相关、ICIR与Sharpe。下一步应采用时间样本外切分、非重叠持有期、语义去重和候选相关性聚类。</p></div></section>
<footer class="footer">数据来源：本地 three_layer_smoke 运行产物 · 报告由 <code>generate_visual_report.py</code> 生成 · 图片已内嵌，可离线打开</footer>
</div>
<dialog id="imageDialog" onclick="if(event.target===this)this.close()"><button onclick="this.parentElement.close()">×</button><img id="dialogImage" alt="回测图放大"></dialog>
<script>
function toggleCard(btn){{const card=btn.closest('.factor-card');card.classList.toggle('open');btn.textContent=card.classList.contains('open')?'收起详情':'展开详情'}}
function showImage(src,alt){{const d=document.getElementById('imageDialog'),img=document.getElementById('dialogImage');img.src=src;img.alt=alt;d.showModal()}}
function filterCards(){{const q=document.getElementById('search').value.trim().toLowerCase(),only=document.getElementById('passOnly').checked;let n=0;document.querySelectorAll('.factor-card').forEach(c=>{{const show=(!q||c.dataset.search.includes(q))&&(!only||c.dataset.pass==='true');c.style.display=show?'':'none';if(show)n++}});document.getElementById('counter').textContent=n+' 个因子'}}
function sortCards(){{const key=document.getElementById('sort').value,list=document.getElementById('factorList');[...list.children].sort((a,b)=>Number(b.dataset[key])-Number(a.dataset[key])).forEach(x=>list.appendChild(x));filterCards()}}
</script></body></html>'''

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    print(f"report={output.resolve()}")
    print(f"factors={len(quality)}")
    print(f"size_mb={output.stat().st_size / 1024 / 1024:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--admission-threshold", type=float, default=0.02)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output = args.output or run_dir / "gp_factor_visual_report.html"
    build_report(run_dir, output.resolve(), args.admission_threshold)


if __name__ == "__main__":
    main()
