#!/usr/bin/env python3
"""Run one reproducible GP evolution and build a node-level HTML report.

The run deliberately disables external LLM calls.  Every node of the best
final-generation expression (including feature leaves) is evaluated as a
standalone factor with quarterly IC and daily five-day-forward backtests.
"""

from __future__ import annotations

import argparse
import ast
import html
import json
import math
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TreeNode:
    node_id: int
    token: Any
    expression: str
    value: np.ndarray
    kind: str
    depth: int
    parent_id: int | None = None
    children: list["TreeNode"] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    daily_ic: pd.Series | None = None
    cumulative_long_short: pd.Series | None = None


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def fmt(value: Any, digits: int = 4, percent: bool = False) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    number = float(value)
    if percent:
        return f"{100 * number:.{digits}f}%"
    return f"{number:.{digits}f}"


def parse_program(tokens, function_map, feature_names, X):
    nodes: list[TreeNode] = []

    def parse_at(position: int, depth: int, parent_id: int | None):
        token = tokens[position]
        node_id = len(nodes)
        placeholder = TreeNode(
            node_id=node_id,
            token=token,
            expression="",
            value=np.empty((0, 0)),
            kind="",
            depth=depth,
            parent_id=parent_id,
        )
        nodes.append(placeholder)
        position += 1
        if isinstance(token, str):
            function = function_map[token]
            children = []
            for _ in range(function.arity):
                child, position = parse_at(position, depth + 1, node_id)
                children.append(child)
            value = function(*[child.value for child in children])
            expression = f"{token}({', '.join(child.expression for child in children)})"
            kind = "cs" if token.startswith("cs_") else "ts"
            placeholder.children = children
            placeholder.value = np.asarray(value, dtype=float)
            placeholder.expression = expression
            placeholder.kind = kind
        elif isinstance(token, int):
            placeholder.value = np.asarray(X[:, token, :], dtype=float)
            placeholder.expression = feature_names[token]
            placeholder.kind = "feature"
        else:
            placeholder.value = np.full((X.shape[0], X.shape[2]), float(token))
            placeholder.expression = f"{float(token):.3f}"
            placeholder.kind = "constant"
        return placeholder, position

    root, end = parse_at(0, 0, None)
    if end != len(tokens):
        raise ValueError(f"program parsing stopped at {end}/{len(tokens)}")
    return root, nodes


def quarter_metrics(values, y, sample_weight, fitness_cache, factor_rate):
    from core.fitness import _mask_eval_data, _cal_total_IC

    masked_y, masked_factor = _mask_eval_data(
        y, values, sample_weight, fitness_cache[0], fitness_cache[1]
    )
    ic, ic_series = _cal_total_IC(masked_y, masked_factor, method="pearson")
    restrict = fitness_cache[1]
    raw = values[np.where(sample_weight == 1)].copy()
    raw = raw[np.where(fitness_cache[0] == 1)]
    raw[restrict != 0] = np.nan
    denominator = np.maximum((restrict == 0).sum(axis=1), 1)
    coverage = np.isfinite(raw).sum(axis=1) / denominator
    return {
        "quarter_ic": float(ic) if np.isfinite(ic) else math.nan,
        "quarter_coverage": float(np.mean(coverage)),
        "quarter_pass_rate": float(np.mean(coverage >= factor_rate)),
        "quarter_valid_ic": int(ic_series.notna().sum()),
    }


def quarter_for_daily_date(date: pd.Timestamp) -> pd.Timestamp:
    if 5 <= date.month <= 8:
        return pd.Timestamp(date.year, 3, 31)
    if 9 <= date.month <= 10:
        return pd.Timestamp(date.year, 6, 30)
    if date.month >= 11:
        return pd.Timestamp(date.year, 9, 30)
    return pd.Timestamp(date.year - 1, 9, 30)


def daily_backtest(values, quarter_dates, stocks, restrict_df, rank_ret_df, raw_ret_df):
    qdf = pd.DataFrame(values, index=pd.to_datetime(quarter_dates), columns=stocks)
    dates = restrict_df.index
    labels = pd.DatetimeIndex([quarter_for_daily_date(pd.Timestamp(d)) for d in dates])
    daily = qdf.reindex(labels).copy()
    daily.index = dates
    daily = daily.reindex(columns=restrict_df.columns)

    common = dates.intersection(rank_ret_df.index).intersection(raw_ret_df.index)
    daily = daily.loc[common].astype(float)
    restrict = restrict_df.loc[common]
    rank_ret = rank_ret_df.loc[common]
    raw_ret = raw_ret_df.loc[common]
    daily[(restrict != 0) | rank_ret.isna()] = np.nan

    available = np.maximum((restrict == 0).sum(axis=1), 1)
    coverage_series = daily.notna().sum(axis=1) / available
    factor_rank = daily.rank(axis=1, pct=True)
    return_rank = rank_ret.rank(axis=1, pct=True)
    daily_ic = factor_rank.corrwith(return_rank, axis=1, method="pearson")
    daily_ic_valid = daily_ic.dropna()
    ic_mean = daily_ic_valid.mean()
    ic_std = daily_ic_valid.std()
    icir = (
        ic_mean / ic_std * math.sqrt(len(daily_ic_valid))
        if len(daily_ic_valid) and np.isfinite(ic_std) and ic_std > 0
        else math.nan
    )

    # Existing Step 3 convention: use contemporaneous daily signal rows with
    # the already forward-shifted five-day return matrix.
    from utility.calc_func import max_drawdown_cal

    # Continuous signals use deciles. Discrete intermediate nodes (such as
    # ts_rank_2) use their available value buckets so they still receive a
    # complete standalone portfolio backtest.
    finite_values = daily.to_numpy()[np.isfinite(daily.to_numpy())]
    distinct = np.unique(finite_values)
    portfolio_groups = min(10, int(distinct.size))
    spread_rows = []
    for position in range(1, len(common)):
        signal = daily.iloc[position].to_numpy(dtype=float)
        returns = raw_ret.iloc[position].to_numpy(dtype=float)
        allowed = restrict.iloc[position].to_numpy() == 0
        mask = np.isfinite(signal) & np.isfinite(returns) & allowed
        signal_t, returns_t = signal[mask], returns[mask]
        unique_t = np.unique(signal_t)
        if portfolio_groups < 2 or unique_t.size < 2:
            spread_rows.append(math.nan)
            continue
        if unique_t.size <= 10:
            bottom_mask = signal_t == unique_t[0]
            top_mask = signal_t == unique_t[-1]
        else:
            percentile = pd.Series(signal_t).rank(method="average", pct=True).to_numpy()
            bottom_mask = percentile <= 0.1
            top_mask = percentile > 0.9
        if not bottom_mask.any() or not top_mask.any():
            spread_rows.append(math.nan)
            continue
        spread_rows.append(float(
            np.nanmean(returns_t[top_mask]) - np.nanmean(returns_t[bottom_mask])
        ))

    top_minus_bottom = np.asarray(spread_rows, dtype=float)
    if np.isfinite(top_minus_bottom).sum() < 2:
        long_short = pd.Series(dtype=float)
        annual_return = sharpe = max_drawdown = math.nan
        direction = "无法分组"
    else:
        if np.nanmean(top_minus_bottom) >= 0:
            spread = top_minus_bottom
            direction = "多高分 / 空低分"
        else:
            spread = -top_minus_bottom
            direction = "多低分 / 空高分"
        long_short = pd.Series(spread, index=common[1:])
        annual_return = float(np.nanmean(spread) * 250)
        std = float(np.nanstd(spread))
        sharpe = float(np.nanmean(spread) / std * math.sqrt(250)) if std > 0 else math.nan
        max_drawdown = float(max_drawdown_cal(np.nancumsum(spread)))

    return {
        "daily_ic": float(ic_mean) if np.isfinite(ic_mean) else math.nan,
        "daily_icir": float(icir) if np.isfinite(icir) else math.nan,
        "daily_coverage": float(coverage_series.mean()),
        "daily_valid_ic": int(daily_ic_valid.size),
        "annual_long_short": annual_return,
        "long_short_sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "direction": direction,
        "portfolio_groups": portfolio_groups,
    }, daily_ic, long_short.cumsum()


def sparkline(series: pd.Series | None, color="#2563eb", width=520, height=150):
    if series is None:
        return '<div class="empty">无有效曲线</div>'
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2:
        return '<div class="empty">有效点不足</div>'
    step = max(1, len(clean) // 180)
    values = clean.iloc[::step].to_numpy(dtype=float)
    low, high = float(np.min(values)), float(np.max(values))
    span = high - low or 1.0
    points = []
    for i, value in enumerate(values):
        x = 8 + i * (width - 16) / max(1, len(values) - 1)
        y = 8 + (high - value) * (height - 24) / span
        points.append(f"{x:.1f},{y:.1f}")
    zero = None
    if low <= 0 <= high:
        zero = 8 + high * (height - 24) / span
    zero_line = (
        f'<line x1="8" y1="{zero:.1f}" x2="{width-8}" y2="{zero:.1f}" class="zero"/>'
        if zero is not None else ""
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" class="spark" role="img">'
        f'{zero_line}<polyline points="{" ".join(points)}" fill="none" '
        f'stroke="{color}" stroke-width="3"/></svg>'
    )


def tree_svg(root: TreeNode, nodes: list[TreeNode]):
    leaf_counter = [0]
    positions = {}

    def place(node):
        if not node.children:
            x = leaf_counter[0]
            leaf_counter[0] += 1
        else:
            child_x = [place(child) for child in node.children]
            x = sum(child_x) / len(child_x)
        positions[node.node_id] = (x, node.depth)
        return x

    place(root)
    max_depth = max(node.depth for node in nodes)
    leaf_count = max(1, leaf_counter[0])
    width = max(900, leaf_count * 210)
    height = max(320, (max_depth + 1) * 150)
    sx = (width - 160) / max(1, leaf_count - 1)
    sy = (height - 110) / max(1, max_depth)

    def xy(node):
        x, depth = positions[node.node_id]
        return 80 + x * sx, 55 + depth * sy

    colors = {"cs": "#2563eb", "ts": "#7c3aed", "feature": "#0f9f6e", "constant": "#d97706"}
    parts = [f'<svg viewBox="0 0 {width} {height}" class="tree" role="img" aria-label="最佳表达式逐节点回测树">']
    for node in nodes:
        x1, y1 = xy(node)
        for child in node.children:
            x2, y2 = xy(child)
            parts.append(f'<path d="M{x1:.1f},{y1+32:.1f} C{x1:.1f},{(y1+y2)/2:.1f} {x2:.1f},{(y1+y2)/2:.1f} {x2:.1f},{y2-32:.1f}" class="edge"/>')
    for node in nodes:
        x, y = xy(node)
        label = str(node.token) if isinstance(node.token, str) else node.expression
        short = label if len(label) <= 22 else label[:20] + "…"
        ic = fmt(node.metrics.get("daily_ic"), 3)
        parts.append(
            f'<g class="tree-node" tabindex="0" onclick="document.getElementById(\'node-{node.node_id}\').scrollIntoView({{behavior:\'smooth\'}})">'
            f'<rect x="{x-82:.1f}" y="{y-34:.1f}" width="164" height="68" rx="13" fill="{colors[node.kind]}"/>'
            f'<text x="{x:.1f}" y="{y-5:.1f}" text-anchor="middle">{esc(short)}</text>'
            f'<text x="{x:.1f}" y="{y+17:.1f}" text-anchor="middle" class="sub">日频IC {ic}</text>'
            f'<title>节点 {node.node_id}：{esc(node.expression)}</title></g>'
        )
    parts.append("</svg>")
    return "".join(parts)


def bars_svg(items, width=760, row_height=32):
    items = list(items)
    height = max(90, len(items) * row_height + 32)
    max_value = max((value for _, value in items), default=1)
    parts = [f'<svg viewBox="0 0 {width} {height}" class="bars">']
    for i, (label, value) in enumerate(items):
        y = 12 + i * row_height
        bar_width = (width - 280) * value / max_value
        parts.append(f'<text x="8" y="{y+17}" class="bar-label">{esc(label[:32])}</text>')
        parts.append(f'<rect x="230" y="{y}" width="{bar_width:.1f}" height="20" rx="6"/>')
        parts.append(f'<text x="{240+bar_width:.1f}" y="{y+16}" class="bar-value">{value}</text>')
    parts.append("</svg>")
    return "".join(parts)


def generation_svg(generation_df):
    width, height = 760, 300
    left, right, top, bottom = 70, 30, 30, 55
    values = list(generation_df["average_ic"]) + list(generation_df["best_ic"])
    low, high = min(values), max(values)
    margin = max(0.005, (high - low) * 0.2)
    low -= margin
    high += margin

    def px(i):
        return left + i * (width - left - right) / max(1, len(generation_df) - 1)

    def py(value):
        return top + (high - value) * (height - top - bottom) / (high - low)

    parts = [f'<svg viewBox="0 0 {width} {height}" class="generation">']
    for j in range(5):
        y = top + j * (height - top - bottom) / 4
        value = high - j * (high - low) / 4
        parts.append(f'<line x1="{left}" y1="{y}" x2="{width-right}" y2="{y}" class="grid"/>')
        parts.append(f'<text x="{left-10}" y="{y+4}" text-anchor="end">{value:.3f}</text>')
    for column, color, label in (("average_ic", "#06b6d4", "平均"), ("best_ic", "#e11d48", "最优")):
        points = " ".join(f"{px(i):.1f},{py(row[column]):.1f}" for i, (_, row) in enumerate(generation_df.iterrows()))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="4"/>')
        for i, (_, row) in enumerate(generation_df.iterrows()):
            parts.append(f'<circle cx="{px(i):.1f}" cy="{py(row[column]):.1f}" r="6" fill="{color}"><title>Gen {int(row.generation)} {label} IC {row[column]:.5f}</title></circle>')
    for i, (_, row) in enumerate(generation_df.iterrows()):
        parts.append(f'<text x="{px(i):.1f}" y="{height-20}" text-anchor="middle">Gen {int(row.generation)}</text>')
    parts.append('</svg>')
    return "".join(parts)


def metric_box(label, value, css=""):
    return f'<div class="metric {css}"><span>{esc(label)}</span><b>{esc(value)}</b></div>'


def node_card(node: TreeNode):
    m = node.metrics
    metrics = "".join([
        metric_box("季度 Pearson IC", fmt(m.get("quarter_ic")), "accent"),
        metric_box("季度覆盖率", fmt(m.get("quarter_coverage"), 1, True)),
        metric_box("日频 Rank IC", fmt(m.get("daily_ic")), "accent"),
        metric_box("日频 ICIR", fmt(m.get("daily_icir"))),
        metric_box("日频覆盖率", fmt(m.get("daily_coverage"), 1, True)),
        metric_box("多空年化收益", fmt(m.get("annual_long_short"), 1, True)),
        metric_box("多空 Sharpe", fmt(m.get("long_short_sharpe"), 2)),
        metric_box("最大回撤", fmt(m.get("max_drawdown"), 4)),
    ])
    label = str(node.token) if isinstance(node.token, str) else node.kind
    note = "该节点具有有效横截面回测。" if np.isfinite(m.get("daily_ic", math.nan)) else "已执行回测，但该节点缺少横截面变化或有效观测不足。"
    grouping = f"组合采用 {m.get('portfolio_groups', 0)} 组"
    return f'''<article class="node-card" id="node-{node.node_id}">
      <div class="node-head"><div><span class="pill {node.kind}">节点 {node.node_id} · {esc(node.kind.upper())}</span><h3>{esc(label)}</h3></div><span class="direction">{esc(m.get("direction", "—"))}</span></div>
      <code>{esc(node.expression)}</code>
      <div class="metrics">{metrics}</div>
      <p class="note">{note} {grouping}；季度有效 IC 期数 {m.get("quarter_valid_ic", 0)}，日频有效 IC 天数 {m.get("daily_valid_ic", 0)}。</p>
      <div class="charts"><section><h4>日频 Rank IC</h4>{sparkline(node.daily_ic, '#2563eb')}</section><section><h4>累计多空收益</h4>{sparkline(node.cumulative_long_short, '#0f9f6e')}</section></div>
    </article>'''


def build_html(output, manifest, generation_df, candidates, root, nodes, operator_counts):
    best = manifest["best"]
    candidate_rows = []
    for _, row in candidates.head(20).iterrows():
        candidate_rows.append(
            f'<tr><td>Gen {int(row.generation)}</td><td><code>{esc(row.formulation)}</code></td>'
            f'<td>{fmt(row.base_fitness, 5)}</td><td>{fmt(row.IC, 5)}</td><td>{int(row.length)}</td></tr>'
        )
    node_cards = "".join(node_card(node) for node in nodes)
    pipeline = [
        ("01", "财务张量", "36 季 × 503 特征 × 5,392 股票"),
        ("02", "Gen 0", f"随机生成 {manifest['parameters']['population_size']} 棵表达式树"),
        ("03", "逐候选评价", "执行因子 → 覆盖率 → 季度 Pearson IC"),
        ("04", "Gen 1", "锦标赛选择、交叉、变异、深度限制"),
        ("05", "逐节点回测", f"最佳树 {len(nodes)} 个节点全部做季度与日频回测"),
    ]
    flow = "".join(f'<div class="flow-step"><b>{n}</b><h3>{esc(title)}</h3><p>{esc(desc)}</p></div>' for n, title, desc in pipeline)
    css = r'''
    :root{--ink:#172033;--muted:#667085;--paper:#f4f7fb;--card:#fff;--line:#dfe5ee;--blue:#2563eb;--violet:#7c3aed;--green:#0f9f6e;--amber:#d97706;--rose:#e11d48;--shadow:0 14px 40px rgba(15,23,42,.08)}
    *{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.65 Inter,"PingFang SC","Microsoft YaHei",system-ui,sans-serif}.shell{max-width:1500px;margin:auto;padding:0 26px 80px}.hero{margin:0 -26px 28px;padding:58px max(26px,calc((100vw - 1448px)/2));color:#fff;background:radial-gradient(circle at 82% 8%,#7c3aed 0,transparent 30%),linear-gradient(135deg,#0f172a,#173b69 65%,#075985)}.eyebrow{color:#93c5fd;letter-spacing:.16em;font-weight:800}.hero h1{font-size:clamp(34px,5vw,62px);line-height:1.08;margin:.18em 0}.hero p{max-width:980px;color:#dbeafe;font-size:17px}.tags{display:flex;gap:9px;flex-wrap:wrap}.tags span,.pill{padding:5px 10px;border-radius:999px;background:#ffffff18;border:1px solid #ffffff2e;font-size:12px}.nav{position:sticky;top:10px;z-index:10;display:flex;gap:5px;flex-wrap:wrap;background:#ffffffdf;backdrop-filter:blur(14px);border:1px solid var(--line);border-radius:14px;padding:9px;box-shadow:var(--shadow)}.nav a{text-decoration:none;color:#344054;padding:7px 10px;border-radius:9px;font-weight:700}.nav a:hover{background:#eef4ff;color:var(--blue)}h2{font-size:29px;margin:45px 0 8px}h3{margin:3px 0}h4{margin:0 0 6px}.lead{color:var(--muted);max-width:1050px}.panel,.node-card{background:var(--card);border:1px solid var(--line);border-radius:18px;box-shadow:var(--shadow);padding:22px;margin:16px 0}.kpis,.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.metric{background:#f8fafc;border-radius:11px;padding:12px}.metric span{display:block;color:var(--muted);font-size:12px}.metric b{font-size:19px}.metric.accent b{color:var(--blue)}.flow{display:grid;grid-template-columns:repeat(5,1fr);gap:26px;margin:22px 0}.flow-step{position:relative;background:white;border:1px solid #cfdaf0;border-radius:15px;padding:18px}.flow-step:not(:last-child):after{content:'→';position:absolute;right:-21px;top:42%;font-size:23px;color:#8190ac}.flow-step>b{font-size:24px;color:var(--blue)}.flow-step p{color:var(--muted);margin:.4em 0}.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}.tree-wrap{overflow:auto;background:#0f172a;border-radius:16px;padding:12px}.tree{min-width:850px;width:100%;height:auto}.edge{fill:none;stroke:#64748b;stroke-width:2}.tree-node{cursor:pointer}.tree-node text{fill:white;font-size:13px;font-weight:750;pointer-events:none}.tree-node text.sub{font-size:11px;font-weight:500;fill:#e2e8f0}.generation,.bars,.spark{width:100%;height:auto}.generation text,.bar-label,.bar-value{font-size:12px;fill:#667085}.grid,.zero{stroke:#d8dee9;stroke-width:1}.bars rect{fill:url(#none);fill:#2563eb}.formula{display:block;background:#0f172a;color:#dbeafe;padding:17px;border-radius:12px;white-space:normal;overflow-wrap:anywhere}.node-list{display:grid;gap:14px}.node-card{scroll-margin-top:85px}.node-head{display:flex;justify-content:space-between;gap:12px}.pill{color:white;border:0}.pill.cs{background:var(--blue)}.pill.ts{background:var(--violet)}.pill.feature{background:var(--green)}.pill.constant{background:var(--amber)}.direction{color:var(--muted);font-weight:700}.node-card code,td code{display:block;white-space:normal;overflow-wrap:anywhere;background:#f8fafc;border:1px solid #e7ebf1;border-radius:9px;padding:10px;margin:10px 0}.note{color:var(--muted)}.charts{display:grid;grid-template-columns:1fr 1fr;gap:12px}.charts section{background:#f8fafc;border-radius:12px;padding:12px}.empty{height:120px;display:grid;place-items:center;color:var(--muted)}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:10px;border-bottom:1px solid var(--line);vertical-align:top}th{color:#475467;background:#f8fafc}.callout{border-left:5px solid var(--amber);background:#fffbeb;padding:15px 18px;border-radius:10px}.footer{margin-top:45px;border-top:1px solid var(--line);padding-top:18px;color:var(--muted)}
    @media(max-width:1050px){.flow{grid-template-columns:1fr}.flow-step:not(:last-child):after{content:'↓';right:50%;top:auto;bottom:-27px}.two{grid-template-columns:1fr}.kpis,.metrics{grid-template-columns:repeat(2,1fr)}}@media(max-width:700px){.shell{padding:0 13px 50px}.hero{margin:0 -13px;padding:38px 17px}.nav{position:static}.charts{grid-template-columns:1fr}.kpis,.metrics{grid-template-columns:1fr 1fr}.panel,.node-card{padding:15px}}
    '''
    document = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GP 一轮迭代与逐节点回测报告</title><style>{css}</style></head><body><div class="shell">
    <header class="hero"><div class="eyebrow">GENETIC PROGRAMMING · ONE EVOLUTION</div><h1>GP 一轮迭代<br>逐节点可视化报告</h1><p>使用新版 90 算子生产空间，从 Gen 0 到 Gen 1 完成一次遗传演化。最佳表达式中的每个特征叶子和算子节点均被独立视作因子，执行季度 IC 与未来 5 日收益日频回测。</p><div class="tags"><span>运行 {esc(manifest['run_name'])}</span><span>随机种子 {manifest['parameters']['random_state']}</span><span>{manifest['evaluations']} 次候选评价</span><span>{len(nodes)} 个树节点逐一回测</span><span>LLM 关闭</span><span>最大深度 4</span></div></header>
    <nav class="nav"><a href="#summary">结果</a><a href="#architecture">架构</a><a href="#evolution">迭代</a><a href="#tree">表达式树</a><a href="#nodes">逐节点回测</a><a href="#candidates">候选</a><a href="#limits">边界</a></nav>
    <section id="summary"><h2>01 · 本轮结果</h2><p class="lead">“一轮”定义为先生成 Gen 0，再由它产生 Gen 1；没有 warm start，也没有发起外部 LLM 请求。</p><div class="kpis">{metric_box('候选评价',str(manifest['evaluations']))}{metric_box('有效唯一表达式',str(manifest['unique_candidates']))}{metric_box('Gen 1 最优 IC',fmt(best['ic'],5),'accent')}{metric_box('最佳树节点',str(len(nodes)))}{metric_box('最佳树深度',str(best['depth']))}{metric_box('最佳节点日频 IC',fmt(max((n.metrics.get('daily_ic',math.nan) for n in nodes),default=math.nan),4),'accent')}{metric_box('根节点日频 IC',fmt(root.metrics.get('daily_ic'),4))}{metric_box('根节点多空 Sharpe',fmt(root.metrics.get('long_short_sharpe'),2))}</div><div class="panel"><h3>Gen 1 最佳表达式</h3><code class="formula">{esc(best['formula'])}</code></div></section>
    <section id="architecture"><h2>02 · 整体架构</h2><p class="lead">数据从季度财务张量进入表达式搜索，候选先逐节点计算，再做覆盖率和季度 IC；只有适应度参与遗传选择。日频回测不参与训练，只用于本报告的事后检验。</p><div class="flow">{flow}</div><div class="panel two"><div><h3>训练路径</h3><ol><li>叶子读取 503 个财务特征之一。</li><li>CS 节点在同一季度跨股票计算；TS 节点沿季度逐股票计算。</li><li>根节点输出完整因子矩阵。</li><li>应用可交易限制与 90% 覆盖规则。</li><li>按季度计算原始 Pearson IC 并取均值。</li><li>锦标赛选择父代，再执行交叉或变异。</li><li>后代深度超过 4 时回退为父代。</li></ol></div><div><h3>报告路径</h3><ol><li>解析最佳表达式的前缀树。</li><li>缓存每个中间节点的季度矩阵。</li><li>每个节点单独计算季度 Pearson IC。</li><li>按现有 Step 3 日期映射展开到日频。</li><li>计算未来 5 日收益 Rank IC。</li><li>连续信号做十分组；离散节点按可用取值自适应分组。</li><li>保存节点指标、IC 序列和累计收益。</li></ol></div></div></section>
    <section id="evolution"><h2>03 · Gen 0 → Gen 1</h2><div class="panel two"><div><h3>代际 IC</h3>{generation_svg(generation_df)}</div><div><h3>有效候选中的高频算子</h3>{bars_svg(operator_counts.most_common(10))}</div></div><div class="panel"><table><thead><tr><th>代</th><th>候选数</th><th>平均 IC</th><th>最优 IC</th><th>平均节点数</th><th>最优节点数</th></tr></thead><tbody>{''.join(f'<tr><td>Gen {int(r.generation)}</td><td>{int(r.candidates)}</td><td>{r.average_ic:.5f}</td><td>{r.best_ic:.5f}</td><td>{r.average_length:.2f}</td><td>{int(r.best_length)}</td></tr>' for _,r in generation_df.iterrows())}</tbody></table></div></section>
    <section id="tree"><h2>04 · 最佳表达式树</h2><p class="lead">蓝色为 CS、紫色为 TS、绿色为财务特征。节点内显示该子表达式独立回测得到的日频 Rank IC；点击任意节点可跳转到详细回测卡片。</p><div class="tree-wrap">{tree_svg(root,nodes)}</div></section>
    <section id="nodes"><h2>05 · 每个节点的独立回测</h2><p class="lead">父节点使用其完整子树输出；叶子节点直接使用原始财务特征。因而可以观察每一步变换究竟改善还是削弱了 IC、覆盖率与组合表现。</p><div class="node-list">{node_cards}</div></section>
    <section id="candidates"><h2>06 · 本轮候选榜单</h2><div class="panel"><table><thead><tr><th>代际</th><th>表达式</th><th>基础 IC</th><th>最终 IC</th><th>节点数</th></tr></thead><tbody>{''.join(candidate_rows)}</tbody></table></div></section>
    <section id="limits"><h2>07 · 使用边界</h2><div class="callout"><b>这是一轮工程与研究诊断，不是正式因子结论。</b>搜索和节点回测使用同一时间区间；未来 5 日收益存在重叠；多空方向依据全样本均值事后选择；本轮只有 80 个体和一次演化；LLM 可解释性惩罚被关闭。因此报告适合检查新版算子空间、表达式生成和节点贡献，不适合直接据此实盘或比较长期泛化能力。</div></section>
    <footer class="footer">报告生成时间：{esc(manifest['finished_at'])} · 原始数据和节点指标保存在同一运行目录 · HTML 不依赖外部 JavaScript 或网络资源</footer></div></body></html>'''
    output.write_text(document, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "runs" / "output_one_round_v2")
    parser.add_argument("--run-name", default="one_round_operator_v2")
    parser.add_argument("--population", type=int, default=80)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=20260910)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    os.environ["GP_OUTPUT_ROOT"] = str(output_root)

    # Project imports must happen after GP_OUTPUT_ROOT is set because path.py
    # reads it at import time.
    from core_func import GpLearnStock
    from core_func.constant.params import (
        DATES_quar, STOCKS, factor_rate, mutation_functions,
        crossover_functions, eval_start_str, eval_end_str,
    )
    from core_func.core.genetic import all_cal_dictionary
    from core_func.data_reader.data_reader_csv import load_csv_daily
    from core_func.data_reader.data_reader_csv_daily import load_rolling_ret_daily

    started = time.time()
    print("[1/6] loading data and configuring one-round GP", flush=True)
    gp = GpLearnStock(
        population_size=args.population,
        hall_of_fame=min(30, args.population),
        n_components=min(10, args.population),
        generations=1,
        tournament_size=min(12, args.population),
        init_depth=[1, 4],
        warm_start=0,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
        llm_interpretability_enabled=False,
        verbose=0,
    )
    gp.set_params_backtest(need_parallel=args.n_jobs > 1)
    gp.dir_kw = args.run_name
    gp.output_root = str(output_root)
    gp._prep_save_dir()
    gp.llm_interpretability_cache_path = os.path.join(gp.save_dir, "llm_interpretability.sqlite")
    gp._prep_logger()
    gp.prep_data_backtest()

    print("[2/6] evolving Gen 0 -> Gen 1", flush=True)
    summary = gp.fit_3D(
        gp.X, gp.y, need_parallel=args.n_jobs > 1, fitness_threshold=-0.999999
    )
    summary = summary.replace([np.inf, -np.inf], np.nan)
    summary.to_csv(Path(gp.save_dir) / f"raw_result_{args.run_name}.csv", index=False)
    valid = summary[summary["IC"].notna() & (summary["IC"] > -0.99)].copy()
    valid["length"] = valid["formulation_stack"].apply(len)
    valid = valid.sort_values(["generation", "IC"], ascending=[True, False])
    unique = valid.drop_duplicates("formulation", keep="first").reset_index(drop=True)
    unique.insert(0, "factor_name", [f"candidate_{i:04d}" for i in range(len(unique))])
    unique.to_csv(Path(gp.save_dir) / f"original_result_{args.run_name}.csv", index=False)
    unique.sort_values("IC", ascending=False).head(30).to_csv(
        Path(gp.save_dir) / f"result_{args.run_name}.csv", index=False
    )

    generation_rows = []
    for generation, frame in valid.groupby("generation"):
        best_row = frame.loc[frame["IC"].idxmax()]
        generation_rows.append({
            "generation": int(generation),
            "candidates": len(frame),
            "average_ic": float(frame["IC"].mean()),
            "best_ic": float(best_row["IC"]),
            "average_length": float(frame["length"].mean()),
            "best_length": int(best_row["length"]),
        })
    generation_df = pd.DataFrame(generation_rows)
    generation_df.to_csv(Path(gp.save_dir) / "generation_summary.csv", index=False)

    final = valid[valid["generation"] == valid["generation"].max()]
    best_row = final.loc[final["IC"].idxmax()]
    tokens = best_row["formulation_stack"]
    if isinstance(tokens, str):
        tokens = ast.literal_eval(tokens)
    root, nodes = parse_program(tokens, all_cal_dictionary, gp.feature_names, gp.X)
    print(f"[3/6] best expression has {len(nodes)} nodes; loading daily returns", flush=True)

    restrict_df = load_csv_daily("Base", "S_RESTRICT", eval_start_str, eval_end_str)
    restrict_df.index = pd.to_datetime(restrict_df.index)
    rank_ret_df = load_rolling_ret_daily(eval_start_str, eval_end_str, h_hori=5, quantile=True).T
    raw_ret_df = load_rolling_ret_daily(eval_start_str, eval_end_str, h_hori=5, quantile=False).T
    rank_ret_df.index = pd.to_datetime(rank_ret_df.index)
    raw_ret_df.index = pd.to_datetime(raw_ret_df.index)

    print("[4/6] backtesting every expression-tree node", flush=True)
    daily_ic_columns = {}
    cumulative_columns = {}
    for index, node in enumerate(nodes, 1):
        node.metrics.update(quarter_metrics(
            node.value, gp.y, gp.sample_weight, gp.fitness_cache, factor_rate
        ))
        daily_metrics, daily_ic, cumulative = daily_backtest(
            node.value, DATES_quar, STOCKS, restrict_df, rank_ret_df, raw_ret_df
        )
        node.metrics.update(daily_metrics)
        node.daily_ic = daily_ic
        node.cumulative_long_short = cumulative
        daily_ic_columns[f"node_{node.node_id}"] = daily_ic
        cumulative_columns[f"node_{node.node_id}"] = cumulative
        print(f"  node {index}/{len(nodes)}: {node.kind} {str(node.token)[:36]}", flush=True)

    node_rows = []
    for node in nodes:
        node_rows.append({
            "node_id": node.node_id,
            "parent_id": node.parent_id,
            "depth": node.depth,
            "kind": node.kind,
            "token": node.token,
            "expression": node.expression,
            **node.metrics,
        })
    pd.DataFrame(node_rows).to_csv(Path(gp.save_dir) / "node_backtest.csv", index=False)
    pd.concat(daily_ic_columns, axis=1).to_csv(Path(gp.save_dir) / "node_daily_ic.csv")
    pd.concat(cumulative_columns, axis=1).to_csv(Path(gp.save_dir) / "node_cumulative_long_short.csv")

    operator_counts = Counter()
    for stack in valid["formulation_stack"]:
        for token in stack:
            if isinstance(token, str):
                operator_counts[token] += 1

    manifest = {
        "status": "completed",
        "run_name": args.run_name,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": time.time() - started,
        "parameters": {
            "population_size": args.population,
            "generations": 1,
            "warm_start": 0,
            "init_depth": [1, 4],
            "n_jobs": args.n_jobs,
            "random_state": args.random_state,
            "llm_interpretability_enabled": False,
            "operator_count": len(mutation_functions) + len(crossover_functions),
        },
        "evaluations": int(args.population * 2),
        "valid_candidate_rows": int(len(valid)),
        "unique_candidates": int(len(unique)),
        "best": {
            "formula": str(best_row["formulation"]),
            "ic": float(best_row["IC"]),
            "base_fitness": float(best_row["base_fitness"]),
            "generation": int(best_row["generation"]),
            "length": int(best_row["length"]),
            "depth": int(root.depth + max((node.depth for node in nodes), default=0)),
            "node_count": len(nodes),
        },
    }
    # Root depth is the maximum edge depth in the parsed tree.
    manifest["best"]["depth"] = max(node.depth for node in nodes)
    manifest_path = Path(gp.save_dir) / "one_round_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("[5/6] rendering standalone HTML", flush=True)
    report_path = Path(gp.save_dir) / "one_round_architecture_report.html"
    candidates = valid.sort_values("IC", ascending=False)
    build_html(report_path, manifest, generation_df, candidates, root, nodes, operator_counts)

    print("[6/6] complete", flush=True)
    print(json.dumps({
        "run_dir": str(Path(gp.save_dir).resolve()),
        "report": str(report_path.resolve()),
        "best_formula": manifest["best"]["formula"],
        "best_ic": manifest["best"]["ic"],
        "node_count": len(nodes),
        "elapsed_seconds": manifest["elapsed_seconds"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
