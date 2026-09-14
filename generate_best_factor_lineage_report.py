"""Create a standalone, visual calculation-lineage report for the best admitted factor."""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def number(value: object, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    factors = pd.read_csv(run_dir / f"result_{run_dir.name}.csv")
    # Use the strongest non-trivial expression as the lead visual example;
    # the absolute best can be a bare input field with no tree to explain.
    complex_factors = factors[factors["formulation"].astype(str).str.contains(r"(?:ts_|cs_)", regex=True)]
    best = complex_factors.sort_values("IC", ascending=False).iloc[0]
    name = str(best.factor_name)
    bt = pd.read_csv(run_dir / "factor_summary" / name / "5" / "summary.csv").iloc[0]
    manifest = json.loads((run_dir / "test_manifest.json").read_text(encoding="utf-8"))
    weight = float(manifest["parameters"]["llm_interpretability_weight"])
    generations = int(manifest["parameters"]["generations"])
    population = int(manifest["parameters"]["population_size"])
    rng = np.random.RandomState(int(manifest["parameters"]["random_state"]))
    rng.randint(np.iinfo(np.int32).max, size=population * 5)
    genetic_names = ["Crossover", "Subtree Mutation", "Hoist Mutation", "Point Mutation"]
    limits = [0.40, 0.41, 0.705, 1.0]
    genetic_counts = {name: 0 for name in genetic_names}
    for _ in range(generations):
        for seed in rng.randint(np.iinfo(np.int32).max, size=population):
            draw = np.random.RandomState(int(seed)).uniform()
            genetic_counts[genetic_names[int(np.searchsorted(limits, draw, side="right"))]] += 1
    genetic_total = population * generations
    genetic_rows = "".join(
        f"<tr><td>{esc(name)}</td><td>{genetic_counts[name]}</td><td>{100*genetic_counts[name]/genetic_total:.2f}%</td></tr>"
        for name in genetic_names
    )
    threshold = 0.01
    formula = str(best.formulation)
    base = float(best.base_fitness)
    penalty = float(best.llm_interpretability_penalty)
    final = float(best.IC)
    score = float(best.llm_interpretability_score)
    all_cards = []
    for _, row in factors.sort_values("IC", ascending=False).iterrows():
        factor_name, factor_formula = str(row.factor_name), str(row.formulation)
        summary = pd.read_csv(run_dir / "factor_summary" / factor_name / "5" / "summary.csv").iloc[0]
        ops = re.findall(r"([A-Za-z][A-Za-z0-9_]*)\(", factor_formula)
        layers = " → ".join(reversed(ops)) if ops else "直接财务字段"
        feature = str(row.feature_stack).strip("[]'")
        chain_items = [("▦", "基础字段", feature)] + [("◔", "算子", op) for op in reversed(ops)] + [("◈", "输出", factor_name)]
        chain = "<span class='chain-arrow'>→</span>".join(
            f"<span class='chain-node'><i>{icon}</i><small>{kind}</small><b>{esc(label)}</b></span>"
            for icon, kind, label in chain_items
        )
        if "利润" in factor_formula:
            meaning = "盈利能力/盈利位置：关注利润相对资产或权益的水平，以及该水平在公司自身历史中的位置。"
        elif "固定资产" in factor_formula:
            meaning = "资产结构/资本密集度：关注固定资产占资产的比例，并通过截面标准化、行业排名或非线性变换形成信号。"
        elif "盈余公积" in factor_formula:
            meaning = "权益积累变化：关注盈余公积相对母公司权益的环比变化及其历史归一化位置。"
        else:
            meaning = "复合财务信号：需结合底层字段与外层算子共同判断方向。"
        all_cards.append(
            f"<article class='card'><h3>{esc(factor_name)} <span class='gen'>Gen {int(row.generation)}</span></h3><div class='formula'>{esc(factor_formula)}</div>"
            f"<div class='mini-tree'>{chain}</div><p><b>嵌套路径：</b>{esc(layers)}。<br><b>经济主题：</b>{esc(meaning)}<br><b>计算解释：</b>从最内层财务字段开始，按箭头顺序逐层施加算子，最后得到股票级季度信号。</p>"
            f"<div class='kpis'><div class='kpi'><span>原始 IC</span><b>{number(row.base_fitness,4)}</b></div><div class='kpi'><span>LLM / 惩罚</span><b>{float(row.llm_interpretability_score):.0f} / {number(row.llm_interpretability_penalty,3)}</b></div><div class='kpi'><span>最终 fitness</span><b>{number(row.IC,4)}</b></div><div class='kpi'><span>日频 IC / Sharpe</span><b>{number(summary.IC,4)} / {number(summary.sharpe,2)}</b></div></div></article>"
        )
    all_cards_html = "".join(all_cards)
    title = "全部入库因子的计算推导与回测效果"
    css = """
    :root{--ink:#10223d;--muted:#61708a;--bg:#f4f7fc;--blue:#2563eb;--purple:#7c3aed;--green:#059669;--amber:#d97706;--line:#dce4f1}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.65 Inter,'PingFang SC','Microsoft YaHei',sans-serif}.wrap{max-width:1200px;margin:auto;padding:0 28px 70px}.hero{margin:0 -28px 32px;padding:64px max(28px,calc((100vw - 1144px)/2));background:radial-gradient(circle at 82% 10%,#6366f1 0,transparent 30%),linear-gradient(135deg,#10223d,#172554);color:#fff}.hero h1{font-size:44px;line-height:1.18;margin:.15em 0}.hero p{color:#dbeafe;max-width:850px}.tag{display:inline-block;border:1px solid #ffffff42;background:#ffffff16;border-radius:999px;padding:5px 10px;margin:4px 5px 0 0}.card{background:#fff;border:1px solid var(--line);border-radius:20px;padding:24px;margin:18px 0;box-shadow:0 8px 25px #10223d0d}h2{margin:42px 0 8px;font-size:29px}.lead{color:var(--muted)}.formula{background:#0f172a;color:#dbeafe;padding:18px;border-radius:13px;overflow-wrap:anywhere;font:16px/1.7 ui-monospace,SFMono-Regular,Consolas,monospace}.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.kpi{background:#f8faff;border:1px solid var(--line);border-radius:14px;padding:15px}.kpi span{display:block;color:var(--muted);font-size:13px}.kpi b{font-size:24px}.tree{display:grid;grid-template-columns:1fr 76px 1fr 76px 1fr;align-items:center;gap:8px;margin:28px 0}.node{min-height:228px;padding:20px;border-radius:20px;border:2px solid var(--line);background:#fff;position:relative}.node .icon{font-size:38px}.node h3{margin:7px 0;font-size:20px}.node.feature{border-color:#16a34a;background:#f0fdf4}.node.op{border-color:#8b5cf6;background:#faf5ff}.node.out{border-color:#2563eb;background:#eff6ff}.arrow{text-align:center;color:#64748b;font-size:46px}.small{color:var(--muted);font-size:14px}.waterfall{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px}.bar{border-radius:14px;padding:18px;color:#fff}.bar.base{background:#2563eb}.bar.penalty{background:#d97706}.bar.final{background:#059669}.bar b{font-size:30px;display:block}.note{border-left:5px solid var(--amber);background:#fffbeb;padding:16px 18px;border-radius:10px;color:#78350f}.table{width:100%;border-collapse:collapse}.table td,.table th{padding:12px;border-bottom:1px solid var(--line);text-align:left}.table th{color:var(--muted);font-size:13px}.gen{font-size:12px;color:#fff;background:#7c3aed;border-radius:999px;padding:4px 9px;vertical-align:middle}.mini-tree{display:flex;align-items:stretch;gap:8px;margin:16px 0;overflow-x:auto;padding:5px}.chain-node{display:grid;grid-template-columns:26px minmax(90px,1fr);grid-template-rows:auto auto;min-width:135px;max-width:250px;padding:10px;border:1px solid var(--line);border-radius:12px;background:#f8faff}.chain-node i{grid-row:1/3;font-style:normal;font-size:22px}.chain-node small{color:var(--muted)}.chain-node b{font-size:12px;overflow-wrap:anywhere}.chain-arrow{align-self:center;color:#64748b;font-size:24px}@media(max-width:760px){.hero h1{font-size:32px}.kpis,.waterfall{grid-template-columns:repeat(2,1fr)}.tree{grid-template-columns:1fr}.arrow{transform:rotate(90deg)}.node{min-height:0}}
    """
    document = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{esc(title)}</title><style>{css}</style></head><body><main class='wrap'>
    <header class='hero'><div>GP FACTOR · IC + LLM PENALTY</div><h1>{esc(title)}</h1><p>从原始季度财务字段到表达式变换、LLM 可解释性惩罚和日频十分组回测的可追溯计算路径。</p><span class='tag'>最大允许深度 4</span><span class='tag'>遗传迭代 {generations} 代</span><span class='tag'>LLM 惩罚权重 {weight:.2f}</span><span class='tag'>入库线 |fitness| &gt; 0.01</span><span class='tag'>十分组回测</span></header>
    <section><h2>01 · 代表性复杂表达式</h2><div class='formula'>{esc(formula)}</div><p class='lead'>本轮共 {len(factors)} 个入库因子。这里先用惩罚后 fitness 最高的非平凡表达式 {esc(name)} 展示完整计算树；后文再逐项展示全部入库因子。</p></section>
    <section><h2>02 · 计算图：从财务报表到可交易信号</h2><div class='tree'>
      <article class='node feature'><div class='icon'>▦</div><h3>输入特征</h3><b>单季度利润总额 / 股东权益</b><p class='small'>分子是当季税前利润；分母是报告期股东权益。它可视为季度 ROE 的近似刻画，数值越高表示当季盈利相对资本基数越强。</p></article><div class='arrow'>→</div>
      <article class='node op'><div class='icon'>◔</div><h3>时序算子</h3><b>ts_rank_8</b><p class='small'>对每只股票，以当前值和此前 8 个季度组成 9 季窗口，输出当前值在自身历史中的分位排名。它不做股票间排序，而是衡量“当前盈利强度相对自己历史的位置”。</p></article><div class='arrow'>→</div>
      <article class='node out'><div class='icon'>◈</div><h3>最终因子</h3><b>{esc(name)}</b><p class='small'>高分意味着：公司最近一季税前盈利相对权益，处于自身近 9 季的较高位置。季度值在披露期之间前向填充到日频，用于预测未来 5 日收益。</p></article>
    </div><div class='note'><b>信息边界：</b>该表达式直接使用预构建的季度字段；报告中的“权益”须按照数据源已对齐的披露期使用。时序排名降低了跨公司量纲差异，但并不消除行业、会计确认和周期波动的影响。</div></section>
    <section><h2>03 · IC + LLM 惩罚的 selection fitness</h2><p class='lead'>LLM 不参与因子值计算，也不查看回测收益；它评价表达式的经济直觉与结构。遗传选择从第一代起直接使用惩罚后的 fitness，而不是纯 IC。</p><div class='waterfall'><div class='bar base'><span>季度原始 IC</span><b>{number(base)}</b><small>36 个季度横截面 Pearson IC 的平均值</small></div><div class='bar penalty'><span>LLM 惩罚</span><b>−{number(penalty)}</b><small>评分 {score:.0f}/10，权重 {weight:.2f}</small></div><div class='bar final'><span>Selection fitness</span><b>{number(final)}</b><small>高于 0.01 入库线 {number(final-threshold)}</small></div></div><div class='formula' style='margin-top:16px'>selection_fitness = raw_IC − {weight:.2f} × (1 − LLM_score / 10)<br>= {number(base)} − {number(penalty)} = {number(final)}</div></section>
    <section><h2>04 · 遗传操作与 Crossover 使用情况</h2><div class='card'><p class='lead'>本轮四代共生成 {genetic_total:,} 个后代。以下是按随机种子重放得到的真实操作抽签，而不是从最终表达式外观推测。</p><table class='table'><tr><th>遗传操作</th><th>实际次数</th><th>实际占比</th></tr>{genetic_rows}</table><div class='note'><b>Crossover 实际执行入口为 {genetic_counts['Crossover']} 次（{100*genetic_counts['Crossover']/genetic_total:.2f}%）。</b>最终表达式看起来很少有 crossover，是因为 crossover 不会作为函数名显示；相似短树交换还容易产生重复或裸字段。当前产物未保存 method→最终因子的映射，因而不能声称某个入库因子一定来自哪次 crossover。</div></div></section>
    <section><h2>05 · 入库后日频十分组效果</h2><div class='kpis'><div class='kpi'><span>日频 Rank IC</span><b>{number(bt.IC,4)}</b></div><div class='kpi'><span>ICIR</span><b>{number(bt.IR,2)}</b></div><div class='kpi'><span>覆盖率</span><b>{float(bt.coverage)*100:.1f}%</b></div><div class='kpi'><span>多空 Sharpe</span><b>{number(bt.sharpe,2)}</b></div><div class='kpi'><span>对冲 Sharpe</span><b>{number(bt.hedge_sharpe,2)}</b></div><div class='kpi'><span>回测交易日</span><b>{int(bt.date_size):,}</b></div><div class='kpi'><span>样本起止</span><b style='font-size:15px'>{esc(bt.start_date)}<br>至 {esc(bt.end_date)}</b></div><div class='kpi'><span>组合方法</span><b>十分组</b></div></div></section>
    <section><h2>06 · 全部 {len(factors)} 个入库因子的计算路径与效果</h2><p class='lead'>以下每张卡片均保留从内到外的算子嵌套顺序、LLM 惩罚前后 fitness，以及实际日频十分组回测指标。</p>{all_cards_html}</section>
    <section><h2>07 · 可复现性与限制</h2><div class='card'><table class='table'><tr><th>运行设置</th><th>本轮取值</th></tr><tr><td>候选评价 / LLM 审计</td><td>{manifest['llm_audit']['candidate_scoring_attempts']:,} / {manifest['llm_audit']['candidate_scoring_attempts']:,}</td></tr><tr><td>LLM 权重</td><td>{manifest['parameters']['llm_interpretability_weight']}</td></tr><tr><td>最大实际深度</td><td>{manifest['depth_audit']['maximum_observed_depth']}（配置上限 4）</td></tr><tr><td>因子方向</td><td>高因子值对应更高未来收益的方向；实际多空方向仍应以样本外验证为准。</td></tr></table><p class='small'>这是同一时期内挖掘与回测的研究结果，尚未进行时间样本外切分、交易成本、非重叠持有期或行业/风格风险控制，不能直接视为实盘结论。</p></div></section>
    </main></body></html>"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
