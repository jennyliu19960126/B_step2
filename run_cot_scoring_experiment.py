"""Paired thinking ablation on fixed cached expressions; never writes GP caches."""
import argparse
import concurrent.futures
import importlib.util
import json
import math
import os
import random
import sqlite3
import time
import urllib.request
from pathlib import Path
for var in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[var] = '4'
ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('scorer', ROOT/'core_func/utility/llm_interpretability.py')
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)

def call(task):
    expression, repeat, mode, config = task
    payload = dict(model=config['model'], messages=[{'role':'system','content':scorer.SYSTEM_PROMPT},
        {'role':'user','content':f'Factor expression:\n{expression}'}], temperature=0,
        max_tokens=config['max_tokens'], response_format={'type':'json_object'}, thinking={'type':mode})
    key = os.getenv('DEEPSEEK_API_KEY') or json.loads((ROOT/'core_func/constant/config.json').read_text())['llm_interpretability_api_key']
    row = dict(expression=expression, repeat=repeat, mode=mode)
    start = time.perf_counter()
    try:
        req=urllib.request.Request(config['base_url'].rstrip('/')+'/chat/completions',
            data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=config['timeout']) as response:
            data=json.load(response)
        choice=data['choices'][0]; message=choice['message']
        row.update(usage=data.get('usage',{}), actual_model=data.get('model'),
            finish_reason=choice.get('finish_reason'), reasoning_chars=len(message.get('reasoning_content') or ''),
            content=message.get('content'))
        score=float(json.loads(message['content'])['score'])
        if not math.isfinite(score) or not 0 <= score <= 10 or choice.get('finish_reason') != 'stop':
            raise ValueError('invalid or truncated score')
        row['score']=score
    except Exception as exc:
        row['error']=type(exc).__name__
        if hasattr(exc, 'code'): row['http_status']=exc.code
    row['seconds']=time.perf_counter()-start
    return row

def summarize(rows):
    import numpy as np
    from scipy import stats
    result={}
    for mode in ('disabled','enabled'):
        rr=[r for r in rows if r['mode']==mode]
        result[mode]=dict(requests=len(rr),success=sum('score' in r for r in rr),
            mean_seconds=float(np.mean([r['seconds'] for r in rr])),
            p95_seconds=float(np.percentile([r['seconds'] for r in rr],95)),
            completion_tokens=sum(r.get('usage',{}).get('completion_tokens',0) for r in rr),
            reasoning_present=sum(r.get('reasoning_chars',0)>0 for r in rr))
    groups={}
    for r in rows:
        if 'score' in r: groups.setdefault(r['expression'],{}).setdefault(r['mode'],[]).append(r['score'])
    pairs=[(np.mean(g['disabled']),np.mean(g['enabled'])) for g in groups.values() if all(len(g.get(m,[]))==ARGS.repeats for m in ('disabled','enabled'))]
    if pairs:
        a,b=np.array(pairs).T; d=b-a
        rng=np.random.default_rng(20260915)
        boot=np.mean(rng.choice(d,size=(10000,len(d)),replace=True),axis=1)
        result['paired']=dict(expressions=len(d),mean_difference_enabled_minus_disabled=float(d.mean()),
            mean_absolute_difference=float(np.abs(d).mean()),mean_difference_bootstrap_95ci=np.percentile(boot,[2.5,97.5]).tolist(),
            spearman=float(stats.spearmanr(a,b)[0]) if len(d)>2 else None,
            wilcoxon_p=float(stats.wilcoxon(d).pvalue) if np.any(d) else 1.0,
            fraction_difference_at_least_1=float(np.mean(np.abs(d)>=1)),
            mean_absolute_penalty_change_at_weight003=float(np.abs(d).mean()*0.003))
    result['notes']='Exploratory stratified cached sample; significance is not equivalence or scoring accuracy. Both arms use identical token caps. Failed/truncated pairs excluded; inspect failure rates.'
    return result

def main():
    global ARGS
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=int,default=100)
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--max-tokens',type=int,default=8192)
    parser.add_argument('--timeout',type=float,default=180)
    parser.add_argument('--output',type=Path)
    ARGS=parser.parse_args()
    if not 1<=ARGS.workers<=100 or ARGS.samples<1 or ARGS.repeats<1: parser.error('invalid sample/repeat/worker count')
    output=ARGS.output or ROOT/'runs'/('cot_scoring_'+time.strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True,exist_ok=False)
    raw=json.loads((ROOT/'core_func/constant/config.json').read_text())
    config=dict(model=raw['llm_interpretability_model'],base_url=raw['llm_interpretability_base_url'],max_tokens=ARGS.max_tokens,timeout=ARGS.timeout)
    cache=ROOT/'runs/llm_cache/scores.sqlite'
    with sqlite3.connect('file:'+str(cache)+'?mode=ro',uri=True) as conn:
        pool=conn.execute('SELECT expression, AVG(score) FROM llm_interpretability GROUP BY expression ORDER BY expression').fetchall()
    bins={}
    for expr,score in pool: bins.setdefault(int(score),[]).append(expr)
    rng=random.Random(20260915)
    for items in bins.values(): rng.shuffle(items)
    selected=[]
    while len(selected)<min(ARGS.samples,len(pool)):
        for k in sorted(bins):
            if bins[k] and len(selected)<ARGS.samples: selected.append(bins[k].pop())
    manifest=dict(config=config,samples=len(selected),repeats=ARGS.repeats,workers=ARGS.workers,seed=20260915,
        expressions=selected,system_prompt=scorer.SYSTEM_PROMPT,cache=str(cache),sampling='balanced across historical integer-score strata',status='running')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    preflight=[call((selected[0],-1,m,config)) for m in ('disabled','enabled')]
    (output/'preflight.json').write_text(json.dumps(preflight,indent=2),encoding='utf-8')
    if any('score' not in row for row in preflight):
        manifest['status']='preflight_failed'
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        raise SystemExit('Preflight failed; inspect preflight.json before a batch run')
    tasks=[(e,r,m,config) for e in selected for r in range(ARGS.repeats) for m in ('disabled','enabled')]
    rng.shuffle(tasks); rows=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=ARGS.workers) as executor, (output/'responses.jsonl').open('w',encoding='utf-8') as handle:
        for future in concurrent.futures.as_completed([executor.submit(call,t) for t in tasks]):
            row=future.result(); rows.append(row); handle.write(json.dumps(row)+'\n'); handle.flush()
            print(f'{len(rows)}/{len(tasks)} {row["mode"]} {row.get("score",row.get("error"))} {row["seconds"]:.1f}s',flush=True)
    summary=summarize(rows)
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    manifest['status']='completed' if summary.get('paired') else 'failed_no_valid_pairs'
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__': main()
