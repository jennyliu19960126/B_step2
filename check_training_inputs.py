import os
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[k]="4"
import sys,json,datetime,concurrent.futures,multiprocessing
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent
os.chdir(ROOT);sys.path.insert(0,str(ROOT/"core_func"))
OUT=ROOT/"runs"/("preflight_249_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
OUT.mkdir();os.environ["GP_OUTPUT_ROOT"]=str(OUT)
from constant.params import DATES_quar,STOCKS,feature_names,feature_tensor_path
from data_reader.data_reader_csv import load_feature,load_y_quarter,load_rolling_ret_quarterly
from data_reader.cache_data import CacheData
X=load_feature(feature_names);Y=load_y_quarter();cache=CacheData()
q=pd.DatetimeIndex(DATES_quar);R=cache.restrict;eligible=R==0
meta=json.loads(Path(feature_tensor_path).with_suffix(".metadata.json").read_text())
assert X.shape==(32,249,len(STOCKS)) and Y.shape==R.shape==(32,len(STOCKS))
assert len(set(STOCKS))==len(STOCKS) and len(set(feature_names))==249
assert q.equals(pd.date_range("2017-03-31","2024-12-31",freq=pd.offsets.QuarterEnd()))
frame=load_rolling_ret_quarterly()
assert frame.index.tolist()==STOCKS and frame.columns.tolist()==DATES_quar
assert np.isfinite(X).sum()==np.count_nonzero(~np.isnan(X))
assert np.isfinite(Y[q.month!=12]).all() and np.isnan(Y[q.month==12]).all()
assert eligible.sum(axis=1).min()>0 and cache.eval_mask.sum()==32
coverage=(np.isfinite(X)&eligible[:,None,:]).sum(axis=2)/eligible.sum(axis=1)[:,None]
def check(j):
    a=coverage[:,j]
    return dict(feature=j,source_id=meta["source_mapping"][j]["source_id"],source_group=meta["source_mapping"][j]["source_group"],formula=feature_names[j],mean_coverage=float(a.mean()),min_coverage=float(a.min()),below90_quarters=[str(q[i].date()) for i in np.flatnonzero(a<.9)],all_quarters_pass=bool((a>=.9).all()))
with concurrent.futures.ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context("fork")) as pool:
    rows=list(pool.map(check,range(249)))
pd.DataFrame(rows).to_csv(OUT/"input_coverage_summary.csv",index=False,encoding="utf-8-sig")
pd.DataFrame(coverage,index=DATES_quar,columns=[r["source_id"] for r in rows]).to_csv(OUT/"input_coverage_by_quarter.csv",encoding="utf-8-sig")
report=dict(input_tensor=str(feature_tensor_path),shape=list(X.shape),axes_passed=True,quarter_count=32,possible_ic_quarters=24,eligible_stocks_by_quarter={str(q[i].date()):int(n) for i,n in enumerate(eligible.sum(axis=1))},all_inputs_mean_pass=all(r["mean_coverage"]>=.9 for r in rows),all_inputs_all_quarters_pass=all(r["all_quarters_pass"] for r in rows),minimum_coverage=float(coverage.min()),failed_inputs=[r for r in rows if not r["all_quarters_pass"]])
(OUT/"preflight.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print("OUTPUT",OUT,flush=True)
print(json.dumps({k:v for k,v in report.items() if k not in ["failed_inputs","eligible_stocks_by_quarter"]},ensure_ascii=False),flush=True)
print("FAILURES",json.dumps(report["failed_inputs"][:15],ensure_ascii=False),flush=True)
