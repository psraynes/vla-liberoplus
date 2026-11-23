"""Validate downloaded raw outcomes and export paired SmolVLA comparisons.

Standard library only. No GPU, model downloads, or simulated outcomes.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

ARMS=('base','combined_v3','v3_spatial_repair')
SUITES=('libero_spatial','libero_object','libero_goal','libero_10')
ROOT=Path(__file__).resolve().parent

def rows(path):
    with path.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def plan(path):
    out=[]
    for line in path.read_text().splitlines():
        if not line or line.startswith('#'):continue
        parts=line.split('\t');out.append((parts[1],int(parts[2])))
    if len(set(out))!=len(out):raise ValueError(f'Duplicate plan key: {path}')
    return set(out)

def load_outcomes(path,standard=False):
    result={}
    for r in rows(path):
        key=(r['suite'],int(r['task_id']))
        if standard:key+=int(r['episode']),
        if key in result:raise ValueError(f'Duplicate key in {path.name}: {key}')
        if r['suite'] not in SUITES:raise ValueError(f'Unexpected suite: {r["suite"]}')
        for arm in ARMS:
            if r[arm] not in ('0','1'):raise ValueError(f'Non-binary outcome in {path.name}')
            r[arm]=int(r[arm])
        result[key]=r
    return result

def exact_mcnemar(improved,regressed):
    n=improved+regressed
    if not n:return 1.0
    k=min(improved,regressed)
    if 2*k==n:return 1.0
    # At p=0.5 the two-sided exact binomial test is twice the lower tail.
    logs=[math.lgamma(n+1)-math.lgamma(i+1)-math.lgamma(n-i+1)-n*math.log(2) for i in range(k+1)]
    top=max(logs)
    return min(1.0,2*math.exp(top)*math.fsum(math.exp(x-top) for x in logs))

def compare(data,protocol,group='ALL'):
    output=[];rs=list(data.values());n=len(rs)
    if not n:raise ValueError('Empty comparison')
    for reference in ('base','combined_v3'):
        improved=sum(r['v3_spatial_repair']==1 and r[reference]==0 for r in rs)
        regressed=sum(r['v3_spatial_repair']==0 and r[reference]==1 for r in rs)
        output.append(dict(protocol=protocol,group=group,reference=reference,n=n,
            reference_successes=sum(r[reference] for r in rs),repair_successes=sum(r['v3_spatial_repair'] for r in rs),
            reference_percent=100*sum(r[reference] for r in rs)/n,
            repair_percent=100*sum(r['v3_spatial_repair'] for r in rs)/n,
            change_pp=100*(improved-regressed)/n,
            repair_only_successes=improved,reference_only_successes=regressed,
            both_successes=sum(r['v3_spatial_repair']==r[reference]==1 for r in rs),
            both_failures=sum(r['v3_spatial_repair']==r[reference]==0 for r in rs),
            mcnemar_exact_nominal_p=exact_mcnemar(improved,regressed)))
    return output

def write_csv(path,values):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)

def check_reported(data,filename,group_column=None):
    for record in rows(ROOT/'reported_aggregates'/filename):
        subset=list(data.values())
        if group_column:
            subset=[r for r in subset if r[group_column]==record[group_column]]
        for arm,column in [('base','base_percent'),('combined_v3','v3_percent'),('v3_spatial_repair','v3_spatial_repair_percent')]:
            value=100*sum(r[arm] for r in subset)/len(subset)
            if abs(value-float(record[column]))>0.00501:
                raise ValueError(f'Raw outcomes disagree with supplied report: {filename} {record.get(group_column,"ALL")} {arm} {value:.5f}')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,default=ROOT/'incoming')
    ap.add_argument('--out',type=Path,default=ROOT/'analysis')
    args=ap.parse_args()
    paths={
      'full':args.input/'clean_full_10030_base_v3_spatial_repair_per_task.csv',
      'holdout':args.input/'clean_holdout_4295_base_v3_spatial_repair_per_task.csv',
      'standard':args.input/'standard_base_v3_spatial_repair_per_episode.csv'}
    data={k:load_outcomes(p,standard=k=='standard') for k,p in paths.items()}
    if set(data['full'])!=plan(ROOT/'plans/full_plan_10030.tsv'):raise ValueError('Full task IDs differ from the frozen 10,030 plan')
    if set(data['holdout'])!=plan(ROOT/'plans/paired_plan_4295.tsv'):raise ValueError('Holdout task IDs differ from the frozen 4,295 plan')
    expected={(s,t,e) for s in SUITES for t in range(10) for e in range(50)}
    if set(data['standard'])!=expected:raise ValueError('Standard LIBERO is not exactly 4 x 10 x 50 episodes')
    for key,row in data['holdout'].items():
        if any(row[a]!=data['full'][key][a] for a in ARMS):raise ValueError(f'Holdout/full outcome mismatch: {key}')
    for label,group in data.items():
        print(f'PASS {label}: {len(group)} unique paired outcomes')
    check_reported(data['full'],'full_clean_by_suite_reported.csv','suite')
    check_reported(data['full'],'full_clean_by_category_reported.csv','category')
    check_reported(data['holdout'],'holdout_clean_by_suite_reported.csv','suite')
    check_reported(data['standard'],'standard_by_suite_reported.csv','suite')
    overall=[];suite=[];category=[]
    for label,records in data.items():
        overall+=compare(records,label)
        for name in SUITES:suite+=compare({k:r for k,r in records.items() if r['suite']==name},label,name)
        if label!='standard':
            for name in sorted({r['category'] for r in records.values()}):
                category+=compare({k:r for k,r in records.items() if r['category']==name},label,name)
    # One predeclared family of six overall comparisons; subgroup p values remain exploratory.
    order=sorted(range(len(overall)),key=lambda i:overall[i]['mcnemar_exact_nominal_p'])
    previous=0.0
    for rank,i in enumerate(order):
        previous=max(previous,min(1.0,(len(order)-rank)*overall[i]['mcnemar_exact_nominal_p']))
        overall[i]['holm_adjusted_p_six_overall']=previous
    args.out.mkdir(parents=True,exist_ok=True)
    write_csv(args.out/'paired_overall.csv',overall)
    write_csv(args.out/'paired_suites.csv',suite)
    write_csv(args.out/'paired_categories.csv',category)
    speed=[]
    for arm in ('base','combined_v3'):
        raw=json.loads((args.input/f'{arm}.json').read_text())
        if raw['arm']!=arm or raw['gpu']!='NVIDIA A30' or raw['warmup']!=50 or raw['iters']!=300:raise ValueError('Speed protocol differs from expected shared run')
        if raw['fresh_model_calls_verified']!=350 or len(raw['raw_latency_ms'])!=300:raise ValueError('Speed query count differs')
        if raw['executed_steps_per_query']!=10:raise ValueError('Speed action count differs')
        if abs(raw['per_executed_action_ms']-raw['end_to_end']['median_ms']/10)>1e-8:raise ValueError('Incorrect per-action normalization')
        speed.append(dict(model=arm,query_median_ms=raw['end_to_end']['median_ms'],model_forward_median_ms=raw['model_forward']['median_ms'],
            executed_actions=10,per_action_median_ms=raw['per_executed_action_ms'],peak_allocated_gib=raw['peak_gpu_mem_gb']))
    a=json.loads((args.input/'base.json').read_text());b=json.loads((args.input/'combined_v3.json').read_text())
    if a['input']['sample_sha256']!=b['input']['sample_sha256'] or a['job_id']!=b['job_id'] or str(a['job_id'])!='2568735':raise ValueError('Speed runs did not share the expected input/job')
    write_csv(args.out/'headline_speed_verified.csv',speed)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.input.iterdir()) if p.is_file()}
    (args.out/'input_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    notes=('Validated exact plans, unique pairs, full/holdout agreement, reported suite/category rates and shared speed counts.\n'
      'Nominal exact McNemar p = two-sided Binomial(improved; improved+regressed, p=0.5).\n'
      'Holm correction applies to the six overall comparisons together; subgroup tests are exploratory.\n'
      'Variants share task families and evaluations use one seed/state per variant. These p values are conditional descriptive analyses, not independent-replication evidence or a measure of generalization to new task families.\n'
      'The full 10,030 set includes some teacher source variants; use the 4,295 set for the source-ID-held-out comparison. The frozen set has been inspected repeatedly during development and is not a final blind test.\n'
      'No paired uncertainty is inferred from aggregate-only records. Repaired-model official-instruction scores and shared latency remain unmeasured.\n'
      'Method reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html\n')
    (args.out/'verification_and_methods.txt').write_text(notes)
    for r in overall:
        print(f"{r['protocol']} repair vs {r['reference']}: {r['change_pp']:+.2f} pp; repair-only={r['repair_only_successes']}; reference-only={r['reference_only_successes']}; nominal p={r['mcnemar_exact_nominal_p']:.4g}")
    print(f'Analysis saved: {args.out.resolve()}')

if __name__=='__main__':main()
