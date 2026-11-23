#!/usr/bin/env python3
import collections
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SUITES=('libero_spatial','libero_object','libero_goal','libero_10')
def plan(path):
    return [(r[1],int(r[2]),r[3]) for line in path.read_text().splitlines()
            if line and not line.startswith('#') for r in [line.split('\t')]]
def clean_results():
    results={}
    for w in (0,1):
        run=ROOT/f'results/clean_v4_w{w}'; expected={(s,t) for s,t,_ in plan(ROOT/f'worker{w}.tsv')}
        seen={};errors=collections.Counter()
        for path in sorted(run.glob('results*.jsonl')):
            for line in path.read_text().splitlines():
                if not line.strip():continue
                r=json.loads(line);key=r['suite'],int(r['task_id'])
                if key not in expected:raise ValueError(f'Unexpected result {key}')
                if r.get('hung'):errors[key]+=1
                else:
                    if key in seen:raise ValueError(f'Duplicate result {key}')
                    seen[key]=int(bool(r['success']))
        print(f'Clean worker {w}: {len(seen)}/{len(expected)}; {sum(errors.values())} error records')
        if seen.keys()!=expected:raise ValueError('Clean evaluation incomplete')
        if results.keys() & seen.keys():raise ValueError('Overlapping workers')
        results.update(seen)
    return results
def standard_results(path):
    observed={}
    for line in path.read_text().splitlines():
        if not line.strip():continue
        r=json.loads(line);key=r['suite'],int(r['task_id']),int(r['episode'])
        if key in observed:raise ValueError('Duplicate standard episode')
        observed[key]=int(bool(r['success']))
    expected={(s,t,e) for s in SUITES for t in range(10) for e in range(50)}
    if observed.keys()!=expected:raise ValueError(f'Standard incomplete: {len(observed)}/2000')
    return observed
def main():
    full=plan(ROOT/'paired_plan_4295.tsv')
    if len(full)!=4295 or len({(s,t) for s,t,_ in full})!=4295:raise ValueError('Wrong frozen plan')
    refpath=ROOT.parent/'smolvla_clean_instruction_eval/smolvla_clean_instruction_paired_4295_per_task.csv'
    refs={(r['suite'],int(r['task_id'])):r for r in csv.DictReader(refpath.open())}
    if refs.keys()!={(s,t) for s,t,_ in full}:raise ValueError('Reference pairs differ')
    new=clean_results();rows=[]
    for s,t,c in full:
        ref=refs[s,t]
        if ref['category']!=c:raise ValueError('Category mismatch')
        rows.append(dict(suite=s,task_id=t,category=c,base=int(ref['base_clean_instruction']),
            combined_v3=int(ref['combined_v3_clean_instruction']),spatial_v4=new[s,t]))
    dest=ROOT/'clean_4295_base_v3_v4_per_task.csv'
    with dest.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('\nFINAL clean-instruction 4,295-task holdout. Rates are raw subset rates.')
    for group in (*SUITES,'ALL'):
        cell=[r for r in rows if group=='ALL' or r['suite']==group]
        rates={arm:100*sum(r[arm] for r in cell)/len(cell) for arm in ('base','combined_v3','spatial_v4')}
        print(f'{group}: n={len(cell)} '+ ' '.join(f'{k}={v:.2f}%' for k,v in rates.items())+
              f' v4-base={rates["spatial_v4"]-rates["base"]:+.2f} pp')
    for category in sorted({r['category'] for r in rows}):
        cell=[r for r in rows if r['category']==category]
        print(category,' '.join(f'{arm}={100*sum(r[arm] for r in cell)/len(cell):.2f}%'
                                for arm in ('base','combined_v3','spatial_v4')))
    print(f'Saved {dest}')
    stdpaths={'base':ROOT.parent/'smolvla_measurements/results/standard_full_base/results.jsonl',
              'combined_v3':ROOT.parent/'smolvla_measurements/results/standard_full_combined_v3/results.jsonl',
              'spatial_v4':ROOT/'results/standard_v4/results.jsonl'}
    data={k:standard_results(v) for k,v in stdpaths.items()}
    print('\nFINAL original standard LIBERO: 2,000 episodes per checkpoint')
    for group in (*SUITES,'ALL'):
        keys=[key for key in data['base'] if group=='ALL' or key[0]==group]
        print(group,' '.join(f'{arm}={100*sum(out[k] for k in keys)/len(keys):.2f}%'
                             for arm,out in data.items()))
    with (ROOT/'standard_base_v3_v4_per_episode.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['suite','task_id','episode',*data])
        for key in sorted(data['base']):w.writerow([*key,*(out[key] for out in data.values())])
if __name__=='__main__':main()
