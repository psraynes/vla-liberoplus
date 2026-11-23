#!/usr/bin/env python3
"""Check collection completeness, clean prompts and held-out/source split separation."""
import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

def audit(root, paired, mode, recipes):
    heldout={(r[1],int(r[2])) for line in paired.read_text().splitlines()
             if line and not line.startswith('#') for r in [line.split('\t')]}
    summary=Counter(); coverage=defaultdict(set); assignments={}; settings=set()
    for recipe in recipes:
        plan_path=Path('spatial_expansion_plans')/f'spatial_{mode}_{recipe}.tsv'
        plan=list(csv.DictReader(plan_path.open(),delimiter='\t'))
        expected={(r['suite'],int(r['task_id'])):r for r in plan}
        assert len(expected)==(15 if mode=='pilot' else 181)
        run=root/f'spatial_repair_v4_{mode}_{recipe}'
        info=json.loads((run/'run_info.json').read_text())
        assert info['instruction_mode']=='clean_v4'
        assert info['plan_sha256']==hashlib.sha256(plan_path.read_bytes()).hexdigest()
        assert info['paired_plan_sha256']==hashlib.sha256(paired.read_bytes()).hexdigest()
        rows=[json.loads(x) for x in (run/'attempts.jsonl').read_text().splitlines() if x.strip()]
        seen=set(); successes=0
        for r in rows:
            key=r['suite'],int(r['task_id'])
            assert key in expected and key not in heldout and key not in seen,key
            seen.add(key); p=expected[key]
            assert 'error' not in r,(key,r.get('error'))
            assert r['instruction_mode']=='clean_v4' and r['seed']==7
            assert r['instruction']==p['base_task_name'].replace('_',' ')
            assert r['split']==p['split']
            assert assignments.setdefault(key,r['split'])==r['split']
            assert r['teacher_checkpoint']==info['teacher_checkpoint']
            assert r['synthetic_task_name']==p['synthetic_task_name']
            if r.get('success') is True:
                ep=(run/r['episode_file']).resolve(strict=True)
                assert ep.is_relative_to(run.resolve())
                assert (r['suite'],r['synthetic_task_name']) not in settings
                settings.add((r['suite'],r['synthetic_task_name']))
                successes+=1; summary[recipe,r['split']]+=1
                coverage[r['split']].add(p['base_task_name'])
        assert seen==set(expected),f'{recipe}: incomplete {len(seen)}/{len(expected)}'
        if mode=='pilot':
            assert successes>=8,f'{recipe}: only {successes}/15 successes; inspect before full collection'
        print(f'{recipe}: {len(rows)} attempted, {successes} successes, zero errors')
    for (recipe,split),count in sorted(summary.items()):print(f'{recipe} / {split}: {count}')
    if mode=='full':
        for split in ('train','validation'):
            assert len(coverage[split])==10,f'{split} has only {len(coverage[split])}/10 Spatial base tasks'
        print('PASS: clean instructions, complete collections, all 10 Spatial tasks in both source-disjoint splits; 4,295 holdout excluded')
    else:print('PASS: clean-instruction pilot passed; full jobs may proceed')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--set',choices=('pilot','full'),required=True)
    p.add_argument('--recipe',choices=('new_view','new_view_blur'))
    a=p.parse_args()
    base=Path(os.environ.get('LP_DIR',f'/scratch/{os.environ["USER"]}/smolvla_turing'))/'teacher_pi05'
    audit(base,Path.home()/'smolvla_turing/eval_kit/paired_plan_4295.tsv',a.set,
          [a.recipe] if a.recipe else ['new_view','new_view_blur'])
