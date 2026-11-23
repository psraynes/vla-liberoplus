#!/usr/bin/env python3
"""Freeze, convert and verify a clean-instruction dataset balanced by suite frames."""
import argparse
import collections
import csv
import hashlib
import json
import random
import re
from pathlib import Path
from convert_combined_teacher_v3 import load_sources, read_episode, FEATURES, TEACHER
from audit_spatial_clean_v4 import audit
from build_teacher_plan import heldout_pairs

SUITES=('libero_spatial','libero_object','libero_goal','libero_10')
COUNTS={'train':310,'validation':39}

def clean(base):
    if base[0].isupper():
        match=re.search(r'SCENE\d+_',base)
        if match is None:raise ValueError(base)
        base=base[match.end():]
    return base.replace('_',' ')

def balance(rows):
    selected=[]
    for split in ('train','validation'):
        pool=[r for r in rows if r['record']['split']==split]
        spatial=[r for r in pool if r['record']['suite']=='libero_spatial']
        if len(spatial)!=COUNTS[split]:raise ValueError(f'Unexpected Spatial {split} count')
        target=sum(int(r['record']['steps']) for r in spatial)
        for i,suite in enumerate(SUITES):
            candidates=[r for r in pool if r['record']['suite']==suite]
            if len({r['base_task_name'] for r in candidates})!=10:
                raise ValueError(f'{split}/{suite}: expected all ten base tasks')
            if i==0:chosen=candidates
            else:
                rng=random.Random(42+i+(100 if split=='validation' else 0))
                rng.shuffle(candidates);chosen=[];covered=set();remaining=[]
                for r in candidates:
                    if r['base_task_name'] not in covered:
                        chosen.append(r);covered.add(r['base_task_name'])
                    else:remaining.append(r)
                frames=sum(int(r['record']['steps']) for r in chosen)
                for r in remaining:
                    n=int(r['record']['steps'])
                    if abs(target-(frames+n))<abs(target-frames):
                        chosen.append(r);frames+=n
                if abs(frames-target)/target>0.05:
                    raise ValueError(f'{split}/{suite}: cannot balance frames within 5%')
            frames=sum(int(r['record']['steps']) for r in chosen)
            print(f'{split}/{suite}: {len(chosen)} episodes, {frames} frames (target {target})')
            selected.extend(chosen)
    return selected

def prepare(args):
    if args.plan.exists():raise FileExistsError('Frozen plan already exists; inspect it before regenerating')
    audit(args.raw,args.paired,'full',['new_view','new_view_blur'])
    old,_=load_sources(args.raw,args.paired)
    mapping={(r['suite'],int(r['task_id'])):r for r in csv.DictReader(args.instructions.open())}
    if len(mapping)!=10030:raise ValueError('Expected full instruction map')
    rows=[]
    for label,root,record,episode in old:
        base=mapping[record['suite'],int(record['task_id'])]['base_task_name']
        rows.append(dict(recipe=label,root=str(root),record=record,episode=str(episode),
                         base_task_name=base,instruction=clean(base)))
    for recipe in ('new_view','new_view_blur'):
        root=args.raw/f'spatial_repair_v4_full_{recipe}'
        for line in (root/'attempts.jsonl').read_text().splitlines():
            r=json.loads(line)
            if r.get('success') is not True:continue
            base=r['base_task_name'];episode=(root/r['episode_file']).resolve(strict=True)
            rows.append(dict(recipe='spatial_'+recipe,root=str(root.resolve()),record=r,
                             episode=str(episode),base_task_name=base,instruction=clean(base)))
    selected=balance(rows)
    args.plan.write_text(json.dumps(dict(seed=42,instruction_mode='canonical_clean',
        paired_plan_sha256=hashlib.sha256(args.paired.read_bytes()).hexdigest(),
        instruction_map_sha256=hashlib.sha256(args.instructions.read_bytes()).hexdigest(),
        selection='whole episodes; each suite frame total within 5% of Spatial; 10 tasks per suite',
        episodes=selected),indent=2),encoding='utf-8')
    print(f'Frozen {len(selected)} episodes in {args.plan}')

def verify_plan(plan,paired):
    data=json.loads(plan.read_text());heldout=heldout_pairs(paired)
    if data['paired_plan_sha256']!=hashlib.sha256(paired.read_bytes()).hexdigest():raise ValueError('Held-out plan changed')
    assigned={};paths=set()
    for row in data['episodes']:
        r=row['record'];key=r['suite'],int(r['task_id'])
        if key in heldout or r.get('success') is not True:raise ValueError('Invalid or held-out episode')
        if r['teacher_checkpoint']!=TEACHER:raise ValueError('Teacher changed')
        if assigned.setdefault(key,r['split'])!=r['split']:raise ValueError('Source split overlap')
        if row['instruction']!=clean(row['base_task_name']):raise ValueError('Instruction is not clean')
        if row['episode'] in paths:raise ValueError('Duplicate saved episode')
        paths.add(row['episode'])
    for split in ('train','validation'):
        totals={}
        for suite in SUITES:
            rows=[x for x in data['episodes'] if x['record']['split']==split and x['record']['suite']==suite]
            if len({x['base_task_name'] for x in rows})!=10:raise ValueError('Missing task coverage')
            totals[suite]=sum(int(x['record']['steps']) for x in rows)
        if any(abs(v-totals['libero_spatial'])/totals['libero_spatial']>0.05 for v in totals.values()):
            raise ValueError('Frame imbalance exceeds 5%')
    return data

def convert(args):
    data=verify_plan(args.plan,args.paired)
    selected=[r for r in data['episodes'] if r['record']['split']==args.split]
    if args.output.exists():raise FileExistsError(f'Refusing dataset overwrite: {args.output}')
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds=LeRobotDataset.create(repo_id=f'capstone/spatial_balanced_clean_v4_{args.split}',fps=20,
        features=FEATURES,root=args.output,robot_type='panda',use_videos=True)
    with (args.output/'collection_manifest.jsonl').open('w',encoding='utf-8') as target:
        for row in selected:
            r=row['record'];cam1,cam2,state,action,original=read_episode(Path(row['episode']),int(r['steps']))
            if original!=r['instruction']:raise ValueError('Saved raw instruction differs from collection record')
            index=ds.num_episodes
            for i in range(len(action)):
                ds.add_frame({'observation.images.camera1':cam1[i],'observation.images.camera2':cam2[i],
                    'observation.state':state[i],'action':action[i],'task':row['instruction']})
            ds.save_episode()
            target.write(json.dumps(dict(episode_index=index,**row))+'\n');target.flush()
            print(f'Saved {args.split}/{index}: {r["suite"]}/{r["task_id"]}',flush=True)
    ds.finalize()
    verify_dataset(args.output,selected)

def verify_dataset(root,rows):
    info=json.loads((root/'meta/info.json').read_text())
    manifest=[json.loads(x) for x in (root/'collection_manifest.jsonl').read_text().splitlines() if x.strip()]
    if info['total_episodes']!=len(rows) or len(manifest)!=len(rows):raise ValueError('Incomplete dataset')
    if info['total_frames']!=sum(int(x['record']['steps']) for x in rows):raise ValueError('Frame count mismatch')
    for i,(record,row) in enumerate(zip(manifest,rows)):
        if record!=dict(episode_index=i,**row):raise ValueError('Final manifest differs from frozen plan')
    for key,feature in FEATURES.items():
        if any(info['features'][key][f]!=feature[f] for f in ('dtype','shape')):raise ValueError('Feature mismatch')
    print(f'PASS: {root.name}: {len(rows)} episodes; {info["total_frames"]} frames')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','convert','verify'))
    p.add_argument('--plan',type=Path,default=Path('balanced_spatial_v4_plan.json'))
    p.add_argument('--paired',type=Path,required=True);p.add_argument('--raw',type=Path)
    p.add_argument('--instructions',type=Path);p.add_argument('--split',choices=('train','validation'))
    p.add_argument('--output',type=Path);p.add_argument('--dataset-base',type=Path)
    a=p.parse_args()
    if a.mode=='prepare':prepare(a)
    elif a.mode=='convert':convert(a)
    else:
        data=verify_plan(a.plan,a.paired)
        for split in ('train','validation'):
            verify_dataset(a.dataset_base/f'spatial_balanced_clean_v4_{split}',
                [r for r in data['episodes'] if r['record']['split']==split])
