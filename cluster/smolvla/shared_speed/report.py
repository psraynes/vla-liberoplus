"""CPU-only report; accepts the result folder printed by the completed job."""
import argparse
import csv
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run',type=Path)
    args=ap.parse_args()
    rows=[]
    for arm in ('base','combined_v3'):
        path=args.run/f'{arm}.json'
        if not path.exists():
            print(f'{arm}: not finished yet')
            continue
        r=json.loads(path.read_text())
        if r['fresh_model_calls_verified'] != r['warmup']+r['iters']:
            raise ValueError('Fresh model prediction count did not match queries')
        rows.append({'model':arm,'gpu':r['gpu'],'parameters_m':r['parameters_m'],
            'llm_layers':r['settings']['num_vlm_layers'],
            'query_median_ms':r['end_to_end']['median_ms'],
            'query_mean_ms':r['end_to_end']['mean_ms'],
            'query_p95_ms':r['end_to_end']['p95_ms'],
            'model_forward_median_ms':r['model_forward']['median_ms'],
            'executed_actions':r['executed_steps_per_query'],
            'per_executed_action_median_ms':r['per_executed_action_ms'],
            'peak_torch_allocated_gib':r['peak_gpu_mem_gb'],
            'peak_torch_reserved_mib':r['peak_torch_reserved_mib']})
        print(f"{arm}: {r['gpu']}; median {r['end_to_end']['median_ms']:.2f} ms/query; "
              f"model {r['model_forward']['median_ms']:.2f} ms; "
              f"{r['per_executed_action_ms']:.2f} ms/action; allocated peak {r['peak_gpu_mem_gb']:.3f} GiB")
    if len(rows)!=2:
        return
    report=args.run/'headline_speed.csv'
    with report.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(f'Saved {report}')
    print('GPU memory is PyTorch allocated; model-forward includes vision/language/action generation, not LLM-only.')
    print('These measurements update speed only; they do not make the success-evaluation protocols identical.')


if __name__=='__main__':main()
