"""Seven real policy rollouts; output is separate from all benchmark result logs."""
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import kitconfig

def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run through verify.sbatch on a compute node.')
    kitconfig.load()
    import imageio
    import numpy as np
    import torch
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from eval_tasks import run_episode, set_seed_everywhere, TASK_MAX_STEPS
    from smolvla_adapter import Adapter

    assert torch.cuda.is_available(), 'CUDA is unavailable'
    assert np.__version__ == '1.26.4'
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=False)
    report = {'job_id': os.environ['SLURM_JOB_ID'], 'device': torch.cuda.get_device_name(),
              'checkpoint': os.environ['SMOLVLA_MODEL'], 'seed': 7,
              'packages': {n: importlib.metadata.version(n) for n in
                           ['lerobot', 'torch', 'transformers', 'mujoco', 'numpy', 'scipy', 'robosuite']},
              'episodes': [], 'status': 'running'}
    path = out / 'verification_report.json'
    path.write_text(json.dumps(report, indent=2))
    tasks = json.loads((kitconfig.KIT / 'smoke_tasks.json').read_text())
    suites = {}
    try:
        adapter = Adapter()
        assert adapter.policy.config.n_action_steps == 10
        for item in tasks:
            suite, task_id = item['suite'], int(item['task_id'])
            if suite not in suites:
                suites[suite] = benchmark.get_benchmark_dict()[suite]()
            task_suite = suites[suite]
            task = task_suite.get_task(task_id)
            set_seed_everywhere(7)
            env = None
            started = time.monotonic()
            try:
                env = OffScreenRenderEnv(
                    bddl_file_name=str(Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file),
                    camera_heights=adapter.image_size, camera_widths=adapter.image_size)
                env.seed(7)
                success, frames = run_episode(env, adapter, task.language,
                    task_suite.get_task_init_states(task_id)[0], TASK_MAX_STEPS[suite])
            finally:
                if env is not None:
                    env.close()
            if not frames:
                raise RuntimeError(f'No frames for {suite}/{task_id}')
            stem = f'{suite}_{task_id}'
            imageio.imwrite(out / f'{stem}_cameras.png', frames[0])
            imageio.mimwrite(out / f'{stem}.mp4', frames, fps=20)
            result = {'suite': suite, 'task_id': task_id, 'category': item['category'],
                      'success': bool(success), 'steps': len(frames),
                      'seconds': round(time.monotonic() - started, 2)}
            report['episodes'].append(result)
            path.write_text(json.dumps(report, indent=2))
            print(result, flush=True)
        report['status'] = 'passed_infrastructure_checks'
        report['note'] = 'Review camera orientation, motion and grasp/release videos. A policy failure is not an infrastructure error.'
        report['peak_allocated_mib'] = torch.cuda.max_memory_allocated() / 1024**2
        report['open_file_handles'] = len(os.listdir('/proc/self/fd'))
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = repr(error)
        raise
    finally:
        path.write_text(json.dumps(report, indent=2))
    print(f'Verification report: {path}', flush=True)

if __name__ == '__main__':
    main()
