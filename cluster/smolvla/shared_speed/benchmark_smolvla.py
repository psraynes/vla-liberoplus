"""Fresh-query A30 benchmark using the team's exact fixed OFT observation.

Use the frozen evaluation adapter, checkpoint processors, dtype and flow steps.
Do not use the old CSV profiling hook: it synchronizes, writes files, and resets
memory peaks inside each prediction. This script does not instantiate a simulator.
"""
import argparse
from collections import Counter
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0 or not np.isfinite(a).all() or (a <= 0).any():
        raise ValueError('Invalid timing samples')
    return dict(median_ms=float(np.median(a)), p95_ms=float(np.percentile(a, 95)),
                mean_ms=float(a.mean()), n=len(a))


def load_observation(path):
    from PIL import Image
    with np.load(path, allow_pickle=False) as sample:
        state = sample['state'].astype(np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError('Expected the OFT sample eight-dimensional robot state')
        instruction = str(sample['task_description'].item())
        images = {}
        for source, dest in [('full_image', 'agentview_image'),
                             ('wrist_image', 'robot0_eye_in_hand_image')]:
            image = sample[source]
            if image.dtype != np.uint8 or image.shape != (224, 224, 3):
                raise ValueError(f'Invalid source image {source}')
            images[dest] = np.asarray(Image.fromarray(image).resize((256, 256), Image.Resampling.BILINEAR))
    # Reconstruct only the adapter's raw observation representation. Its normal
    # image rotation/state construction/preprocessors remain inside timed calls.
    rotvec = state[3:6].astype(np.float64)
    angle = float(np.linalg.norm(rotvec))
    quaternion = np.r_[rotvec * (np.sin(angle / 2) / angle if angle > 1e-12 else 0.5), np.cos(angle / 2)]
    images.update(robot0_eef_pos=state[:3].copy(),
                  robot0_eef_quat=quaternion,
                  robot0_gripper_qpos=state[6:].copy())
    return images, state, instruction


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arm', choices=['base', 'combined_v3'], required=True)
    ap.add_argument('--adapter-dir', type=Path, required=True)
    ap.add_argument('--sample', type=Path, default=Path(__file__).with_name('oft_sample_observation.npz'))
    ap.add_argument('--warmup', type=int, default=50)
    ap.add_argument('--iters', type=int, default=300)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.warmup < 5 or args.iters < 10:
        raise ValueError('Use at least 5 warmup and 10 measured queries')
    if args.out.exists():
        raise FileExistsError(args.out)
    if os.getenv('SMOLVLA_PROFILE_CSV'):
        raise RuntimeError('Unset SMOLVLA_PROFILE_CSV; its profiling hook changes timings and memory peaks')
    if not (args.adapter_dir / 'smolvla_adapter.py').is_file():
        raise FileNotFoundError(args.adapter_dir / 'smolvla_adapter.py')
    model_path = Path(os.environ['SMOLVLA_MODEL'])
    if not (model_path / 'model.safetensors').is_file():
        raise FileNotFoundError(model_path)
    provenance = json.loads(Path(__file__).with_name('sample_provenance.json').read_text())
    if sha256(args.sample) != provenance['npz_sha256']:
        raise RuntimeError('Fixed observation checksum mismatch')

    import torch
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('This benchmark needs exactly one SLURM-assigned CUDA GPU')
    gpu = torch.cuda.get_device_name(0)
    if 'A30' not in gpu:
        raise RuntimeError(f'Expected an A30 for the shared comparison; found {gpu}')
    torch.manual_seed(7)
    np.random.seed(7)
    sys.path.insert(0, str(args.adapter_dir.resolve()))
    from smolvla_adapter import Adapter
    adapter = Adapter()
    cfg = adapter.policy.config
    if cfg.n_action_steps != 10:
        raise RuntimeError('Expected ten executed actions per query, matching our evaluated checkpoints')
    obs, state, instruction = load_observation(args.sample)
    frame = adapter.observation_frame(obs, instruction)
    np.testing.assert_allclose(frame['observation.state'].numpy(), state, atol=2e-5, rtol=2e-5)
    if any(frame[k].shape != (3,256,256) for k in ['observation.images.camera1','observation.images.camera2']):
        raise RuntimeError('Adapter changed camera frame dimensions')

    policy = adapter.policy
    original_sample_actions = policy.model.sample_actions
    counter = {'calls': 0, 'shape': None}
    def counted_sample_actions(*a, **kw):
        counter['calls'] += 1
        result = original_sample_actions(*a, **kw)
        counter['shape'] = list(result.shape)
        return result
    policy.model.sample_actions = counted_sample_actions
    def query():
        before = counter['calls']
        action = adapter.predict(obs, instruction)
        if counter['calls'] != before + 1:
            raise RuntimeError('Query reused queued actions or made multiple model predictions')
        return action

    # The frozen adapter returns the first of ten queued actions; its fresh call
    # computes the full model chunk. Reset outside timing to avoid cached actions.
    for _ in range(args.warmup):
        adapter.reset()
        query()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    raw_e2e = []
    for i in range(args.iters):
        adapter.reset()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        action = query()
        raw_e2e.append((time.perf_counter() - t0) * 1000)
        if i in (0, args.iters - 1):
            print(f'[{args.arm}] fresh end-to-end queries: {i+1}/{args.iters}', flush=True)
    torch.cuda.synchronize()
    generated_shape = counter['shape']
    expected_calls = args.warmup + args.iters
    if counter['calls'] != expected_calls:
        raise RuntimeError(f'Expected {expected_calls} generated chunks; got {counter["calls"]}')
    e2e_allocated = torch.cuda.max_memory_allocated()
    e2e_reserved = torch.cuda.max_memory_reserved()
    policy.model.sample_actions = original_sample_actions
    if action.shape != (1, 7) or generated_shape[:2] != [1, cfg.chunk_size]:
        raise RuntimeError(f'Unexpected adapter/model output shapes: {action.shape}, {generated_shape}')

    # Separate pass, matching the team's distinction between adapter wall time
    # and CUDA-event model-forward time. Never add or subtract these medians.
    pre, core = [], []
    with torch.inference_mode():
        for i in range(args.warmup + args.iters):
            adapter.reset()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            batch = adapter.preprocess(adapter.observation_frame(obs, instruction))
            torch.cuda.synchronize()
            preprocess_ms = (time.perf_counter() - t0) * 1000
            batch = policy._prepare_batch(batch)
            images, masks = policy.prepare_images(batch)
            prepared_state = policy.prepare_state(batch)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            amp = torch.autocast('cuda') if cfg.use_amp else nullcontext()
            with amp:
                start.record()
                pred = original_sample_actions(images, masks, batch[OBS_LANGUAGE_TOKENS],
                                               batch[OBS_LANGUAGE_ATTENTION_MASK], prepared_state)
                end.record()
            torch.cuda.synchronize()
            if i >= args.warmup:
                pre.append(preprocess_ms)
                core.append(start.elapsed_time(end))
    result = {
        'schema_version': 'smolvla-shared-speed-v1', 'arm': args.arm,
        'checkpoint': str(model_path), 'job_id': os.getenv('SLURM_JOB_ID'),
        'node': platform.node(), 'gpu': gpu, 'batch': 1, 'seed': 7,
        'warmup': args.warmup, 'iters': args.iters,
        'end_to_end': stats(raw_e2e), 'preprocess': stats(pre), 'model_forward': stats(core),
        'executed_steps_per_query': cfg.n_action_steps,
        'per_executed_action_ms': stats(raw_e2e)['median_ms'] / cfg.n_action_steps,
        'per_executed_action_mean_ms': stats(raw_e2e)['mean_ms'] / cfg.n_action_steps,
        'predicted_chunk_shape': generated_shape[1:],
        'adapter_returned_action_shape': list(action.shape),
        'fresh_model_calls_verified': counter['calls'],
        'parameters_m': sum(x.numel() for x in policy.parameters()) / 1e6,
        'trainable_parameters_m': sum(x.numel() for x in policy.parameters() if x.requires_grad) / 1e6,
        'peak_gpu_mem_gb': e2e_allocated / 2**30,
        'peak_torch_allocated_mib': e2e_allocated / 2**20,
        'peak_torch_reserved_mib': e2e_reserved / 2**20,
        'peak_all_stages_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
        'memory_definition': 'PyTorch peak allocated including resident model; timed end-to-end pass after warmup. GiB despite legacy *_gb key; not total process/device use.',
        'input': {'sample_sha256': sha256(args.sample), 'source_git_blob':provenance['source_git_blob'],
                  'image_size':256, 'resize':'PIL bilinear once before timing, matching TurboVLA',
                  'rotation':'180 degrees inside frozen adapter', 'state':state.tolist(), 'instruction':instruction},
        'timing_boundaries': {'end_to_end':'Frozen adapter.predict on a fresh chunk; rotation, preprocessing, model, first-action postprocess and CPU copy. Reset and fixture resize excluded.',
                              'model_forward':'SmolVLA sample_actions (vision/language/action generator) CUDA events; separate pass. Not LLM-only.',
                              'preprocess':'Adapter observation_frame + checkpoint preprocessor, including rotation/tokenization/H2D; excludes policy prepare_images/prepare_state.',
                              'per_action':'Amortized query time divided by ten executed actions, not observed control-loop period.',
                              'adapter_queue':'Adapter returns one action and queues nine; fresh query computes configured full chunk. Thin call-count check prevents timing cached actions.'},
        'settings': {k:getattr(cfg,k,None) for k in ('use_amp','num_steps','num_vlm_layers','chunk_size','n_action_steps','device')},
        'parameter_dtypes': dict(Counter(str(x.dtype) for x in policy.parameters())),
        'provenance': {'adapter':str(args.adapter_dir/'smolvla_adapter.py'),
                       'adapter_sha256':sha256(args.adapter_dir/'smolvla_adapter.py'),
                       'benchmark_sha256':sha256(__file__),
                       'model_config_sha256':sha256(model_path/'config.json')},
        'versions': {'torch':torch.__version__, 'cuda':torch.version.cuda, 'python':platform.python_version(),
                     **{k:importlib.metadata.version(k) for k in ('lerobot','transformers','numpy','Pillow')}},
        'raw_latency_ms':raw_e2e, 'raw_preprocess_ms':pre, 'raw_model_forward_ms':core,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as out:
        json.dump(result,out,indent=2,default=str)
        out.write('\n')
    print(f'[{args.arm}] median {result["end_to_end"]["median_ms"]:.2f} ms/query; '
          f'{result["per_executed_action_ms"]:.2f} ms/executed action; '
          f'{result["peak_gpu_mem_gb"]:.3f} GiB allocated; saved {args.out}',flush=True)


if __name__ == '__main__':
    main()
