"""Prepare the team's fixed MIT-licensed OFT sample without committing data."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import pickle
from urllib.request import urlopen

import numpy as np

COMMIT = 'e4287e94541f459edc4feabc4e181f537cd569a8'
SOURCE_PATH = 'experiments/robot/libero/sample_libero_spatial_observation.pkl'
SHA256 = '326db6c78dd0a9d91c11f05af03b93fa3095338ee3cb5a5eb15adf3d87eb0109'
URL = f'https://raw.githubusercontent.com/moojink/openvla-oft/{COMMIT}/{SOURCE_PATH}'


class ArrayOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {('numpy', 'dtype'): np.dtype,
                   ('numpy.core.numeric', '_frombuffer'): np.core.numeric._frombuffer}
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f'Unexpected global {module}.{name}')
        return allowed[module, name]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=Path('.'))
    parser.add_argument('--source-pickle', type=Path, help='Optional local copy of the pinned source')
    args = parser.parse_args()
    if args.source_pickle:
        raw = args.source_pickle.read_bytes()
    else:
        with urlopen(URL, timeout=60) as response:
            raw = response.read()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError('OFT source SHA-256 mismatch; refusing to decode')
    obs = ArrayOnlyUnpickler(io.BytesIO(raw)).load()
    if set(obs) != {'full_image', 'wrist_image', 'state', 'task_description'}:
        raise ValueError('Unexpected observation fields')
    for key in ('full_image', 'wrist_image'):
        if obs[key].shape != (224, 224, 3) or obs[key].dtype != np.uint8:
            raise ValueError(f'Unexpected {key} shape or dtype')
    if np.asarray(obs['state']).shape != (8,) or not np.isfinite(obs['state']).all():
        raise ValueError('Expected finite eight-dimensional state')
    if not isinstance(obs['task_description'], str):
        raise ValueError('Expected instruction string')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / 'oft_sample_observation.npz'
    np.savez_compressed(target, **obs)
    meta = dict(source_repository='https://github.com/moojink/openvla-oft',
                source_commit=COMMIT, source_path=SOURCE_PATH, source_sha256=SHA256,
                npz_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                source_license='MIT; see LICENSE-openvla-oft-MIT.txt',
                conversion='Lossless arrays and string; no resize or rotation')
    (args.output_dir / 'sample_provenance.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(f'Prepared hash-verified OFT sample: {target}')


if __name__ == '__main__':
    main()
