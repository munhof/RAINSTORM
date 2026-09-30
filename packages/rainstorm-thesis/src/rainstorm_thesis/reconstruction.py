"""Content-addressed scientific inventory and explicit benchmark alignment."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import pandas as pd

from .splits import build_experiment_splits, infer_session_type

RAINSTORM_ROOT = Path(__file__).resolve().parents[4]
HISTORICAL_SPLIT_NOTEBOOK_SHA256 = 'f80e193c0f07b80f117d2cfb3ac6b492e6174e8b510ecdd07e5255858ce38109'

def file_reference(path: Path | str) -> dict:
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        before = os.fstat(handle.fileno())
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f'Source changed during hashing: {path}')
    return {'path': str(path), 'sha256': digest.hexdigest(), 'size_bytes': after.st_size}


def build_inventory(data_dir: Path | str) -> dict:
    root = Path(data_dir)
    sessions = []
    ids = set()
    for path in sorted(root.glob('*.h5')):
        sid = path.stem.split('DLC_')[0].rstrip('_')
        if sid in ids:
            raise ValueError(f'Duplicate session ID: {sid}')
        ids.add(sid)
        sessions.append({'session_id': sid, 'session_type': infer_session_type(sid),
                         'pose': file_reference(path)})
    if not sessions:
        raise ValueError('No H5 sessions found')
    return {'schema_version': 1, 'sessions': sessions,
            'roi': file_reference(root / 'ROIs.json')}


def build_supervised_model_references(project_root: Path | str = RAINSTORM_ROOT) -> list[dict]:
    """Hash the two shipped model bundles and record their verified signatures."""
    model_dir = Path(project_root) / 'examples' / 'models' / 'trained_models'
    shapes = {'example_simple.keras': ([None, 12], [None, 1]),
              'example_wide.keras': ([None, 7, 12], [None, 1])}
    references = []
    for name, (inputs, outputs) in shapes.items():
        references.append({'model_id': name.removesuffix('.keras'),
                           'artifact': file_reference(model_dir / name),
                           'input_shape': inputs, 'output_shape': outputs,
                           'task': 'binary_classification',
                           'training_population': 'unknown',
                           'training_recipe': 'not_in_supplied_checkout',
                           'capabilities': {'inference': 'verified_isolated_runtime',
                                            'architecture': 'verified_weight_reconstruction',
                                            'retraining': 'unsupported'}})
    return references


def build_benchmark(label_path: Path | str, *, pose_index, fps: float,
                    first_frame: int = 1, last_frame: int = 7200) -> dict:
    """CSV Frame is one-based; H5 and decoded video positions are zero-based.

    The mapping is accepted only for the verified contiguous H5 convention.
    Ambiguous, absent, and malformed one-hot rows are excluded, never class zero.
    """
    index = np.asarray(pose_index)
    if index.ndim != 1 or not np.array_equal(index, np.arange(len(index))):
        raise ValueError('Unsupported pose index: explicit alignment is required')
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be finite and positive')
    if not 1 <= first_frame <= last_frame <= len(index):
        raise ValueError('Invalid benchmark crop')
    table = pd.read_csv(label_path)
    if 'Frame' not in table or len(table.columns) < 2:
        raise ValueError('Expected Frame and one-hot taxonomy columns')
    frames = pd.to_numeric(table['Frame'], errors='raise')
    if (frames.isna().any() or frames.duplicated().any()
            or not np.isfinite(frames).all() or (frames % 1 != 0).any()
            or not frames.between(1, len(index)).all()):
        raise ValueError('Invalid or duplicate CSV Frame')
    taxonomy = [str(c) for c in table.columns if c != 'Frame']
    table.index = frames.astype(int)
    csv_frames = np.arange(first_frame, last_frame + 1)
    values = table[taxonomy].apply(pd.to_numeric, errors='coerce').reindex(csv_frames).to_numpy()
    valid = np.isin(values, [0, 1]).all(axis=1) & ((values == 1).sum(axis=1) == 1)
    labels = [int(np.argmax(row)) if ok else None for row, ok in zip(values, valid)]
    return {'schema_version': 1, 'taxonomy': taxonomy,
            'alignment': {'csv_frame_base': 1, 'pose_index_base': 0,
                          'video_frame_base': 0, 'fps': fps},
            'crop_inclusive': [first_frame, last_frame],
            'csv_frames': csv_frames.tolist(), 'pose_indices': (csv_frames - 1).tolist(),
            'video_frames': (csv_frames - 1).tolist(),
            'video_seconds': ((csv_frames - 1) / fps).tolist(),
            'labels': labels, 'evaluation_mask': valid.tolist(),
            'valid_label_count': int(valid.sum()), 'pose_frame_count': len(index),
            'source_labels': file_reference(label_path),
            'discontinuities': None,
            'discontinuity_status': 'unknown_requires_review'}


def inspect_video_timeline(timestamps, *, expected_fps: float,
                           expected_frame_count: int) -> dict:
    """Mark timestamp gaps in a decoded video stream, or keep the timeline unknown."""
    try:
        rate = float(expected_fps)
    except (TypeError, ValueError):
        return {'discontinuities': None,
                'discontinuity_status': 'unknown_requires_review'}
    if not math.isfinite(rate) or rate <= 0 or not isinstance(timestamps, list):
        return {'discontinuities': None,
                'discontinuity_status': 'unknown_requires_review'}
    if type(expected_frame_count) is not int or expected_frame_count < 1:
        return {'discontinuities': None,
                'discontinuity_status': 'unknown_requires_review'}
    if len(timestamps) != expected_frame_count:
        return {'discontinuities': None,
                'discontinuity_status': 'unknown_requires_review'}
    try:
        seconds = [float(value) for value in timestamps]
    except (TypeError, ValueError):
        return {'discontinuities': None,
                'discontinuity_status': 'unknown_requires_review'}
    if any(not math.isfinite(value) for value in seconds):
        return {'discontinuities': None,
                'discontinuity_status': 'unknown_requires_review'}

    expected_step = 1.0 / rate
    discontinuities = []
    for index, (left, right) in enumerate(zip(seconds, seconds[1:])):
        delta = right - left
        if delta <= 0 or delta > expected_step * 1.5:
            discontinuities.append({
                'after_frame': index,
                'before_frame': index + 1,
                'delta_seconds': round(delta, 9),
            })
    return {
        'discontinuities': discontinuities,
        'discontinuity_status': (
            'gaps_detected' if discontinuities else 'verified_contiguous'),
    }


def atomic_write_json(path: Path | str, payload: dict) -> None:
    """Publish only a complete, fsynced JSON document; preserve the old revision."""
    path = Path(path)
    encoded = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.' + path.name, delete=False) as handle:
            name = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def reconstruct(data_dir: Path, manual_dir: Path) -> dict:
    inventory = build_inventory(data_dir)
    splits = build_experiment_splits([s['pose']['path'] for s in inventory['sessions']])
    split_ids = {name: [r.session_id for r in getattr(splits, name)] for name in
                 ('sweep', 'final_train', 'final_test', 'final_val_internal')}
    pose_paths = list(manual_dir.glob('*.h5'))
    csv_paths = list(manual_dir.glob('*.csv'))
    videos = list(manual_dir.glob('*.avi'))
    if not len(pose_paths) == len(csv_paths) == len(videos) == 1:
        raise ValueError('Manual directory must contain exactly one H5, CSV, and AVI')
    pose = pd.read_hdf(pose_paths[0])
    probe = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames',
        '-show_entries',
        'stream=nb_frames,r_frame_rate:frame=best_effort_timestamp_time',
        '-of', 'json', str(videos[0])], text=True))
    stream = probe['streams'][0]
    numerator, denominator = map(int, stream['r_frame_rate'].split('/'))
    if int(stream['nb_frames']) != len(pose):
        raise ValueError('Pose/video frame counts disagree')
    benchmark = build_benchmark(csv_paths[0], pose_index=pose.index, fps=numerator/denominator)
    timeline = inspect_video_timeline(
        [frame.get('best_effort_timestamp_time') for frame in probe.get('frames', [])],
        expected_fps=numerator / denominator,
        expected_frame_count=int(stream['nb_frames']),
    )
    benchmark.update({'pose': file_reference(pose_paths[0]), 'video': file_reference(videos[0]),
                      'video_frame_count': int(stream['nb_frames']), **timeline})
    return {'schema_version': 1, 'status': 'reconstructed_data_and_supervised_inference',
            'inventory': inventory,
            'supervised_models': build_supervised_model_references(),
            'split_recipe': {'seed': 156, 'sweep_size': 10, 'ordering': 'sorted_filename',
                             'rng': 'numpy.default_rng/PCG64', 'numpy_version': np.__version__,
                             'origin': 'Tesis_Facu/src/experiment/generate_splits.py',
                             'historical_notebook_sha256': HISTORICAL_SPLIT_NOTEBOOK_SHA256},
            'splits': split_ids, 'benchmark': benchmark,
            'capabilities': {
                'vame_native': (
                    'training_saved_run_inference_and_epoch_resume_verified_on_synthetic_data; '
                    'pose_ego_two_session_one_epoch_biological_training_artifact_reload_inference_smoke_verified; '
                    'historical_62_session_retraining_unverified'),
                'vame_official': (
                    'two_session_one_epoch_biological_training_artifact_reload_inference_smoke_verified; '
                    'historical_62_session_retraining_unverified'),
                'supervised_simple': (
                    'real_bundle_storm_inference_and_artifact_reload_smoke_verified; '
                    'training_population_unknown'),
                'supervised_wide': (
                    'real_bundle_storm_inference_and_artifact_reload_smoke_verified; '
                    'training_population_unknown')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--manual-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    atomic_write_json(args.output, reconstruct(args.data_dir, args.manual_dir))


if __name__ == '__main__':
    main()
