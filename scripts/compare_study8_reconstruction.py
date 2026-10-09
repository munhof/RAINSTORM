"""Compare recovered filtering/smoothing stages on all train/validation sources.

This compares formulas under the explicit v2 calibration. It does not establish
that the historical native experiment selected the same calibration or trainer.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def compare(prepared_root, historical):
    import numpy as np
    import pandas as pd
    from storm.pipeline import PipelineContext
    from rainstorm_thesis.reconstruction_v2 import ReconstructPoseV2
    from rainstorm_thesis.experiment_plugin import _sha256
    graph = json.loads((prepared_root / 'graph.json').read_text())
    recipe = next(n['config'] for n in graph['nodes'] if n['id'] == 'train_prepare')
    parts = recipe['bodyparts']
    root = historical / 'src/rainstorm/domains/pose/preprocessing'
    modules, source_hashes = {}, {}
    for name in ('refinement', 'filtration'):
        path = root / f'{name}.py'
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[name] = module
        source_hashes[name] = _sha256(path)
    source = next(n['config'] for n in graph['nodes'] if n['id'] == 'source')
    if _sha256(source['manifest_path']) != source['sha256']:
        raise ValueError('Manifest changed')
    records = []
    for session in json.loads(Path(source['manifest_path']).read_text())['sessions']:
        if session['partition'] == 'test':
            continue
        if _sha256(session['path']) != session['sha256']:
            raise ValueError('Pose source changed')
        table = pd.read_hdf(session['path'])
        table.columns = ['_'.join(map(str, c[-2:])) for c in table.columns]
        values = table[[f'{p}_{a}' for p in parts for a in ('x', 'y')]].to_numpy(dtype=np.float32).copy()
        confidence = table[[p+'_likelihood' for p in parts]].to_numpy(dtype=np.float32)
        values[:, 0::2] /= recipe['px_per_cm_x']
        values[:, 1::2] = (recipe['flip_y_height']-values[:, 1::2])/recipe['px_per_cm_y']
        count = len(values)
        context = PipelineContext(data=values, metadata={'units': 'cm', 'sessions': [session['id']]*count,
            'partitions': [session['partition']]*count, 'segments': ['0']*count,
            'frames': table.index.tolist(), 'likelihoods': confidence})
        ReconstructPoseV2().process(context)
        original = values.reshape(count, len(parts), 2).astype(float)
        original[(confidence < .6) | ~np.isfinite(confidence)] = np.nan
        original[~np.isfinite(original)] = np.nan
        jumped = modules['refinement'].jump_threshold_filter(original, max_v=120, fps=30)
        spatial = modules['filtration'].SpatialFiltration(parts, mask_until_full_body=False).apply(jumped)
        expected = modules['refinement'].temporal_smoothing(spatial, med_window=5, sigma=.6)
        actual = np.asarray(context.data).reshape(expected.shape)
        masks = np.isfinite(spatial).all(axis=2)
        mask_differences = int((masks != np.asarray(context.state['reconstruction_v2']['valid_mask'])).sum())
        difference = float(np.max(np.abs(actual-expected)))
        records.append({'session': session['id'], 'partition': session['partition'], 'frames': count,
            'source_sha256': session['sha256'], 'invalid_after_likelihood': context.state['reconstruction_v2']['invalid_after_likelihood'],
            'invalid_after_spatial': int((~masks).sum()), 'mask_differences': mask_differences,
            'max_absolute_coordinate_difference_cm': difference,
            'within_tolerance': bool(np.allclose(actual, expected, atol=1e-6, rtol=1e-6))})
    return {'scope': 'recovered formulas under explicit v2 calibration; not historical experiment equivalence',
        'test_policy': 'test pose files never opened', 'historical_source_sha256': source_hashes,
        'sessions': records, 'compared_sessions': len(records),
        'all_masks_equal': all(r['mask_differences'] == 0 for r in records),
        'all_coordinates_within_tolerance': all(r['within_tolerance'] for r in records)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared-root', type=Path, required=True)
    parser.add_argument('--historical', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.prepared_root, args.historical)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'sessions'}), flush=True)
