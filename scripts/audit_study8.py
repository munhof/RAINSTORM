"""Read-only inventory and pose diagnostics. Never reads held-out test pose data."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3


def audit(workspace, historical):
    import numpy as np
    import pandas as pd
    connection = sqlite3.connect(f'file:{workspace / "studio.sqlite3"}?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    study = dict(connection.execute('select * from storm_studio_study where id=8').fetchone())
    row = connection.execute('select * from storm_studio_datasetrevision where id=?',
                             (study['dataset_revision_id'],)).fetchone()
    config, inventory = json.loads(row['config']), json.loads(row['inventory'])
    partitions = config['session_partitions']
    result = {'study_id': 8, 'dataset_revision_id': row['id'],
        'source_fingerprint': inventory['source_fingerprint'],
        'inventory_session_count': inventory['session_count'],
        'partition_session_counts': dict(Counter(partitions.values())),
        'partition_frame_counts': inventory['partition_counts'],
        'test_policy': 'test assets not opened; existing partition preserved',
        'historical_sources': {}, 'sessions': [],
        'acceptance_status': 'blocked_before_training',
        'blockers': ['Per-session coordinate calibration and Y-axis convention are not yet verified',
                     'Strict v2 preparation of every included session must pass before training'],
        'recovered_parameters': {'likelihood': .6, 'max_velocity_cm_s': 120., 'fps': 30.,
            'dist_near_cm': 4., 'dist_far_cm': 12., 'max_outlier_connections': 3,
            'mask_until_full_body': False, 'interpolation': 'pchip', 'edge_fill': 'ffill/bfill',
            'med_window': 5, 'sigma': .6, 'alignment': 'nose minus body to pi/2',
            'native_offsets': list(range(-9, 11))}}
    for relative in ['src/rainstorm/experiments/study/input_builder.py',
        'src/rainstorm/application/steps/preprocessing/pose_filtration_step.py',
        'src/rainstorm/application/steps/preprocessing/pose_smoothing_step.py',
        'src/rainstorm/domains/pose/preprocessing/refinement.py',
        'src/rainstorm/domains/pose/preprocessing/filtration.py',
        'src/rainstorm/application/steps/ingestion/pose_ingestion_step.py']:
        path = historical / relative
        result['historical_sources'][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    asset_ids = json.loads(row['asset_ids'])
    for asset_id in asset_ids:
        asset = connection.execute('select * from storm_studio_datasetasset where id=?', (asset_id,)).fetchone()
        if asset['role'] != 'pose':
            continue
        session = config.get('asset_sessions', {}).get(str(asset_id), asset['session_id'])
        partition = partitions[session]
        if partition == 'test':
            continue
        path = workspace / asset['relative_path']
        table = pd.read_hdf(path)
        if hasattr(table.columns, 'levels'):
            table.columns = ['_'.join(map(str, c[-2:])) for c in table.columns]
        parts = [name[:-2] for name in inventory['feature_names'] if name.endswith('_x')]
        values = table[[c for p in parts for c in (p+'_x', p+'_y')]].to_numpy().reshape(len(table), -1, 2)
        confidence = table[[p+'_likelihood' for p in parts]].to_numpy()
        invalid = (~np.isfinite(values).all(axis=2)) | (confidence < .6) | ~np.isfinite(confidence)
        nose, body = parts.index('nose'), parts.index('body')
        coincident = (values[:, nose] == values[:, body]).all(axis=1)
        result['sessions'].append({'session': session, 'partition': partition, 'frames': len(values),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'registered_sha256': asset['sha256'], 'missing_or_low_confidence_points': int(invalid.sum()),
            'invalid_by_bodypart': dict(zip(parts, invalid.sum(axis=0).tolist())),
            'raw_coincident_nose_body_frames': int(coincident.sum()),
            'first_coincident_frames': np.flatnonzero(coincident)[:10].tolist(),
            'raw_zero_coordinate_pairs': int((values == 0).all(axis=2).sum())})
    result['audited_sessions'] = len(result['sessions'])
    result['source_hashes_match'] = all(r['sha256'] == r['registered_sha256'] for r in result['sessions'])
    if not result['source_hashes_match']:
        result['blockers'].append('One or more source assets changed')
    connection.close()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--historical', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.workspace, args.historical)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: report[k] for k in ('audited_sessions', 'partition_session_counts',
          'source_hashes_match', 'acceptance_status', 'blockers')}, ensure_ascii=False))
