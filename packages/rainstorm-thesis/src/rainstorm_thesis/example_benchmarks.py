"""Bundled, inference-only evaluation datasets for RAINSTORM Studio."""
from __future__ import annotations

import json
from pathlib import Path


RAINSTORM_ROOT = Path(__file__).resolve().parents[4]


def nor_test_benchmark() -> dict:
    """Describe the ten labeled NOR test sessions without including train data."""
    source_root = (RAINSTORM_ROOT / 'examples' / 'NOR').resolve(strict=True)
    reference_path = source_root / 'reference.json'
    reference = json.loads(reference_path.read_text(encoding='utf-8'))
    session_ids = sorted(
        session for session in reference.get('files', {})
        if session.startswith('NOR_TS_')
    )
    if len(session_ids) != 10:
        raise ValueError('The bundled NOR benchmark must contain ten TS sessions.')

    assets = []
    label_mapping_by_session = {}
    for session_id in session_ids:
        pose_matches = sorted(source_root.glob(f'{session_id}DLC_*.h5'))
        labels = source_root / 'TS_manual_labels' / f'{session_id}_labels.csv'
        video = source_root / 'videos' / f'{session_id}.mp4'
        if len(pose_matches) != 1 or not labels.is_file() or not video.is_file():
            raise ValueError(f'Incomplete pose, labels, or video for {session_id}.')
        targets = reference['files'][session_id].get('targets', {})
        if set(targets) != {'obj_1', 'obj_2'} or set(targets.values()) != {'Known', 'Novel'}:
            raise ValueError(f'Unsupported NOR target mapping for {session_id}.')
        label_mapping_by_session[session_id] = targets
        assets.extend((
            {'role': 'pose', 'source_path': str(pose_matches[0]),
             'session_id': session_id},
            {'role': 'video', 'source_path': str(video),
             'session_id': session_id},
            {'role': 'labels', 'source_path': str(labels),
             'session_id': session_id},
        ))

    roi = source_root / 'ROIs.json'
    if not roi.is_file():
        raise ValueError('The bundled NOR benchmark has no ROIs.json.')
    assets.extend((
        {'role': 'roi', 'source_path': str(roi), 'session_id': 'global'},
        {'role': 'labels', 'source_path': str(reference_path), 'session_id': 'global'},
    ))
    return {
        'id': 'rainstorm.nor_test.v1',
        'label': 'Benchmark NOR · sesiones de prueba',
        'description': (
            'Diez sesiones NOR-TS con pose H5, video a 25 FPS, ROI y etiquetas '
            'manuales. Solo se permite inferencia; las etiquetas se comparan como '
            'Known/Novel según reference.json.'
        ),
        'source_root': str(source_root),
        'connector': 'dlc_h5',
        'assets': assets,
        'config': {
            'benchmark_preset_id': 'rainstorm.nor_test.v1',
            'inference_only': True,
            'fps': 25.0,
            'hdf_key': '',
            'csv_frame_base': 1,
            'pose_frame_base': 0,
            'label_frame_reference': 'pose',
            'canonical_taxonomy': ['Known', 'Novel'],
            'label_mapping_by_session': label_mapping_by_session,
            'session_partitions': {session: 'test' for session in session_ids},
        },
    }
