"""Portable reconstruction checks; no thesis checkout or experimental data needed."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rainstorm_thesis.reconstruction import (
    build_inventory, build_benchmark, atomic_write_json,
)
from rainstorm_thesis.splits import build_experiment_splits

FIXTURE = json.loads((Path(__file__).parent / 'fixtures/historical_inventory.json').read_text())


def test_historical_split_matches_preserved_notebook_outputs():
    paths = [Path(name) for name in FIXTURE['filenames']]
    splits = build_experiment_splits(paths, seed=156)
    for name, count in FIXTURE['expected_sizes'].items():
        ids = [r.session_id for r in getattr(splits, name)]
        assert len(ids) == count
        expected = FIXTURE['visible_notebook_ids'][name]
        if name in FIXTURE['truncated_notebook_splits']:
            assert set(expected) <= set(ids)
        else:
            assert ids == expected
    assert splits == build_experiment_splits(list(reversed(paths)), seed=156)
    train, test, val = [set(r.session_id for r in getattr(splits, n)) for n in
                        ('final_train', 'final_test', 'final_val_internal')]
    assert not (train & test or train & val or test & val)
    assert train | test | val == {p.stem.split('DLC_')[0] for p in paths}
    assert {r.session_id for r in splits.sweep} <= train


def test_duplicate_session_ids_rejected():
    paths = [Path(n) for n in FIXTURE['filenames']]
    with pytest.raises(ValueError, match='Duplicate'):
        build_experiment_splits(paths + [paths[0]])


def test_inventory_hash_changes_when_contents_change(tmp_path):
    pose = tmp_path / 'Hab1DLC_model.h5'
    pose.write_bytes(b'first')
    roi = tmp_path / 'ROIs.json'
    roi.write_text('{}')
    first = build_inventory(tmp_path)
    pose.write_bytes(b'second')
    second = build_inventory(tmp_path)
    assert first['sessions'][0]['session_id'] == 'Hab1'
    assert first['sessions'][0]['pose']['sha256'] != second['sessions'][0]['pose']['sha256']
    assert first['roi'] == second['roi']


def test_benchmark_aligns_by_frame_and_excludes_missing_or_ambiguous_labels(tmp_path):
    labels = tmp_path / 'labels.csv'
    pd.DataFrame({'Frame':[4, 1, 3], 'walk':[0, 1, 1], 'turn':[1, 0, 1]}).to_csv(labels,index=False)
    result = build_benchmark(labels, pose_index=np.arange(5), fps=30, first_frame=1, last_frame=4)
    assert result['csv_frames'] == [1, 2, 3, 4]
    assert result['pose_indices'] == [0, 1, 2, 3]
    assert result['video_frames'] == [0, 1, 2, 3]
    assert result['labels'] == [0, None, None, 1]
    assert result['evaluation_mask'] == [True, False, False, True]
    assert result['video_seconds'] == pytest.approx([0, 1/30, 2/30, 3/30])


@pytest.mark.parametrize('frames', [[1,1], [0,1], [1,6], [1,1.5]])
def test_benchmark_rejects_invalid_frame_mapping(tmp_path, frames):
    labels = tmp_path / 'labels.csv'
    pd.DataFrame({'Frame':frames,'walk':[1,1]}).to_csv(labels,index=False)
    with pytest.raises(ValueError):
        build_benchmark(labels, pose_index=np.arange(5), fps=30, last_frame=4)


def test_benchmark_requires_explicit_supported_pose_index(tmp_path):
    labels = tmp_path / 'labels.csv'
    pd.DataFrame({'Frame':[1], 'walk':[1]}).to_csv(labels,index=False)
    with pytest.raises(ValueError, match='pose index'):
        build_benchmark(labels, pose_index=[3,4,5], fps=30, last_frame=2)


def test_video_timeline_reports_timestamp_gaps_and_unknown_timelines():
    from rainstorm_thesis.reconstruction import inspect_video_timeline

    continuous = inspect_video_timeline(
        [0.0, 1 / 30, 2 / 30], expected_fps=30, expected_frame_count=3)
    assert continuous == {
        'discontinuities': [], 'discontinuity_status': 'verified_contiguous'}

    gap = inspect_video_timeline(
        [0.0, 1 / 30, 3 / 30], expected_fps=30, expected_frame_count=3)
    assert gap['discontinuity_status'] == 'gaps_detected'
    assert gap['discontinuities'] == [{
        'after_frame': 1, 'before_frame': 2, 'delta_seconds': pytest.approx(2 / 30),
    }]

    unknown = inspect_video_timeline([0.0, None, 2 / 30], expected_fps=30,
                                     expected_frame_count=3)
    assert unknown == {
        'discontinuities': None, 'discontinuity_status': 'unknown_requires_review'}


def test_atomic_publish_preserves_previous_manifest_on_serialization_failure(tmp_path):
    target = tmp_path / 'manifest.json'
    atomic_write_json(target, {'revision':1})
    with pytest.raises((TypeError, ValueError)):
        atomic_write_json(target, {'revision':object()})
    assert json.loads(target.read_text()) == {'revision':1}
    assert list(tmp_path.iterdir()) == [target]


def test_manifest_registers_real_supervised_bundles_with_distinct_signatures():
    from rainstorm_thesis.reconstruction import build_supervised_model_references
    models=build_supervised_model_references(Path(__file__).parents[1])
    assert [item['input_shape'] for item in models] == [[None,12],[None,7,12]]
    assert all(len(item['artifact']['sha256']) == 64 for item in models)
    assert all(item['training_population']=='unknown' for item in models)


def test_pose_reference_tracks_contents_and_recipe(tmp_path):
    from rainstorm_thesis.data import PoseDatasetConfig
    pose = tmp_path / 'pose.csv'
    pose.write_text('nose_x,nose_y\n1,2\n')
    first = PoseDatasetConfig(pose, max_frames=1).data_ref()
    pose.write_text('nose_x,nose_y\n3,4\n')
    second = PoseDatasetConfig(pose, max_frames=1).data_ref()
    assert first.fingerprint != second.fingerprint
    assert second.fingerprint != PoseDatasetConfig(pose, max_frames=2).data_ref().fingerprint


def test_loader_preserves_human_categories_frame_alignment_and_mask(tmp_path):
    from rainstorm_thesis.data import PoseDatasetConfig, build_pose_dataset_loader
    pose = tmp_path / 'pose.csv'
    labels = tmp_path / 'labels.csv'
    pd.DataFrame({'nose_x':[1,2,3], 'nose_y':[4,5,6]}).to_csv(pose,index=False)
    pd.DataFrame({'Frame':[3,1], 'walk':[0,1], 'turn':[1,0]}).to_csv(labels,index=False)
    config = PoseDatasetConfig(pose, labels, max_frames=None)
    dataset = build_pose_dataset_loader(config)(config.data_ref())
    assert dataset.targets.iloc[0] == 0
    assert pd.isna(dataset.targets.iloc[1])
    assert dataset.targets.iloc[2] == 1
    assert dataset.metadata['evaluation_mask'] == [True,False,True]
    assert dataset.metadata['taxonomy'] == ['walk','turn']


def test_loader_rejects_changed_source_after_registration(tmp_path):
    from rainstorm_thesis.data import PoseDatasetConfig, build_pose_dataset_loader
    pose = tmp_path / 'pose.csv'
    pose.write_text('nose_x,nose_y\n1,2\n')
    config = PoseDatasetConfig(pose)
    reference = config.data_ref()
    pose.write_text('nose_x,nose_y\n3,4\n')
    with pytest.raises(ValueError, match='changed'):
        build_pose_dataset_loader(config)(reference)


def test_unimplemented_scientific_models_do_not_emit_fabricated_group_ids():
    from rainstorm_thesis.models import ExternalSegmentationModel
    model=ExternalSegmentationModel({'model_family':'vame_native','variant':'pose_ego'})
    with pytest.raises(NotImplementedError,match='pending implementation'):
        model.fit(np.zeros((10,4)))
    with pytest.raises(NotImplementedError,match='pending implementation'):
        model.predict(np.zeros((10,4)))


def test_split_recipe_pins_historical_notebook_content():
    from rainstorm_thesis.reconstruction import HISTORICAL_SPLIT_NOTEBOOK_SHA256
    assert HISTORICAL_SPLIT_NOTEBOOK_SHA256 == FIXTURE['source_sha256']


def test_manifest_reports_implemented_vame_adapter_limits(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import rainstorm_thesis.reconstruction as reconstruction

    manual_dir = tmp_path / 'manual'
    manual_dir.mkdir()
    (manual_dir / 'pose.h5').touch()
    pd.DataFrame({'Frame': [1], 'walk': [1]}).to_csv(manual_dir / 'labels.csv', index=False)
    (manual_dir / 'video.avi').touch()

    monkeypatch.setattr(reconstruction, 'build_inventory', lambda _path: {'sessions': []})
    monkeypatch.setattr(reconstruction, 'build_experiment_splits',
                        lambda _paths: SimpleNamespace(sweep=[], final_train=[],
                                                        final_test=[], final_val_internal=[]))
    monkeypatch.setattr(pd, 'read_hdf', lambda _path: pd.DataFrame(index=np.arange(1)))
    monkeypatch.setattr(reconstruction.subprocess, 'check_output',
                        lambda *_args, **_kwargs: json.dumps({'streams': [
                            {'nb_frames': '1', 'r_frame_rate': '30/1'}]}))
    monkeypatch.setattr(reconstruction, 'file_reference',
                        lambda path: {'path': str(path), 'sha256': '0' * 64, 'size_bytes': 0})
    monkeypatch.setattr(reconstruction, 'build_supervised_model_references', lambda: [])
    monkeypatch.setattr(reconstruction, 'build_benchmark', lambda *_args, **_kwargs: {})

    manifest = reconstruction.reconstruct(tmp_path, manual_dir)

    assert manifest['capabilities']['vame_native'] == (
        'training_saved_run_inference_and_epoch_resume_verified_on_synthetic_data; '
        'pose_ego_two_session_one_epoch_biological_training_artifact_reload_inference_smoke_verified; '
        'historical_62_session_retraining_unverified')
    assert manifest['capabilities']['vame_official'] == (
        'two_session_one_epoch_biological_training_artifact_reload_inference_smoke_verified; '
        'historical_62_session_retraining_unverified')
    for model_name in ('supervised_simple', 'supervised_wide'):
        assert manifest['capabilities'][model_name] == (
            'real_bundle_storm_inference_and_artifact_reload_smoke_verified; '
            'training_population_unknown')


@pytest.mark.skipif(
    not os.environ.get('RAINSTORM_RECONSTRUCTION_ROOT'),
    reason='Set RAINSTORM_RECONSTRUCTION_ROOT to run the local biological data check.',
)
def test_local_data_inventory_and_manual_labels_match_the_historical_recipe():
    """Verify the registered H5/CSV/video path against the supplied study files."""
    from rainstorm_thesis.data import load_dlc_h5
    from rainstorm_thesis.reconstruction import reconstruct

    root = Path(os.environ['RAINSTORM_RECONSTRUCTION_ROOT'])
    data_dir = root / 'rainstorm_analysis_simon'
    manual_dir = root / 'Manual Labels'
    manifest = reconstruct(data_dir, manual_dir)
    splits = manifest['splits']

    assert len(manifest['inventory']['sessions']) == 86
    assert len(manifest['inventory']['roi']['sha256']) == 64
    assert {name: len(ids) for name, ids in splits.items()} == {
        'sweep': 10, 'final_train': 62, 'final_test': 22,
        'final_val_internal': 2,
    }
    for split_name, expected_ids in FIXTURE['visible_notebook_ids'].items():
        assert set(expected_ids) <= set(splits[split_name])
    assert manifest['benchmark']['crop_inclusive'] == [1, 7200]
    assert manifest['benchmark']['pose_frame_count'] == 33000
    assert manifest['benchmark']['video_frame_count'] == 33000
    assert manifest['benchmark']['valid_label_count'] == 5303
    assert manifest['benchmark']['discontinuities'] == []
    assert manifest['benchmark']['discontinuity_status'] == 'verified_contiguous'

    pose_path = manual_dir / 'merge_2DLC_Resnet50_simonSep2shuffle1_snapshot_100.h5'
    video_path = manual_dir / 'merge_2.avi'
    label_path = manual_dir / 'merge_2_labels(3).csv'
    session_id = 'merge_2'
    loaded = load_dlc_h5({
        'pose_paths': [str(pose_path)], 'pose_session_ids': [session_id],
        'video_paths': [str(video_path)], 'labels_path': str(label_path), 'fps': 30,
        'test_session_ids': [session_id], 'frame_start': 0, 'frame_stop': 7200,
        'csv_frame_base': 1, 'pose_frame_base': 0, 'label_frame_reference': 'pose',
    })
    benchmark = manifest['benchmark']
    assert len(loaded['inputs']) == 7200
    assert len(loaded['feature_names']) == 30
    assert loaded['targets'] == benchmark['labels']
    assert loaded['evaluation_mask'] == benchmark['evaluation_mask']
    assert loaded['frames'] == benchmark['pose_indices']
    assert loaded['video_frames'] == benchmark['video_frames']
    assert loaded['source_files'][0]['frame_count'] == 33000
    assert loaded['source_files'][0]['selected_frame_count'] == 7200
    assert loaded['source_files'][0]['video_path'] == str(video_path.resolve())
    assert len(loaded['source_files'][0]['sha256']) == 64
    assert len(loaded['source_files'][0]['video_sha256']) == 64


@pytest.mark.skipif(
    not os.environ.get('RAINSTORM_NATIVE_VAME_BIOLOGICAL_SMOKE')
    or not os.environ.get('RAINSTORM_RECONSTRUCTION_ROOT'),
    reason=('Set RAINSTORM_NATIVE_VAME_BIOLOGICAL_SMOKE=1 and '
            'RAINSTORM_RECONSTRUCTION_ROOT to run the isolated native VAME smoke.'),
)
def test_native_vame_pose_ego_trains_and_reloads_on_biological_h5_sessions(tmp_path):
    """Exercise registration, codeless preparation and saved inference on DLC H5."""
    from rainstorm_thesis.plugin import register
    from storm.suite import default_catalog, execute

    root = Path(os.environ['RAINSTORM_RECONSTRUCTION_ROOT']) / 'rainstorm_analysis_simon'
    pose_paths = [
        root / 'TR0a_2025-08-25T16_48_43DLC_Resnet50_simonSep2shuffle1_snapshot_100.h5',
        root / 'TR0a_2025-08-25T16_55_48DLC_Resnet50_simonSep2shuffle1_snapshot_100.h5',
        root / 'TR0a_2025-08-25T17_09_46DLC_Resnet50_simonSep2shuffle1_snapshot_100.h5',
    ]
    session_ids = ['native-train-a', 'native-train-b', 'native-inference']
    bodyparts = ['nose', 'left_ear', 'right_ear', 'head', 'neck', 'body']
    coordinate_names = [f'{part}_{axis}' for part in bodyparts for axis in ('x', 'y')]
    coordinate_pairs = [[index, index + 1]
                        for index in range(0, len(coordinate_names), 2)]
    body_x, nose_x = coordinate_names.index('body_x'), coordinate_names.index('nose_x')
    steps = [
        {'type': 'pose.select_coordinates', 'config': {'names': coordinate_names}},
        {'type': 'pose.likelihood_filter',
         'config': {'threshold': 0.6, 'coordinate_pairs': coordinate_pairs}},
        {'type': 'pose.recenter',
         'config': {'center_indices': [body_x, body_x + 1],
                    'coordinate_pairs': coordinate_pairs}},
        {'type': 'pose.orient_coordinates', 'config': {
            'reference_pairs': [[body_x, body_x + 1], [nose_x, nose_x + 1]],
            'coordinate_pairs': coordinate_pairs,
            'degenerate_reference_policy': 'identity',
        }},
        {'type': 'pose.temporal_windows',
         'config': {'offsets': [-3, -2, -1, 0, 1, 2, 3]}},
    ]
    model_config = {
        'n_states': 3, 'latent_dim': 3, 'hidden_dim': 8, 'epochs': 1,
        'batch_size': 128, 'seed': 156, 'device': 'cpu', 'num_threads': 1,
    }
    training_spec = {
        'model': 'vame_native', 'connector': 'dlc_h5', 'config': model_config,
        'steps': steps, 'metrics': [], 'seed': 156,
        'data': {
            'pose_paths': [str(path) for path in pose_paths],
            'pose_session_ids': session_ids,
            'bodyparts': bodyparts,
            'train_session_ids': session_ids[:2],
            'test_session_ids': [session_ids[2]],
            'max_frames': 512, 'fps': 30,
        },
    }
    inference_spec = {
        **training_spec, 'operation': 'infer',
        'data': {
            'pose_paths': [str(pose_paths[2])],
            'pose_session_ids': [session_ids[2]],
            'bodyparts': bodyparts,
            'max_frames': 512, 'fps': 30,
        },
    }
    catalog = default_catalog()
    register(catalog)
    artifact_root = tmp_path / 'artifacts'

    trained = execute(training_spec, artifact_root, 'native-biological-training', catalog)
    inferred = execute(
        inference_spec, artifact_root, 'native-biological-inference', catalog,
        inference_from=trained)

    assert trained['model'] == 'vame_native'
    assert trained['output_metadata']['backend'] == 'rainstorm_native_vame'
    assert len(trained['indices']) == 1012
    assert len(trained['fitted_steps']) == len(steps)
    assert trained['output_metadata']['training_history'][0]['epoch'] == 1
    assert inferred['source_execution'] == trained['execution_id']
    assert len(inferred['predictions']) == 506
    assert inferred['model_ref'] == trained['model_ref']


@pytest.mark.skipif(
    not os.environ.get('RAINSTORM_VAME_BIOLOGICAL_SMOKE'),
    reason='Set RAINSTORM_VAME_BIOLOGICAL_SMOKE=1 to run the isolated VAME fit/inference smoke.',
)
def test_official_vame_trains_and_infers_on_biological_h5_sessions(tmp_path, monkeypatch):
    """Train, persist, reload and apply official VAME across biological sessions."""
    from rainstorm_thesis.data import load_dlc_h5
    from rainstorm_thesis.vame_models import VAMEOfficialModel
    from storm.artifacts import FileArtifactStore

    root = Path(os.environ['RAINSTORM_RECONSTRUCTION_ROOT']) / 'rainstorm_analysis_simon'
    train_poses = [
        root / 'TR0a_2025-08-25T16_48_43DLC_Resnet50_simonSep2shuffle1_snapshot_100.h5',
        root / 'TR0a_2025-08-25T16_55_48DLC_Resnet50_simonSep2shuffle1_snapshot_100.h5',
    ]
    infer_pose = root / 'TR0a_2025-08-25T17_09_46DLC_Resnet50_simonSep2shuffle1_snapshot_100.h5'
    monkeypatch.setattr(Path, 'home', classmethod(lambda _cls: tmp_path / 'worker-home'))
    import vame
    from movement.utils.logging import logger

    logger.configure(log_directory=tmp_path / 'logs', console=False)
    training_data = load_dlc_h5({
        'pose_paths': [str(path) for path in train_poses],
        'pose_session_ids': ['registered-train-a', 'registered-train-b'], 'fps': 30,
    })
    training_data['partitions'] = ['train'] * len(training_data['inputs'])
    training_data['reserved_evaluation'] = [False] * len(training_data['inputs'])
    inference_data = load_dlc_h5({
        'pose_paths': [str(infer_pose)], 'pose_session_ids': ['registered-inference'], 'fps': 30,
    })
    model = VAMEOfficialModel({
        'project_name': 'biological_smoke', 'working_directory': str(tmp_path / 'vame'),
        'source_software': 'DeepLabCut', 'fps': 30, 'segmentation_algorithm': 'kmeans',
        'centered_reference_keypoint': 'body', 'orientation_reference_keypoint': 'nose',
        'config_kwargs': {'n_clusters': 2, 'model_snapshot': 1,
            'model_convergence': 10, 'time_window': 19, 'zdims': 2, 'max_epochs': 2,
            'batch_size': 32, 'confidence': 0.6, 'seed': 156, 'learning_rate': 0.0005},
    })
    model.bind_data(training_data, list(range(len(training_data['inputs']))))
    trained = model.fit_predict(training_data['inputs'])
    store = FileArtifactStore(tmp_path / 'artifacts')
    reference = store.save(
        kind='models', artifact_id='official-biological-smoke', value=model)
    recovered = store.load(reference)
    inferred = recovered.predict_with_context(
        inference_data['inputs'], inference_data, list(range(len(inference_data['inputs']))))

    assert trained.metadata['backend'] == 'vame_py_official'
    assert len(trained.predictions) == len(training_data['inputs'])
    assert len(inferred.predictions) == len(inference_data['inputs'])
    assert sum(inferred.metadata['prediction_mask']) > 0
    assert inferred.metadata['discretizer_scope'] == 'shared_training_model'
