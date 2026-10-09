import json
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import pytest
from storm.experiment_nodes import experiment_registry
from storm.experiments import ExperimentExecutor
from storm.suite import default_catalog


def manifest(tmp_path):
    entries = []
    columns = pd.MultiIndex.from_product([['scorer'], ['nose', 'body'], ['x', 'y', 'likelihood']])
    for name, partition in [('a', 'train'), ('b', 'validation'), ('reserved', 'test')]:
        path = tmp_path / f'{name}.h5'
        if partition != 'test':
            values = np.array([[40+i/20, 30., 1., 50+i/20, 35., 1.] for i in range(20)])
            pd.DataFrame(values, columns=columns).to_hdf(path, key='pose')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            digest = 'reserved-never-opened'
        entries.append({'id': name, 'partition': partition, 'path': str(path), 'sha256': digest})
    path = tmp_path / 'sessions.json'
    path.write_text(json.dumps({'version': '1', 'sessions': entries}))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def graph(tmp_path):
    path, digest = manifest(tmp_path)
    nodes = [
        {'id': 'source', 'type': 'data.pose_sessions', 'config': {'manifest_path': str(path), 'sha256': digest}},
        {'id': 'train_prepare', 'type': 'pose.prepare_sessions', 'config': {'bodyparts': ['nose', 'body'],
            'px_per_cm_x': 10., 'px_per_cm_y': 10., 'flip_y_height': 100., 'offsets': [-1, 0, 1]}},
        {'id': 'valid_prepare', 'type': 'pose.prepare_sessions', 'config': {'bodyparts': ['nose', 'body'],
            'px_per_cm_x': 10., 'px_per_cm_y': 10., 'flip_y_height': 100., 'offsets': [-1, 0, 1]}},
        {'id': 'sample', 'type': 'pose.sample_windows', 'config': {'limit': 5, 'seed': 156}}]
    return {'version': '1', 'nodes': nodes, 'edges': [
        {'source': 'source.train', 'target': 'train_prepare.context'},
        {'source': 'source.validation', 'target': 'valid_prepare.context'},
        {'source': 'train_prepare.context', 'target': 'sample.context'}]}


def registry(tmp_path):
    from rainstorm_thesis.experiment_plugin import register_experiment_nodes
    catalog = default_catalog()
    catalog.experiment_artifact_root = tmp_path / 'artifacts'
    register_experiment_nodes(catalog)
    return experiment_registry(catalog)


def test_pose_pipeline_prepares_all_sessions_then_samples_windows_without_test(tmp_path):
    result = ExperimentExecutor(registry(tmp_path)).run(graph(tmp_path))
    assert result.status == 'complete', result.records
    prepared = result.outputs['train_prepare']['context']
    sample = result.outputs['sample']['context']
    assert len(prepared.data) == 18
    assert len(sample.data) == 5
    assert sample.metadata['partition'] == 'train'
    assert sample.metadata['units'] == 'cm'
    assert sample.state['preparation_audit'][0]['frames'] == 20
    assert len(sample.state['window_parents']) == 5
    assert len(sample.state['source_validity_mask']) == 5
    assert np.asarray(sample.state['source_validity_mask']).shape == (5, 1, 2)
    assert np.asarray(sample.data).shape == (5, 3, 4)
    assert sample.artifacts['prepared_sessions'][0]['sha256']
    assert result.outputs['source']['train'].data[0]['id'] == 'a'


def test_source_identity_changes_are_rejected_before_preparation(tmp_path):
    spec = graph(tmp_path)
    spec['nodes'][0]['config']['sha256'] = 'wrong'
    result = ExperimentExecutor(registry(tmp_path)).run(spec)
    assert result.status == 'failed'
    assert 'changed' in result.records['source']['error']
    assert result.records['train_prepare']['status'] == 'blocked'


def test_prepared_windows_are_picklable_and_isolated_without_copying_resources(tmp_path):
    import pickle
    from copy import deepcopy
    result = ExperimentExecutor(registry(tmp_path)).run(graph(tmp_path))
    context = result.outputs['train_prepare']['context']
    restored = pickle.loads(pickle.dumps(context))
    np.testing.assert_array_equal(restored.data[0], context.data[0])
    branch = deepcopy(context)
    with pytest.raises(ValueError):
        branch.data[0][0, 0] = 100
    np.testing.assert_array_equal(restored.data[0], context.data[0])


def test_pose_preview_is_bounded_and_marked_as_sample(tmp_path):
    result = ExperimentExecutor(registry(tmp_path)).run(graph(tmp_path), preview=True)
    assert result.status == 'complete', result.records
    source = result.outputs['source']['train']
    assert source.metadata['preview_scope'] == 'sample_only'
    assert source.data[0]['preview_frames'] == 256
    assert result.outputs['train_prepare']['context'].metadata['preview_scope'] == 'sample_only'


@pytest.mark.parametrize('confidence,expected', [(0., 'complete'), (1., 'partial')])
def test_resolution_check_uses_only_confident_coordinates(tmp_path, confidence, expected):
    spec = graph(tmp_path)
    path = Path(spec['nodes'][0]['config']['manifest_path'])
    entries = json.loads(path.read_text())
    source = entries['sessions'][0]
    table = pd.read_hdf(source['path'])
    table.iloc[5, 1] = 110.
    table.iloc[5, 2] = confidence
    table.to_hdf(source['path'], key='pose', mode='w')
    source['sha256'] = hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()
    path.write_text(json.dumps(entries))
    spec['nodes'][0]['config']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = ExperimentExecutor(registry(tmp_path)).run(spec)
    assert result.status == expected, result.records
    if confidence:
        assert 'verify source resolution' in result.records['train_prepare']['error']


def test_calibration_reference_is_verified_before_preparation(tmp_path):
    spec = graph(tmp_path)
    roi = tmp_path / 'roi.json'
    roi.write_text(json.dumps({'scale': 10., 'frame_shape': [100, 100]}))
    digest = hashlib.sha256(roi.read_bytes()).hexdigest()
    for node in spec['nodes'][1:3]:
        node['config']['calibration_ref'] = {'path': str(roi), 'sha256': digest}
    roi.write_text(json.dumps({'scale': 20., 'frame_shape': [100, 100]}))
    result = ExperimentExecutor(registry(tmp_path)).run(spec)
    assert result.records['train_prepare']['status'] == 'failed'
    assert 'Calibration contents changed' in result.records['train_prepare']['error']


def test_pose_windows_declare_frame_clock_and_keep_times_after_sampling(tmp_path):
    result = ExperimentExecutor(registry(tmp_path)).run(graph(tmp_path))
    sample = result.outputs['sample']['context']
    assert sample.metadata['time_axis'] == {'clock': 'pose_frame', 'unit': 'frame',
        'origin': 'source_frame_id', 'scope': 'session'}
    assert sample.metadata['times'] == sample.metadata['frames']
    assert len(sample.metadata['times']) == len(sample.data)


def test_preview_keeps_raw_pose_aligned_to_prepared_window_anchor(tmp_path):
    result = ExperimentExecutor(registry(tmp_path)).run(graph(tmp_path), preview=True)
    assert result.status == 'complete', result.records
    context = result.outputs['valid_prepare']['context']
    raw = context.state['preview_input_context']
    assert raw['metadata']['observation_ids'] == context.metadata['observation_ids'][:5]
    assert raw['metadata']['units'] == 'px'
    assert raw['rows'][0] == pytest.approx([40.05,30,50.05,35])
    assert context.metadata['feature_names'] == ['nose_x','nose_y','body_x','body_y']
    assert len(raw['rows']) <= 5


def test_frame_loader_preserves_source_frames_and_records_confidence_discards(tmp_path):
    from rainstorm_thesis.experiment_plugin import _frames
    from storm.pipeline import PipelineContext
    path, _ = manifest(tmp_path)
    source = json.loads(path.read_text())['sessions'][1]
    table = pd.read_hdf(source['path'])
    table.iloc[3, 2] = 0.1
    table.to_hdf(source['path'], key='pose', mode='w')
    source['sha256'] = hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()
    context = PipelineContext(data=[source], metadata={'partition': 'validation'})
    result = _frames({'context': context}, {'bodyparts': ['nose', 'body'], 'threshold': .6})['context']
    assert len(result.data) == 19
    assert result.metadata['frames'] == [i for i in range(20) if i != 3]
    assert result.metadata['observation_ids'][3] == 'b:4'
    assert result.metadata['units'] == 'px'
    assert result.state['preparation_audit'][0]['discarded'] == 1
    assert result.artifacts['prepared_sessions'][0]['source_sha256'] == source['sha256']


def test_frame_loader_rejects_partition_mismatch_before_reading():
    from rainstorm_thesis.experiment_plugin import _frames
    from storm.pipeline import PipelineContext
    context = PipelineContext(data=[{'partition': 'test', 'path': '/never-read'}], metadata={'partition': 'validation'})
    with pytest.raises(ValueError, match='partition'):
        _frames({'context': context}, {'bodyparts': ['nose'], 'threshold': .6})
