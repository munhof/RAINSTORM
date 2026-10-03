from types import SimpleNamespace
import numpy as np
import pytest


def pose_data():
    rng = np.random.default_rng(42)
    values = rng.normal(size=(24, 6)).tolist()
    for row in values:
        row[2] = row[5] = .9
    return {'inputs': values, 'feature_names': ['nose_x', 'nose_y', 'nose_likelihood',
        'body_x', 'body_y', 'body_likelihood'],
        'sessions': ['a'] * 12 + ['b'] * 12, 'frames': list(range(12)) * 2,
        'reserved_evaluation': [False] * 24}


def test_pose_binding_preserves_sessions_frames_and_confidence():
    from rainstorm_thesis.kpms_models import pose_sequences
    data = pose_data()
    indices = [0, 1, 4, 5, 12, 13]
    bundle = pose_sequences([data['inputs'][i] for i in indices], data, indices, {})
    assert bundle['lengths'] == [2, 2, 2]
    assert bundle['frames'] == [0, 1, 4, 5, 0, 1]
    assert bundle['bodyparts'] == ['nose', 'body']
    assert bundle['coordinates'].shape == (6, 2, 2)
    assert np.array_equal(bundle['confidences'][:, 0], np.asarray(data['inputs'])[indices, 2])


def test_official_pca_precedes_init_and_phases_continue_iteration(tmp_path):
    from rainstorm_thesis.kpms_models import KPMSOfficialModel
    calls = []
    def pca(**kwargs):
        calls.append('pca')
        assert kwargs['Y'] == 'formatted poses'
        return 'fitted PCA'
    def init(**kwargs):
        calls.append('init')
        assert kwargs['pca'] == 'fitted PCA'
        return {'model': 'initial'}
    def fit(**kwargs):
        calls.append(('fit', kwargs['ar_only'], kwargs['start_iter'], kwargs['num_iters']))
        return {'model': 'fitted'}, 'test'
    backend = SimpleNamespace(generate_config=lambda **kw: None,
        load_config=lambda *args: {}, fit_pca=pca, init_model=init, fit_model=fit,
        format_data=lambda **kw: ({'Y': 'formatted poses', 'mask': 'mask'}, 'metadata'),
        apply_model=lambda **kw: {f'sequence_{i}': {'syllable': np.zeros(12, dtype=int)} for i in range(2)})
    model = KPMSOfficialModel({'project_dir': str(tmp_path), 'ar_iters': 2, 'full_iters': 3,
                               'fit_full_model': True}, kpms_module=backend)
    data = pose_data()
    model.bind_data(data, list(range(24)))
    output = model.fit_predict(data['inputs'])
    assert len(output.predictions) == 24
    assert calls == ['pca', 'init', ('fit', True, 0, 2), ('fit', False, 2, 5)]
    assert output.metadata['frames'] == data['frames']


def test_native_kpms_cpu_roundtrip_and_inference():
    pytest.importorskip('sklearn')
    from rainstorm_thesis.kpms_models import KPMSNativeModel
    data = pose_data()
    for row in data['inputs']:
        row[2] = row[5] = 1.0
    model = KPMSNativeModel({'num_states': 2, 'latent_dim': 2, 'ar_order': 1,
                            'num_iters': 1})
    model.bind_data(data, list(range(24)))
    output = model.fit_predict(data['inputs'])
    import pickle
    model = pickle.loads(pickle.dumps(model))
    prediction = model.predict_with_context(data['inputs'], data, list(range(24)))
    assert len(output.predictions) == len(prediction.predictions) == 24
    assert output.metadata['backend'] == 'rainstorm_pca_arhmm'
    assert output.metadata['sequence_lengths'] == [12, 12]
    swapped = {**data, 'feature_names': data['feature_names'][3:] + data['feature_names'][:3]}
    with pytest.raises(ValueError, match='pose schema'):
        model.predict_with_context(data['inputs'], swapped, list(range(24)))


def test_pose_binding_rejects_reserved_training():
    from rainstorm_thesis.kpms_models import KPMSNativeModel
    data = pose_data()
    data['reserved_evaluation'][0] = True
    with pytest.raises(ValueError, match='reserved'):
        KPMSNativeModel({}).bind_data(data, [0, 1])


def test_pose_binding_rejects_invalid_confidence():
    from rainstorm_thesis.kpms_models import pose_sequences
    data = pose_data(); data['inputs'][0][2] = -1
    with pytest.raises(ValueError, match='confidence'):
        pose_sequences(data['inputs'], data, list(range(24)), {})


@pytest.mark.skipif(__import__('importlib').util.find_spec('keypoint_moseq') is None,
                    reason='requires isolated KPMS worker dependencies')
@pytest.mark.parametrize('full_model', [False, True])
def test_official_kpms_real_cpu_fixture(tmp_path, full_model):
    from rainstorm_thesis.kpms_models import KPMSOfficialModel
    data = pose_data()
    for row in data['inputs']:
        row[2] = row[5] = 1.0
    data['feature_names'] += ['tail_x', 'tail_y', 'tail_likelihood']
    for row, xy in zip(data['inputs'], np.random.default_rng(7).normal(size=(24, 2))):
        row.extend([*xy.tolist(), 1.0])
    model = KPMSOfficialModel({'project_dir': str(tmp_path), 'num_states': 2,
        'latent_dim': 2, 'ar_iters': 1, 'inference_iters': 1, 'seed': 42,
        'fit_full_model': full_model, 'full_iters': 1})
    model.bind_data(data, list(range(24)))
    output = model.fit_predict(data['inputs'])
    assert len(output.predictions) == 24
    assert model.pca is not None
    assert output.metadata['frames'] == data['frames']
    import pickle
    restored = pickle.loads(pickle.dumps(model))
    assert len(restored.predict_with_context(data['inputs'], data, list(range(24))).predictions) == 24
