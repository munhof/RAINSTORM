import numpy as np
import pytest
from rainstorm.models import RainstormSupervisedModel


def test_real_adapter_preserves_binary_probability_semantics():
    model = RainstormSupervisedModel({'model_path':'simple.keras', 'output_meaning':'object exploration'},
        predictor=lambda path, values: np.array([[.2],[.8]]))
    result = model.predict(np.zeros((2,12)))
    assert result.predictions == [0,1]
    assert result.metadata['probabilities'] == [.2,.8]
    assert result.metadata['output_meaning'] == 'object exploration'
    assert result.metadata['training_population'] == 'unknown'
    assert result.metadata['capabilities']['retrain'] is False
    assert result.metadata['category_mapping'] == {'0': 'unknown', '1': 'object exploration'}
    assert result.metadata['category_mapping_version'] == '1'


def test_adapter_persists_both_labels_in_a_declared_binary_mapping():
    model = RainstormSupervisedModel({
        'model_path': 'simple.keras', 'output_meaning': 'approach',
        'negative_output_meaning': 'rest', 'category_mapping_version': 'benchmark-v2',
    }, predictor=lambda path, values: np.array([[.2], [.8]]))

    result = model.predict(np.zeros((2, 12)))

    assert result.metadata['category_mapping'] == {'0': 'rest', '1': 'approach'}
    assert result.metadata['category_mapping_version'] == 'benchmark-v2'


@pytest.mark.parametrize('values', [np.zeros((2,11)), np.zeros((2,5,12)), np.full((2,12),np.nan)])
def test_adapter_rejects_wrong_features(values):
    model = RainstormSupervisedModel({'model_path':'simple.keras'}, predictor=lambda *args: [.2,.8])
    with pytest.raises(ValueError):
        model.predict(values)


@pytest.mark.parametrize('probabilities', [[.2], [.2,1.2], [.2,np.nan], [[.2,.3],[.4,.5]]])
def test_adapter_rejects_invalid_probabilities(probabilities):
    model = RainstormSupervisedModel({'model_path':'simple.keras'}, predictor=lambda *args: probabilities)
    with pytest.raises(ValueError):
        model.predict(np.zeros((2,12)))


def test_adapter_does_not_claim_fit_support():
    model = RainstormSupervisedModel({'model_path':'simple.keras'})
    with pytest.raises(NotImplementedError):
        model.fit(np.zeros((2,12)), [0,1])


def test_feature_recipe_retains_original_frames_and_respects_discontinuities():
    import pandas as pd
    from rainstorm.models.features import prepare_supervised_inputs
    data = pd.DataFrame({f'{bp}_{axis}':np.arange(9,dtype=float) + n
                         for n,bp in enumerate(['a','b','c','d','e','f']) for axis in ['x','y']})
    values, centers = prepare_supervised_inputs(data, bodyparts=['a','b','c','d','e','f'],
        frames=list(range(9)), sessions=['s']*9, segments=['one']*4+['two']*5,
        partitions=['benchmark']*9, reserved=[True]*9, offsets=(-1,0,1))
    assert values.shape == (5,3,12)
    assert centers == [1,2,5,6,7]
    np.testing.assert_array_equal(values[0,1], data.iloc[1].to_numpy())


def test_feature_recipe_fails_without_six_explicit_bodyparts():
    import pandas as pd
    from rainstorm.models.features import prepare_supervised_inputs
    with pytest.raises(ValueError, match='six'):
        prepare_supervised_inputs(pd.DataFrame(), bodyparts=[], frames=[],sessions=[],segments=[],
                                 partitions=[],reserved=[], offsets=(0,))


def test_feature_recipe_preserves_historical_rotation_formula():
    import pandas as pd
    from rainstorm.models.features import prepare_supervised_inputs
    bodyparts = ['nose','left_ear','right_ear','head','neck','body']
    data = pd.DataFrame({f'{b}_{axis}':[2. if axis == 'x' else 0.]
                         for b in bodyparts for axis in ('x', 'y')})
    data['body_x'] = 0.
    values, centers = prepare_supervised_inputs(data, bodyparts=bodyparts,
        frames=[0], sessions=['s'], segments=['a'], partitions=['train'], reserved=[False],
        offsets=(0,), center='body', orientation=('body','nose'))
    np.testing.assert_allclose(values[0,:2], [-np.sqrt(2),-np.sqrt(2)], atol=1e-6)
    assert centers == [0]


@pytest.mark.parametrize('variant,key', [('simple','X_ts'),('wide','X_ts_wide')])
def test_imported_model_real_inference(variant,key,tmp_path):
    import os
    import subprocess
    from pathlib import Path
    runtime = os.environ.get('RAINSTORM_RUNTIME_PYTHON')
    examples = os.environ.get('RAINSTORM_EXAMPLE_ROOT')
    if not runtime or not examples:
        pytest.skip('Set RAINSTORM_RUNTIME_PYTHON and RAINSTORM_EXAMPLE_ROOT for real model checks')
    examples = Path(examples)
    path = examples / 'models' / 'trained_models' / f'example_{variant}.keras'
    features = tmp_path / 'features.npy'
    subprocess.run([runtime,'-c',
        'import h5py,numpy as np,sys; '
        'f=h5py.File(sys.argv[1]); np.save(sys.argv[3],f[sys.argv[2]][:16])',
        str(examples / 'models' / 'splits' / 'split_example.h5'),key,str(features)],check=True)
    model = RainstormSupervisedModel({'model_path':path,'runtime_python':runtime})
    result = model.predict(np.load(features,allow_pickle=False))
    probabilities = np.array(result.metadata['probabilities'])
    assert len(result.predictions) == 16
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert np.ptp(probabilities) > 0

    from rainstorm_thesis.plugin import register
    from storm.suite import default_catalog, execute, infer

    model_name = f'supervised_{variant}'
    catalog = default_catalog()
    register(catalog)
    config = catalog.normalize(model_name, {})
    config.update({'model_path': str(path), 'runtime_python': runtime})
    inputs = np.load(features, allow_pickle=False).tolist()
    execution = execute({
        'operation': 'infer', 'model': model_name, 'connector': 'json_records',
        'config': config, 'steps': [], 'metrics': [],
        'data': {'inputs': inputs, 'train': [], 'test': list(range(len(inputs)))},
    }, tmp_path, f'real-{variant}', catalog)
    recovered_predictions = infer(execution, inputs[:4], tmp_path, catalog)

    assert len(execution['predictions']) == 16
    assert len(recovered_predictions) == 4
    assert execution['model_ref']['kind'] == 'models'
    assert execution['output_metadata']['task'] == 'binary_classification'
    assert execution['output_metadata']['training_population'] == 'unknown'


@pytest.mark.parametrize('variant', ['simple','wide'])
def test_reconstructed_architecture_loads_historical_weights(variant):
    import os
    import subprocess
    from pathlib import Path
    runtime = os.environ.get('RAINSTORM_RUNTIME_PYTHON')
    examples = os.environ.get('RAINSTORM_EXAMPLE_ROOT')
    if not runtime or not examples:
        pytest.skip('Set runtime environment variables to compare reconstructed weights')
    model_path = Path(examples) / 'models' / 'trained_models' / f'example_{variant}.keras'
    source = (
        'import sys,numpy as np,tensorflow as tf; '
        'from rainstorm_supervised.architecture import build_simple,build_wide; '
        'saved=tf.keras.models.load_model(sys.argv[1],compile=False); '
        'model=(build_simple() if sys.argv[2]=="simple" else build_wide()); '
        'model.set_weights(saved.get_weights()); '
        'shape=((3,12) if sys.argv[2]=="simple" else (3,7,12)); '
        'x=np.random.default_rng(4).normal(size=shape).astype(np.float32); '
        'np.testing.assert_allclose(saved.predict(x,verbose=0),model.predict(x,verbose=0),rtol=1e-6,atol=1e-6)'
    )
    subprocess.run([runtime,'-c',source,str(model_path),variant],check=True)
