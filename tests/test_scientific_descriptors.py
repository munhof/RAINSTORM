import json
import subprocess
import sys
import pytest


def test_catalog_declares_vame_stages_and_validates_nested_controls():
    from storm.suite import default_catalog
    from rainstorm_thesis.plugin import register
    catalog = default_catalog()
    register(catalog)
    with pytest.raises(ValueError, match='config_kwargs.max_epochs'):
        catalog.normalize('vame_official', {'config_kwargs': {'max_epochs': 1}})
    with pytest.raises(ValueError, match='trainset_options.keypoints_to_include'):
        catalog.normalize('vame_official', {'trainset_options': {'keypoints_to_include': [7]}})
    config = catalog.normalize('vame_official', {'preprocessing_options': {'run_savgol_filtering': False},
        'trainset_options': {'split_mode': 'mode_2', 'keypoints_to_include': ['nose', 'body']}})
    assert config['trainset_options']['split_mode'] == 'mode_2'
    descriptor = next(c for c in catalog.describe() if c['name'] == 'vame_official')['descriptor']
    assert descriptor['backend'] == 'vame-py'
    assert 'create_trainset' in descriptor['stages']
    json.dumps(catalog.describe())


def test_catalog_exposes_distinct_kpms_backends_without_loading_them():
    code = '''
import sys
from storm.suite import default_catalog
from rainstorm_thesis.plugin import register
c = default_catalog(); register(c)
assert c.get('kpms_native').input_contract.input_type == 'pose_frames'
assert c.get('kpms_official').input_contract.input_type == 'pose_frames'
assert c.get('kpms_native').descriptor['algorithm'] == 'PCA + minimal AR-HMM'
assert not any(m in sys.modules for m in ('torch', 'jax', 'keypoint_moseq', 'sklearn'))
'''
    subprocess.run([sys.executable, '-c', code], check=True)


def test_official_projects_are_unique_before_any_training(tmp_path):
    from types import SimpleNamespace
    from rainstorm_thesis.vame_models import VAMEOfficialModel
    pose = tmp_path / 'session.h5'; pose.write_bytes(b'fixture')
    projects = []
    def init(**kwargs):
        projects.append((kwargs['working_directory'], kwargs['project_name']))
        raise RuntimeError('stop before training')
    for _ in range(2):
        model = VAMEOfficialModel({'pose_paths': [str(pose)], 'working_directory': str(tmp_path)},
                                   vame_module=SimpleNamespace(init_new_project=init))
        with pytest.raises(RuntimeError, match='stop before training'):
            model.fit_predict([[1], [2], [3]])
    assert projects[0] != projects[1]


def test_native_seed_zero_matches_declared_configuration():
    from rainstorm_thesis.vame_models import VAMENativeModel
    assert VAMENativeModel({'seed': 0}).config['seed'] == 0
