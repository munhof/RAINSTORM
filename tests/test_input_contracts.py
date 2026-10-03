from copy import deepcopy
import json
import os
import subprocess
import sys


def test_declared_model_inputs_and_lazy_catalog():
    from storm.contracts import validate_plan
    from storm.suite import default_catalog
    from rainstorm_thesis.plugin import register
    catalog = default_catalog()
    register(catalog)
    assert catalog.get('vame_native').input_contract.required_steps == ('pose.temporal_windows',)
    assert catalog.get('vame_official').input_contract.granularity == 'session'
    assert catalog.get('supervised_simple').input_contract.shape == (12,)
    assert catalog.get('supervised_wide').input_contract.shape == (7, 12)
    assert any(p.code == 'preparation.external_forbidden' for p in validate_plan(
        {'model': 'vame_official', 'steps': [{'type': 'center'}]}, catalog))
    assert any(p.code == 'preparation.required_step' for p in validate_plan(
        {'model': 'vame_native'}, catalog))
    assert any(p.code == 'input.shape' for p in validate_plan(
        {'model': 'supervised_wide'}, catalog, {'shape': [12]}))
    subprocess.run([sys.executable, '-c',
        'import sys, json; from storm.suite import default_catalog; '
        'from rainstorm_thesis.plugin import register; c=default_catalog(); register(c); '
        'json.dumps(c.describe()); assert "torch" not in sys.modules; '
        'assert "tensorflow" not in sys.modules'], check=True, env=os.environ.copy())


def test_pose_resolver_preserves_names_indices_and_input():
    from rainstorm_thesis.preparation import resolve_preparation_steps
    steps = [{'type': 'pose.select_coordinates', 'config': {'names': ['body_x', 'body_y', 'nose_x', 'nose_y']}},
             {'type': 'pose.recenter', 'config': {'center_bodypart': 'body', 'bodyparts': ['nose'], 'device': 'cpu'}}]
    features = ['nose_x', 'nose_y', 'body_x', 'body_y']
    original = deepcopy(steps)
    resolved = resolve_preparation_steps(steps, features)
    assert resolved[0]['config']['names'] == original[0]['config']['names']
    assert resolved[1]['config'] == {'center_indices': [0, 1], 'coordinate_pairs': [[2, 3]], 'device': 'cpu'}
    assert steps == original
    assert features == ['nose_x', 'nose_y', 'body_x', 'body_y']
