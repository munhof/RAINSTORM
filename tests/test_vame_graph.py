from copy import deepcopy
import json
import pytest


def test_default_graph_roundtrip_and_registered_replacement():
    from rainstorm_thesis.vame_graph import default_graph, compile_graph
    graph = json.loads(json.dumps(default_graph()))
    assert compile_graph(graph)['architecture'] == {}
    encoder = next(node for node in graph['nodes'] if node['id'] == 'encoder')
    encoder['type'] = 'encoder.lstm'
    encoder['config'] = {'encoder_layers': 2, 'encoder_bidirectional': False}
    compiled = compile_graph(graph)
    assert compiled['architecture']['encoder_type'] == 'lstm'
    assert compiled['architecture']['encoder_layers'] == 2


def test_graph_rewiring_and_unregistered_code_are_rejected_preflight():
    from rainstorm_thesis.vame_graph import default_graph
    from rainstorm_thesis.plugin import register
    from storm.suite import default_catalog
    from storm.contracts import validate_plan
    catalog = default_catalog(); register(catalog)
    graph = default_graph()
    graph['nodes'][1]['type'] = 'arbitrary.module:Code'
    problems = validate_plan({'model': 'vame_native', 'config': {'architecture_graph': graph}}, catalog)
    assert any(p.code == 'config.invalid' and 'registered' in p.message for p in problems)
    graph = default_graph()
    graph['edges'].append({'source': 'output.out', 'target': 'encoder.in'})
    assert any(p.code == 'config.invalid' for p in validate_plan(
        {'model': 'vame_native', 'config': {'architecture_graph': graph}}, catalog))


def test_graph_drives_real_native_network():
    pytest.importorskip('torch')
    import numpy as np
    from rainstorm_thesis.vame_graph import default_graph
    from rainstorm_thesis.vame_models import VAMENativeModel
    graph = default_graph()
    encoder = next(node for node in graph['nodes'] if node['id'] == 'encoder')
    encoder['type'] = 'encoder.lstm'
    model = VAMENativeModel({'architecture_graph': graph, 'n_states': 2, 'latent_dim': 2,
        'hidden_dim': 4, 'epochs': 1, 'batch_size': 2, 'device': 'cpu', 'num_threads': 1})
    values = np.random.default_rng(42).normal(size=(6, 3, 4)).tolist()
    assert len(model.fit_predict(values).predictions) == 6
    assert model.checkpoint['model_state']['encoder.weight_ih_l0'].shape[0] == 16
    assert len(model.predict(values).predictions) == 6
