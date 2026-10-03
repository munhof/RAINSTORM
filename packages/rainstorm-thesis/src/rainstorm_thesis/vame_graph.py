"""First executable VAME graph: fixed VAE wiring, replaceable recurrent nodes.

The declared wiring preserves the existing loss/training semantics. Arbitrary
topologies require a separately registered plugin implementation.
"""
from copy import deepcopy
from storm.graphs import validate_graph


ARCHITECTURE_SCHEMA = {'type': 'object', 'properties': {
    'encoder_layers': {'type': 'integer', 'minimum': 1},
    'encoder_bidirectional': {'type': 'boolean'},
    'encoder_type': {'type': 'string', 'enum': ['gru', 'lstm']},
    'decoder_layers': {'type': 'integer', 'minimum': 1},
    'decoder_hidden_dim': {'type': 'integer', 'minimum': 1}}}
OPTIMIZER_SCHEMA = {'type': 'object', 'properties': {
    'type': {'type': 'string', 'enum': ['adam', 'adamw', 'sgd']},
    'weight_decay': {'type': 'number', 'minimum': 0}}}
DISCRETIZER_SCHEMA = {'type': 'object', 'properties': {
    'n_init': {'type': 'integer', 'minimum': 1}, 'max_iter': {'type': 'integer', 'minimum': 1}}}


def node(inputs, outputs, properties=None):
    return {'inputs': inputs, 'outputs': outputs,
            'schema': {'type': 'object', 'properties': properties or {}}}


NODE_DESCRIPTORS = {
    'windows': node({}, {'out': 'windows'}),
    'encoder.gru': node({'in': 'windows'}, {'out': 'hidden'},
        {k: v for k, v in ARCHITECTURE_SCHEMA['properties'].items() if k in ('encoder_layers', 'encoder_bidirectional')}),
    'latent.mu': node({'in': 'hidden'}, {'out': 'latent'}),
    'latent.logvar': node({'in': 'hidden'}, {'out': 'latent'}),
    'sample': node({'mu': 'latent', 'logvar': 'latent'}, {'out': 'latent'}),
    'project': node({'in': 'latent'}, {'out': 'decoder_input'}),
    'decoder.gru': node({'in': 'decoder_input'}, {'out': 'decoded'},
        {k: v for k, v in ARCHITECTURE_SCHEMA['properties'].items() if k.startswith('decoder_')}),
    'output': node({'in': 'decoded'}, {'out': 'windows'}),
    'loss.mse': node({'target': 'windows', 'prediction': 'windows'}, {'out': 'loss'}),
    'loss.kl': node({'mu': 'latent', 'logvar': 'latent'}, {'out': 'loss'}),
    'loss.weighted': node({'mse': 'loss', 'kl': 'loss'}, {'out': 'loss'}, {
        'kld_weight': {'type': 'number', 'minimum': 0}}),
    'optimizer': node({'in': 'loss'}, {}, OPTIMIZER_SCHEMA['properties']),
    'discretizer.kmeans': node({'in': 'latent'}, {'out': 'states'}, DISCRETIZER_SCHEMA['properties']),
}
NODE_DESCRIPTORS['encoder.lstm'] = deepcopy(NODE_DESCRIPTORS['encoder.gru'])
SLOTS = {'windows': ['windows'], 'encoder': ['encoder.gru', 'encoder.lstm'],
    'mu': ['latent.mu'], 'logvar': ['latent.logvar'], 'sample': ['sample'], 'project': ['project'],
    'decoder': ['decoder.gru'], 'output': ['output'], 'mse': ['loss.mse'], 'kl': ['loss.kl'],
    'loss': ['loss.weighted'], 'optimizer': ['optimizer'], 'discretizer': ['discretizer.kmeans']}
EDGES = [('windows.out', 'encoder.in'), ('encoder.out', 'mu.in'), ('encoder.out', 'logvar.in'),
    ('mu.out', 'sample.mu'), ('logvar.out', 'sample.logvar'), ('sample.out', 'project.in'),
    ('project.out', 'decoder.in'), ('decoder.out', 'output.in'), ('windows.out', 'mse.target'),
    ('output.out', 'mse.prediction'), ('mu.out', 'kl.mu'), ('logvar.out', 'kl.logvar'),
    ('mse.out', 'loss.mse'), ('kl.out', 'loss.kl'), ('loss.out', 'optimizer.in'),
    ('mu.out', 'discretizer.in')]


def default_graph():
    return {'version': '1', 'nodes': [{'id': name, 'type': kinds[0], 'config': {}}
                                     for name, kinds in SLOTS.items()],
            'edges': [{'source': src, 'target': dst} for src, dst in EDGES]}


def compile_graph(graph):
    problems = validate_graph(graph, NODE_DESCRIPTORS)
    if problems:
        raise ValueError('; '.join(p.message for p in problems))
    nodes = {node['id']: node for node in graph['nodes']}
    if (set(nodes) != set(SLOTS) or any(nodes[name]['type'] not in kinds for name, kinds in SLOTS.items())
            or {(e['source'], e['target']) for e in graph['edges']} != set(EDGES)):
        raise ValueError('This VAME implementation requires the declared VAE wiring and node slots')
    architecture = {**nodes['encoder'].get('config', {}), **nodes['decoder'].get('config', {})}
    if nodes['encoder']['type'] == 'encoder.lstm':
        architecture['encoder_type'] = 'lstm'
    result = {'architecture': architecture}
    for key in ('optimizer', 'discretizer'):
        result[key] = deepcopy(nodes[key].get('config', {}))
    result.update(nodes['loss'].get('config', {}))
    return result


def validate_native_config(config):
    graph = config.get('architecture_graph')
    if graph:
        compile_graph(graph)
        if any(config.get(key) for key in ('architecture', 'optimizer', 'discretizer')):
            raise ValueError('Configure architecture/optimizer/discretizer in the graph or in fields, not both')
