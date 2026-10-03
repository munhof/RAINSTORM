"""Lightweight declared scientific controls; backend reflection runs in workers."""
from copy import deepcopy


def names(default=None):
    result = {'type': 'array', 'items': {'type': 'string'}}
    if default is not None:
        result['default'] = default
    return result


PREPROCESSING_OPTIONS = {'type': 'object', 'default': {}, 'properties': {
    key: {'type': 'boolean'} for key in ('run_lowconf_cleaning', 'run_egocentric_alignment',
        'run_outlier_cleaning', 'run_savgol_filtering', 'run_rescaling')}}
TRAINSET_OPTIONS = {'type': 'object', 'default': {}, 'properties': {
    'test_fraction': {'type': 'number', 'minimum': 0.000001, 'maximum': .999999},
    'split_mode': {'type': 'string', 'enum': ['mode_1', 'mode_2']},
    'keypoints_to_include': names(), 'keypoints_to_exclude': names(),
    'read_from_variable': {'type': 'string'}}}


def vame_config_properties():
    """Supported controls from vame-py 0.14.4 ProjectSchema; no model construction."""
    properties = {}
    for name in ('n_clusters', 'model_snapshot', 'model_convergence', 'time_window', 'zdims',
                 'batch_size', 'num_features', 'hidden_size_layer_1', 'hidden_size_layer_2',
                 'hidden_size_rec', 'hidden_size_pred', 'n_layers', 'prediction_steps',
                 'scheduler_step_size', 'steps_per_epoch', 'hmm_n_iter', 'n_init_kmeans',
                 'savgol_length', 'savgol_order'):
        properties[name] = {'type': 'integer', 'minimum': 1}
    properties['max_epochs'] = {'type': 'integer', 'minimum': 2}
    properties['seed'] = {'type': 'integer', 'minimum': 0}
    for name in ('learning_rate', 'beta', 'kmeans_lambda', 'scheduler_gamma', 'scheduler_threshold'):
        properties[name] = {'type': 'number', 'minimum': 0}
    for name in ('confidence', 'dropout_encoder', 'dropout_rec', 'dropout_pred'):
        properties[name] = {'type': 'number', 'minimum': 0, 'maximum': 1}
    for name in ('individual_segmentation', 'noise', 'beta_norm', 'softplus', 'robust',
                 'savgol_filter', 'egocentric_data'):
        properties[name] = {'type': 'boolean'}
    for name in ('prediction_decoder', 'scheduler', 'kmeans_loss', 'kl_start', 'annealtime', 'iqr_factor'):
        properties[name] = {'type': 'integer', 'minimum': 0}
    properties.update(keypoints=names(), extra_features=names(),
        transition_function={'type': 'string', 'enum': ['GRU']},
        anneal_function={'type': 'string', 'enum': ['linear', 'sigmoid']})
    for name in ('mse_reconstruction_reduction', 'mse_prediction_reduction'):
        properties[name] = {'type': 'string', 'enum': ['sum', 'mean']}
    return properties


def kpms_schema(official=False):
    props = {'num_states': {'type': 'integer', 'minimum': 2, 'default': 100 if official else 20},
        'latent_dim': {'type': 'integer', 'minimum': 1},
        'kappa': {'type': 'number', 'minimum': 0, 'default': 1e6 if official else 100},
        'seed': {'type': 'integer', 'minimum': 0, 'default': 156},
        'confidence_threshold': {'type': 'number', 'minimum': .000001, 'maximum': 1, 'default': .6},
        'bodyparts': names([]), 'fps': {'type': 'number', 'minimum': .000001, 'default': 30}}
    if official:
        props.update(project_dir={'type': 'string', 'default': '.storm/rainstorm/kpms_official'},
            use_bodyparts=names([]), anterior_bodyparts=names([]), posterior_bodyparts=names([]),
            fit_arhmm={'type': 'boolean', 'default': True},
            fit_full_model={'type': 'boolean', 'default': False},
            **{key: {'type': 'integer', 'minimum': 1, 'default': 50}
               for key in ('ar_iters', 'full_iters', 'inference_iters')})
    else:
        props.update(pca_variance_threshold={'type': 'number', 'minimum': .000001, 'maximum': .999999},
            whiten_pca={'type': 'boolean', 'default': True},
            ar_order={'type': 'integer', 'minimum': 1, 'default': 3},
            num_iters={'type': 'integer', 'minimum': 1, 'default': 50},
            sticky_alpha={'type': 'number', 'minimum': 0, 'default': 1},
            use_sticky_prior={'type': 'boolean', 'default': False},
            show_progress={'type': 'boolean', 'default': False})
    return {'type': 'object', 'properties': props}


def scientific_descriptor(backend, algorithm, stages, effects):
    dependencies = {'torch': ['torch', 'numpy', 'scikit-learn'],
                    'vame-py': ['vame-py', 'torch', 'numpy'],
                    'keypoint-moseq': ['keypoint-moseq', 'jax-moseq', 'jax', 'numpy'],
                    'numpy/sklearn': ['numpy', 'scikit-learn']}
    return {'version': '1', 'backend': backend, 'algorithm': algorithm,
            'dependencies': dependencies.get(backend, []),
            'source_modules': {'torch': ['rainstorm_thesis.vame_native_runtime', 'rainstorm_thesis.vame_graph'],
                'vame-py': ['rainstorm_thesis.vame_official_runtime'],
                'numpy/sklearn': ['rainstorm_thesis.kpms_initializer', 'rainstorm_thesis.kpms_preprocessing']
                }.get(backend, []),
            'stages': deepcopy(stages), 'effects': effects,
            'extension_policy': 'installed Python plugin packages with declared contracts'}


def validate_official_config(config):
    options = config.get('trainset_options', {})
    if options.get('keypoints_to_include') and options.get('keypoints_to_exclude'):
        raise ValueError('Choose keypoints_to_include or keypoints_to_exclude, not both')
    recipe = config.get('config_kwargs', {})
    length, order = recipe.get('savgol_length', 5), recipe.get('savgol_order', 2)
    if config.get('preprocessing_options', {}).get('run_savgol_filtering', True):
        if length % 2 == 0 or order >= length:
            raise ValueError('savgol_length must be odd and greater than savgol_order')
