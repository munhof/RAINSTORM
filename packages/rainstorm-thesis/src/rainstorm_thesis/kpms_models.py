"""Lazy STORM adapters for distinct native AR-HMM and official KPMS backends."""
from copy import deepcopy
import importlib
from pathlib import Path
from uuid import uuid4
from storm import ModelOutput


def pose_sequences(inputs, data, indices, config):
    """Select declared pose columns and split at session/segment/frame boundaries."""
    import numpy as np
    values = np.asarray(inputs, dtype=float)
    names = data.get('feature_names', [])
    bodyparts = list(config.get('bodyparts') or dict.fromkeys(
        name[:-2] for name in names if name.endswith('_x')))
    if values.ndim == 3 and values.shape[-1] in (2, 3):
        if not bodyparts:
            bodyparts = [f'point_{i}' for i in range(values.shape[1])]
        coordinates = values.copy()
        confidences = np.ones(values.shape[:2])
    elif values.ndim == 2 and bodyparts:
        coordinates = values[:, [[names.index(f'{part}_{axis}') for axis in ('x', 'y')]
                                  for part in bodyparts]]
        if all(f'{part}_likelihood' in names for part in bodyparts):
            confidences = values[:, [names.index(f'{part}_likelihood') for part in bodyparts]]
        elif 'likelihoods' in data and data.get('likelihood_bodyparts'):
            columns = [data['likelihood_bodyparts'].index(part) for part in bodyparts]
            confidences = np.asarray(data['likelihoods'])[indices][:, columns]
        else:
            raise ValueError('KPMS requires declared confidence columns or likelihood metadata')
    else:
        raise ValueError('KPMS requires pose frames and declared feature_names/bodyparts')
    if len(coordinates) != len(indices) or coordinates.shape[1] != len(bodyparts):
        raise ValueError('KPMS observation mapping does not match pose shape')
    if not len(indices) or not np.isfinite(coordinates).all() or not np.isfinite(confidences).all():
        raise ValueError('KPMS pose/confidence values must be finite and nonempty')
    if ((confidences < 0) | (confidences > 1)).any():
        raise ValueError('KPMS confidence values must lie between 0 and 1')
    sessions = [data.get('sessions', ['session'] * len(data.get('inputs', inputs)))[i] for i in indices]
    frames = [data.get('frames', list(range(len(data.get('inputs', inputs)))))[i] for i in indices]
    segments = [data.get('segments', ['segment'] * len(data.get('inputs', inputs)))[i] for i in indices]
    lengths = []
    for j, index in enumerate(indices):
        if (j == 0 or sessions[j] != sessions[j-1] or segments[j] != segments[j-1]
                or index != indices[j-1] + 1 or frames[j] != frames[j-1] + 1):
            lengths.append(1)
        else:
            lengths[-1] += 1
    return dict(coordinates=coordinates, confidences=confidences, bodyparts=bodyparts,
                lengths=lengths, sessions=sessions, frames=frames)


class KPMSNativeModel:
    """PCA plus minimal AR-HMM; this is not the official keypoint-SLDS model."""
    supports_pipeline_steps = False
    requires_registered_data_for_inference = True

    def __init__(self, config=None):
        self.config = deepcopy(config or {})
        self._bound = None
        self.pca = self.initializer = None

    def configuration_snapshot(self):
        return deepcopy(self.config)

    def bind_data(self, data, observation_indices):
        if any(data.get('reserved_evaluation', [False] * len(data['inputs']))[i]
               for i in observation_indices):
            raise ValueError('KPMS training cannot include reserved observations')
        self._bound = (data, list(observation_indices))

    def _bundle(self, inputs, data=None, indices=None):
        if data is None:
            data, indices = self._bound or ({'inputs': inputs}, list(range(len(inputs))))
        return pose_sequences(inputs, data, indices, self.config)

    def _preprocess(self, bundle):
        import numpy as np
        from .kpms_preprocessing import mask_low_confidence, interpolate_masked, add_initialization_noise
        # Interpolation must never cross a session, partition or missing-frame boundary.
        prepared, offset = [], 0
        for length in bundle['lengths']:
            masked, mask = mask_low_confidence(bundle['coordinates'][offset:offset+length],
                bundle['confidences'][offset:offset+length], self.config.get('confidence_threshold', .6))
            prepared.append(interpolate_masked(masked, mask))
            offset += length
        return add_initialization_noise(np.concatenate(prepared), seed=self.config.get('seed', 156))

    def fit_predict(self, inputs, constraints=()):
        if constraints:
            raise ValueError('KPMS does not accept pairwise constraints')
        from .kpms_config import KeypointMoseqConfig
        from .kpms_initializer import KPMSInitializer
        from .kpms_preprocessing import fit_pose_pca
        bundle = self._bundle(inputs)
        self._bodyparts = tuple(bundle['bodyparts'])
        config = {key: value for key, value in self.config.items() if key != 'bodyparts'}
        if 'latent_dim' in config and 'pca_variance_threshold' not in config:
            config['pca_variance_threshold'] = None
        options = KeypointMoseqConfig(**config)
        components = options.pca_variance_threshold or options.latent_dim
        self.pca, latent = fit_pose_pca(self._preprocess(bundle), components, options.whiten_pca, options.seed)
        config.update(latent_dim=latent.shape[1], pca_variance_threshold=None)
        self.initializer = KPMSInitializer(KeypointMoseqConfig(**config))
        result = self.initializer.fit(latent, sequence_lengths=bundle['lengths'])
        self._bound = None  # Source metadata belongs to the run, not the model artifact.
        return self._output(result.state_sequence, bundle)

    def predict(self, inputs):
        return self._predict(self._bundle(inputs))

    def predict_with_context(self, inputs, data, indices):
        return self._predict(self._bundle(inputs, data, indices))

    def _predict(self, bundle):
        if tuple(bundle['bodyparts']) != getattr(self, '_bodyparts', None):
            raise ValueError('KPMS inference pose schema differs from training')
        if self.pca is None or self.initializer is None:
            raise ValueError('Fit native KPMS before inference')
        from .kpms_preprocessing import apply_pose_pca
        latent = apply_pose_pca(self._preprocess(bundle), self.pca)
        return self._output(self.initializer.predict(latent, sequence_lengths=bundle['lengths']), bundle)

    @staticmethod
    def _output(labels, bundle):
        return ModelOutput(labels.tolist(), {'backend': 'rainstorm_pca_arhmm',
            'algorithm': 'PCA + minimal AR-HMM', 'requires_registered_data_for_inference': True,
            'discretizer_scope': 'shared_training_model',
            'sequence_lengths': bundle['lengths'], 'sessions': bundle['sessions'],
            'frames': bundle['frames'], 'bodyparts': bundle['bodyparts']})


class KPMSOfficialModel(KPMSNativeModel):
    """Official KPMS stages, with fitted PCA supplied before model initialization."""
    def __init__(self, config=None, *, kpms_module=None):
        super().__init__(config)
        self._kpms_module = kpms_module
        self.model = None

    def _backend(self):
        return self._kpms_module or importlib.import_module('keypoint_moseq')

    def _format(self, kpms, bundle):
        coordinates, confidences, offset = {}, {}, 0
        for index, length in enumerate(bundle['lengths']):
            name = f'sequence_{index}'
            coordinates[name] = bundle['coordinates'][offset:offset+length]
            confidences[name] = bundle['confidences'][offset:offset+length]
            offset += length
        return coordinates, confidences, kpms.format_data(coordinates=coordinates,
            confidences=confidences, **self._resolved_config)

    def fit_predict(self, inputs, constraints=()):
        if constraints:
            raise ValueError('KPMS does not accept pairwise constraints')
        kpms, bundle = self._backend(), self._bundle(inputs)
        self._bodyparts = tuple(bundle['bodyparts'])
        base = Path(self.config.get('project_dir', '.storm/rainstorm/kpms_official')).expanduser()
        self.project_dir = str((base / uuid4().hex).resolve())
        Path(self.project_dir).mkdir(parents=True, exist_ok=False)
        parts = bundle['bodyparts']
        kpms.generate_config(project_dir=self.project_dir, bodyparts=parts,
            use_bodyparts=self.config.get('use_bodyparts') or parts,
            anterior_bodyparts=self.config.get('anterior_bodyparts') or parts[:1],
            posterior_bodyparts=self.config.get('posterior_bodyparts') or parts[-1:],
            skeleton=[], num_states=self.config.get('num_states', 100),
            latent_dim=self.config.get('latent_dim', 10), kappa=self.config.get('kappa', 1e6),
            fps=self.config.get('fps', 30), conf_threshold=self.config.get('confidence_threshold', .6))
        self._resolved_config = kpms.load_config(self.project_dir)
        coordinates, confidences, (data, metadata) = self._format(kpms, bundle)
        self.pca = kpms.fit_pca(**data, **self._resolved_config)
        self.model = kpms.init_model(data=data, pca=self.pca, seed=self.config.get('seed', 156),
                                    **self._resolved_config)
        name, iteration = 'model', 0
        for enabled, ar_only, count in ((self.config.get('fit_arhmm', True), True, self.config.get('ar_iters', 50)),
                (self.config.get('fit_full_model', False), False, self.config.get('full_iters', 50))):
            if enabled:
                self.model, name = kpms.fit_model(model=self.model, data=data, metadata=metadata,
                    project_dir=self.project_dir, model_name=name, ar_only=ar_only,
                    start_iter=iteration, num_iters=iteration+count, generate_progress_plots=False)
                iteration += count
        self._bound = None
        return self._predict(bundle)

    def _predict(self, bundle):
        if tuple(bundle['bodyparts']) != getattr(self, '_bodyparts', None):
            raise ValueError('KPMS inference pose schema differs from training')
        if self.model is None:
            raise ValueError('Fit official KPMS before inference')
        import numpy as np
        kpms = self._backend()
        _, _, (data, metadata) = self._format(kpms, bundle)
        results = kpms.apply_model(model=self.model, data=data, metadata=metadata,
            num_iters=self.config.get('inference_iters', 100),
            ar_only=not self.config.get('fit_full_model', False), save_results=False,
            **self._resolved_config)
        labels = []
        for index, length in enumerate(bundle['lengths']):
            values = np.asarray(results[f'sequence_{index}']['syllable']).reshape(-1)
            if len(values) != length:
                raise ValueError('Official KPMS predictions do not match source frames')
            labels.extend(values.tolist())
        return ModelOutput(labels, {'backend': 'keypoint_moseq_official',
            'requires_registered_data_for_inference': True,
            'discretizer_scope': 'shared_training_model', 'sequence_lengths': bundle['lengths'],
            'sessions': bundle['sessions'], 'frames': bundle['frames'],
            'project_dir': self.project_dir})
