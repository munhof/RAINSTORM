"""STORM inference adapter for imported binary models in a separate runtime."""
from pathlib import Path
import os
import subprocess
import tempfile

import numpy as np
from storm import ModelOutput


class RainstormSupervisedModel:
    capabilities = {'retrain': False, 'incremental': False, 'constraints': False,
                    'inference': True}

    def __init__(self, config, *, predictor=None):
        self.model_path = Path(config['model_path'])
        self.threshold = float(config.get('threshold', .5))
        if not 0 <= self.threshold <= 1:
            raise ValueError('Threshold must be between zero and one')
        self.runtime_python = config.get('runtime_python')
        self.output_meaning = config.get('output_meaning', 'unknown')
        self.negative_output_meaning = config.get('negative_output_meaning', 'unknown')
        self.category_mapping_version = str(config.get('category_mapping_version', '1'))
        self.training_population = config.get('training_population', 'unknown')
        self.predictor = predictor or self._subprocess_predict

    def fit(self, inputs, targets=None):
        raise NotImplementedError('Imported model supports inference only')

    def predict(self, inputs):
        values = np.asarray(inputs, dtype=np.float32)
        if values.shape[1:] not in ((12,), (7,12)) or not np.isfinite(values).all():
            raise ValueError('Expected finite inputs with shape (N,12) or (N,7,12)')
        if len(values) == 0:
            raise ValueError('No observations supplied')
        probabilities = np.asarray(self.predictor(self.model_path, values))
        if probabilities.shape not in ((len(values),), (len(values),1)):
            raise ValueError('Expected one binary probability per observation')
        probabilities = probabilities.reshape(-1)
        if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError('Invalid binary probabilities')
        return ModelOutput(predictions=(probabilities >= self.threshold).astype(int).tolist(),
            metadata={'adapter':'rainstorm-supervised', 'probabilities': probabilities.tolist(),
                      'threshold':self.threshold, 'output_meaning':self.output_meaning,
                      'negative_output_meaning':self.negative_output_meaning,
                      'category_mapping': {
                          '0': self.negative_output_meaning,
                          '1': self.output_meaning,
                      },
                      'category_mapping_version': self.category_mapping_version,
                      'training_population':self.training_population,
                      'task':'binary_classification', 'capabilities':dict(self.capabilities)})

    def _subprocess_predict(self, model_path, values):
        if not self.runtime_python:
            raise ValueError('runtime_python must point to the isolated model environment')
        with tempfile.TemporaryDirectory(prefix='rainstorm-inference-') as folder:
            inputs = Path(folder) / 'inputs.npy'
            output = Path(folder) / 'probabilities.npy'
            np.save(inputs, values, allow_pickle=False)
            env = dict(os.environ)
            env.pop('PYTHONPATH', None)
            subprocess.run([str(self.runtime_python), '-m', 'rainstorm_supervised',
                            '--model', str(model_path.resolve()), '--inputs', str(inputs),
                            '--output', str(output)], check=True, env=env)
            return np.load(output, allow_pickle=False)
