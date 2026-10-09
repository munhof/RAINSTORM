"""Strict successor to the zero-filling likelihood recipe.

Stage order and formulas recovered from Tesis_Facu's PoseFiltrationStep,
SpatialFiltration and refinement.temporal_smoothing. Unlike the historical
fallback, fewer than two valid coordinates in a segment is a diagnostic error.
Scientific libraries are imported only when the step executes in a worker.
"""
from math import isfinite
from storm.pipeline import PipelineStep


class ReconstructPoseV2(PipelineStep):
    step_type = 'pose.reconstruct_v2'
    version = '2'

    def __init__(self, threshold=.6, fps=30., max_velocity=120., dist_near=4.,
                 dist_far=12., max_outlier_connections=3, med_window=5, sigma=.6):
        if any(not isfinite(v) or v <= 0 for v in (fps, max_velocity, dist_near, dist_far, sigma)):
            raise ValueError('Reconstruction scales must be finite and positive')
        if not isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError('threshold must be between zero and one')
        if type(med_window) is not int or med_window < 1 or med_window % 2 == 0:
            raise ValueError('med_window must be a positive odd integer')
        if type(max_outlier_connections) is not int or max_outlier_connections < 0:
            raise ValueError('max_outlier_connections must be nonnegative')
        self.threshold, self.fps, self.max_velocity = threshold, fps, max_velocity
        self.dist_near, self.dist_far = dist_near, dist_far
        self.max_outlier_connections = max_outlier_connections
        self.med_window, self.sigma = med_window, sigma

    def process(self, context):
        import numpy as np
        import pandas as pd
        from scipy import ndimage, signal
        if context.metadata.get('units') != 'cm':
            raise ValueError('Reconstruction requires verified centimeters; supply an explicit calibration')
        values = np.asarray(context.data, dtype=float)
        if values.ndim != 2 or not len(values) or values.shape[1] % 2:
            raise ValueError('Expected frame rows with paired x/y coordinates')
        values = values.reshape(len(values), -1, 2).copy()
        m, count = context.metadata, len(values)
        for key in ('sessions', 'partitions', 'segments', 'frames', 'likelihoods'):
            if key not in m or len(m[key]) != count:
                raise ValueError(f'Declare aligned {key} before reconstructing pose')
        confidence = np.asarray(m['likelihoods'], dtype=float)
        if confidence.shape != values.shape[:2]:
            raise ValueError('Confidence must align with pose bodyparts')
        values[(confidence < self.threshold) | ~np.isfinite(confidence)] = np.nan
        values[~np.isfinite(values)] = np.nan
        likelihood_mask = np.isfinite(values).all(axis=2)
        boundaries = [0]
        for i in range(1, count):
            if (any(m[k][i] != m[k][i-1] for k in ('sessions', 'partitions', 'segments'))
                    or m['frames'][i] != m['frames'][i-1] + 1):
                boundaries.append(i)
        boundaries.append(count)
        filtered_mask = np.zeros(values.shape[:2], dtype=bool)
        audit = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            out = values[start:end].copy()
            jumps = np.linalg.norm(np.diff(out, axis=0), axis=2) * self.fps > self.max_velocity
            out[1:][jumps] = np.nan
            valid = np.isfinite(out).all(axis=2)
            distances = np.linalg.norm(out[:, :, None, :] - out[:, None, :, :], axis=3)
            diagonal = np.arange(out.shape[1]); distances[:, diagonal, diagonal] = np.inf
            nearby = valid & (distances <= self.dist_near).any(axis=2)
            pairs = nearby[:, :, None] & nearby[:, None, :]
            long_connections = (pairs & (distances > self.dist_far)).sum(axis=2)
            keep = nearby & (long_connections <= self.max_outlier_connections)
            out[~keep] = np.nan
            # Historical spatial stage returned float32 before interpolation.
            out = out.astype(np.float32)
            filtered_mask[start:end] = keep
            for bodypart in range(out.shape[1]):
                for coordinate in range(2):
                    series = out[:, bodypart, coordinate]
                    if np.isfinite(series).sum() < 2:
                        raise ValueError(f"{m['sessions'][start]} / {m['partitions'][start]} / "
                            f"frames {m['frames'][start]}-{m['frames'][end-1]} / bodypart {bodypart}: "
                            'fewer than two valid coordinates; zero filling is forbidden')
                    interpolated = pd.Series(series).interpolate(
                        method='pchip', limit_area='inside').ffill().bfill().to_numpy()
                    out[:, bodypart, coordinate] = ndimage.gaussian_filter1d(
                        signal.medfilt(interpolated, kernel_size=self.med_window), sigma=self.sigma)
            values[start:end] = out
            audit.append({'session': m['sessions'][start], 'partition': m['partitions'][start],
                'first_frame': m['frames'][start], 'last_frame': m['frames'][end-1],
                'frames': end-start, 'invalid_after_spatial': int((~keep).sum())})
        context.data = values.reshape(count, -1).astype(np.float32).tolist()
        context.state['reconstruction_v2'] = {'recipe_version': self.version,
            'invalid_after_likelihood': int((~likelihood_mask).sum()),
            'invalid_after_spatial': int((~filtered_mask).sum()),
            'valid_mask': filtered_mask.tolist(), 'likelihood_mask': likelihood_mask.tolist(),
            'segments': audit, 'parameters': dict(vars(self)),
            'edge_policy': 'historical ffill/bfill within each segment; not multistream alignment'}
        return context
