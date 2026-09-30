"""Small STORM pipeline for explicit pose selection, centering, and windows."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from storm.pipeline import PipelineContext, PipelineRunner, PipelineStep


class SelectPoseFeatures(PipelineStep):
    step_type = 'select_pose_features'

    def __init__(self, bodyparts):
        self.bodyparts = tuple(bodyparts)
        self.feature_names = tuple(f'{part}_{axis}' for part in self.bodyparts
                                   for axis in ('x', 'y'))

    def process(self, context):
        missing = [name for name in self.feature_names if name not in context.data.columns]
        if missing:
            raise ValueError(f'Missing pose features: {missing}')
        context.data = context.data.loc[:, self.feature_names].astype(float).copy()
        context.metadata['feature_names'] = list(self.feature_names)
        return context


class RecenterPose(PipelineStep):
    step_type = 'recenter_pose'

    def __init__(self, bodyparts, center):
        self.bodyparts, self.center = tuple(bodyparts), center

    def process(self, context):
        names = (f'{self.center}_x', f'{self.center}_y')
        if any(name not in context.data for name in names):
            raise ValueError(f'Unknown centering bodypart: {self.center}')
        values = context.data.to_numpy(dtype=float, copy=True).reshape(-1, len(self.bodyparts), 2)
        origin = context.data.loc[:, names].to_numpy(dtype=float)
        values -= origin[:, None, :]
        context.data.iloc[:, :] = values.reshape(len(values), -1)
        return context


class TemporalPoseWindows(PipelineStep):
    step_type = 'temporal_pose_windows'

    def __init__(self, offsets):
        self.offsets = tuple(offsets)
        if not self.offsets or 0 not in self.offsets or tuple(sorted(self.offsets)) != self.offsets:
            raise ValueError('Temporal offsets must be sorted and include zero')

    def process(self, context):
        values = context.data.to_numpy(dtype=float)
        count = len(values)
        frames = context.metadata.get('frames', list(range(count)))
        sessions = context.metadata.get('sessions', ['session'] * count)
        segments = context.metadata.get('segments', ['segment'] * count)
        partitions = context.metadata.get('partitions', ['inference'] * count)
        reserved = context.metadata.get('reserved_evaluation', [False] * count)
        if any(len(series) != count for series in (frames, sessions, segments, partitions, reserved)):
            raise ValueError('Pose boundaries must align with rows')
        windows, centers, fit_mask = [], [], []
        for center in range(count):
            start = stop = center
            while start > 0 and self._same_source(start - 1, start, frames, sessions, segments, partitions):
                start -= 1
            while stop + 1 < count and self._same_source(stop, stop + 1, frames, sessions, segments, partitions):
                stop += 1
            positions = [min(stop, max(start, center + offset)) for offset in self.offsets]
            windows.append(values[positions])
            centers.append(center)
            fit_mask.append(partitions[center] == 'train' and not any(
                reserved[index] for index in range(min(positions), max(positions) + 1)))
        context.data = np.asarray(windows, dtype=float)
        context.metadata['observation_indices'] = centers
        context.metadata['fit_mask'] = fit_mask
        return context

    @staticmethod
    def _same_source(left, right, frames, sessions, segments, partitions):
        return (sessions[left] == sessions[right]
                and segments[left] == segments[right]
                and partitions[left] == partitions[right]
                and frames[right] == frames[left] + 1)


def build_pose_pipeline(*, bodyparts, recenter_on=None, temporal_window=None):
    """Build the standard pose steps while keeping original rows addressable."""
    bodyparts = tuple(bodyparts)
    if not bodyparts or len(set(bodyparts)) != len(bodyparts):
        raise ValueError('Bodyparts must be distinct and nonempty')
    steps = [SelectPoseFeatures(bodyparts)]
    if recenter_on:
        if recenter_on not in bodyparts:
            raise ValueError('Center point must be a selected bodypart')
        steps.append(RecenterPose(bodyparts, recenter_on))
    if temporal_window:
        past, future, broad = temporal_window
        if past < 0 or future < 0 or broad < 1:
            raise ValueError('Invalid temporal window parameters')
        frames = range(-past, future + 1)
        offsets = tuple(-int(abs(frame) ** broad) if frame < 0 else int(frame ** broad)
                        for frame in frames)
        steps.append(TemporalPoseWindows(offsets))
    return PipelineRunner(steps)
