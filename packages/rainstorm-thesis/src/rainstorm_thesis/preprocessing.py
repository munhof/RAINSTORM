from __future__ import annotations

from time import monotonic
from math import ceil, atan2, cos, isfinite, pi, radians, sin

from storm.pipeline import PipelineContext, PipelineStep


POSE_GPU_MIN_ROWS = 32_768
POSE_GPU_CHUNK_ROWS = 65_536


def _report_pose_progress(context, completed, total, started, label):
    callback = context.progress_callback
    if callback is None or (completed not in (0, total) and completed % POSE_GPU_CHUNK_ROWS):
        return
    elapsed = max(0, monotonic() - started)
    speed = completed / elapsed if completed and elapsed else None
    callback({'label': label, 'device': 'cpu',
              'batch_step': ceil(completed / POSE_GPU_CHUNK_ROWS),
              'batch_total': ceil(total / POSE_GPU_CHUNK_ROWS),
              'processed_observations': completed, 'total_observations': total,
              'throughput': speed, 'batch_elapsed_seconds': elapsed,
              'batch_eta_seconds': ceil((total - completed) / speed) if speed else None})


def _pose_compute_device(requested: str, row_count: int) -> str:
    """Resolve the optional accelerator for large, numeric pose transforms."""
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    if requested == "cpu" or (requested == "auto" and row_count < POSE_GPU_MIN_ROWS):
        return "cpu"
    try:
        import torch
    except ImportError as error:
        if requested == "cuda":
            raise RuntimeError(
                "A GPU was requested for pose preprocessing, but PyTorch is not installed."
            ) from error
        return "cpu"
    available = torch.cuda.is_available()
    if requested == "cuda" and not available:
        raise RuntimeError(
            "A GPU was requested for pose preprocessing, but this worker has no "
            "available CUDA or ROCm device. Choose auto or cpu."
        )
    return "cuda" if available else "cpu"


def _apply_pose_gpu(values, operation, progress_callback=None):
    """Apply one pose operation in bounded float64 batches on CUDA or ROCm."""
    import numpy as np
    import torch

    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Pose coordinates must form a rectangular numeric matrix") from error
    if matrix.ndim != 2 or not len(matrix):
        raise ValueError("Pose coordinates must form a nonempty numeric matrix")
    for start in range(0, len(matrix), POSE_GPU_CHUNK_ROWS):
        end = min(start + POSE_GPU_CHUNK_ROWS, len(matrix))
        batch = torch.as_tensor(matrix[start:end], dtype=torch.float64, device="cuda")
        operation(batch, torch)
        matrix[start:end] = batch.cpu().numpy()
        if progress_callback is not None:
            progress_callback({'label': 'Transformando coordenadas de pose en GPU',
                               'batch_step': (start // POSE_GPU_CHUNK_ROWS) + 1,
                               'batch_total': (len(matrix) + POSE_GPU_CHUNK_ROWS - 1) // POSE_GPU_CHUNK_ROWS,
                               'processed_observations': end, 'total_observations': len(matrix),
                               'device': 'cuda'})
    return matrix.tolist()


class SelectPoseCoordinates(PipelineStep):
    step_type = "pose.select_coordinates"

    def __init__(self, indices: list[int] | None = None, names: list[str] | None = None):
        if (indices is None) == (names is None):
            raise ValueError("Provide exactly one of indices or names")
        self.indices = tuple(indices) if indices is not None else None
        self.names = tuple(names) if names is not None else None
        if self.indices is not None and (not self.indices or any(type(i) is not int or i < 0 for i in self.indices)):
            raise ValueError("Coordinate indices must be nonnegative integers")
        if self.names is not None and (not self.names or len(set(self.names)) != len(self.names)):
            raise ValueError("Coordinate names must be distinct and nonempty")

    def process(self, context: PipelineContext) -> PipelineContext:
        names = context.metadata.get("feature_names", [])
        indices = self.indices
        if self.names is not None:
            missing = [name for name in self.names if name not in names]
            if missing:
                raise ValueError(f"Unknown pose coordinate names: {missing}")
            indices = tuple(names.index(name) for name in self.names)
        rows = [[float(value) for value in row] for row in context.data]
        if not rows or max(indices) >= len(rows[0]) or any(len(row) != len(rows[0]) for row in rows):
            raise ValueError("Coordinate selection does not match the pose rows")
        context.data = [[row[index] for index in indices] for row in rows]
        if names:
            selected_names = [names[index] for index in indices]
            context.metadata["feature_names"] = selected_names
            likelihood_bodyparts = context.metadata.get("likelihood_bodyparts", [])
            bodypart_order = [
                name[:-2] for name in selected_names
                if name.endswith("_x") and f"{name[:-2]}_y" in selected_names
            ]
            if likelihood_bodyparts and "likelihoods" in context.metadata:
                confidence_indices = [likelihood_bodyparts.index(part) for part in bodypart_order
                                      if part in likelihood_bodyparts]
                context.metadata["likelihoods"] = [
                    [row[index] for index in confidence_indices]
                    for row in context.metadata["likelihoods"]
                ]
                context.metadata["likelihood_bodyparts"] = [
                    likelihood_bodyparts[index] for index in confidence_indices
                ]
        return context


class RecenterPose(PipelineStep):
    step_type = "pose.recenter"

    def __init__(self, center_indices: list[int], coordinate_pairs: list[list[int]],
                 device: str = "auto"):
        if len(center_indices) != 2 or any(type(i) is not int or i < 0 for i in center_indices):
            raise ValueError("center_indices must contain x and y indices")
        if not coordinate_pairs or any(len(pair) != 2 for pair in coordinate_pairs):
            raise ValueError("coordinate_pairs must contain x/y index pairs")
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
        self.center_indices = tuple(center_indices)
        self.coordinate_pairs = tuple(tuple(pair) for pair in coordinate_pairs)
        self.device = device

    def process(self, context: PipelineContext) -> PipelineContext:
        requested_device = _pose_compute_device(self.device, len(context.data))
        if requested_device == "cuda":
            coordinate_indices = [index for pair in self.coordinate_pairs for index in pair]
            center_indices = self.center_indices

            def recenter(batch, torch):
                selected = torch.as_tensor(coordinate_indices, device=batch.device)
                center = torch.as_tensor(center_indices, device=batch.device)
                coordinates = batch.index_select(1, selected).reshape(
                    len(batch), len(self.coordinate_pairs), 2)
                origin = batch.index_select(1, center).unsqueeze(1)
                batch[:, selected] = (coordinates - origin).reshape(len(batch), -1)

            context.data = _apply_pose_gpu(context.data, recenter, context.progress_callback)
            return context
        values = [[float(value) for value in row] for row in context.data]
        if not values or max((*self.center_indices, *(i for pair in self.coordinate_pairs for i in pair))) >= len(values[0]):
            raise ValueError("Center or coordinate indices are outside the pose rows")
        for row in values:
            origin_x, origin_y = (row[index] for index in self.center_indices)
            for x_index, y_index in self.coordinate_pairs:
                row[x_index] -= origin_x
                row[y_index] -= origin_y
        context.data = values
        return context


class OrientPose(PipelineStep):
    """Rotate centered body coordinates to the historical 45-degree axis."""

    step_type = "pose.orient_coordinates"

    def __init__(self, reference_pairs: list[list[int]],
                 coordinate_pairs: list[list[int]], target_angle_degrees: float = 45,
                 degenerate_reference_policy: str = "error", device: str = "auto"):
        pairs = [*reference_pairs, *coordinate_pairs]
        if (len(reference_pairs) != 2 or not coordinate_pairs or
                any(len(pair) != 2 or any(type(index) is not int or index < 0 for index in pair)
                    for pair in pairs)):
            raise ValueError("Orientation requires two reference pairs and coordinate pairs")
        if not isfinite(target_angle_degrees):
            raise ValueError("target_angle_degrees must be finite")
        if degenerate_reference_policy not in {"error", "identity"}:
            raise ValueError(
                "degenerate_reference_policy must be 'error' or 'identity'"
            )
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
        self.reference_pairs = tuple(tuple(pair) for pair in reference_pairs)
        self.coordinate_pairs = tuple(tuple(pair) for pair in coordinate_pairs)
        self.target_angle = radians(float(target_angle_degrees))
        self.degenerate_reference_policy = degenerate_reference_policy
        self.device = device

    def process(self, context: PipelineContext) -> PipelineContext:
        requested_device = _pose_compute_device(self.device, len(context.data))
        if requested_device == "cuda":
            reference_indices = [index for pair in self.reference_pairs for index in pair]
            coordinate_indices = [index for pair in self.coordinate_pairs for index in pair]
            target_angle = self.target_angle
            policy = self.degenerate_reference_policy

            def orient(batch, torch):
                reference_columns = torch.as_tensor(reference_indices, device=batch.device)
                coordinate_columns = torch.as_tensor(coordinate_indices, device=batch.device)
                references = batch.index_select(1, reference_columns).reshape(-1, 2, 2)
                source, target = references[:, 0], references[:, 1]
                dx, dy = target[:, 0] - source[:, 0], target[:, 1] - source[:, 1]
                degenerate = (dx == 0) & (dy == 0)
                if policy == "error" and torch.any(degenerate).item():
                    raise ValueError("Orientation reference points coincide")
                theta = target_angle - torch.atan2(-dy, -dx)
                if policy == "identity":
                    theta = torch.where(degenerate, torch.zeros_like(theta), theta)
                cosine, sine = torch.cos(theta).unsqueeze(1), torch.sin(theta).unsqueeze(1)
                coordinates = batch.index_select(1, coordinate_columns).reshape(
                    len(batch), len(self.coordinate_pairs), 2)
                x, y = coordinates[..., 0], coordinates[..., 1]
                rotated = torch.stack((x * cosine - y * sine,
                                       x * sine + y * cosine), dim=-1)
                batch[:, coordinate_columns] = rotated.reshape(len(batch), -1)

            context.data = _apply_pose_gpu(context.data, orient, context.progress_callback)
            return context
        values = [[float(value) for value in row] for row in context.data]
        required_indices = [index for pair in (*self.reference_pairs, *self.coordinate_pairs)
                            for index in pair]
        if not values or max(required_indices) >= len(values[0]):
            raise ValueError("Orientation coordinates are outside the pose rows")
        for row in values:
            (from_x, from_y), (toward_x, toward_y) = (
                (row[x_index], row[y_index]) for x_index, y_index in self.reference_pairs)
            dx, dy = toward_x - from_x, toward_y - from_y
            if dx == 0 and dy == 0:
                if self.degenerate_reference_policy == "error":
                    raise ValueError("Orientation reference points coincide")
                continue
            theta = self.target_angle - atan2(-dy, -dx)
            cosine, sine = cos(theta), sin(theta)
            for x_index, y_index in self.coordinate_pairs:
                x, y = row[x_index], row[y_index]
                row[x_index] = x * cosine - y * sine
                row[y_index] = x * sine + y * cosine
        context.data = values
        return context


class LikelihoodFilter(PipelineStep):
    step_type = "pose.likelihood_filter"

    def __init__(self, threshold: float, coordinate_pairs: list[list[int]]):
        if not isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if not coordinate_pairs or any(len(pair) != 2 for pair in coordinate_pairs):
            raise ValueError("coordinate_pairs must contain x/y index pairs")
        self.threshold = float(threshold)
        self.coordinate_pairs = tuple(tuple(pair) for pair in coordinate_pairs)

    def process(self, context: PipelineContext) -> PipelineContext:
        values = [[float(value) for value in row] for row in context.data]
        confidence = context.metadata.get("likelihoods", [])
        if len(confidence) != len(values) or any(
            len(row) != len(self.coordinate_pairs) for row in confidence
        ):
            raise ValueError("Likelihood rows must align with the selected pose bodyparts")
        sessions = context.metadata.get("sessions", ["session"] * len(values))
        segments = context.metadata.get("segments", ["segment"] * len(values))
        frames = context.metadata.get("frames", list(range(len(values))))
        if any(len(item) != len(values) for item in (sessions, segments, frames)):
            raise ValueError("Pose boundary metadata must align with likelihood rows")
        started = monotonic()
        _report_pose_progress(context, 0, len(values), started, "Filtrando confianza de pose")
        last_good: dict[tuple[str, str, int], float] = {}
        for row_index in range(len(values)):
            _report_pose_progress(context, row_index, len(values), started, "Filtrando confianza de pose")
            for part_index, (x_index, y_index) in enumerate(self.coordinate_pairs):
                identity = (sessions[row_index], segments[row_index], part_index)
                if confidence[row_index][part_index] >= self.threshold:
                    last_good[identity] = row_index
                    continue
                previous = last_good.get(identity)
                if previous is None or frames[previous] + 1 != frames[row_index]:
                    values[row_index][x_index] = values[row_index][y_index] = 0.0
                else:
                    values[row_index][x_index] = values[previous][x_index]
                    values[row_index][y_index] = values[previous][y_index]
        _report_pose_progress(context, len(values), len(values), started, "Filtrando confianza de pose")
        context.data = values
        return context


class TemporalPoseWindows(PipelineStep):
    step_type = "pose.temporal_windows"

    def __init__(self, offsets: list[int]):
        if (not offsets or 0 not in offsets or any(type(value) is not int for value in offsets)
                or tuple(sorted(set(offsets))) != tuple(offsets)):
            raise ValueError("offsets must be sorted, unique integers including zero")
        self.offsets = tuple(offsets)

    def process(self, context: PipelineContext) -> PipelineContext:
        values = [[float(value) for value in row] for row in context.data]
        if not values or any(len(row) != len(values[0]) for row in values):
            raise ValueError("Temporal windows expect one pose feature vector per observation")
        count = len(values)
        observation_indices = context.metadata.get("observation_indices", list(range(count)))
        frames = context.metadata.get("frames", observation_indices)
        sessions = context.metadata.get("sessions", ["session"] * count)
        segments = context.metadata.get("segments", ["segment"] * count)
        partitions = context.metadata.get("partitions", ["train"] * count)
        reserved = context.metadata.get("reserved_evaluation", [False] * count)
        aligned = (observation_indices, frames, sessions, segments, partitions, reserved)
        if any(len(item) != count for item in aligned):
            raise ValueError("Pose boundaries must align with window rows")
        by_frame = {}
        for position, frame in enumerate(frames):
            key = (sessions[position], segments[position], frame, partitions[position])
            if key in by_frame:
                raise ValueError("Frame identity is duplicated inside a temporal partition")
            by_frame[key] = position
        started = monotonic()
        _report_pose_progress(context, 0, count, started, "Construyendo ventanas temporales")
        windows, centers, window_sources = [], [], []
        for center in range(count):
            _report_pose_progress(context, center, count, started, "Construyendo ventanas temporales")
            positions = []
            for offset in self.offsets:
                key = (sessions[center], segments[center], frames[center] + offset,
                       partitions[center])
                position = by_frame.get(key)
                if position is None:
                    break
                positions.append(position)
            if len(positions) != len(self.offsets):
                continue
            if partitions[center] == "train" and any(reserved[position] for position in positions):
                continue
            windows.append([values[position] for position in positions])
            centers.append(observation_indices[center])
            window_sources.append([observation_indices[position] for position in positions])
        _report_pose_progress(context, count, count, started, "Construyendo ventanas temporales")
        context.data = windows
        context.metadata["observation_indices"] = centers
        context.metadata["window_source_indices"] = window_sources
        return context
