from __future__ import annotations

from math import atan2, cos, isfinite, pi, radians, sin

from storm.pipeline import PipelineContext, PipelineStep


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

    def __init__(self, center_indices: list[int], coordinate_pairs: list[list[int]]):
        if len(center_indices) != 2 or any(type(i) is not int or i < 0 for i in center_indices):
            raise ValueError("center_indices must contain x and y indices")
        if not coordinate_pairs or any(len(pair) != 2 for pair in coordinate_pairs):
            raise ValueError("coordinate_pairs must contain x/y index pairs")
        self.center_indices = tuple(center_indices)
        self.coordinate_pairs = tuple(tuple(pair) for pair in coordinate_pairs)

    def process(self, context: PipelineContext) -> PipelineContext:
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
                 degenerate_reference_policy: str = "error"):
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
        self.reference_pairs = tuple(tuple(pair) for pair in reference_pairs)
        self.coordinate_pairs = tuple(tuple(pair) for pair in coordinate_pairs)
        self.target_angle = radians(float(target_angle_degrees))
        self.degenerate_reference_policy = degenerate_reference_policy

    def process(self, context: PipelineContext) -> PipelineContext:
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
        last_good: dict[tuple[str, str, int], float] = {}
        for row_index in range(len(values)):
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
        windows, centers, window_sources = [], [], []
        for center in range(count):
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
        context.data = windows
        context.metadata["observation_indices"] = centers
        context.metadata["window_source_indices"] = window_sources
        return context
