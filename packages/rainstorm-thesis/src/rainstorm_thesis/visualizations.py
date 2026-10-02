from __future__ import annotations

from html import escape
from math import isfinite

from storm.visualization import Visualization, VisualizationRequest, VisualizationResult


class RainstormPoseTimeline(Visualization):
    visualization_type = "rainstorm_pose_timeline"

    def __init__(self, x_index: int = 0, y_index: int = 1):
        self.x_index, self.y_index = int(x_index), int(y_index)
        if min(self.x_index, self.y_index) < 0 or self.x_index == self.y_index:
            raise ValueError("x_index and y_index must be distinct nonnegative columns")

    def render(self, request: VisualizationRequest) -> VisualizationResult:
        values = _rows(request.data)
        if values and values[0] and isinstance(values[0][0], list):
            offsets = request.metadata.get("offsets")
            if offsets is not None:
                if (not isinstance(offsets, (list, tuple))
                        or len(offsets) != len(values[0])
                        or any(type(offset) is not int for offset in offsets)
                        or offsets.count(0) != 1):
                    raise ValueError(
                        "Temporal pose offsets must align with the window and include zero once"
                    )
                center = offsets.index(0)
            else:
                # Even windows use the lower middle frame, as in VAME's centered alignment.
                center = (len(values[0]) - 1) // 2
            if any(len(row) <= center for row in values):
                raise ValueError("Temporal pose windows must have aligned frame counts")
            values = [row[center] for row in values]
        indices = request.metadata.get("indices")
        if indices is not None and len(indices) != len(values):
            if any(type(index) is not int or index < 0 or index >= len(values) for index in indices):
                raise ValueError("Pose observation indices are outside the registered source rows")
            values = [values[index] for index in indices]
        if not values or max(self.x_index, self.y_index) >= len(values[0]):
            raise ValueError("Pose timeline needs aligned rows with x/y coordinate columns")
        selected = [
            [row[self.x_index], row[self.y_index]] for row in values
            if isfinite(row[self.x_index]) and isfinite(row[self.y_index])
        ]
        if not len(selected):
            raise ValueError("Pose timeline has no finite x/y coordinates")
        x, y = _scale_points(selected, 740, 190)
        points = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, y))
        frames = request.metadata.get("frames", request.metadata.get("indices", list(range(len(values)))))
        if len(frames) != len(values):
            raise ValueError("Frame IDs must align with the selected pose rows")
        label = escape(str(request.metadata.get("bodypart", "selected pose point")))
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 250" role="img" '
            f'aria-label="Pose trajectory for {label}">'
            f'<title>Pose trajectory for {label}</title>'
            '<rect width="800" height="250" fill="#fff"/>'
            '<text x="16" y="24" font-size="14">Frame-ordered pose trajectory</text>'
            f'<polyline points="{points}" fill="none" stroke="#315fbd" stroke-width="2"/>'
            f'<text x="16" y="232" font-size="12">{len(selected)} observations; '
            f'frame range {escape(str(frames[0] if frames else "?"))}–'
            f'{escape(str(frames[-1] if frames else "?"))}</text></svg>'
        )
        return VisualizationResult(svg, "image/svg+xml", {"observations": len(selected)})


class RainstormStateTimeline(Visualization):
    visualization_type = "rainstorm_state_timeline"

    def __init__(self, lane_height: int = 32):
        self.lane_height = int(lane_height)
        if not 16 <= self.lane_height <= 96:
            raise ValueError("lane_height must be between 16 and 96 pixels")

    def render(self, request: VisualizationRequest) -> VisualizationResult:
        output = request.output
        predictions = getattr(output, "predictions", None)
        if predictions is None:
            raise ValueError("State timeline requires model predictions")
        predictions = list(predictions)
        indices = list(request.metadata.get("indices", range(len(predictions))))
        frames = list(request.metadata.get("frames", indices))
        if len(indices) != len(predictions) or len(frames) != len(predictions):
            raise ValueError("Frame IDs and observation IDs must align with predictions")
        semantics = str(getattr(output, "metadata", {}).get(
            "semantics", "categorical state IDs"
        ))
        unique_states = list(dict.fromkeys(predictions))
        colors = {state: _state_color(index) for index, state in enumerate(unique_states)}
        width = 760 / max(1, len(predictions))
        rectangles = []
        labels = []
        for position, (state, frame) in enumerate(zip(predictions, frames)):
            x = 20 + position * width
            color = colors[state]
            text = f"state {state}"
            rectangles.append(
                f'<rect x="{x:.2f}" y="52" width="{max(1.0, width):.2f}" '
                f'height="{self.lane_height}" fill="{color}" stroke="#fff" stroke-width=".5">'
                f'<title>frame {escape(str(frame))}: {escape(text)}</title></rect>'
            )
            if width >= 32:
                labels.append(
                    f'<text x="{x + 2:.2f}" y="{52 + self.lane_height - 8}" '
                    f'font-size="10">{escape(str(state))}</text>'
                )
        legend = " ".join(
            f'<text x="{20 + index * 100}" y="112" font-size="12" '
            f'fill="{colors[state]}">state {escape(str(state))}</text>'
            for index, state in enumerate(unique_states)
        )
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 140" role="img" '
            'aria-label="Categorical state timeline">'
            '<title>Categorical state timeline</title>'
            f'<text x="20" y="22" font-size="14">{escape(semantics)}</text>'
            f'<text x="20" y="40" font-size="11">{len(predictions)} observations, '
            f'frames {escape(str(frames[0] if frames else "?"))}–'
            f'{escape(str(frames[-1] if frames else "?"))}</text>'
            f'{"".join(rectangles)}{"".join(labels)}{legend}</svg>'
        )
        return VisualizationResult(
            svg,
            "image/svg+xml",
            {"observation_indices": indices, "frames": frames, "states": unique_states},
        )


def _rows(data):
    try:
        if hasattr(data, "tolist"):
            data = data.tolist()
        values = [[float(value) if not isinstance(value, list) else
                   [float(item) for item in value] for value in row] for row in data]
    except (TypeError, ValueError) as error:
        raise ValueError("Pose visualization needs numeric data") from error
    def finite(value):
        return all(finite(item) for item in value) if isinstance(value, list) else isfinite(value)
    if not all(finite(row) for row in values):
        raise ValueError("Pose visualization cannot render non-finite coordinates")
    return values


def _scale_points(values, width, height):
    low = [min(row[index] for row in values) for index in range(2)]
    high = [max(row[index] for row in values) for index in range(2)]
    span = [high[index] - low[index] or 1.0 for index in range(2)]
    x = [30 + (row[0] - low[0]) / span[0] * width for row in values]
    y = [35 + height - (row[1] - low[1]) / span[1] * height for row in values]
    return x, y


def _state_color(index: int) -> str:
    palette = ("#315fbd", "#7b3fb5", "#bd6400", "#13826f", "#a33a52", "#536875")
    return palette[index % len(palette)]
