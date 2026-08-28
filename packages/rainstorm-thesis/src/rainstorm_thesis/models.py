from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from storm import ModelOutput


class ExternalSegmentationModel:
    """Traceable placeholder for an external VAME or KPMS adapter."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.model_family = str(config.get("model_family", "segmentation"))
        self.variant = str(config.get("variant", "pose_base"))
        self.n_states = int(config.get("n_states", 12))

    def fit(self, inputs: Any, targets: Any = None) -> "ExternalSegmentationModel":
        return self

    def predict(self, inputs: Any) -> ModelOutput:
        values = np.asarray(inputs, dtype=float)
        if values.ndim != 2 or values.shape[0] == 0:
            labels = np.array([], dtype=int)
        else:
            score = np.nan_to_num(values).sum(axis=1)
            edges = np.quantile(score, np.linspace(0, 1, self.n_states + 1))
            labels = np.digitize(score, edges[1:-1], right=False)
        return ModelOutput(
            predictions=labels.tolist(),
            metadata={
                "adapter": self.model_family,
                "variant": self.variant,
                "n_states": self.n_states,
                "mode": "notebook-placeholder",
            },
        )


class RainstormSupervisedProxyModel:
    """Small supervised proxy that keeps the TensorFlow ANN module isolated."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.threshold = float(config.get("threshold", 0.5))
        self.model_path = str(config.get("model_path", ""))

    def fit(self, inputs: Any, targets: Any = None) -> "RainstormSupervisedProxyModel":
        return self

    def predict(self, inputs: Any) -> ModelOutput:
        values = np.asarray(inputs, dtype=float)
        movement = np.nan_to_num(np.abs(np.diff(values, axis=0)).mean(axis=1))
        movement = np.concatenate([[0.0], movement])
        if movement.max() > 0:
            movement = movement / movement.max()
        predictions = (movement >= self.threshold).astype(int)
        return ModelOutput(
            predictions=predictions.tolist(),
            metadata={
                "adapter": "rainstorm-supervised-proxy",
                "model_path": self.model_path,
                "threshold": self.threshold,
            },
        )
