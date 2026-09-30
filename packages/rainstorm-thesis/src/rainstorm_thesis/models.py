from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from storm import ModelOutput


class ExternalSegmentationModel:
    """Unavailable external segmentation adapter.

    Keep the public name for workflow migration compatibility, but fail closed:
    this scaffold cannot produce scientific states or be ranked as a model.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.model_family = str(config.get("model_family", "external segmentation"))
        self.variant = str(config.get("variant", "pose_base"))
        self.n_states = int(config.get("n_states", 12))

    def fit(self, inputs: Any, targets: Any = None) -> "ExternalSegmentationModel":
        raise NotImplementedError(
            f"The {self.model_family} adapter is pending implementation; no model was fitted."
        )

    def predict(self, inputs: Any) -> ModelOutput:
        raise NotImplementedError(
            f"The {self.model_family} adapter is pending implementation; no states were predicted."
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
