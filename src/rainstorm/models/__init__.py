from __future__ import annotations

SUPERVISED_MODEL_MODULE = "rainstorm.modeling"

__all__ = ["SUPERVISED_MODEL_MODULE", "RainstormSupervisedModel"]


def __getattr__(name):
    if name == "RainstormSupervisedModel":
        from .supervised import RainstormSupervisedModel

        return RainstormSupervisedModel
    raise AttributeError(name)
