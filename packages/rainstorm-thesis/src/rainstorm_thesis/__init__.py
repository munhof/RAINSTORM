from __future__ import annotations

from importlib import import_module

__all__ = [
    "ExternalSegmentationModel",
    "PoseDatasetConfig",
    "RainstormSupervisedProxyModel",
    "build_pose_dataset_loader",
    "build_segmentation_study_spec",
    "build_supervised_study_spec",
    "example_colabels_file",
    "example_pose_file",
    "load_pose_table",
]

_EXPORTS = {
    "ExternalSegmentationModel": ("models", "ExternalSegmentationModel"),
    "RainstormSupervisedProxyModel": ("models", "RainstormSupervisedProxyModel"),
    "PoseDatasetConfig": ("data", "PoseDatasetConfig"),
    "build_pose_dataset_loader": ("data", "build_pose_dataset_loader"),
    "example_colabels_file": ("data", "example_colabels_file"),
    "example_pose_file": ("data", "example_pose_file"),
    "load_pose_table": ("data", "load_pose_table"),
    "build_segmentation_study_spec": ("specs", "build_segmentation_study_spec"),
    "build_supervised_study_spec": ("specs", "build_supervised_study_spec"),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    return getattr(import_module(f"rainstorm_thesis.{module_name}"), attribute)
