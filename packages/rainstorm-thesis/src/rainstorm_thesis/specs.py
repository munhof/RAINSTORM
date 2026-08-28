from __future__ import annotations

from storm import DataRef, RunSpec, StudySpec


def build_segmentation_study_spec(
    *,
    study_id: str,
    data: DataRef,
    model_family: str,
    variants: tuple[str, ...],
    n_states: int,
) -> StudySpec:
    return StudySpec(
        study_id=study_id,
        data=data,
        runs=tuple(
            RunSpec(
                run_id=f"{model_family}-{variant}",
                model_type=model_family,
                model_config={
                    "model_family": model_family,
                    "variant": variant,
                    "n_states": n_states,
                },
            )
            for variant in variants
        ),
    )


def build_supervised_study_spec(
    *,
    study_id: str,
    data: DataRef,
    model_path: str,
    thresholds: tuple[float, ...] = (0.35, 0.5, 0.65),
) -> StudySpec:
    return StudySpec(
        study_id=study_id,
        data=data,
        runs=tuple(
            RunSpec(
                run_id=f"supervised-threshold-{threshold:g}",
                model_type="supervised",
                model_config={
                    "model_path": model_path,
                    "threshold": threshold,
                },
            )
            for threshold in thresholds
        ),
    )
