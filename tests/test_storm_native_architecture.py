from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_backend_contains_only_storm_domain_processing():
    backend = ROOT / "src" / "rainstorm" / "backend"
    removed_legacy_modules = {
        "BehavioralLabeler",
        "DrawROIs",
        "VideoHandling",
        "geometric_analysis",
        "modeling",
        "prepare_positions",
        "seize_labels",
    }

    assert not any((backend / name).exists() for name in removed_legacy_modules)

    source = "\n".join(
        path.read_text(encoding="utf-8") for path in backend.rglob("*.py")
    ).lower()
    for forbidden_import in ("tensorflow", "keras", "tkinter", "customtkinter", "cv2"):
        assert forbidden_import not in source


def test_pose_pipeline_runs_as_storm_steps():
    from storm import PipelineContext

    from rainstorm.backend import build_pose_pipeline

    positions = pd.DataFrame(
        {
            "nose_x": [2.0, 3.0, 4.0],
            "nose_y": [1.0, 1.0, 1.0],
            "body_x": [1.0, 1.0, 1.0],
            "body_y": [1.0, 1.0, 1.0],
            "unused": [9.0, 9.0, 9.0],
        }
    )
    runner = build_pose_pipeline(
        bodyparts=("nose", "body"),
        recenter_on="body",
        temporal_window=(1, 1, 1.0),
    )

    result = runner.run(PipelineContext(data=positions, dataset_id="example"))

    assert result.data.shape == (3, 3, 4)
    assert result.metadata["feature_names"] == [
        "nose_x",
        "nose_y",
        "body_x",
        "body_y",
    ]
    assert [execution.status for execution in result.executions] == [
        "completed",
        "completed",
        "completed",
    ]
    np.testing.assert_allclose(result.data[1, 1], [2.0, 0.0, 0.0, 0.0])


def test_supervised_model_is_a_storm_adapter_without_tensorflow_import():
    from rainstorm.models import RainstormSupervisedModel

    calls = []

    def predictor(model_path, values):
        calls.append((model_path, values.shape))
        return np.array([[0.2], [0.8]])

    model = RainstormSupervisedModel(
        {"model_path": "weights/example_wide.keras", "threshold": 0.5},
        predictor=predictor,
    )
    output = model.predict(np.zeros((2, 7, 12)))

    assert output.predictions == [0, 1]
    assert output.metadata["probabilities"] == [0.2, 0.8]
    assert calls == [(Path("weights/example_wide.keras"), (2, 7, 12))]

    source = (ROOT / "src" / "rainstorm" / "models" / "supervised.py").read_text(
        encoding="utf-8"
    )
    assert "tensorflow" not in source.lower()


def test_tensorflow_runtime_is_isolated_and_keeps_legacy_architecture():
    runtime = ROOT / "packages" / "rainstorm-supervised"
    pyproject = (runtime / "pyproject.toml").read_text(encoding="utf-8")
    architecture = (
        runtime
        / "src"
        / "rainstorm_supervised"
        / "architecture.py"
    ).read_text(encoding="utf-8")

    assert 'requires-python = ">=3.9,<3.10"' in pyproject
    assert '"tensorflow==2.10.1"' in pyproject
    for layer_name in (
        "conv1d_motion",
        "bn_conv",
        "dropout_conv",
        "global_max_pooling",
        "dense_8",
        "dense_4",
        "binary_out",
    ):
        assert layer_name in architecture
    assert "Bidirectional" in architecture
    assert "LSTM" in architecture


def test_temporal_pose_windows_stop_at_frame_discontinuities():
    from storm import PipelineContext
    from rainstorm.backend import build_pose_pipeline

    positions = pd.DataFrame({
        'nose_x':[10.0, 11.0, 50.0, 51.0], 'nose_y':[0.0]*4,
        'body_x':[0.0]*4, 'body_y':[0.0]*4,
    })
    result = build_pose_pipeline(bodyparts=('nose','body'),
        recenter_on='body', temporal_window=(1,1,1.0)).run(PipelineContext(
            data=positions, metadata={'frames':[0,1,10,11], 'sessions':['s']*4,
                'segments':['video']*4, 'partitions':['test']*4}))
    assert result.data.shape == (4,3,4)
    np.testing.assert_allclose(result.data[1,:,0], [10,11,11])
    np.testing.assert_allclose(result.data[2,:,0], [50,50,51])
