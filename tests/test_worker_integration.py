from __future__ import annotations

import os
import subprocess
import sys

import pytest


class RecordingGroupModel:
    def __init__(self, config):
        self.config = config

    def fit_predict(self, inputs, constraints=()):
        from storm import ModelOutput

        return ModelOutput(
            [int(window[1][0]) for window in inputs],
            {"semantics": "test-only categorical states"},
        )


class ContextInferenceModel:
    def __init__(self, _config):
        pass

    def predict(self, _inputs):
        raise AssertionError("registered data context must be used for this adapter")

    def predict_with_context(self, inputs, data, observation_indices):
        from storm import ModelOutput

        assert data["sessions"] == ["mouse-a", "mouse-a"]
        assert observation_indices == [0, 1]
        return ModelOutput(
            [len(inputs), len(data["source_files"])],
            {"semantics": "context-bound predictions"},
        )


def build_catalog():
    from storm.suite import default_catalog

    from rainstorm_thesis.plugin import register

    catalog = default_catalog()
    register(catalog)
    return catalog


def test_catalog_registers_recoverable_historical_vame_recipes():
    catalog = build_catalog()

    native = catalog.recipe_presets['vame_native_pose_ego']
    assert native['model'] == 'vame_native'
    assert native['config'] == {
        'n_states': 50, 'latent_dim': 20, 'epochs': 25,
        'batch_size': 256, 'kld_weight': 0.5, 'seed': 156,
    }
    bodyparts = [
        'nose', 'left_ear', 'right_ear', 'head', 'neck', 'body',
        'left_shoulder', 'right_shoulder', 'left_midside', 'right_midside',
        'left_hip', 'right_hip', 'tail_base', 'tail_mid', 'tail_end',
    ]
    assert [step['type'] for step in native['steps']] == [
        'pose.select_coordinates', 'pose.likelihood_filter', 'pose.recenter',
        'pose.orient_coordinates', 'pose.temporal_windows',
    ]
    assert native['steps'][0]['config']['names'] == [
        name for part in bodyparts for name in (f'{part}_x', f'{part}_y')
    ]
    assert native['steps'][1]['config'] == {
        'threshold': 0.6, 'bodyparts': bodyparts,
    }
    assert native['steps'][2]['config'] == {
        'center_bodypart': 'body', 'bodyparts': bodyparts,
    }
    assert native['steps'][3]['config'] == {
        'from_bodypart': 'nose', 'toward_bodypart': 'body',
        'target_angle_degrees': 90,
    }
    assert native['steps'][4]['config']['offsets'] == list(range(-9, 11))
    assert 'filtrado anatómico/por velocidad' in native['description']
    assert 'Tesis_Facu' in native['provenance']
    assert 'input_builder.py' in native['provenance']
    assert 'no están verificados' in native['status']

    official = catalog.recipe_presets['vame_official_ego_roi']
    assert official['model'] == 'vame_official'
    assert official['config']['segmentation_algorithm'] == 'kmeans'
    assert official['config']['config_kwargs']['time_window'] == 19
    assert official['config']['config_kwargs']['learning_rate'] == 0.0005
    assert official['steps'] == []
    assert 'análisis contextual posterior' in official['description']


def test_native_vame_device_config_automatically_uses_available_accelerator():
    component = build_catalog().get('vame_native')
    device = component.schema['properties']['device']

    assert device['default'] == 'auto'
    assert device['enum'] == ['auto', 'cpu', 'cuda']


def test_native_vame_auto_device_prefers_gpu_and_explicit_gpu_fails_clearly(monkeypatch):
    torch = pytest.importorskip('torch')
    from rainstorm_thesis.vame_native_runtime import resolve_device

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    assert str(resolve_device('auto')) == 'cuda'

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    assert str(resolve_device('auto')) == 'cpu'
    with pytest.raises(RuntimeError, match='GPU device was requested'):
        resolve_device('cuda')


def test_native_vame_finishes_a_cpu_checkpoint_on_rocm_torch_version():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('Requires a CUDA or ROCm worker for the continuation phase')
    import numpy as np

    from rainstorm_thesis.vame_native_runtime import train_native_vame

    values = np.asarray([
        [[float(index), 0.0], [float(index) + 0.1, 0.2],
         [float(index) + 0.2, 0.3]]
        for index in range(8)
    ], dtype=np.float32)
    config = {
        'n_states': 2, 'latent_dim': 2, 'hidden_dim': 4, 'epochs': 1,
        'batch_size': 2, 'learning_rate': 0.001, 'kld_weight': 0.5,
        'seed': 7, 'device': 'cpu', 'num_threads': 1,
    }
    checkpoints = []
    train_native_vame(values, config, checkpoint=checkpoints.append)
    checkpoint = checkpoints[-1]
    checkpoint['torch_version'] = '2.14.0+cpu'

    config['device'] = 'cuda'
    result = train_native_vame(values, config, resume_state=checkpoint)

    assert result['config']['device'] == 'cuda'
    assert len(result['labels']) == len(values)


def test_native_vame_rejects_torch_version_change_before_final_epoch():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('Requires a CUDA or ROCm worker for the continuation phase')
    import numpy as np

    from rainstorm_thesis.vame_native_runtime import train_native_vame

    values = np.asarray([
        [[float(index), 0.0], [float(index) + 0.1, 0.2],
         [float(index) + 0.2, 0.3]]
        for index in range(8)
    ], dtype=np.float32)
    config = {
        'n_states': 2, 'latent_dim': 2, 'hidden_dim': 4, 'epochs': 2,
        'batch_size': 2, 'learning_rate': 0.001, 'kld_weight': 0.5,
        'seed': 7, 'device': 'cpu', 'num_threads': 1,
    }
    checkpoints = []

    def stop_after_first_epoch(state):
        checkpoints.append(state)
        raise RuntimeError('pause before final epoch')

    with pytest.raises(RuntimeError, match='pause before final epoch'):
        train_native_vame(values, config, checkpoint=stop_after_first_epoch)
    checkpoint = checkpoints[-1]
    checkpoint['torch_version'] = '2.14.0+cpu'
    config['device'] = 'cuda'

    with pytest.raises(ValueError, match='checkpoint is incompatible'):
        train_native_vame(values, config, resume_state=checkpoint)


def test_compose_gives_studio_the_supervised_worker_runtime_default():
    from pathlib import Path

    compose = (Path(__file__).parents[1] / "compose.storm-plugin.yaml").read_text()
    studio = compose.split("  studio:", 1)[1].split("  worker:", 1)[0]
    worker = compose.split("  worker:", 1)[1]
    assert "RAINSTORM_SUPERVISED_PYTHON: /opt/workspace/RAINSTORM/supervised-runtime/bin/python" in studio
    assert "../RAINSTORM/examples/models/trained_models:/opt/rainstorm/examples/models/trained_models:ro,Z" in worker


def test_compose_mounts_labeled_nor_benchmark_into_worker():
    from pathlib import Path

    compose = (Path(__file__).parents[1] / "compose.storm-plugin.yaml").read_text()
    worker = compose.split("  worker:", 1)[1]
    assert "../RAINSTORM/examples/NOR:/opt/workspace/RAINSTORM/examples/NOR:ro,Z" in worker


def test_scientific_worker_image_installs_ffmpeg_for_video_previews():
    from pathlib import Path

    containerfile = (Path(__file__).parents[1] / "Containerfile.storm-worker").read_text()
    assert "apt-get install -y --no-install-recommends libgomp1 ffmpeg" in containerfile


def test_catalog_can_register_supervised_models_without_numpy_installed():
    script = """
import importlib.abc
import sys

class BlockNumpy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "numpy" or fullname.startswith("numpy."):
            raise ModuleNotFoundError("NumPy is intentionally unavailable in Studio")

sys.meta_path.insert(0, BlockNumpy())
from storm.suite import default_catalog
from rainstorm_thesis.plugin import register

catalog = default_catalog()
register(catalog)
assert {"supervised_simple", "supervised_wide"} <= {
    component["name"] for component in catalog.describe()
}
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )

    assert result.returncode == 0, result.stderr


def test_rainstorm_plugin_registers_models_h5_preprocessing_and_visuals():
    catalog = build_catalog()

    assert {"dlc_h5", "dlc_csv"} <= set(catalog.connectors)
    assert {"vame_native", "vame_official"} <= {
        component["name"] for component in catalog.describe()
    }
    assert "infer" in catalog.get("vame_official").capabilities
    assert {
        "pose.select_coordinates",
        "pose.recenter",
        "pose.orient_coordinates",
        "pose.likelihood_filter",
        "pose.temporal_windows",
    } <= set(catalog.steps.available)
    assert {"rainstorm_pose_timeline", "rainstorm_state_timeline"} <= set(
        catalog.visualizations.available
    )
    for model_name in ("vame_native", "vame_official"):
        model = catalog.build(model_name, {})
        assert callable(model.fit_predict)
    official_config = catalog.normalize("vame_official", {})
    assert official_config["config_kwargs"]["n_clusters"] == 50
    assert official_config["config_kwargs"]["time_window"] == 19
    assert "sesiones de pose registradas" in catalog.get("vame_official").schema[
        "properties"
    ]["pose_paths"]["description"]


def test_official_vame_rejects_epoch_budget_that_upstream_would_not_execute():
    from rainstorm_thesis.vame_models import VAMEOfficialModel

    with pytest.raises(ValueError, match="max_epochs must be at least 2"):
        VAMEOfficialModel({"config_kwargs": {"max_epochs": 1}})


def test_registered_temporal_preprocessing_keeps_session_frame_and_partition_edges():
    from storm.suite import transform_aligned

    catalog = build_catalog()
    values = [[float(index)] for index in range(8)]
    metadata = {
        "frames": [0, 1, 2, 3, 10, 11, 12, 13],
        "sessions": ["mouse-a"] * 8,
        "segments": ["clip-1"] * 8,
        "partitions": ["train"] * 8,
        "reserved_evaluation": [False] * 8,
    }

    prepared, fitted, centers = transform_aligned(
        values,
        list(range(8)),
        [{"type": "pose.temporal_windows", "config": {"offsets": [-1, 0, 1]}}],
        catalog=catalog,
        context_metadata=metadata,
    )

    assert centers == [1, 2, 5, 6]
    assert prepared == [
        [[0.0], [1.0], [2.0]],
        [[1.0], [2.0], [3.0]],
        [[4.0], [5.0], [6.0]],
        [[5.0], [6.0], [7.0]],
    ]
    assert fitted[0]["type"] == "pose.temporal_windows"


def test_registered_pose_selection_recentering_and_likelihood_filter_are_traced():
    from storm.suite import transform_aligned

    catalog = build_catalog()
    values = [[5.0, 7.0, 1.0, 1.0], [10.0, 15.0, 2.0, 3.0]]
    metadata = {
        "feature_names": ["nose_x", "nose_y", "body_x", "body_y"],
        "likelihood_bodyparts": ["nose", "body"],
        "likelihoods": [[0.9, 0.9], [0.2, 0.9]],
        "frames": [0, 1],
        "sessions": ["mouse-a"] * 2,
        "segments": ["clip-1"] * 2,
        "partitions": ["train"] * 2,
        "reserved_evaluation": [False] * 2,
    }
    steps = [
        {"type": "pose.select_coordinates", "config": {"names": [
            "nose_x", "nose_y", "body_x", "body_y"]}},
        {"type": "pose.recenter", "config": {
            "center_indices": [2, 3], "coordinate_pairs": [[0, 1], [2, 3]]}},
        {"type": "pose.likelihood_filter", "config": {
            "threshold": 0.6, "coordinate_pairs": [[0, 1], [2, 3]]}},
    ]

    prepared, fitted, indices = transform_aligned(
        values, [0, 1], steps, catalog=catalog, context_metadata=metadata
    )

    assert indices == [0, 1]
    assert prepared == [[4.0, 6.0, 0.0, 0.0], [4.0, 6.0, 0.0, 0.0]]
    assert [step["type"] for step in fitted] == [step["type"] for step in steps]


def test_dlc_label_mask_is_serializable_by_storm(monkeypatch, tmp_path):
    import pandas as pd
    from rainstorm_thesis.data import load_dlc_h5
    from storm.config import json_compatible

    pose = tmp_path / "mouseDLC_pose.h5"
    pose.touch()
    labels = tmp_path / "labels.csv"
    pd.DataFrame({"Frame": [1, 2], "explore": [1, 0], "other": [0, 1]}).to_csv(
        labels, index=False)
    columns = pd.MultiIndex.from_tuples([
        ("DLC", "nose", "x"), ("DLC", "nose", "y"),
        ("DLC", "nose", "likelihood"),
    ])
    table = pd.DataFrame([[1.0, 2.0, 0.9], [3.0, 4.0, 0.8]],
                         index=[0, 1], columns=columns)
    monkeypatch.setattr(pd, "read_hdf", lambda *_args, **_kwargs: table)

    data = load_dlc_h5({"pose_path": str(pose), "labels_path": str(labels)})

    assert data["evaluation_mask"] == [True, True]
    assert all(type(value) is bool for value in data["evaluation_mask"])
    json_compatible(data)


def test_dlc_h5_connector_flattens_bodyparts_and_preserves_video_frames(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("tables")
    import numpy as np

    catalog = build_catalog()
    path = tmp_path / "mouse-aDLC_resnet.h5"
    columns = pd.MultiIndex.from_tuples(
        [
            ("DLC", "nose", "x"),
            ("DLC", "nose", "y"),
            ("DLC", "nose", "likelihood"),
            ("DLC", "body", "x"),
            ("DLC", "body", "y"),
        ],
        names=["scorer", "bodyparts", "coords"],
    )
    table = pd.DataFrame(
        [[1, 2, 0.9, 3, 4], [2, 3, 0.8, 4, 5], [8, 9, 0.9, 5, 6]],
        index=[0, 1, 10],
        columns=columns,
    )
    table.to_hdf(path, key="df")
    video = tmp_path / "camera-file.mp4"
    video.write_bytes(b"video")
    labels = tmp_path / "manual.csv"
    pd.DataFrame({"Frame": [6, 7], "explore": [1, 0], "other": [0, 1]}).to_csv(labels, index=False)

    data = catalog.connectors["dlc_h5"](
        {
            "pose_paths": [str(path)],
            "pose_session_ids": ["experiment-one"],
            "video_paths": [str(video)],
            "labels_by_session": {"experiment-one": str(labels)},
            "label_frame_reference": "video",
            "video_frame_offsets_by_session": {"experiment-one": 5},
            "video_discontinuities_by_session": {"experiment-one": [6]},
            "reserved_evaluation_ranges_by_session": {
                "experiment-one": {"start": 1, "stop": 2}},
            "fps": 30.0,
            "key": "df",
        }
    )

    assert data["inputs"] == [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [8.0, 9.0, 5.0, 6.0]]
    assert data["feature_names"] == ["nose_x", "nose_y", "body_x", "body_y"]
    assert data["frames"] == [0, 1, 10]
    assert data["video_frames"] == [5, 6, 15]
    assert data["reserved_evaluation"] == [False, True, False]
    assert data["partitions"] == ["train", "test", "train"]
    assert data["train"] == [0, 2]
    assert data["test"] == [1]
    assert data["segments"] == [
        "experiment-one:segment-0", "experiment-one:segment-1",
        "experiment-one:segment-2",
    ]
    assert data["sessions"] == ["experiment-one"] * 3
    assert data["source_files"][0]["video_path"] == str(video.resolve())
    assert data["source_files"][0]["session_id"] == "experiment-one"
    np.testing.assert_allclose(data["likelihoods"], [[0.9], [0.8], [0.9]])
    assert data["targets"] == [0, 1, None]
    assert data["evaluation_mask"] == [True, True, False]
    assert all(type(value) is bool for value in data["evaluation_mask"])
    from storm.config import json_compatible
    json_compatible(data)
    assert data["data_fingerprint"].startswith("sha256:")
    assert "groups" not in data


def test_dlc_session_with_no_saved_partition_stays_unassigned():
    from rainstorm_thesis.data import _session_partition

    assert _session_partition(
        "new-session", {"trained-session"}, {"validation-session"}, {"held-out"}
    ) == "unassigned"


def test_rainstorm_state_visualization_returns_labeled_svg():
    from storm import ModelOutput
    from storm.visualization import VisualizationManager, VisualizationRequest, VisualizationSpec

    catalog = build_catalog()
    rendered = VisualizationManager(catalog.visualizations).render(
        VisualizationSpec("rainstorm_state_timeline"),
        VisualizationRequest(
            data=[[0.0], [1.0], [2.0]],
            output=ModelOutput([2, 2, 7], {"semantics": "VAME motif IDs"}),
            metadata={"indices": [4, 5, 9], "frames": [100, 101, 120]},
        ),
    )

    assert rendered.media_type == "image/svg+xml"
    assert "VAME motif IDs" in rendered.content
    assert "frame 120" in rendered.content
    assert "state 7" in rendered.content


def test_pose_visualization_uses_zero_offset_for_asymmetric_windows():
    from storm.visualization import VisualizationManager, VisualizationRequest, VisualizationSpec

    catalog = build_catalog()
    rendered = VisualizationManager(catalog.visualizations).render(
        VisualizationSpec("rainstorm_pose_timeline"),
        VisualizationRequest(
            data=[
                [[-2, 0], [0, 0], [0, 1], [0, 2], [0, 3]],
                [[-1, 0], [1, 1], [1, 0], [1, 2], [1, 3]],
            ],
            metadata={
                "offsets": [-2, 0, 1, 2, 3],
                "indices": [10, 11],
                "frames": [100, 101],
            },
        ),
    )

    assert 'points="30.0,225.0 770.0,35.0"' in rendered.content


def test_native_vame_adapter_calls_scientific_runtime(monkeypatch):
    import sys
    import types

    from rainstorm_thesis.vame_models import VAMENativeModel

    calls = []
    runtime = types.ModuleType("rainstorm_thesis.vame_native_runtime")
    runtime.train_native_vame = lambda values, config: (
        calls.append((values, config)) or {
            "labels": [0, 1], "embeddings": [[0.0], [1.0]],
            "training_history": [{"epoch": 1}],
        }
    )
    runtime.predict_native_vame = lambda values, checkpoint: ([1, 0], [[1.0], [0.0]])
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    model = VAMENativeModel({"n_states": 2, "latent_dim": 1})

    output = model.fit_predict([[[0.0], [1.0]], [[1.0], [2.0]]])
    replay = model.predict([[[1.0], [2.0]], [[0.0], [1.0]]])

    assert output.predictions == [0, 1]
    assert output.metadata["backend"] == "rainstorm_native_vame"
    assert replay.predictions == [1, 0]
    assert len(calls) == 1


def test_native_vame_saved_run_can_infer_with_its_fitted_state(tmp_path, monkeypatch):
    import sys
    import types

    from storm.suite import execute, infer

    runtime = types.ModuleType("rainstorm_thesis.vame_native_runtime")
    def train_native_vame(values, config, *, resume_state=None, checkpoint=None):
        if checkpoint:
            checkpoint({"epoch": 1, "training_history": [{"epoch": 1}]})
        return {
            "labels": [0, 1], "embeddings": [[0.0], [1.0]],
            "training_history": [{"epoch": 1}],
        }

    runtime.train_native_vame = train_native_vame
    runtime.predict_native_vame = lambda values, checkpoint: (
        [1] * len(values), [[1.0] for _ in values]
    )
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    catalog = build_catalog()
    values = [[[0.0], [1.0]], [[1.0], [2.0]]]

    result = execute({
        "model": "vame_native", "connector": "json_records",
        "config": {"n_states": 2, "latent_dim": 1, "epochs": 1},
        "preapplied_steps": ["pose.temporal_windows"],
        "data": {"inputs": values, "train": [0, 1], "test": []},
    }, tmp_path, "native-vame", catalog)

    assert "infer" in result["capabilities"]
    assert infer(result, [values[1]], tmp_path, catalog) == [1]


def test_official_vame_adapter_calls_official_pipeline_and_maps_labels(tmp_path, monkeypatch):
    import io
    import sys
    import types

    import numpy as real_numpy
    from ruamel.yaml import YAML

    from rainstorm_thesis.vame_models import VAMEOfficialModel

    class Array(list):
        def reshape(self, *_shape):
            return self

        def tolist(self):
            return list(self)

    numpy = types.ModuleType("numpy")
    numpy.load = lambda _path: Array([2])
    numpy.asarray = lambda values: Array(values)
    numpy.ndarray = real_numpy.ndarray
    numpy.generic = real_numpy.generic
    monkeypatch.setitem(sys.modules, "numpy", numpy)

    calls = []
    written_configs = []
    vame = types.ModuleType("vame")

    def init_new_project(*, project_name, working_directory, **kwargs):
        project = tmp_path / "official" / project_name
        project.mkdir(parents=True)
        (project / "config.yaml").write_text("project: test\n")
        kwargs["config_kwargs"]["keypoints"] = [real_numpy.str_("nose")]
        calls.append("init")
        return str(project), str(project)

    def segment_session(*, config):
        project = tmp_path / "official" / config["project_name"]
        session = config["session_names"][0]
        algorithm = config["segmentation_algorithms"][0]
        clusters = config["n_clusters"]
        result = project / "results" / session / "VAME" / f"{algorithm}-{clusters}"
        result.mkdir(parents=True)
        (result / f"{clusters}_{algorithm}_label_{session}.npy").write_bytes(b"fake labels")
        calls.append("segment")

    vame.init_new_project = init_new_project
    def write_config(_path, config):
        YAML().dump(config, io.StringIO())
        written_configs.append(dict(config))

    vame.write_config = write_config
    vame.read_config = lambda _path: {
        "project_name": "rainstorm_vame_official",
        "session_names": ["mouse-aDLC_pose"],
        "keypoints": [real_numpy.str_("nose")],
    }

    def preprocessing(**kwargs):
        assert kwargs["config"]["pose_confidence"] == 0.6
        kwargs["config"]["keypoints"] = [real_numpy.str_("nose")]
        calls.append("preprocess")

    def create_trainset(**kwargs):
        YAML().dump(kwargs["config"], io.StringIO())
        calls.append("trainset")

    vame.preprocessing = preprocessing
    vame.create_trainset = create_trainset
    vame.train_model = lambda **_kwargs: calls.append("train")
    vame.segment_session = segment_session
    monkeypatch.setitem(sys.modules, "vame", vame)

    pose = tmp_path / "mouse-aDLC_pose.h5"
    pose.write_bytes(b"registered H5 source")
    model = VAMEOfficialModel({
        "pose_paths": [str(pose)],
        "working_directory": str(tmp_path / "official"),
        "config_kwargs": {"time_window": 3},
    })

    output = model.fit_predict([[0.0], [1.0], [2.0]])
    assert output.predictions == [None, 2, None]
    assert output.metadata["backend"] == "vame_py_official"
    assert output.metadata["prediction_mask"] == [False, True, False]
    with pytest.raises(ValueError, match="registered pose files and session metadata"):
        model.predict([[0.0], [1.0], [2.0]])
    assert written_configs[0]["pose_confidence"] == 0.6
    assert "confidence" not in written_configs[0]
    assert calls == ["init", "preprocess", "trainset", "train", "segment"]


def test_official_vame_applies_saved_encoder_and_shared_kmeans_to_new_sessions(
        tmp_path, monkeypatch):
    import json
    import sys
    import types
    from pathlib import Path

    import numpy as np

    from rainstorm_thesis.vame_official_runtime import predict_official_vame

    training_root = tmp_path / "trained"
    training_config = {
        "project_name": "trained", "project_path": str(training_root),
        "session_names": ["mouse-train"], "model_name": "VAME",
        "num_features": 2, "keypoints": ["nose"], "n_clusters": 2,
        "time_window": 3, "zdims": 1, "egocentric_data": False,
        "individual_segmentation": False, "all_data": "yes",
        "project_random_state": 7,
    }
    training_root.mkdir()
    (training_root / "config.yaml").write_text(json.dumps(training_config))
    model_file = training_root / "model" / "best_model" / "VAME_trained.pkl"
    model_file.parent.mkdir(parents=True)
    model_file.write_bytes(b"saved VAME weights")
    metadata_file = training_root / "data" / "train" / "metadata.json"
    metadata_file.parent.mkdir(parents=True)
    feature_mapping = [{"index": 0, "keypoint": "nose", "coordinate": "x"},
                       {"index": 1, "keypoint": "nose", "coordinate": "y"}]
    metadata_file.write_text(json.dumps({
        "feature_mapping": feature_mapping,
        "parameters": {"keypoints_used": ["nose"], "read_from_variable": "position_processed",
                       "extra_features": []},
    }))
    centers = np.asarray([[0.0], [10.0]])
    centers_file = (training_root / "results" / "mouse-train" / "VAME" / "kmeans-2"
                    / "cluster_center_mouse-train.npy")
    centers_file.parent.mkdir(parents=True)
    np.save(centers_file, centers)

    pose_path = tmp_path / "raw-export-name.h5"
    pose_path.write_bytes(b"registered pose source")
    created_configs = {}
    vame = types.ModuleType("vame")
    vame.__path__ = []
    vame.read_config = lambda path: json.loads(Path(path).read_text())

    def init_new_project(*, project_name, working_directory, poses_estimations,
                         videos, source_software, fps, config_kwargs):
        assert all(Path(path).name == Path(path).resolve().name
                   for path in poses_estimations)
        project_directory = Path(working_directory) / project_name
        for directory in ("model/best_model", "data/processed", "results", "states"):
            (project_directory / directory).mkdir(parents=True, exist_ok=True)
        config = {
            **training_config, **config_kwargs,
            "project_name": project_name,
            "project_path": str(project_directory),
            # Official VAME resolves staged symlinks before naming sessions.
            "session_names": [Path(path).resolve().stem for path in poses_estimations],
            "num_features": 2, "keypoints": ["nose"],
        }
        config_path = project_directory / "config.yaml"
        config_path.write_text(json.dumps(config))
        created_configs[project_name] = config
        return str(config_path), config

    vame.init_new_project = init_new_project
    vame.preprocessing = lambda **_kwargs: "position_processed"
    monkeypatch.setitem(sys.modules, "vame", vame)

    io_module = types.ModuleType("vame.io")
    io_module.__path__ = []
    load_poses = types.ModuleType("vame.io.load_poses")
    load_poses.read_pose_estimation_file = lambda **_kwargs: (None, None, object())
    preprocessing = types.ModuleType("vame.preprocessing")
    preprocessing.__path__ = []
    to_model = types.ModuleType("vame.preprocessing.to_model")
    to_model.format_xarray_for_rnn = lambda **_kwargs: (None, {
        "feature_mapping": feature_mapping,
        "parameters": {"total_features": 2},
    })
    monkeypatch.setitem(sys.modules, "vame.io", io_module)
    monkeypatch.setitem(sys.modules, "vame.io.load_poses", load_poses)
    monkeypatch.setitem(sys.modules, "vame.preprocessing", preprocessing)
    monkeypatch.setitem(sys.modules, "vame.preprocessing.to_model", to_model)

    analysis = types.ModuleType("vame.analysis")
    analysis.__path__ = []
    pose_segmentation = types.ModuleType("vame.analysis.pose_segmentation")

    def embed(config, sessions, fixed, overwrite):
        assert sessions == ["raw-export-name"]
        assert fixed is False and overwrite is True
        inference_project = Path(config["project_path"])
        assert (inference_project / "model" / "best_model"
                / f"VAME_{config['project_name']}.pkl").is_file()
        assert (inference_project / "data" / "train" / "metadata.json").is_file()
        assert (inference_project / "results" / "raw-export-name" / "VAME").is_dir()
        return [np.asarray([[0.2], [9.8]])]

    pose_segmentation.embed_latent_vectors_optimized = embed
    monkeypatch.setitem(sys.modules, "vame.analysis", analysis)
    monkeypatch.setitem(sys.modules, "vame.analysis.pose_segmentation", pose_segmentation)

    model_config = {
        "project_name": "trained", "working_directory": str(tmp_path),
        "segmentation_algorithm": "kmeans", "source_software": "DeepLabCut",
        "fps": 30.0, "centered_reference_keypoint": "body",
        "orientation_reference_keypoint": "nose",
        "config_kwargs": {"n_clusters": 2, "time_window": 3},
    }
    data = {
        "sessions": ["mouse-new"] * 4, "frames": [0, 1, 2, 3],
        "segments": ["segment-1"] * 4, "fps": 30.0,
        "source_files": [{
            "session_id": "mouse-new", "path": str(pose_path),
            "frame_count": 4, "selected_frame_count": 4, "video_path": "",
        }],
    }

    output = predict_official_vame(model_config, [[0.0], [1.0], [2.0], [3.0]],
                                   data, [0, 1, 2, 3])

    assert output.predictions == [None, 0, 1, None]
    assert output.metadata["prediction_mask"] == [False, True, True, False]
    assert output.metadata["discretizer_scope"] == "shared_training_model"
    assert created_configs[Path(output.metadata["inference_project"]).name][
        "segmentation_algorithms"] == ["kmeans"]
    np.testing.assert_array_equal(np.load(centers_file), centers)


def test_official_vame_project_defaults_inside_storm_workspace(tmp_path, monkeypatch):
    from rainstorm_thesis.vame_models import VAMEOfficialModel

    workspace = tmp_path / "shared-workspace"
    monkeypatch.setenv("STORM_WORKSPACE", str(workspace))
    model = VAMEOfficialModel({"working_directory": ".storm/rainstorm/vame_official"})

    assert model.config["working_directory"] == str(workspace / "rainstorm/vame_official")


def test_worker_engine_source_points_to_named_engine_package():
    import tomllib
    from pathlib import Path

    package_root = Path(__file__).parents[1] / "packages" / "rainstorm-thesis"
    package_manifest = tomllib.loads((package_root / "pyproject.toml").read_text())
    package_source = package_manifest["tool"]["uv"]["sources"]["storm-traceable"]
    engine_root = (package_root / package_source["path"]).resolve()
    engine_manifest = tomllib.loads((engine_root / "pyproject.toml").read_text())
    worker_root = package_root.parents[1] / "envs" / "vame_worker"
    worker_manifest = tomllib.loads((worker_root / "pyproject.toml").read_text())
    worker_source = worker_manifest["tool"]["uv"]["sources"]["storm-traceable"]

    assert engine_manifest["project"]["name"] == "storm-traceable"
    assert package_source["editable"] is True
    assert worker_source["editable"] is True
    assert (worker_root / worker_source["path"]).resolve() == engine_root


def test_official_vame_extra_is_limited_to_supported_worker_python():
    import tomllib
    from pathlib import Path

    package_root = Path(__file__).parents[1] / "packages" / "rainstorm-thesis"
    manifest = tomllib.loads((package_root / "pyproject.toml").read_text())
    official_dependencies = manifest["project"]["optional-dependencies"]["vame_official"]

    assert official_dependencies == [
        "vame-py>=0.14.0; python_version >= '3.12' and sys_platform != 'win32'"
    ]


def test_worker_uses_cpu_torch_and_does_not_build_a_dummy_package():
    import tomllib
    from pathlib import Path

    worker_root = Path(__file__).parents[1] / "envs" / "vame_worker"
    manifest = tomllib.loads((worker_root / "pyproject.toml").read_text())

    assert manifest["tool"]["uv"]["package"] is False
    assert manifest["tool"]["uv"]["sources"]["torch"]["index"] == "pytorch-cpu"
    assert {index["name"] for index in manifest["tool"]["uv"]["index"]} >= {
        "pytorch-cpu"
    }
    assert any(dependency.startswith("torch>=") for dependency in manifest["project"]["dependencies"])


def test_official_vame_binds_complete_training_sessions_and_blocks_reserved_frames(tmp_path):
    from rainstorm_thesis.vame_models import VAMEOfficialModel

    pose = tmp_path / "mouse-aDLC_pose.h5"
    pose.write_bytes(b"pose source")
    source = {
        "path": str(pose),
        "session_id": "mouse-a",
        "frame_count": 3,
        "selected_frame_count": 3,
        "video_path": "",
    }
    data = {
        "source_files": [source],
        "sessions": ["mouse-a"] * 3,
        "partitions": ["train"] * 3,
        "segments": ["clip-1"] * 3,
        "reserved_evaluation": [False] * 3,
    }

    model = VAMEOfficialModel()
    model.bind_data(data, [0, 1, 2])
    assert model.config["pose_paths"] == [str(pose.resolve())]

    data["reserved_evaluation"] = [False, True, False]
    with pytest.raises(ValueError, match="reserved evaluation"):
        VAMEOfficialModel().bind_data(data, [0, 1, 2])


def test_official_vame_tracks_lengths_for_multiple_whole_sessions(tmp_path):
    from pathlib import Path

    from rainstorm_thesis.vame_models import VAMEOfficialModel

    sources = []
    for session, frame_count in (("mouse-a", 2), ("mouse-b", 3)):
        pose = tmp_path / f"{session}DLC_pose.h5"
        pose.write_bytes(b"pose source")
        sources.append({
            "path": str(pose), "session_id": session, "frame_count": frame_count,
            "selected_frame_count": frame_count, "video_path": "",
        })
    data = {
        "source_files": sources,
        "sessions": ["mouse-a", "mouse-a", "mouse-b", "mouse-b", "mouse-b"],
        "partitions": ["train"] * 5,
        "segments": ["clip-a", "clip-a", "clip-b", "clip-b", "clip-b"],
        "reserved_evaluation": [False] * 5,
    }

    model = VAMEOfficialModel()
    model.bind_data(data, [0, 1, 2, 3, 4])

    assert model._session_lengths == [2, 3]
    assert model.config["pose_paths"] == [str(Path(source["path"]).resolve()) for source in sources]


def test_native_vame_trains_and_replays_small_pose_windows_when_runtime_is_installed():
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    import numpy as np

    from rainstorm_thesis.vame_models import VAMENativeModel

    windows = np.asarray([
        [[0.0, 0.0], [0.1, 0.0], [0.2, 0.1]],
        [[0.1, 0.0], [0.2, 0.1], [0.3, 0.1]],
        [[2.0, 2.0], [2.1, 2.0], [2.2, 2.1]],
        [[2.1, 2.0], [2.2, 2.1], [2.3, 2.1]],
    ])
    model = VAMENativeModel({
        "n_states": 2, "latent_dim": 2, "hidden_dim": 4,
        "epochs": 1, "batch_size": 4, "seed": 7,
    })

    trained = model.fit_predict(windows)
    replay = model.predict(windows)

    assert len(trained.predictions) == len(windows)
    assert len(replay.predictions) == len(windows)
    assert trained.metadata["embedding_shape"] == [4, 2]
    assert trained.metadata["training_history"][0]["epoch"] == 1


def test_native_vame_encodes_training_and_inference_outputs_in_batches(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    import numpy as np

    from rainstorm_thesis.vame_native_runtime import (
        NativeVAMENetwork,
        predict_native_vame,
        train_native_vame,
    )

    values = np.asarray([
        [[float(index), 0.0], [float(index) + 0.1, 0.2],
         [float(index) + 0.2, 0.3]]
        for index in range(8)
    ], dtype=np.float32)
    config = {
        "n_states": 2, "latent_dim": 2, "hidden_dim": 4, "epochs": 1,
        "batch_size": 2, "learning_rate": 0.001, "kld_weight": 0.5,
        "seed": 7, "device": "cpu", "num_threads": 1,
    }
    encoded_batch_sizes = []
    original_encode = NativeVAMENetwork.encode

    def record_encode_batch(model, batch):
        encoded_batch_sizes.append(len(batch))
        return original_encode(model, batch)

    monkeypatch.setattr(NativeVAMENetwork, "encode", record_encode_batch)
    trained = train_native_vame(values, config)
    assert max(encoded_batch_sizes) <= config["batch_size"]
    assert trained["embeddings"].shape == (len(values), config["latent_dim"])

    encoded_batch_sizes.clear()
    labels, embeddings = predict_native_vame(values, trained)
    assert max(encoded_batch_sizes) <= config["batch_size"]
    assert len(labels) == len(embeddings) == len(values)


def test_native_vame_runtime_resumes_exactly_from_an_epoch_boundary():
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    import numpy as np

    from rainstorm_thesis.vame_native_runtime import train_native_vame

    values = np.asarray([
        [[0.0, 0.0], [0.1, 0.0], [0.2, 0.1]],
        [[0.1, 0.0], [0.2, 0.1], [0.3, 0.1]],
        [[2.0, 2.0], [2.1, 2.0], [2.2, 2.1]],
        [[2.1, 2.0], [2.2, 2.1], [2.3, 2.1]],
    ], dtype=np.float32)
    config = {
        "n_states": 2, "latent_dim": 2, "hidden_dim": 4, "epochs": 3,
        "batch_size": 2, "learning_rate": 0.001, "kld_weight": 0.5,
        "seed": 7, "device": "cpu", "num_threads": 1,
    }
    saved = {}

    def interrupt_after_first_epoch(state):
        saved["state"] = state
        if state["epoch"] == 1:
            raise RuntimeError("simulated worker interruption")

    with pytest.raises(RuntimeError, match="simulated worker interruption"):
        train_native_vame(values, config, checkpoint=interrupt_after_first_epoch)

    state = saved["state"]
    assert state["epoch"] == 1
    assert {"optimizer_state", "numpy_rng_state", "torch_rng_state"} <= set(state)
    resumed = train_native_vame(values, config, resume_state=state)
    uninterrupted = train_native_vame(values, config)

    assert resumed["training_history"] == uninterrupted["training_history"]
    assert resumed["labels"] == uninterrupted["labels"]
    np.testing.assert_array_equal(resumed["embeddings"], uninterrupted["embeddings"])
    for name, weights in uninterrupted["model_state"].items():
        np.testing.assert_array_equal(resumed["model_state"][name], weights)
    incompatible = dict(state, numpy_version="incompatible")
    with pytest.raises(ValueError, match="incompatible with this data, recipe, or runtime"):
        train_native_vame(values, config, resume_state=incompatible)


def test_native_vame_resumes_training_from_a_persisted_epoch_checkpoint(tmp_path, monkeypatch):
    import sys
    import types

    from rainstorm_thesis.vame_models import VAMENativeModel
    from storm.artifacts import FileArtifactStore
    from storm.suite import execute

    catalog = build_catalog()
    assert "checkpoint" in catalog.get("vame_native").capabilities
    catalog.connectors["test.pose_windows"] = lambda data: data
    runtime = types.ModuleType("rainstorm_thesis.vame_native_runtime")

    def train(values, config, *, resume_state=None, checkpoint=None):
        state = dict(resume_state or {"epoch": 0, "training_history": []})
        history = list(state["training_history"])
        for epoch in range(state["epoch"], config["epochs"]):
            history.append({"epoch": epoch + 1, "loss": float(config["epochs"] - epoch)})
            state = {"epoch": epoch + 1, "training_history": history}
            checkpoint(state)
        count = len(values)
        return {
            "labels": [index % config["n_states"] for index in range(count)],
            "embeddings": [[float(index), 0.0] for index in range(count)],
            "model_state": {},
            "cluster_centers": [[0.0, 0.0], [1.0, 1.0]],
            "input_spec": {"feature_dim": 2, "sequence_len": 3},
            "config": {"latent_dim": 2, "hidden_dim": 4, "device": "cpu"},
            "training_history": history,
            "seed": config["seed"],
        }

    runtime.train_native_vame = train
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    inputs = [
        [[0.0, 0.0], [0.1, 0.0], [0.2, 0.1]],
        [[0.1, 0.0], [0.2, 0.1], [0.3, 0.1]],
        [[2.0, 2.0], [2.1, 2.0], [2.2, 2.1]],
        [[2.1, 2.0], [2.2, 2.1], [2.3, 2.1]],
    ]
    spec = {
        "model": "vame_native",
        "connector": "test.pose_windows",
        "preapplied_steps": ["pose.temporal_windows"],
        "config": {
            "n_states": 2,
            "latent_dim": 2,
            "hidden_dim": 4,
            "epochs": 3,
            "batch_size": 4,
            "seed": 7,
        },
        "data": {"inputs": inputs, "train": [0, 1, 2, 3], "test": []},
    }
    original = VAMENativeModel.fit_predict_with_checkpoints

    class SimulatedWorkerInterruption(RuntimeError):
        pass

    def interrupt_after_checkpoint(self, values, constraints, save):
        def persist_then_interrupt(state):
            save(state)
            raise SimulatedWorkerInterruption("worker stopped after epoch checkpoint")

        return original(self, values, constraints, persist_then_interrupt)

    monkeypatch.setattr(VAMENativeModel, "fit_predict_with_checkpoints", interrupt_after_checkpoint)
    store_root = tmp_path / "interrupted"
    with pytest.raises(SimulatedWorkerInterruption, match="after epoch checkpoint"):
        execute(spec, store_root, "interrupted", catalog)

    store = FileArtifactStore(store_root)
    reference = store.resolve(kind="checkpoints", artifact_id="interrupted")
    checkpoint = store.load(reference)
    assert checkpoint["state"]["epoch"] == 1
    monkeypatch.setattr(VAMENativeModel, "fit_predict_with_checkpoints", original)

    resumed = execute(spec, store_root, "resumed", catalog, resume_from=reference)
    uninterrupted = execute(spec, tmp_path / "uninterrupted", "complete", catalog)

    assert resumed["predictions"] == uninterrupted["predictions"]
    assert resumed["output_metadata"]["training_history"] == uninterrupted["output_metadata"]["training_history"]
    assert [item["epoch"] for item in resumed["output_metadata"]["training_history"]] == [1, 2, 3]


def test_unsupervised_vame_can_train_when_some_reference_labels_are_masked():
    from storm.suite import validate_data

    data = {
        "inputs": [[0.0], [1.0], [2.0]],
        "targets": [0, None, 1],
        "evaluation_mask": [True, False, True],
        "train": [0, 1],
        "validation": [],
        "test": [2],
    }

    validate_data(data, numeric=False, allow_missing_targets=True)


def test_worker_executes_registered_pose_pipeline_and_keeps_sparse_labels_inspection_only(tmp_path):
    from storm.suite import Component, execute

    catalog = build_catalog()
    catalog.register(Component(
        "recording_group",
        RecordingGroupModel,
        ("group",),
        {"type": "object", "properties": {}},
    ))
    catalog.connectors["fixture_pose"] = lambda _config: {
        "inputs": [[float(index)] for index in range(8)],
        "targets": [0, 1, 2, 3, None, 5, 6, 7],
        "evaluation_mask": [True, True, True, True, False, True, True, True],
        "reserved_evaluation": [False, False, False, False, True, False, False, False],
        "train": [0, 1, 2, 3, 5, 6, 7],
        "validation": [],
        "test": [4],
        "frames": [0, 1, 2, 3, 4, 5, 6, 7],
        "sessions": ["mouse-a"] * 8,
        "segments": ["clip-1"] * 8,
        "partitions": ["train", "train", "train", "train", "test", "train", "train", "train"],
        "feature_names": ["nose_x"],
        "likelihoods": [[] for _ in range(8)],
        "likelihood_bodyparts": [],
        "taxonomy": ["explore", "other"],
    }

    result = execute(
        {
            "model": "recording_group",
            "connector": "fixture_pose",
            "data": {},
            "config": {},
            "steps": [{
                "type": "pose.temporal_windows",
                "config": {"offsets": [-1, 0, 1]},
            }],
            "metrics": [],
            "seed": 156,
        },
        tmp_path,
        "rainstorm-test-execution",
        catalog,
    )

    assert result["indices"] == [1, 2, 6]
    assert result["predictions"] == [1, 2, 6]
    assert result["metrics"] == {}
    assert result["evaluation_status"] == "inspection_only_no_group_metric"
    assert result["preparation_trace"]["train"][0]["step_type"] == "pose.temporal_windows"


def test_catalog_runs_imported_supervised_models_as_inference_only(monkeypatch, tmp_path):
    import numpy as np
    from pathlib import Path
    from rainstorm.models.supervised import RainstormSupervisedModel
    from storm.suite import execute

    monkeypatch.setattr(
        RainstormSupervisedModel,
        "_subprocess_predict",
        lambda self, model_path, values: np.array([0.2, 0.45, 0.8]),
    )
    catalog = build_catalog()
    examples = Path(__file__).parents[1] / "examples" / "models" / "trained_models"

    for model_name, shape, filename in (
        ("supervised_simple", (12,), "example_simple.keras"),
        ("supervised_wide", (7, 12), "example_wide.keras"),
    ):
        component = catalog.get(model_name)
        assert component.capabilities == ("infer",)
        config = catalog.normalize(model_name, {})
        assert Path(config["model_path"]) == examples / filename

        result = execute(
            {
                "operation": "infer",
                "model": model_name,
                "connector": "json_records",
                "config": {},
                "metrics": [],
                "data": {
                    "inputs": np.zeros((3, *shape)).tolist(),
                    "train": [],
                    "validation": [],
                    "test": [0, 1, 2],
                },
            },
            tmp_path,
            model_name,
            catalog,
        )

        assert result["predictions"] == [0, 0, 1]
        assert result["output_metadata"]["task"] == "binary_classification"
        assert result["output_metadata"]["probabilities"] == [0.2, 0.45, 0.8]


def test_registered_pose_orientation_matches_historical_supervised_features():
    import numpy as np
    import pandas as pd
    from rainstorm.models.features import prepare_supervised_inputs
    from storm import PipelineContext

    bodyparts = ["nose", "left_ear", "right_ear", "head", "neck", "body"]
    values = {
        "nose": (3.0, 1.0), "left_ear": (2.0, 2.0),
        "right_ear": (4.0, 2.0), "head": (3.0, 3.0),
        "neck": (3.0, 4.0), "body": (1.0, 1.0),
    }
    table = pd.DataFrame({
        f"{bodypart}_{axis}": [values[bodypart][coordinate]]
        for bodypart in bodyparts
        for coordinate, axis in enumerate(("x", "y"))
    })
    expected, _ = prepare_supervised_inputs(
        table,
        bodyparts=bodyparts,
        frames=[0],
        sessions=["s"],
        segments=["a"],
        partitions=["test"],
        reserved=[True],
        offsets=(0,),
        center="body",
        orientation=("body", "nose"),
    )

    catalog = build_catalog()
    step = catalog.steps.build("pose.orient_coordinates", {
        "reference_pairs": [[10, 11], [0, 1]],
        "coordinate_pairs": [[index, index + 1] for index in range(0, 12, 2)],
        "target_angle_degrees": 45,
    })
    centered = table[[f"{bodypart}_{axis}" for bodypart in bodyparts
                      for axis in ("x", "y")]].to_numpy()
    centered = centered.reshape(1, 6, 2) - np.array([values["body"]])[:, None, :]
    context = PipelineContext(data=centered.reshape(1, 12).tolist())

    actual = step.process(context).data

    np.testing.assert_allclose(actual, expected, atol=1e-7)


def test_pose_orientation_can_leave_rows_with_coincident_references_unrotated():
    import numpy as np
    from storm import PipelineContext
    from rainstorm_thesis.preprocessing import OrientPose

    step = OrientPose(
        reference_pairs=[[0, 1], [2, 3]],
        coordinate_pairs=[[0, 1], [2, 3], [4, 5]],
        degenerate_reference_policy="identity",
    )
    source = [[0.0, 0.0, 0.0, 0.0, 1.0, 2.0],
              [0.0, 0.0, 1.0, 0.0, 1.0, 2.0]]

    actual = step.process(PipelineContext(data=source)).data

    np.testing.assert_allclose(actual[0], source[0])
    np.testing.assert_allclose(actual[1][2:4], [-2**-0.5, -2**-0.5])


def test_pose_numeric_steps_choose_gpu_for_large_batches_and_allow_cpu_override(monkeypatch):
    torch = pytest.importorskip("torch")
    from rainstorm_thesis.preprocessing import _pose_compute_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert _pose_compute_device("auto", 4) == "cpu"
    assert _pose_compute_device("auto", 32_768) == "cuda"
    assert _pose_compute_device("cpu", 100_000) == "cpu"
    assert _pose_compute_device("cuda", 4) == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _pose_compute_device("auto", 100_000) == "cpu"
    with pytest.raises(RuntimeError, match="no available CUDA or ROCm device"):
        _pose_compute_device("cuda", 100_000)


@pytest.mark.parametrize("step_type", ["recenter", "orient"])
def test_pose_numeric_steps_gpu_match_cpu_results(step_type):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("ROCm/CUDA worker required")
    import numpy as np
    from storm import PipelineContext
    from rainstorm_thesis.preprocessing import OrientPose, RecenterPose

    source = [[2.0, 1.0, 1.0, 1.0, 5.0, 3.0],
              [4.0, 2.0, 0.0, 1.0, 6.0, 4.0]]
    if step_type == "recenter":
        make_step = lambda device: RecenterPose(
            center_indices=[2, 3], coordinate_pairs=[[0, 1], [2, 3], [4, 5]],
            device=device)
    else:
        make_step = lambda device: OrientPose(
            reference_pairs=[[2, 3], [0, 1]],
            coordinate_pairs=[[0, 1], [2, 3], [4, 5]],
            target_angle_degrees=45, degenerate_reference_policy="identity",
            device=device)

    expected = make_step("cpu").process(PipelineContext(data=source)).data
    actual = make_step("cuda").process(PipelineContext(data=source)).data

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_engine_passes_registered_dataset_context_to_contextual_inference_adapters(tmp_path):
    from storm.suite import Component, default_catalog, execute

    catalog = default_catalog()
    catalog.register(Component(
        "test.context_inference", ContextInferenceModel, ("infer",),
        {"type": "object", "properties": {}},
    ))
    catalog.connectors["test.pose_context"] = lambda data: data
    data = {
        "inputs": [[1.0], [2.0]], "train": [], "validation": [], "test": [0, 1],
        "sessions": ["mouse-a", "mouse-a"],
        "source_files": [{"session_id": "mouse-a", "path": "/pose/mouse-a.h5"}],
    }

    result = execute({
        "operation": "infer", "model": "test.context_inference",
        "connector": "test.pose_context", "config": {}, "data": data,
    }, tmp_path, "context-inference", catalog)

    assert result["predictions"] == [2, 1]
    assert result["output_metadata"]["semantics"] == "context-bound predictions"


def test_native_vame_reports_training_and_inference_batches():
    pytest.importorskip('torch')
    pytest.importorskip('sklearn')
    import numpy as np
    from rainstorm_thesis.vame_native_runtime import train_native_vame, predict_native_vame
    values = np.arange(48, dtype=np.float32).reshape(8, 3, 2) / 48
    config = dict(n_states=2, latent_dim=2, hidden_dim=4, epochs=1,
                  batch_size=2, learning_rate=0.001, kld_weight=0.5,
                  seed=7, device='cpu', num_threads=1)
    events = []
    trained = train_native_vame(values, config, progress_callback=events.append)
    training = [event for event in events if event['phase'] == 'training']
    assert [event['batch_step'] for event in training] == [0, 1, 2, 3, 4]
    assert training[-1]['processed_observations'] == 8
    assert training[-1]['throughput'] > 0
    events.clear()
    predict_native_vame(values, trained, progress_callback=events.append)
    assert events[-1]['batch_step'] == events[-1]['batch_total'] == 4
    assert events[-1]['batch_eta_seconds'] == 0


def test_native_vame_can_select_native_rnn_backend():
    pytest.importorskip('torch')
    import torch
    from rainstorm_thesis.vame_native_runtime import NativeVAMENetwork
    model = NativeVAMENetwork(2, 3, 2, 4, rnn_backend='native')
    output, mu, logvar = model(torch.zeros(2, 3, 2), sample=False)
    output.square().mean().backward()
    assert output.shape == (2, 3, 2)
    assert model.rnn_backend == 'native'
    with pytest.raises(ValueError):
        NativeVAMENetwork(2, 3, 2, 4, rnn_backend='invalid')


def test_pose_filter_and_windows_report_real_observations_without_changing_boundaries():
    from storm.pipeline import PipelineContext
    from rainstorm_thesis.preprocessing import LikelihoodFilter, TemporalPoseWindows
    for step in (LikelihoodFilter(0.5, [[0, 1]]), TemporalPoseWindows([-1, 0, 1])):
        events = []
        context = PipelineContext(data=[[1., 2.], [3., 4.], [5., 6.]],
                                  metadata={'likelihoods': [[1.], [1.], [1.]],
                                            'frames': [0, 1, 3],
                                            'observation_indices': [0, 1, 2]},
                                  progress_callback=events.append)
        output = step.process(context)
        assert events[0]['processed_observations'] == 0
        assert events[-1]['processed_observations'] == 3
        assert events[-1]['total_observations'] == 3
        assert events[-1]['batch_step'] == events[-1]['batch_total']
        if isinstance(step, TemporalPoseWindows):
            assert output.data == []
