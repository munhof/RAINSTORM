from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import os
from pathlib import Path
import importlib
import shutil

from storm import ModelOutput


class VAMENativeModel:
    """Native PyTorch VAME representation followed by model-local KMeans states."""

    required_pipeline_steps = ("pose.temporal_windows",)

    def __init__(self, config: Mapping | None = None):
        self.config = {
            "n_states": 50,
            "latent_dim": 20,
            "hidden_dim": 128,
            "epochs": 25,
            "batch_size": 256,
            "learning_rate": 0.001,
            "kld_weight": 0.5,
            "seed": 156,
            "device": "auto",
            **dict(config or {}),
        }
        for key in ("n_states", "latent_dim", "hidden_dim", "epochs", "batch_size", "seed"):
            if type(self.config[key]) is not int or self.config[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.config["n_states"] < 2:
            raise ValueError("n_states must be at least 2")
        if self.config["learning_rate"] <= 0 or self.config["kld_weight"] < 0:
            raise ValueError("learning_rate must be positive and kld_weight nonnegative")
        self.checkpoint = None
        self._resume_state = None
        self._training_state = None
        self._progress_callback = None

    def set_progress_callback(self, callback):
        self._progress_callback = callback

    def fit_predict(self, inputs, constraints=()):
        return self._fit_predict(inputs, constraints)

    def fit_predict_with_checkpoints(self, inputs, constraints=(), checkpoint=None):
        if not callable(checkpoint):
            raise ValueError("VAME checkpoint training requires a checkpoint callback")
        return self._fit_predict(inputs, constraints, checkpoint)

    def _fit_predict(self, inputs, constraints=(), checkpoint=None):
        if constraints:
            raise ValueError("VAME does not accept pairwise grouping constraints")
        try:
            from .vame_native_runtime import train_native_vame
        except ImportError as error:
            raise RuntimeError(
                "Native VAME needs NumPy, PyTorch, and scikit-learn; install "
                "RAINSTORM's vame-native worker dependencies."
            ) from error

        progress_options = ({'progress_callback': getattr(self, '_progress_callback', None)}
                            if getattr(self, '_progress_callback', None) is not None else {})
        if checkpoint is None and self._resume_state is None:
            self.checkpoint = train_native_vame(inputs, self.config, **progress_options)
        else:
            def record_checkpoint(state):
                self._training_state = deepcopy(state)
                checkpoint(state)

            self.checkpoint = train_native_vame(
                inputs,
                self.config,
                resume_state=deepcopy(self._resume_state),
                checkpoint=record_checkpoint,
                **progress_options,
            )
        self._resume_state = None
        predictions = self.checkpoint.pop("labels")
        embeddings = self.checkpoint.pop("embeddings")
        training_history = self.checkpoint["training_history"]
        return ModelOutput(
            predictions=predictions,
            metadata={
                "semantics": "model-local VAME state IDs; IDs may permute between runs",
                "backend": "rainstorm_native_vame",
                "latent_dim": self.config["latent_dim"],
                "n_states": self.config["n_states"],
                "embedding_shape": [len(embeddings), self.config["latent_dim"]],
                "training_history": training_history,
                "checkpoint_capability": "epoch-boundary continuation with model, optimizer, and RNG state",
            },
        )

    def save_checkpoint(self):
        state = self._training_state or self._resume_state
        if state is None:
            raise ValueError("No native VAME training checkpoint is available")
        return deepcopy(state)

    def load_checkpoint(self, state):
        if not isinstance(state, Mapping) or "epoch" not in state:
            raise ValueError("Invalid native VAME training checkpoint")
        self._resume_state = deepcopy(dict(state))

    def predict(self, inputs):
        if self.checkpoint is None:
            raise ValueError("Fit VAMENativeModel before prediction")
        try:
            from .vame_native_runtime import predict_native_vame
        except ImportError as error:
            raise RuntimeError(
                "Native VAME prediction needs NumPy, PyTorch, and scikit-learn."
            ) from error

        progress_options = ({'progress_callback': getattr(self, '_progress_callback', None)}
                            if getattr(self, '_progress_callback', None) is not None else {})
        labels, embeddings = predict_native_vame(inputs, self.checkpoint, **progress_options)
        return ModelOutput(
            predictions=labels,
            metadata={
                "semantics": "model-local VAME state IDs; IDs may permute between runs",
                "backend": "rainstorm_native_vame",
                "embeddings": embeddings,
            },
        )


class VAMEOfficialModel:
    """Worker adapter for the official EthoML ``vame-py`` pipeline."""

    requires_registered_data_for_inference = True
    supports_pipeline_steps = False

    def __init__(self, config: Mapping | None = None, *, vame_module=None):
        official_recipe = {
            "n_clusters": 50,
            "model_snapshot": 10,
            "model_convergence": 10,
            "time_window": 19,
            "zdims": 10,
            "max_epochs": 50,
            "batch_size": 32,
            "confidence": 0.60,
            "seed": 156,
        }
        supplied = dict(config or {})
        official_recipe.update(supplied.pop("config_kwargs", {}) or {})
        self.config = {
            "project_name": "rainstorm_vame_official",
            "working_directory": ".storm/rainstorm/vame_official",
            "pose_paths": [],
            "videos": [],
            "source_software": "DeepLabCut",
            "fps": 30.0,
            "segmentation_algorithm": "hmm",
            "centered_reference_keypoint": "body",
            "orientation_reference_keypoint": "nose",
            **supplied,
            "config_kwargs": official_recipe,
        }
        if self.config["working_directory"] == ".storm/rainstorm/vame_official":
            workspace = Path(os.environ.get("STORM_WORKSPACE", ".storm")).expanduser()
            self.config["working_directory"] = str(
                workspace / "rainstorm" / "vame_official"
            )
        if self.config["segmentation_algorithm"] not in {"hmm", "kmeans"}:
            raise ValueError("segmentation_algorithm must be 'hmm' or 'kmeans'")
        if float(self.config["fps"]) <= 0:
            raise ValueError("fps must be positive")
        max_epochs = self.config["config_kwargs"]["max_epochs"]
        if type(max_epochs) is not int or max_epochs < 2:
            raise ValueError(
                "Official VAME max_epochs must be at least 2; its training loop starts at epoch 1"
            )
        if not 0 <= float(self.config["config_kwargs"]["confidence"]) <= 1:
            raise ValueError("confidence must be between 0 and 1")
        self._vame_module = vame_module
        self._labels = None
        self._prediction_mask = None
        self._prediction_window = None
        self._session_lengths = None
        self._embeddings = None

    def bind_data(self, data: Mapping, observation_indices: list[int]) -> None:
        """Bind only complete, unreserved training sessions to official VAME."""
        source_files = data.get("source_files")
        sessions = data.get("sessions")
        if not isinstance(source_files, list) or not isinstance(sessions, list):
            raise ValueError("Official VAME needs DLC source-file and session metadata")
        if any(type(index) is not int or index < 0 or index >= len(sessions)
               for index in observation_indices):
            raise ValueError("Official VAME received an invalid observation mapping")
        selected_sessions = list(dict.fromkeys(sessions[index] for index in observation_indices))
        ordered_indices = [
            index for session in selected_sessions
            for index, value in enumerate(sessions) if value == session
        ]
        if ordered_indices != observation_indices:
            raise ValueError(
                "Official VAME requires complete sessions in their source order"
            )
        source_by_session = {item["session_id"]: item for item in source_files}
        reserved = data.get("reserved_evaluation", [False] * len(sessions))
        partitions = data.get("partitions", ["train"] * len(sessions))
        segments = data.get("segments", ["segment"] * len(sessions))
        bound_sources = []
        session_lengths = []
        for session in selected_sessions:
            source = source_by_session.get(session)
            if source is None:
                raise ValueError(f"No pose file is registered for training session {session}")
            row_indices = [index for index, value in enumerate(sessions) if value == session]
            selected_indices = [index for index in observation_indices if sessions[index] == session]
            if len(selected_indices) != len(row_indices):
                raise ValueError(
                    "Official VAME needs every frame from each training session; "
                    "use whole-session partitions without temporal-window filtering."
                )
            if source["selected_frame_count"] != source["frame_count"]:
                raise ValueError("Official VAME cannot train on a cropped H5 source")
            if any(reserved[index] for index in row_indices):
                raise ValueError(
                    "Official VAME cannot train on a session that contains reserved evaluation frames"
                )
            if any(partitions[index] != "train" for index in row_indices):
                raise ValueError("Official VAME requires training-only whole-session inputs")
            if len({segments[index] for index in row_indices}) != 1:
                raise ValueError(
                    "Official VAME cannot cross video discontinuities; split the source into segments"
                )
            bound_sources.append(source)
            session_lengths.append(len(row_indices))
        pose_paths = [str(Path(source["path"]).resolve()) for source in bound_sources]
        configured_paths = [str(Path(path).expanduser().resolve())
                            for path in self.config.get("pose_paths", [])]
        if configured_paths and configured_paths != pose_paths:
            raise ValueError("Configured VAME pose_paths do not match the selected training sessions")
        self.config["pose_paths"] = pose_paths
        self._session_lengths = session_lengths
        video_paths = [source.get("video_path", "") for source in bound_sources]
        video_paths = [path for path in video_paths if path]
        configured_videos = [str(Path(path).expanduser().resolve())
                             for path in self.config.get("videos", [])]
        if configured_videos and configured_videos != video_paths:
            raise ValueError("Configured VAME videos do not match the selected training sessions")
        self.config["videos"] = video_paths

    def fit_predict(self, inputs, constraints=()):
        import numpy as np

        if constraints:
            raise ValueError("Official VAME does not accept pairwise grouping constraints")
        pose_paths = [Path(path).expanduser().resolve() for path in self.config["pose_paths"]]
        if not pose_paths or any(not path.is_file() for path in pose_paths):
            raise ValueError("vame_official requires registered pose_paths in its model config")
        video_paths = [Path(path).expanduser().resolve() for path in self.config["videos"]]
        if video_paths and (len(video_paths) != len(pose_paths)
                            or any(not path.is_file() for path in video_paths)):
            raise ValueError("videos must be existing paths aligned with pose_paths")
        try:
            vame = self._vame_module or importlib.import_module("vame")
        except ImportError as error:
            raise RuntimeError(
                "Official VAME needs vame-py in its Python 3.12 runtime; set up "
                "RAINSTORM's envs/vame_official environment."
            ) from error

        working_directory = Path(self.config["working_directory"]).expanduser().resolve()
        project_directory = working_directory / self.config["project_name"]
        working_directory.mkdir(parents=True, exist_ok=True)
        config_kwargs = dict(self.config["config_kwargs"])
        config_kwargs["pose_confidence"] = float(config_kwargs.pop("confidence"))
        if not (project_directory / "config.yaml").is_file():
            created = vame.init_new_project(
                project_name=self.config["project_name"],
                working_directory=str(working_directory),
                videos=[str(path) for path in video_paths] or None,
                poses_estimations=[str(path) for path in pose_paths],
                source_software=self.config["source_software"],
                fps=float(self.config["fps"]),
                config_kwargs=config_kwargs,
            )
            self._official_project = created[1] if isinstance(created, tuple) and len(created) == 2 else created
        config_path = project_directory / "config.yaml"
        official_config = (vame.read_config(str(config_path))
                           if hasattr(vame, "read_config") and config_path.is_file()
                           else str(config_path))
        if isinstance(official_config, dict):
            official_config = _plain_vame_config(official_config, np)
            official_config.update(config_kwargs)
            official_config = _plain_vame_config(official_config, np)
            vame.write_config(str(config_path), official_config)
        vame.preprocessing(
            config=official_config,
            centered_reference_keypoint=self.config["centered_reference_keypoint"],
            orientation_reference_keypoint=self.config["orientation_reference_keypoint"],
        )
        if isinstance(official_config, dict):
            official_config = _plain_vame_config(official_config, np)
        session_names = list((official_config.get("session_names") or [])
                             if isinstance(official_config, dict) else [])
        if not session_names:
            session_names = [path.stem for path in pose_paths]
        session_names = [str(name) for name in session_names]
        if self._session_lengths is None:
            if len(session_names) != 1:
                raise ValueError(
                    "Bind multi-session VAME input with STORM session metadata before training"
                )
            self._session_lengths = [len(inputs)]
        if len(session_names) != len(self._session_lengths):
            raise ValueError("VAME session names do not match the bound training sessions")
        if sum(self._session_lengths) != len(inputs):
            raise ValueError("Bound VAME session frame counts do not match model inputs")
        vame.create_trainset(config=official_config)
        vame.train_model(config=official_config)
        self._ensure_best_model_alias(project_directory)
        if isinstance(official_config, dict):
            official_config["segmentation_algorithms"] = [self.config["segmentation_algorithm"]]
        vame.segment_session(config=official_config)
        window = int(self.config["config_kwargs"]["time_window"])
        labels_paths, embeddings_paths = [], []
        aligned_labels, prediction_mask = [], []
        for session_name, frame_count in zip(session_names, self._session_lengths):
            labels_path, embeddings_path = _discover_vame_session_outputs(
                project_directory, session_name,
                self.config["segmentation_algorithm"],
                self.config["config_kwargs"]["n_clusters"],
            )
            labels = np.asarray(np.load(labels_path)).reshape(-1).tolist()
            session_labels, session_mask = _align_vame_window_labels(
                labels, frame_count, window
            )
            aligned_labels.extend(session_labels)
            prediction_mask.extend(session_mask)
            labels_paths.append(labels_path)
            embeddings_paths.append(embeddings_path)
        self._labels = aligned_labels
        self._prediction_mask = prediction_mask
        self._prediction_window = window
        self._embeddings_paths = embeddings_paths
        return ModelOutput(
            predictions=list(self._labels),
            metadata={
                "semantics": "model-local VAME motif IDs; IDs may permute between runs",
                "backend": "vame_py_official",
                "project_name": self.config["project_name"],
                "project_dir": str(project_directory),
                "requires_registered_data_for_inference": True,
                "segmentation_algorithm": self.config["segmentation_algorithm"],
                "labels_path": str(labels_paths[0]) if len(labels_paths) == 1 else None,
                "labels_paths": [str(path) for path in labels_paths],
                "embeddings_path": (
                    str(embeddings_paths[0])
                    if len(embeddings_paths) == 1 and embeddings_paths[0] is not None
                    else None
                ),
                "embeddings_paths": [str(path) if path is not None else None
                                     for path in embeddings_paths],
                "prediction_mask": list(self._prediction_mask),
                "prediction_indices": [i for i, valid in enumerate(self._prediction_mask) if valid],
                "prediction_alignment": {
                    "kind": "centered_temporal_window",
                    "window": window,
                    "anchor_offset": window // 2,
                },
            },
        )

    def predict(self, inputs):
        raise ValueError(
            "VAME official inference needs registered pose files and session metadata; "
            "apply the saved model to a registered pose dataset."
        )

    def predict_with_context(self, inputs, data: Mapping, observation_indices: list[int]):
        """Run the saved VAME encoder and fitted discretizer on registered sessions."""
        try:
            from .vame_official_runtime import predict_official_vame
        except ImportError as error:
            raise RuntimeError(
                "Official VAME inference needs vame-py in RAINSTORM's scientific worker."
            ) from error
        return predict_official_vame(self.config, inputs, data, observation_indices)

    def _ensure_best_model_alias(self, project_directory: Path) -> None:
        best_model_dir = project_directory / "model" / "best_model"
        target = best_model_dir / f"VAME_{self.config['project_name']}.pkl"
        if target.is_file():
            return
        snapshots_dir = best_model_dir / "snapshots"
        snapshots = sorted(snapshots_dir.glob("VAME_*_epoch_*.pkl"),
                           key=lambda path: path.stat().st_mtime) if snapshots_dir.is_dir() else []
        if snapshots:
            shutil.copy2(snapshots[-1], target)


def _plain_vame_config(value, numpy):
    """Convert NumPy values from VAME preprocessing before its YAML writes."""
    if isinstance(value, numpy.ndarray):
        return _plain_vame_config(value.tolist(), numpy)
    if isinstance(value, numpy.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            _plain_vame_config(key, numpy): _plain_vame_config(item, numpy)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_plain_vame_config(item, numpy) for item in value]
    return value


def _align_vame_window_labels(labels: list, frame_count: int, window: int):
    """Map VAME's one-label-per-window output to source frame centers."""
    if window < 1:
        raise ValueError("VAME time_window must be a positive integer")
    if len(labels) == frame_count:
        return list(labels), [True] * frame_count
    expected = frame_count - window + 1
    if expected < 0 or len(labels) != expected:
        raise ValueError(
            f"Official VAME produced {len(labels)} labels for {frame_count} frames; "
            f"expected {frame_count} frame labels or {max(0, expected)} window labels."
        )
    offset = window // 2
    aligned = [None] * frame_count
    mask = [False] * frame_count
    for index, label in enumerate(labels):
        frame = index + offset
        aligned[frame] = label
        mask[frame] = True
    return aligned, mask


def _discover_vame_session_outputs(project_directory: Path, session_name: str,
                                   algorithm: str, n_clusters: int):
    session_directory = project_directory / "results" / session_name
    expected_name = f"{n_clusters}_{algorithm}_label_{session_name}.npy"
    labels = sorted(session_directory.rglob(expected_name))
    if not labels:
        labels = sorted(
            path for path in session_directory.rglob("*_label_*.npy")
            if f"_{algorithm}_label_{session_name}.npy" in path.name
        )
    if not labels:
        raise FileNotFoundError(
            f"Official VAME wrote no {algorithm} labels for session {session_name}"
        )
    embeddings = _first_existing(
        session_directory, ("**/latent_vectors.npy", "**/embeddings.npy")
    )
    return labels[0], embeddings


def _first_existing(root: Path, patterns: tuple[str, ...]):
    return next((match for pattern in patterns for match in sorted(root.glob(pattern))), None)
