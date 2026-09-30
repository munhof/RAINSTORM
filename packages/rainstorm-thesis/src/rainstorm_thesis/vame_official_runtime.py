"""Registered-dataset inference for saved official VAME projects.

The upstream VAME pipeline does not expose a standalone prediction function.
This adapter creates an isolated analysis project, loads the saved encoder and
training metadata, then applies the training project's persisted HMM or KMeans
discretizer without refitting it on the new sessions.
"""
from __future__ import annotations

import importlib
import json
import shutil
import uuid
from pathlib import Path

from storm import ModelOutput


_PROJECT_KEYS = {
    "vame_version", "project_name", "project_path", "creation_datetime",
    "session_names", "pose_estimation_filetype", "video_type", "num_features",
    "keypoints",
}


def predict_official_vame(model_config, inputs, data, observation_indices):
    import numpy as np

    from .vame_models import _align_vame_window_labels, _plain_vame_config

    sessions = data.get("sessions")
    frames = data.get("frames")
    segments = data.get("segments")
    source_files = data.get("source_files")
    if not all(isinstance(value, list) for value in (sessions, frames, segments, source_files)):
        raise ValueError("VAME official inference needs registered session, frame, and source metadata")
    if not observation_indices or len(inputs) != len(observation_indices):
        raise ValueError("VAME official inference needs complete registered session inputs")
    if any(type(index) is not int or index < 0 or index >= len(sessions)
           for index in observation_indices):
        raise ValueError("VAME official inference received an invalid observation mapping")

    selected_sessions = list(dict.fromkeys(sessions[index] for index in observation_indices))
    expected_indices = [index for index, session in enumerate(sessions)
                        if session in selected_sessions]
    if list(observation_indices) != expected_indices:
        raise ValueError("VAME official inference requires complete sessions in source order")
    if len(frames) != len(sessions) or len(segments) != len(sessions):
        raise ValueError("VAME official session metadata does not align with source frames")

    source_by_session = {item.get("session_id"): item for item in source_files
                         if isinstance(item, dict)}
    bound_sources = []
    frame_counts = []
    for session in selected_sessions:
        source = source_by_session.get(session)
        if source is None:
            raise ValueError(f"No registered pose file is available for session {session}")
        row_indices = [index for index, value in enumerate(sessions) if value == session]
        if (source.get("selected_frame_count") != source.get("frame_count")
                or len(row_indices) != source.get("frame_count")):
            raise ValueError("VAME official inference requires every frame from each source session")
        if len({segments[index] for index in row_indices}) != 1:
            raise ValueError("VAME official inference cannot cross a video discontinuity")
        session_frames = [frames[index] for index in row_indices]
        if any(right != left + 1 for left, right in zip(session_frames, session_frames[1:])):
            raise ValueError("VAME official inference needs contiguous source frames")
        pose_path = Path(source.get("path", "")).expanduser().resolve()
        if not pose_path.is_file():
            raise FileNotFoundError(f"Registered pose source does not exist: {pose_path}")
        bound_sources.append((str(session), source, pose_path))
        frame_counts.append(len(row_indices))

    training_root = (Path(model_config["working_directory"]).expanduser().resolve()
                     / model_config["project_name"])
    training_config_path = training_root / "config.yaml"
    if not training_config_path.is_file():
        raise FileNotFoundError(f"Saved VAME project config was not found: {training_config_path}")
    try:
        vame = importlib.import_module("vame")
    except ImportError as error:
        raise RuntimeError("Official VAME inference requires vame-py in the scientific worker") from error
    training_config = vame.read_config(str(training_config_path))
    if not isinstance(training_config, dict):
        raise ValueError("Saved VAME project config is not an object")
    if training_config.get("individual_segmentation"):
        raise ValueError(
            "This VAME project fitted session-local clusters and has no shared saved discretizer"
        )
    algorithm = model_config["segmentation_algorithm"]
    n_clusters = int(training_config["n_clusters"])
    if (n_clusters != int(model_config["config_kwargs"]["n_clusters"])
            or int(training_config["time_window"])
            != int(model_config["config_kwargs"]["time_window"])):
        raise ValueError("Saved VAME project recipe does not match the registered model recipe")

    model_name = str(training_config["model_name"])
    model_path = _find_saved_model(training_root, model_name, model_config["project_name"])
    metadata_path = training_root / "data" / "train" / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Saved VAME training metadata was not found: {metadata_path}")
    with metadata_path.open(encoding="utf-8") as handle:
        training_metadata = json.load(handle)
    if not training_metadata.get("parameters", {}).get("keypoints_used"):
        raise ValueError("Saved VAME metadata does not declare the trained keypoints")

    if algorithm == "kmeans":
        training_centers = _find_shared_kmeans_centers(
            training_root, training_config, n_clusters, np
        )
    elif algorithm == "hmm":
        hmm_path = training_root / "results" / "hmm_trained.pkl"
        if not hmm_path.is_file():
            raise FileNotFoundError("The saved VAME project has no fitted HMM discretizer")
        training_centers = None
    else:
        raise ValueError(f"Unsupported official VAME discretizer: {algorithm}")

    run_id = uuid.uuid4().hex[:12]
    inference_name = f"rainstorm_inference_{run_id}"
    workspace = Path(model_config["working_directory"]).expanduser().resolve()
    inference_root = workspace / "inference_runs"
    source_stage = inference_root / f"sources_{run_id}"
    inference_root.mkdir(parents=True, exist_ok=True)
    source_stage.mkdir(parents=True)
    pose_paths = []
    video_paths = []
    has_all_videos = all(source.get("video_path") for _, source, _ in bound_sources)
    used_names = set()
    for _, source, pose_path in bound_sources:
        # vame-py resolves symlinks to derive its session name, but writes its
        # processed file under the staged link name. Preserve the source stem
        # so both paths agree.
        stem = pose_path.stem
        if stem in used_names:
            raise ValueError("Pose filenames must map to unique VAME session names")
        used_names.add(stem)
        alias = source_stage / f"{stem}{pose_path.suffix.lower()}"
        alias.symlink_to(pose_path)
        pose_paths.append(str(alias))
        if has_all_videos:
            video_path = Path(source["video_path"]).expanduser().resolve()
            if not video_path.is_file():
                raise FileNotFoundError(f"Registered video source does not exist: {video_path}")
            video_alias = source_stage / f"{stem}{video_path.suffix.lower()}"
            video_alias.symlink_to(video_path)
            video_paths.append(str(video_alias))

    config_kwargs = {
        key: value for key, value in training_config.items() if key not in _PROJECT_KEYS
    }
    config_kwargs["segmentation_algorithms"] = [algorithm]
    config_kwargs["individual_segmentation"] = False
    config_kwargs["hmm_trained"] = algorithm == "hmm"
    fps = float(data.get("fps", model_config["fps"]))
    created = vame.init_new_project(
        project_name=inference_name,
        working_directory=str(inference_root),
        poses_estimations=pose_paths,
        videos=video_paths or None,
        source_software=model_config["source_software"],
        fps=fps,
        config_kwargs=config_kwargs,
    )
    project_directory = inference_root / inference_name
    inference_config = (created[1] if isinstance(created, tuple) and len(created) == 2
                        and isinstance(created[1], dict)
                        else vame.read_config(str(project_directory / "config.yaml")))
    inference_config = _plain_vame_config(inference_config, np)
    if inference_config.get("keypoints") != training_config.get("keypoints"):
        raise ValueError("Registered pose keypoints do not match the trained VAME input schema")
    inference_config["num_features"] = int(training_config["num_features"])
    expected_session_names = [Path(path).stem for path in pose_paths]
    if inference_config.get("session_names") != expected_session_names:
        raise ValueError("VAME inference session order does not match the registered source order")

    _copy_inference_artifacts(project_directory, model_name, model_path, metadata_path,
                              algorithm, training_root / "results" / "hmm_trained.pkl"
                              if algorithm == "hmm" else None)
    vame.preprocessing(
        config=inference_config,
        centered_reference_keypoint=model_config["centered_reference_keypoint"],
        orientation_reference_keypoint=model_config["orientation_reference_keypoint"],
        save_logs=False,
    )

    from vame.io.load_poses import read_pose_estimation_file
    from vame.preprocessing.to_model import format_xarray_for_rnn

    parameters = training_metadata["parameters"]
    for session_name in inference_config["session_names"]:
        processed_path = (project_directory / "data" / "processed"
                          / f"{session_name}_processed.nc")
        _, _, dataset = read_pose_estimation_file(file_path=str(processed_path))
        _, feature_metadata = format_xarray_for_rnn(
            ds=dataset,
            read_from_variable=parameters["read_from_variable"],
            keypoints=parameters["keypoints_used"],
            extra_features=parameters.get("extra_features", []),
        )
        if (feature_metadata.get("feature_mapping") != training_metadata.get("feature_mapping")
                or feature_metadata["parameters"]["total_features"]
                != int(training_config["num_features"])):
            raise ValueError("Registered pose features do not match the trained VAME input schema")

    # VAME's embedding writer saves directly below results/<session>/<model> but
    # does not create that directory for a fresh analysis project.
    for session_name in inference_config["session_names"]:
        (project_directory / "results" / session_name / model_name).mkdir(
            parents=True, exist_ok=True
        )

    from vame.analysis.pose_segmentation import embed_latent_vectors_optimized

    embeddings = embed_latent_vectors_optimized(
        config=inference_config,
        sessions=inference_config["session_names"],
        fixed=inference_config["egocentric_data"],
        overwrite=True,
    )
    if algorithm == "hmm":
        vame.segment_session(
            config=inference_config,
            overwrite_segmentation=True,
            overwrite_embeddings=False,
            save_logs=False,
            optimized=True,
        )

    predictions, prediction_mask, embedding_paths = [], [], []
    for session_name, frame_count, session_embeddings in zip(
            inference_config["session_names"], frame_counts, embeddings):
        latent = np.asarray(session_embeddings)
        if latent.ndim != 2 or latent.shape[1] != int(training_config["zdims"]):
            raise ValueError("VAME encoder output does not match the trained latent schema")
        if algorithm == "kmeans":
            if latent.shape[1] != training_centers.shape[1]:
                raise ValueError("Saved VAME KMeans centers do not match the encoder output")
            labels = np.square(latent[:, None, :] - training_centers[None, :, :]).sum(axis=2).argmin(axis=1)
        else:
            label_path, _ = _discover_vame_session_outputs(
                project_directory, session_name, algorithm, n_clusters
            )
            labels = np.asarray(np.load(label_path)).reshape(-1)
        aligned, valid = _align_vame_window_labels(
            np.asarray(labels).reshape(-1).tolist(), frame_count,
            int(training_config["time_window"]),
        )
        predictions.extend(aligned)
        prediction_mask.extend(valid)
        embedding_paths.append(str(project_directory / "results" / session_name
                                   / model_name / "latent_vectors.npy"))

    return ModelOutput(
        predictions=predictions,
        metadata={
            "semantics": "motif IDs from the saved VAME encoder and training discretizer",
            "backend": "vame_py_official_saved_model",
            "segmentation_algorithm": algorithm,
            "prediction_mask": prediction_mask,
            "prediction_indices": [index for index, valid in enumerate(prediction_mask) if valid],
            "prediction_alignment": {
                "kind": "centered_temporal_window",
                "window": int(training_config["time_window"]),
                "anchor_offset": int(training_config["time_window"]) // 2,
            },
            "inference_project": str(project_directory),
            "embeddings_paths": embedding_paths,
            "discretizer_scope": "shared_training_model",
        },
    )


def _find_saved_model(project_directory, model_name, project_name):
    best_model = project_directory / "model" / "best_model"
    expected = best_model / f"{model_name}_{project_name}.pkl"
    if expected.is_file():
        return expected
    snapshots = sorted(best_model.glob(f"{model_name}_*_epoch_*.pkl"),
                       key=lambda path: path.stat().st_mtime)
    if not snapshots:
        raise FileNotFoundError(f"Saved VAME encoder weights were not found in {best_model}")
    return snapshots[-1]


def _find_shared_kmeans_centers(project_directory, config, n_clusters, numpy):
    if config.get("individual_segmentation"):
        raise ValueError("VAME session-local KMeans centers cannot be applied as a shared model")
    centers = []
    for session in config.get("session_names", []):
        expected = (project_directory / "results" / session / config["model_name"]
                    / f"kmeans-{n_clusters}" / f"cluster_center_{session}.npy")
        if not expected.is_file():
            expected = next((path for path in (project_directory / "results" / session).rglob(
                f"cluster_center_{session}.npy")), None)
        if expected is None or not expected.is_file():
            continue
        centers.append(numpy.asarray(numpy.load(expected), dtype=float))
    if not centers:
        raise FileNotFoundError("Saved VAME project has no training KMeans centers")
    if any(center.shape != centers[0].shape
           or not numpy.allclose(center, centers[0]) for center in centers[1:]):
        raise ValueError("VAME training sessions do not share one KMeans discretizer")
    return centers[0]


def _copy_inference_artifacts(inference_root, model_name, model_path, metadata_path,
                              algorithm, hmm_path):
    destination_model = inference_root / "model" / "best_model"
    destination_model.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_path, destination_model / f"{model_name}_{inference_root.name}.pkl")
    destination_metadata = inference_root / "data" / "train" / "metadata.json"
    destination_metadata.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(metadata_path, destination_metadata)
    if algorithm == "hmm":
        destination_hmm = inference_root / "results" / "hmm_trained.pkl"
        destination_hmm.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(hmm_path, destination_hmm)


def _discover_vame_session_outputs(project_directory, session_name, algorithm, n_clusters):
    expected_name = f"{n_clusters}_{algorithm}_label_{session_name}.npy"
    labels = sorted((project_directory / "results" / session_name).rglob(expected_name))
    if not labels:
        raise FileNotFoundError(
            f"Official VAME wrote no {algorithm} labels for session {session_name}"
        )
    return labels[0], None
