from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

from storm import DataRef, Dataset


@dataclass(frozen=True)
class PoseDatasetConfig:
    pose_path: Path
    label_path: Path | None = None
    max_frames: int | None = 2_000
    coordinate_suffixes: tuple[str, ...] = ("_x", "_y")
    fps: float = 30.0

    def data_ref(self, *, identifier: str | None = None) -> DataRef:
        from .reconstruction import file_reference

        pose = file_reference(self.pose_path)
        labels = file_reference(self.label_path) if self.label_path else None
        recipe = {"max_frames": self.max_frames,
                  "coordinate_suffixes": list(self.coordinate_suffixes), "fps": self.fps}
        identity = {"pose": pose["sha256"],
                    "labels": labels["sha256"] if labels else None, "recipe": recipe}
        return DataRef(
            identifier=identifier or self.pose_path.stem,
            fingerprint="sha256:" + hashlib.sha256(
                json.dumps(identity, sort_keys=True).encode()
            ).hexdigest(),
            metadata={
                "pose_path": pose["path"],
                "label_path": labels["path"] if labels else "",
                "source_hashes": {"pose": pose["sha256"],
                                  "labels": labels["sha256"] if labels else None},
                "recipe": recipe,
            },
        )


def project_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (
            (candidate / "examples").is_dir()
            and (candidate / "src" / "rainstorm" / "backend").is_dir()
        ):
            return candidate
    raise RuntimeError("Could not locate the RAINSTORM project root.")


def example_pose_file(pattern: str = "NOR_TS_01*.h5") -> Path:
    return next((project_root() / "examples" / "NOR").glob(pattern))


def example_colabels_file() -> Path:
    return project_root() / "examples" / "models" / "colabels.csv"


def load_pose_table(path: Path | str, *, max_frames: int | None = None,
                    preserve_index: bool = False) -> pd.DataFrame:
    import pandas as pd

    path = Path(path)
    if path.suffix.lower() == ".csv":
        table = pd.read_csv(path)
    elif path.suffix.lower() in {".h5", ".hdf", ".hdf5"}:
        table = pd.read_hdf(path)
    else:
        raise ValueError(f"Unsupported pose table format: {path.suffix}")

    table = table.copy()
    if isinstance(table.columns, pd.MultiIndex):
        table.columns = [_flatten_pose_column(column) for column in table.columns]
    if max_frames is not None:
        table = table.head(max_frames)
    return table if preserve_index else table.reset_index(drop=True)


def load_dlc_h5(data: dict) -> dict:
    """Load DLC H5 pose files into aligned, JSON-compatible worker rows."""
    import pandas as pd

    return _load_dlc_tables(data, pd, lambda path, key: pd.read_hdf(path, key=key))


def load_dlc_csv(data: dict) -> dict:
    """Load standard three-header-row DLC CSV exports."""
    import pandas as pd

    return _load_dlc_tables(
        data, pd, lambda path, _key: pd.read_csv(path, header=[0, 1, 2], index_col=0)
    )


def _load_dlc_tables(data: dict, pd, table_reader) -> dict:
    """Shared DeepLabCut table alignment for HDF5 and CSV sources."""

    pose_paths = data.get("pose_paths") or ([data["pose_path"]] if data.get("pose_path") else [])
    if not pose_paths:
        raise ValueError("dlc_h5 requires pose_path or a nonempty pose_paths list")
    pose_session_ids = data.get("pose_session_ids")
    if pose_session_ids is not None and (
        not isinstance(pose_session_ids, list)
        or len(pose_session_ids) != len(pose_paths)
        or any(not isinstance(session, str) or not session for session in pose_session_ids)
    ):
        raise ValueError("pose_session_ids must provide one nonempty ID per pose file")
    video_paths = data.get("video_paths") or ([data["video_path"]] if data.get("video_path") else [])
    if video_paths and len(video_paths) != len(pose_paths):
        raise ValueError("video_paths must align one-to-one with pose_paths")
    resolved_videos = [Path(path).expanduser().resolve() for path in video_paths]
    if any(not path.is_file() for path in resolved_videos):
        raise FileNotFoundError(next(path for path in resolved_videos if not path.is_file()))
    fps = float(data.get("fps", 30.0))
    if fps <= 0:
        raise ValueError("fps must be positive")
    csv_frame_base = int(data.get("csv_frame_base", 1))
    pose_frame_base = int(data.get("pose_frame_base", 0))
    label_frame_reference = data.get("label_frame_reference", "pose")
    if label_frame_reference not in {"pose", "video"}:
        raise ValueError("label_frame_reference must be 'pose' or 'video'")
    video_frame_offsets = data.get("video_frame_offsets_by_session", {})
    if (not isinstance(video_frame_offsets, dict)
            or any(not isinstance(session, str) or type(offset) is not int
                   for session, offset in video_frame_offsets.items())):
        raise ValueError("video_frame_offsets_by_session must map session IDs to integers")
    video_discontinuities = data.get("video_discontinuities_by_session", {})
    if not isinstance(video_discontinuities, dict):
        raise ValueError("video_discontinuities_by_session must be an object")
    for session, boundaries in video_discontinuities.items():
        if (not isinstance(session, str) or not isinstance(boundaries, list)
                or any(type(frame) is not int or frame < 0 for frame in boundaries)
                or boundaries != sorted(set(boundaries))):
            raise ValueError(
                "Video discontinuities must be sorted, unique, nonnegative frame indices"
            )
    requested = data.get("bodyparts")
    key = data.get("key")
    train_sessions = set(data.get("train_session_ids", []))
    validation_sessions = set(data.get("validation_session_ids", []))
    test_sessions = set(data.get("test_session_ids", []))
    frame_start = data.get("frame_start")
    frame_stop = data.get("frame_stop")
    max_frames = data.get("max_frames")
    if frame_start is not None and (type(frame_start) is not int or frame_start < 0):
        raise ValueError("frame_start must be a nonnegative integer")
    if frame_stop is not None and (type(frame_stop) is not int or frame_stop < 1):
        raise ValueError("frame_stop must be a positive integer")
    if frame_start is not None and frame_stop is not None and frame_start >= frame_stop:
        raise ValueError("Pose frame crop must be a nonempty half-open interval")
    if max_frames is not None and (type(max_frames) is not int or max_frames < 1):
        raise ValueError("max_frames must be a positive integer")
    overlap = ((train_sessions & validation_sessions) | (train_sessions & test_sessions)
               | (validation_sessions & test_sessions))
    if overlap:
        raise ValueError(f"Sessions occur in multiple partitions: {sorted(overlap)}")
    if set(video_discontinuities) - set(pose_session_ids or []):
        raise ValueError("Video discontinuities refer to an unregistered pose session")

    inputs: list[list[float]] = []
    likelihoods: list[list[float]] = []
    frames: list[int] = []
    video_frames: list[int] = []
    sessions: list[str] = []
    segments: list[str] = []
    partition_names: list[str] = []
    observation_ids: list[str] = []
    source_files: list[dict[str, str]] = []
    feature_names: list[str] | None = None
    likelihood_bodyparts: list[str] = []
    for file_index, raw_path in enumerate(pose_paths):
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        table = table_reader(path, key)
        if isinstance(table.columns, pd.MultiIndex):
            table = table.copy()
            table.columns = [_flatten_pose_column(column) for column in table.columns]
        else:
            table = table.copy()
            table.columns = [str(column) for column in table.columns]
        session = (pose_session_ids[file_index] if pose_session_ids is not None
                   else _dlc_session_id(path))
        if feature_names is None:
            available_bodyparts = list(dict.fromkeys(
                name[:-2] for name in table.columns if name.endswith(("_x", "_y"))
            ))
            bodyparts = list(requested) if requested is not None else available_bodyparts
            if not bodyparts or len(set(bodyparts)) != len(bodyparts):
                raise ValueError("DLC H5 must contain distinct bodyparts with x/y coordinates")
            missing = [f"{part}_{axis}" for part in bodyparts for axis in ("x", "y")
                       if f"{part}_{axis}" not in table.columns]
            if missing:
                raise ValueError(f"DLC H5 is missing coordinate columns: {missing}")
            feature_names = [f"{part}_{axis}" for part in bodyparts for axis in ("x", "y")]
            likelihood_names = [f"{part}_likelihood" for part in bodyparts
                                if f"{part}_likelihood" in table.columns]
            likelihood_bodyparts = [name.removesuffix("_likelihood") for name in likelihood_names]
        elif any(name not in table.columns for name in feature_names):
            raise ValueError("DLC H5 files do not share the same bodypart coordinate columns")
        else:
            likelihood_names = [name for name in feature_names
                                if name.endswith("_x") and name[:-1] + "likelihood" in table.columns]
            likelihood_names = [name[:-1] + "likelihood" for name in likelihood_names]

        table = table.loc[:, feature_names + likelihood_names]
        try:
            session_frames = [int(value) for value in table.index]
        except (TypeError, ValueError):
            session_frames = list(range(len(table)))
        if len(set(session_frames)) != len(session_frames):
            raise ValueError(f"DLC H5 frame index is duplicated for session {session}")
        values = table.loc[:, feature_names].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
        confidence = (table.loc[:, likelihood_names].apply(pd.to_numeric, errors="coerce")
                      .to_numpy(dtype=float)) if likelihood_names else [[] for _ in range(len(table))]
        session_partition = _session_partition(session, train_sessions, validation_sessions,
                                               test_sessions)
        session_video_gaps = set(video_discontinuities.get(session, []))
        segment_number = 0
        previous_frame = None
        included_frames = 0
        for row_index, frame in enumerate(session_frames):
            if frame_start is not None and frame < frame_start:
                continue
            if frame_stop is not None and frame >= frame_stop:
                continue
            if max_frames is not None and included_frames >= max_frames:
                break
            video_frame = frame - pose_frame_base + video_frame_offsets.get(session, 0)
            if (previous_frame is not None
                    and (frame != previous_frame + 1 or video_frame in session_video_gaps)):
                segment_number += 1
            inputs.append(values[row_index].tolist())
            likelihoods.append(confidence[row_index].tolist())
            frames.append(frame)
            video_frames.append(video_frame)
            sessions.append(session)
            segments.append(f"{session}:segment-{segment_number}")
            partition_names.append(session_partition)
            observation_ids.append(f"{session}:{frame}")
            previous_frame = frame
            included_frames += 1
        video = resolved_videos[file_index] if resolved_videos else None
        source_files.append({
            "path": str(path),
            "sha256": _sha256_file(path),
            "session_id": session,
            "frame_count": len(session_frames),
            "selected_frame_count": included_frames,
            "video_path": str(video) if video else "",
            "video_sha256": _sha256_file(video) if video else "",
        })

    if len(set(observation_ids)) != len(observation_ids):
        raise ValueError("DLC H5 files contain duplicate session/frame identities")
    labels_input = data.get("labels_path") or data.get("labels_by_session")
    targets, evaluation_mask, taxonomy = _load_aligned_labels(
        labels_input, sessions, frames,
        csv_frame_base=csv_frame_base,
        pose_frame_base=pose_frame_base,
        frame_offsets_by_session=(video_frame_offsets
                                  if label_frame_reference == "video" else {}),
    )
    reserved = list(data.get("reserved_evaluation", [False] * len(inputs)))
    if len(reserved) != len(inputs) or any(type(value) is not bool for value in reserved):
        raise ValueError("reserved_evaluation must contain one boolean per observation")
    reserved_ranges = data.get("reserved_evaluation_ranges_by_session", {})
    if not isinstance(reserved_ranges, dict):
        raise ValueError("reserved_evaluation_ranges_by_session must be an object")
    for session, interval in reserved_ranges.items():
        if (not isinstance(session, str) or not isinstance(interval, dict)
                or type(interval.get("start")) is not int
                or type(interval.get("stop")) is not int
                or interval["start"] < 0 or interval["stop"] <= interval["start"]):
            raise ValueError("Reserved frame ranges must use nonnegative [start, stop) integers")
        indices = [index for index, value in enumerate(sessions) if value == session]
        if not indices or not any(
            interval["start"] <= frames[index] < interval["stop"] for index in indices
        ):
            raise ValueError(f"Reserved frame range does not overlap pose frames for {session}")
        for index in indices:
            if interval["start"] <= frames[index] < interval["stop"]:
                reserved[index] = True
                partition_names[index] = "test"
    groups = sorted(set(sessions))
    train = [i for i, part in enumerate(partition_names) if part == "train"]
    validation = [i for i, part in enumerate(partition_names) if part == "validation"]
    test = [i for i, part in enumerate(partition_names) if part == "test"]
    if isinstance(labels_input, dict):
        label_reference = {}
        for session, raw_path in labels_input.items():
            path = Path(raw_path).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            label_reference[str(session)] = {"path": str(path), "sha256": _sha256_file(path)}
    elif labels_input:
        path = Path(labels_input).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        label_reference = {"path": str(path), "sha256": _sha256_file(path)}
    else:
        label_reference = None
    from storm.config import fingerprint

    data_fingerprint = fingerprint({
        "source_files": source_files,
        "labels": label_reference,
        "recipe": {
            "fps": fps,
            "feature_names": feature_names,
            "train_session_ids": sorted(train_sessions),
            "validation_session_ids": sorted(validation_sessions),
            "test_session_ids": sorted(test_sessions),
            "reserved_evaluation": reserved,
            "reserved_evaluation_ranges_by_session": reserved_ranges,
            "csv_frame_base": csv_frame_base,
            "pose_frame_base": pose_frame_base,
            "label_frame_reference": label_frame_reference,
            "video_frame_offsets_by_session": video_frame_offsets,
            "video_discontinuities_by_session": video_discontinuities,
            "frame_start": frame_start,
            "frame_stop": frame_stop,
            "max_frames": max_frames,
            "key": key,
        },
    })
    result = {
        "data_fingerprint": data_fingerprint,
        "label_source": label_reference,
        "inputs": inputs,
        "targets": targets,
        "train": train,
        "validation": validation,
        "test": test,
        "observation_ids": observation_ids,
        "feature_names": feature_names,
        "likelihood_bodyparts": likelihood_bodyparts,
        "likelihoods": likelihoods,
        "frames": frames,
        "video_frames": video_frames,
        "sessions": sessions,
        "segments": segments,
        "partitions": partition_names,
        "evaluation_mask": evaluation_mask,
        "reserved_evaluation": reserved,
        "reserved_evaluation_ranges_by_session": reserved_ranges,
        "video_discontinuities_by_session": video_discontinuities,
        "taxonomy": taxonomy,
        "fps": fps,
        "source_files": source_files,
    }
    if train_sessions or validation_sessions or test_sessions:
        result["groups"] = sessions
    return result


def _dlc_session_id(path: Path) -> str:
    stem = path.stem
    for marker in ("DLC_", "DeepCut_"):
        if marker in stem:
            stem = stem.split(marker, 1)[0].rstrip("_")
            break
    return stem or path.stem


def _session_partition(session, train_sessions, validation_sessions, test_sessions):
    if session in validation_sessions:
        return "validation"
    if session in test_sessions:
        return "test"
    if session in train_sessions:
        return "train"
    if not (train_sessions or validation_sessions or test_sessions):
        return "train"
    return "unassigned"


def _load_aligned_labels(label_source, sessions, frames, *, csv_frame_base, pose_frame_base,
                         frame_offsets_by_session=None):
    if not label_source:
        return None, [True] * len(frames), []
    import pandas as pd

    sessions_present = list(dict.fromkeys(sessions))
    if isinstance(label_source, dict):
        label_paths = {str(session): Path(path).expanduser().resolve()
                       for session, path in label_source.items()}
    elif len(sessions_present) == 1:
        label_paths = {sessions_present[0]: Path(label_source).expanduser().resolve()}
    else:
        raise ValueError("Multi-session labels must be supplied as labels_by_session")
    tables = {}
    taxonomy = None
    for session, path in label_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        labels = pd.read_csv(path)
        if "Frame" not in labels or len(labels.columns) < 2:
            raise ValueError("Labels must have a Frame column and at least one taxonomy column")
        current_taxonomy = [str(column) for column in labels.columns if column != "Frame"]
        if taxonomy is None:
            taxonomy = current_taxonomy
        elif current_taxonomy != taxonomy:
            raise ValueError("Session label CSVs must declare the same taxonomy in the same order")
        frame_values = pd.to_numeric(labels["Frame"], errors="raise")
        if (frame_values.isna().any() or frame_values.duplicated().any()
                or (frame_values % 1 != 0).any()):
            raise ValueError(f"Labels contain missing, duplicate, or noninteger frames for {session}")
        labels.index = frame_values.astype(int)
        tables[session] = labels[taxonomy].apply(pd.to_numeric, errors="coerce")
    targets, mask = [], []
    for session, frame in zip(sessions, frames, strict=True):
        table = tables.get(session)
        if table is None:
            targets.append(None)
            mask.append(False)
            continue
        frame_offset = (frame_offsets_by_session or {}).get(session, 0)
        csv_frame = frame - pose_frame_base + csv_frame_base + frame_offset
        row = table.reindex([csv_frame]).to_numpy()[0]
        valid = bool(all(value in (0, 1) for value in row) and sum(
            value == 1 for value in row
        ) == 1)
        targets.append(int(row.argmax()) if valid else None)
        mask.append(valid)
    return targets, mask, taxonomy


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def coordinate_columns(
    table: pd.DataFrame,
    *,
    suffixes: Iterable[str] = ("_x", "_y"),
) -> list[str]:
    suffixes = tuple(suffixes)
    return [column for column in table.columns if str(column).endswith(suffixes)]


def build_pose_dataset_loader(config: PoseDatasetConfig):
    def load(reference: DataRef) -> Dataset:
        import pandas as pd

        from .reconstruction import build_benchmark

        pose_path = Path(reference.metadata.get("pose_path") or config.pose_path)
        labels_path = reference.metadata.get("label_path") or config.label_path
        if reference.metadata.get("source_hashes"):
            current = PoseDatasetConfig(pose_path, Path(labels_path) if labels_path else None,
                                        config.max_frames, config.coordinate_suffixes, config.fps)
            if current.data_ref().fingerprint != reference.fingerprint:
                raise ValueError("Registered source or preparation recipe changed")
        table = load_pose_table(pose_path, preserve_index=True)
        pose_index = table.index.to_numpy()
        if config.max_frames is not None:
            table = table.head(config.max_frames)
        table = table.reset_index(drop=True)
        columns = coordinate_columns(table, suffixes=config.coordinate_suffixes)
        inputs = table[columns].astype(float)
        targets = None
        annotation_metadata = {}
        if labels_path:
            header = pd.read_csv(labels_path, nrows=0).columns
            if "Frame" in header and "labels" not in header and not any(
                c.startswith("Labeler_") for c in header
            ):
                benchmark = build_benchmark(labels_path, pose_index=pose_index,
                                            fps=config.fps, last_frame=len(inputs))
                targets = pd.Series(benchmark["labels"], dtype="Int64")
                annotation_metadata = {key: benchmark[key] for key in
                                       ("taxonomy", "evaluation_mask", "alignment",
                                        "csv_frames", "pose_indices", "video_frames")}
            else:
                targets = _load_targets(Path(labels_path), len(inputs))
        return Dataset(
            inputs=inputs,
            targets=targets,
            metadata={
                "pose_path": str(pose_path),
                "label_path": str(labels_path) if labels_path else "",
                "coordinate_columns": columns,
                "frame_count": len(inputs),
                **annotation_metadata,
            },
        )

    return load


def _load_targets(path: Path, n_rows: int) -> pd.Series:
    labels = pd.read_csv(path).head(n_rows)
    if "labels" in labels.columns:
        return labels["labels"]
    label_columns = [column for column in labels.columns if column.startswith("Labeler_")]
    if not label_columns:
        numeric = labels.select_dtypes(include="number")
        label_columns = [
            column
            for column in numeric.columns
            if column.lower() != "frame" and not column.endswith(("_x", "_y"))
        ]
    if not label_columns:
        raise ValueError(f"No label columns found in {path}")
    return labels[label_columns].max(axis=1)


def _flatten_pose_column(column: tuple[object, ...]) -> str:
    parts = [str(part) for part in column if str(part)]
    if len(parts) >= 2 and parts[-1] in {"x", "y", "likelihood"}:
        return f"{parts[-2]}_{parts[-1]}"
    return "_".join(parts)
