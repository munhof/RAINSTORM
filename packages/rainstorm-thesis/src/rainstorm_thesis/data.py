from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from storm import DataRef, Dataset


@dataclass(frozen=True)
class PoseDatasetConfig:
    pose_path: Path
    label_path: Path | None = None
    max_frames: int | None = 2_000
    coordinate_suffixes: tuple[str, ...] = ("_x", "_y")

    def data_ref(self, *, identifier: str | None = None) -> DataRef:
        return DataRef(
            identifier=identifier or self.pose_path.stem,
            fingerprint=f"path:{self.pose_path.name}",
            metadata={
                "pose_path": str(self.pose_path),
                "label_path": str(self.label_path) if self.label_path else "",
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


def load_pose_table(path: Path | str, *, max_frames: int | None = None) -> pd.DataFrame:
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
    return table.reset_index(drop=True)


def coordinate_columns(
    table: pd.DataFrame,
    *,
    suffixes: Iterable[str] = ("_x", "_y"),
) -> list[str]:
    suffixes = tuple(suffixes)
    return [column for column in table.columns if str(column).endswith(suffixes)]


def build_pose_dataset_loader(config: PoseDatasetConfig):
    def load(reference: DataRef) -> Dataset:
        pose_path = Path(reference.metadata.get("pose_path") or config.pose_path)
        labels_path = reference.metadata.get("label_path") or config.label_path
        table = load_pose_table(pose_path, max_frames=config.max_frames)
        columns = coordinate_columns(table, suffixes=config.coordinate_suffixes)
        inputs = table[columns].astype(float)
        targets = _load_targets(Path(labels_path), len(inputs)) if labels_path else None
        return Dataset(
            inputs=inputs,
            targets=targets,
            metadata={
                "pose_path": str(pose_path),
                "label_path": str(labels_path) if labels_path else "",
                "coordinate_columns": columns,
                "frame_count": len(inputs),
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
