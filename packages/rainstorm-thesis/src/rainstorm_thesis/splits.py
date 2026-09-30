"""Generate reproducible experiment split files for RAINSTORM sessions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Migrated from Tesis_Facu/src/experiment/generate_splits.py.
# Original copyright and permission notice: docs/reconstruction/THESIS_LICENSE.
@dataclass(frozen=True)
class SessionRecord:
    session_id: str
    path: Path


@dataclass(frozen=True)
class ExperimentSplits:
    sweep: list[SessionRecord]
    final_train: list[SessionRecord]
    final_test: list[SessionRecord]
    final_val_internal: list[SessionRecord]


def infer_session_type(session_id: str) -> str:
    if session_id.startswith("Hab"):
        return "Hab"
    if session_id.startswith("TR"):
        return "TR"
    if session_id.startswith("TS"):
        return "TS"
    return "Other"


def build_experiment_splits(
    session_paths: list[str | Path],
    *,
    seed: int = 156,
    sweep_size: int = 10,
) -> ExperimentSplits:
    session_paths = sorted(map(Path, session_paths))
    records = [
        SessionRecord(
            session_id=Path(path).stem.split("DLC_")[0].rstrip("_"), path=Path(path)
        )
        for path in session_paths
    ]
    ids = [record.session_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate session IDs")
    if any(infer_session_type(sid) == "Other" for sid in ids):
        raise ValueError("Unknown session type")
    if not 3 <= sweep_size <= len(records) - 2:
        raise ValueError("Sweep must leave two validation sessions")
    rng = np.random.default_rng(seed)
    grouped = _group_by_type(records)

    sweep = _stratified_take(grouped, sweep_size, rng)
    if len(sweep) != sweep_size:
        raise ValueError("Insufficient sessions for stratified sweep")
    remaining = [record for record in records if record not in set(sweep)]

    val_internal = _take_internal_validation(remaining, rng)
    train_test_pool = [
        record for record in remaining if record not in set(val_internal)
    ]
    shuffled_pool = _shuffled(train_test_pool, rng)
    train_count = round(0.70 * len(shuffled_pool))
    final_train = sorted(
        sweep + shuffled_pool[:train_count], key=lambda record: record.session_id
    )
    final_test = sorted(
        shuffled_pool[train_count:], key=lambda record: record.session_id
    )

    return ExperimentSplits(
        sweep=sorted(sweep, key=lambda record: record.session_id),
        final_train=final_train,
        final_test=final_test,
        final_val_internal=sorted(val_internal, key=lambda record: record.session_id),
    )


def write_split_files(splits: ExperimentSplits, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_split(output / "sweep_sessions.txt", splits.sweep)
    _write_split(output / "final_train_sessions.txt", splits.final_train)
    _write_split(output / "final_test_sessions.txt", splits.final_test)
    _write_split(output / "final_val_internal.txt", splits.final_val_internal)


def _group_by_type(records: list[SessionRecord]) -> dict[str, list[SessionRecord]]:
    grouped: dict[str, list[SessionRecord]] = {"Hab": [], "TR": [], "TS": []}
    for record in records:
        session_type = infer_session_type(record.session_id)
        if session_type in grouped:
            grouped[session_type].append(record)
    if any(not values for values in grouped.values()):
        raise ValueError(
            "Sessions must include at least one Hab, TR, and TS recording."
        )
    return grouped


def _stratified_take(
    grouped: dict[str, list[SessionRecord]],
    total: int,
    rng: np.random.Generator,
) -> list[SessionRecord]:
    sizes = {key: len(values) for key, values in grouped.items()}
    total_available = sum(sizes.values())
    raw = {key: total * size / total_available for key, size in sizes.items()}
    counts = {key: max(1, int(value)) for key, value in raw.items()}

    while sum(counts.values()) < total:
        key = max(raw, key=lambda item: raw[item] - int(raw[item]))
        counts[key] += 1
        raw[key] = int(raw[key])
    while sum(counts.values()) > total:
        key = max(counts, key=counts.get)
        if counts[key] > 1:
            counts[key] -= 1
        else:
            break

    selected: list[SessionRecord] = []
    for key, values in grouped.items():
        selected.extend(_shuffled(values, rng)[: counts[key]])
    return selected


def _take_internal_validation(
    records: list[SessionRecord],
    rng: np.random.Generator,
) -> list[SessionRecord]:
    hab = [
        record for record in records if infer_session_type(record.session_id) == "Hab"
    ]
    tr_ts = [
        record
        for record in records
        if infer_session_type(record.session_id) in {"TR", "TS"}
    ]
    if not hab or not tr_ts:
        raise ValueError("Internal validation needs one Hab and one TR/TS session.")
    return [_shuffled(hab, rng)[0], _shuffled(tr_ts, rng)[0]]


def _shuffled(
    records: list[SessionRecord], rng: np.random.Generator
) -> list[SessionRecord]:
    indexes = rng.permutation(len(records))
    return [records[int(index)] for index in indexes]


def _write_split(path: Path, records: list[SessionRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(f"{record.session_id} {record.path}\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/rainstorm_analysis_simon")
    parser.add_argument("--output-dir", default="notebooks/model_experiments/data/splits")
    parser.add_argument("--seed", type=int, default=156)
    parser.add_argument("--pattern", default="*.h5")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    session_paths = sorted(Path(args.data_dir).glob(args.pattern))
    splits = build_experiment_splits(session_paths, seed=args.seed)
    write_split_files(splits, args.output_dir)


if __name__ == "__main__":
    main()
