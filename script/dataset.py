#!/usr/bin/env python3
"""Strict release-manifest, rubric, and candidate loaders."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable, Iterator


class DataError(ValueError):
    pass


PUBLIC_RUBRIC_SETS = ("basic_rubric", "rubric")


def rubric_asset_key(rubric_set: str) -> str:
    """Map the public rubric name to its release-manifest asset key."""
    if rubric_set == "basic_rubric":
        return "simple_rubric"
    if rubric_set == "rubric":
        return "rubric"
    raise DataError("rubric set must be basic_rubric or rubric")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DataError(f"JSON root must be an object: {path}")
    return value


def resolve_release_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise DataError(f"unsafe release-relative path: {relative!r}")
    root = root.resolve()
    resolved = (root / path).resolve()
    if root not in resolved.parents:
        raise DataError(f"path escapes release root: {relative!r}")
    return resolved


def load_manifest(root: Path, *, verify_hashes: bool = True) -> list[dict]:
    records = []
    seen_names, seen_seq = set(), set()
    path = root / "data/manifest.jsonl"
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataError(f"invalid manifest JSON at line {line_number}: {exc}") from exc
            required = {"name", "seq", "language", "level1", "level1_name_en",
                        "level2", "level2_name_en", "assets"}
            if not isinstance(row, dict) or set(row) != required:
                raise DataError(f"invalid manifest fields at line {line_number}")
            if row["name"] in seen_names or row["seq"] in seen_seq:
                raise DataError(f"duplicate manifest identity at line {line_number}")
            seen_names.add(row["name"]); seen_seq.add(row["seq"])
            if set(row["assets"]) != {"image", "ocr", "verified_translation", "simple_rubric", "rubric"}:
                raise DataError(f"invalid asset set at line {line_number}")
            for asset in row["assets"].values():
                if not isinstance(asset, dict) or set(asset) != {"path", "sha256"}:
                    raise DataError(f"invalid asset descriptor at line {line_number}")
                target = resolve_release_path(root, asset["path"])
                if not target.is_file():
                    raise DataError(f"missing asset: {asset['path']}")
                if verify_hashes and sha256(target) != asset["sha256"]:
                    raise DataError(f"asset hash mismatch: {asset['path']}")
            records.append(row)
    return records


def projected_criteria(rubric: dict, target_lang: str, *, profile: str = "image") -> list[dict]:
    if target_lang not in rubric.get("target_langs", []):
        raise DataError(f"target language not in rubric: {target_lang}")
    override = (rubric.get("language_overrides") or {}).get(target_lang) or {}
    criteria = list(rubric.get("criteria") or []) + list(override.get("criteria") or [])
    allowed_dimensions = {"image_understanding", "translation_quality", "knowledge_entailment"}
    if profile == "text":
        criteria = [row for row in criteria if row.get("dimension") != "image_understanding"]
    elif profile != "image":
        raise DataError("profile must be image or text")
    ids = []
    for index, criterion in enumerate(criteria):
        if not isinstance(criterion, dict):
            raise DataError(f"criterion {index} is not an object")
        identifier = criterion.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise DataError(f"criterion {index} has no ID")
        if criterion.get("dimension") not in allowed_dimensions:
            raise DataError(f"criterion {identifier} has invalid dimension")
        if criterion.get("importance") not in {"core", "supporting", "minor"}:
            raise DataError(f"criterion {identifier} has invalid importance")
        if not isinstance(criterion.get("expected"), list) or not criterion["expected"]:
            raise DataError(f"criterion {identifier} has no expected checks")
        if not isinstance(criterion.get("forbidden", []), list):
            raise DataError(f"criterion {identifier} has invalid forbidden checks")
        ids.append(identifier)
    if len(ids) != len(set(ids)):
        raise DataError("duplicate projected criterion IDs")
    return criteria


def load_candidate(row: dict, *, expected_task_id: str | None = None) -> dict:
    required = {"schema_version", "task_id", "image_name", "source_lang", "target_lang", "candidate_blocks"}
    if not isinstance(row, dict) or set(row) != required:
        raise DataError("invalid candidate fields")
    if row["schema_version"] != "vista_self_segmented_candidate_v1":
        raise DataError("unsupported candidate schema")
    if expected_task_id is not None and row["task_id"] != expected_task_id:
        raise DataError("candidate task identity mismatch")
    blocks = row["candidate_blocks"]
    if not isinstance(blocks, list):
        raise DataError("candidate_blocks must be a list")
    ids = []
    for block in blocks:
        if not isinstance(block, dict) or set(block) != {"id", "source_text", "text"}:
            raise DataError("invalid candidate block")
        if not isinstance(block["id"], int) or block["id"] < 1:
            raise DataError("invalid candidate block ID")
        if not isinstance(block["source_text"], str) or not isinstance(block["text"], str):
            raise DataError("candidate block text must be strings")
        ids.append(block["id"])
    if len(ids) != len(set(ids)):
        raise DataError("duplicate candidate block IDs")
    return row


def select_tasks(tasks: Iterable[dict], *, target_langs: Iterable[str] | None = None,
                 limit: int | None = None, done: Iterable[str] = ()) -> list[dict]:
    """Fix the filtered/limited task range before removing completed tasks."""
    if limit is not None and limit < 0:
        raise DataError("limit must be non-negative")
    task_list = list(tasks)
    allowed = set(target_langs) if target_langs else None
    if allowed is not None:
        available = {task.get("target_lang") for task in task_list}
        unknown = sorted(allowed - available)
        if unknown:
            raise DataError("unknown target language(s): " + ", ".join(unknown))
    selected = [task for task in task_list
                if allowed is None or task.get("target_lang") in allowed]
    if limit is not None:
        selected = selected[:limit]
    completed = set(done)
    return [task for task in selected if task.get("task_id") not in completed]


def write_jsonl_atomic(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataError(f"invalid JSONL at line {line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise DataError(f"JSONL row {line_number} must be an object")
            yield value
