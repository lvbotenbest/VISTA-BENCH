#!/usr/bin/env python3
"""Publish independent V/T/K scores from criterion pools and direction macros."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path

from dataset import DataError, iter_jsonl, load_json

SCORING_PROTOCOL = "binary_expected_independent_vtk"
DIMENSIONS = ("translation_quality", "image_understanding", "knowledge_entailment")
RANK_FIELDS = dict(zip(DIMENSIONS, ("rank_translation", "rank_image", "rank_knowledge")))
COUNT_FIELDS = dict(zip(DIMENSIONS, ("translation_directions", "image_directions", "knowledge_directions")))
LEADERBOARD_FIELDS = ["model", *RANK_FIELDS.values(), *DIMENSIONS, *COUNT_FIELDS.values()]
RUN_COUNT_FIELDS = ("selected", "candidate_succeeded", "candidate_failed", "judge_ok", "judge_zero_criteria")
STATUS_FIELDS = ("ok", "unscorable", "judge_failed")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (f"{value:.6f}" if key in DIMENSIONS and value is not None else value)
                             for key, value in row.items()})


def parse_inputs(values: list[str]) -> list[tuple[str, Path]]:
    result = []
    for value in values:
        if "=" not in value:
            raise DataError("--scores must use MODEL=PATH")
        model, path = value.split("=", 1)
        if not model.strip() or not path.strip():
            raise DataError("--scores must use nonempty MODEL=PATH")
        result.append((model, Path(path)))
    if len({model for model, _ in result}) != len(result):
        raise DataError("duplicate model label")
    return result


def number(value: object, field: str, *, maximum: int = 1) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataError(f"{field} must be a finite number")
    result = Decimal(str(value))
    if not result.is_finite():
        raise DataError(f"{field} must be a finite number")
    if not 0 <= result <= maximum:
        raise DataError(f"{field} must be between 0 and {maximum}")
    return result


def validate_score(row: dict, profile: str, rubric_set: str) -> None:
    required = {"schema_version", "task_id", "image_name", "source_lang", "target_lang",
                "level1", "level1_name_en", "profile", "rubric_set", "scoring_protocol",
                "status", "dimension_scores", "dimension_scores_100", "criterion_scores"}
    if not isinstance(row, dict) or set(row) != required:
        legacy = any(key == "overall" or key.startswith("overall_") for key in row) if isinstance(row, dict) else False
        legacy = legacy or (isinstance(row, dict) and any(
            key in row for key in ("active_dimension_weights", "dimension_weights")))
        if legacy:
            raise DataError("legacy overall/dimension-weight fields are not allowed")
        actual = set(row) if isinstance(row, dict) else set()
        missing = ", ".join(sorted(required - actual)) or "none"
        extra = ", ".join(sorted(actual - required)) or "none"
        raise DataError(f"score fields do not match the current protocol; missing={missing}; extra={extra}")
    for key, expected in (("schema_version", "vista_score"), ("scoring_protocol", SCORING_PROTOCOL),
                          ("profile", profile), ("rubric_set", rubric_set)):
        if row[key] != expected:
            raise DataError(f"{key} mismatch: expected {expected!r}")
    if row["status"] not in STATUS_FIELDS:
        raise DataError("invalid score status")
    for key in ("task_id", "image_name", "source_lang", "target_lang", "level1", "level1_name_en"):
        if not isinstance(row[key], str) or not row[key].strip():
            raise DataError(f"{key} must be a nonempty string")
    dimensions = row["dimension_scores"]
    dimensions_100 = row["dimension_scores_100"]
    if not isinstance(dimensions, dict) or set(dimensions) != set(DIMENSIONS):
        raise DataError("dimension_scores must contain exactly the three independent dimensions")
    if not isinstance(dimensions_100, dict) or set(dimensions_100) != set(DIMENSIONS):
        raise DataError("dimension_scores_100 must contain exactly the three independent dimensions")
    for dimension in DIMENSIONS:
        value, value_100 = dimensions[dimension], dimensions_100[dimension]
        if (value is None) != (value_100 is None):
            raise DataError(f"dimension score scales disagree: {dimension}")
        if value is not None:
            number(value, f"dimension_scores.{dimension}")
            number(value_100, f"dimension_scores_100.{dimension}", maximum=100)
    criteria = row["criterion_scores"]
    if not isinstance(criteria, list):
        raise DataError("criterion_scores must be a list")
    if row["status"] == "ok":
        if not criteria or all(value is None for value in dimensions.values()):
            raise DataError("ok score must contain criterion scores and a non-null dimension")
    elif criteria or any(value is not None for value in dimensions.values()):
        raise DataError("unscorable/judge_failed scores must have null dimensions and empty criteria")
    if profile == "text" and dimensions["image_understanding"] is not None:
        raise DataError("text profile cannot contain image_understanding scores")
    seen = set()
    criterion_fields = {"criterion_id", "dimension", "importance", "weight", "expected_met",
                        "expected_total", "expected_ratio", "forbidden_triggered",
                        "criterion_score", "rationale"}
    for criterion in criteria:
        if not isinstance(criterion, dict) or set(criterion) != criterion_fields:
            raise DataError("criterion score fields do not match the current protocol")
        identifier = criterion["criterion_id"]
        if not isinstance(identifier, str) or not identifier.strip() or identifier in seen:
            raise DataError("invalid or duplicate criterion_id")
        seen.add(identifier)
        dimension = criterion["dimension"]
        if dimension not in DIMENSIONS:
            raise DataError("invalid criterion dimension")
        if profile == "text" and dimension == "image_understanding":
            raise DataError("text profile cannot contain image_understanding criteria")
        importance = criterion["importance"]
        if importance not in ("core", "supporting", "minor"):
            raise DataError("invalid criterion importance")
        if type(criterion["weight"]) is not int or criterion["weight"] != {"core": 3, "supporting": 2, "minor": 1}[importance]:
            raise DataError("criterion weight does not match importance")
        expected_met, expected_total = criterion["expected_met"], criterion["expected_total"]
        if (type(expected_met) is not int or type(expected_total) is not int or
                expected_total < 1 or not 0 <= expected_met <= expected_total):
            raise DataError("invalid expected_met/expected_total")
        ratio = number(criterion["expected_ratio"], "expected_ratio")
        if abs(ratio - Decimal(expected_met) / Decimal(expected_total)) > Decimal("1e-15"):
            raise DataError("expected_ratio does not match expected_met/expected_total")
        if not isinstance(criterion["forbidden_triggered"], bool):
            raise DataError("forbidden_triggered must be boolean")
        criterion_score = number(criterion["criterion_score"], "criterion_score")
        expected_score = Decimal(0) if criterion["forbidden_triggered"] else ratio
        if criterion_score != expected_score:
            raise DataError("criterion_score does not follow forbidden veto and expected ratio")
        if not isinstance(criterion["rationale"], str) or not criterion["rationale"].strip():
            raise DataError("criterion rationale must be a nonempty string")


def pool(rows: list[dict]) -> dict[str, Decimal | None]:
    numerators = dict.fromkeys(DIMENSIONS, Decimal(0))
    denominators = dict.fromkeys(DIMENSIONS, Decimal(0))
    for row in rows:
        for criterion in row["criterion_scores"]:
            dimension = criterion["dimension"]
            weight = Decimal(criterion["weight"])
            numerators[dimension] += Decimal(str(criterion["criterion_score"])) * weight
            denominators[dimension] += weight
    return {dimension: (numerators[dimension] / denominators[dimension]
                        if denominators[dimension] else None) for dimension in DIMENSIONS}


def macro(scores: list[dict]) -> dict:
    result = {}
    for dimension in DIMENSIONS:
        values = [row[dimension] for row in scores if row[dimension] is not None]
        result[dimension] = sum(values, Decimal(0)) / len(values) if values else None
        result[COUNT_FIELDS[dimension]] = len(values)
    return result


def public_scores(scores: dict) -> dict:
    # Round only at publication, never before a direction or domain macro.
    return {key: (None if value is None else
                  float((value * 100).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)) + 0.0)
            if key in DIMENSIONS else value for key, value in scores.items()}


def rank_leaderboard(rows: list[dict]) -> None:
    def published(value: float) -> Decimal:
        return Decimal(str(value)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)

    for dimension, rank_field in RANK_FIELDS.items():
        for row in rows:
            row[rank_field] = None
        ranked = sorted((row for row in rows if row[dimension] is not None),
                        key=lambda row: -published(row[dimension]))
        previous, rank = None, None
        for position, row in enumerate(ranked, 1):
            current = published(row[dimension])
            if current != previous:
                rank = position
                previous = current
            row[rank_field] = rank
    rows.sort(key=lambda row: (row["translation_quality"] is None,
                              -published(row["translation_quality"] or 0), row["model"]))


def load_run_counts(path: Path, profile: str, rubric_set: str) -> dict:
    metadata = load_json(path)
    if metadata.get("schema_version") != "vista_benchmark_run":
        raise DataError("run metadata schema_version mismatch")
    if metadata.get("scoring_protocol") != SCORING_PROTOCOL:
        raise DataError("run metadata scoring_protocol mismatch")
    parameters = metadata.get("parameters")
    if not isinstance(parameters, dict):
        raise DataError("run metadata parameters must be an object")
    for key, expected in (("track", profile), ("rubric_set", rubric_set),
                          ("scoring_protocol", SCORING_PROTOCOL)):
        if parameters.get(key) != expected:
            raise DataError(f"run metadata {key} mismatch")
    counts = metadata.get("counts") or {}
    if not isinstance(counts, dict):
        raise DataError("run metadata counts must be an object")
    result = {key: counts.get(key, 0) for key in RUN_COUNT_FIELDS}
    if any(type(value) is not int or value < 0 for value in result.values()):
        raise DataError("run metadata counts must be nonnegative integers")
    return result


def summarize(args: argparse.Namespace) -> None:
    inputs = parse_inputs(args.scores)
    if args.run_metadata and len(inputs) != 1:
        raise DataError("--run-metadata is only allowed with a single model")
    run_counts = (load_run_counts(args.run_metadata, args.profile, args.rubric_set)
                  if args.run_metadata else {})
    records = {}
    coverage = []
    for model, path in inputs:
        rows, seen = [], set()
        for index, row in enumerate(iter_jsonl(path), 1):
            try:
                validate_score(row, args.profile, args.rubric_set)
                if row["task_id"] in seen:
                    raise DataError(f"duplicate task_id: {row['task_id']}")
            except DataError as exc:
                raise DataError(f"{path}: record {index}: {exc}") from exc
            seen.add(row["task_id"])
            rows.append(row)
        records[model] = rows
        coverage.append({"model": model, **run_counts, "records": len(rows),
                         **{status: sum(row["status"] == status for row in rows) for status in STATUS_FIELDS}})
    coverage.sort(key=lambda row: row["model"])

    direction_rows, direction_level1_rows, level1_rows, leaderboard = [], [], [], []
    for model, rows in sorted(records.items()):
        directions, direction_domains = defaultdict(list), defaultdict(list)
        level1_names = {}
        for row in rows:
            if row["status"] != "ok":
                continue
            direction = (row["source_lang"], row["target_lang"])
            directions[direction].append(row)
            direction_domains[(*direction, row["level1"])].append(row)
            if row.get("level1_name_en"):
                level1_names.setdefault(row["level1"], row["level1_name_en"])
        direction_scores = []
        for (source, target), group in sorted(directions.items()):
            score = pool(group)
            direction_scores.append(score)
            direction_rows.append({"model": model, "source_lang": source, "target_lang": target,
                                   **public_scores(score)})
        domain_scores = defaultdict(list)
        for (source, target, level1), group in sorted(direction_domains.items()):
            score = pool(group)
            domain_scores[level1].append(score)
            direction_level1_rows.append({"model": model, "source_lang": source, "target_lang": target,
                                          "level1": level1, "level1_name_en": level1_names.get(level1, ""),
                                          **public_scores(score)})
        for level1, scores in sorted(domain_scores.items()):
            level1_rows.append({"model": model, "level1": level1,
                                "level1_name_en": level1_names.get(level1, ""), **public_scores(macro(scores))})
        leaderboard.append({"model": model, **public_scores(macro(direction_scores))})
    rank_leaderboard(leaderboard)
    leaderboard = [{field: row[field] for field in LEADERBOARD_FIELDS} for row in leaderboard]
    summary = {
        "schema_version": "vista_public_summary",
        "scoring_protocol": SCORING_PROTOCOL,
        "profile": args.profile,
        "rubric_set": args.rubric_set,
        "cross_track_comparable": False,
        "formula": {dimension: {
            "direction": "sum(weight * criterion_score) / sum(weight) over ok criteria in model/source/target/dimension; weight is core=3, supporting=2, minor=1",
            "leaderboard": "equal-weight macro of non-null direction scores for this model and dimension",
            "direction_level1": "sum(weight * criterion_score) / sum(weight) over ok criteria in model/source/target/level1/dimension; weight is core=3, supporting=2, minor=1",
            "level1": "equal-weight macro of non-null direction_level1 scores within this model, level1 and dimension",
        } for dimension in DIMENSIONS},
        "ranking": {
            "dimensions": RANK_FIELDS,
            "score_scale": 100,
            "public_decimals": 6,
            "rounding": "ROUND_HALF_UP",
            "method": "competition",
            "order": "descending",
            "null_rank": None,
            "row_order": ["translation_quality descending", "null last", "model ascending"],
        },
        "coverage": coverage,
        "leaderboard": leaderboard,
    }
    if args.rubric_set == "rubric":
        summary.update({"selection_leakage": True, "analysis_status": "exploratory",
                        "official_leaderboard": False})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "leaderboard.csv", leaderboard, LEADERBOARD_FIELDS)
    write_csv(args.output_dir / "scores_by_direction.csv", direction_rows,
              ["model", "source_lang", "target_lang", *DIMENSIONS])
    write_csv(args.output_dir / "scores_by_direction_level1.csv", direction_level1_rows,
              ["model", "source_lang", "target_lang", "level1", "level1_name_en", *DIMENSIONS])
    write_csv(args.output_dir / "scores_by_level1.csv", level1_rows,
              ["model", "level1", "level1_name_en", *DIMENSIONS, *COUNT_FIELDS.values()])
    write_csv(args.output_dir / "coverage.csv", coverage,
              ["model", *run_counts, "records", *STATUS_FIELDS])
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", action="append", required=True, help="MODEL=PATH; repeat per model")
    parser.add_argument("--profile", choices=("image", "text"), required=True)
    parser.add_argument(
        "--rubric-set", choices=("basic_rubric", "rubric"), required=True,
        help=("rubric=主要评测/强模型 (primary evaluation for stronger models); "
              "basic_rubric=简单规则/小模型 (easier criteria for smaller models)"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-metadata", type=Path, help="run coverage metadata; single-model summaries only")
    args = parser.parse_args()
    try:
        summarize(args)
    except (DataError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
