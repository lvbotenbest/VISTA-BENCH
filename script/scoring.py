#!/usr/bin/env python3
"""Deterministic binary-expected scoring with independent V/T/K dimensions."""
from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from dataset import (DataError, PUBLIC_RUBRIC_SETS, iter_jsonl, load_json, load_manifest,
                     projected_criteria, resolve_release_path, rubric_asset_key, write_jsonl_atomic)
from judge import validate_judgment, validate_judge_record

SCORING_PROTOCOL = "binary_expected_independent_vtk"
IMPORTANCE_WEIGHTS = {"core": 3, "supporting": 2, "minor": 1}
DIMENSIONS = ("image_understanding", "translation_quality", "knowledge_entailment")


def rounded(value: Decimal, digits: int) -> float:
    return float(value.quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))


def empty_score(image_name: str, target_lang: str, profile: str, rubric_set: str,
                status: str) -> dict:
    return {"schema_version": "vista_score", "image_name": image_name,
            "target_lang": target_lang, "profile": profile, "rubric_set": rubric_set,
            "scoring_protocol": SCORING_PROTOCOL, "status": status,
            "dimension_scores": {key: None for key in DIMENSIONS},
            "dimension_scores_100": {key: None for key in DIMENSIONS},
            "criterion_scores": []}


def score_task(rubric: dict, target_lang: str, judge_output: dict | None,
               *, profile: str = "image", rubric_set: str = "basic_rubric",
               precision: int = 6) -> dict:
    if rubric_set not in PUBLIC_RUBRIC_SETS:
        raise DataError("invalid rubric set")
    if type(precision) is not int or precision < 0:
        raise DataError("precision must be a non-negative integer")
    image_name = rubric.get("image_name")
    if not isinstance(image_name, str) or not image_name.strip():
        raise DataError("rubric must define image_name")
    criteria = projected_criteria(rubric, target_lang, profile=profile)
    result = empty_score(image_name, target_lang, profile, rubric_set, "unscorable")
    if judge_output is not None or criteria:
        validate_judgment(judge_output, image_name, target_lang, criteria)
    if not criteria:
        return result
    by_judgment = {row["criterion_id"]: row for row in judge_output["judgments"]}
    numerators = {key: Decimal("0") for key in DIMENSIONS}
    denominators = {key: Decimal("0") for key in DIMENSIONS}
    details = []
    for criterion in criteria:
        judgment = by_judgment[criterion["id"]]
        expected = judgment["expected_checks"]
        expected_met = sum(check["result"] == "met" for check in expected)
        expected_total = len(expected)
        expected_ratio = Decimal(expected_met) / Decimal(expected_total)
        forbidden_triggered = any(check["matched"] for check in judgment["forbidden_checks"])
        value = Decimal("0") if forbidden_triggered else expected_ratio
        dimension = criterion["dimension"]
        weight = IMPORTANCE_WEIGHTS[criterion["importance"]]
        numerators[dimension] += weight * value
        denominators[dimension] += weight
        details.append({"criterion_id": criterion["id"], "dimension": dimension,
                        "importance": criterion["importance"], "weight": weight,
                        "expected_met": expected_met, "expected_total": expected_total,
                        "expected_ratio": float(expected_ratio),
                        "forbidden_triggered": forbidden_triggered,
                        "criterion_score": float(value), "rationale": judgment["rationale"]})
    scores = {key: (numerators[key] / denominators[key] if denominators[key] else None)
              for key in DIMENSIONS}
    result.update({"status": "ok",
                   "dimension_scores": {k: None if v is None else rounded(v, precision)
                                        for k, v in scores.items()},
                   "dimension_scores_100": {k: None if v is None else rounded(v * 100, 2)
                                            for k, v in scores.items()},
                   "criterion_scores": details})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--judge-records", type=Path, required=True)
    parser.add_argument(
        "--rubric-set", choices=PUBLIC_RUBRIC_SETS, required=True,
        help=("rubric=主要评测/强模型 (primary evaluation for stronger models); "
              "basic_rubric=简单规则/小模型 (easier criteria for smaller models)"),
    )
    parser.add_argument("--profile", choices=("image", "text"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = {row["name"]: row for row in load_manifest(args.root, verify_hashes=False)}

    def score_records():
        seen = set()
        for record in iter_jsonl(args.judge_records):
            if record.get("schema_version") != "vista_judge_record":
                raise DataError("invalid judge record schema_version")
            if record.get("profile") != args.profile or record.get("rubric_set") != args.rubric_set:
                raise DataError("judge record profile/rubric set does not match scoring arguments")
            image_name = record.get("image_name")
            if not isinstance(image_name, str) or image_name not in manifest:
                raise DataError(f"judge record is outside manifest: {image_name}")
            manifest_row = manifest[image_name]
            rubric = load_json(resolve_release_path(
                args.root, manifest_row["assets"][rubric_asset_key(args.rubric_set)]["path"]))
            if rubric.get("image_name") != image_name:
                raise DataError("rubric image_name does not match judge record")
            target_lang = record.get("target_lang")
            if not isinstance(target_lang, str) or len(target_lang.strip()) < 2:
                raise DataError("invalid judge record target_lang")
            criteria = projected_criteria(rubric, target_lang, profile=args.profile)
            validate_judge_record(record, criteria)
            if (record["task_id"] != f"{manifest_row['seq']}:{target_lang}" or
                    record["source_lang"] != manifest_row["language"]):
                raise DataError("judge record identity does not match manifest")
            if record["task_id"] in seen:
                raise DataError("duplicate judge record task_id")
            seen.add(record["task_id"])
            base = {k: record[k] for k in ("task_id", "image_name", "source_lang", "target_lang")}
            base.update({"level1": manifest_row["level1"], "level1_name_en": manifest_row["level1_name_en"]})
            if record["status"] == "failed":
                result = empty_score(image_name, target_lang, args.profile, args.rubric_set, "judge_failed")
            else:
                result = score_task(rubric, target_lang, record["judge_output"],
                                    profile=args.profile, rubric_set=args.rubric_set)
            yield {**base, **result}

    write_jsonl_atomic(args.output, score_records())


if __name__ == "__main__":
    main()
