from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

from jsonschema import Draft202012Validator, ValidationError
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from dataset import DataError
from scoring import DIMENSIONS, SCORING_PROTOCOL, score_task

SCRIPT = Path(__file__).parents[1]
SCHEMA = json.loads((SCRIPT / "schemas/score.schema.json").read_text())
VALIDATOR = Draft202012Validator(SCHEMA)


def rubric(criteria):
    return {"image_name": "a.jpg", "target_langs": ["fr"], "criteria": criteria,
            "language_overrides": {}}


def criterion(identifier, dimension="translation_quality", importance="core", expected=3, forbidden=1):
    return {"id": identifier, "dimension": dimension, "importance": importance,
            "expected": ["x"] * expected, "forbidden": ["bad"] * forbidden, "source_blocks": [1]}


def judgment(identifier, results, matched=False):
    return {"criterion_id": identifier,
            "expected_checks": [{"index": i, "result": value} for i, value in enumerate(results)],
            "forbidden_checks": [{"index": 0, "matched": matched}],
            "evidence": [], "rationale": "checked"}


def output(*judgments):
    return {"schema_version": "vista_judge_output", "image_name": "a.jpg", "target_lang": "fr",
            "judgments": list(judgments)}


def check_score_schema(score):
    VALIDATOR.validate({"task_id": "1:fr", "source_lang": "English", "level1": "D01",
                        "level1_name_en": "Transportation", **score})
    assert score["schema_version"] == "vista_score"
    assert score["scoring_protocol"] == SCORING_PROTOCOL == "binary_expected_independent_vtk"
    assert not {"overall_score", "overall_score_100", "active_dimension_weights"} & score.keys()


def test_two_of_three_expected_atoms_are_met():
    score = score_task(rubric([criterion("T1")]), "fr", output(judgment("T1", ["met", "met", "unmet"])))
    assert score["dimension_scores"] == dict.fromkeys(DIMENSIONS) | {"translation_quality": 0.666667}
    assert score["dimension_scores_100"]["translation_quality"] == 66.67
    assert score["criterion_scores"] == [{
        "criterion_id": "T1", "dimension": "translation_quality", "importance": "core", "weight": 3,
        "expected_met": 2, "expected_total": 3, "expected_ratio": 2 / 3,
        "forbidden_triggered": False, "criterion_score": 2 / 3, "rationale": "checked"}]
    check_score_schema(score)


def test_importance_weighting_is_per_criterion_not_per_atom():
    r = rubric([criterion("T1", importance="core"),
                criterion("T2", importance="supporting", expected=1),
                criterion("T3", importance="minor", expected=2)])
    score = score_task(r, "fr", output(judgment("T1", ["met", "met", "unmet"]),
                                       judgment("T2", ["met"]), judgment("T3", ["unmet", "unmet"])))
    assert score["dimension_scores"]["translation_quality"] == 0.666667
    assert [item["weight"] for item in score["criterion_scores"]] == [3, 2, 1]
    check_score_schema(score)


@pytest.mark.parametrize("results", [["met", "met", "met"], ["met", "met", "unmet"]])
def test_any_forbidden_match_vetoes_only_its_criterion(results):
    r = rubric([criterion("T1", forbidden=2), criterion("T2", importance="minor")])
    first = judgment("T1", results)
    first["forbidden_checks"].append({"index": 1, "matched": True})
    score = score_task(r, "fr", output(first, judgment("T2", ["met"] * 3)))
    detail = score["criterion_scores"][0]
    assert detail["expected_met"] == results.count("met")
    assert detail["expected_ratio"] == results.count("met") / 3
    assert detail["forbidden_triggered"] is True
    assert detail["criterion_score"] == 0
    assert score["dimension_scores"]["translation_quality"] == 0.25
    check_score_schema(score)


def test_no_forbidden_atoms():
    row = judgment("T1", ["met"] * 3)
    row["forbidden_checks"] = []
    score = score_task(rubric([criterion("T1", forbidden=0)]), "fr", output(row))
    assert score["criterion_scores"][0]["forbidden_triggered"] is False
    assert score["dimension_scores"]["translation_quality"] == 1
    check_score_schema(score)


def test_vtk_dimensions_are_independent_and_rubric_weights_do_not_override_protocol():
    r = rubric([criterion("V1", "image_understanding"), criterion("T1"),
                criterion("K1", "knowledge_entailment", "minor")])
    judged = output(judgment("V1", ["met"] * 3), judgment("T1", ["met", "met", "unmet"]),
                    judgment("K1", ["unmet"] * 3))
    original = score_task(r, "fr", judged)
    r["scoring"] = {"dimension_weights": {"translation_quality": 999},
                    "importance_weights": {"core": 0, "supporting": 100, "minor": -1}}
    assert score_task(r, "fr", judged) == original
    assert original["dimension_scores"] == {"image_understanding": 1.0,
                                            "translation_quality": 0.666667, "knowledge_entailment": 0.0}
    judged["judgments"][0] = judgment("V1", ["unmet"] * 3)
    changed = score_task(r, "fr", judged)
    for dimension in ("translation_quality", "knowledge_entailment"):
        assert changed["dimension_scores"][dimension] == original["dimension_scores"][dimension]
    check_score_schema(original)
    check_score_schema(changed)


def test_text_profile_excludes_visual_criteria():
    r = rubric([criterion("V1", "image_understanding"), criterion("T1"),
                criterion("K1", "knowledge_entailment")])
    score = score_task(r, "fr", output(judgment("T1", ["met"] * 3),
                                       judgment("K1", ["unmet"] * 3)), profile="text")
    assert score["dimension_scores"] == {"image_understanding": None,
                                         "translation_quality": 1.0, "knowledge_entailment": 0.0}
    assert score["dimension_scores_100"]["image_understanding"] is None
    assert [item["criterion_id"] for item in score["criterion_scores"]] == ["T1", "K1"]
    check_score_schema(score)


@pytest.mark.parametrize("rubric_set", ["basic_rubric", "rubric"])
def test_score_preserves_rubric_set(rubric_set):
    score = score_task(rubric([criterion("T1")]), "fr", output(judgment("T1", ["met"] * 3)),
                       rubric_set=rubric_set)
    assert score["rubric_set"] == rubric_set
    check_score_schema(score)


@pytest.mark.parametrize("visual_only", [False, True])
@pytest.mark.parametrize("judge_output", [None, output()])
def test_zero_criteria_is_unscorable(visual_only, judge_output):
    r = rubric([criterion("V1", "image_understanding")] if visual_only else [])
    score = score_task(r, "fr", judge_output, profile="text")
    assert score["status"] == "unscorable"
    assert score["dimension_scores"] == dict.fromkeys(DIMENSIONS)
    assert score["dimension_scores_100"] == dict.fromkeys(DIMENSIONS)
    assert score["criterion_scores"] == []
    check_score_schema(score)


def test_language_override_criteria_are_scored():
    r = rubric([criterion("T1")])
    r["language_overrides"] = {"fr": {"criteria": [criterion("K1", "knowledge_entailment")]}}
    score = score_task(r, "fr", output(judgment("T1", ["met"] * 3), judgment("K1", ["met"] * 3)))
    assert score["dimension_scores"]["knowledge_entailment"] == 1
    check_score_schema(score)


@pytest.mark.parametrize("value", ["present", "partial", "absent", None, [], True])
def test_scoring_rejects_old_and_invalid_expected_results(value):
    with pytest.raises(DataError):
        score_task(rubric([criterion("T1")]), "fr", output(judgment("T1", ["met", value, "unmet"])))


@pytest.mark.parametrize("change", ["no_envelope", "old_schema", "wrong_image", "wrong_language",
                                    "missing_check", "duplicate_index", "boolean_index", "extra_check_field",
                                    "missing_forbidden", "non_boolean_forbidden", "bad_evidence",
                                    "empty_rationale", "duplicate_criterion", "unknown_criterion", "no_output"])
def test_scoring_validates_complete_judge_output(change):
    row = judgment("T1", ["met"] * 3)
    judged = output(row)
    if change == "no_envelope":
        judged = {"judgments": [row]}
    elif change == "old_schema":
        judged["schema_version"] = "vista_judge_output_v2"
    elif change == "wrong_image":
        judged["image_name"] = "other.jpg"
    elif change == "wrong_language":
        judged["target_lang"] = "de"
    elif change == "missing_check":
        row["expected_checks"].pop()
    elif change == "duplicate_index":
        row["expected_checks"][2]["index"] = 0
    elif change == "boolean_index":
        row["expected_checks"][0]["index"] = False
    elif change == "extra_check_field":
        row["expected_checks"][0]["score"] = 1
    elif change == "missing_forbidden":
        row["forbidden_checks"] = []
    elif change == "non_boolean_forbidden":
        row["forbidden_checks"][0]["matched"] = 1
    elif change == "bad_evidence":
        row["evidence"] = [1]
    elif change == "empty_rationale":
        row["rationale"] = " "
    elif change == "duplicate_criterion":
        judged["judgments"].append(deepcopy(row))
    elif change == "unknown_criterion":
        row["criterion_id"] = "T2"
    else:
        judged = None
    with pytest.raises(DataError):
        score_task(rubric([criterion("T1")]), "fr", judged)


@pytest.fixture
def valid_score():
    return {"task_id": "1:fr", "source_lang": "English", "level1": "D01",
            "level1_name_en": "Transportation",
            **score_task(rubric([criterion("T1")]), "fr", output(judgment("T1", ["met"] * 3)))}


@pytest.mark.parametrize("field", ["overall_score", "overall_score_100", "active_dimension_weights"])
def test_schema_rejects_overall_fields(valid_score, field):
    valid_score[field] = None
    with pytest.raises(ValidationError):
        VALIDATOR.validate(valid_score)


@pytest.mark.parametrize("field", ["rubric_set", "scoring_protocol", "dimension_scores", "dimension_scores_100", "criterion_scores"])
def test_schema_requires_protocol_and_complete_scores(valid_score, field):
    valid_score.pop(field)
    with pytest.raises(ValidationError):
        VALIDATOR.validate(valid_score)


@pytest.mark.parametrize("field,value", [("weight", 2), ("expected_met", -1), ("expected_met", True),
                                        ("expected_total", 0), ("expected_ratio", 1.1),
                                        ("criterion_score", -0.1), ("forbidden_triggered", 1),
                                        ("rationale", " "), ("importance_weight", 3)])
def test_schema_rejects_invalid_criterion_details(valid_score, field, value):
    valid_score["criterion_scores"][0][field] = value
    with pytest.raises(ValidationError):
        VALIDATOR.validate(valid_score)


@pytest.mark.parametrize("change", ["version", "protocol", "rubric_set", "missing_dimension", "extra_dimension",
                                    "out_of_range", "empty_ok", "failed_with_scores", "forbidden_not_zero",
                                    "zero_met_nonzero_ratio", "text_with_visual_score"])
def test_schema_enforces_score_constraints(valid_score, change):
    Draft202012Validator.check_schema(SCHEMA)
    if change == "version":
        valid_score["schema_version"] = "vista_score_v1"
    elif change == "protocol":
        valid_score["scoring_protocol"] = "binary_expected_independent_vtk_v1"
    elif change == "rubric_set":
        valid_score["rubric_set"] = "unknown"
    elif change == "missing_dimension":
        valid_score["dimension_scores"].pop("image_understanding")
    elif change == "extra_dimension":
        valid_score["dimension_scores"]["overall"] = 1
    elif change == "out_of_range":
        valid_score["dimension_scores_100"]["translation_quality"] = 101
    elif change == "empty_ok":
        valid_score["criterion_scores"] = []
    elif change == "failed_with_scores":
        valid_score["status"] = "judge_failed"
    elif change == "forbidden_not_zero":
        valid_score["criterion_scores"][0]["forbidden_triggered"] = True
    elif change == "zero_met_nonzero_ratio":
        valid_score["criterion_scores"][0]["expected_met"] = 0
    else:
        valid_score["profile"] = "text"
        valid_score["dimension_scores"]["image_understanding"] = 1
    with pytest.raises(ValidationError):
        VALIDATOR.validate(valid_score)


@pytest.fixture
def scoring_cli(tmp_path):
    (tmp_path / "data").mkdir()
    rubric_path = tmp_path / "data/rubric.json"
    rubric_path.write_text(json.dumps(rubric([criterion("T1")])))
    assets = {key: {"path": "data/rubric.json", "sha256": "unused"}
              for key in ("image", "ocr", "verified_translation", "simple_rubric", "rubric")}
    manifest = {"name": "a.jpg", "seq": 1, "language": "English", "level1": "D01",
                "level1_name_en": "Domain", "level2": "D01.01", "level2_name_en": "Group", "assets": assets}
    (tmp_path / "data/manifest.jsonl").write_text(json.dumps(manifest) + "\n")
    record = {"schema_version": "vista_judge_record", "task_id": "1:fr", "image_name": "a.jpg",
              "source_lang": "English", "target_lang": "fr", "profile": "text", "rubric_set": "rubric",
              "status": "ok", "attempts": 1, "judge_output": output(judgment("T1", ["met", "met", "unmet"]))}
    source = tmp_path / "judgments.jsonl"
    destination = tmp_path / "scores.jsonl"
    command = [sys.executable, "-B", str(SCRIPT / "scoring.py"), "--root", str(tmp_path),
               "--judge-records", str(source), "--rubric-set", "rubric", "--profile", "text",
               "--output", str(destination)]
    return record, rubric_path, source, destination, command


@pytest.mark.parametrize("status", ["ok", "zero_criteria", "failed"])
def test_cli_outputs_valid_uniform_scores(scoring_cli, status):
    record, rubric_path, source, destination, command = scoring_cli
    record["status"] = status
    if status == "zero_criteria":
        rubric_path.write_text(json.dumps(rubric([])))
        record.update(attempts=0, judge_output=None)
    elif status == "failed":
        record.update(attempts=3, error="API failure", judge_output=None)
    source.write_text(json.dumps(record) + "\n")
    subprocess.run(command, text=True, capture_output=True, check=True)
    score = json.loads(destination.read_text())
    check_score_schema(score)
    assert score["rubric_set"] == "rubric"
    assert score["status"] == {"ok": "ok", "zero_criteria": "unscorable", "failed": "judge_failed"}[status]
    if status == "ok":
        assert score["dimension_scores"]["translation_quality"] == 0.666667
    else:
        assert score["dimension_scores"] == dict.fromkeys(DIMENSIONS)
        assert score["dimension_scores_100"] == dict.fromkeys(DIMENSIONS)
        assert score["criterion_scores"] == []


@pytest.mark.parametrize("change", ["old_record", "old_zero", "old_output", "old_result", "wrong_profile",
                                    "wrong_rubric", "wrong_task", "wrong_source", "wrong_image",
                                    "invalid_status", "false_zero", "failed_with_output", "missing_error",
                                    "bad_attempts", "duplicate_record"])
def test_cli_rejects_invalid_records_without_overwriting_scores(scoring_cli, change):
    record, rubric_path, source, destination, command = scoring_cli
    if change in ("old_record", "old_zero"):
        record["schema_version"] = "vista_judge_record_v1"
        if change == "old_zero":
            rubric_path.write_text(json.dumps(rubric([])))
            record.update(status="zero_criteria", attempts=0, judge_output=None)
    elif change == "old_output":
        record["judge_output"]["schema_version"] = "vista_judge_output_v2"
    elif change == "old_result":
        record["judge_output"]["judgments"][0]["expected_checks"][0]["result"] = "partial"
    elif change == "wrong_profile":
        record["profile"] = "image"
    elif change == "wrong_rubric":
        record["rubric_set"] = "basic_rubric"
    elif change == "wrong_task":
        record["task_id"] = "2:fr"
    elif change == "wrong_source":
        record["source_lang"] = "Other"
    elif change == "wrong_image":
        record["image_name"] = "other.jpg"
    elif change == "invalid_status":
        record["status"] = "unknown"
    elif change == "false_zero":
        record.update(status="zero_criteria", attempts=0, judge_output=None)
    elif change == "failed_with_output":
        record.update(status="failed", error="failure")
    elif change == "missing_error":
        record.update(status="failed", judge_output=None)
    elif change == "bad_attempts":
        record["attempts"] = True
    source.write_text((json.dumps(record) + "\n") * (2 if change == "duplicate_record" else 1))
    destination.write_text("previous scores\n")
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode != 0
    assert "DataError" in result.stderr
    assert destination.read_text() == "previous scores\n"
