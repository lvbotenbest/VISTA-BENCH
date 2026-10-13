import copy
import csv
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import subprocess
import sys

from jsonschema import Draft202012Validator
import pytest


SCRIPT = Path(__file__).parents[1] / "summarize.py"
SCHEMA = json.loads((SCRIPT.parent / "schemas/summary.schema.json").read_text(encoding="utf-8"))
T, I, K = "translation_quality", "image_understanding", "knowledge_entailment"
DIMENSIONS = (T, I, K)
RANKS = ("rank_translation", "rank_image", "rank_knowledge")
COUNTS = ("translation_directions", "image_directions", "knowledge_directions")
FIELDS = ["model", *RANKS, *DIMENSIONS, *COUNTS]
CSV_FILES = ("leaderboard.csv", "scores_by_direction.csv", "scores_by_direction_level1.csv",
             "scores_by_level1.csv", "coverage.csv")


def criterion(dimension, value, weight=1):
    ratio = Decimal(str(value))
    expected_met, expected_total = ratio.as_integer_ratio()
    importance = {3: "core", 2: "supporting", 1: "minor"}[weight]
    return {"dimension": dimension, "importance": importance, "weight": weight,
            "expected_met": expected_met, "expected_total": expected_total,
            "expected_ratio": float(ratio), "forbidden_triggered": False,
            "criterion_score": float(ratio), "rationale": "checked"}


def score(task_id="1:fr", criteria=(), *, source="English", target="fr", level1="D01",
          profile="image", rubric_set="basic_rubric", status="ok"):
    criteria = list(criteria)
    if status == "ok" and not criteria:
        criteria = [criterion(T, 1)]
    if status != "ok":
        criteria = []
    criteria = [{"criterion_id": f"C{index}", **item} for index, item in enumerate(criteria, 1)]
    dimensions = {}
    for dimension in DIMENSIONS:
        selected = [item for item in criteria if item["dimension"] == dimension]
        dimensions[dimension] = (sum(item["criterion_score"] * item["weight"] for item in selected)
                                 / sum(item["weight"] for item in selected) if selected else None)
    dimensions_100 = {dimension: (None if value is None else float(
        (Decimal(str(value)) * 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)))
                      for dimension, value in dimensions.items()}
    return {"schema_version": "vista_score", "scoring_protocol": "binary_expected_independent_vtk",
            "rubric_set": rubric_set, "task_id": task_id, "image_name": "a.jpg",
            "source_lang": source, "target_lang": target, "level1": level1,
            "level1_name_en": "Transportation" if level1 == "D01" else "Other",
            "profile": profile, "status": status, "dimension_scores": dimensions,
            "dimension_scores_100": dimensions_100, "criterion_scores": criteria}


def invoke(tmp_path, models, *, profile="image", rubric_set="basic_rubric", metadata=None):
    output = tmp_path / "summary"
    command = [sys.executable, str(SCRIPT), "--profile", profile, "--rubric-set", rubric_set,
               "--output-dir", str(output)]
    for index, (model, rows) in enumerate(models.items()):
        source = tmp_path / f"scores-{index}.jsonl"
        source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        command.extend(["--scores", f"{model}={source}"])
    if metadata is not None:
        path = tmp_path / "run.json"
        path.write_text(json.dumps(metadata), encoding="utf-8")
        command.extend(["--run-metadata", str(path)])
    return subprocess.run(command, text=True, capture_output=True), output


def summarize(tmp_path, models, **kwargs):
    result, output = invoke(tmp_path, models, **kwargs)
    assert result.returncode == 0, result.stderr
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    Draft202012Validator(SCHEMA).validate(summary)
    return output, summary


def csv_rows(output, name):
    with (output / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def reject(tmp_path, models, message, **kwargs):
    result, output = invoke(tmp_path, models, **kwargs)
    assert result.returncode != 0
    assert message in result.stderr
    assert not output.exists(), "invalid input must not produce partial summaries"


def test_independent_rankings_fixed_columns_and_numeric_json(tmp_path):
    models = {name: [score(criteria=[criterion(d, value) for d, value in zip(DIMENSIONS, values)])]
              for name, values in (("C", (.7, .5, .9)), ("B", (.8, .9, .2)), ("A", (.9, .2, .6)))}
    output, summary = summarize(tmp_path, models)
    rows = csv_rows(output, "leaderboard.csv")
    assert list(rows[0]) == FIELDS
    assert [row["model"] for row in rows] == ["A", "B", "C"]
    assert [tuple(row[key] for key in RANKS) for row in rows] == [
        ("1", "3", "2"), ("2", "1", "3"), ("3", "2", "1")]
    for csv_row, json_row in zip(rows, summary["leaderboard"]):
        assert list(json_row) == FIELDS
        for dimension, rank, count in zip(DIMENSIONS, RANKS, COUNTS):
            assert type(json_row[dimension]) in (int, float)
            assert csv_row[dimension] == f"{json_row[dimension]:.6f}"
            assert type(json_row[rank]) is int
            assert json_row[count] == 1
    assert summary["schema_version"] == SCHEMA["$id"] == "vista_public_summary"
    assert summary["scoring_protocol"] == "binary_expected_independent_vtk"
    assert set(summary["formula"]) == set(DIMENSIONS)
    assert summary["ranking"]["dimensions"] == dict(zip(DIMENSIONS, RANKS))
    assert summary["ranking"]["method"] == "competition"
    assert summary["ranking"]["public_decimals"] == 6
    assert set(path.name for path in output.iterdir()) == {*CSV_FILES, "summary.json"}
    for name in (*CSV_FILES, "summary.json"):
        text = (output / name).read_text(encoding="utf-8")
        assert "overall" not in text
        assert "dimension_weights" not in text
        assert "0.3/0.5/0.2" not in text
    for flag in ("selection_leakage", "analysis_status", "official_leaderboard"):
        assert flag not in summary


@pytest.mark.parametrize("dimension,rank_field", zip(DIMENSIONS, RANKS))
def test_six_decimal_ties_use_competition_ranks_and_keep_zero(tmp_path, dimension, rank_field):
    models = {name: [score(criteria=[criterion(dimension, value)] if value is not None else [],
                           status="ok" if value is not None else "unscorable")]
              for name, value in (("z-top", .900000004), ("zero", 0), ("null", None),
                                  ("third", .8), ("a-top", .900000003))}
    output, summary = summarize(tmp_path, models)
    rows = {row["model"]: row for row in summary["leaderboard"]}
    assert {name: row[rank_field] for name, row in rows.items()} == {
        "a-top": 1, "z-top": 1, "third": 3, "zero": 4, "null": None}
    assert rows["a-top"][dimension] == rows["z-top"][dimension] == 90.0
    csv_by_model = {row["model"]: row for row in csv_rows(output, "leaderboard.csv")}
    assert csv_by_model["null"][rank_field] == csv_by_model["null"][dimension] == ""
    assert csv_by_model["zero"][dimension] == "0.000000"
    expected = (["a-top", "z-top", "third", "zero", "null"] if dimension == T
                else sorted(models))
    assert [row["model"] for row in summary["leaderboard"]] == expected


def test_public_rounding_boundary_distinguishes_ranks(tmp_path):
    _, summary = summarize(tmp_path, {
        "low": [score(criteria=[criterion(T, .500000004)])],
        "high": [score(criteria=[criterion(T, .500000005)])],
    })
    assert [(row[T], row["rank_translation"]) for row in summary["leaderboard"]] == [
        (50.000001, 1), (50.0, 2)]


def test_text_null_and_zero_have_independent_direction_counts(tmp_path):
    output, summary = summarize(tmp_path, {
        "z-null": [score(profile="text", status="judge_failed")],
        "a-empty": [],
        "zero": [score("t", [criterion(T, 0)], profile="text"),
                 score("k", [criterion(K, 0)], target="de", profile="text"),
                 score("failed", [criterion(T, 1)], target="es", status="judge_failed", profile="text"),
                 score("unscorable", target="it", status="unscorable", profile="text")],
    }, profile="text")
    rows = summary["leaderboard"]
    assert [row["model"] for row in rows] == ["zero", "a-empty", "z-null"]
    assert [rows[0][key] for key in DIMENSIONS] == [0.0, None, 0.0]
    assert [rows[0][key] for key in RANKS] == [1, None, 1]
    assert [rows[0][key] for key in COUNTS] == [1, 0, 1]
    for row in rows[1:]:
        assert all(row[key] is None for key in (*DIMENSIONS, *RANKS))
        assert all(row[key] == 0 for key in COUNTS)
    for name in CSV_FILES[:-1]:
        for row in csv_rows(output, name):
            assert row[I] == ""
            if "image_directions" in row:
                assert row["image_directions"] == "0"
    directions = csv_rows(output, "scores_by_direction.csv")
    assert len(directions) == 2
    assert {row["target_lang"]: (row[T], row[K]) for row in directions} == {
        "de": ("", "0.000000"), "fr": ("0.000000", "")}
    coverage = next(row for row in summary["coverage"] if row["model"] == "zero")
    assert coverage == {"model": "zero", "records": 4, "ok": 2, "unscorable": 1, "judge_failed": 1}


def test_criterion_pooling_direction_macro_and_level1_macro(tmp_path):
    records = [
        score("fr-a", [criterion(T, 1), criterion(T, 0), criterion(K, 0)]),
        score("fr-b", [criterion(T, 1, 3)]),
        score("fr-other", [criterion(T, 0, 2), criterion(K, 1, 3)], level1="D02"),
        score("de", [criterion(T, 0), criterion(I, 1, 2)], target="de"),
        score("ja-fr", [criterion(T, 1, 2)], source="Japanese"),
    ]
    # Stored task dimension means are not inputs to criterion pooling.
    for row in records:
        row["dimension_scores"] = dict.fromkeys(DIMENSIONS, .123)
        row["dimension_scores_100"] = dict.fromkeys(DIMENSIONS, 12.3)
    records[1]["level1_name_en"] = "Alternative display name"
    output, summary = summarize(tmp_path, {"model": records})
    directions = {(row["source_lang"], row["target_lang"]): row
                  for row in csv_rows(output, "scores_by_direction.csv")}
    assert len(directions) == 3
    assert directions["English", "fr"][T] == "57.142857"
    assert directions["English", "fr"][K] == "75.000000"
    assert directions["English", "fr"][I] == ""
    assert directions["English", "de"][T] == "0.000000"
    assert directions["Japanese", "fr"][T] == "100.000000"
    result = summary["leaderboard"][0]
    assert [result[key] for key in DIMENSIONS] == [52.380952, 100.0, 75.0]
    assert [result[key] for key in COUNTS] == [3, 1, 1]
    cells = {(row["source_lang"], row["target_lang"], row["level1"]): row
             for row in csv_rows(output, "scores_by_direction_level1.csv")}
    assert len(cells) == 4
    assert cells["English", "fr", "D01"][T] == "80.000000"
    assert cells["English", "fr", "D02"][T] == "0.000000"
    domains = {row["level1"]: row for row in csv_rows(output, "scores_by_level1.csv")}
    assert [domains["D01"][key] for key in DIMENSIONS] == ["60.000000", "100.000000", "0.000000"]
    assert [domains["D01"][key] for key in COUNTS] == ["3", "1", "1"]
    assert [domains["D02"][key] for key in DIMENSIONS] == ["0.000000", "", "100.000000"]
    assert [domains["D02"][key] for key in COUNTS] == ["1", "0", "1"]


def test_macro_uses_unrounded_direction_scores(tmp_path):
    output, summary = summarize(tmp_path, {"model": [
        score("fr", [criterion(T, .0000000049)]),
        score("de", [criterion(T, .0000000050)], target="de"),
    ]})
    assert sorted(row[T] for row in csv_rows(output, "scores_by_direction.csv")) == ["0.000000", "0.000001"]
    assert summary["leaderboard"][0][T] == 0.0
    assert summary["leaderboard"][0]["translation_directions"] == 2
    assert summary["leaderboard"][0]["rank_translation"] == 1
    assert csv_rows(output, "scores_by_level1.csv")[0][T] == "0.000000"


def test_empty_scores_emit_headers_and_null_leaderboard(tmp_path):
    output, summary = summarize(tmp_path, {"model": []})
    for name in CSV_FILES[1:-1]:
        assert csv_rows(output, name) == []
        assert (output / name).read_text(encoding="utf-8").startswith("model,")
    row = summary["leaderboard"][0]
    assert all(row[key] is None for key in (*DIMENSIONS, *RANKS))
    assert all(row[key] == 0 for key in COUNTS)


def test_rubric_keeps_exploratory_unofficial_flags_and_run_coverage(tmp_path):
    metadata = {"schema_version": "vista_benchmark_run",
                "scoring_protocol": "binary_expected_independent_vtk",
                "parameters": {"track": "text", "rubric_set": "rubric",
                               "scoring_protocol": "binary_expected_independent_vtk"},
                "counts": {"selected": 3, "candidate_succeeded": 2, "candidate_failed": 1,
                           "judge_ok": 1, "judge_zero_criteria": 0, "judge_failed": 999}}
    output, summary = summarize(tmp_path, {"model": [
        score(profile="text", rubric_set="rubric", status="judge_failed")
    ]}, profile="text", rubric_set="rubric", metadata=metadata)
    assert summary["selection_leakage"] is True
    assert summary["analysis_status"] == "exploratory"
    assert summary["official_leaderboard"] is False
    coverage = summary["coverage"][0]
    assert coverage == {"model": "model", "records": 1, "ok": 0, "unscorable": 0,
                        "judge_failed": 1, "selected": 3, "candidate_succeeded": 2,
                        "candidate_failed": 1, "judge_ok": 1, "judge_zero_criteria": 0}
    assert csv_rows(output, "coverage.csv")[0] == {key: str(value) for key, value in coverage.items()}


def test_run_metadata_rejects_multiple_models(tmp_path):
    reject(tmp_path, {"a": [], "b": []}, "single model", metadata={})


@pytest.mark.parametrize("location,field,value", [
    ("parameters", "track", "text"),
    ("parameters", "rubric_set", "rubric"),
    ("parameters", "scoring_protocol", "old"),
    ("root", "scoring_protocol", "old"),
    ("root", "schema_version", "vista_benchmark_run_v1"),
])
def test_run_metadata_rejects_mismatch(tmp_path, location, field, value):
    metadata = {"schema_version": "vista_benchmark_run",
                "scoring_protocol": "binary_expected_independent_vtk",
                "parameters": {"track": "image", "rubric_set": "basic_rubric",
                               "scoring_protocol": "binary_expected_independent_vtk"},
                "counts": {}}
    if location == "parameters":
        metadata["parameters"][field] = value
    else:
        metadata[field] = value
    reject(tmp_path, {"model": []}, f"run metadata {field} mismatch", metadata=metadata)


@pytest.mark.parametrize("value", [-1, 1.5, True, "1"])
def test_run_metadata_rejects_invalid_counts(tmp_path, value):
    metadata = {"schema_version": "vista_benchmark_run",
                "scoring_protocol": "binary_expected_independent_vtk",
                "parameters": {"track": "image", "rubric_set": "basic_rubric",
                               "scoring_protocol": "binary_expected_independent_vtk"},
                "counts": {"selected": value}}
    reject(tmp_path, {"model": []}, "nonnegative integers", metadata=metadata)


@pytest.mark.parametrize("field,value", [("schema_version", "vista_score_v1"), ("profile", "text"),
                                        ("rubric_set", "rubric"), ("scoring_protocol", "old")])
@pytest.mark.parametrize("across_models", [False, True])
def test_rejects_mixed_contracts(tmp_path, field, value, across_models):
    valid, invalid = score("valid"), score("invalid")
    invalid[field] = value
    models = {"a": [valid], "b": [invalid]} if across_models else {"model": [valid, invalid]}
    reject(tmp_path, models, f"{field} mismatch")


@pytest.mark.parametrize("field", ["schema_version", "scoring_protocol", "rubric_set", "profile", "status",
                                   "task_id", "dimension_scores", "dimension_scores_100", "criterion_scores"])
def test_rejects_missing_new_contract_fields(tmp_path, field):
    row = score()
    del row[field]
    reject(tmp_path, {"model": [row]}, field)


@pytest.mark.parametrize("field", ["overall", "overall_score", "overall_score_100",
                                   "dimension_weights", "active_dimension_weights"])
def test_rejects_legacy_fields_even_with_new_protocol(tmp_path, field):
    row = score()
    row[field] = None
    reject(tmp_path, {"model": [row]}, "legacy")


@pytest.mark.parametrize("status", ["ok", "unscorable", "judge_failed"])
def test_rejects_duplicate_task_id_within_model(tmp_path, status):
    reject(tmp_path, {"model": [score(), score(status=status, target="de")]}, "duplicate task_id")


@pytest.mark.parametrize("field,value,message", [
    ("status", "failed", "status"), ("task_id", "", "task_id"),
    ("source_lang", None, "source_lang"), ("target_lang", [], "target_lang"),
    ("level1", None, "level1"), ("level1_name_en", 1, "level1_name_en"),
    ("dimension_scores", {}, "dimension_scores"), ("dimension_scores", [], "dimension_scores"),
    ("criterion_scores", None, "criterion_scores"), ("criterion_scores", [None], "criterion score fields"),
    ("criterion_scores", [{"dimension": "unknown"}], "criterion score fields"),
])
def test_rejects_malformed_score_fields(tmp_path, field, value, message):
    row = score()
    row[field] = value
    reject(tmp_path, {"model": [row]}, message)


@pytest.mark.parametrize("field,value", [
    ("weight", 0), ("weight", -1), ("weight", None),
    ("weight", True), ("weight", "3"), ("weight", float("inf")),
    ("criterion_score", None), ("criterion_score", True), ("criterion_score", "0.5"),
    ("criterion_score", -.1), ("criterion_score", 1.1), ("criterion_score", float("nan")),
    ("criterion_score", float("inf")),
])
def test_rejects_invalid_criterion_numbers(tmp_path, field, value):
    row = score(criteria=[criterion(T, .5)])
    row["criterion_scores"][0][field] = value
    reject(tmp_path, {"model": [row]}, field)


@pytest.mark.parametrize("value", [True, "0.5", -1, 2, float("nan"), float("inf")])
def test_rejects_invalid_dimension_numbers(tmp_path, value):
    row = score()
    row["dimension_scores"][T] = value
    reject(tmp_path, {"model": [row]}, "dimension_scores.translation_quality")


@pytest.mark.parametrize("in_criteria", [False, True])
def test_text_rejects_image_scores_and_criteria(tmp_path, in_criteria):
    row = score(profile="text")
    if in_criteria:
        row["criterion_scores"] = [{"criterion_id": "visual", **criterion(I, 1)}]
    else:
        row["dimension_scores"][I] = 0
        row["dimension_scores_100"][I] = 0
    reject(tmp_path, {"model": [row]}, "text profile", profile="text")


@pytest.mark.parametrize("arguments,message", [
    (["invalid"], "MODEL=PATH"), (["=path"], "nonempty"), (["model="], "nonempty"),
    (["model=path", "model=other"], "duplicate model label"),
])
def test_rejects_invalid_score_arguments(tmp_path, arguments, message):
    command = [sys.executable, str(SCRIPT), "--profile", "image", "--rubric-set", "basic_rubric",
               "--output-dir", str(tmp_path / "summary")]
    for argument in arguments:
        command.extend(["--scores", argument])
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode != 0
    assert message in result.stderr


def test_summary_schema_rejects_old_versions_overall_and_wrong_json_types(tmp_path):
    Draft202012Validator.check_schema(SCHEMA)
    _, summary = summarize(tmp_path, {"model": [score(criteria=[criterion(T, 0)])]})
    validator = Draft202012Validator(SCHEMA)
    for key, value in (("overall", 0), (T, "0.000000"), ("rank_translation", "1"),
                       ("rank_translation", 1.5), ("rank_translation", 0), ("image_directions", None)):
        bad = copy.deepcopy(summary)
        bad["leaderboard"][0][key] = value
        assert not validator.is_valid(bad), (key, value)
    for key, value in (("schema_version", "vista_public_summary_v1"), ("scoring_protocol", "old"),
                       ("overall", None)):
        bad = {**summary, key: value}
        assert not validator.is_valid(bad), (key, value)
