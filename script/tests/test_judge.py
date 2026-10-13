from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

from jsonschema import Draft202012Validator, ValidationError
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from dataset import DataError
import judge
from judge import make_packet, validate_judgment, validate_judge_record

SCRIPT = Path(__file__).parents[1]
SCHEMA = json.loads((SCRIPT / "schemas/judge_output.schema.json").read_text())
VALIDATOR = Draft202012Validator(SCHEMA)
RECORD_VALIDATOR = VALIDATOR.evolve(schema={"$ref": "#/$defs/judge_record"})


@pytest.fixture
def criteria():
    return [{"id": "T1", "dimension": "translation_quality", "importance": "core",
             "expected": ["x", "y", "z"], "forbidden": ["bad"]}]


@pytest.fixture
def output():
    return {"schema_version": "vista_judge_output", "image_name": "a.jpg", "target_lang": "fr",
            "judgments": [{"criterion_id": "T1",
                           "expected_checks": [{"index": 0, "result": "met"},
                                               {"index": 1, "result": "met"},
                                               {"index": 2, "result": "unmet"}],
                           "forbidden_checks": [{"index": 0, "matched": False}],
                           "evidence": ["x", "y"], "rationale": "Third atom is only partially satisfied."}]}


@pytest.fixture
def record(output):
    return {"schema_version": "vista_judge_record", "task_id": "1:fr", "image_name": "a.jpg",
            "source_lang": "English", "target_lang": "fr", "profile": "text",
            "rubric_set": "basic_rubric", "status": "ok", "attempts": 1, "judge_output": output}


def test_keyed_checklist_validation(criteria, output):
    Draft202012Validator.check_schema(SCHEMA)
    assert validate_judgment(output, "a.jpg", "fr", criteria) == output
    VALIDATOR.validate(output)
    output["judgments"][0]["expected_checks"].reverse()
    assert validate_judgment(output, "a.jpg", "fr", criteria) == output


@pytest.mark.parametrize("result", ["present", "partial", "absent", "MET", True, None, [], {}])
def test_old_and_invalid_expected_results_rejected(criteria, output, result):
    output["judgments"][0]["expected_checks"][0]["result"] = result
    with pytest.raises(DataError):
        validate_judgment(output, "a.jpg", "fr", criteria)
    with pytest.raises(ValidationError):
        VALIDATOR.validate(output)


@pytest.mark.parametrize("field,value", [
    ("expected_checks", None), ("expected_checks", {}), ("expected_checks", [None]),
    ("expected_checks", []), ("forbidden_checks", None), ("forbidden_checks", [False]),
    ("evidence", "quote"), ("evidence", [1]), ("evidence", [{}]),
    ("rationale", ""), ("rationale", " \n "), ("rationale", None),
    ("criterion_id", []), ("criterion_id", "unknown"),
])
def test_invalid_judgment_fields_fail_closed(criteria, output, field, value):
    output["judgments"][0][field] = value
    with pytest.raises(DataError):
        validate_judgment(output, "a.jpg", "fr", criteria)


@pytest.mark.parametrize("field", ["expected_checks", "forbidden_checks"])
@pytest.mark.parametrize("index", [True, False, 0.0, -1, 99, "0", [], None])
def test_invalid_checklist_indices_fail_closed(criteria, output, field, index):
    output["judgments"][0][field][0]["index"] = index
    with pytest.raises(DataError):
        validate_judgment(output, "a.jpg", "fr", criteria)


@pytest.mark.parametrize("matched", [1, 0, "true", "false", None, []])
def test_forbidden_requires_boolean(criteria, output, matched):
    output["judgments"][0]["forbidden_checks"][0]["matched"] = matched
    with pytest.raises(DataError):
        validate_judgment(output, "a.jpg", "fr", criteria)
    with pytest.raises(ValidationError):
        VALIDATOR.validate(output)


@pytest.mark.parametrize("change", ["missing_check", "duplicate_index", "extra_check_field",
                                    "extra_row_field", "missing_row_field", "duplicate_criterion",
                                    "missing_criterion", "extra_root_field", "old_schema", "wrong_identity"])
def test_checklist_completeness_and_output_identity(criteria, output, change):
    row = output["judgments"][0]
    if change == "missing_check":
        row["expected_checks"].pop()
    elif change == "duplicate_index":
        row["expected_checks"][2]["index"] = 0
    elif change == "extra_check_field":
        row["expected_checks"][0]["score"] = 1
    elif change == "extra_row_field":
        row["score"] = 1
    elif change == "missing_row_field":
        row.pop("evidence")
    elif change == "duplicate_criterion":
        output["judgments"].append(deepcopy(row))
    elif change == "missing_criterion":
        output["judgments"] = []
    elif change == "extra_root_field":
        output["overall_score"] = 1
    elif change == "old_schema":
        output["schema_version"] = "vista_judge_output_v2"
    else:
        output["target_lang"] = "de"
    with pytest.raises(DataError):
        validate_judgment(output, "a.jpg", "fr", criteria)


@pytest.mark.parametrize("value", [None, [], "output", 1])
def test_non_object_output_fails_closed(criteria, value):
    with pytest.raises(DataError):
        validate_judgment(value, "a.jpg", "fr", criteria)


@pytest.mark.parametrize("status", ["ok", "zero_criteria", "failed"])
def test_record_validation_and_schema(criteria, record, status):
    record["status"] = status
    if status != "ok":
        record["judge_output"] = None
    if status == "zero_criteria":
        criteria = []
        record["attempts"] = 0
    elif status == "failed":
        record["error"] = "API unavailable"
    assert validate_judge_record(record, criteria) == record
    RECORD_VALIDATOR.validate(record)


@pytest.mark.parametrize("field,value", [
    ("schema_version", "vista_judge_record_v1"), ("attempts", True), ("attempts", 0),
    ("attempts", 4), ("attempts", "1"), ("status", "unknown"), ("task_id", []),
    ("profile", []), ("rubric_set", {}), ("source_lang", " "), ("judge_output", None),
])
def test_invalid_records_rejected(criteria, record, field, value):
    record[field] = value
    with pytest.raises(DataError):
        validate_judge_record(record, criteria)
    with pytest.raises(ValidationError):
        RECORD_VALIDATOR.validate(record)


@pytest.fixture
def release(tmp_path, criteria):
    (tmp_path / "script/prompts").mkdir(parents=True)
    (tmp_path / "script/prompts/judge.txt").write_text((SCRIPT / "prompts/judge.txt").read_text())
    (tmp_path / "data").mkdir()
    assets = {}
    contents = {
        "image": b"image", "ocr": json.dumps({"text": "Hello"}).encode(),
        "verified_translation": json.dumps({"target_langs": ["fr"],
                                               "blocks": [{"id": 1, "text": "Hello"}]}).encode(),
        "simple_rubric": json.dumps({"image_name": "a.jpg", "target_langs": ["fr"],
                                      "language_overrides": {}, "criteria": criteria}).encode(),
        "rubric": json.dumps({"image_name": "a.jpg", "target_langs": ["fr"],
                               "language_overrides": {}, "criteria": []}).encode(),
    }
    for kind, content in contents.items():
        path = tmp_path / "data" / f"{kind}.json"
        path.write_bytes(content)
        assets[kind] = {"path": f"data/{kind}.json", "sha256": "unused"}
    manifest = {"name": "a.jpg", "seq": 1, "language": "English", "level1": "D01",
                "level1_name_en": "Domain", "level2": "D01.01", "level2_name_en": "Group",
                "assets": assets}
    (tmp_path / "data/manifest.jsonl").write_text(json.dumps(manifest) + "\n")
    candidate = {"schema_version": "vista_self_segmented_candidate_v1", "task_id": "1:fr",
                 "image_name": "a.jpg", "source_lang": "English", "target_lang": "fr",
                 "candidate_blocks": [{"id": 1, "source_text": "Hello", "text": "Bonjour"}]}
    (tmp_path / "candidates.jsonl").write_text(json.dumps(candidate) + "\n")
    return tmp_path


@pytest.mark.parametrize("case,skipped", [("current", 1), ("zero_current", 1), ("failed", 0)])
def test_resume_only_reuses_current_valid_records_without_writing(release, record, case, skipped):
    rubric_set = "rubric" if case == "zero_current" else "basic_rubric"
    record["rubric_set"] = rubric_set
    if case == "zero_current":
        record.update(status="zero_criteria", attempts=0, judge_output=None)
    elif case == "failed":
        record.update(status="failed", attempts=3, error="retryable failure", judge_output=None)
    output_path = release / "judgments.jsonl"
    original = json.dumps(record) + "\n"
    output_path.write_text(original)
    result = subprocess.run([
        sys.executable, "-B", str(SCRIPT / "judge.py"), "--root", str(release),
        "--config", str(release / "missing.toml"), "--candidates", str(release / "candidates.jsonl"),
        "--output", str(output_path), "--profile", "text", "--rubric-set", rubric_set,
        "--resume", "--dry-run"], text=True, capture_output=True, check=True)
    stats = json.loads(result.stdout)
    assert stats["tasks"] == 1 - skipped
    assert stats["skipped"] == skipped
    assert stats["api_requests"] == 0
    assert output_path.read_text() == original


@pytest.mark.parametrize("case", [
    "old_record", "zero_old", "old_output", "present", "partial", "absent", "malformed",
    "invalid_attempts", "wrong_identity", "invalid_zero", "missing_version", "duplicate",
])
def test_resume_rejects_old_or_invalid_records_without_writing(release, record, case):
    rubric_set = "rubric" if case == "zero_old" else "basic_rubric"
    record["rubric_set"] = rubric_set
    if case in ("zero_old", "invalid_zero"):
        record.update(status="zero_criteria", attempts=0, judge_output=None)
    if case in ("old_record", "zero_old"):
        record["schema_version"] = "vista_judge_record_v1"
    elif case == "old_output":
        record["judge_output"]["schema_version"] = "vista_judge_output_v2"
    elif case in ("present", "partial", "absent"):
        record["judge_output"]["judgments"][0]["expected_checks"][0]["result"] = case
    elif case == "invalid_attempts":
        record["attempts"] = True
    elif case == "wrong_identity":
        record["source_lang"] = "Other"
    elif case == "missing_version":
        record.pop("schema_version")
    output_path = release / "judgments.jsonl"
    original = "{invalid\n[]\n" if case == "malformed" else json.dumps(record) + "\n"
    if case == "duplicate":
        original *= 2
    output_path.write_text(original)
    result = subprocess.run([
        sys.executable, "-B", str(SCRIPT / "judge.py"), "--root", str(release),
        "--config", str(release / "missing.toml"), "--candidates", str(release / "candidates.jsonl"),
        "--output", str(output_path), "--profile", "text", "--rubric-set", rubric_set,
        "--resume", "--dry-run"], text=True, capture_output=True)
    assert result.returncode != 0
    assert "DataError" in result.stderr
    assert output_path.read_text() == original


def test_prompt_and_packet_remain_text_only(criteria):
    candidate = {"task_id": "1:fr", "target_lang": "fr",
                 "candidate_blocks": [{"id": 1, "source_text": "Hello", "text": "Bonjour"}]}
    packet = make_packet({"name": "a.jpg", "language": "English"}, candidate,
                         {"tolerances": ["punctuation"]}, criteria, [{"id": 1, "text": "Hello"}])
    assert packet == {"task_id": "1:fr", "image_name": "a.jpg", "source_lang": "English",
                      "target_lang": "fr", "source_blocks": [{"id": 1, "text": "Hello"}],
                      "candidate_blocks": candidate["candidate_blocks"], "criteria": criteria,
                      "tolerances": ["punctuation"]}
    prompt = (SCRIPT / "prompts/judge.txt").read_text().format(packet=json.dumps(packet))
    assert "vista_judge_output_v" not in prompt
    assert "met or unmet" in prompt
    assert "Partial satisfaction" in prompt
    assert "binary_expected_independent_vtk" in prompt
    assert json.loads(prompt.split("Packet:\n", 1)[1]) == packet


@pytest.mark.parametrize("repair", [False, True])
def test_judge_emits_current_record_and_repairs_with_protocol(release, output, criteria, monkeypatch, repair):
    config = release / "config.toml"
    config.write_text('[judge_api]\nbase_url="https://example.invalid"\nendpoint="chat/completions"\n'
                      'model="mock"\nkey_env="VISTA_JUDGE_TEST_KEY"\n')
    monkeypatch.setenv("VISTA_JUDGE_TEST_KEY", "test-only")
    prompts = []

    def generate(self, prompt, *, temperature):
        prompts.append(prompt)
        if repair and len(prompts) == 1:
            legacy = deepcopy(output)
            legacy["judgments"][0]["expected_checks"][0]["result"] = "partial"
            return json.dumps(legacy)
        return json.dumps(output)

    monkeypatch.setattr(judge.OpenAICompatibleClient, "generate", generate)
    output_path = release / "judgments.jsonl"
    monkeypatch.setattr(sys, "argv", ["judge.py", "--root", str(release), "--config", str(config),
                                    "--candidates", str(release / "candidates.jsonl"),
                                    "--output", str(output_path), "--profile", "text", "--workers", "1"])
    assert judge.main() == 0
    record = json.loads(output_path.read_text())
    assert record["schema_version"] == "vista_judge_record"
    assert record["status"] == "ok"
    validate_judge_record(record, criteria)
    RECORD_VALIDATOR.validate(record)
    assert len(prompts) == 1 + int(repair)
    assert all("met or unmet" in prompt and "Packet:\n" in prompt for prompt in prompts)
