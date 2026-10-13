import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))
import run_benchmark
from run_benchmark import (FILES, RUN_SCHEMA_VERSION, SCORING_PROTOCOL, RunConfigError,
                           build_parser, merge_resume, merge_resume_parameters, normalize_api_url,
                           normalize_rubric, validate_resume_metadata)


def test_api_url_accepts_base_and_full_endpoints():
    assert normalize_api_url("https://example.test/v1") == (
        "https://example.test/v1", "chat/completions")
    assert normalize_api_url("https://example.test/v1/responses") == (
        "https://example.test/v1", "responses")
    assert normalize_api_url("https://example.test/v1/chat/completions/") == (
        "https://example.test/v1", "chat/completions")


def test_api_url_rejects_endpoint_conflict_and_embedded_credentials():
    with pytest.raises(RunConfigError):
        normalize_api_url("https://example.test/v1/responses", "chat/completions")
    with pytest.raises(RunConfigError):
        normalize_api_url("https://user:secret@example.test/v1")


def test_resume_rejects_mixed_parameters_and_fills_omitted_values():
    stored = {"track": "image", "candidate_workers": 4}
    assert merge_resume({"track": None, "candidate_workers": 4}, stored) == stored
    with pytest.raises(RunConfigError, match="track"):
        merge_resume({"track": "text", "candidate_workers": None}, stored)


def test_resume_allows_execution_settings_and_key_environment_to_change():
    stored = {"track": "image", "rubric_set": "rubric", "key_env": "OLD_KEY",
              "candidate_workers": 8, "judge_workers": 8, "max_attempts": 3,
              "timeout": 120.0}
    provided = {"track": None, "rubric_set": None, "key_env": "NEW_KEY",
                "candidate_workers": 2, "judge_workers": 3, "max_attempts": 2,
                "timeout": 240.0}
    merged = merge_resume_parameters(provided, stored)
    assert merged == {**stored, "key_env": "NEW_KEY", "candidate_workers": 2,
                      "judge_workers": 3, "max_attempts": 2, "timeout": 240.0}
    with pytest.raises(RunConfigError, match="track"):
        merge_resume_parameters({**provided, "track": "text"}, stored)


def test_documented_cli_options_are_accepted():
    args = build_parser().parse_args([
        "--rubric-set", "basic_rubric",
        "--candidate-workers", "2",
        "--judge-workers", "3",
        "--timeout-seconds", "45",
    ])
    assert args.rubric_set == "basic_rubric"
    assert args.candidate_workers == 2
    assert args.judge_workers == 3
    assert args.timeout_seconds == 45


def test_rubric_normalization_accepts_canonical_names_and_simple_alias_only():
    assert normalize_rubric("rubric") == "rubric"
    assert normalize_rubric("basic_rubric") == "basic_rubric"
    assert normalize_rubric("basic") == "basic_rubric"
    assert normalize_rubric("simple_rubric") == "basic_rubric"
    assert normalize_rubric("simple") == "basic_rubric"
    with pytest.raises(RunConfigError):
        normalize_rubric("hard")


def test_rubric_help_describes_model_positioning():
    help_text = build_parser().format_help()
    assert "rubric=主要评测/强模型" in help_text
    assert "basic_rubric=简单规则/小模型" in help_text


def test_new_run_defaults_to_primary_rubric(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_benchmark.py", "--dry-run", "--non-interactive"])
    monkeypatch.setattr(
        run_benchmark.subprocess, "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout='{"selected": 0}\n', stderr=""),
    )
    assert run_benchmark.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["parameters"]["rubric_set"] == "rubric"


def test_resume_uses_stored_rubric_instead_of_new_default(tmp_path, monkeypatch, capsys):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    parameters = {
        "track": "image", "rubric_set": "basic_rubric", "scoring_protocol": SCORING_PROTOCOL,
        "target_langs": [], "limit": None,
        "api_base_url": "https://example.test/v1", "endpoint": "chat/completions",
        "candidate_model": "candidate", "judge_model": "judge", "key_env": None,
        "candidate_workers": 4, "judge_workers": 4, "max_attempts": 3, "timeout": 120.0,
    }
    (output_dir / "run.json").write_text(json.dumps({
        "schema_version": RUN_SCHEMA_VERSION, "scoring_protocol": SCORING_PROTOCOL,
        "parameters": parameters, "files": FILES, "counts": {},
    }), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["run_benchmark.py", "--output-dir", str(output_dir),
                                      "--resume", "--dry-run", "--non-interactive"])
    monkeypatch.setattr(
        run_benchmark.subprocess, "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout='{"selected": 0}\n', stderr=""),
    )
    assert run_benchmark.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["parameters"]["rubric_set"] == "basic_rubric"
    assert output["parameters"]["scoring_protocol"] == SCORING_PROTOCOL
    assert output["files"]["scores_by_direction_level1"] == "scores_by_direction_level1.csv"


def test_resume_rejects_old_metadata_before_subprocess(tmp_path, monkeypatch, capsys):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "run.json").write_text(json.dumps({
        "schema_version": "vista_benchmark_run_v1", "parameters": {}, "counts": {},
    }), encoding="utf-8")
    called = False

    def run(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("subprocess must not run for old metadata")

    monkeypatch.setattr(run_benchmark.subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", ["run_benchmark.py", "--output-dir", str(output_dir),
                                      "--resume", "--dry-run", "--non-interactive"])
    assert run_benchmark.main() == run_benchmark.EXIT_CONFIG
    assert called is False
    assert "resume requires current run metadata schema" in capsys.readouterr().err


@pytest.mark.parametrize("change", ["schema", "root_protocol", "parameter_protocol", "files", "counts"])
def test_validate_resume_metadata_requires_current_contract(change):
    metadata = {"schema_version": RUN_SCHEMA_VERSION, "scoring_protocol": SCORING_PROTOCOL,
                "parameters": {"scoring_protocol": SCORING_PROTOCOL},
                "files": FILES, "counts": {}}
    if change == "schema":
        metadata["schema_version"] += "_v1"
    elif change == "root_protocol":
        metadata["scoring_protocol"] += "_v1"
    elif change == "parameter_protocol":
        metadata["parameters"]["scoring_protocol"] += "_v1"
    elif change == "files":
        metadata["files"] = {key: value for key, value in FILES.items()
                             if key != "scores_by_direction_level1"}
    else:
        metadata["counts"] = []
    with pytest.raises(RunConfigError):
        validate_resume_metadata(metadata)
