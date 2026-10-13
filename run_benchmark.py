#!/usr/bin/env python3
"""Safe interactive and non-interactive VISTA benchmark runner."""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parent
EXIT_CONFIG = 2
EXIT_CANDIDATE = 3
EXIT_JUDGE = 4
EXIT_SCORING = 5
EXIT_SUMMARY = 6
SCORING_PROTOCOL = "binary_expected_independent_vtk"
RUN_SCHEMA_VERSION = "vista_benchmark_run"
RESUME_OVERRIDABLE = {"key_env", "candidate_workers", "judge_workers", "max_attempts", "timeout"}
FILES = {
    "candidates": "candidates.jsonl",
    "candidate_failures": "candidate_failures.jsonl",
    "judgments": "judgments.jsonl",
    "scores": "scores.jsonl",
    "run_metadata": "run.json",
    "summary": "summary.json",
    "leaderboard": "leaderboard.csv",
    "coverage": "coverage.csv",
    "scores_by_direction": "scores_by_direction.csv",
    "scores_by_direction_level1": "scores_by_direction_level1.csv",
    "scores_by_level1": "scores_by_level1.csv",
}


class RunConfigError(ValueError):
    pass


def normalize_api_url(value: str, endpoint: str | None = None) -> tuple[str, str]:
    value = value.strip()
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RunConfigError("--api-url must be an explicit HTTP(S) URL")
    if parsed.query or parsed.fragment:
        raise RunConfigError("--api-url must not contain a query or fragment")
    if parsed.username or parsed.password:
        raise RunConfigError("--api-url must not contain embedded credentials")
    path = parsed.path.rstrip("/")
    inferred = None
    for suffix in ("chat/completions", "responses"):
        marker = "/" + suffix
        if path.endswith(marker):
            inferred = suffix
            path = path[:-len(marker)].rstrip("/")
            break
    if inferred and endpoint and inferred != endpoint:
        raise RunConfigError("--endpoint conflicts with the full --api-url")
    endpoint = inferred or endpoint or "chat/completions"
    if endpoint not in {"chat/completions", "responses"}:
        raise RunConfigError("endpoint must be chat/completions or responses")
    base_url = urlunsplit((parsed.scheme, parsed.netloc, path, "", "")).rstrip("/")
    return base_url, endpoint


def atomic_json(path: Path, value: dict) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunConfigError(f"cannot read run metadata: {exc}") from exc
    if not isinstance(value, dict):
        raise RunConfigError("run metadata root must be an object")
    return value


def validate_resume_metadata(metadata: dict) -> None:
    if metadata.get("schema_version") != RUN_SCHEMA_VERSION:
        raise RunConfigError("resume requires current run metadata schema")
    if metadata.get("scoring_protocol") != SCORING_PROTOCOL:
        raise RunConfigError("resume run metadata scoring_protocol mismatch")
    parameters = metadata.get("parameters")
    if not isinstance(parameters, dict) or parameters.get("scoring_protocol") != SCORING_PROTOCOL:
        raise RunConfigError("resume run parameters scoring_protocol mismatch")
    if metadata.get("files") != FILES:
        raise RunConfigError("resume run metadata files mismatch")
    counts = metadata.get("counts")
    if not isinstance(counts, dict):
        raise RunConfigError("resume run metadata counts must be an object")


def prompt_value(label: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    value = input(f"{label}{suffix}: ").strip()
    return value or (default or "")


def normalize_rubric(value: str | None) -> str | None:
    if value is None:
        return None
    aliases = {"basic": "basic_rubric", "basic_rubric": "basic_rubric",
               "simple": "basic_rubric", "simple_rubric": "basic_rubric", "rubric": "rubric"}
    try:
        return aliases[value]
    except KeyError as exc:
        raise RunConfigError(
            "--rubric-set must be rubric or basic_rubric"
        ) from exc


def normalize_langs(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None
    langs = []
    for value in values:
        langs.extend(item.strip() for item in value.split(",") if item.strip())
    return sorted(set(langs))


def merge_resume(provided: dict, stored: dict) -> dict:
    merged = {}
    for key, value in provided.items():
        old = stored.get(key)
        if value is not None and value != old:
            raise RunConfigError(f"resume parameter mismatch: {key}")
        merged[key] = old if value is None else value
    return merged


def merge_resume_parameters(provided: dict, stored: dict) -> dict:
    """Keep the run identity fixed while allowing execution settings to change."""
    strict = {key: value for key, value in provided.items() if key not in RESUME_OVERRIDABLE}
    merged = merge_resume(strict, stored)
    for key in RESUME_OVERRIDABLE:
        value = provided.get(key)
        merged[key] = stored.get(key) if value is None else value
    return merged


def parse_stats(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    return {}


def toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def make_config(path: Path, parameters: dict, config_key_env: str) -> None:
    common = (f"base_url = {toml_string(parameters['api_base_url'])}\n"
              f"endpoint = {toml_string(parameters['endpoint'])}\n")
    text = ("[candidate_api]\n" + common +
            f"model = {toml_string(parameters['candidate_model'])}\n"
            f"key_env = {toml_string(config_key_env)}\n"
            f"timeout_seconds = {parameters['timeout']}\n\n"
            "[judge_api]\n" + common +
            f"model = {toml_string(parameters['judge_model'])}\n"
            f"key_env = {toml_string(config_key_env)}\n"
            f"timeout_seconds = {parameters['timeout']}\n")
    path.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the VISTA benchmark pipeline")
    parser.add_argument("--api-url")
    parser.add_argument("--endpoint", choices=("chat/completions", "responses"))
    parser.add_argument("--candidate-model")
    parser.add_argument("--judge-model")
    parser.add_argument("--key-env", help="environment variable containing the shared API key")
    parser.add_argument("--track", choices=("image", "text"))
    parser.add_argument(
        "--rubric-set", "--rubric", dest="rubric_set", metavar="SET",
        help=("rubric=主要评测/强模型 (primary evaluation for stronger models); "
              "basic_rubric=简单规则/小模型 (easier criteria for smaller models)"),
    )
    parser.add_argument("--target-lang", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--candidate-workers", type=int)
    parser.add_argument("--judge-workers", type=int)
    parser.add_argument("--workers", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--timeout-seconds", "--timeout", dest="timeout_seconds", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--interactive", action="store_true")
    mode.add_argument("--non-interactive", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.limit is not None and args.limit < 0:
            raise RunConfigError("--limit must be non-negative")
        if args.workers is not None:
            if args.candidate_workers is not None or args.judge_workers is not None:
                raise RunConfigError("--workers cannot be combined with stage-specific worker options")
            args.candidate_workers = args.workers
            args.judge_workers = args.workers
        if any(value is not None and value < 1
               for value in (args.candidate_workers, args.judge_workers)):
            raise RunConfigError("worker counts must be positive")
        if args.max_attempts is not None and not 1 <= args.max_attempts <= 3:
            raise RunConfigError("--max-attempts must be between 1 and 3")
        if args.timeout_seconds is not None and args.timeout_seconds <= 0:
            raise RunConfigError("--timeout-seconds must be positive")

        missing = any(value is None for value in
                      (args.output_dir, args.api_url, args.candidate_model))
        wizard = args.interactive or (not args.dry_run and not args.non_interactive and
                                      sys.stdin.isatty() and missing and not args.resume)
        if wizard:
            print("VISTA benchmark setup")
            args.track = args.track or prompt_value("Track (image/text)", "image")
            args.rubric_set = args.rubric_set or prompt_value(
                "Rubric (rubric=harder criteria/strong; basic_rubric=easier criteria/small)", "rubric")
            args.api_url = args.api_url or prompt_value("Shared API URL")
            args.candidate_model = args.candidate_model or prompt_value("Candidate model")
            args.judge_model = args.judge_model or prompt_value("Judge model", args.candidate_model)
            if args.target_lang is None:
                value = prompt_value("Target languages, comma-separated (blank for all)", "")
                args.target_lang = [value] if value else []
            if args.limit is None:
                value = prompt_value("Task limit (blank for all)", "")
                args.limit = int(value) if value else None
            args.output_dir = args.output_dir or Path(prompt_value("Output directory", "outputs/benchmark"))

        metadata_path = args.output_dir / FILES["run_metadata"] if args.output_dir else None
        stored_metadata = None
        if args.resume:
            if metadata_path is None or not metadata_path.is_file():
                raise RunConfigError("--resume requires an existing output-dir/run.json")
            stored_metadata = load_json(metadata_path)
            validate_resume_metadata(stored_metadata)

        rubric = normalize_rubric(args.rubric_set)
        target_langs = normalize_langs(args.target_lang)
        supplied_base = supplied_endpoint = None
        if args.api_url is not None:
            supplied_base, supplied_endpoint = normalize_api_url(args.api_url, args.endpoint)
        elif args.endpoint is not None:
            supplied_endpoint = args.endpoint
        provided = {
            "track": args.track,
            "rubric_set": rubric,
            "scoring_protocol": SCORING_PROTOCOL,
            "target_langs": target_langs,
            "limit": args.limit,
            "api_base_url": supplied_base,
            "endpoint": supplied_endpoint,
            "candidate_model": args.candidate_model,
            "judge_model": args.judge_model,
            "key_env": args.key_env,
            "candidate_workers": args.candidate_workers,
            "judge_workers": args.judge_workers,
            "max_attempts": args.max_attempts,
            "timeout": args.timeout_seconds,
        }
        if stored_metadata is not None:
            parameters = merge_resume_parameters(provided, stored_metadata.get("parameters") or {})
        else:
            parameters = provided
            parameters.update({
                "track": parameters["track"] or "image",
                "rubric_set": parameters["rubric_set"] or "rubric",
                "target_langs": parameters["target_langs"] or [],
                "endpoint": parameters["endpoint"] or "chat/completions",
                "judge_model": parameters["judge_model"] or parameters["candidate_model"],
                "candidate_workers": parameters["candidate_workers"] or 4,
                "judge_workers": parameters["judge_workers"] or 4,
                "max_attempts": parameters["max_attempts"] or 3,
                "timeout": parameters["timeout"] or 120.0,
            })
        if args.dry_run:
            parameters.update({
                "api_base_url": parameters["api_base_url"] or "https://dry-run.invalid/v1",
                "candidate_model": parameters["candidate_model"] or "dry-run",
                "judge_model": parameters["judge_model"] or parameters["candidate_model"] or "dry-run",
            })
        required = ("track", "rubric_set", "api_base_url", "endpoint",
                    "candidate_model", "judge_model", "candidate_workers", "judge_workers",
                    "max_attempts", "timeout")
        absent = [key for key in required if parameters.get(key) in {None, ""}]
        if args.output_dir is None and not args.dry_run:
            absent.append("output_dir")
        if absent:
            raise RunConfigError("missing required arguments: " + ", ".join(absent))
        if "=" in parameters["candidate_model"]:
            raise RunConfigError("candidate model must not contain '='")

        output_dir = (args.output_dir or (ROOT / "outputs/dry-run")).resolve()
        paths = {key: output_dir / name for key, name in FILES.items()}
        if not args.resume and not args.dry_run:
            occupied = [path.name for path in paths.values() if path.exists()]
            if occupied:
                raise RunConfigError("output directory contains benchmark artifacts; use --resume: " +
                                     ", ".join(sorted(occupied)))

        child_env = os.environ.copy()
        config_key_env = parameters.get("key_env")
        if not args.dry_run:
            if config_key_env:
                if not child_env.get(config_key_env):
                    raise RunConfigError(f"API key environment variable {config_key_env!r} is not set")
            else:
                if args.non_interactive or not sys.stdin.isatty():
                    raise RunConfigError("non-interactive runs require --key-env")
                secret = getpass.getpass("Shared API key: ")
                if not secret:
                    raise RunConfigError("API key must not be empty")
                suffix = hashlib.sha256(os.urandom(16)).hexdigest()[:12].upper()
                config_key_env = f"VISTA_RUN_KEY_{suffix}"
                child_env[config_key_env] = secret
                secret = ""

        with tempfile.TemporaryDirectory(prefix="vista-benchmark-") as temporary_dir:
            config_path = Path(temporary_dir) / "config.toml"
            make_config(config_path, parameters, config_key_env or "VISTA_DRY_RUN_KEY")
            candidate_script = (ROOT / "script" /
                                ("generate_image_candidates.py" if parameters["track"] == "image"
                                 else "generate_text_candidates.py"))
            candidate_command = [sys.executable, str(candidate_script), "--root", str(ROOT),
                                 "--config", str(config_path), "--output", str(paths["candidates"]),
                                 "--failures-output", str(paths["candidate_failures"]),
                                 "--workers", str(parameters["candidate_workers"]), "--max-attempts",
                                 str(parameters["max_attempts"])]
            for lang in parameters["target_langs"]:
                candidate_command.extend(["--target-lang", lang])
            if parameters["limit"] is not None:
                candidate_command.extend(["--limit", str(parameters["limit"])])
            if args.resume:
                candidate_command.append("--resume")
            if args.dry_run:
                candidate_command.append("--dry-run")
                result = subprocess.run(candidate_command, cwd=ROOT, env=child_env,
                                        text=True, capture_output=True)
                if result.returncode:
                    if result.stderr:
                        print(result.stderr.rstrip(), file=sys.stderr)
                    return EXIT_CANDIDATE
                stats = parse_stats(result.stdout)
                public_parameters = {key: parameters[key] for key in
                                     ("track", "rubric_set", "scoring_protocol", "target_langs", "limit",
                                      "candidate_workers", "judge_workers", "max_attempts", "timeout")}
                print(json.dumps({"dry_run": True, "parameters": public_parameters,
                                  "plan": stats, "files": FILES}, ensure_ascii=False, indent=2))
                return 0

            output_dir.mkdir(parents=True, exist_ok=True)
            metadata = stored_metadata or {
                "schema_version": RUN_SCHEMA_VERSION,
                "scoring_protocol": SCORING_PROTOCOL,
                "parameters": parameters, "files": FILES, "counts": {},
            }
            metadata.update({"status": "running", "stage": "candidate",
                             "scoring_protocol": SCORING_PROTOCOL})
            metadata["parameters"] = parameters
            metadata["files"] = FILES
            atomic_json(paths["run_metadata"], metadata)

            def run_stage(stage: str, command: list[str], exit_code: int) -> dict:
                metadata["stage"] = stage
                atomic_json(paths["run_metadata"], metadata)
                result = subprocess.run(command, cwd=ROOT, env=child_env,
                                        text=True, capture_output=True)
                if result.returncode:
                    metadata.update({"status": "failed", "failed_stage": stage,
                                     "exit_status": exit_code})
                    atomic_json(paths["run_metadata"], metadata)
                    if result.stderr:
                        print(result.stderr.rstrip(), file=sys.stderr)
                    raise SystemExit(exit_code)
                stats = parse_stats(result.stdout)
                print(f"{stage}: {json.dumps(stats, ensure_ascii=False)}")
                return stats

            candidate_stats = run_stage("candidate", candidate_command, EXIT_CANDIDATE)
            metadata["counts"].update({key: candidate_stats.get(key, 0) for key in
                                       ("selected", "candidate_succeeded", "candidate_failed")})
            judge_command = [sys.executable, str(ROOT / "script/judge.py"),
                             "--root", str(ROOT), "--config", str(config_path),
                             "--candidates", str(paths["candidates"]),
                             "--output", str(paths["judgments"]), "--rubric-set",
                             parameters["rubric_set"], "--profile", parameters["track"],
                             "--workers", str(parameters["judge_workers"]), "--max-attempts",
                             str(parameters["max_attempts"])]
            if args.resume:
                judge_command.append("--resume")
            judge_stats = run_stage("judge", judge_command, EXIT_JUDGE)
            metadata["counts"].update({key: judge_stats.get(key, 0) for key in
                                       ("judge_ok", "judge_zero_criteria", "judge_failed")})
            scoring_command = [sys.executable, str(ROOT / "script/scoring.py"),
                               "--root", str(ROOT), "--judge-records", str(paths["judgments"]),
                               "--rubric-set", parameters["rubric_set"], "--profile",
                               parameters["track"], "--output", str(paths["scores"])]
            run_stage("scoring", scoring_command, EXIT_SCORING)
            with paths["scores"].open(encoding="utf-8") as stream:
                metadata["counts"]["scored"] = sum(bool(line.strip()) for line in stream)
            atomic_json(paths["run_metadata"], metadata)
            summary_command = [sys.executable, str(ROOT / "script/summarize.py"),
                               "--scores", f"{parameters['candidate_model']}={paths['scores']}",
                               "--profile", parameters["track"], "--rubric-set",
                               parameters["rubric_set"], "--output-dir", str(output_dir),
                               "--run-metadata", str(paths["run_metadata"])]
            run_stage("summarize", summary_command, EXIT_SUMMARY)
            partial = bool(metadata["counts"].get("candidate_failed") or
                           metadata["counts"].get("judge_failed"))
            exit_status = EXIT_CANDIDATE if partial else 0
            metadata.update({"status": "partial" if partial else "completed",
                             "stage": "completed", "exit_status": exit_status})
            metadata.pop("failed_stage", None)
            atomic_json(paths["run_metadata"], metadata)
            print(f"{'partial' if partial else 'completed'}: {output_dir}")
            return exit_status
    except RunConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG


if __name__ == "__main__":
    raise SystemExit(main())
