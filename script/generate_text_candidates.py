#!/usr/bin/env python3
"""Generate self-segmented translations from released OCR text only."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import sys
import threading
import tomllib

from api import APIConfig, APIError, OpenAICompatibleClient
from dataset import (DataError, load_candidate, load_json, load_manifest,
                     resolve_release_path, select_tasks, write_jsonl_atomic)


def parse_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1])
    value = json.loads(text)
    if not isinstance(value, dict) or not isinstance(value.get("candidate_blocks"), list):
        raise DataError("response must be an object containing candidate_blocks")
    return value


def safe_error(exc: Exception) -> str:
    return re.sub(r"https?://[^\s'\"<>]+", "[REDACTED_URL]",
                  f"{type(exc).__name__}: {exc}")


def valid_existing(path: Path, tasks: dict[str, dict]) -> dict[str, dict]:
    valid = {}
    if not path.exists():
        return valid
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = load_candidate(json.loads(line))
                task = tasks.get(row["task_id"])
                if (task is None or row["image_name"] != task["record"]["name"] or
                        row["source_lang"] != task["record"]["language"] or
                        row["target_lang"] != task["target_lang"]):
                    continue
                valid.setdefault(row["task_id"], row)
            except (DataError, json.JSONDecodeError, TypeError):
                continue
    return valid


def existing_failures(path: Path, tasks: dict[str, dict]) -> dict[str, dict]:
    failures = {}
    if not path.exists():
        return failures
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
                task_id = row.get("task_id")
                if (isinstance(row, dict) and task_id in tasks and
                        row.get("status") == "failed"):
                    failures[task_id] = row
            except (json.JSONDecodeError, AttributeError):
                continue
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failures-output", type=Path)
    parser.add_argument("--target-lang", action="append")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if not 1 <= args.max_attempts <= 3:
        parser.error("--max-attempts must be between 1 and 3")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")
    failures_output = args.failures_output or args.output.with_name(
        f"{args.output.stem}.failures.jsonl")

    all_tasks = []
    for record in load_manifest(args.root, verify_hashes=False):
        ocr = load_json(resolve_release_path(args.root, record["assets"]["ocr"]["path"]))
        lines = [line for line in ocr["text"].splitlines() if line.strip()]
        source_blocks = [{"id": index + 1, "source_text": line}
                         for index, line in enumerate(lines)]
        translation = load_json(resolve_release_path(
            args.root, record["assets"]["verified_translation"]["path"]))
        for lang in translation["target_langs"]:
            all_tasks.append({"record": record, "target_lang": lang,
                              "task_id": f"{record['seq']}:{lang}",
                              "source_blocks": source_blocks})
    tasks_by_id = {task["task_id"]: task for task in all_tasks}
    existing = valid_existing(args.output, tasks_by_id) if args.resume else {}
    fixed = select_tasks(all_tasks, target_langs=args.target_lang, limit=args.limit)
    pending = select_tasks(all_tasks, target_langs=args.target_lang, limit=args.limit,
                           done=existing)
    if args.dry_run:
        print(json.dumps({"mode": "text", "selected": len(fixed),
                          "tasks": len(pending), "skipped": len(fixed) - len(pending),
                          "api_requests": 0}))
        return 0

    config = tomllib.loads(args.config.read_text(encoding="utf-8"))["candidate_api"]
    api_config = APIConfig(**config)
    api_config.api_key()
    prompt_template = (args.root / "script/prompts/candidate_text.txt").read_text(encoding="utf-8")
    failures = existing_failures(failures_output, tasks_by_id) if args.resume else {}
    request_count = 0
    count_lock = threading.Lock()
    fatal_event = threading.Event()

    def generate(task: dict) -> tuple[str, dict | None, dict | None, bool]:
        nonlocal request_count
        if fatal_event.is_set():
            return task["task_id"], None, None, False
        client = OpenAICompatibleClient(api_config)
        prompt = prompt_template.format(
            target_lang=task["target_lang"],
            source_blocks=json.dumps(task["source_blocks"], ensure_ascii=False))
        last_error = ""
        fatal = False
        for attempt in range(1, args.max_attempts + 1):
            try:
                with count_lock:
                    request_count += 1
                response = parse_object(client.generate(prompt, temperature=0.0))
                record = task["record"]
                output = load_candidate({
                    "schema_version": "vista_self_segmented_candidate_v1",
                    "task_id": task["task_id"], "image_name": record["name"],
                    "source_lang": record["language"], "target_lang": task["target_lang"],
                    "candidate_blocks": response["candidate_blocks"],
                }, expected_task_id=task["task_id"])
                return task["task_id"], output, None, False
            except Exception as exc:
                last_error = safe_error(exc)
                fatal = isinstance(exc, APIError) and not exc.retryable
                if fatal:
                    fatal_event.set()
                    break
        failure = {"schema_version": "vista_candidate_failure_v1",
                   "task_id": task["task_id"], "status": "failed",
                   "attempts": attempt, "error": last_error}
        return task["task_id"], None, failure, fatal

    succeeded = failed = 0
    fatal_messages = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for task_id, candidate, failure, fatal in pool.map(generate, pending):
            if candidate is not None:
                existing[task_id] = candidate
                failures.pop(task_id, None)
                succeeded += 1
            elif failure is not None:
                failures[task_id] = failure
                failed += 1
                if fatal:
                    fatal_messages.append(failure["error"])

    write_jsonl_atomic(args.output, (existing[task["task_id"]]
                                     for task in all_tasks if task["task_id"] in existing))
    write_jsonl_atomic(failures_output, (failures[task["task_id"]]
                                         for task in all_tasks if task["task_id"] in failures))
    fixed_ids = {task["task_id"] for task in fixed}
    stats = {"mode": "text", "selected": len(fixed), "tasks": len(pending),
             "succeeded": succeeded, "failed": failed,
             "candidate_succeeded": sum(task_id in existing for task_id in fixed_ids),
             "candidate_failed": sum(task_id in failures and task_id not in existing
                                     for task_id in fixed_ids),
             "api_requests": request_count}
    print(json.dumps(stats))
    if fatal_messages:
        print(f"candidate generation stopped: {fatal_messages[0]}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
