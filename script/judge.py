#!/usr/bin/env python3
"""Run a text-only keyed-checklist judge over self-segmented candidates."""
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
from dataset import (DataError, PUBLIC_RUBRIC_SETS, iter_jsonl, load_candidate, load_json,
                     load_manifest, projected_criteria, resolve_release_path, rubric_asset_key,
                     write_jsonl_atomic)


def parse_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1])
    value = json.loads(text)
    if not isinstance(value, dict):
        raise DataError("judge response must be a JSON object")
    return value


def validate_judgment(output: dict, image_name: str, target_lang: str,
                      criteria: list[dict]) -> dict:
    if not isinstance(output, dict) or set(output) != {"schema_version", "image_name", "target_lang", "judgments"}:
        raise DataError("invalid judge output fields")
    if output["schema_version"] != "vista_judge_output":
        raise DataError("invalid judge output schema_version")
    if (not isinstance(output["image_name"], str) or not output["image_name"].strip() or
            not isinstance(output["target_lang"], str) or len(output["target_lang"].strip()) < 2 or
            output["image_name"] != image_name or output["target_lang"] != target_lang):
        raise DataError("judge output identity mismatch")
    judgments = output["judgments"]
    if not isinstance(judgments, list):
        raise DataError("judgments must be a list")
    by_id = {row["id"]: row for row in criteria}
    seen = set()
    for row in judgments:
        required = {"criterion_id", "expected_checks", "forbidden_checks", "evidence", "rationale"}
        if not isinstance(row, dict) or set(row) != required:
            raise DataError("invalid judgment fields")
        criterion_id = row["criterion_id"]
        if not isinstance(criterion_id, str) or criterion_id not in by_id or criterion_id in seen:
            raise DataError("unknown or duplicate criterion ID")
        seen.add(criterion_id)
        criterion = by_id[criterion_id]
        expected = row["expected_checks"]
        forbidden = row["forbidden_checks"]
        if not isinstance(expected, list) or any(
                not isinstance(x, dict) or set(x) != {"index", "result"} or
                type(x["index"]) is not int or not isinstance(x["result"], str) or
                x["result"] not in {"met", "unmet"} for x in expected):
            raise DataError(f"invalid expected checklist: {criterion_id}")
        if (not expected or len(expected) != len(criterion["expected"]) or
                {x["index"] for x in expected} != set(range(len(criterion["expected"])))):
            raise DataError(f"expected checklist mismatch: {criterion_id}")
        if not isinstance(forbidden, list) or any(
                not isinstance(x, dict) or set(x) != {"index", "matched"} or
                type(x["index"]) is not int or not isinstance(x["matched"], bool)
                for x in forbidden):
            raise DataError(f"invalid forbidden checklist: {criterion_id}")
        if (len(forbidden) != len(criterion.get("forbidden", [])) or
                {x["index"] for x in forbidden} != set(range(len(criterion.get("forbidden", []))))):
            raise DataError(f"forbidden checklist mismatch: {criterion_id}")
        if (not isinstance(row["evidence"], list) or
                any(not isinstance(item, str) for item in row["evidence"]) or
                not isinstance(row["rationale"], str) or not row["rationale"].strip()):
            raise DataError(f"invalid evidence or rationale: {criterion_id}")
    if seen != set(by_id):
        raise DataError("judge did not return every criterion")
    return output


def validate_judge_record(record: dict, criteria: list[dict]) -> dict:
    required = {"schema_version", "task_id", "image_name", "source_lang", "target_lang",
                "profile", "rubric_set", "status", "attempts", "judge_output"}
    if not isinstance(record, dict):
        raise DataError("judge record must be an object")
    if record.get("status") == "failed":
        required.add("error")
    if set(record) != required:
        raise DataError("invalid judge record fields")
    if record["schema_version"] != "vista_judge_record":
        raise DataError("invalid judge record schema_version")
    for key in ("task_id", "image_name", "source_lang", "target_lang"):
        if not isinstance(record[key], str) or not record[key].strip():
            raise DataError(f"invalid judge record {key}")
    if len(record["target_lang"].strip()) < 2:
        raise DataError("invalid judge record target_lang")
    if record["profile"] not in ("image", "text") or record["rubric_set"] not in PUBLIC_RUBRIC_SETS:
        raise DataError("invalid judge record profile/rubric set")
    if type(record["attempts"]) is not int:
        raise DataError("invalid judge record attempts")
    status = record["status"]
    if status == "zero_criteria":
        if criteria or record["attempts"] != 0 or record["judge_output"] is not None:
            raise DataError("invalid zero_criteria judge record")
    elif status in ("ok", "failed"):
        if not criteria or not 1 <= record["attempts"] <= 3:
            raise DataError("invalid judge record criteria/attempts")
        if status == "ok":
            validate_judgment(record["judge_output"], record["image_name"], record["target_lang"], criteria)
        elif (record["judge_output"] is not None or not isinstance(record["error"], str) or
              not record["error"].strip()):
            raise DataError("invalid failed judge record")
    else:
        raise DataError("invalid judge record status")
    return record


def make_packet(record: dict, candidate: dict, rubric: dict, criteria: list[dict],
                source_blocks: list[dict]) -> dict:
    return {
        "task_id": candidate["task_id"], "image_name": record["name"],
        "source_lang": record["language"], "target_lang": candidate["target_lang"],
        "source_blocks": [{"id": row["id"], "text": row["text"]} for row in source_blocks],
        "candidate_blocks": [{"id": row["id"], "source_text": row["source_text"], "text": row["text"]}
                             for row in candidate["candidate_blocks"]],
        "criteria": criteria, "tolerances": rubric.get("tolerances") or [],
    }


def safe_error(exc: Exception) -> str:
    return re.sub(r"https?://[^\s'\"<>]+", "[REDACTED_URL]",
                  f"{type(exc).__name__}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--rubric-set", choices=PUBLIC_RUBRIC_SETS, default="basic_rubric",
        help=("rubric=主要评测/强模型 (primary evaluation for stronger models); "
              "basic_rubric=简单规则/小模型 (easier criteria for smaller models)"),
    )
    parser.add_argument("--profile", choices=("image", "text"), required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_attempts <= 3:
        parser.error("--max-attempts must be between 1 and 3")
    if args.workers < 1:
        parser.error("--workers must be positive")

    prompt_template = (args.root / "script/prompts/judge.txt").read_text(encoding="utf-8")
    manifest = {row["name"]: row for row in load_manifest(args.root, verify_hashes=False)}
    candidates = []
    by_task = {}
    for raw in iter_jsonl(args.candidates):
        candidate = load_candidate(raw)
        if candidate["task_id"] in by_task:
            raise DataError(f"duplicate candidate task: {candidate['task_id']}")
        record = manifest.get(candidate["image_name"])
        if record is None:
            raise DataError(f"candidate image is outside manifest: {candidate['image_name']}")
        expected_task_id = f"{record['seq']}:{candidate['target_lang']}"
        if candidate["task_id"] != expected_task_id or candidate["source_lang"] != record["language"]:
            raise DataError(f"candidate identity mismatch: {candidate['task_id']}")
        by_task[candidate["task_id"]] = candidate
        candidates.append(candidate)

    prepared = {}
    def prepare(candidate: dict) -> tuple[dict, dict, list[dict], list[dict]]:
        cached = prepared.get(candidate["task_id"])
        if cached is not None:
            return cached
        record = manifest[candidate["image_name"]]
        rubric = load_json(resolve_release_path(
            args.root, record["assets"][rubric_asset_key(args.rubric_set)]["path"]))
        translation = load_json(resolve_release_path(
            args.root, record["assets"]["verified_translation"]["path"]))
        criteria = projected_criteria(rubric, candidate["target_lang"], profile=args.profile)
        value = (record, rubric, criteria, translation["blocks"])
        prepared[candidate["task_id"]] = value
        return value

    existing = {}
    resume_seen = set()
    if args.resume and args.output.exists():
        with args.output.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict) or row.get("schema_version") != "vista_judge_record":
                        raise DataError("resume requires current judge records")
                    task_id = row.get("task_id")
                    candidate = by_task.get(task_id)
                    if candidate is None:
                        raise DataError(f"resume record has unknown task_id: {task_id}")
                    if task_id in resume_seen:
                        raise DataError(f"duplicate resume judge record task_id: {task_id}")
                    resume_seen.add(task_id)
                    if (row.get("profile") != args.profile or
                            row.get("rubric_set") != args.rubric_set or
                            row.get("image_name") != candidate["image_name"] or
                            row.get("source_lang") != candidate["source_lang"] or
                            row.get("target_lang") != candidate["target_lang"]):
                        raise DataError(f"resume judge record contract mismatch: {task_id}")
                    _, _, criteria, _ = prepare(candidate)
                    validate_judge_record(row, criteria)
                except (DataError, json.JSONDecodeError, TypeError) as exc:
                    raise DataError(f"invalid resume judge record at line {line_number}: {exc}") from exc
                if row["status"] in ("zero_criteria", "ok"):
                    existing[task_id] = row
    pending = [candidate for candidate in candidates if candidate["task_id"] not in existing]

    if args.dry_run:
        zero = sum(not prepare(candidate)[2] for candidate in pending)
        print(json.dumps({"tasks": len(pending), "skipped": len(existing),
                          "zero_criteria": zero, "api_requests": 0,
                          "text_only_judge": True}))
        return 0

    config = tomllib.loads(args.config.read_text(encoding="utf-8"))["judge_api"]
    api_config = APIConfig(**config)
    api_config.api_key()
    request_count = 0
    count_lock = threading.Lock()
    fatal_event = threading.Event()

    def evaluate(candidate: dict) -> tuple[dict | None, bool]:
        nonlocal request_count
        if fatal_event.is_set():
            return None, False
        record, rubric, criteria, source_blocks = prepare(candidate)
        base = {"schema_version": "vista_judge_record", "task_id": candidate["task_id"],
                "image_name": candidate["image_name"], "source_lang": candidate["source_lang"],
                "target_lang": candidate["target_lang"], "profile": args.profile,
                "rubric_set": args.rubric_set}
        if not criteria:
            return {**base, "status": "zero_criteria", "attempts": 0,
                    "judge_output": None}, False
        packet = make_packet(record, candidate, rubric, criteria, source_blocks)
        prompt = prompt_template.format(packet=json.dumps(packet, ensure_ascii=False))
        client = OpenAICompatibleClient(api_config)
        last_error = ""
        repair_used = False
        fatal = False
        for attempt in range(1, args.max_attempts + 1):
            try:
                with count_lock:
                    request_count += 1
                raw = client.generate(prompt, temperature=0.0)
                try:
                    output = validate_judgment(parse_json_object(raw), candidate["image_name"],
                                               candidate["target_lang"], criteria)
                except (DataError, json.JSONDecodeError):
                    if repair_used:
                        raise
                    repair_used = True
                    with count_lock:
                        request_count += 1
                    repair = client.generate(
                        prompt + "\nReturn only valid JSON matching the protocol above. "
                        "Re-evaluate any invalid checks. Repair this response:\n" + raw,
                        temperature=0.0)
                    output = validate_judgment(parse_json_object(repair), candidate["image_name"],
                                               candidate["target_lang"], criteria)
                return {**base, "status": "ok", "attempts": attempt,
                        "judge_output": output}, False
            except Exception as exc:
                last_error = safe_error(exc)
                fatal = isinstance(exc, APIError) and not exc.retryable
                if fatal:
                    fatal_event.set()
                    break
        return {**base, "status": "failed", "attempts": attempt,
                "error": last_error, "judge_output": None}, fatal

    fatal_messages = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result, fatal in pool.map(evaluate, pending):
            if result is not None:
                existing[result["task_id"]] = result
                if fatal:
                    fatal_messages.append(result["error"])

    rows = [existing[candidate["task_id"]] for candidate in candidates
            if candidate["task_id"] in existing]
    write_jsonl_atomic(args.output, rows)
    stats = {"tasks": len(pending), "records": len(rows),
             "judge_ok": sum(row.get("status") == "ok" for row in rows),
             "judge_zero_criteria": sum(row.get("status") == "zero_criteria" for row in rows),
             "judge_failed": sum(row.get("status") == "failed" for row in rows),
             "api_requests": request_count, "text_only_judge": True}
    print(json.dumps(stats))
    if fatal_messages:
        print(f"judging stopped: {fatal_messages[0]}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
