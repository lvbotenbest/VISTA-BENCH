from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading


ROOT = Path(__file__).parents[2]


class MockHandler(BaseHTTPRequestHandler):
    requests = 0

    def do_POST(self):
        type(self).requests += 1
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        prompt = body["messages"][0]["content"]
        if body["model"] == "candidate-model":
            output = {"candidate_blocks": [{"id": 1, "source_text": "mock", "text": "mock"}]}
        else:
            packet_text = prompt.split("Packet:", 1)[1]
            packet = json.loads(packet_text.replace("\\n", "\n").strip())
            output = {
                "schema_version": "vista_judge_output",
                "image_name": packet["image_name"],
                "target_lang": packet["target_lang"],
                "judgments": [
                    {
                        "criterion_id": criterion["id"],
                        "expected_checks": [
                            {"index": index, "result": "met"}
                            for index, _ in enumerate(criterion["expected"])
                        ],
                        "forbidden_checks": [
                            {"index": index, "matched": False}
                            for index, _ in enumerate(criterion.get("forbidden", []))
                        ],
                        "evidence": [],
                        "rationale": "mock judge result",
                    }
                    for criterion in packet["criteria"]
                ],
            }
        response = json.dumps({"choices": [{"message": {"content": json.dumps(output)}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format, *args):
        pass


def test_unified_runner_completes_and_resumes_without_duplicate_requests(tmp_path):
    MockHandler.requests = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    output = tmp_path / "run"
    env = {**os.environ, "VISTA_TEST_KEY": "integration-secret"}
    command = [
        sys.executable, str(ROOT / "run_benchmark.py"),
        "--non-interactive",
        "--api-url", f"http://127.0.0.1:{server.server_port}/v1",
        "--key-env", "VISTA_TEST_KEY",
        "--candidate-model", "candidate-model",
        "--judge-model", "judge-model",
        "--track", "text",
        "--rubric-set", "basic_rubric",
        "--target-lang", "en",
        "--limit", "1",
        "--output-dir", str(output),
        "--candidate-workers", "1",
        "--judge-workers", "1",
    ]
    try:
        first = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
        assert first.returncode == 0, first.stderr or first.stdout
        assert MockHandler.requests == 2
        for name in ("run.json", "candidates.jsonl", "judgments.jsonl", "scores.jsonl",
                     "summary.json", "leaderboard.csv", "coverage.csv",
                     "scores_by_direction.csv", "scores_by_direction_level1.csv",
                     "scores_by_level1.csv"):
            assert (output / name).is_file()
        assert "integration-secret" not in "".join(
            path.read_text(encoding="utf-8", errors="ignore")
            for path in output.rglob("*") if path.is_file()
        )
        metadata = json.loads((output / "run.json").read_text(encoding="utf-8"))
        assert metadata["schema_version"] == "vista_benchmark_run"
        assert metadata["scoring_protocol"] == "binary_expected_independent_vtk"
        assert metadata["parameters"]["scoring_protocol"] == "binary_expected_independent_vtk"

        resumed = subprocess.run(command + ["--resume"], cwd=ROOT, env=env,
                                 text=True, capture_output=True)
        assert resumed.returncode == 0, resumed.stderr or resumed.stdout
        assert MockHandler.requests == 2
        candidates = [json.loads(line) for line in
                      (output / "candidates.jsonl").read_text(encoding="utf-8").splitlines()]
        judgments = [json.loads(line) for line in
                     (output / "judgments.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(candidates) == len({row["task_id"] for row in candidates}) == 1
        assert len(judgments) == len({row["task_id"] for row in judgments}) == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
