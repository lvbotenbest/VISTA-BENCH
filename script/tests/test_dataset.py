from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from dataset import DataError, load_candidate, projected_criteria, resolve_release_path, select_tasks


def criterion(identifier="T1", dimension="translation_quality"):
    return {"id": identifier, "dimension": dimension, "importance": "core",
            "expected": ["meaning"], "forbidden": [], "source_blocks": [1]}


def test_resolve_release_path_rejects_escape(tmp_path):
    with pytest.raises(DataError):
        resolve_release_path(tmp_path, "../outside")


def test_target_projection_and_text_filter():
    rubric = {"target_langs": ["fr"], "criteria": [criterion("V1", "image_understanding")],
              "language_overrides": {"fr": {"criteria": [criterion()]}}}
    assert [x["id"] for x in projected_criteria(rubric, "fr")] == ["V1", "T1"]
    assert [x["id"] for x in projected_criteria(rubric, "fr", profile="text")] == ["T1"]


def test_candidate_schema_is_strict():
    row = {"schema_version": "vista_self_segmented_candidate_v1", "task_id": "1:fr",
           "image_name": "a.jpg", "source_lang": "English", "target_lang": "fr",
           "candidate_blocks": [{"id": 1, "source_text": "Hi", "text": "Salut"}]}
    assert load_candidate(row)["task_id"] == "1:fr"
    with pytest.raises(DataError):
        load_candidate({**row, "extra": True})


def test_select_tasks_limits_fixed_range_before_excluding_done():
    tasks = [{"task_id": name, "target_lang": "fr"} for name in ("a", "b", "c")]
    assert [row["task_id"] for row in select_tasks(tasks, limit=2, done={"a"})] == ["b"]


def test_select_tasks_filters_before_limit():
    tasks = [{"task_id": "a", "target_lang": "de"},
             {"task_id": "b", "target_lang": "fr"},
             {"task_id": "c", "target_lang": "fr"}]
    assert [row["task_id"] for row in
            select_tasks(tasks, target_langs=["fr"], limit=1)] == ["b"]


def test_select_tasks_rejects_unknown_target_language():
    with pytest.raises(DataError, match="unknown target language"):
        select_tasks([{"task_id": "a", "target_lang": "fr"}], target_langs=["xx"])
