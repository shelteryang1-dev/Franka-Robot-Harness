"""Test benchmark result validation."""

import json

import pytest

from scripts.run_benchmark import _validate_fresh_results


def test_validation_accepts_new_summary_with_expected_episode_count(tmp_path):

    old_summary = json.dumps({"total_episodes": 1}).encode()
    (tmp_path / "summary.json").write_text(json.dumps({"total_episodes": 32}), encoding="utf-8")
    _validate_fresh_results(tmp_path, old_summary, expected_episodes=32)


def test_validation_rejects_unchanged_old_summary(tmp_path):

    old_summary = json.dumps({"total_episodes": 32}).encode()
    (tmp_path / "summary.json").write_bytes(old_summary)
    with pytest.raises(RuntimeError, match="Summary unchanged"):
        _validate_fresh_results(tmp_path, old_summary, expected_episodes=32)


def test_validation_rejects_wrong_episode_count(tmp_path):

    (tmp_path / "summary.json").write_text(json.dumps({"total_episodes": 8}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Episode count mismatch"):
        _validate_fresh_results(tmp_path, None, expected_episodes=32)
