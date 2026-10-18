"""Tests for scripts/eval_ragas.py viva report generator."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.eval_ragas import compile_viva_report


def test_compile_viva_report_success(tmp_path: Path, capsys):
    test_eval_file = tmp_path / "final_eval.json"
    data = {
        "summary": {
            "total_queries": 50,
            "script_preservation_rate": 0.84,
            "language_fidelity_rate": 0.86,
            "citation_validity_rate": 1.0,
            "citation_grounding_rate": 1.0,
        }
    }
    test_eval_file.write_text(json.dumps(data), encoding="utf-8")

    compile_viva_report(str(test_eval_file))
    captured = capsys.readouterr()
    assert "BHASHA" in captured.out
    assert "84.0%" in captured.out
    assert "86.0%" in captured.out


def test_compile_viva_report_missing_file(capsys):
    compile_viva_report("nonexistent_file_path.json")
    captured = capsys.readouterr()
    assert "not found" in captured.out
