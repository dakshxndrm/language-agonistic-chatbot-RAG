"""Tests for scripts/preflight.py air-gap verification and port sanity checks."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from scripts.preflight import check_port, run_preflight


def test_check_port_success():
    with patch("socket.socket") as mock_sock:
        mock_instance = mock_sock.return_value.__enter__.return_value
        mock_instance.connect_ex.return_value = 0
        assert check_port("PostgreSQL", "localhost", 5432) is True


def test_check_port_failure():
    with patch("socket.socket") as mock_sock:
        mock_instance = mock_sock.return_value.__enter__.return_value
        mock_instance.connect_ex.return_value = 111
        assert check_port("PostgreSQL", "localhost", 5432) is False


def test_run_preflight_onprem_fails_on_cloud_keys(monkeypatch):
    monkeypatch.setenv("BHASHA_PROFILE", "onprem")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_leaked_key")

    with pytest.raises(SystemExit) as exc:
        run_preflight()
    assert exc.value.code == 1


def test_run_preflight_success(monkeypatch):
    monkeypatch.setenv("BHASHA_PROFILE", "onprem")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with patch("scripts.preflight.check_port", return_value=True):
        # Should complete without sys.exit
        run_preflight()
