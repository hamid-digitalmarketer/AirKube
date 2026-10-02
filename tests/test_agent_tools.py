import sys
import os
from unittest.mock import MagicMock, patch

sys.path.append(os.getcwd())

from agent import tools


def test_check_airflow_health_not_configured(monkeypatch):
    """
    Regression test for #14: when Airflow env vars are missing, the health
    check must say so honestly instead of claiming the scheduler is running.
    """
    monkeypatch.delenv("AIRFLOW_BASE_URL", raising=False)
    monkeypatch.delenv("AIRFLOW_USERNAME", raising=False)
    monkeypatch.delenv("AIRFLOW_PASSWORD", raising=False)

    result = tools._check_airflow_health()
    assert "not configured" in result
    assert "Simulated" not in result
    assert result.startswith("[WARN]")


def test_check_airflow_health_unreachable(monkeypatch):
    """When AIRFLOW_BASE_URL is set but the API can't be reached, report an error."""
    monkeypatch.setenv("AIRFLOW_BASE_URL", "http://localhost:8080")
    monkeypatch.delenv("AIRFLOW_USERNAME", raising=False)
    monkeypatch.delenv("AIRFLOW_PASSWORD", raising=False)

    with patch("agent.tools.requests.get", side_effect=ConnectionError("refused")):
        result = tools._check_airflow_health()

    assert result.startswith("[ERR]")
    assert "Unreachable" in result
    assert "Simulated" not in result


def test_check_airflow_health_healthy(monkeypatch):
    """When Airflow's health API reports the scheduler as healthy, surface that."""
    monkeypatch.setenv("AIRFLOW_BASE_URL", "http://localhost:8080")
    monkeypatch.setenv("AIRFLOW_USERNAME", "admin")
    monkeypatch.setenv("AIRFLOW_PASSWORD", "admin")

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"scheduler": {"status": "healthy"}}

    with patch("agent.tools.requests.get", return_value=mock_resp):
        result = tools._check_airflow_health()

    assert result.startswith("[OK]")
    assert "Simulated" not in result


def test_check_airflow_health_degraded(monkeypatch):
    """When Airflow reports a non-healthy scheduler status, warn instead of OK."""
    monkeypatch.setenv("AIRFLOW_BASE_URL", "http://localhost:8080")
    monkeypatch.setenv("AIRFLOW_USERNAME", "admin")
    monkeypatch.setenv("AIRFLOW_PASSWORD", "admin")

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"scheduler": {"status": "unhealthy"}}

    with patch("agent.tools.requests.get", return_value=mock_resp):
        result = tools._check_airflow_health()

    assert result.startswith("[WARN]")


def test_check_system_health_never_fabricates_simulated_ok(monkeypatch):
    """
    End-to-end: check_system_health's output must never contain the old
    hardcoded "Scheduler Running (Simulated)" string.
    """
    monkeypatch.delenv("AIRFLOW_BASE_URL", raising=False)
    monkeypatch.setattr(tools, "get_connector", lambda: MagicMock())

    result = tools.check_system_health.invoke({})
    assert "Simulated" not in result
