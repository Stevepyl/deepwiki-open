from __future__ import annotations

import subprocess
from pathlib import Path

from fastapi.testclient import TestClient

from api.api import app
from api.knowledge_store import RepoKnowledgeStore
from api.pr_analysis import (
    analyze_diff,
    build_reference_index,
    parse_python_references,
    parse_python_symbols,
    parse_unified_diff,
)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return result.stdout.strip()


def make_fixture_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "airflow-fixture"
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "Test User")

    write(repo / "airflow" / "__init__.py", "")
    write(
        repo / "airflow" / "settings.py",
        """from airflow.configuration import conf

SQL_ALCHEMY_CONN = conf.get("database", "SQL_ALCHEMY_CONN")

def block_orm_access():
    return "blocked"
""",
    )
    write(
        repo / "airflow" / "sdk" / "execution_time" / "supervisor.py",
        """from airflow import settings

def start_task():
    conn = settings.SQL_ALCHEMY_CONN
    settings.block_orm_access()
    return conn
""",
    )
    git(repo, "add", ".")
    git(repo, "commit", "-m", "base")
    git(repo, "branch", "base")

    write(
        repo / "airflow" / "settings.py",
        """from airflow.configuration import conf

def get_sql_alchemy_conn():
    return conf.get("database", "SQL_ALCHEMY_CONN")

def block_orm_access():
    return "blocked"
""",
    )
    git(repo, "add", ".")
    git(repo, "commit", "-m", "head")
    git(repo, "branch", "head")
    return repo


def test_parse_unified_diff_detects_python_file(tmp_path: Path):
    repo = make_fixture_repo(tmp_path)
    patch = git(repo, "diff", "base..head", "--")

    files = parse_unified_diff(patch)

    assert len(files) == 1
    assert files[0].path == "airflow/settings.py"
    assert files[0].hunks[0].to_dict()["removed_lines"]
    assert files[0].hunks[0].to_dict()["added_lines"]


def test_ast_extracts_symbols_and_references():
    settings_source = """from airflow.configuration import conf
SQL_ALCHEMY_CONN = conf.get("database", "SQL_ALCHEMY_CONN")
def get_sql_alchemy_conn():
    return conf.get("database", "SQL_ALCHEMY_CONN")
"""
    supervisor_source = """from airflow import settings
def start_task():
    settings.block_orm_access()
"""

    symbols = parse_python_symbols(settings_source, "airflow/settings.py")
    settings_refs = parse_python_references(settings_source, "airflow/settings.py")
    refs = parse_python_references(supervisor_source, "airflow/sdk/execution_time/supervisor.py")

    assert any(s.id == "airflow.settings.SQL_ALCHEMY_CONN" for s in symbols)
    assert any(s.id == "airflow.settings.get_sql_alchemy_conn" for s in symbols)
    assert any(
        r.ref_type == "config_read" and r.target == "config.database.SQL_ALCHEMY_CONN"
        for r in settings_refs
    )
    assert any(
        r.ref_type == "attribute_access" and r.target == "airflow.settings.block_orm_access"
        for r in refs
    )
    assert any(r.ref_type == "call" and r.target == "airflow.settings.block_orm_access" for r in refs)


def test_impact_expansion_and_risk_rules(tmp_path: Path):
    repo = make_fixture_repo(tmp_path)
    store = RepoKnowledgeStore(tmp_path / "metadata.db")
    repo_id = "fixture"
    build_reference_index(repo_id, repo, store)
    result = analyze_diff(repo_id, repo, "base", "head", store)

    assert any(s["id"] == "airflow.settings.SQL_ALCHEMY_CONN" for s in result["changed_symbols"])
    assert any(path["nodes"][0]["id"] == "airflow.settings.SQL_ALCHEMY_CONN" for path in result["impact_paths"])
    assert any(risk["rule"] == "module_global_removed" for risk in result["risks"])
    assert any(risk["rule"] == "settings_attribute_compat" for risk in result["risks"])
    assert "task execution" in result["report"].lower()
    store.close()


def test_api_analyze_diff_and_followup(tmp_path: Path, monkeypatch):
    repo = make_fixture_repo(tmp_path)
    db_path = tmp_path / "api-metadata.db"

    def test_store_factory():
        return RepoKnowledgeStore(db_path)

    monkeypatch.setattr("api.api.RepoKnowledgeStore", test_store_factory)
    client = TestClient(app)

    response = client.post(
        "/api/pr-analysis/analyze",
        json={
            "repo_id": "fixture-api",
            "repo_path": str(repo),
            "base": "base",
            "head": "head",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"]
    assert payload["changed_symbols"]
    assert payload["risks"]

    followup = client.post(
        f"/api/pr-analysis/{payload['session_id']}/ask",
        json={"question": "Why is this risky?"},
    )
    assert followup.status_code == 200
    assert "risk" in followup.json()["answer"].lower() or "path" in followup.json()["answer"].lower()


def test_api_rejects_invalid_github_pr_url():
    client = TestClient(app)

    response = client.post(
        "/api/pr-analysis/analyze",
        json={"pr_url": "https://example.com/not/a/pr"},
    )

    assert response.status_code == 400
    assert "github" in response.json()["detail"].lower()
