"""
Shared fixtures for tool tests.

Creates a temporary "fake repo" file tree that all tool tests can operate on,
isolating tests from the real filesystem.
"""

import os

import pytest


@pytest.fixture()
def fake_repo(tmp_path):
    """
    Build a minimal repo-like directory tree under tmp_path:

        tmp_path/
            src/
                main.py          — Python with a function
                utils.py         — Python helper
                components/
                    button.tsx   — TypeScript React component
            .github/
                workflows/
                    ci.yml       — hidden dir file (should be found with --hidden)
            .git/
                HEAD             — must be EXCLUDED by tools
            README.md
            setup.cfg
    """
    # src/
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text(
        'def main():\n    print("hello world")\n\ndef helper():\n    return 42\n',
        encoding="utf-8",
    )
    (src / "utils.py").write_text(
        'import os\n\ndef read_file(path):\n    with open(path) as f:\n        return f.read()\n',
        encoding="utf-8",
    )

    components = src / "components"
    components.mkdir()
    (components / "button.tsx").write_text(
        'export function Button({ label }: { label: string }) {\n  return <button>{label}</button>\n}\n',
        encoding="utf-8",
    )

    # .github/ (hidden, but should be searchable)
    gh = tmp_path / ".github" / "workflows"
    gh.mkdir(parents=True)
    (gh / "ci.yml").write_text("name: CI\non: [push]\njobs:\n  build:\n    runs-on: ubuntu-latest\n", encoding="utf-8")

    # .git/ (should be EXCLUDED by tools)
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")

    # Root files
    (tmp_path / "README.md").write_text("# Fake Repo\n\nThis is a test repo.\n", encoding="utf-8")
    (tmp_path / "setup.cfg").write_text("[metadata]\nname = fakerepo\n", encoding="utf-8")

    return tmp_path


@pytest.fixture()
def repo_path(fake_repo):
    """Return the fake repo path as a string (matches tool constructor signature)."""
    return str(fake_repo)
