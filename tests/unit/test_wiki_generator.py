"""
Unit tests for api/agent/wiki_generator.py.

Tests cover:
- parse_wiki_structure: multi-layer JSON parsing with all failure modes
- _flatten_pages_in_section_order: section-order traversal, orphan pages
- _compute_repo_name: URL parsing
- _validate_wiki_structure: schema validation
- Agent config registration: wiki-planner and wiki-writer
"""

import pytest

from api.agent.config import get_agent_config
from api.agent.wiki_generator import (
    _compute_repo_name,
    _flatten_pages_in_section_order,
    _validate_wiki_structure,
    parse_wiki_structure,
    AgentWikiRequest,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_page(pid: str, title: str = "Page", file_paths=None, related=None) -> dict:
    return {
        "id": pid,
        "title": title,
        "content": "",
        "filePaths": file_paths or [],
        "importance": "medium",
        "relatedPages": related or [],
    }


def _make_structure(pages, sections=None, root_sections=None) -> dict:
    return {
        "id": "wiki-root",
        "title": "Test Wiki",
        "description": "A test repository wiki.",
        "pages": pages,
        "sections": sections,
        "rootSections": root_sections,
    }


# ---------------------------------------------------------------------------
# parse_wiki_structure
# ---------------------------------------------------------------------------


class TestParseWikiStructure:
    def test_plain_json(self):
        raw = """{
            "id": "wiki-root",
            "title": "My Repo",
            "description": "desc",
            "pages": [
                {"id": "p1", "title": "Overview", "content": "", "filePaths": [], "importance": "high", "relatedPages": []}
            ]
        }"""
        result = parse_wiki_structure(raw)
        assert result is not None
        assert result["title"] == "My Repo"
        assert len(result["pages"]) == 1

    def test_strips_markdown_code_fence(self):
        raw = '```json\n{"id":"r","title":"T","description":"d","pages":[{"id":"p1","title":"P"}]}\n```'
        result = parse_wiki_structure(raw)
        assert result is not None
        assert result["title"] == "T"

    def test_strips_code_fence_without_json_label(self):
        raw = '```\n{"id":"r","title":"T","description":"d","pages":[{"id":"p1","title":"P"}]}\n```'
        result = parse_wiki_structure(raw)
        assert result is not None

    def test_ignores_leading_explanatory_text(self):
        raw = 'Here is the wiki structure:\n{"id":"r","title":"T","description":"d","pages":[{"id":"p1","title":"P"}]}\n\nNote: ...'
        result = parse_wiki_structure(raw)
        assert result is not None

    def test_returns_none_for_invalid_json(self):
        raw = '{"id":"r","title":"T","description":"d","pages":['  # truncated
        result = parse_wiki_structure(raw)
        assert result is None

    def test_returns_none_for_no_json_object(self):
        result = parse_wiki_structure("I could not produce a JSON structure.")
        assert result is None

    def test_returns_none_for_missing_required_fields(self):
        raw = '{"title": "only title"}'
        result = parse_wiki_structure(raw)
        assert result is None

    def test_returns_none_for_pages_not_a_list(self):
        raw = '{"id":"r","title":"T","description":"d","pages":"not a list"}'
        result = parse_wiki_structure(raw)
        assert result is None

    def test_fills_missing_optional_page_fields(self):
        raw = '{"id":"r","title":"T","description":"d","pages":[{"id":"p1","title":"Overview"}]}'
        result = parse_wiki_structure(raw)
        assert result is not None
        page = result["pages"][0]
        assert page["content"] == ""
        assert page["filePaths"] == []
        assert page["importance"] == "medium"
        assert page["relatedPages"] == []

    def test_returns_none_for_empty_string(self):
        result = parse_wiki_structure("")
        assert result is None


# ---------------------------------------------------------------------------
# _validate_wiki_structure
# ---------------------------------------------------------------------------


class TestValidateWikiStructure:
    def test_valid_minimal_structure(self):
        data = {"id": "r", "title": "T", "description": "d", "pages": [{"id": "p1", "title": "P"}]}
        assert _validate_wiki_structure(data) is True

    def test_invalid_non_dict(self):
        assert _validate_wiki_structure([]) is False  # type: ignore[arg-type]

    def test_invalid_missing_pages(self):
        data = {"id": "r", "title": "T", "description": "d"}
        assert _validate_wiki_structure(data) is False

    def test_invalid_pages_not_list(self):
        data = {"id": "r", "title": "T", "description": "d", "pages": {}}
        assert _validate_wiki_structure(data) is False

    def test_invalid_page_missing_id(self):
        data = {"id": "r", "title": "T", "description": "d", "pages": [{"title": "No ID"}]}
        assert _validate_wiki_structure(data) is False

    def test_invalid_page_not_dict(self):
        data = {"id": "r", "title": "T", "description": "d", "pages": ["not a dict"]}
        assert _validate_wiki_structure(data) is False


# ---------------------------------------------------------------------------
# _flatten_pages_in_section_order
# ---------------------------------------------------------------------------


class TestFlattenPagesInSectionOrder:
    def test_no_sections_returns_pages_as_is(self):
        pages = [_make_page("p1"), _make_page("p2"), _make_page("p3")]
        structure = _make_structure(pages)
        result = _flatten_pages_in_section_order(structure)
        assert [p["id"] for p in result] == ["p1", "p2", "p3"]

    def test_sections_order_respected(self):
        pages = [_make_page("p1"), _make_page("p2"), _make_page("p3")]
        sections = [
            {"id": "s1", "title": "First", "pages": ["p2"], "subsections": []},
            {"id": "s2", "title": "Second", "pages": ["p1"], "subsections": []},
        ]
        structure = _make_structure(pages, sections=sections, root_sections=["s1", "s2"])
        result = _flatten_pages_in_section_order(structure)
        # p2 comes before p1 because s1 is listed first in rootSections
        assert result[0]["id"] == "p2"
        assert result[1]["id"] == "p1"
        # p3 is orphan — appended last
        assert result[2]["id"] == "p3"

    def test_subsections_traversed_recursively(self):
        pages = [_make_page("p1"), _make_page("p2"), _make_page("p3")]
        sections = [
            {"id": "s1", "title": "Root", "pages": ["p1"], "subsections": ["s2"]},
            {"id": "s2", "title": "Child", "pages": ["p2"], "subsections": []},
        ]
        structure = _make_structure(pages, sections=sections, root_sections=["s1"])
        result = _flatten_pages_in_section_order(structure)
        assert [p["id"] for p in result[:3]] == ["p1", "p2", "p3"]

    def test_duplicate_page_refs_deduplicated(self):
        pages = [_make_page("p1"), _make_page("p2")]
        sections = [
            {"id": "s1", "title": "A", "pages": ["p1", "p1"], "subsections": []},
            {"id": "s2", "title": "B", "pages": ["p2"], "subsections": []},
        ]
        structure = _make_structure(pages, sections=sections, root_sections=["s1", "s2"])
        result = _flatten_pages_in_section_order(structure)
        ids = [p["id"] for p in result]
        assert ids.count("p1") == 1
        assert ids.count("p2") == 1

    def test_empty_pages_returns_empty(self):
        structure = _make_structure([])
        result = _flatten_pages_in_section_order(structure)
        assert result == []

    def test_empty_root_sections_falls_back_to_pages(self):
        pages = [_make_page("p1"), _make_page("p2")]
        structure = _make_structure(pages, root_sections=[])
        result = _flatten_pages_in_section_order(structure)
        assert [p["id"] for p in result] == ["p1", "p2"]


# ---------------------------------------------------------------------------
# _compute_repo_name
# ---------------------------------------------------------------------------


class TestComputeRepoName:
    def _req(self, url: str, repo_type: str = "github") -> AgentWikiRequest:
        return AgentWikiRequest(repo_url=url, type=repo_type)

    def test_github_url(self):
        req = self._req("https://github.com/owner/my-repo")
        assert _compute_repo_name(req) == "owner_my-repo"

    def test_gitlab_url(self):
        req = self._req("https://gitlab.com/org/project", "gitlab")
        assert _compute_repo_name(req) == "org_project"

    def test_strips_dot_git(self):
        req = self._req("https://github.com/alice/repo.git")
        assert _compute_repo_name(req) == "alice_repo"

    def test_trailing_slash_stripped(self):
        req = self._req("https://github.com/owner/repo/")
        assert _compute_repo_name(req) == "owner_repo"

    def test_unknown_type_uses_last_segment(self):
        req = self._req("https://example.com/somerepo", "local")
        assert _compute_repo_name(req) == "somerepo"


# ---------------------------------------------------------------------------
# Agent config registration
# ---------------------------------------------------------------------------


class TestAgentConfigRegistration:
    def test_wiki_planner_registered(self):
        config = get_agent_config("wiki-planner")
        assert config.name == "wiki-planner"
        assert "grep" in config.allowed_tools
        assert "glob" in config.allowed_tools
        assert "ls" in config.allowed_tools
        assert "read" in config.allowed_tools
        assert "bash" not in config.allowed_tools
        assert "task" not in config.allowed_tools
        assert config.max_steps == 20

    def test_wiki_writer_registered(self):
        config = get_agent_config("wiki-writer")
        assert config.name == "wiki-writer"
        assert "grep" in config.allowed_tools
        assert "glob" in config.allowed_tools
        assert "ls" in config.allowed_tools
        assert "read" in config.allowed_tools
        assert "bash" in config.allowed_tools
        assert "task" not in config.allowed_tools
        assert config.max_steps == 25

    def test_wiki_planner_prompt_has_required_placeholders(self):
        config = get_agent_config("wiki-planner")
        tmpl = config.system_prompt_template
        for placeholder in ("{repo_type}", "{repo_url}", "{repo_name}", "{language_name}", "{comprehensive_instruction}"):
            assert placeholder in tmpl, f"Missing placeholder: {placeholder}"

    def test_wiki_writer_prompt_has_required_placeholders(self):
        config = get_agent_config("wiki-writer")
        tmpl = config.system_prompt_template
        for placeholder in ("{repo_type}", "{repo_url}", "{repo_name}", "{language_name}"):
            assert placeholder in tmpl, f"Missing placeholder: {placeholder}"
        # writer must NOT have comprehensive_instruction (different vars per agent)
        assert "{comprehensive_instruction}" not in tmpl
