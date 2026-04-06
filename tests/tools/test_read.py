"""Tests for api.tools.read -- ReadTool."""

import os

import pytest

from api.tools.read import ReadTool, _is_binary_file


@pytest.fixture()
def read(repo_path):
    return ReadTool(repo_path)


# ---------------------------------------------------------------------------
# Basic file reading
# ---------------------------------------------------------------------------

class TestReadFile:

    @pytest.mark.asyncio
    async def test_reads_file_with_line_numbers(self, read, fake_repo):
        result = await read.execute({"file_path": str(fake_repo / "src" / "main.py")})
        assert "def main" in result.output
        assert "1:" in result.output  # line number prefix
        assert "<type>file</type>" in result.output

    @pytest.mark.asyncio
    async def test_reads_relative_path(self, read):
        result = await read.execute({"file_path": "src/main.py"})
        assert "def main" in result.output

    @pytest.mark.asyncio
    async def test_end_of_file_marker(self, read, fake_repo):
        result = await read.execute({"file_path": str(fake_repo / "README.md")})
        assert "End of file" in result.output


# ---------------------------------------------------------------------------
# Offset and limit
# ---------------------------------------------------------------------------

class TestReadOffsetLimit:

    @pytest.mark.asyncio
    async def test_offset_skips_lines(self, read, fake_repo):
        result = await read.execute({
            "file_path": str(fake_repo / "src" / "main.py"),
            "offset": 2,
        })
        # Line 1 ("def main") should NOT appear at start
        assert "2:" in result.output
        assert not result.output.split("<content>")[1].startswith("\n1:")

    @pytest.mark.asyncio
    async def test_limit_restricts_lines(self, read, fake_repo):
        result = await read.execute({
            "file_path": str(fake_repo / "src" / "main.py"),
            "limit": 2,
        })
        assert result.metadata["truncated"] is True
        assert "Use offset=" in result.output

    @pytest.mark.asyncio
    async def test_offset_out_of_range(self, read, fake_repo):
        result = await read.execute({
            "file_path": str(fake_repo / "README.md"),
            "offset": 9999,
        })
        assert result.metadata.get("error") == "offset_out_of_range"

    @pytest.mark.asyncio
    async def test_invalid_offset_rejected(self, read):
        result = await read.execute({"file_path": "src/main.py", "offset": 0})
        assert result.metadata.get("error") == "invalid_offset"

    @pytest.mark.asyncio
    async def test_invalid_limit_rejected(self, read):
        result = await read.execute({"file_path": "src/main.py", "limit": 0})
        assert result.metadata.get("error") == "invalid_limit"


# ---------------------------------------------------------------------------
# Directory mode
# ---------------------------------------------------------------------------

class TestReadDirectory:

    @pytest.mark.asyncio
    async def test_lists_directory_entries(self, read, fake_repo):
        result = await read.execute({"file_path": str(fake_repo / "src")})
        assert "<type>directory</type>" in result.output
        assert "main.py" in result.output
        assert "utils.py" in result.output
        assert "components/" in result.output

    @pytest.mark.asyncio
    async def test_directory_with_offset(self, read, fake_repo):
        result = await read.execute({
            "file_path": str(fake_repo / "src"),
            "offset": 2,
            "limit": 1,
        })
        assert result.metadata["truncated"] is True


# ---------------------------------------------------------------------------
# Binary file detection
# ---------------------------------------------------------------------------

class TestReadBinary:

    def test_binary_extension_detected(self, tmp_path):
        bin_file = tmp_path / "image.png"
        bin_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        assert _is_binary_file(str(bin_file)) is True

    def test_text_file_not_binary(self, tmp_path):
        txt_file = tmp_path / "hello.txt"
        txt_file.write_text("Hello, world!")
        assert _is_binary_file(str(txt_file)) is False

    def test_null_bytes_detected_as_binary(self, tmp_path):
        bin_file = tmp_path / "data.custom"
        bin_file.write_bytes(b"some text\x00more text\x00\x00\x00")
        assert _is_binary_file(str(bin_file)) is True

    @pytest.mark.asyncio
    async def test_binary_file_returns_error(self, tmp_path):
        bin_file = tmp_path / "data.zip"
        bin_file.write_bytes(b"PK\x03\x04" + b"\x00" * 100)
        tool = ReadTool(str(tmp_path))
        result = await tool.execute({"file_path": str(bin_file)})
        assert result.metadata.get("error") == "binary_file"

    def test_empty_file_not_binary(self, tmp_path):
        empty = tmp_path / "empty.unknown"
        empty.write_bytes(b"")
        assert _is_binary_file(str(empty)) is False


# ---------------------------------------------------------------------------
# File not found and suggestions
# ---------------------------------------------------------------------------

class TestReadNotFound:

    @pytest.mark.asyncio
    async def test_file_not_found(self, read, fake_repo):
        result = await read.execute({
            "file_path": str(fake_repo / "src" / "nonexistent.py"),
        })
        assert result.metadata.get("error") == "not_found"
        assert "File not found" in result.output

    @pytest.mark.asyncio
    async def test_suggests_similar_files(self, read, fake_repo):
        # _suggest_similar checks bidirectional substring: base in item or item in base
        # "main" is a substring of "main.py", so searching for "main" should suggest main.py
        result = await read.execute({
            "file_path": str(fake_repo / "src" / "main"),
        })
        assert result.metadata.get("error") == "not_found"
        assert "main.py" in result.output or "Did you mean" in result.output


# ---------------------------------------------------------------------------
# Path traversal
# ---------------------------------------------------------------------------

class TestReadPathTraversal:

    @pytest.mark.asyncio
    async def test_path_traversal_blocked(self, read, fake_repo):
        result = await read.execute({
            "file_path": str(fake_repo / ".." / "escape" / "secrets.txt"),
        })
        assert result.metadata.get("error") == "path_traversal"

    @pytest.mark.asyncio
    async def test_absolute_path_within_repo_works(self, read, fake_repo):
        result = await read.execute({
            "file_path": str(fake_repo / "src" / "main.py"),
        })
        assert "def main" in result.output


# ---------------------------------------------------------------------------
# Long line truncation
# ---------------------------------------------------------------------------

class TestReadLongLines:

    @pytest.mark.asyncio
    async def test_long_line_truncated(self, tmp_path):
        long_file = tmp_path / "long.txt"
        long_file.write_text("x" * 5000 + "\n")
        tool = ReadTool(str(tmp_path))
        result = await tool.execute({"file_path": str(long_file)})
        assert "line truncated" in result.output.lower()


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

class TestReadParams:

    @pytest.mark.asyncio
    async def test_missing_file_path(self, read):
        result = await read.execute({})
        assert result.metadata.get("error") == "missing_param"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestReadSchema:

    def test_function_schema_structure(self, read):
        schema = read.to_function_schema()
        assert schema["function"]["name"] == "read"
        props = schema["function"]["parameters"]["properties"]
        assert "file_path" in props
        assert "offset" in props
        assert "limit" in props
        assert "file_path" in schema["function"]["parameters"]["required"]
