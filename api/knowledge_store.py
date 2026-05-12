"""SQLite-backed repository knowledge graph shared by RAG and PR analysis."""

from __future__ import annotations

import json
import logging
import os
import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from adalflow.utils import get_adalflow_default_root_path

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeChunk:
    chunk_id: str
    repo_id: str
    rel_path: str
    language: str
    chunk_type: str
    symbol_name: str
    qualified_name: str
    parent_symbol: str
    start_line: int
    end_line: int
    content: str


class RepoKnowledgeStore:
    """Small SQLite store for code chunks, graph edges, symbols, refs and analysis sessions."""

    def __init__(self, db_path: str | Path | None = None):
        if db_path is None:
            root = Path(get_adalflow_default_root_path())
            db_path = root / "knowledge" / "metadata.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_tables()

    def _init_tables(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                repo_id TEXT,
                rel_path TEXT,
                language TEXT,
                chunk_type TEXT,
                symbol_name TEXT,
                qualified_name TEXT,
                parent_symbol TEXT,
                start_line INTEGER,
                end_line INTEGER,
                content TEXT
            );
            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                repo_id TEXT,
                source TEXT,
                target TEXT,
                relation TEXT
            );
            CREATE TABLE IF NOT EXISTS symbols (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                repo_id TEXT,
                symbol_id TEXT,
                name TEXT,
                symbol_type TEXT,
                file_path TEXT,
                module TEXT,
                start_line INTEGER,
                end_line INTEGER,
                code_hash TEXT,
                signature TEXT,
                runtime_tags TEXT
            );
            CREATE TABLE IF NOT EXISTS refs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                repo_id TEXT,
                file_path TEXT,
                source_scope TEXT,
                ref_type TEXT,
                target TEXT,
                ref_name TEXT,
                line INTEGER,
                snippet TEXT
            );
            CREATE TABLE IF NOT EXISTS repos (
                repo_id TEXT PRIMARY KEY,
                name TEXT,
                path TEXT,
                chunk_count INTEGER,
                status TEXT
            );
            CREATE TABLE IF NOT EXISTS analysis_sessions (
                session_id TEXT PRIMARY KEY,
                repo_id TEXT,
                base TEXT,
                head TEXT,
                data TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_repo ON chunks(repo_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(repo_id, rel_path);
            CREATE INDEX IF NOT EXISTS idx_chunks_symbol ON chunks(repo_id, symbol_name);
            CREATE INDEX IF NOT EXISTS idx_chunks_qualified ON chunks(repo_id, qualified_name);
            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(repo_id, source);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(repo_id, target);
            CREATE INDEX IF NOT EXISTS idx_symbols_repo_symbol ON symbols(repo_id, symbol_id);
            CREATE INDEX IF NOT EXISTS idx_symbols_repo_file ON symbols(repo_id, file_path);
            CREATE INDEX IF NOT EXISTS idx_refs_target ON refs(repo_id, target);
            CREATE INDEX IF NOT EXISTS idx_refs_source ON refs(repo_id, source_scope);
            CREATE INDEX IF NOT EXISTS idx_refs_type_target ON refs(repo_id, ref_type, target);
            """
        )
        self.conn.commit()

    def save_repo(self, repo_id: str, name: str, path: str, chunk_count: int, status: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO repos VALUES (?,?,?,?,?)",
            (repo_id, name, path, chunk_count, status),
        )
        self.conn.commit()

    def get_repo(self, repo_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT repo_id, name, path, chunk_count, status FROM repos WHERE repo_id=?",
            (repo_id,),
        ).fetchone()
        if not row:
            return None
        return {
            "repo_id": row[0],
            "name": row[1],
            "path": row[2],
            "chunk_count": int(row[3] or 0),
            "status": row[4] or "unknown",
        }

    def list_symbols(self, repo_id: str, query: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit or 100), 500))
        if query:
            q = f"%{query}%"
            rows = self.conn.execute(
                """SELECT symbol_id, name, symbol_type, file_path, module, start_line,
                          end_line, signature, runtime_tags
                   FROM symbols
                   WHERE repo_id=? AND (symbol_id LIKE ? OR name LIKE ? OR file_path LIKE ?)
                   ORDER BY file_path, start_line
                   LIMIT ?""",
                (repo_id, q, q, q, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """SELECT symbol_id, name, symbol_type, file_path, module, start_line,
                          end_line, signature, runtime_tags
                   FROM symbols
                   WHERE repo_id=?
                   ORDER BY file_path, start_line
                   LIMIT ?""",
                (repo_id, limit),
            ).fetchall()
        return [self._symbol_row_to_dict(row) for row in rows]

    def list_references(
        self,
        repo_id: str,
        *,
        source_scope: str | None = None,
        target: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit or 100), 500))
        clauses = ["repo_id=?"]
        params: list[Any] = [repo_id]
        if source_scope:
            clauses.append("source_scope LIKE ?")
            params.append(f"%{source_scope}%")
        if target:
            clauses.append("target LIKE ?")
            params.append(f"%{target}%")
        params.append(limit)
        rows = self.conn.execute(
            f"""SELECT file_path, source_scope, ref_type, target, ref_name, line, snippet
                FROM refs
                WHERE {' AND '.join(clauses)}
                ORDER BY file_path, line
                LIMIT ?""",
            params,
        ).fetchall()
        return [self._ref_row_to_dict(row) for row in rows]

    def list_edges(self, repo_id: str, symbol: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit or 100), 500))
        if symbol:
            q = f"%{symbol}%"
            rows = self.conn.execute(
                """SELECT source, target, relation
                   FROM edges
                   WHERE repo_id=? AND (source LIKE ? OR target LIKE ?)
                   ORDER BY relation, source, target
                   LIMIT ?""",
                (repo_id, q, q, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """SELECT source, target, relation
                   FROM edges
                   WHERE repo_id=?
                   ORDER BY relation, source, target
                   LIMIT ?""",
                (repo_id, limit),
            ).fetchall()
        return [
            {
                "source": row[0],
                "target": row[1],
                "relation": row[2],
            }
            for row in rows
        ]

    def replace_chunks(self, repo_id: str, chunks: Iterable[KnowledgeChunk]) -> None:
        rows = [
            (
                chunk.chunk_id,
                chunk.repo_id,
                chunk.rel_path,
                chunk.language,
                chunk.chunk_type,
                chunk.symbol_name,
                chunk.qualified_name,
                chunk.parent_symbol,
                chunk.start_line,
                chunk.end_line,
                chunk.content,
            )
            for chunk in chunks
        ]
        self.conn.execute("DELETE FROM chunks WHERE repo_id=?", (repo_id,))
        self.conn.executemany(
            """INSERT OR REPLACE INTO chunks
               (chunk_id, repo_id, rel_path, language, chunk_type, symbol_name,
                qualified_name, parent_symbol, start_line, end_line, content)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        self.conn.commit()

    def replace_edges(self, repo_id: str, edges: Iterable[dict[str, str]]) -> None:
        rows = [
            (repo_id, edge["source"], edge["target"], edge["relation"])
            for edge in edges
            if edge.get("source") and edge.get("target") and edge.get("relation")
        ]
        self.conn.execute("DELETE FROM edges WHERE repo_id=?", (repo_id,))
        self.conn.executemany(
            "INSERT INTO edges (repo_id, source, target, relation) VALUES (?,?,?,?)",
            rows,
        )
        self.conn.commit()

    def clear_code_graph(self, repo_id: str) -> None:
        self.conn.execute("DELETE FROM symbols WHERE repo_id=?", (repo_id,))
        self.conn.execute("DELETE FROM refs WHERE repo_id=?", (repo_id,))
        self.conn.commit()

    def save_symbols(self, repo_id: str, symbols: list[Any]) -> None:
        self.conn.executemany(
            """INSERT INTO symbols
               (repo_id, symbol_id, name, symbol_type, file_path, module, start_line,
                end_line, code_hash, signature, runtime_tags)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            [
                (
                    repo_id,
                    symbol.id,
                    symbol.name,
                    symbol.symbol_type,
                    symbol.file_path,
                    symbol.module,
                    symbol.start_line,
                    symbol.end_line,
                    symbol.code_hash,
                    symbol.signature,
                    json.dumps(symbol.runtime_tags),
                )
                for symbol in symbols
            ],
        )
        self.conn.commit()

    def save_references(self, repo_id: str, references: list[Any]) -> None:
        self.conn.executemany(
            """INSERT INTO refs
               (repo_id, file_path, source_scope, ref_type, target, ref_name, line, snippet)
               VALUES (?,?,?,?,?,?,?,?)""",
            [
                (
                    repo_id,
                    ref.file_path,
                    ref.source_scope,
                    ref.ref_type,
                    ref.target,
                    ref.ref_name,
                    ref.line,
                    ref.snippet,
                )
                for ref in references
            ],
        )
        self.conn.commit()

    def get_chunk(self, chunk_id: str) -> KnowledgeChunk | None:
        row = self.conn.execute("SELECT * FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone()
        return self._row_to_chunk(row) if row else None

    def search_exact(self, repo_id: str, query: str, limit: int = 10) -> list[KnowledgeChunk]:
        q = f"%{query}%"
        rows = self.conn.execute(
            """SELECT * FROM chunks WHERE repo_id=? AND
               (symbol_name LIKE ? OR rel_path LIKE ? OR qualified_name LIKE ?)
               LIMIT ?""",
            (repo_id, q, q, q, limit),
        ).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def get_neighbors(self, repo_id: str, symbol: str, limit: int = 5) -> list[KnowledgeChunk]:
        if not symbol:
            return []
        edge_rows = self.conn.execute(
            """SELECT source, target FROM edges
               WHERE repo_id=? AND (source LIKE ? OR target LIKE ?) LIMIT 40""",
            (repo_id, f"%{symbol}%", f"%{symbol}%"),
        ).fetchall()
        related = set()
        for source, target in edge_rows:
            related.add(source)
            related.add(target)
        related.discard(symbol)

        chunks: list[KnowledgeChunk] = []
        for name in list(related)[:limit]:
            rows = self.conn.execute(
                """SELECT * FROM chunks WHERE repo_id=? AND
                   (symbol_name=? OR qualified_name=? OR qualified_name LIKE ?)
                   LIMIT 2""",
                (repo_id, name, name, f"%{name}"),
            ).fetchall()
            chunks.extend(self._row_to_chunk(row) for row in rows)
        return chunks[:limit]

    def find_references_to(self, repo_id: str, target: str, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """SELECT file_path, source_scope, ref_type, target, ref_name, line, snippet
               FROM refs WHERE repo_id=? AND target=? LIMIT ?""",
            (repo_id, target, limit),
        ).fetchall()
        return [self._ref_row_to_dict(row) for row in rows]

    def find_references_by_prefix(self, repo_id: str, target_prefix: str, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """SELECT file_path, source_scope, ref_type, target, ref_name, line, snippet
               FROM refs WHERE repo_id=? AND target LIKE ? LIMIT ?""",
            (repo_id, f"{target_prefix}%", limit),
        ).fetchall()
        return [self._ref_row_to_dict(row) for row in rows]

    def get_reference_neighbors(self, repo_id: str, target: str, limit: int = 10) -> list[dict[str, Any]]:
        refs = self.find_references_to(repo_id, target, limit=limit)
        if not refs:
            refs = self.find_references_by_prefix(repo_id, f"{target}.", limit=limit)
        return refs

    def save_analysis_session(self, session_id: str, repo_id: str, base: str, head: str, data: dict[str, Any]) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO analysis_sessions
               (session_id, repo_id, base, head, data) VALUES (?,?,?,?,?)""",
            (session_id, repo_id, base, head, json.dumps(data, ensure_ascii=False)),
        )
        self.conn.commit()

    def get_analysis_session(self, session_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT session_id, repo_id, base, head, data, created_at FROM analysis_sessions WHERE session_id=?",
            (session_id,),
        ).fetchone()
        if not row:
            return None
        return {
            "session_id": row[0],
            "repo_id": row[1],
            "base": row[2],
            "head": row[3],
            "data": json.loads(row[4]) if row[4] else {},
            "created_at": row[5],
        }

    def close(self) -> None:
        self.conn.close()

    @staticmethod
    def _row_to_chunk(row: Any) -> KnowledgeChunk:
        return KnowledgeChunk(
            chunk_id=row[0],
            repo_id=row[1],
            rel_path=row[2],
            language=row[3],
            chunk_type=row[4],
            symbol_name=row[5] or "",
            qualified_name=row[6] or "",
            parent_symbol=row[7] or "",
            start_line=int(row[8] or 0),
            end_line=int(row[9] or 0),
            content=row[10] or "",
        )

    @staticmethod
    def _ref_row_to_dict(row: Any) -> dict[str, Any]:
        return {
            "file_path": row[0],
            "source_scope": row[1],
            "ref_type": row[2],
            "target": row[3],
            "ref_name": row[4],
            "line": row[5],
            "snippet": row[6] or "",
        }

    @staticmethod
    def _symbol_row_to_dict(row: Any) -> dict[str, Any]:
        return {
            "symbol_id": row[0],
            "name": row[1],
            "symbol_type": row[2],
            "file_path": row[3],
            "module": row[4],
            "start_line": int(row[5] or 0),
            "end_line": int(row[6] or 0),
            "signature": row[7] or "",
            "runtime_tags": json.loads(row[8]) if row[8] else [],
        }


def default_repo_id(repo_url_or_path: str) -> str:
    normalized = repo_url_or_path.strip().replace("\\", "/").rstrip("/")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def repo_display_name(repo_path: str) -> str:
    return os.path.basename(repo_path.rstrip("\\/")) or repo_path
