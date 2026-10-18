"""One SQLite file for everything: documents, chunks, BM25, vectors, graph, jobs.

WAL + one connection per thread. Writers stay short; the indexing job is the only
long writer and the job runner allows one at a time.

shortcut: SQLite is single-writer and vec0 is brute-force KNN (~40ms @100k chunks,
~400ms @1M). Past ~1M chunks, swap graph_rag/retrieval/vectors.py for LanceDB.
"""

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import sqlite_vec

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY,
  filename TEXT NOT NULL,
  content_sha TEXT NOT NULL UNIQUE,
  char_count INTEGER NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chunks (
  id TEXT PRIMARY KEY,
  document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  idx INTEGER NOT NULL,
  text TEXT NOT NULL,
  token_count INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_document ON chunks(document_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  text, content='chunks', content_rowid='rowid', tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
  INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
END;

CREATE TABLE IF NOT EXISTS entities (
  name TEXT PRIMARY KEY,
  type TEXT NOT NULL DEFAULT 'CONCEPT',
  description TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS edges (
  src TEXT NOT NULL REFERENCES entities(name) ON DELETE CASCADE,
  dst TEXT NOT NULL REFERENCES entities(name) ON DELETE CASCADE,
  rel TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  weight REAL NOT NULL DEFAULT 1.0,
  PRIMARY KEY (src, dst, rel)
);

CREATE TABLE IF NOT EXISTS entity_mentions (
  name TEXT NOT NULL REFERENCES entities(name) ON DELETE CASCADE,
  chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
  description TEXT NOT NULL DEFAULT '',
  PRIMARY KEY (name, chunk_id)
);

CREATE TABLE IF NOT EXISTS edge_mentions (
  src TEXT NOT NULL,
  dst TEXT NOT NULL,
  rel TEXT NOT NULL,
  chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
  description TEXT NOT NULL DEFAULT '',
  weight REAL NOT NULL DEFAULT 1.0,
  PRIMARY KEY (src, dst, rel, chunk_id)
);

CREATE VIEW IF NOT EXISTS adjacency AS
  SELECT src AS u, dst AS v FROM edges
  UNION
  SELECT dst AS u, src AS v FROM edges;

CREATE VIRTUAL TABLE IF NOT EXISTS vec_items USING vec0(
  kind TEXT, ref TEXT, embedding float[384]
);

CREATE TABLE IF NOT EXISTS communities (
  id INTEGER PRIMARY KEY,
  title TEXT NOT NULL,
  summary TEXT NOT NULL,
  findings TEXT NOT NULL DEFAULT '[]',
  rating REAL NOT NULL DEFAULT 5.0,
  members_hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS communities_hash ON communities(members_hash);

CREATE TABLE IF NOT EXISTS community_members (
  community_id INTEGER NOT NULL REFERENCES communities(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  PRIMARY KEY (community_id, name)
);

CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  status TEXT NOT NULL,
  phase TEXT NOT NULL DEFAULT '',
  done INTEGER NOT NULL DEFAULT 0,
  total INTEGER NOT NULL DEFAULT 0,
  message TEXT NOT NULL DEFAULT '',
  error TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS extraction_cache (
  key TEXT PRIMARY KEY,
  payload TEXT NOT NULL
);
"""


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self.conn().executescript(SCHEMA)

    def conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=10, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            self._local.conn = conn
        return conn

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        conn = self.conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
