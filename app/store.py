import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np


class Store:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "documents.sqlite3"
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY, filename TEXT NOT NULL, created_at TEXT NOT NULL,
                embedding_model TEXT NOT NULL, chunks TEXT NOT NULL, embeddings TEXT NOT NULL
            )""")

            db.execute("""CREATE TABLE IF NOT EXISTS csv_tables (
                document_id TEXT PRIMARY KEY, content TEXT NOT NULL
            )""")

    def add_csv(self, filename, table):
        identifier = str(uuid4())
        with self.connect() as db:
            db.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?)",
                (identifier, filename, datetime.now(UTC).isoformat(), "structured-csv", "[]", "[]"),
            )
            db.execute("INSERT INTO csv_tables VALUES (?, ?)", (identifier, json.dumps(table)))
        return self.get(identifier)[0]

    def csv_table(self, identifier):
        with self.connect() as db:
            row = db.execute(
                "SELECT content FROM csv_tables WHERE document_id = ?", (identifier,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def metadata(row):
        return {
            "id": row[0],
            "filename": row[1],
            "created_at": row[2],
            "embedding_model": row[3],
            "chunk_count": len(json.loads(row[4])),
        }

    def add(self, filename, model, chunks, embeddings):
        identifier = str(uuid4())
        with self.connect() as db:
            db.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?)",
                (
                    identifier,
                    filename,
                    datetime.now(UTC).isoformat(),
                    model,
                    json.dumps(chunks),
                    json.dumps(embeddings),
                ),
            )
        return self.get(identifier)[0]

    def list(self):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
        return [self.metadata(row) for row in rows]

    def get(self, identifier):
        with self.connect() as db:
            row = db.execute("SELECT * FROM documents WHERE id = ?", (identifier,)).fetchone()
        if row is None:
            raise KeyError(identifier)
        return self.metadata(row), json.loads(row[4]), json.loads(row[5])

    def delete(self, identifier):
        with self.connect() as db:
            db.execute("DELETE FROM csv_tables WHERE document_id = ?", (identifier,))
            return db.execute("DELETE FROM documents WHERE id = ?", (identifier,)).rowcount > 0

    def search(self, identifier, query_vector, model, limit):
        metadata, chunks, embeddings = self.get(identifier)
        if metadata["embedding_model"] != model:
            raise ValueError("Embedding model changed; re-upload this document")
        matrix = np.asarray(embeddings, dtype=float)
        query = np.asarray(query_vector, dtype=float)
        if matrix.shape[1] != query.shape[0]:
            raise ValueError("Embedding dimensions changed; re-upload this document")
        scores = (matrix @ query) / (
            np.maximum(np.linalg.norm(matrix, axis=1) * np.linalg.norm(query), 1e-12)
        )
        return [{**chunks[int(i)], "score": float(scores[i])} for i in np.argsort(-scores)[:limit]]
