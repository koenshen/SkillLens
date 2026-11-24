"""BIRD execution accuracy for materialized SQLite tasks."""
from __future__ import annotations

import re
import sqlite3
import time
from contextlib import closing
from pathlib import Path


def database_path(root: Path, db_id: str) -> Path:
    if not db_id or db_id in {".", ".."} or db_id != Path(db_id).name:
        raise ValueError(f"Invalid BIRD database ID: {db_id!r}")
    return root / db_id / f"{db_id}.sqlite"


def extract_sql(content: str) -> str:
    if " response" in content:
        content = content.rsplit(" response", 1)[1]
    fenced = re.search(r"```(?:sql|sqlite|mysql|postgresql)?\s*([\s\S]*?)```", content, re.I)
    if fenced:
        content = fenced.group(1)
    start = re.search(r"(?im)^\s*(?:SELECT|WITH|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\b", content)
    if start:
        content = content[start.start():]
    content = content.strip()
    quote = None
    i = 0
    while i < len(content):
        char = content[i]
        if quote:
            if char == quote:
                if i + 1 < len(content) and content[i + 1] == quote:
                    i += 2
                    continue
                quote = None
        elif char in ("'", '"', "`"):
            quote = char
        elif char == ";":
            return content[:i + 1].strip()
        i += 1
    return content


# Only result-producing reads belong in an EX evaluation. Opening the main
# DB read-only alone would still permit attaching a second, writable database.
_READ_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}


def _authorize(action, _arg1, _arg2, _database, _trigger):
    return sqlite3.SQLITE_OK if action in _READ_ACTIONS else sqlite3.SQLITE_DENY


def score_response(response: str, gold_sql: str, database: Path, timeout: float = 30.0) -> dict:
    predicted = extract_sql(response)
    result = {"predicted_sql": predicted, "ex": 0, "exec_status": "error", "exec_error": ""}
    if not predicted:
        result["exec_error"] = "empty SQL"
        return result
    deadline = time.monotonic() + timeout
    timed_out = False

    def check_deadline():
        nonlocal timed_out
        timed_out = time.monotonic() > deadline
        return int(timed_out)

    stage = "connect"
    try:
        uri = database.resolve(strict=True).as_uri() + "?mode=ro&immutable=1"
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA query_only=ON")
            conn.set_authorizer(_authorize)
            conn.set_progress_handler(check_deadline, 1000)
            stage = "predicted SQL"
            predicted_rows = set(conn.execute(predicted))
            stage = "gold SQL"
            gold_rows = set(conn.execute(gold_sql))
        result.update(ex=int(predicted_rows == gold_rows), exec_status="ok")
    except (sqlite3.Error, OSError, ValueError, OverflowError) as exc:
        result.update(exec_status="timeout" if timed_out else "error",
                      exec_error=f"{stage}: {exc}")
    return result
