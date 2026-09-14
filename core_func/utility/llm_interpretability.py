"""Cached DeepSeek-based interpretability scoring for GP expressions."""

from __future__ import annotations

import json
import hashlib
import math
import fcntl
import os
import re
import sqlite3
import time
import urllib.error
import urllib.request
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a conservative quantitative equity researcher. Score only the "
    "interpretability and financial plausibility of a symbolic stock factor. "
    "Prefer short expressions with an understandable financial mechanism. "
    "Penalize arbitrary nesting, unstable divisions/logs, redundant transforms, "
    "and expressions with no economic intuition. Do not score historical "
    "performance. Reply with JSON only: {\"score\": <number 0-10>}."
)


def _connect(path: str) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS llm_interpretability (
        expression TEXT NOT NULL, model TEXT NOT NULL, score REAL NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (expression, model))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS llm_interpretability_audit (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        expression TEXT NOT NULL,
        model TEXT NOT NULL,
        outcome TEXT NOT NULL,
        score REAL,
        penalty REAL NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
    return conn


def _cached_score(cache_path: str, expression: str, model: str) -> float | None:
    with _connect(cache_path) as conn:
        row = conn.execute("SELECT score FROM llm_interpretability WHERE expression=? AND model=?", (expression, model)).fetchone()
    return float(row[0]) if row else None


def _store_score(cache_path: str, expression: str, model: str, score: float) -> None:
    for attempt in range(4):
        try:
            with _connect(cache_path) as conn:
                conn.execute("INSERT OR IGNORE INTO llm_interpretability(expression, model, score) VALUES (?, ?, ?)", (expression, model, score))
            return
        except sqlite3.OperationalError:
            time.sleep(0.1 * (attempt + 1))


def _record_audit(cache_path: str, expression: str, model: str,
                  outcome: str, score: float | None, penalty: float) -> None:
    """Persist one row for every candidate that enters LLM scoring."""
    for attempt in range(8):
        try:
            with _connect(cache_path) as conn:
                conn.execute(
                    "INSERT INTO llm_interpretability_audit"
                    "(expression, model, outcome, score, penalty) VALUES (?, ?, ?, ?, ?)",
                    (expression, model, outcome, score, penalty),
                )
            return
        except sqlite3.OperationalError:
            time.sleep(0.1 * (attempt + 1))


def _request_score(expression: str, model: str, base_url: str, api_key: str, timeout: float) -> float:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Factor expression:\n{expression}"},
        ],
        "temperature": 0,
        "max_tokens": 32,
        "response_format": {"type": "json_object"},
        # Interpretability scoring only needs a short deterministic JSON value.
        # DeepSeek thinking mode can otherwise consume the entire token budget
        # and return an empty content field.
        "thinking": {"type": "disabled"},
    }
    request = urllib.request.Request(f"{base_url.rstrip('/')}/chat/completions", data=json.dumps(payload).encode("utf-8"), headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response_data = json.loads(response.read().decode("utf-8"))
    usage_dir = os.getenv("GP_LLM_USAGE_DIR")
    if usage_dir:
        Path(usage_dir).mkdir(parents=True, exist_ok=True)
        record = {"request_id": response_data.get("id"), "model": response_data.get("model", model),
                  "usage": response_data.get("usage"), "timestamp": time.time()}
        with (Path(usage_dir) / f"usage_{os.getpid()}.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    content = response_data["choices"][0]["message"]["content"]
    try:
        score = float(json.loads(content)["score"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        match = re.search(r"(?:score\D*)?(10(?:\.0+)?|[0-9](?:\.\d+)?)", content)
        if not match:
            raise ValueError(f"unparseable LLM response: {content!r}")
        score = float(match.group(1))
    return min(10.0, max(0.0, score))


def _score_namespace(model: str, base_url: str) -> str:
    settings = [model, base_url.rstrip('/'), SYSTEM_PROMPT,
                'Factor expression:\n', 0, 32, 'json_object', 'thinking_disabled']
    return model + ':' + hashlib.sha256(json.dumps(settings).encode()).hexdigest()


def _shared_cache() -> str:
    return os.getenv('GP_LLM_SHARED_CACHE', str(
        Path(__file__).resolve().parents[2] / 'runs' / 'llm_cache' / 'scores.sqlite'))


def interpretability_penalty(expression: str, base_fitness: float, *, enabled: bool, min_fitness: float, penalty_weight: float, model: str, base_url: str, cache_path: str, timeout: float, api_key: str = "") -> tuple[float | None, float]:
    """Skip invalid numeric candidates; cache valid scores across runs.

    min_fitness is retained for compatibility, not used to bypass selection.
    Run-local score/audit tables remain available to existing reports.
    """
    if not enabled or penalty_weight == 0:
        return None, 0.0
    if not math.isfinite(base_fitness) or base_fitness < 0:
        _record_audit(cache_path, expression, model, 'invalid_fitness', None, 0.0)
        return None, 0.0
    namespace = _score_namespace(model, base_url)
    shared = _shared_cache()
    lock_dir = Path(shared).parent / 'locks'
    lock_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256((namespace + '\0' + expression).encode()).hexdigest()
    # Process-scoped flock is released automatically even if a worker crashes.
    with (lock_dir / key).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        score = _cached_score(shared, expression, namespace)
        outcome = 'cache_hit'
        if score is None:
            api_key = api_key or os.getenv('DEEPSEEK_API_KEY', '')
            if not api_key:
                _record_audit(cache_path, expression, model, 'no_api_key', None, 0.0)
                return None, 0.0
            try:
                score = _request_score(expression, model, base_url, api_key, timeout)
                _store_score(shared, expression, namespace, score)
                outcome = 'api_success'
            except (urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError):
                _record_audit(cache_path, expression, model, 'api_failure', None, 0.0)
                return None, 0.0
        _store_score(cache_path, expression, model, score)
    penalty = penalty_weight * (1.0 - score / 10.0)
    _record_audit(cache_path, expression, model, outcome, score, penalty)
    return score, penalty
