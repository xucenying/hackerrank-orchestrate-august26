"""Preflight - decide what actually needs rebuilding before spending money.

Two failure modes, deliberately different:
  schema / referential drift -> STOP the run. The data contract changed.
  content drift              -> rebuild quietly, only the affected items.

Caches are content-addressed, so invalidation is automatic and precise:
  media key = sha256(file bytes) + extractor version + model
  llm key   = sha256(system prompt + model + fully rendered per-message context)

Because the rendered context is derived from the CSVs, editing one user's row in
users.csv invalidates only that user's messages - no dependency graph needed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

MEDIA_EXTRACTOR_VERSION = "1"
LLM_PROMPT_VERSION = "1"


def sha(*parts: str | bytes) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else p.encode("utf-8"))
    return h.hexdigest()


def media_key(path: Path, model: str) -> str:
    return sha(path.read_bytes(), MEDIA_EXTRACTOR_VERSION, model)


def llm_key(system_prompt: str, model: str, rendered: str) -> str:
    return sha(system_prompt, model, LLM_PROMPT_VERSION, rendered)


class Cache:
    """Tiny content-addressed JSON store."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data: dict = {}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.data = {}

    def get(self, key: str):
        return self.data.get(key)

    def put(self, key: str, value) -> None:
        self.data[key] = value

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1, sort_keys=True), encoding="utf-8")


@dataclass
class Report:
    dataset_ok: bool
    problems: list[str] = field(default_factory=list)
    media_total: int = 0
    media_stale: list[tuple[str, str]] = field(default_factory=list)
    llm_total: int = 0
    llm_stale: int = 0
    est_cost_usd: float = 0.0

    def render(self) -> str:
        lines = ["PREFLIGHT"]
        lines.append(f"  dataset ......... {'schema OK, joins OK' if self.dataset_ok else 'FAILED'}")
        for p in self.problems[:10]:
            lines.append(f"                    ! {p}")
        lines.append("  address book .... rebuilt from CSV (never cached)")
        fresh_media = self.media_total - len(self.media_stale)
        lines.append(f"  media cache ..... {fresh_media} of {self.media_total} fresh")
        for mid, why in self.media_stale[:8]:
            lines.append(f"                    ~ {mid:<10} {why}")
        fresh_llm = self.llm_total - self.llm_stale
        lines.append(f"  llm cache ....... {fresh_llm} of {self.llm_total} fresh")
        if self.llm_stale:
            lines.append(f"                    ~ {self.llm_stale} message(s) need classification")
        lines.append(f"  estimated cost .. ${self.est_cost_usd:.2f}")
        return "\n".join(lines)


def check_media(ds, cache: Cache, model: str) -> tuple[int, list[tuple[str, str]]]:
    """Which media files need (re)extraction, and why."""
    used = {r["media_id"] for r in ds.messages + ds.samples + ds.history if r.get("media_id")}
    stale: list[tuple[str, str]] = []
    for mid in sorted(used):
        path = ds.media_file(mid)
        if path is None or not path.exists():
            stale.append((mid, "file missing on disk"))
            continue
        key = media_key(path, model)
        if cache.get(key) is None:
            known = any(v.get("media_id") == mid for v in cache.data.values() if isinstance(v, dict))
            stale.append((mid, "content changed" if known else "new file"))
    return len(used), stale


def estimate_cost(n_llm_calls: int, avg_in: int = 6000, avg_out: int = 900) -> float:
    """claude-opus-5 at $5/MTok in, $25/MTok out, assuming most input is cache-read."""
    cached_share = 0.65
    in_full = avg_in * (1 - cached_share) * n_llm_calls
    in_cached = avg_in * cached_share * n_llm_calls
    out = avg_out * n_llm_calls
    return (in_full * 5 + in_cached * 0.5 + out * 25) / 1_000_000
