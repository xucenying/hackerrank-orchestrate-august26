"""Stage 7 - validate hard, then write output.csv.

Every assertion here is a contract from problem_statement.md. A failure raises
rather than writing a malformed file.
"""

from __future__ import annotations

import csv
from pathlib import Path

from classify import ACTIONS, TYPES

COLUMNS = ["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"]


class ContractError(RuntimeError):
    pass


def to_rows(results: list[tuple[str, object]], ds) -> list[dict]:
    out = []
    for message_id, verdict in results:
        ids = verdict.evidence_ids or []
        out.append(
            {
                "message_id": message_id,
                "action": verdict.action,
                "message_type": verdict.message_type,
                "reason": verdict.reason,
                "confidence": f"{verdict.confidence:.2f}",
                "evidence_message_ids": ";".join(ids) if ids else "none",
            }
        )
    return out


def validate(rows: list[dict], ds, expect_order: list[str] | None = None) -> None:
    problems: list[str] = []

    if expect_order is not None:
        got = [r["message_id"] for r in rows]
        if got != expect_order:
            if set(got) != set(expect_order):
                missing = set(expect_order) - set(got)
                extra = set(got) - set(expect_order)
                problems.append(f"id mismatch: missing={sorted(missing)[:5]} extra={sorted(extra)[:5]}")
            else:
                problems.append("rows are not in output.csv order")

    for r in rows:
        mid = r["message_id"]
        if r["action"] not in ACTIONS:
            problems.append(f"{mid}: bad action {r['action']!r}")
        if r["message_type"] not in TYPES:
            problems.append(f"{mid}: bad message_type {r['message_type']!r}")
        try:
            conf = float(r["confidence"])
            if not 0.0 <= conf <= 1.0:
                problems.append(f"{mid}: confidence {conf} out of range")
        except ValueError:
            problems.append(f"{mid}: non-numeric confidence {r['confidence']!r}")
        if not r["reason"].strip():
            problems.append(f"{mid}: empty reason")

        ev = r["evidence_message_ids"]
        if ev != "none":
            owner = _owner(mid, ds)
            for eid in ev.split(";"):
                if eid not in ds.history_by_id:
                    problems.append(f"{mid}: evidence {eid} not in message_history.csv")
                elif owner and ds.history_by_id[eid]["user_id"] != owner:
                    problems.append(
                        f"{mid}: evidence {eid} belongs to "
                        f"{ds.history_by_id[eid]['user_id']}, not {owner}"
                    )

    if problems:
        raise ContractError("output contract violated:\n  " + "\n  ".join(problems[:20]))


def _owner(message_id: str, ds) -> str | None:
    for row in ds.messages:
        if row["message_id"] == message_id:
            return row["user_id"]
    for row in ds.samples:
        if row["message_id"] == message_id:
            return row["user_id"]
    return None


def write(rows: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path
