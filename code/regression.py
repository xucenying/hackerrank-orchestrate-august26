"""Regression check on output.csv - the safety net the sample eval cannot be.

The 30 solved samples are the only rows with a known answer, so every other
check in this pipeline is blind to the other 110. That gap is not theoretical:
an unguarded urgent_floor rule once scored a clean 30/30 while promoting five
phishing messages from `mute` to `notify`, three of which the receiving user had
already reported. It was caught by eye, not by a test.

This module compares a fresh run against the last accepted output.csv and grades
each change by consequence rather than counting diffs. It never asks whether a
prediction is correct - it cannot, there is no answer key. It only asks whether
anything alarming moved.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

# Only two transitions are dangerous, and they are dangerous in opposite ways.
CRITICAL = {
    ("mute", "notify"): "a suppressed message now interrupts the user",
    ("notify", "mute"): "an interrupting message is now silently suppressed",
}
WARN = {("notify", "digest"), ("digest", "notify"), ("digest", "mute"), ("mute", "digest")}


@dataclass
class Change:
    message_id: str
    before: str
    after: str
    severity: str
    detail: str = ""
    text: str = ""
    evidence: str = ""


@dataclass
class Diff:
    total: int = 0
    changes: list[Change] = field(default_factory=list)
    baseline: Path | None = None

    @property
    def critical(self) -> list[Change]:
        return [c for c in self.changes if c.severity == "CRITICAL"]

    @property
    def ok(self) -> bool:
        return not self.critical

    def render(self) -> str:
        if self.baseline is None:
            return "REGRESSION  no baseline output.csv yet - nothing to compare against."

        L = [f"REGRESSION vs {self.baseline.name}        "
             f"{self.total} rows, {len(self.changes)} changed"]
        if not self.changes:
            L.append("  no changes")
            return "\n".join(L)

        for severity in ("CRITICAL", "warn", "info"):
            group = [c for c in self.changes if c.severity == severity]
            if not group:
                continue
            L.append("")
            label = group[0].detail if severity == "CRITICAL" else {
                "warn": "action changed (low consequence)",
                "info": "message_type / reason / confidence only",
            }[severity]
            L.append(f"  {severity:<9} {label:<52} {len(group)} rows")
            for c in group if severity != "info" else group[:5]:
                L.append(f"    {c.message_id:<9} {c.before} -> {c.after:<7} {c.text[:58]}")
                if c.evidence:
                    L.append(f"              prior evidence: {c.evidence}")
            if severity == "info" and len(group) > 5:
                L.append(f"    ... and {len(group) - 5} more")

        reported = [c for c in self.critical if "reported" in c.evidence]
        if reported:
            L.append("")
            L.append(f"  {len(reported)} of {len(self.critical)} promoted rows cite evidence "
                     f"this user REPORTED.")
        return "\n".join(L)


def _load(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as fh:
        return {r["message_id"]: r for r in csv.DictReader(fh)}


def compare(new_rows: list[dict], baseline_path: str | Path, ds, results=None) -> Diff:
    """Grade a fresh run against the last accepted output.csv."""
    baseline_path = Path(baseline_path)
    old = _load(baseline_path)
    diff = Diff(total=len(new_rows), baseline=baseline_path if old else None)
    if not old:
        return diff

    texts = {r["message_id"]: (r["message_text"] or "(media only)").replace("\n", " ")
             for r in ds.messages + ds.samples}
    verdicts = dict(results or [])

    for row in new_rows:
        mid = row["message_id"]
        prev = old.get(mid)
        if prev is None:
            continue
        before, after = prev["action"], row["action"]
        if before == after:
            if (prev["message_type"], prev["reason"], prev["confidence"]) != (
                row["message_type"], row["reason"], row["confidence"]
            ):
                diff.changes.append(Change(mid, before, after, "info", text=texts.get(mid, "")))
            continue

        severity = "CRITICAL" if (before, after) in CRITICAL else (
            "warn" if (before, after) in WARN else "info")
        detail = CRITICAL.get((before, after), "")

        # Surface the contradiction directly: is the system now interrupting
        # someone with a message they previously reported or dismissed?
        evidence = ""
        verdict = verdicts.get(mid)
        if verdict is not None and getattr(verdict, "evidence_ids", None):
            labels = [f"{e} [{ds.engagement.get(e, '?')}]" for e in verdict.evidence_ids]
            evidence = ", ".join(labels)

        diff.changes.append(
            Change(mid, before, after, severity, detail, texts.get(mid, ""), evidence)
        )

    order = {"CRITICAL": 0, "warn": 1, "info": 2}
    diff.changes.sort(key=lambda c: (order[c.severity], c.message_id))
    return diff
