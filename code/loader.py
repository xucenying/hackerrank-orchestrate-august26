"""Stage 0 - load every dataset CSV and build the lookup tables.

The address book is rebuilt from disk on every run. It is never cached: the whole
dataset is ~180KB and rebuilding costs milliseconds, so caching would buy nothing
and add a staleness class. What this module does need is validation - see check().
"""

from __future__ import annotations

import csv
import datetime as dt
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

MESSAGE_FILES = ("messages.csv", "sample_messages.csv", "message_history.csv")


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_ts(value: str) -> dt.datetime | None:
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(value, fmt)
        except (TypeError, ValueError):
            continue
    return None


def in_dnd(window: str, when: dt.datetime | None) -> bool:
    """Quiet-hours windows all wrap midnight (e.g. 22:00-07:00)."""
    if not window or when is None or "-" not in window:
        return False
    start_s, _, end_s = window.partition("-")
    try:
        start = dt.time.fromisoformat(start_s.strip())
        end = dt.time.fromisoformat(end_s.strip())
    except ValueError:
        return False
    now = when.time()
    if start <= end:
        return start <= now < end
    return now >= start or now < end


@dataclass
class Dataset:
    root: Path

    users: dict[str, dict] = field(default_factory=dict)
    groups: dict[str, dict] = field(default_factory=dict)
    businesses: dict[str, dict] = field(default_factory=dict)

    # (group_id, user_id) -> membership row
    membership: dict[tuple[str, str], dict] = field(default_factory=dict)
    # (user_id, business_id) -> relationship row; absence is itself a signal
    biz_rel: dict[tuple[str, str], dict] = field(default_factory=dict)

    messages: list[dict] = field(default_factory=list)
    samples: list[dict] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)

    history_by_id: dict[str, dict] = field(default_factory=dict)
    history_by_user: dict[str, list[dict]] = field(default_factory=lambda: defaultdict(list))
    engagement: dict[str, str] = field(default_factory=dict)

    media_path: dict[str, str] = field(default_factory=dict)
    fatigue: dict[str, dict] = field(default_factory=dict)
    output_order: list[str] = field(default_factory=list)

    # --------------------------------------------------------------- build

    @classmethod
    def load(cls, root: str | Path) -> "Dataset":
        root = Path(root)
        ds = cls(root=root)

        ds.users = {r["user_id"]: r for r in _read(root / "users.csv")}
        ds.groups = {r["group_id"]: r for r in _read(root / "groups.csv")}
        ds.businesses = {r["business_id"]: r for r in _read(root / "business_accounts.csv")}
        ds.membership = {(r["group_id"], r["user_id"]): r for r in _read(root / "group_members.csv")}
        ds.biz_rel = {(r["user_id"], r["business_id"]): r for r in _read(root / "user_business_history.csv")}

        ds.messages = _read(root / "messages.csv")
        ds.samples = _read(root / "sample_messages.csv")
        ds.history = _read(root / "message_history.csv")

        ds.history_by_id = {r["message_id"]: r for r in ds.history}
        for row in ds.history:
            ds.history_by_user[row["user_id"]].append(row)

        for ev in _read(root / "message_events.csv"):
            ds.engagement[ev["message_id"]] = _engagement_label(ev)

        for row in _read(root / "images.csv"):
            ds.media_path[row["image_id"]] = row["file_path"]
        for row in _read(root / "voice_notes.csv"):
            ds.media_path[row["voice_note_id"]] = row["file_path"]

        totals: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
        for row in _read(root / "daily_notification_summary.csv"):
            acc = totals[row["user_id"]]
            acc[0] += _int(row["notifications_sent"])
            acc[1] += _int(row["notifications_dismissed"])
            acc[2] += 1
        for uid, (sent, dismissed, days) in totals.items():
            ds.fatigue[uid] = {
                "sent": sent,
                "dismissed": dismissed,
                "days": days,
                "dismiss_rate": (dismissed / sent) if sent else 0.0,
                "per_day": (sent / days) if days else 0.0,
            }

        ds.output_order = [r["message_id"] for r in _read(root / "output.csv")]
        return ds

    # ---------------------------------------------------------- validation

    def check(self) -> list[str]:
        """Referential integrity. Schema/reference drift must fail loudly."""
        problems: list[str] = []

        for name, rows in (("messages.csv", self.messages), ("message_history.csv", self.history)):
            for row in rows:
                mid = row["message_id"]
                if row["user_id"] not in self.users:
                    problems.append(f"{name}:{mid} unknown user_id {row['user_id']!r}")
                if row["group_id"] and row["group_id"] not in self.groups:
                    problems.append(f"{name}:{mid} unknown group_id {row['group_id']!r}")
                if row["business_id"] and row["business_id"] not in self.businesses:
                    problems.append(f"{name}:{mid} unknown business_id {row['business_id']!r}")
                if row["sender_user_id"] and row["sender_user_id"] not in self.users:
                    problems.append(f"{name}:{mid} unknown sender_user_id {row['sender_user_id']!r}")
                if row["media_id"] and row["media_id"] not in self.media_path:
                    problems.append(f"{name}:{mid} unknown media_id {row['media_id']!r}")

        missing_events = {r["message_id"] for r in self.history} - set(self.engagement)
        if missing_events:
            problems.append(f"message_history rows with no event row: {sorted(missing_events)[:5]}")

        ids = [r["message_id"] for r in self.messages]
        if self.output_order and ids != self.output_order:
            if set(ids) != set(self.output_order):
                problems.append("messages.csv and output.csv message_id sets differ")
            else:
                problems.append("messages.csv and output.csv are in different orders")

        for mid, rel in self.media_path.items():
            if not (self.root / rel).exists():
                problems.append(f"media file missing on disk: {mid} -> {rel}")

        return problems

    # ------------------------------------------------------------ helpers

    def role_in_group(self, group_id: str, user_id: str) -> str | None:
        row = self.membership.get((group_id, user_id))
        return row["role"] if row else None

    def group_muted(self, group_id: str, user_id: str) -> bool:
        row = self.membership.get((group_id, user_id))
        return bool(row and row["group_muted_by_user"] == "1")

    def media_file(self, media_id: str) -> Path | None:
        rel = self.media_path.get(media_id)
        return (self.root / rel) if rel else None


def _engagement_label(ev: dict) -> str:
    if ev["message_reported"] == "1":
        return "reported"
    if ev["notification_dismissed"] == "1":
        return "dismissed"
    if ev["message_replied"] == "1":
        return "replied"
    return "read"


# Engagement label -> the action it argues for (see message_events analysis).
ENGAGEMENT_ACTION = {
    "replied": "notify",
    "read": "digest",
    "dismissed": "mute",
    "reported": "mute",
}
