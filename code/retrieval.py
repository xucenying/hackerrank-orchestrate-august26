"""Stage 4 - evidence retrieval, scoped to the receiving user.

Every cited evidence ID must belong to the same user as the message being routed
(verified across all 31 citations in sample_messages.csv). Cross-user matches on
the same media file are kept separately as context and must never be cited.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

STOP = {
    "the", "a", "an", "and", "or", "but", "if", "is", "are", "was", "were", "be",
    "to", "of", "in", "on", "at", "for", "with", "as", "by", "from", "this",
    "that", "it", "its", "will", "can", "you", "your", "we", "our", "us", "i",
    "not", "no", "so", "do", "does", "has", "have", "had", "please", "pls",
}
TOKEN_RE = re.compile(r"[a-z0-9]+")

# Structural agreement is a boost on content similarity, never a replacement.
# Diagnosing the solved samples showed the gold citation is usually the top
# lexical match, while a hard tier cascade kept returning same-sender rows about
# unrelated topics - so the joins are worth roughly a fifth of a similarity
# point, not an override.
STRUCTURAL_BONUS = {
    "same_media": 0.35,
    "same_business": 0.25,
    "same_sender_group": 0.20,
    "same_group": 0.10,
}
MIN_SCORE = 0.15
TOP_K = 1


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall((text or "").lower()) if t not in STOP and len(t) > 2]


@dataclass
class Evidence:
    message_id: str
    engagement: str
    how: str
    score: float
    text: str


@dataclass
class Retrieved:
    citable: list[Evidence] = field(default_factory=list)
    context_only: list[Evidence] = field(default_factory=list)

    @property
    def ids(self) -> list[str]:
        return [e.message_id for e in self.citable]

    @property
    def primary(self) -> Evidence | None:
        return self.citable[0] if self.citable else None


class Index:
    """Per-user TF-IDF over message_history text. Small corpus, no sklearn needed."""

    def __init__(self, ds):
        self.ds = ds
        self._df: Counter = Counter()
        self._docs: dict[str, Counter] = {}
        for row in ds.history:
            tokens = tokenize(row["message_text"])
            self._docs[row["message_id"]] = Counter(tokens)
            self._df.update(set(tokens))
        self._n = max(len(self._docs), 1)

    def _idf(self, term: str) -> float:
        return math.log((self._n + 1) / (self._df.get(term, 0) + 1)) + 1.0

    def _vector(self, counts: Counter) -> dict[str, float]:
        vec = {t: (1 + math.log(c)) * self._idf(t) for t, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def similarity(self, query_counts: Counter, message_id: str) -> float:
        q = self._vector(query_counts)
        d = self._vector(self._docs.get(message_id, Counter()))
        if not q or not d:
            return 0.0
        small, large = (q, d) if len(q) < len(d) else (d, q)
        return sum(v * large.get(t, 0.0) for t, v in small.items())


def is_cold_sender(row: dict, ds) -> bool:
    """First ever contact from this sender to this user, in a 1:1 conversation."""
    if row["conversation_type"] != "personal":
        return False
    sender = row.get("sender_user_id") or ""
    if not sender:
        return True
    return not any(
        h.get("sender_user_id") == sender
        for h in ds.history_by_user.get(row["user_id"], [])
    )


def retrieve(row: dict, ds, index: Index, media_text: str = "", top_k: int = TOP_K) -> Retrieved:
    user = row["user_id"]
    pool = ds.history_by_user.get(user, [])
    seen: set[str] = set()
    out = Retrieved()

    # A first-ever message from an unknown sender has no relevant history, even
    # when similar words appear elsewhere. The samples emit `none` here.
    if is_cold_sender(row, ds):
        return out

    def add(hist_row: dict, how: str, score: float) -> None:
        mid = hist_row["message_id"]
        if mid in seen:
            return
        seen.add(mid)
        out.citable.append(
            Evidence(
                message_id=mid,
                engagement=ds.engagement.get(mid, "unknown"),
                how=how,
                score=score,
                text=(hist_row["message_text"] or "")[:160],
            )
        )

    query = Counter(tokenize(f"{row.get('message_text', '')} {media_text}"))
    media_id = row.get("media_id") or ""
    bid = row.get("business_id") or ""
    sender = row.get("sender_user_id") or ""
    group = row.get("group_id") or ""

    # Same media, different user: usable as background, never citable.
    if media_id:
        for h in ds.history:
            if h.get("media_id") == media_id and h["user_id"] != user:
                out.context_only.append(
                    Evidence(
                        message_id=h["message_id"],
                        engagement=ds.engagement.get(h["message_id"], "unknown"),
                        how="same_media_other_user",
                        score=0.0,
                        text=(h["message_text"] or "")[:160],
                    )
                )

    # Blended scoring. Structural joins are a *boost* on top of content
    # similarity, not an override of it - a same-sender match about an unrelated
    # topic is worse evidence than a different sender saying the same thing.
    scored: list[tuple[float, str, dict]] = []
    for h in pool:
        sim = index.similarity(query, h["message_id"]) if query else 0.0
        bonus, how = 0.0, "lexical"
        if media_id and h.get("media_id") == media_id:
            bonus, how = STRUCTURAL_BONUS["same_media"], "same_media"
        elif bid and h.get("business_id") == bid:
            bonus, how = STRUCTURAL_BONUS["same_business"], "same_business"
        elif sender and group and h.get("sender_user_id") == sender and h.get("group_id") == group:
            bonus, how = STRUCTURAL_BONUS["same_sender_group"], "same_sender_group"
        elif group and h.get("group_id") == group:
            bonus, how = STRUCTURAL_BONUS["same_group"], "same_group"
        scored.append((sim + bonus, how, h))

    scored.sort(key=lambda p: p[0], reverse=True)
    for score, how, h in scored:
        if len(out.citable) >= top_k or score < MIN_SCORE:
            break
        add(h, how, round(score, 3))
    return out
