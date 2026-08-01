"""Stage 2b - evidence retrieval, scoped to the receiving user.

Every cited evidence ID must belong to the same user as the message being routed.
Cross-user matches on the same media file are kept separately as context and must
never be cited.

Retrieval is LLM-based: Claude reads the incoming message alongside the user's
history pool and picks the single most relevant prior message. This gives
semantic matching ("delivery" ↔ "shipment") that word-overlap methods miss.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class Evidence:
    message_id: str
    engagement: str
    how: str
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


RETRIEVAL_SYSTEM = """\
You are an evidence retrieval assistant. You will receive an incoming WhatsApp \
message and a list of past messages that the same user received.

Find the past messages that are most relevant to the incoming message. There \
may be zero, one, or several relevant matches.

Respond with ONLY a JSON object:
{"matches": [{"message_id": "<id>", "how": "<reason>"}, ...]}

- message_id: the ID of a relevant past message.
- how: one of "same_media", "same_business", "same_sender", "same_group", "semantic" \
  — describing WHY this past message is relevant.
- Return {"matches": []} if no past message is meaningfully related.

Rules:
- Only choose from the listed past messages. Never invent an ID.
- A past message about a completely different topic is NOT relevant, even if it \
  shares a sender. Prefer topical similarity over structural overlap.
- Return at most 3 matches. Prefer fewer, stronger matches over many weak ones.
"""


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


def _format_history(pool: list[dict], ds) -> str:
    lines = []
    for h in pool:
        mid = h["message_id"]
        eng = ds.engagement.get(mid, "unknown")
        text = (h["message_text"] or "")[:200]
        bid = h.get("business_id") or ""
        sender = h.get("sender_user_id") or ""
        group = h.get("group_id") or ""
        media = h.get("media_id") or ""
        parts = [f"id={mid}", f"engagement={eng}"]
        if bid:
            parts.append(f"business={bid}")
        if sender:
            parts.append(f"sender={sender}")
        if group:
            parts.append(f"group={group}")
        if media:
            parts.append(f"media={media}")
        parts.append(f"text={text!r}")
        lines.append("  ".join(parts))
    return "\n".join(lines)


def _format_incoming(row: dict, media_text: str) -> str:
    parts = [
        f"message_id: {row['message_id']}",
        f"conversation_type: {row['conversation_type']}",
        f"text: {(row.get('message_text') or '')!r}",
    ]
    if row.get("business_id"):
        parts.append(f"business_id: {row['business_id']}")
    if row.get("sender_user_id"):
        parts.append(f"sender_user_id: {row['sender_user_id']}")
    if row.get("group_id"):
        parts.append(f"group_id: {row['group_id']}")
    if row.get("media_id"):
        parts.append(f"media_id: {row['media_id']}")
    if media_text:
        parts.append(f"media_content: {media_text[:300]!r}")
    return "\n".join(parts)


def retrieve(row: dict, ds, client, model: str, cache=None,
             media_text: str = "") -> Retrieved:
    user = row["user_id"]
    pool = ds.history_by_user.get(user, [])
    out = Retrieved()

    if is_cold_sender(row, ds):
        return out

    if not pool:
        return out

    media_id = row.get("media_id") or ""

    # Same media, different user: usable as background, never citable.
    if media_id:
        for h in ds.history:
            if h.get("media_id") == media_id and h["user_id"] != user:
                out.context_only.append(
                    Evidence(
                        message_id=h["message_id"],
                        engagement=ds.engagement.get(h["message_id"], "unknown"),
                        how="same_media_other_user",
                        text=(h["message_text"] or "")[:160],
                    )
                )

    # Check cache before calling the API.
    cache_key = f"retrieval:{row['message_id']}"
    valid_ids = {h["message_id"] for h in pool}
    pool_by_id = {h["message_id"]: h for h in pool}

    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            for match in cached:
                mid = match["message_id"]
                if mid in valid_ids:
                    h = pool_by_id[mid]
                    out.citable.append(Evidence(
                        message_id=mid,
                        engagement=ds.engagement.get(mid, "unknown"),
                        how=match.get("how", "semantic"),
                        text=(h["message_text"] or "")[:160],
                    ))
            return out

    if client is None:
        return out

    prompt = (
        "INCOMING MESSAGE:\n"
        + _format_incoming(row, media_text)
        + "\n\nPAST MESSAGES THIS USER RECEIVED:\n"
        + _format_history(pool, ds)
    )

    response = client.messages.create(
        model=model,
        max_tokens=300,
        system=[{"type": "text", "text": RETRIEVAL_SYSTEM,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": prompt}],
    )

    text_out = ""
    for block in response.content:
        if hasattr(block, "text"):
            text_out = block.text.strip()
            break
    try:
        result = json.loads(text_out)
    except json.JSONDecodeError:
        result = {"matches": []}

    matches = result.get("matches", [])
    validated = []
    for match in matches:
        mid = match.get("message_id", "")
        how = match.get("how", "semantic")
        if mid in valid_ids:
            validated.append({"message_id": mid, "how": how})
            h = pool_by_id[mid]
            out.citable.append(Evidence(
                message_id=mid,
                engagement=ds.engagement.get(mid, "unknown"),
                how=how,
                text=(h["message_text"] or "")[:160],
            ))

    if cache is not None:
        cache.put(cache_key, validated)
        cache.save()

    return out
