"""Stage 5 - deterministic policy arbitration.

Rules carry an explicit precedence number rather than relying on list position,
so reordering the list for readability cannot silently change behaviour. Every
rule proposes an adjustment; the highest-precedence rule that fires wins.

A rule can fire (its condition is true) and still change nothing - a cap only
bites when the current action is above it. Both are recorded separately.
"""

from __future__ import annotations

from dataclasses import dataclass

from classify import Verdict

RANK = {"mute": 0, "digest": 1, "notify": 2}
UNRANK = {v: k for k, v in RANK.items()}

URGENCY_ABUSE_FLAGS = (
    "PAYMENT_REQUEST", "CREDENTIAL_REQUEST", "PRESSURE", "PRIZE_BAIT",
    "SHORTENER", "VIRAL_FORWARD", "FORWARDED", "UNKNOWN_BRAND", "YOUNG_DOMAIN",
)

HIGH_DISMISS = 0.45


@dataclass
class Fired:
    rule: str
    precedence: int
    changed: bool
    detail: str


def _cap(action: str, ceiling: str) -> str:
    return UNRANK[min(RANK[action], RANK[ceiling])]


def _floor(action: str, floor: str) -> str:
    return UNRANK[max(RANK[action], RANK[floor])]


def apply(verdict: Verdict, ctx) -> tuple[Verdict, list[Fired]]:
    s, f = ctx.safety, ctx.facts
    fired: list[Fired] = []
    action, mtype, reason = verdict.action, verdict.message_type, verdict.reason

    def record(rule: str, prec: int, before: str, after: str, detail: str) -> str:
        fired.append(Fired(rule, prec, before != after, detail))
        return after

    # 100 - hard override. The only two flags trusted to force the answer.
    override = s.override
    if override:
        new = "mute"
        action = record("hard_override", 100, action, new, override)
        mtype = "scam"
        return _finish(verdict, action, mtype, reason, ctx, fired)

    # 90 - genuine time pressure interrupts.
    prior = ctx.evidence.primary
    urgency_is_trustworthy = (
        not s.has(*URGENCY_ABUSE_FLAGS)
        and (prior is None or prior.engagement not in ("reported", "dismissed"))
    )
    if mtype == "urgent" and urgency_is_trustworthy:
        action = record("urgent_floor", 90, action, _floor(action, "notify"), "time-critical content")

    # 80 - admin floor. Guarded: authority does not extend to money requests.
    admin_ok = (
        f.get("sender_is_admin")
        and mtype in ("urgent", "event")
        and not s.has("PAYMENT_REQUEST", "CREDENTIAL_REQUEST")
    )
    if admin_ok:
        action = record("admin_floor", 80, action, _floor(action, "notify"), "admin operational notice")

    # 60 - opted-out marketing.
    if mtype in ("promotion", "spam") and (
        f.get("opted_out_at") or (f.get("business_id") and not f.get("allows_promotions", True))
    ):
        action = record("opt_out", 60, action, "mute", "user opted out of promotions")

    # 60 - muted group cap (skipped when the admin floor already fired).
    if f.get("group_muted") and not admin_ok and mtype != "urgent":
        action = record("group_muted", 60, action, _cap(action, "digest"), "receiver muted this group")

    # 55 - cold sender cap. A stranger doesn't earn an interrupt.
    if f.get("cold_sender") and mtype != "urgent":
        action = record("cold_sender", 55, action, _cap(action, "digest"),
                        "first-ever message from this sender")

    # 50 - viral forward cap, narrowed to non-admin senders.
    if f.get("forwarded", 0) >= 5 and not f.get("sender_is_admin") and mtype != "urgent":
        action = record("viral_forward", 50, action, _cap(action, "digest"), "forwarded 5+ times")

    # 45 - the sender explicitly said it can wait.
    if f.get("deferrable") and mtype != "urgent":
        action = record("deferrable", 45, action, _cap(action, "digest"),
                        "sender indicated this is not time-critical")

    # 40 - quiet hours.
    if f.get("in_dnd") and mtype != "urgent":
        action = record("quiet_hours", 40, action, _cap(action, "digest"), "inside receiver quiet hours")

    # 20 - notification fatigue, only on borderline calls.
    if f.get("dismiss_rate", 0.0) >= HIGH_DISMISS and verdict.confidence < 0.85 and mtype != "urgent":
        action = record("fatigue", 20, action, _cap(action, "digest"),
                        f"receiver dismisses {f['dismiss_rate']:.0%}")

    return _finish(verdict, action, mtype, reason, ctx, fired)


def _finish(verdict, action, mtype, reason, ctx, fired):
    out = Verdict(
        action=action,
        message_type=mtype,
        reason=reason,
        confidence=verdict.confidence,
        evidence_ids=list(verdict.evidence_ids),
        trace=list(verdict.trace) + [f"{x.rule}({'changed' if x.changed else 'no-op'})" for x in fired],
    )
    return out, fired
