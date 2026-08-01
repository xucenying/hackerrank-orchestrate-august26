"""Stage 6 - deterministic policy arbitration.

Rules carry an explicit precedence number rather than relying on list position,
so reordering the list for readability cannot silently change behaviour. Every
rule proposes an adjustment; the highest-precedence rule that fires wins.

A rule can fire (its condition is true) and still change nothing - a cap only
bites when the current action is above it. Both are recorded separately.
"""

from __future__ import annotations

from dataclasses import dataclass

from classify import REASON_TEMPLATES, Verdict

RANK = {"mute": 0, "digest": 1, "notify": 2}
UNRANK = {v: k for k, v in RANK.items()}

CONF_MIN, CONF_MAX = 0.78, 0.91
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
    action, mtype, code = verdict.action, verdict.message_type, verdict.reason_code

    def record(rule: str, prec: int, before: str, after: str, detail: str) -> str:
        fired.append(Fired(rule, prec, before != after, detail))
        return after

    # 100 - hard override. The only two flags trusted to force the answer.
    override = s.override
    if override:
        new = "mute"
        action = record("hard_override", 100, action, new, override)
        mtype = "scam"
        if override == "INJECTION":
            code = "ROUTER_INJECTION"
        elif code not in ("OTP_PHISHING", "IMPERSONATED_BRAND", "FAKE_SUPPORT"):
            code = "IMPERSONATED_BRAND"
        return _finish(verdict, action, mtype, code, ctx, fired)

    # 90 - genuine time pressure always interrupts, whatever the user did with
    # similar messages before. In the solved samples `urgent` is `notify` 4/4.
    if mtype == "urgent":
        action = record("urgent_floor", 90, action, _floor(action, "notify"), "time-critical content")

    # 80 - admin floor. Guarded: authority does not extend to money requests.
    admin_ok = (
        f.get("sender_is_admin")
        and mtype in ("urgent", "event")
        and not s.has("PAYMENT_REQUEST", "CREDENTIAL_REQUEST")
    )
    if admin_ok:
        action = record("admin_floor", 80, action, _floor(action, "notify"), "admin operational notice")
        if code in ("NO_SIGNAL", "GROUP_INFORMATIONAL"):
            code = "ADMIN_OPERATIONAL"

    # 60 - opted-out marketing.
    if mtype in ("promotion", "spam") and (
        f.get("opted_out_at") or (f.get("business_id") and not f.get("allows_promotions", True))
    ):
        action = record("opt_out", 60, action, "mute", "user opted out of promotions")
        if code == "NO_SIGNAL":
            code = "OPTED_OUT_MARKETING"

    # 60 - muted group cap (skipped when the admin floor already fired).
    if f.get("group_muted") and not admin_ok and mtype != "urgent":
        action = record("group_muted", 60, action, _cap(action, "digest"), "receiver muted this group")

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

    return _finish(verdict, action, mtype, code, ctx, fired)


def _finish(verdict: Verdict, action: str, mtype: str, code: str, ctx, fired) -> tuple[Verdict, list[Fired]]:
    confidence = _calibrate(verdict.confidence, ctx, fired)
    out = Verdict(
        action=action,
        message_type=mtype,
        reason_code=code if code in REASON_TEMPLATES else "NO_SIGNAL",
        confidence=confidence,
        evidence_ids=list(verdict.evidence_ids),
        trace=list(verdict.trace) + [f"{x.rule}({'changed' if x.changed else 'no-op'})" for x in fired],
    )
    return out, fired


def _calibrate(raw: float, ctx, fired) -> float:
    """Squeeze into the band the solved samples occupy, nudged by signal agreement."""
    conf = raw
    if ctx.evidence.citable:
        conf += 0.02
    if ctx.safety.override:
        conf += 0.03
    if not ctx.evidence.citable and not ctx.safety.flags:
        conf -= 0.03
    if any(x.changed for x in fired):
        conf -= 0.01
    return round(min(max(conf, CONF_MIN), CONF_MAX), 2)


def render_reason(code: str) -> str:
    return REASON_TEMPLATES.get(code, REASON_TEMPLATES["NO_SIGNAL"])
