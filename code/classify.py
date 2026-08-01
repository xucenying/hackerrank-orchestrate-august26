"""Stage 5 - decide action / message_type / reason_code / confidence / evidence.

Two interchangeable backends:
  rules  - deterministic, offline, no API key. The baseline.
  claude - claude-opus-5 with structured outputs and a cached system prefix.

Both emit the same Verdict, so policy.py and writer.py are backend-agnostic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from loader import ENGAGEMENT_ACTION

ACTIONS = ("notify", "digest", "mute")
TYPES = (
    "personal", "urgent", "event", "payment", "business_update", "promotion",
    "greeting", "forward", "spam", "scam", "unknown",
)

REASON_TEMPLATES = {
    "TRUSTED_ADMIN_URGENT": "A trusted group admin sent a time-sensitive update that should interrupt the user.",
    "ADMIN_OPERATIONAL": "A group admin sent a same-day operational update that the user is likely to need immediately.",
    "CLOSE_CONTACT_URGENT": "A close contact sent a short urgent request that should interrupt the user.",
    "TRANSACTIONAL_EXPECTED": "The message relates to an active order, booking, or payment the user is expecting.",
    "TRUSTED_LOW_PRIORITY": "The sender is trusted, but the message has no urgent action or safety relevance.",
    "VERIFIED_LOW_PRIORITY": "The verified business message is legitimate but does not require immediate attention.",
    "MATCHES_INTEREST": "The message matches the user's known interests but is still low priority.",
    "GROUP_INFORMATIONAL": "The group update is useful background but does not need to interrupt the user.",
    "IGNORED_SIMILAR": "Similar historical messages were ignored, dismissed, or muted by this user.",
    "OPTED_OUT_MARKETING": "The user has opted out of or repeatedly dismissed similar marketing messages.",
    "REPEATED_FORWARDS": "The sender has a pattern of repeated forwards or greetings that the user usually ignores.",
    "MUTED_LOW_VALUE": "The user muted this conversation and the message carries no urgent or personal request.",
    "OTP_PHISHING": "The message asks for urgent OTP or account verification through a suspicious flow.",
    "FAKE_SUPPORT": "The message uses fake support language and account-blocking pressure to push the user into action.",
    "IMPERSONATED_BRAND": "The sender imitates a known brand from an unverified account on a lookalike domain.",
    "COLD_SENSITIVE_REQUEST": "This is the first message from the sender and it asks for sensitive verification or payment.",
    "PRIZE_BAIT": "The message promises a prize or reward to push the user toward an unsafe link.",
    "ROUTER_INJECTION": "The message tries to instruct the router, but the routing decision should be based on the actual content and risk.",
    "UNSOLICITED_BULK": "The sender has no relationship with this user and the message is unsolicited bulk marketing.",
    "NO_SIGNAL": "There is no strong signal that this message needs immediate attention.",
}


@dataclass
class Verdict:
    action: str
    message_type: str
    reason_code: str
    confidence: float
    evidence_ids: list[str] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)

    def note(self, msg: str) -> "Verdict":
        self.trace.append(msg)
        return self


# --------------------------------------------------------------- type rules

# Urgency must be time-pressure, not merely a mention of minutes. "when you get
# 5 mins, call me" is not urgent; "bus leaves 15 mins early" is.
URGENT_RE = re.compile(
    r"\b(urgent|immediately|asap|right away|emergency|evacuat\w+|leaking|"
    r"tanker|ambulance|pulled (to|up|forward)|moved up|escalat\w+|"
    r"alert threshold|come online now)\b"
    r"|leav\w+[^.]{0,25}(early|\d{1,2}\s*min)"
    r"|\d{1,2}\s*min(ute)?s?\b[^.]{0,25}\b(max|left|only|before|then)\b"
    r"|\b(starts?|begins?) in \d{1,2}\s*min"
    # An explicit immediate request, or a health emergency.
    r"|\bcall (me |us )?now\b"
    r"|\b(is |feeling )?unwell\b|\btaken to (the|a) (hospital|clinic)\b",
    re.I,
)
# "Nothing urgent" is not urgent, and neither is "don't call now" - which is
# the same four words as the urgent form, inverted.
NEGATED_URGENT_RE = re.compile(
    r"\b(nothing|not|no|isn'?t|non)[- ]?\s*urgent\b"
    r"|\b(don'?t|dont|do not|no need to) call (me |us )?now\b",
    re.I,
)

# The sender has explicitly said this can wait. Trust them over any other signal.
DEFERRABLE_RE = re.compile(
    r"\b(till|until|by) next\b|\bnext (week|month|sunday|monday|saturday)\b"
    r"|\bno rush\b|\bwhenever (you|it)\b|\btalk tomorrow\b|\bend of (the )?month\b"
    r"|\bnothing urgent\b|\bdon'?t call now\b", re.I,
)

# A scheduled service changing time is an event, even when the change is today.
SCHEDULED_RE = re.compile(
    r"\b(bus|route [a-z0-9]|class|drill|alarm test|assembly|form is open|"
    r"rehearsal|practice session)\b", re.I,
)
PAYMENT_RE = re.compile(
    r"\b(payment (due|link|pending|failed|received|request|reminder)|"
    r"pay (the|your|now|before|online)|invoice|bill(ing)? (due|cycle)|due date|"
    r"maintenance (charge|due|payment)|emi|instal?ment|challan|dues|penalty|"
    r"outstanding|renewal (due|amount))\b", re.I,
)
PROMO_RE = re.compile(
    r"\b(\d{1,3}\s*%\s*off|discount|sale|offer|deal|coupon|promo code|flat \d+|"
    r"buy \d|cashback|free delivery|limited time|shop now|book now|lowest price)\b"
    r"|\b(rs\.?|inr|₹)\s*[\d,]{3,}", re.I,
)
SELLING_RE = re.compile(
    r"\b(selling|for sale|barely used|gently used|no damage|price is|negotiable|"
    r"dm me|interested\?|size [smlx]\b|bought (last|it))\b", re.I,
)
EVENT_RE = re.compile(
    r"\b(meeting|rsvp|venue|circular|assembly|rehearsal|workshop|webinar|"
    r"drill|alarm test|form is open|registration|schedule[sd]? (for|on)|"
    r"timing|slot|appointment|consent note|parent[- ]teacher)\b", re.I,
)
BIZ_UPDATE_RE = re.compile(
    r"\b(order|delivery|shipped|dispatch|out for delivery|booking|reservation|"
    r"confirmed|statement|ticket|prescription|advisory|refund (processed|issued))\b", re.I,
)
GREETING_RE = re.compile(
    r"\b(good morning|good evening|blessings|stay positive|keep smiling|"
    r"happy (birthday|anniversary|new year|diwali)|warm wishes|god bless|"
    r"share (this )?with everyone)\b", re.I,
)

# Group character drives the default category when nothing else matches.
SOCIAL_GROUPS = {
    "family", "extended_family", "friends", "book_club", "alumni", "sports",
    "college_students", "caregiving", "dance_class",
}
NOTICE_GROUPS = {
    "society", "school_group", "coworker", "college_faculty", "safety",
    "tech_community", "investment_tips", "finance_help",
}
TRADE_GROUPS = {"marketplace", "local_food", "real_estate"}


def _classify_type(ctx, safety) -> str:
    blob = f"{ctx.row.get('message_text', '')}\n{ctx.media_text}"
    conv = ctx.row["conversation_type"]
    f = ctx.facts
    trusted = safety.has("TRUSTED_SENDER")

    # --- danger first -----------------------------------------------------
    if safety.has("INJECTION") or safety.has("IMPERSONATION"):
        return "scam"
    if not trusted:
        if safety.text_scam_score >= 2:
            return "scam"
        if safety.has("CREDENTIAL_REQUEST") and (
            safety.has("PRESSURE", "PAYMENT_REQUEST") or f.get("relationship") is None
        ):
            return "scam"
        if safety.has("PRIZE_BAIT") and safety.has("PAYMENT_REQUEST", "PRESSURE", "UNKNOWN_BRAND"):
            return "scam"
        # Unsolicited bulk needs corroboration beyond a missing relationship row.
        if safety.has("UNKNOWN_BRAND") or (
            conv == "business"
            and f.get("relationship") is None
            and safety.has("YOUNG_DOMAIN", "HIGH_REPORTS", "SHORTENER")
        ):
            return "spam"

    # --- genuine time pressure outranks topic ------------------------------
    urgent = bool(URGENT_RE.search(blob)) and not NEGATED_URGENT_RE.search(blob)
    if urgent and not SCHEDULED_RE.search(blob):
        return "urgent"

    # --- conversation-type aware -------------------------------------------
    if conv == "personal":
        if PAYMENT_RE.search(blob):
            return "payment"
        if f.get("cold_sender"):
            return "unknown"
        return "personal"

    if conv == "business":
        if PAYMENT_RE.search(blob):
            return "payment"
        if PROMO_RE.search(blob):
            return "promotion"
        if EVENT_RE.search(blob):
            return "event"
        return "business_update"

    # --- group -------------------------------------------------------------
    gtype = f.get("group_type") or ""
    if GREETING_RE.search(blob):
        return "greeting"
    if f.get("forwarded", 0) >= 5 and not f.get("sender_is_admin"):
        return "forward"
    if PAYMENT_RE.search(blob):
        return "payment"
    if PROMO_RE.search(blob) or SELLING_RE.search(blob):
        return "promotion"
    if gtype in TRADE_GROUPS:
        return "promotion"  # commerce groups: posts are listings by default
    if EVENT_RE.search(blob) or urgent:
        return "event"
    if gtype in SOCIAL_GROUPS:
        return "personal"
    if gtype in NOTICE_GROUPS:
        return "event"
    return "personal"


# ------------------------------------------------------------ rules backend

def classify_rules(ctx) -> Verdict:
    s, f = ctx.safety, ctx.facts
    ev = ctx.evidence
    mtype = _classify_type(ctx, s)
    ids = ev.ids

    if s.has("INJECTION"):
        return Verdict("mute", "scam", "ROUTER_INJECTION", 0.88, ids).note("injection")
    if s.has("IMPERSONATION"):
        code = "OTP_PHISHING" if s.has("CREDENTIAL_REQUEST") else "IMPERSONATED_BRAND"
        return Verdict("mute", "scam", code, 0.89, ids).note("impersonation")
    if mtype == "scam":
        if f.get("relationship") is None and ctx.row["conversation_type"] == "business" and not ids:
            code = "COLD_SENSITIVE_REQUEST"
        elif s.has("PRESSURE"):
            code = "FAKE_SUPPORT"
        elif s.has("PRIZE_BAIT"):
            code = "PRIZE_BAIT"
        else:
            code = "OTP_PHISHING"
        return Verdict("mute", "scam", code, 0.86, ids).note("scam text")

    # opted out of / historically dismisses this business's marketing
    if mtype in ("promotion", "spam") and ctx.row["conversation_type"] == "business":
        opted_out = f.get("opted_out_at") or not f.get("allows_promotions", False)
        if f.get("relationship") is None and not s.has("TRUSTED_SENDER"):
            return Verdict("mute", "spam", "UNSOLICITED_BULK", 0.83, ids).note("no relationship")
        if f.get("relationship") is None or opted_out:
            # Preserve the classified type: an opted-out message from an unknown
            # brand on a shortener is spam, not merely an unwanted promotion.
            return Verdict("mute", mtype, "OPTED_OUT_MARKETING", 0.84, ids).note("opted out")

    if mtype in ("greeting", "forward") and f.get("forwarded", 0) >= 1:
        return Verdict("mute", mtype, "REPEATED_FORWARDS", 0.83, ids).note("viral forward")

    # engagement-driven default: what did this user do with the closest match?
    primary = ev.primary
    if primary is not None:
        action = ENGAGEMENT_ACTION.get(primary.engagement, "digest")
        if action == "notify":
            code = "TRUSTED_ADMIN_URGENT" if f.get("sender_is_admin") else "CLOSE_CONTACT_URGENT"
            if mtype in ("business_update", "payment"):
                code = "TRANSACTIONAL_EXPECTED"
            conf = 0.86
        elif action == "mute":
            code = "IGNORED_SIMILAR"
            conf = 0.84
        else:
            code = "VERIFIED_LOW_PRIORITY" if ctx.row["conversation_type"] == "business" else "TRUSTED_LOW_PRIORITY"
            if mtype == "promotion":
                code = "MATCHES_INTEREST"
            conf = 0.82
        return Verdict(action, mtype, code, conf, ids).note(f"evidence={primary.engagement}")

    # no evidence at all
    if f.get("sender_is_admin") and mtype in ("urgent", "event"):
        return Verdict("notify", mtype, "ADMIN_OPERATIONAL", 0.85, ids).note("admin, no evidence")
    if mtype == "urgent":
        return Verdict("notify", mtype, "CLOSE_CONTACT_URGENT", 0.82, ids).note("urgent text")
    if mtype in ("promotion", "spam", "greeting", "forward"):
        return Verdict("mute", mtype, "NO_SIGNAL", 0.79, ids).note("low value, no evidence")
    return Verdict("digest", mtype, "NO_SIGNAL", 0.80, ids).note("default")


# ----------------------------------------------------------- claude backend

SYSTEM_PROMPT_PATH = "prompts/system.md"


def classify_claude(ctx, client, system_prompt: str, model: str = "claude-opus-5") -> Verdict:
    """One classification call. System block is frozen so it caches across all rows."""
    from pydantic import BaseModel, Field  # local import: rules backend needs no pydantic

    class Routing(BaseModel):
        action: str = Field(description="notify, digest, or mute")
        message_type: str
        reason_code: str
        confidence: float
        evidence_ids: list[str] = []

    allowed = [e.message_id for e in ctx.evidence.citable]
    # The system block is byte-identical on every call, so it caches after the
    # first request. Per-message content goes in the user turn, after the
    # breakpoint. Effort defaults to high on claude-opus-5; passing
    # output_config alongside output_format would conflict.
    response = client.messages.parse(
        model=model,
        max_tokens=2000,
        system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": ctx.render()}],
        output_format=Routing,
    )
    parsed = response.parsed_output
    if parsed is None:
        return classify_rules(ctx).note("claude returned no parse; fell back to rules")
    action = parsed.action if parsed.action in ACTIONS else "digest"
    mtype = parsed.message_type if parsed.message_type in TYPES else "unknown"
    code = parsed.reason_code if parsed.reason_code in REASON_TEMPLATES else "NO_SIGNAL"
    ids = [i for i in (parsed.evidence_ids or []) if i in allowed]
    return Verdict(action, mtype, code, float(parsed.confidence), ids).note("claude")
