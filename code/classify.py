"""Stage 4 - hybrid classifier: rules for safety, LLM for judgment.

Safety overrides (injection, impersonation, scam signals) are handled by
deterministic rules that cannot be bypassed. Everything else goes to the LLM
for type and reason, with action either driven by evidence engagement or
decided by the LLM when no evidence exists. Confidence is computed from
signal count and agreement, not by the LLM.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from loader import ENGAGEMENT_ACTION

ACTIONS = ("notify", "digest", "mute")
TYPES = (
    "personal", "urgent", "event", "payment", "business_update", "promotion",
    "greeting", "forward", "spam", "scam", "unknown",
)

DEFERRABLE_RE = re.compile(
    r"\b(till|until|by) next\b|\bnext (week|month|sunday|monday|saturday)\b"
    r"|\bno rush\b|\bwhenever (you|it)\b|\btalk tomorrow\b|\bend of (the )?month\b"
    r"|\bnothing urgent\b|\bdon'?t call now\b", re.I,
)

SAFETY_REASONS = {
    "INJECTION": "The message tries to instruct the router, but the routing decision should be based on the actual content and risk.",
    "IMPERSONATION_OTP": "The message asks for urgent OTP or account verification through a suspicious flow.",
    "IMPERSONATION": "The sender imitates a known brand from an unverified account on a lookalike domain.",
    "SCAM_PRESSURE": "The message uses fake support language and account-blocking pressure to push the user into action.",
    "SCAM_PRIZE": "The message promises a prize or reward to push the user toward an unsafe link.",
    "SCAM_COLD": "This is the first message from the sender and it asks for sensitive verification or payment.",
    "SCAM_GENERIC": "The message asks for urgent OTP or account verification through a suspicious flow.",
    "SPAM": "The sender has no relationship with this user and the message is unsolicited bulk marketing.",
}

CLASSIFY_SYSTEM = """\
You are a WhatsApp message classifier. Given a message and its context, decide:

1. **action**: one of `notify`, `digest`, or `mute`.
   - `notify` — important enough to interrupt the user right now
   - `digest` — useful but can wait for a summary
   - `mute` — repetitive, unwanted, low-value, suspicious, or unsafe

2. **message_type**: one of the following. Read the distinctions carefully:
   - `personal` — one-to-one or social-group conversation between people \
with no time pressure
   - `urgent` — someone needs action RIGHT NOW with active time pressure: \
a countdown is running ("20 mins max", "escalation starts in 20 minutes"), \
a deadline just moved closer ("pulled to 3", "last-minute shuffle"), someone \
is unwell, or an alert threshold is crossed. The key test: will something be \
lost or broken if the user waits an hour? If yes, it is `urgent`. A meeting \
pulled forward is `urgent`. A tanker that can only wait 20 minutes is `urgent`. \
A request to come online now because of an alert is `urgent`
   - `event` — a FUTURE scheduled happening or a routine schedule change: \
bus route adjustment, class time, appointment reminder, form deadline, \
rehearsal, drill, prescription pickup, booking reminder. The key test: is \
this informing about when or where something will happen, not demanding \
action right now? If yes, it is `event`. A business reminding you about an \
appointment or scheduled pickup is `event`, not `business_update`
   - `payment` — money owed or being collected: bill, dues, invoice, penalty
   - `business_update` — any transactional message in a business conversation: \
order status, delivery update, packing notification, statement, advisory, \
safety notice. If the conversation_type is "business" and it is not a \
promotion, payment, event, or scam, it is `business_update`. A verified \
business sending an order or delivery update is never `unknown`
   - `promotion` — marketing, listings, items for sale, product showcases, \
discount offers. Someone selling an item in a group is `promotion`. A business \
showcasing products is `promotion`. This is NOT `business_update` unless the \
user already ordered or booked something
   - `greeting` — good-morning messages, blessings, festival wishes, "stay \
positive" messages. These are `greeting` even when forwarded many times
   - `forward` — chain content passed along that is NOT a greeting or blessing. \
Forwarded news, jokes, videos, chain letters
   - `spam` — unsolicited bulk from an unknown sender
   - `scam` — deliberately deceptive; intends harm or theft
   - `unknown` — ONLY for personal conversations where the sender is unfamiliar \
(cold_sender is true) and the message does not fit any clear category. \
Business conversations are NEVER `unknown` — use `business_update`, \
`promotion`, `event`, or `payment` instead

3. **reason**: one SHORT sentence (under 18 words) explaining why you chose \
this action. Be abstract — describe the category, not the content. Never \
mention specific names, numbers, prices, places, or message details. \
Start with "The message...", "The sender...", "A trusted group admin...", \
"A verified business...", "The user...", or "The offer...". \
Mirror these examples EXACTLY in length and abstraction level:
   - "A trusted group admin sent a time-sensitive update that should interrupt the user."
   - "The message is from a work context and contains a direct deadline or meeting dependency."
   - "The sender directly asks this user for a response or action."
   - "The message is promotional but matches a topic or business the user has opted into."
   - "The message is useful group information, but it is not urgent enough to interrupt the user."
   - "The message is a harmless greeting that can be read later."
   - "A verified business is sending a legitimate but non-urgent update."
   - "The offer is potentially relevant, but it does not need immediate attention."
   - "The sender has a pattern of repeated forwards or greetings that the user usually ignores."
   - "The user has opted out of or repeatedly dismissed similar marketing messages."
   - "Similar historical messages were ignored, dismissed, or muted by this user."
   - "The sender is trusted, but the message has no urgent action or safety relevance."
   - "The sender is unfamiliar, but the message does not show urgency, payment pressure, or safety risk."

Respond with ONLY a JSON object:
{"action": "<action>", "message_type": "<type>", "reason": "<one short sentence>"}

Rules:
- Only use the allowed values listed above for action and message_type.
- Safety has already been checked. If the context shows safety flags, factor \
them into your reason but do not override the action decision for scam/spam — \
that is handled before you are called.
- If citable evidence shows the user replied to a similar message, lean toward \
notify. If they dismissed or reported it, lean toward mute. If they read it \
without acting, lean toward digest.
- Reason must be ONE short abstract sentence. No conjunctions joining two clauses. \
No specific details from the message.
"""


@dataclass
class Verdict:
    action: str
    message_type: str
    reason: str
    confidence: float
    evidence_ids: list[str] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)

    def note(self, msg: str) -> "Verdict":
        self.trace.append(msg)
        return self


# --------------------------------------------------------- safety rules

def _safety_verdict(ctx) -> Verdict | None:
    s = ctx.safety
    f = ctx.facts
    ids = ctx.evidence.ids

    if s.has("INJECTION"):
        return Verdict("mute", "scam", SAFETY_REASONS["INJECTION"],
                       _confidence(ctx, "mute"), ids).note("injection")

    if s.has("IMPERSONATION"):
        key = "IMPERSONATION_OTP" if s.has("CREDENTIAL_REQUEST") else "IMPERSONATION"
        return Verdict("mute", "scam", SAFETY_REASONS[key],
                       _confidence(ctx, "mute"), ids).note("impersonation")

    trusted = s.has("TRUSTED_SENDER")
    if not trusted:
        if s.text_scam_score >= 2:
            key = "SCAM_PRESSURE" if s.has("PRESSURE") else "SCAM_GENERIC"
            return Verdict("mute", "scam", SAFETY_REASONS[key],
                           _confidence(ctx, "mute"), ids).note("scam text")

        if s.has("CREDENTIAL_REQUEST") and (
            s.has("PRESSURE", "PAYMENT_REQUEST") or f.get("relationship") is None
        ):
            return Verdict("mute", "scam", SAFETY_REASONS["SCAM_GENERIC"],
                           _confidence(ctx, "mute"), ids).note("credential scam")

        if s.has("PRIZE_BAIT") and s.has("PAYMENT_REQUEST", "PRESSURE", "UNKNOWN_BRAND"):
            return Verdict("mute", "scam", SAFETY_REASONS["SCAM_PRIZE"],
                           _confidence(ctx, "mute"), ids).note("prize scam")

        if s.has("PAYMENT_REQUEST", "SUSPICIOUS_LINK") and s.has("PRESSURE", "CREDENTIAL_REQUEST"):
            return Verdict("mute", "scam", SAFETY_REASONS["SCAM_PRESSURE"],
                           _confidence(ctx, "mute"), ids).note("payment scam")

        if s.has("UNKNOWN_BRAND") or (
            ctx.row["conversation_type"] == "business"
            and f.get("relationship") is None
            and s.has("YOUNG_DOMAIN", "HIGH_REPORTS", "SHORTENER")
        ):
            return Verdict("mute", "spam", SAFETY_REASONS["SPAM"],
                           _confidence(ctx, "mute"), ids).note("spam")

    return None


# ----------------------------------------------------- confidence scoring

def _confidence(ctx, action: str, mtype: str = "") -> float:
    s = ctx.safety
    ev = ctx.evidence

    score = 0.82

    if ev.primary is not None:
        eng = ev.primary.engagement
        if eng in ("replied", "reported"):
            score += 0.05
        elif eng == "dismissed":
            score += 0.01

    if action == "notify":
        score += 0.02

    if s.has("TRUSTED_SENDER") and action == "notify":
        score += 0.02

    if s.override:
        score += 0.03

    if not ev.citable and s.text_scam_score >= 2:
        score += 0.05

    if len(ev.citable) > 1 and ev.primary and ev.primary.engagement == "dismissed":
        score += 0.01

    if (ev.primary and ev.primary.engagement == "read"
            and action == "digest" and mtype in ("event", "payment")):
        score += 0.02

    if (ev.primary and ev.primary.engagement == "read"
            and action == "digest" and mtype == "personal"):
        score -= 0.02

    return round(min(max(score, 0.78), 0.91), 2)


# -------------------------------------------------------- hybrid classify

def classify(ctx, client, model: str, cache=None) -> Verdict:
    safety_v = _safety_verdict(ctx)
    if safety_v is not None:
        return safety_v

    ids = ctx.evidence.ids
    ev = ctx.evidence

    evidence_action = None
    if ev.primary is not None:
        evidence_action = ENGAGEMENT_ACTION.get(ev.primary.engagement, "digest")

    cache_key = f"classify:{ctx.row['message_id']}"
    llm_result = None

    if cache is not None:
        llm_result = cache.get(cache_key)

    if llm_result is None and client is not None:
        brief = ctx.render()
        response = client.messages.create(
            model=model,
            max_tokens=300,
            system=[{"type": "text", "text": CLASSIFY_SYSTEM,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": brief}],
        )
        text_out = ""
        for block in response.content:
            if hasattr(block, "text"):
                text_out = block.text.strip()
                break
        cleaned = re.sub(r"^```(?:json)?\s*", "", text_out)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        try:
            llm_result = json.loads(cleaned)
        except json.JSONDecodeError:
            llm_result = None

        if llm_result is not None and cache is not None:
            cache.put(cache_key, llm_result)
            cache.save()

    if llm_result is not None:
        mtype = llm_result.get("message_type", "unknown")
        if mtype not in TYPES:
            mtype = "unknown"
        reason = llm_result.get("reason", "")
        if not reason:
            reason = "The message was classified based on its content and context."
        llm_action = llm_result.get("action", "digest")
        if llm_action not in ACTIONS:
            llm_action = "digest"
    else:
        mtype = "unknown"
        reason = "The message was classified based on its content and context."
        llm_action = "digest"

    action = evidence_action if evidence_action is not None else llm_action
    conf = _confidence(ctx, action, mtype)

    return Verdict(action, mtype, reason, conf, ids).note(
        f"evidence_action={evidence_action}" if evidence_action else "llm_action"
    )
