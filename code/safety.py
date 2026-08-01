"""Stage 3 - the danger check.

Pure functions over CSV rows and text. No API calls, so this is the one stage
that is fully unit-testable offline. Output is consumed twice: as features in
the classifier prompt, and as override authority in policy.py.

Only IMPERSONATION and INJECTION are override-grade. Everything else is a weight.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

YOUNG_DOMAIN_DAYS = 60
HIGH_REPORTS = 30

SHORTENERS = (
    "shorturl.at", "weurl.co", "vl.gl", "link.wame.pro", "bit.ly",
    "tinyurl.com", "t.co", "rb.gy", "cutt.ly", "is.gd",
)

CREDENTIAL_RE = re.compile(
    r"\b(otp|one[- ]time (pass|code)|pin\b|cvv|kyc|"
    r"(confirm|share|enter|reply with)[^.]{0,30}\b(password|pin|otp|code|card number)|"
    r"verify (your )?(account|profile|identity|wallet)|"
    r"account (re[- ]?)?verification|re[- ]?verify|update (your )?kyc|"
    # Asking for bank or card details is a credential request even when the
    # words OTP and password never appear. The object has to be specific:
    # a bare "card" also matches "ID card" and "report card", which is how a
    # school consent form briefly got classified as phishing.
    r"(bank|account|card) (details|number)|"
    r"(fill|share|send|provide)[^.]{0,25}"
    r"(bank details|account (number|details)|card (number|details)|ifsc|upi id))\b",
    re.I,
)
PRESSURE_RE = re.compile(
    r"\b(blocked? (in|within|after)\s*\d|will be (blocked|suspended|deactivated|closed)|"
    r"expire[sd]?\s*(today|soon|in\s*\d)|before midnight|last chance|"
    r"immediately or|within \d+\s*(min|hour)s?\b[^.]{0,40}(block|suspend|penalt))\b",
    re.I,
)
PRIZE_RE = re.compile(
    r"\b(congrats|congratulations|you(r number)? (have|has|were|was) (been )?(selected|chosen|won)|"
    r"lucky draw|prize|voucher|scratch card|claim (your|the) (reward|prize|gift|cashback))\b",
    re.I,
)
SHORTENER_IN_TEXT_RE = re.compile(
    r"\b(" + "|".join(s.replace(".", r"\.") for s in SHORTENERS) + r")\b"
    r"|\b[a-z0-9-]+\.(?:link|click|top|xyz|gq|tk)\b",
    re.I,
)
INJECTION_RE = re.compile(
    r"(ignore (all )?(previous|prior|above)|disregard (the )?(previous|prior|above|rules)|"
    r"mark this (message )?as|route this as|you are an? (ai|assistant|model)|"
    r"system prompt|new instructions?:|override (the )?(rules|routing))",
    re.I,
)
PAYMENT_REQUEST_RE = re.compile(
    r"\b(scan (this |the )?qr|pay (now|the|via|small|reattempt)|make (the )?payment|"
    r"transfer (the )?(amount|money|fund)|upi|reattempt fee|processing fee|"
    r"pay[^.]{0,20}(link|online)|complete (the )?payment)\b",
    re.I,
)

# Signals that argue for scam even without a bad domain.
STRONG_TEXT_FLAGS = ("CREDENTIAL_REQUEST", "PRESSURE", "PRIZE_BAIT")


@dataclass
class Safety:
    flags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def has(self, *names: str) -> bool:
        return any(n in self.flags for n in names)

    @property
    def override(self) -> str | None:
        """The two flags trusted enough to force the final answer."""
        if "INJECTION" in self.flags:
            return "INJECTION"
        if "IMPERSONATION" in self.flags:
            return "IMPERSONATION"
        return None

    @property
    def text_scam_score(self) -> int:
        return sum(1 for f in STRONG_TEXT_FLAGS if f in self.flags)


def evaluate(row: dict, ds, media_text: str = "") -> Safety:
    """Danger check for one message row. `media_text` is OCR/ASR output if any."""
    out = Safety()
    blob = f"{row.get('message_text', '')}\n{media_text}"

    bid = row.get("business_id") or ""
    if bid and bid in ds.businesses:
        acct = ds.businesses[bid]
        official = (acct.get("official_domain") or "").strip()
        used = (acct.get("domain_used_by_sender") or "").strip()
        verified = acct.get("verified") == "1"

        # A verified brand, on its own domain, with a long history is trusted.
        # It can legitimately mention OTPs (fraud advisories) or payments
        # (real invoices) without that being evidence of phishing.
        try:
            account_age = int(acct.get("account_age_days") or 0)
            reports = int(acct.get("user_reports_30d") or 0)
        except ValueError:
            account_age, reports = 0, 999
        if verified and official and official == used and account_age >= 365 and reports < HIGH_REPORTS:
            out.flags.append("TRUSTED_SENDER")

        # Blank official_domain is "nothing to compare against", not a mismatch.
        if official and used and official != used and not verified:
            out.flags.append("IMPERSONATION")
            out.notes.append(
                f"claims {acct.get('brand_name')!r} but links {used!r} "
                f"instead of {official!r}, unverified"
            )
        elif official and used and official != used:
            out.flags.append("DOMAIN_MISMATCH_VERIFIED")
            out.notes.append(f"verified account using off-brand domain {used!r}")

        if (acct.get("brand_name") or "").strip().lower() in ("", "unknown"):
            out.flags.append("UNKNOWN_BRAND")

        try:
            age = int(acct.get("domain_used_by_sender_age_days") or 0)
            if age and age < YOUNG_DOMAIN_DAYS:
                out.flags.append("YOUNG_DOMAIN")
                out.notes.append(f"sender domain is {age} days old")
        except ValueError:
            pass

        try:
            if int(acct.get("user_reports_30d") or 0) >= HIGH_REPORTS:
                out.flags.append("HIGH_REPORTS")
        except ValueError:
            pass

        if any(s in used.lower() for s in SHORTENERS):
            out.flags.append("SHORTENER")

    # A shortened or lookalike link is a risk signal wherever it appears. The
    # domain checks above only inspect business_accounts.csv, so a bit.ly in the
    # body of a group or personal message was previously invisible.
    if SHORTENER_IN_TEXT_RE.search(blob):
        out.flags.append("SUSPICIOUS_LINK")
        out.notes.append("message body contains a shortened or redirect link")

    if CREDENTIAL_RE.search(blob):
        out.flags.append("CREDENTIAL_REQUEST")
    if PRESSURE_RE.search(blob):
        out.flags.append("PRESSURE")
    if PRIZE_RE.search(blob):
        out.flags.append("PRIZE_BAIT")
    if INJECTION_RE.search(blob):
        out.flags.append("INJECTION")
        out.notes.append("message text attempts to instruct the router")
    if PAYMENT_REQUEST_RE.search(blob):
        out.flags.append("PAYMENT_REQUEST")

    try:
        fwd = int(row.get("forwarded_count") or 0)
    except ValueError:
        fwd = 0
    if fwd >= 5:
        out.flags.append("VIRAL_FORWARD")
    elif fwd >= 1:
        out.flags.append("FORWARDED")

    # No prior relationship with a business is a cold-start signal.
    if bid and (row.get("user_id"), bid) not in ds.biz_rel:
        out.flags.append("NO_BUSINESS_RELATIONSHIP")

    return out
