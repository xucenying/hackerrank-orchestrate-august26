"""Stage 1 - turn media files into text, once, and cache by content hash.

images -> Claude vision (structured extraction, not a description)
audio  -> local Whisper; the Messages API has no audio input, so this cannot
          go through Claude. faster-whisper keeps it offline and deterministic.

Both are optional. When a backend is unavailable the pipeline still runs; the
message is simply routed without its media content, and that is reported rather
than silently ignored.
"""

from __future__ import annotations

import base64
from pathlib import Path

from preflight import Cache, media_key

VISION_PROMPT = """Extract facts from this image for a message-routing system.
Reply with compact lines, no prose:
text: <all readable text, verbatim>
kind: poster|screenshot|receipt|circular|photo|other
brand_claimed: <brand name or none>
has_url: yes|no  url: <url or none>
has_qr: yes|no
asks_for_payment: yes|no
asks_for_credentials: yes|no
urgency_language: yes|no"""

# File extensions lie in this dataset: 10 of the 20 .jpg files are actually PNG,
# WebP or AVIF, and the API rejects a mismatched media_type. Sniff the bytes.
MAGIC = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)
SUPPORTED = {"image/jpeg", "image/png", "image/gif", "image/webp"}
AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}


def detect_media_type(path: Path) -> str | None:
    """Real image type from magic bytes. None means 'not an image we can send'."""
    head = path.read_bytes()[:16]
    for sig, mime in MAGIC:
        if head.startswith(sig):
            return mime
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    if head[4:8] == b"ftyp":  # AVIF / HEIC - not accepted by the vision API
        return "image/avif"
    return None


class Extractor:
    def __init__(self, ds, cache_path: str | Path, client=None, model: str = "claude-opus-5",
                 whisper_model: str = "base"):
        self.ds = ds
        self.cache = Cache(cache_path)
        self.client = client
        self.model = model
        self.whisper_model = whisper_model
        self._whisper = None
        self.unavailable: list[str] = []
        self.skipped: list[str] = []

    # ------------------------------------------------------------------ api

    def text_for(self, media_id: str) -> str:
        if not media_id:
            return ""
        path = self.ds.media_file(media_id)
        if path is None or not path.exists():
            return ""
        key = media_key(path, self.model)
        hit = self.cache.get(key)
        if hit is not None:
            return hit.get("text", "")

        if path.suffix.lower() in AUDIO_SUFFIXES:
            text = self._transcribe(path)
        else:
            text = self._ocr(path)

        if text is None:
            self.unavailable.append(media_id)
            return ""
        self.cache.put(key, {"media_id": media_id, "text": text})
        # Persist immediately. Each entry cost an API call or a minute of CPU;
        # a later crash must not throw away work already paid for.
        self.cache.save()
        return text

    def save(self) -> None:
        self.cache.save()

    # -------------------------------------------------------------- backends

    def _ocr(self, path: Path) -> str | None:
        if self.client is None:
            return None
        media_type = detect_media_type(path)
        if media_type not in SUPPORTED:
            # AVIF/HEIC or an unrecognised container: the vision API would 400.
            # Report it rather than crash the run mid-way.
            self.skipped.append(f"{path.name} ({media_type or 'unknown format'})")
            return None
        data = base64.standard_b64encode(path.read_bytes()).decode("ascii")
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": data,
                    }},
                    {"type": "text", "text": VISION_PROMPT},
                ],
            }],
        )
        return "\n".join(b.text for b in resp.content if b.type == "text").strip()

    def _transcribe(self, path: Path) -> str | None:
        if self._whisper is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError:
                return None
            self._whisper = WhisperModel(self.whisper_model, device="cpu", compute_type="int8")
        segments, _ = self._whisper.transcribe(str(path), beam_size=1, language="en")
        return " ".join(s.text.strip() for s in segments).strip()
