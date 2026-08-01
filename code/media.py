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

MEDIA_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


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

        if path.suffix.lower() in MEDIA_TYPES:
            text = self._ocr(path)
        else:
            text = self._transcribe(path)

        if text is None:
            self.unavailable.append(media_id)
            return ""
        self.cache.put(key, {"media_id": media_id, "text": text})
        return text

    def save(self) -> None:
        self.cache.save()

    # -------------------------------------------------------------- backends

    def _ocr(self, path: Path) -> str | None:
        if self.client is None:
            return None
        data = base64.standard_b64encode(path.read_bytes()).decode("ascii")
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            output_config={"effort": "medium"},
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64",
                        "media_type": MEDIA_TYPES[path.suffix.lower()],
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
