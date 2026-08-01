# Message Notification Router

Routes every message in `dataset/messages.csv` to `notify`, `digest`, or `mute`,
personalised to the receiving user.

## Setup

```bash
python -m pip install -r requirements.txt
```

Python 3.10+. The default backend is fully offline and needs no API key.

Optional, for the media and LLM stages:

```bash
export ANTHROPIC_API_KEY=...        # image OCR + the claude classifier backend
python -m pip install faster-whisper  # voice-note transcription (local, offline)
```

## Run

```bash
python code/main.py                    # write output.csv
python code/main.py --eval             # score against the 30 solved samples
python code/main.py --dry-run          # preflight report only, no work
python code/main.py --backend claude   # LLM classifier (needs ANTHROPIC_API_KEY)
python code/main.py --offline          # fail rather than touch the network
```

## Design

Rules -> model -> rules. Deterministic stages bracket the judgment call on both
sides, so safety policy is never left to the classifier.

| Stage | Module | What it does |
|---|---|---|
| 0 | `loader.py` | Load 13 CSVs, build lookup tables, validate referential integrity |
| 1 | `media.py` | Image OCR (Claude vision) and voice transcription (Whisper), cached by content hash |
| 2 | `features.py` | Assemble one context bundle per message |
| 3 | `safety.py` | Danger check - impersonation, credential requests, pressure, injection |
| 4 | `retrieval.py` | Find similar past messages **this same user** received |
| 5 | `classify.py` | Decide action / type / reason_code / confidence (`rules` or `claude`) |
| 6 | `policy.py` | Deterministic arbitration with explicit precedence |
| 7 | `writer.py` | Validate the output contract, then write |
| 8 | `evaluate.py` | Score against `sample_messages.csv` |

Key decisions:

- **Evidence is scoped per user.** Every cited ID belongs to the receiving user;
  the writer enforces this. Cross-user matches on the same media file are kept
  as context and never cited.
- **Retrieval blends content and structure.** Structural joins (same media, same
  business, same sender) are a bonus on lexical similarity, not an override.
- **Reasons come from a fixed template table**, selected by an enum `reason_code`,
  so phrasing stays consistent across all rows.
- **Safety overrides engagement.** Opening a phishing message is not a request
  for more of them.
- **Caches are content-addressed**, so editing a CSV row or replacing a media
  file invalidates exactly the affected items and nothing else.

## Results (rules backend, offline)

Measured with `python code/main.py --eval`:

```
action accuracy ........ 29/30  97%
message_type accuracy .. 28/30  93%
both correct ........... 28/30  93%
evidence same-decision . 26/28  93%
```

Both remaining misses are voice notes routed without transcription. Install
`faster-whisper` to close them.
