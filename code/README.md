# Message Notification Router

Routes every message in `dataset/messages.csv` to `notify`, `digest`, or `mute`,
personalised to the receiving user.

## Setup

Python 3.10+. The default backend is fully offline and needs no API key.

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Windows Git Bash
source .venv/Scripts/activate

python -m pip install -r requirements.txt
```

**The baseline needs no dependencies at all.** The `rules` backend is
stdlib-only and scores 29/30 on the solved samples in an empty virtualenv;
verify with `python code/main.py --eval` before installing anything. Everything
in `requirements.txt` unlocks an optional stage:

| Package | Unlocks | Rows affected |
|---|---|---|
| `anthropic` | `--backend claude`, image OCR | 15 image rows |
| `pydantic` | structured outputs for the claude backend | - |
| `faster-whisper` | local voice transcription, no API key | 8 voice rows |

`faster-whisper` is heavy (~18 transitive packages, plus a ~145 MB model on
first run). Skip it and those 8 rows route on metadata alone, with a printed
NOTE rather than a silent degradation.

### Configuration

Secrets are read from the environment. `.env` is loaded automatically at
startup by `load_env()` in `main.py` (no extra dependency), and a real
environment variable always wins over the file.

```bash
cp .env.sample .env      # then paste your key into .env
```

| Variable | Needed for | Default |
|---|---|---|
| `ANTHROPIC_API_KEY` | `--backend claude`, image OCR | unset - rules backend runs offline |
| `ANTHROPIC_MODEL` | overriding the model | `claude-opus-5` |
| `WHISPER_MODEL` | voice-note transcription size | `base` |

`.env` is gitignored and must never be committed. `.env.sample` is the tracked
template and must never contain a real value.

Optional, for voice notes (local, offline, no key):

```bash
python -m pip install faster-whisper
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
