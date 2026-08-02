# Message Notification Router

Routes every message in `dataset/messages.csv` to `notify`, `digest`, or `mute`,
personalised to the receiving user.

## Setup

Python 3.10+. Requires `ANTHROPIC_API_KEY` for retrieval and classification.

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

| Package | What it does | Required? |
|---|---|---|
| `anthropic` | LLM retrieval, classification, image OCR | Yes |
| `faster-whisper` | local voice transcription, no API key | Optional (8 voice rows) |

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
| `ANTHROPIC_API_KEY` | retrieval, classification, image OCR | required |
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
```

`--eval` runs the full pipeline on the 30 gold rows in `sample_messages.csv`
and reports action accuracy, type accuracy, confidence MAE, evidence overlap,
and a confusion matrix. It writes `message_sample_test.csv` for inspection.

## Design

Rules -> LLM -> rules. Deterministic safety and policy stages bracket the
LLM judgment call on both sides, so safety is never left to the model.

| Stage | Module | What it does |
|---|---|---|
| 0 | `loader.py` | Load 13 CSVs, build lookup tables, validate referential integrity |
| 1 | `media.py` | Image OCR (Claude vision) and voice transcription (Whisper), cached by content hash |
| 2a | `safety.py` | Danger check - impersonation, credential requests, pressure, injection |
| 2b | `retrieval.py` | LLM finds the most relevant past messages **this same user** received |
| 3 | `features.py` | Assemble one context bundle from safety + retrieval results |
| 4 | `classify.py` | Hybrid: safety rules force scam/spam, LLM decides type + reason, evidence or LLM decides action, confidence from engagement tier |
| 5 | `policy.py` | Deterministic arbitration with explicit precedence |
| 6 | `writer.py` | Validate the output contract, then write |
| 7 | `evaluate.py` | Score against `sample_messages.csv` |

Key decisions:

- **Evidence is scoped per user.** Every cited ID belongs to the receiving user;
  the writer enforces this. Cross-user matches on the same media file are kept
  as context and never cited.
- **Retrieval is LLM-based.** Claude picks the most semantically relevant past
  messages, handling paraphrases and multilingual content that word-overlap
  methods miss.
- **Reasons are LLM-generated**, one sentence per message, mirroring the style
  in `sample_messages.csv`. Safety overrides use fixed reason sentences.
- **Confidence is engagement-tier based**, not guessed by the LLM. Base 0.82,
  adjusted by evidence engagement (replied/reported +0.05, dismissed +0.01),
  action decisiveness, trusted-sender status, and safety signals. Clamped 0.78-0.91.
- **Safety overrides engagement.** Opening a phishing message is not a request
  for more of them.
- **Caches are content-addressed**, so editing a CSV row or replacing a media
  file invalidates exactly the affected items and nothing else.
