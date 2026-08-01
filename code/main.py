"""Message Notification Router - entry point.

  python code/main.py                    route dataset/messages.csv -> output.csv
  python code/main.py --eval             score against the 30 solved sample rows
  python code/main.py --dry-run          preflight report only, no work
  python code/main.py --backend claude   use the API (needs ANTHROPIC_API_KEY)

Secrets come from the environment only. Nothing is hardcoded.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import classify  # noqa: E402
import evaluate  # noqa: E402
import features  # noqa: E402
import policy  # noqa: E402
import retrieval  # noqa: E402
import safety  # noqa: E402
import writer  # noqa: E402
from loader import Dataset  # noqa: E402
from media import Extractor  # noqa: E402
from preflight import Cache, Report, check_media, estimate_cost, llm_key  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
CACHE_DIR = ROOT / "code" / "cache"
PROMPT_PATH = ROOT / "code" / "prompts" / "system.md"
ENV_PATH = ROOT / ".env"


def load_env(path: Path = ENV_PATH) -> None:
    """Read KEY=VALUE lines from .env into the environment.

    A real environment variable always wins, so CI and shell exports override
    the file. Blank values are skipped so an unfilled template is a no-op.
    Kept dependency-free on purpose - this is ~15 lines, not a package.
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and value and key not in os.environ:
            os.environ[key] = value


def get_client():
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return None
    try:
        import anthropic
    except ImportError:
        return None
    return anthropic.Anthropic()


def run(rows, ds, extractor, backend, client, system_prompt, llm_cache, model):
    index = retrieval.Index(ds)
    results = []
    for row in rows:
        media_text = extractor.text_for(row.get("media_id") or "")
        flags = safety.evaluate(row, ds, media_text)
        found = retrieval.retrieve(row, ds, index, media_text)
        ctx = features.build(row, ds, flags, found, media_text)

        if backend == "claude" and client is not None:
            key = llm_key(system_prompt, model, ctx.render())
            cached = llm_cache.get(key)
            if cached:
                verdict = classify.Verdict(**cached)
            else:
                verdict = classify.classify_claude(ctx, client, system_prompt, model)
                llm_cache.put(key, verdict.__dict__)
                llm_cache.save()  # each entry is a paid call - persist immediately
        else:
            verdict = classify.classify_rules(ctx)

        final, _ = policy.apply(verdict, ctx)
        results.append((row["message_id"], final))
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="WhatsApp message notification router")
    ap.add_argument("--eval", action="store_true", help="score against sample_messages.csv")
    ap.add_argument("--dry-run", action="store_true", help="preflight report only")
    ap.add_argument("--backend", choices=("rules", "claude"), default="rules")
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--out", default=str(ROOT / "output.csv"))
    ap.add_argument("--offline", action="store_true", help="fail if anything needs the network")
    args = ap.parse_args()

    load_env()
    if args.model == "claude-opus-5" and os.environ.get("ANTHROPIC_MODEL"):
        args.model = os.environ["ANTHROPIC_MODEL"]

    ds = Dataset.load(DATASET)
    problems = ds.check()

    client = get_client() if args.backend == "claude" else None
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8") if PROMPT_PATH.exists() else ""

    llm_cache = Cache(CACHE_DIR / "llm_cache.json")
    extractor = Extractor(
        ds,
        CACHE_DIR / "media_cache.json",
        client=client,
        model=args.model,
        whisper_model=os.environ.get("WHISPER_MODEL") or "base",
    )

    media_total, media_stale = check_media(ds, extractor.cache, args.model)
    target = ds.samples if args.eval else ds.messages
    llm_stale = len(target) if args.backend == "claude" else 0

    report = Report(
        dataset_ok=not problems,
        problems=problems,
        media_total=media_total,
        media_stale=media_stale,
        llm_total=len(target),
        llm_stale=llm_stale,
        est_cost_usd=estimate_cost(llm_stale) if llm_stale else 0.0,
    )
    print(report.render())
    print()

    if problems:
        print("STOP: dataset contract changed. Fix the above before running.", file=sys.stderr)
        return 2
    if args.dry_run:
        return 0
    if args.offline and (media_stale or llm_stale):
        print("STOP: --offline set but work requires the network.", file=sys.stderr)
        return 3
    if args.backend == "claude" and client is None:
        print("STOP: --backend claude needs ANTHROPIC_API_KEY.", file=sys.stderr)
        return 4

    try:
        results = run(target, ds, extractor, args.backend, client, system_prompt, llm_cache, args.model)
    finally:
        # A crash must never discard work already paid for.
        extractor.save()
        llm_cache.save()

    if extractor.unavailable:
        missing = sorted(set(extractor.unavailable))
        print(f"NOTE: {len(missing)} media file(s) routed without extracted content "
              f"(no vision/ASR backend available): {', '.join(missing)}\n")

    if args.eval:
        stats = evaluate.score(dict(results), ds)
        print(evaluate.render(stats))
        return 0

    rows = writer.to_rows(results, ds)
    writer.validate(rows, ds, expect_order=ds.output_order or None)
    path = writer.write(rows, args.out)
    print(f"wrote {len(rows)} rows -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
