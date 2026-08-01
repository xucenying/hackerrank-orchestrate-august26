"""Message Notification Router - entry point.

  python code/main.py                    route dataset/messages.csv -> output.csv
  python code/main.py --eval             score against the 30 solved sample rows
  python code/main.py --dry-run          preflight report only, no work

Requires ANTHROPIC_API_KEY for retrieval and classification.
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
import regression  # noqa: E402
import retrieval  # noqa: E402
import safety  # noqa: E402
import writer  # noqa: E402
from loader import Dataset  # noqa: E402
from media import Extractor  # noqa: E402
from preflight import Cache, Report, check_media, estimate_cost  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
CACHE_DIR = ROOT / "code" / "cache"
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


def run(rows, ds, extractor, client, llm_cache, model):
    results = []
    for row in rows:
        media_text = extractor.text_for(row.get("media_id") or "")
        flags = safety.evaluate(row, ds, media_text)
        found = retrieval.retrieve(row, ds, client, model, cache=llm_cache, media_text=media_text)
        ctx = features.build(row, ds, flags, found, media_text)
        verdict = classify.classify(ctx, client, model, cache=llm_cache)
        final, _ = policy.apply(verdict, ctx)
        results.append((row["message_id"], final))
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="WhatsApp message notification router")
    ap.add_argument("--eval", action="store_true", help="score against sample_messages.csv")
    ap.add_argument("--dry-run", action="store_true", help="preflight report only")
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--out", default=str(ROOT / "output.csv"))
    ap.add_argument("--diff", action="store_true",
                    help="compare against the committed output.csv without writing")
    ap.add_argument("--accept", action="store_true",
                    help="write the new output.csv even when the diff flags CRITICAL moves")
    args = ap.parse_args()

    load_env()
    if args.model == "claude-opus-5" and os.environ.get("ANTHROPIC_MODEL"):
        args.model = os.environ["ANTHROPIC_MODEL"]

    ds = Dataset.load(DATASET)
    problems = ds.check()

    client = get_client()
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
    llm_stale = len(target)

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
    if client is None:
        print("STOP: retrieval requires ANTHROPIC_API_KEY.", file=sys.stderr)
        return 4

    try:
        results = run(target, ds, extractor, client, llm_cache, args.model)
    finally:
        # A crash must never discard work already paid for.
        extractor.save()
        llm_cache.save()

    if extractor.unavailable:
        missing = sorted(set(extractor.unavailable))
        print(f"NOTE: {len(missing)} media file(s) routed without extracted content "
              f"(no vision/ASR backend available): {', '.join(missing)}\n")

    if args.eval:
        pred = dict(results)
        stats = evaluate.score(pred, ds)
        print(evaluate.render(stats))
        test_path = evaluate.write_test_csv(pred, ds, ROOT / "message_sample_test.csv")
        print(f"\nwrote {len(pred)} rows -> {test_path}")
        return 0

    rows = writer.to_rows(results, ds)
    writer.validate(rows, ds, expect_order=ds.output_order or None)

    # The sample eval only ever sees 30 rows. This is the only check that looks
    # at all 110, and the only one that can catch a change moving messages
    # across the mute/notify boundary.
    diff = regression.compare(rows, args.out, ds, results)
    print(diff.render())
    print()

    if args.diff:
        return 0 if diff.ok else 1
    if not diff.ok and not args.accept:
        print("STOP: changes crossed the mute/notify boundary. Review the rows "
              "above, then re-run with --accept to write anyway.", file=sys.stderr)
        return 5

    path = writer.write(rows, args.out)
    print(f"wrote {len(rows)} rows -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
