"""Stage 8 - score the pipeline against the 30 solved rows in sample_messages.csv.

Separate ID space from messages.csv, so there is no leakage. 30 rows is a smoke
test and regression guard, not a leaderboard.
"""

from __future__ import annotations

from collections import Counter, defaultdict

ACTION_ORDER = ["notify", "digest", "mute"]


def score(predictions: dict[str, object], ds) -> dict:
    gold = {r["message_id"]: r for r in ds.samples}
    rows = [(mid, predictions[mid], gold[mid]) for mid in gold if mid in predictions]

    action_hits = sum(1 for _, p, g in rows if p.action == g["action"])
    type_hits = sum(1 for _, p, g in rows if p.message_type == g["message_type"])
    both = sum(1 for _, p, g in rows if p.action == g["action"] and p.message_type == g["message_type"])

    ev_gold = [(mid, p, g) for mid, p, g in rows if g["evidence_message_ids"] != "none"]
    ev_hit = sum(1 for _, p, g in ev_gold if set(p.evidence_ids) & set(g["evidence_message_ids"].split(";")))
    ev_none_gold = [(mid, p, g) for mid, p, g in rows if g["evidence_message_ids"] == "none"]
    ev_none_hit = sum(1 for _, p, _ in ev_none_gold if not p.evidence_ids)

    # The gold citations in sample_messages.csv are purpose-built companion rows
    # (sample_msg_NNN pairs with message_00NN), and the corpus contains exact
    # text duplicates of them. Citing a duplicate is materially the same
    # evidence, so also score whether the citation carries the same engagement
    # label - i.e. whether it argues for the same routing decision.
    ev_equiv = 0
    for _, p, g in ev_gold:
        goldset = set(g["evidence_message_ids"].split(";"))
        got = set(p.evidence_ids)
        if got & goldset:
            ev_equiv += 1
        elif got:
            gold_states = {ds.engagement.get(x) for x in goldset}
            got_states = {ds.engagement.get(x) for x in got}
            if gold_states & got_states:
                ev_equiv += 1

    confusion: dict[str, Counter] = defaultdict(Counter)
    for _, p, g in rows:
        confusion[g["action"]][p.action] += 1

    type_confusion: dict[str, Counter] = defaultdict(Counter)
    for _, p, g in rows:
        if p.message_type != g["message_type"]:
            type_confusion[g["message_type"]][p.message_type] += 1

    by_conv: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_media: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for mid, p, g in rows:
        by_conv[g["conversation_type"]][1] += 1
        by_media[g["media_type"] or "text"][1] += 1
        if p.action == g["action"]:
            by_conv[g["conversation_type"]][0] += 1
            by_media[g["media_type"] or "text"][0] += 1

    misses = [
        {
            "message_id": mid,
            "gold": f"{g['action']}/{g['message_type']}",
            "pred": f"{p.action}/{p.message_type}",
            "conf": p.confidence,
            "trace": ", ".join(p.trace),
            "text": (g["message_text"] or "(media only)")[:70],
        }
        for mid, p, g in rows
        if p.action != g["action"] or p.message_type != g["message_type"]
    ]

    confs = [p.confidence for _, p, _ in rows]
    return {
        "n": len(rows),
        "action_acc": action_hits / len(rows) if rows else 0.0,
        "type_acc": type_hits / len(rows) if rows else 0.0,
        "both_acc": both / len(rows) if rows else 0.0,
        "action_hits": action_hits,
        "type_hits": type_hits,
        "both_hits": both,
        "evidence_recall": ev_hit / len(ev_gold) if ev_gold else 0.0,
        "evidence_n": len(ev_gold),
        "evidence_hits": ev_hit,
        "evidence_equiv": ev_equiv,
        "evidence_equiv_rate": ev_equiv / len(ev_gold) if ev_gold else 0.0,
        "none_correct": ev_none_hit,
        "none_n": len(ev_none_gold),
        "confusion": confusion,
        "type_confusion": type_confusion,
        "by_conv": dict(by_conv),
        "by_media": dict(by_media),
        "conf_min": min(confs) if confs else 0,
        "conf_max": max(confs) if confs else 0,
        "misses": misses,
    }


def render(s: dict) -> str:
    L = [f"EVALUATION on sample_messages.csv  (n={s['n']})", ""]
    L.append(f"  action accuracy ........ {s['action_hits']:>2}/{s['n']}  {s['action_acc']:.0%}")
    L.append(f"  message_type accuracy .. {s['type_hits']:>2}/{s['n']}  {s['type_acc']:.0%}")
    L.append(f"  both correct ........... {s['both_hits']:>2}/{s['n']}  {s['both_acc']:.0%}")
    L.append(f"  evidence exact id ...... {s['evidence_hits']:>2}/{s['evidence_n']}  {s['evidence_recall']:.0%}")
    L.append(f"  evidence same-decision . {s['evidence_equiv']:>2}/{s['evidence_n']}  {s['evidence_equiv_rate']:.0%}"
             f"   (correct 'none': {s['none_correct']}/{s['none_n']})")
    L.append(f"  confidence range ....... {s['conf_min']:.2f} - {s['conf_max']:.2f}")
    L.append("")
    L.append("  action confusion (rows = gold, cols = predicted)")
    L.append("           " + "".join(f"{a:>9}" for a in ACTION_ORDER))
    for gold_a in ACTION_ORDER:
        row = s["confusion"].get(gold_a, {})
        L.append(f"  {gold_a:<8}" + "".join(f"{row.get(a, 0):>9}" for a in ACTION_ORDER))
    L.append("")
    L.append("  action accuracy by slice")
    for name, table in (("conversation_type", s["by_conv"]), ("media_type", s["by_media"])):
        L.append(f"    {name}")
        for k, (hit, tot) in sorted(table.items()):
            L.append(f"      {k:<12} {hit:>2}/{tot:<3} {hit / tot:.0%}")
    if s["misses"]:
        L.append("")
        L.append(f"  misses ({len(s['misses'])})")
        L.append(f"    {'id':<16}{'gold':<22}{'predicted':<22}{'conf':<6}text")
        for m in s["misses"]:
            L.append(f"    {m['message_id']:<16}{m['gold']:<22}{m['pred']:<22}{m['conf']:<6.2f}{m['text']}")
    return "\n".join(L)
