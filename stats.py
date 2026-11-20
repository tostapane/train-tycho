# Reads out/data (or the folder given) and prints how generation went; writes cases.csv, one row per drawn recipe.
# Also estimates the hours for TARGET passing cases. Safe to run any time, even while generate.py is running.
# Usage: uv run stats.py [DIR]

import csv
import json
import re
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from gen import recipes

TARGET = 50_000  # passing cases wanted from the full run


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()] if path.exists() else []


def pct(a: int, b: int) -> str:
    return f"{100 * a / b:.0f}%" if b else "-"


def mean(values: list[float]) -> str:
    return f"{sum(values) / len(values):.0f}" if values else "n/a"


# A message with its recipe values (names, titles, numbers) blanked: what is left is its wording.
def mask(case: dict) -> str:
    text = case["messages"][0].lower()
    for phrase in sorted((p for p in case["say"] if "|" not in p), key=len, reverse=True):
        text = text.replace(phrase.lower(), "_")
    return re.sub(r"\d+", "#", text)


# Seconds when at least one case was running: pauses between runs (a resumed pass) don't count.
def busy(used: list[dict]) -> float:
    spans = sorted((datetime.fromisoformat(u["at"]) - timedelta(seconds=u["seconds"]), datetime.fromisoformat(u["at"]))
                   for u in used)
    total, end = 0.0, None
    for a, b in spans:
        a = max(a, end or a)
        total += max(0.0, (b - a).total_seconds())
        end = max(b, end or b)
    return total


# Speed over a pass: cases per minute and completion tokens per second, while busy.
def rate(used: list[dict]) -> str:
    seconds = busy(used)
    if not seconds:
        return "n/a"
    return f"{60 * len(used) / seconds:.1f} cases/min, {sum(u['completion_tokens'] for u in used) / seconds:.0f} tokens/s"


# What was asked of the teacher for one case: tokens and seconds, from both passes (blank if the record has none).
def cost(case: dict) -> dict:
    row = {}
    for name in ("write", "play"):
        for field in ("prompt_tokens", "completion_tokens", "seconds", "requests"):
            row[f"{name}_{field}"] = case.get(name, {}).get(field, "")
    play = case.get("play", {})
    return row | {"teacher": play.get("teacher") or case.get("write", {}).get("teacher", ""),
                  "think_budget": play.get("think_budget", ""), "at": play.get("at") or case.get("write", {}).get("at", "")}


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "out/data"
    kept, dropped, runs = (load(folder / n) for n in ("messages.jsonl", "messages-dropped.jsonl", "runs.jsonl"))
    ran = {r["seed"] for r in runs}
    passed = [r for r in runs if r["passed"]]
    calls = lambda r: [e["tool"] for e in r["run"]["log"] if "tool" in e]
    empty = lambda r: r["run"]["answer"] == "" and not r["run"].get("confirm")

    print(f"{folder}: drawn {len(kept) + len(dropped)}, messages kept {len(kept)}, dropped {len(dropped)}; "
          f"runs {len(runs)}, passed {len(passed)} ({pct(len(passed), len(runs))})\n")

    print("Message drops:")
    for reason, n in Counter(c["dropped"].split(" [")[0] for c in dropped).most_common():
        print(f"  {n:>4}  {reason}")

    print("\nPer category (share = of drawn recipes; target from recipes.py):")
    for name, (weight, _) in recipes.CATEGORIES.items():
        drawn = sum(c.get("category") == name for c in kept + dropped)
        mine = [r for r in runs if r["category"] == name]
        print(f"  {name:<10} drawn {drawn:>4} ({pct(drawn, len(kept) + len(dropped)):>4}, target {weight:.0%})"
              f"  runs {len(mine):>4}  passed {pct(sum(r['passed'] for r in mine), len(mine)):>4}")

    print("\nWording variety per kind (kept messages, values blanked):")
    for kind in sorted({c["kind"] for c in kept}):
        mine = [mask(c) for c in kept if c["kind"] == kind]
        print(f"  {kind:<16} kept {len(mine):>4}  distinct wordings {len(set(mine)):>4}")

    print("\nPer kind (accepted share of all passing runs):")
    for kind in sorted({c["kind"] for c in kept + dropped}):
        mine = [r for r in runs if r["kind"] == kind]
        why = Counter(r["problem"] for r in mine if not r["passed"]).most_common(2)
        count = sum(r["passed"] for r in mine)
        print(f"  {kind:<16} runs {len(mine):>4}  accepted {count:>4}  rate {pct(count, len(mine)):>4}"
              f"  share {pct(count, len(passed)):>4}  "
              + "; ".join(f"{n} {p}" for p, n in why))

    print("\nPassing runs, tools called:")
    tools = Counter(t for r in passed for t in set(calls(r)))
    for tool, n in tools.most_common():
        print(f"  {tool:<16} {n:>4} {pct(n, len(passed)):>4}")
    print(f"  (no tool)        {sum(not calls(r) for r in passed):>4}")

    print("\nPassing runs, first word of the message:")
    first = Counter(re.findall(r"[a-z0-9']+", r["messages"][0].lower())[0] for r in passed
                    if re.search(r"[a-z0-9]", r["messages"][0].lower()))
    for word, n in first.most_common(10):
        print(f"  {word:<16} {n:>4} {pct(n, len(passed)):>4}")

    print("\nCost (teacher, both passes):")
    write = [c["write"] for c in kept + dropped if "write" in c]
    play = [r["play"] for r in runs if "play" in r]
    for label, used in (("messages", write), ("runs", play)):
        print(f"  {label:<9} completion tokens/case {mean([u['completion_tokens'] for u in used])}, "
              f"{rate(used)}  ({len(used)} cases measured)")
    print(f"  tool calls per run {sum(len(calls(r)) for r in runs) / len(runs):.1f}" if runs else "")
    print(f"  empty answers {sum(map(empty, runs))} of {len(runs)}")
    steps = [e for r in runs for e in r["run"]["log"] if e.get("think") and "think_cut" in e]  # older records lack it
    if steps:
        print(f"  thinking cut at the budget: {sum(e['think_cut'] for e in steps)} of {len(steps)} steps")
    batches = load(folder / "batches.jsonl")
    elapsed = Counter()
    for batch in batches:
        elapsed[batch["stage"]] += batch["seconds"]
    if batches:
        print(f"  timed batches: {sum(elapsed.values()) / 3600:.3f} h, "
              f"{sum(b['errors'] for b in batches)} retryable errors")
    if passed and elapsed["messages"] and elapsed["runs"]:
        drawn = len(kept) + len(dropped)
        keep_rate = len(kept) / drawn
        need = TARGET / (keep_rate * len(passed) / len(runs))
        hours = need * (elapsed["messages"] / drawn + keep_rate * elapsed["runs"] / len(runs)) / 3600
        print(f"  for {TARGET:,} passing cases at this speed: ~{need:,.0f} recipes, ~{hours:.1f} generation hours")
        print("  add billed setup, model loading, pauses and transfers at the instance rate")
    elif not batches:
        print("  historical folder has no batch wall times; rental cost unknown")
    print("  accepted split:", dict(Counter(r.get("split", "unassigned") for r in passed)))

    rows = []
    for c in dropped:
        rows.append({"stage": "dropped", "reason": c["dropped"], **c})
    for c in kept:
        if c["seed"] not in ran:
            rows.append({"stage": "not run", "reason": "", **c})
    for r in runs:
        rows.append({"stage": "passed" if r["passed"] else "failed", "reason": r["problem"] or "", **r,
                     "tool_calls": len(calls(r)), "empty_answer": empty(r)})
    fields = ["seed", "split", "category", "kind", "style", "shape", "stage", "reason", "messages", "tool_calls", "empty_answer",
              "write_prompt_tokens", "write_completion_tokens", "write_seconds", "write_requests",
              "play_prompt_tokens", "play_completion_tokens", "play_seconds", "play_requests",
              "teacher", "think_budget", "at"]
    with (folder / "cases.csv").open("w", newline="") as f:
        out = csv.DictWriter(f, fields, extrasaction="ignore")
        out.writeheader()
        for row in sorted(rows, key=lambda x: x["seed"]):
            out.writerow(row | {"messages": " >> ".join(row["messages"]), **cost(row)})
    print(f"\nwrote {folder / 'cases.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
