# Reads out/data (or the folder given) and prints how generation went; writes cases.csv, one row per drawn recipe.
# Safe to run any time, even while generate.py is running. Usage: uv run stats.py [DIR]

import csv
import json
import re
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from gen import recipes


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


# Speed over the whole pass: cases per minute and completion tokens per second, from first start to last end.
def rate(used: list[dict]) -> str:
    if len(used) < 2:
        return "n/a"
    ends = [datetime.fromisoformat(u["at"]) for u in used]
    span = (max(ends) - min(e - timedelta(seconds=u["seconds"]) for e, u in zip(ends, used))).total_seconds()
    return f"{60 * len(used) / span:.1f} cases/min, {sum(u['completion_tokens'] for u in used) / span:.0f} tokens/s"


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
    for c in dropped:
        c.setdefault("category", recipes.draw(c["seed"])["category"])
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
        drawn = sum(c["category"] == name for c in kept + dropped)
        mine = [r for r in runs if r["category"] == name]
        print(f"  {name:<10} drawn {drawn:>4} ({pct(drawn, len(kept) + len(dropped)):>4}, target {weight:.0%})"
              f"  runs {len(mine):>4}  passed {pct(sum(r['passed'] for r in mine), len(mine)):>4}")

    print("\nWording variety per kind (kept messages, values blanked):")
    for kind in sorted({c["kind"] for c in kept}):
        mine = [mask(c) for c in kept if c["kind"] == kind]
        print(f"  {kind:<16} kept {len(mine):>4}  distinct wordings {len(set(mine)):>4}")

    print("\nPer kind:")
    for kind in sorted({r["kind"] for r in runs}):
        mine = [r for r in runs if r["kind"] == kind]
        why = Counter(r["problem"] for r in mine if not r["passed"]).most_common(2)
        print(f"  {kind:<16} runs {len(mine):>4}  passed {pct(sum(r['passed'] for r in mine), len(mine)):>4}  "
              + "; ".join(f"{n} {p}" for p, n in why))

    print(f"\nPassing runs, tools called:")
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
    for label, key in (("messages", "write"), ("runs", "play")):
        used = [c[key] for c in kept + runs if key in c]
        print(f"  {label:<9} completion tokens/case {mean([u['completion_tokens'] for u in used])}, "
              f"{rate(used)}  ({len(used)} cases measured)")
    print(f"  tool calls per run {sum(len(calls(r)) for r in runs) / len(runs):.1f}" if runs else "")
    print(f"  empty answers {sum(map(empty, runs))} of {len(runs)}")

    rows = []
    for c in dropped:
        rows.append({"stage": "dropped", "reason": c["dropped"], **c})
    for c in kept:
        if c["seed"] not in ran:
            rows.append({"stage": "not run", "reason": "", **c})
    for r in runs:
        rows.append({"stage": "passed" if r["passed"] else "failed", "reason": r["problem"] or "", **r,
                     "tool_calls": len(calls(r)), "empty_answer": empty(r)})
    fields = ["seed", "category", "kind", "style", "shape", "stage", "reason", "messages", "tool_calls", "empty_answer",
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
