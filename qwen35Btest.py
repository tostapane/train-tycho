# Pilot for a teacher model: both passes on N recipes into out/pilot/, then the stats and a time estimate for 50k.
# Start the teacher first (vLLM, see the notes in ../documentation), then: uv run qwen35Btest.py [--count 300]
# Delete out/pilot/ to start over; a second run continues where the first stopped.

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
FOLDER = "out/pilot"
TARGET = 50_000


# Runs one of this repo's scripts and returns how long it took, in seconds.
def run(script: str, *args: str) -> float:
    start = time.time()
    subprocess.run([sys.executable, script, *args], check=True, cwd=HERE)
    return time.time() - start


def load(name: str) -> list[dict]:
    path = HERE / FOLDER / name
    return [json.loads(line) for line in path.open()] if path.exists() else []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8080")
    parser.add_argument("--count", type=int, default=300, help="recipes to draw")
    parser.add_argument("--workers", type=int, default=64, help="cases in flight at once (vLLM --max-num-seqs)")
    parser.add_argument("--think-budget", type=int, default=512)
    args = parser.parse_args()

    try:
        with urllib.request.urlopen(args.server + "/v1/models", timeout=10) as response:
            print("teacher:", json.loads(response.read())["data"][0]["id"])
    except Exception as e:
        sys.exit(f"No teacher at {args.server} ({e}). Start vLLM first.")

    common = ["--server", args.server, "--workers", str(args.workers), "--out", FOLDER]
    seconds_messages = run("generate.py", "messages", "--count", str(args.count), *common)
    seconds_runs = run("generate.py", "runs", "--count", str(args.count), *common,
                       "--think-budget", str(args.think_budget))
    run("stats.py", FOLDER)

    kept, dropped, runs = load("messages.jsonl"), load("messages-dropped.jsonl"), load("runs.jsonl")
    drawn, passed = len(kept) + len(dropped), sum(r["passed"] for r in runs)
    if not runs or not passed:
        print("\nNo passing runs: nothing to estimate from.")
        return
    keep_rate, pass_rate = len(kept) / drawn, passed / len(runs)
    recipes_needed = TARGET / (keep_rate * pass_rate)
    hours = (recipes_needed * seconds_messages / drawn + recipes_needed * keep_rate * seconds_runs / len(runs)) / 3600
    print(f"\nPilot: {drawn} recipes -> {len(kept)} messages kept ({keep_rate:.0%}) -> {passed} runs passed ({pass_rate:.0%} of {len(runs)})")
    print(f"Time: messages {seconds_messages / 60:.1f} min, runs {seconds_runs / 60:.1f} min, {args.workers} workers")
    print(f"For {TARGET} passing cases at this speed: ~{recipes_needed:,.0f} recipes, ~{hours:.1f} h")
    print("Read the fail reasons and a few passing runs in out/pilot/ before deciding.")


if __name__ == "__main__":
    main()
