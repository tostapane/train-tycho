# Smoke test for a new machine: is the repo intact, is the GPU/disk/vLLM there, does the teacher answer, and does
# the real generator write, play, save and resume a few cases? Run it before starting vLLM (1-2) and after (3-4).
# Usage: uv run smoke.py [--server http://127.0.0.1:8080] [--parallel 32]. Exit code 1 if a FAIL line was printed.

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
FAILS = 0
MIN_VLLM = (0, 17, 0)
MIN_DISK_GB = 100  # both teachers: 27B 30.9 GB + 35B-A3B 37.5 GB, plus room for the data


def say(level: str, text: str):
    global FAILS
    FAILS += level == "FAIL"
    print(f"{level:<5} {text}", flush=True)


# Runs a command and returns its output, or None if it isn't there or fails.
def shell(*command: str) -> str | None:
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=60, cwd=HERE)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return (done.stdout + done.stderr).strip() if done.returncode == 0 else None


def check_repo():
    print("\n1. Repo")
    say("ok" if sys.version_info[:2] == (3, 14) else "WARN", f"python {sys.version.split()[0]} (pyproject pins 3.14)")
    commit = shell("git", "rev-parse", "--short", "HEAD")
    dirty = shell("git", "status", "--short")
    say("ok" if commit and not dirty else "WARN", f"git commit {commit or '?'}{' (uncommitted changes)' if dirty else ''}")
    try:
        from app.grammar import GRAMMAR_FILE, build
        from gen import checks, recipes
        say("ok" if GRAMMAR_FILE.read_text() == build() else "FAIL", "app/hven.gbnf matches app/tools.json")
        kinds = {recipes.draw(seed)["kind"] for seed in range(300)}
        missing = kinds - set(recipes.ALLOWED_TOOLS)
        say("FAIL" if missing else "ok", f"300 recipes drawn, {len(kinds)} kinds"
            + (f"; no ALLOWED_TOOLS for {', '.join(sorted(missing))}" if missing else ""))
        say("ok" if checks.says("push yoga to sat 5pm", "move|push") and checks.says("an hour and a half", "90 minutes") else "FAIL",
            "message checks answer as expected")
    except Exception as e:
        say("FAIL", f"{type(e).__name__}: {e}")


def check_machine():
    print("\n2. Machine")
    gpu = shell("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader")
    say("ok" if gpu else "WARN", f"gpu: {gpu or 'nvidia-smi not found'}")
    free = shutil.disk_usage(Path.home()).free / 1e9
    say("ok" if free >= MIN_DISK_GB else "WARN", f"disk free {free:.0f} GB (want {MIN_DISK_GB}+ for both teachers)")
    version = shell("vllm", "--version")
    found = re.findall(r"\d+\.\d+\.\d+", version or "")
    numbers = tuple(map(int, found[-1].split("."))) if found else None
    say("ok" if numbers and numbers >= MIN_VLLM else "WARN",
        f"vllm {'.'.join(map(str, numbers)) if numbers else 'not on PATH'} (want {'.'.join(map(str, MIN_VLLM))}+)")
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    have = sorted(p.name.removeprefix("models--").replace("--", "/") for p in hub.glob("models--Qwen--Qwen3.5*")) if hub.exists() else []
    say("ok" if have else "WARN", f"downloaded: {', '.join(have) or 'no Qwen3.5 model yet'}")


def check_teacher(server: str, parallel: int) -> bool:
    print(f"\n3. Teacher at {server}")
    try:
        with urllib.request.urlopen(server + "/v1/models", timeout=10) as response:
            name = json.loads(response.read())["data"][0]["id"]
    except Exception as e:
        say("WARN", f"no teacher answering ({type(e).__name__}); start vLLM and run again for steps 3-4")
        return False
    say("ok", f"model: {name}")

    import generate
    teacher = generate.Teacher(server)
    try:
        (text, _), used = teacher.measure(teacher.complete, "Count from one to five: one,", 24, [], 0.0, 1)
        say("ok" if used["completion_tokens"] > 0 else "FAIL",
            f"one completion: {used['completion_tokens']} tokens in {used['seconds']} s, usage reported, text {text.strip()[:30]!r}")
    except Exception as e:
        say("FAIL", f"completion failed: {type(e).__name__}: {e}")
        return False

    start = time.time()
    with ThreadPoolExecutor(parallel) as pool:
        results = list(pool.map(lambda i: teacher.measure(teacher.complete, "Write a long story about a lighthouse.", 128, [], 1.0, i), range(parallel)))
    seconds = time.time() - start
    tokens = sum(used["completion_tokens"] for _, used in results)
    say("ok", f"{parallel} completions at once: {tokens / seconds:.0f} tokens/s in total, {tokens / seconds / parallel:.1f} per stream")
    return True


# The real generator in a throwaway folder: write 8 recipes, play 2, resume and play the rest, read it all back.
def check_generation(server: str):
    print("\n4. Generation end to end, in out/smoke-generation (kept until the next smoke)")
    folder = HERE / "out/smoke-generation"
    shutil.rmtree(folder, ignore_errors=True)
    for stage, count in [("messages", 8), ("runs", 2), ("runs", 8)]:
        done = subprocess.run([sys.executable, "generate.py", stage, "--count", str(count), "--workers", "4",
                               "--server", server, "--out", "out/smoke-generation"], cwd=HERE, capture_output=True, text=True)
        if done.returncode:
            say("FAIL", f"generate.py {stage} --count {count}: {(done.stdout + done.stderr).strip()[-300:]}")
            return
    try:
        kept = [json.loads(line) for line in (folder / "messages.jsonl").open()]
        runs = [json.loads(line) for line in (folder / "runs.jsonl").open()]
        manifest = json.loads((folder / "manifest.json").read_text())
    except Exception as e:
        say("FAIL", f"saved files unreadable: {type(e).__name__}: {e}")
        return
    for r in runs:
        say("ok", f"{r['kind']:<15} {' >> '.join(r['messages'])[:60]!r} -> {'PASS' if r['passed'] else 'fail: ' + r['problem']}")
    seeds = [r["seed"] for r in runs]
    whole = len(seeds) == len(set(seeds)) == len(kept) and set(seeds) == {c["seed"] for c in kept}
    say("ok" if kept and whole else "FAIL", f"{len(kept)} of 8 messages kept, {len(runs)} runs saved over two batches, "
        + ("each kept case exactly once" if whole else "runs don't match the kept messages one to one"))
    say("ok" if {"messages", "runs"} <= manifest.keys() else "FAIL",
        f"manifest: messages by {manifest.get('messages', {}).get('teacher', '?')}, runs by {manifest.get('runs', {}).get('teacher', '?')}")
    steps = [e for r in runs for e in r["run"]["log"] if "think" in e]
    thought, cut = sum(bool(e["think"]) for e in steps), sum(e["think_cut"] for e in steps)
    say("ok" if thought and cut <= len(steps) // 4 else "WARN", f"thinking in {thought} of {len(steps)} steps, {cut} cut at the budget")
    stats = subprocess.run([sys.executable, "stats.py", str(folder)], cwd=HERE, capture_output=True, text=True)
    say("ok" if stats.returncode == 0 else "FAIL", stats.stdout.splitlines()[0] if stats.returncode == 0 else stats.stderr.strip()[-300:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8080")
    parser.add_argument("--parallel", type=int, default=32, help="completions at once in the speed test")
    args = parser.parse_args()
    check_repo()
    check_machine()
    if check_teacher(args.server, args.parallel):
        check_generation(args.server)
    print(f"\n{'FAILED: fix the FAIL lines above' if FAILS else 'No failures.'} (WARN lines are worth a look.)")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
