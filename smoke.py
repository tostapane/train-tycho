# Smoke test for a new machine: is the repo intact, is the GPU/disk/vLLM there, does the teacher answer and
# do both generation passes work on a few cases? Run it once before starting vLLM (steps 1-2), once after (step 3).
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
MIN_DISK_GB = 130  # both candidate teachers: 37.5 GB + 78.9 GB, plus room


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


def check_teacher(server: str, parallel: int):
    print(f"\n3. Teacher at {server}")
    try:
        with urllib.request.urlopen(server + "/v1/models", timeout=10) as response:
            name = json.loads(response.read())["data"][0]["id"]
    except Exception as e:
        say("WARN", f"no teacher answering ({type(e).__name__}); start vLLM and run again for this step")
        return
    say("ok", f"model: {name}")

    import generate
    from gen import recipes
    teacher = generate.Teacher(server)
    try:
        (text, _), used = teacher.measure(teacher.complete, "Count from one to five: one,", 24, [], 0.0, 1)
        say("ok" if used["completion_tokens"] > 0 else "FAIL",
            f"one completion: {used['completion_tokens']} tokens in {used['seconds']} s, usage reported, text {text.strip()[:30]!r}")
    except Exception as e:
        say("FAIL", f"completion failed: {type(e).__name__}: {e}")
        return

    # Thinking must end at </think> by itself: if the server ignores that stop, every step runs to the budget.
    thinking, cut, shown = generate.step(teacher, "<|im_start|>user\nWhat is 2 + 2? One word.<|im_end|>\n", 2048, 1)
    say("ok" if thinking and not cut else "FAIL", f"thinking stopped at </think> after {len(thinking)} characters, answer {shown.strip()[:20]!r}"
        if thinking and not cut else "thinking ran to the budget: the server doesn't stop at </think>")

    start = time.time()
    with ThreadPoolExecutor(parallel) as pool:
        results = list(pool.map(lambda i: teacher.measure(teacher.complete, "Write a long story about a lighthouse.", 128, [], 1.0, i), range(parallel)))
    seconds = time.time() - start
    tokens = sum(used["completion_tokens"] for _, used in results)
    say("ok", f"{parallel} completions at once: {tokens / seconds:.0f} tokens/s in total, {tokens / seconds / parallel:.1f} per stream")

    kept = []
    for seed in range(4):
        case, used = teacher.measure(generate.write_messages, teacher, recipes.draw(seed))
        what = f"dropped ({case['dropped']})" if "dropped" in case else "kept"
        say("ok", f"pass 1, {case['kind']}: {' >> '.join(case['messages'])!r} {what}")
        if "dropped" not in case:
            kept.append(case)
    if not kept:
        say("WARN", "no kept message to play in pass 2; run again")
        return
    case = kept[0]
    try:
        run, used = teacher.measure(generate.live_run, teacher, case, 128)
        problem = generate.check(case, run)
        calls = [e["tool"] for e in run["log"] if "tool" in e]
        thought = any(e.get("think") for e in run["log"])
        say("ok", f"pass 2, {case['kind']}: calls {calls or 'none'}, answer {run['answer'][:50]!r}, "
                  f"{used['requests']} requests, {used['seconds']} s, {'passed' if problem is None else 'failed: ' + problem}")
        say("ok" if thought else "WARN", "thinking text came back" if thought else "no thinking text: check the server's template")
    except Exception as e:
        say("FAIL", f"live run failed: {type(e).__name__}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8080")
    parser.add_argument("--parallel", type=int, default=32, help="completions at once in the speed test")
    args = parser.parse_args()
    check_repo()
    check_machine()
    check_teacher(args.server, args.parallel)
    print(f"\n{'FAILED: fix the FAIL lines above' if FAILS else 'No failures.'} (WARN lines are worth a look.)")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
