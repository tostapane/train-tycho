# Makes training cases with a teacher model behind an OpenAI-style /v1/completions server (llama-server or vLLM).
# Pass 1 (messages): the teacher writes the user's message(s) for each recipe; duplicates are dropped.
# Pass 2 (runs): the teacher plays the assistant live on the fake phone; code checks the outcome.
# Usage: uv run generate.py messages|runs --count N [--server URL] [--workers W] [--think-budget T] [--out DIR]

import argparse
import difflib
import json
import random
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from app import grammar, prompt, sim
from eval import MAX_CALLS, in_answer, matches, parse_call, visible
from gen import recipes
from gen.checks import ASSISTANT, RING, RING_OK, says

HERE = Path(__file__).parent
WRITER_FILE = json.loads((HERE / "gen/writer.json").read_text())
STYLES = WRITER_FILE["styles"]

# The rules the teacher follows in the live run. Training text replaces them with a short prompt from the pool.
RULES = """You are a private assistant that runs offline on the user's phone. Follow these rules exactly.
- Look up before acting: find_contact before call, draft_message or draft_email, and use the number or email exactly as found; find_events before edit_events or delete_events.
- Two contacts or events match: ask which one, naming both in full. Nothing matches: say so, and offer the closest match if there is one.
- Information missing (a time for something that happens at a time, a recipient, a duration): ask one short question. Never guess it.
- "Remind me to..." (also "in 12 minutes") is add_reminder with the task as text. A timer only when the user says timer or countdown. Something that happens at a time (appointment, meeting, sport, dinner) is add_event. Waking up is set_alarm, with days only if it repeats.
- Bare hours: waking up means morning; an activity means its sensible time (padel at 4 is 16:00); am/pm always wins. Dates are never in the past: "the 1st" is the next 1st.
- Messages go by SMS unless the user names WhatsApp, Telegram or Signal. Keep the user's words in the message text.
- Change or delete many events in one call, with all their ids. Clearing a day deletes every event listed on it.
- Event titles are short and clean: "Gym", "Dinner with Marco", never the whole request or a date or time.
- After the tool result, answer and stop: no extra actions. After an error, fix the call once if the intent is clear, otherwise ask. Never repeat a failing call.
- A timer can't be changed: for "make it 15", start a new timer and say the old one is still running and can be stopped in the Clock app.
- No tool for chat, thanks, questions you can answer, "don't...", or "I should ... later". The dates before the user's message are reference only.
- Final answer: one short sentence that repeats what was done with its key values, e.g. "Added Gym on Tuesday at 19:00 for an hour." Contacts by full name. Dates: today, tomorrow, a weekday within a week, else "3 December". Times in the user's clock format. No offers, no emojis."""

WRITER = """You play a phone USER typing to their offline assistant. You are not the assistant: never answer, offer
help or apologize. Write like a real person: never mention tools, ids or this task. Use only the facts given, and the
day and time words exactly as given, and words in quotes word for word. A style or shape may add a filler or a short
reason, never anything that changes what is asked. Reply with JSON only: {"message": "...", "followup": "..." or null}.

Examples:
"""
EXAMPLES = WRITER_FILE["examples"]


# The writer's instructions for one recipe: four examples drawn by its seed (one in its style), so no single one gets copied.
def writer(recipe: dict) -> str:
    rng = random.Random(recipe["seed"])
    same = [e for e in EXAMPLES if e["style"] == recipe["style"]]
    others = [e for e in EXAMPLES if e["style"] != recipe["style"]]
    picks = rng.sample(same, min(1, len(same))) + rng.sample(others, 3)
    rng.shuffle(picks)
    shown = []
    for e in picks:
        shape = f"Shape of the first message: {recipes.SHAPES[e['shape']]}.\n" if e.get("shape") else ""
        shown.append(f"What the user wants: {e['wants']}\nStyle: {e['style']}: {STYLES[e['style']]}.\n{shape}"
                     + json.dumps({"message": e["message"], "followup": e["followup"]}, ensure_ascii=False))
    return WRITER + "\n\n".join(shown)


class Teacher:
    def __init__(self, url: str):
        self.url = url
        self.name = ""
        self.mine = threading.local()

    # Raw completion of `text`, cut at any of `stop`. Returns the generated text.
    def complete(self, text: str, max_tokens: int, stop: list[str], temperature: float, seed: int) -> str:
        body = {"prompt": text, "max_tokens": max_tokens, "stop": stop, "temperature": temperature, "top_p": 0.95,
                "seed": seed}
        request = urllib.request.Request(self.url + "/v1/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=600) as response:
            reply = json.loads(response.read())
        self.name = reply.get("model", self.name)
        used = getattr(self.mine, "used", None)
        if used is not None:
            used["requests"] += 1
            for key in ("prompt_tokens", "completion_tokens"):
                used[key] += reply.get("usage", {}).get(key, 0)
        return reply["choices"][0]["text"]

    # Runs f(*args) and returns (its result, what the teacher calls inside it cost).
    def measure(self, f, *args):
        self.mine.used = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0}
        start = time.time()
        result = f(*args)
        used = self.mine.used | {"seconds": round(time.time() - start, 1), "teacher": self.name,
                                 "at": datetime.now().isoformat(timespec="seconds")}
        return result, used


# A chat turn in Qwen's template with thinking off: system, user, then the start of the answer.
def chat(system: str, user: str) -> str:
    return (f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
            + prompt.assistant_start(False))


# Pass 1 for one recipe: the teacher writes the message (and the follow-up, if the recipe has one).
# A message that breaks the recipe comes back with the reason in "dropped".
def write_messages(teacher: Teacher, recipe: dict) -> dict:
    now = datetime.fromisoformat(recipe["now"])
    request = (f"Now: {now.strftime('%A %d %B %Y, %H:%M')}.\nWhat the user wants: {recipe['task']}\n"
               f"Style: {recipe['style']}: {STYLES[recipe['style']]}.\n"
               + (f"Shape of the first message: {recipes.SHAPES[recipe['shape']]}.\n" if recipe.get("shape") else "")
               + (f"Then, after the assistant answers, a follow-up: {recipe['followup']}\n" if recipe["followup"]
                  else "No follow-up (null).\n"))
    reply = teacher.complete(chat(writer(recipe), request), 200, ["<|im_end|>"], 1.0, recipe["seed"])
    found = re.search(r"\{.*\}", reply, re.S)
    try:
        written = json.loads(found.group(0)) if found else None
    except json.JSONDecodeError:
        written = None
    if not isinstance(written, dict) or not isinstance(written.get("message"), str) or not written["message"].strip():
        return recipe | {"messages": [reply], "dropped": "no message in the reply"}
    messages = [written["message"].strip()]
    if recipe["followup"]:
        if not isinstance(written.get("followup"), str) or not written["followup"].strip():
            return recipe | {"messages": messages, "dropped": "no follow-up"}
        messages.append(written["followup"].strip())
    text = " ".join(messages)
    if any(ASSISTANT.search(m) for m in messages):
        return recipe | {"messages": messages, "dropped": "sounds like the assistant"}
    if recipe["kind"] not in RING_OK and RING.search(text):
        return recipe | {"messages": messages, "dropped": "uses 'ring' for something other than a clock"}
    missing = [s for s in recipe["say"] if not says(text, s)]
    if missing:
        return recipe | {"messages": messages, "dropped": f"missing {missing}"}
    return recipe | {"messages": messages}


# Lowercase words only, to spot duplicates that differ by case or punctuation.
def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def too_close(a: str, b: str) -> bool:
    x, y = set(a.split()), set(b.split())
    return bool(x | y) and len(x & y) / len(x | y) >= 0.8


# One step of the assistant: thinking (cut at `budget` tokens and closed by force), then the visible part.
def step(teacher: Teacher, text: str, budget: int, seed: int) -> tuple[str, str]:
    if budget == 0:
        return "", teacher.complete(text + prompt.assistant_start(False), 400, ["<|im_end|>"], 0.7, seed)
    start = text + prompt.assistant_start(True)
    thinking = teacher.complete(start, budget, ["</think>"], 0.6, seed).strip()
    shown = teacher.complete(start + thinking + "\n</think>\n\n", 400, ["<|im_end|>"], 0.6, seed)
    return thinking, shown


# A whole tool call the grammar allows, at the end of the model's text.
CALL = re.compile(grammar.call_pattern() + r"\s*$")
# A thinking block, empty or not. Qwen's template drops them from turns before the last user message.
THINK = re.compile(r"<think>\n.*?</think>\n\n", re.S)


# Pass 2 for one case: plays every message live on the fake phone, as eval.play does for the 2B.
# "confirm" is true when the last turn ended on a confirm card: then there's no answer to check.
def live_run(teacher: Teacher, case: dict, budget: int) -> dict:
    now = datetime.fromisoformat(case["now"])
    phone = sim.Phone(case["phone"], now)
    text = prompt.system_turn(prompt.load_tools(), tuple(case["formats"])).replace(prompt.SYSTEM_PROMPT, RULES)
    log, answer, pending, close = [], "", None, False
    for message in case["messages"]:
        text = THINK.sub("", text)
        text += "".join(t for t, _ in prompt.user_turn(message, now, budget > 0, close, pending))
        text = text.rsplit("<|im_start|>assistant\n", 1)[0]
        log.append({"user": message})
        phone.in_background = False
        pending, close, calls, confirm = None, True, 0, False
        while True:
            thinking, shown = step(teacher, text, budget, case["seed"] * 10 + len(log))
            text += prompt.assistant_start(False) + shown if budget == 0 else \
                prompt.assistant_start(True) + thinking + "\n</think>\n\n" + shown
            log.append({"think": thinking, "model": shown})
            answer = visible(shown)
            if calls >= MAX_CALLS or "<tool_call>" not in shown:
                break
            if shown.count("<tool_call>") > 1 or not CALL.search(shown):
                return {"phone": phone.state, "answer": answer, "log": log, "problem": "call the grammar forbids"}
            name, args = parse_call(shown)
            result = phone.run(name, args)
            calls += 1
            log.append({"tool": name, "args": args, "result": result})
            if sim.TOOLS[name]["hven"]["confirm"]:
                pending, confirm = result, True
                break
            text += "".join(t for t, _ in prompt.tool_response(result, budget > 0)).rsplit("<|im_start|>assistant\n", 1)[0]
    return {"phone": phone.state, "answer": answer, "log": log, "confirm": confirm}


BAD_TITLE = re.compile(r"\d|\b(today|tonight|tomorrow|am|pm|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I)
OFFER = re.compile(r"would you like|anything else|let me know|feel free|is there anything", re.I)
RAW = re.compile(r"\b\d+ h \d+ min \d+ s\b|\bid \d+\b|</?tool|<function|</?think", re.I)
EMOJI = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")


# Lowercase, with times joined as in `words`: what answer patterns are matched against.
def flat(text: str) -> str:
    text = re.sub(r"(\d{1,2}):00\s*(am|pm)\b", r"\1\2", text.lower())
    return re.sub(r"(\d)\s+(am|pm)\b", r"\1\2", text)


# Why one answer breaks the general rules, or None: emojis, raw tool output, wrong clock, too long, offers.
def answer_problem(case: dict, answer: str, acted: bool) -> str | None:
    if EMOJI.search(answer) or RAW.search(answer):
        return "emoji or raw tool output"
    if case["formats"][1] == "24-hour" and re.search(r"\d\s?(am|pm)\b", answer, re.I):
        return "am/pm for a 24-hour user"
    if case["formats"][1] == "12-hour" and re.search(r"\b(1[3-9]|2[0-3]):[0-5]\d\b", answer):
        return "24-hour time for a 12-hour user"
    if len(answer) > (500 if case["category"] == "notool" else 250):
        return "too long"
    if acted and OFFER.search(answer):
        return "offer after an action"
    return None


# Why a run fails, or None if it passes: end state, answer patterns, names, titles and the general answer rules.
# A turn that ended on a confirm card has no answer: the card shows what will happen.
def check(case: dict, run: dict) -> str | None:
    if run.get("problem"):
        return run["problem"]
    start = sim.EMPTY_STATE | case["phone"]
    answer = flat(run["answer"])
    state_ok = [o for o in case["expect"] if matches(start | o.get("changes", {}), run["phone"])]
    if not state_ok:
        return "wrong end state"
    if not run["confirm"] and not answer:
        return "no answer"
    if not run["confirm"] and not any(
            all(in_answer(answer, s) for s in o.get("answer_has", []))
            and all(any(in_answer(answer, flat(alt)) for alt in group) for group in o.get("answer_any", []))
            and not any(in_answer(answer, s) for s in o.get("answer_not", [])) for o in state_ok):
        return "answer misses what the recipe needs"
    acted = run["phone"] != start
    for text in [visible(e["model"]) for e in run["log"] if "model" in e]:
        problem = text and answer_problem(case, text, acted)
        if problem:
            return problem
    contacts = run["phone"]["contacts"]
    for opened in run["phone"]["opened"]:
        person = next((c for c in contacts if opened.get("number") and re.sub(r"[^0-9+]", "", c["phone"]) == opened["number"]
                       or opened.get("to") and c["email"] == opened["to"]), None)
        if person and not run["confirm"] and person["name"].lower() not in answer:
            return "contact not named in full"
    for name in re.findall(r"did you mean ([a-z][\w'-]*)", answer):
        if not any(c["name"].lower().startswith(name) for c in contacts):
            return "suggests a contact that doesn't exist"
    if any(BAD_TITLE.search(e["title"]) for e in run["phone"]["events"] if e not in case["phone"]["events"]):
        return "event title has a date or time"
    return None


# Runs f(*args); on any error (server down, odd output) prints it and returns None, so one case can't stop the run.
# The case isn't written, so the next run tries it again.
def safe(f, *args):
    try:
        return f(*args)
    except Exception as e:
        print(f"ERROR {type(e).__name__}: {e}", flush=True)
        return None


def done_seeds(path: Path) -> set:
    return {json.loads(line)["seed"] for line in path.open()} if path.exists() else set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["messages", "runs"])
    parser.add_argument("--count", type=int, default=100, help="recipes to draw (messages) or cases to run (runs)")
    parser.add_argument("--server", default="http://127.0.0.1:8080")
    parser.add_argument("--workers", type=int, default=1, help="cases in flight at once (~256 on a big GPU)")
    parser.add_argument("--think-budget", type=int, default=512, help="thinking tokens per step; 0 = thinking off")
    parser.add_argument("--out", default="out/data", help="folder for the results, inside this repo")
    args = parser.parse_args()
    teacher = Teacher(args.server)
    folder = HERE / args.out
    folder.mkdir(parents=True, exist_ok=True)

    if args.stage == "messages":
        path, dropped_path = folder / "messages.jsonl", folder / "messages-dropped.jsonl"
        done = done_seeds(path) | done_seeds(dropped_path)
        todo = [recipes.draw(s) for s in range(args.count) if s not in done]
        seen = {normalize(json.loads(l)["messages"][0]) for l in path.open()} if path.exists() else set()
        evals = [normalize(m) for l in (HERE / "cases/core.jsonl").open() for m in json.loads(l)["messages"]]
        with path.open("a") as out, dropped_path.open("a") as dropped, ThreadPoolExecutor(args.workers) as pool:
            for result in pool.map(lambda r: safe(teacher.measure, write_messages, teacher, r), todo):
                if result is None:
                    continue
                case, used = result
                case |= {"write": used}
                key = normalize(case["messages"][0])
                if "dropped" not in case and (key in seen or any(too_close(key, e) for e in evals)):
                    case["dropped"] = "duplicate, or too close to an eval case"
                if "dropped" in case:
                    dropped.write(json.dumps({k: case[k] for k in ("seed", "category", "kind", "messages", "say", "dropped", "write")},
                                             ensure_ascii=False) + "\n")
                    dropped.flush()
                    print(f"DROP {case['kind']:<16} {' >> '.join(case['messages'])[:60]:<60} {case['dropped']}", flush=True)
                    continue
                seen.add(key)
                out.write(json.dumps(case, ensure_ascii=False) + "\n")
                out.flush()
                print(f"{case['category']:<10} {case['kind']:<16} {' >> '.join(case['messages'])}", flush=True)
        return

    path = folder / "runs.jsonl"
    done = done_seeds(path)
    cases = [json.loads(l) for l in (folder / "messages.jsonl").open()]
    todo = [c for c in cases if c["seed"] not in done][:args.count]
    with path.open("a") as out, ThreadPoolExecutor(args.workers) as pool:
        for case, result in zip(todo, pool.map(lambda c: safe(teacher.measure, live_run, teacher, c, args.think_budget), todo)):
            if result is None:
                continue
            run, used = result
            problem = check(case, run)
            out.write(json.dumps(case | {"run": run, "play": used | {"think_budget": args.think_budget},
                                         "passed": problem is None, "problem": problem}, ensure_ascii=False) + "\n")
            out.flush()
            calls = " -> ".join(f"{e['tool']}({e['args']})" for e in run["log"] if "tool" in e)
            print(f"{'PASS' if problem is None else 'FAIL'}  {case['kind']:<16} {' >> '.join(case['messages'])[:45]:<45} "
                  f"{calls[:70]} | {run['answer'][:45]!r} {problem or ''}", flush=True)


if __name__ == "__main__":
    main()
