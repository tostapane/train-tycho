# Makes training cases with a teacher model behind an OpenAI-style /v1/completions server (llama-server or vLLM).
# Pass 1 (messages): the teacher writes the user's message(s) for each recipe; a 4th copy of the same messages is dropped.
# Pass 2 (runs): the teacher plays the assistant live on the fake phone; code checks the outcome.
# Usage: uv run generate.py messages|runs --count N [--server URL] [--workers W] [--think-budget T] [--out DIR]

import argparse
import fcntl
import hashlib
import json
import random
import re
import threading
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from itertools import islice
from pathlib import Path

from app import grammar, prompt, sim
from eval import MAX_CALLS, ends_on_card, grounded_actions, in_answer, matches, parse_call, searched_day, visible
from gen import recipes
from gen.checks import ASSISTANT, RING, RING_OK, says

HERE = Path(__file__).parent
MAX_COPIES = 3  # the same messages, at most this many times
WRITER_FILE = json.loads((HERE / "gen/writer.json").read_text())
STYLES = WRITER_FILE["styles"]

# The rules the teacher follows in the live run. Training text replaces them with a short prompt from the pool.
RULES = """You are a private assistant that runs offline on the user's phone. Follow these rules exactly.
- Look up before acting: find_contact before call, draft_message or draft_email, and use the number or email exactly as found; find_events before edit_events or delete_events, with a query when the user names the event.
- Questions about the calendar or a contact ("what do I have Sunday?", "what's Priya's number?"): look it up and answer from the result. Change nothing, call no one.
- Two contacts or events match: ask which one, naming both in full. Nothing matches: say so, and offer the closest match if there is one.
- Information missing (a time for something that happens at a time, a recipient, a duration): ask one short question. Never guess it.
- "Remind me to..." is add_reminder with the task as text: in_minutes for "in 25 minutes" or "in 2 hours", else date and time. A timer only when the user says timer or countdown. Something that happens at a time (appointment, meeting, sport, dinner) is add_event. Waking up is set_alarm, with days only if it repeats.
- Bare hours: waking up means morning; an activity means its sensible time (tennis at 2 is 14:00); am/pm always wins. Dates are never in the past: "the 3rd" is the next 3rd.
- Messages go by SMS unless the user names WhatsApp, Telegram or Signal. Keep the user's words in the message text.
- Drafts are not sent: the user presses send. Say the message or email is ready, never that it was sent.
- Change or delete many events in one call, with all their ids. Clearing a day deletes every event listed on it, with that day's date so repeating ones go only for that day.
- Event titles are short and clean: "Swimming", "Dinner with Omar", never the whole request or a date or time.
- Call only the tools the request needs. Once it's done, answer and stop: no extra actions. After an error, fix the call once if the intent is clear, otherwise ask. Never repeat a failing call.
- Two requests in one message: do both, then one answer for both. "If I'm free, add X": find_events first, add it only if nothing is on at that time, else say what is.
- A timer can't be changed: for "make it 20", start a new timer and say the old one is still running and can be stopped in the Clock app.
- No tool for chat, thanks, questions you can answer, "don't...", "I should ... later", or remarks about people, events or times that ask for nothing ("I called mum yesterday"). The dates before the user's message are reference only.
- No tool can get the weather, news, the web, prices or any live information: say so in one short sentence and never make up an answer.
- Final answer: one short sentence that repeats what was done with its key values, e.g. "Added Swimming on Monday at 18:30 for 45 minutes." Contacts by full name. Dates: today, tomorrow, a weekday within a week, else "3 December". Times in the user's clock format. Plain text: no markdown, no lists. No offers, no emojis.
- Think briefly: a few short sentences on what the user wants, which tool, and which values.
- In your thinking, reason about the request itself; never mention these rules or instructions."""

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
        self.model_id = ""
        self.mine = threading.local()

    # Raw completion of `text`, cut at any of `stop`. Returns the generated text, and True if it hit max_tokens.
    def complete(self, text: str, max_tokens: int, stop: list[str], temperature: float, seed: int) -> tuple[str, bool]:
        body = {"prompt": text, "max_tokens": max_tokens, "stop": stop, "temperature": temperature, "top_p": 0.95,
                "seed": seed}
        if self.model_id:
            body["model"] = self.model_id
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
        choice = reply["choices"][0]
        return choice["text"], choice.get("finish_reason") == "length"

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
    reply, _ = teacher.complete(chat(writer(recipe), request), 200, ["<|im_end|>"], 1.0, recipe["seed"])
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
    facts = [recipe["first_turn"]["say"], recipe["follow_say"]] if "first_turn" in recipe else [recipe["say"]]
    missing = [s for message, required in zip(messages, facts, strict=True) for s in required if not says(message, s)]
    if phrase := recipe.get("verbatim"):
        words = " " + normalize(messages[-1]) + " "
        if " " + normalize(phrase) + " " not in words:
            missing.append(phrase)
    if missing:
        return recipe | {"messages": messages, "dropped": f"missing {missing}"}
    return recipe | {"messages": messages}


# Lowercase words only, to spot duplicates that differ by case or punctuation.
def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def too_close(a: str, b: str) -> bool:
    x, y = set(a.split()), set(b.split())
    return bool(x | y) and len(x & y) / len(x | y) >= 0.8


# True if the message contains the words of a held-out request.
def contains(text: str, case: str) -> bool:
    have = text.split()
    if len(case.split()) == 2:
        polite = {"please", "pls", "hey", "can", "could", "you", "for", "me", "thanks"}
        return [w for w in have if w not in polite] == case.split()
    return len(case.split()) >= 3 and all(any(h == w or len(w) >= 3 and h.startswith(w) for h in have) for w in case.split())


# Generate thinking up to the budget, then complete the visible answer.
def step(teacher: Teacher, text: str, budget: int, seed: int) -> tuple[str, bool, str]:
    if budget == 0:
        return "", False, teacher.complete(text + prompt.assistant_start(False), 400, ["<|im_end|>"], 0.7, seed)[0]
    start = text + prompt.assistant_start(True)
    thinking, cut = teacher.complete(start, budget, ["</think>"], 0.6, seed)
    thinking = thinking.strip()
    shown, _ = teacher.complete(start + thinking + "\n</think>\n\n", 400, ["<|im_end|>"], 0.6, seed)
    return thinking, cut, shown


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
            thinking, cut, shown = step(teacher, text, budget, case["seed"] * 10 + len(log))
            text += prompt.assistant_start(False) + shown if budget == 0 else \
                prompt.assistant_start(True) + thinking + "\n</think>\n\n" + shown
            log.append({"think": thinking, "think_cut": cut, "model": shown})
            answer = visible(shown)
            if "<tool_call>" not in shown:
                break
            if calls >= MAX_CALLS:
                return {"phone": phone.state, "answer": answer, "log": log, "problem": "too many tool calls"}
            if shown.count("<tool_call>") > 1 or not CALL.search(shown):
                return {"phone": phone.state, "answer": answer, "log": log, "problem": "call the grammar forbids"}
            name, args = parse_call(shown)
            result = phone.run(name, args)
            calls += 1
            log.append({"tool": name, "args": args, "result": result})
            if sim.TOOLS[name]["hven"]["confirm"] and not result.startswith("Error:"):
                pending, confirm = result, True
                break
            text += "".join(t for t, _ in prompt.tool_response(result, budget > 0)).rsplit("<|im_start|>assistant\n", 1)[0]
    return {"phone": phone.state, "answer": answer, "log": log, "confirm": confirm}


BAD_TITLE = re.compile(r"\d|\b(today|tonight|tomorrow|am|pm|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I)
OFFER = re.compile(r"would you like|anything else|let me know|feel free|is there anything", re.I)
RAW = re.compile(r"\b\d+ h \d+ min \d+ s\b|\bid \d+\b|\b\d{4}-\d{2}-\d{2}\b|</?tool|<function|</?think", re.I)
EMOJI = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")


# Reject formatting that the Android plain text viewer would display literally.
MARKDOWN = re.compile(r"\*\*|`|^\s*(#+ |[-*•] |\d+[.)] )", re.M)


# Lowercase, with times joined as in `words`: what answer patterns are matched against.
def flat(text: str) -> str:
    text = re.sub(r"(\d{1,2}):00\s*(am|pm)\b", r"\1\2", text.lower())
    return re.sub(r"(\d)\s+(am|pm)\b", r"\1\2", text)


# Why one answer breaks the general rules, or None: emojis, raw tool output, wrong clock, too long, offers.
def answer_problem(case: dict, answer: str, acted: bool) -> str | None:
    if EMOJI.search(answer) or RAW.search(answer):
        return "emoji or raw tool output"
    if MARKDOWN.search(answer):
        return "markdown"
    if case["formats"][1] == "24-hour" and re.search(r"\d\s?(am|pm)\b", answer, re.I):
        return "am/pm for a 24-hour user"
    if case["formats"][1] == "12-hour" and re.search(r"\b(1[3-9]|2[0-3]):[0-5]\d\b", answer):
        return "24-hour time for a 12-hour user"
    if len(answer) > (500 if case["category"] == "notool" else 250):
        return "too long"
    if acted and OFFER.search(answer):
        return "offer after an action"
    return None


# Check each turn against the facts available when the user asked it.
def check(case: dict, run: dict) -> str | None:
    if run.get("problem"):
        return run["problem"]
    starts = [i for i, entry in enumerate(run["log"]) if "user" in entry]
    if len(starts) == 1 and not case.get("followup"):
        return check_turn(case, run, [])
    if len(starts) != 2 or "first_turn" not in case:
        return "missing per-turn expectations"
    first, second = run["log"][:starts[1]], run["log"][starts[1]:]
    phone = sim.Phone(case["phone"], datetime.fromisoformat(case["now"]))
    for entry in first:
        if "tool" in entry:
            phone.run(entry["tool"], entry["args"])
    answer = next((visible(e["model"]) for e in reversed(first) if "model" in e), "")
    first_run = {"phone": phone.state, "log": first, "confirm": ends_on_card(first), "answer": answer}
    first_case = {k: case[k] for k in ("phone", "category", "formats")} | case["first_turn"]
    problem = check_turn(first_case, first_run, [])
    if problem:
        return "first turn: " + problem
    return check_turn(case, run | {"log": second}, [] if case.get("earlier") else first)


# A read must cover the requested information before the assistant uses it.
def checked_lookup(case: dict, calls: list) -> bool:
    reads = [e for e in calls if sim.TOOLS[e["tool"]]["hven"]["kind"] == "read" and not e["result"].startswith("Error:")]
    if not all(any(in_answer(e["result"], fact) for e in reads) for fact in case.get("read_has", [])):
        return False
    if app := case.get("missing_app"):
        if not any(e["tool"] == "open_app" and e["args"]["name"].casefold() == app.casefold()
                   and e["result"].startswith("Error:") for e in calls):
            return False
    if topic := case.get("note_topic"):
        if not any(e["tool"] == "find_notes" and e["result"] == "No notes match." and e["args"]["query"].strip()
                   and all(word in sim.fold(topic) for word in sim.fold(e["args"]["query"]).split()) for e in reads):
            return False
    if day := case.get("calendar_day"):
        return searched_day(calls, day)
    return True


# Score the state, actions and answer of one turn. Earlier reads can ground a genuine follow-up.
def check_turn(case: dict, run: dict, history: list) -> str | None:
    start = sim.EMPTY_STATE | case["phone"]
    answer = flat(run["answer"])
    state_ok = [o for o in case["expect"] if matches(start | o.get("changes", {}), run["phone"])]
    if not state_ok:
        return "wrong end state"
    calls = [e for e in run["log"] if "tool" in e]
    called = [e["tool"] for e in calls]
    kind = case["kind"]
    if any(t not in recipes.ALLOWED_TOOLS[kind] for t in called):
        return "tool not needed"
    if any(t not in called for t in recipes.NEEDED_TOOLS.get(kind, [])):
        return "skipped a needed lookup"
    if not grounded_actions(history + calls) or not checked_lookup(case, calls):
        return "missing relevant lookup"
    if not run["confirm"] and not answer:
        return "no answer"
    if not run["confirm"] and not any(
            all(in_answer(answer, s) for s in o.get("answer_has", []))
            and all(any(in_answer(answer, flat(alt)) for alt in group) for group in o.get("answer_any", []))
            and not any(in_answer(answer, s) for s in o.get("answer_not", [])) for o in state_ok):
        return "answer misses what the recipe needs"
    acted = False
    for e in run["log"]:
        if "user" in e:
            acted = False
        elif "tool" in e:
            acted |= sim.TOOLS[e["tool"]]["hven"]["kind"] == "action" and not e["result"].startswith("Error")
        else:
            text = visible(e["model"])
            problem = text and answer_problem(case, text, acted)
            if problem:
                return problem
    if any(e.get("tool") in ("draft_message", "draft_email") for e in run["log"]) and re.search(r"\bsent\b", answer):
        return "draft called sent"
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


# Keep each folder tied to the exact generation sources and the teacher used for each pass.
def prepare_folder(folder: Path, stage: str, teacher: Teacher, budget: int):
    path = folder / "manifest.json"
    if not path.exists() and any(folder.glob("*.jsonl")):
        raise SystemExit("This folder contains unversioned data. Choose a fresh --out folder.")
    digest = hashlib.sha256()
    for pattern in ("generate.py", "eval.py", "app/*.py", "app/*.json", "gen/*.py", "gen/*.json", "cases/core.jsonl"):
        for source in sorted(HERE.glob(pattern)):
            digest.update(str(source.relative_to(HERE)).encode() + b"\0" + source.read_bytes() + b"\0")
    signature = digest.hexdigest()
    manifest = json.loads(path.read_text()) if path.exists() else {"sources": signature}
    if manifest["sources"] != signature:
        raise SystemExit("Generation sources differ from this folder's manifest. Choose a fresh --out folder.")
    with urllib.request.urlopen(teacher.url + "/v1/models", timeout=15) as response:
        teacher.model_id = json.load(response)["data"][0]["id"]
    config = {"teacher": teacher.model_id}
    if stage == "runs":
        config["think_budget"] = budget
    if stage in manifest and manifest[stage] != config:
        raise SystemExit(f"The {stage} settings differ from this folder's manifest.")
    manifest[stage] = config
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(manifest, indent=2) + "\n")
    temp.replace(path)


# Put all copies of an opening message on the same side before making any training variants.
def conversation_split(messages: list[str]) -> str:
    group = hashlib.sha256(normalize(messages[0]).encode()).digest()
    return "validation" if int.from_bytes(group[:8]) % 20 == 0 else "train"


# Read records one at a time so full phones and conversations need not all fit in memory.
def records(path: Path):
    if path.exists():
        with path.open() as source:
            for line in source:
                yield json.loads(line)


def done_seeds(path: Path) -> set:
    return {record["seed"] for record in records(path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["messages", "runs"])
    parser.add_argument("--count", type=int, default=100, help="recipes to draw (messages) or cases to run (runs)")
    parser.add_argument("--server", default="http://127.0.0.1:8080")
    parser.add_argument("--kinds", default="", help="messages only: draw just these comma-separated recipe kinds")
    parser.add_argument("--workers", type=int, default=1, help="cases in flight at once (~256 on a big GPU)")
    parser.add_argument("--think-budget", type=int, default=512, help="thinking tokens per step; 0 = thinking off")
    parser.add_argument("--out", default="out/data", help="folder for the results, inside this repo")
    args = parser.parse_args()
    teacher = Teacher(args.server)
    folder = HERE / args.out
    folder.mkdir(parents=True, exist_ok=True)

    args.kinds = set(filter(None, args.kinds.split(",")))
    if args.kinds - recipes.ALLOWED_TOOLS.keys() or (args.kinds and args.stage != "messages"):
        parser.error("--kinds needs valid recipe kinds and the messages stage")
    if args.count < 0 or args.workers < 1 or args.think_budget < 0:
        parser.error("count and budget must be nonnegative and workers must be positive")
    with (folder / ".generation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another generator is writing this folder.")
        prepare_folder(folder, args.stage, teacher, args.think_budget)
        totals = {"written": 0, "errors": 0}
        start = time.monotonic()
        try:
            generate_batch(args, folder, teacher, totals)
        finally:
            batch = totals | {"stage": args.stage, "seconds": time.monotonic() - start}
            with (folder / "batches.jsonl").open("a") as log:
                log.write(json.dumps(batch) + "\n")
        if totals["errors"]:
            raise SystemExit(f"{totals['errors']} cases failed to run. Rerun this pass to retry their seeds.")


# Teacher failures leave the seed available for retry and count against the elapsed batch time.
def attempt(teacher, function, case, *args):
    try:
        return teacher.measure(function, teacher, case, *args)
    except Exception as error:
        print(f"ERROR seed {case['seed']}: {type(error).__name__}: {error}", flush=True)
        return None


# Process one bounded batch, keeping writes in seed order.
def generate_batch(args, folder, teacher, totals):
    if args.stage == "messages":
        path, dropped_path = folder / "messages.jsonl", folder / "messages-dropped.jsonl"
        done = done_seeds(path) | done_seeds(dropped_path)
        todo = (recipes.draw(s) for s in range(args.count) if s not in done)
        if args.kinds:
            todo = (case for case in todo if case["kind"] in args.kinds)
        seen = Counter(normalize(" ".join(c["messages"])) for c in records(path))
        evals = [normalize(m) for l in (HERE / "cases/core.jsonl").open() for m in json.loads(l)["messages"]]
        with path.open("a") as out, dropped_path.open("a") as dropped, ThreadPoolExecutor(args.workers) as pool:
            for result in pool.map(lambda r: attempt(teacher, write_messages, r), todo, buffersize=2 * args.workers):
                if result is None:
                    totals["errors"] += 1
                    continue
                case, used = result
                totals["written"] += 1
                case |= {"write": used}
                key = normalize(" ".join(case["messages"]))
                if "dropped" not in case and seen[key] >= MAX_COPIES:
                    case["dropped"] = f"{MAX_COPIES} copies already"
                if "dropped" not in case and any(too_close(normalize(m), e) or contains(normalize(m), e)
                                                 for m in case["messages"] for e in evals):
                    case["dropped"] = "too close to an eval case"
                if "dropped" in case:
                    dropped.write(json.dumps({k: case[k] for k in ("seed", "category", "kind", "messages", "say", "dropped", "write")},
                                             ensure_ascii=False) + "\n")
                    dropped.flush()
                    print(f"DROP {case['kind']:<16} {' >> '.join(case['messages'])[:60]:<60} {case['dropped']}", flush=True)
                    continue
                case["split"] = conversation_split(case["messages"])
                seen[key] += 1
                out.write(json.dumps(case, ensure_ascii=False) + "\n")
                out.flush()
                print(f"{case['category']:<10} {case['kind']:<16} {' >> '.join(case['messages'])}", flush=True)
        return

    path = folder / "runs.jsonl"
    done = done_seeds(path)
    messages = folder / "messages.jsonl"
    if not messages.exists():
        raise SystemExit("Run the messages pass in this folder first.")
    todo = islice((c for c in records(messages) if c["seed"] not in done), args.count)

    def play(case):
        return case, attempt(teacher, live_run, case, args.think_budget)

    with path.open("a") as out, ThreadPoolExecutor(args.workers) as pool:
        for case, result in pool.map(play, todo, buffersize=2 * args.workers):
            if result is None:
                totals["errors"] += 1
                continue
            totals["written"] += 1
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
