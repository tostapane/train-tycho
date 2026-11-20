# Runs the eval cases through llama-server, with the app's prompt, grammar and sampling, and scores
# each case on the fake phone: the end state and the answer must match what the case expects.
# Start the server first (see AGENTS.md), then: uv run eval.py [--think] [--runs N] [--only PREFIX]

import argparse
import copy
import json
import re
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from app import prompt, sim
from app.sim import fold
from app.grammar import GRAMMAR_FILE, build

HERE = Path(__file__).parent
MAX_CALLS = 3          # tool calls per user message; after that the model must answer
CONTEXT_SIZE = 4096    # as ModelConfig.CONTEXT_SIZE

# Sampling as in the app (ModelConfig.kt): top-k, temperature, then a random pick. The app's presence penalty
# is left out: llama-server also counts prompt tokens in it (the user's own words get penalized), the app doesn't.
SAMPLING = {"temperature": 1.0, "top_k": 20, "samplers": ["top_k", "temperature"]}


class Server:
    def __init__(self, url: str, grammar: str, think: bool):
        self.url = url
        self.grammar = grammar
        self.n_predict = 2048 if think else 512
        self.token_cache = {}

    def post(self, path: str, body: dict) -> dict:
        request = urllib.request.Request(self.url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read())

    # Tokens of one prompt segment. Plain segments keep control markers as text, as the app does.
    def tokens(self, text: str, special: bool, first: bool = False) -> list[int]:
        key = (text, special, first)
        if key not in self.token_cache:
            body = {"content": text, "add_special": first, "parse_special": special}
            self.token_cache[key] = self.post("/tokenize", body)["tokens"]
        return self.token_cache[key]

    # Generates until the end of the answer. With tools off, <tool_call> is banned.
    # Returns the text and the exact tokens, which are kept as they are (re-tokenizing could differ).
    def generate(self, tokens: list[int], tools: bool, seed: int) -> tuple[str, list[int]]:
        body = SAMPLING | {"prompt": tokens, "n_predict": self.n_predict, "seed": seed,
                           "return_tokens": True, "cache_prompt": True}
        if tools:
            trigger = {"type": 0, "value": "<tool_call>", "token": prompt.TOOL_CALL_TOKEN}
            body |= {"grammar": self.grammar, "grammar_lazy": True, "grammar_triggers": [trigger]}
        else:
            body["logit_bias"] = [[prompt.TOOL_CALL_TOKEN, False]]
        result = self.post("/completion", body)
        generated = result["tokens"]
        # Android stops before feeding the sampled end token back into the context.
        return result["content"], generated[:-1] if result["stop_type"] == "eos" else generated

    # The GGUF's file name, e.g. Qwen3.5-2B-Q4_K_M.
    def model_name(self) -> str:
        with urllib.request.urlopen(self.url + "/props") as response:
            return Path(json.loads(response.read())["model_path"]).stem


# The tool call in a model's output as (name, args with typed values), or None if it only wrote text.
def parse_call(text: str) -> tuple[str, dict] | None:
    call = re.search(r"<tool_call>(.*)</tool_call>", text, re.S)
    if not call:
        return None
    name = re.search(r"<function=([^>]+)>", call.group(1)).group(1)
    schema = sim.TOOLS[name]["parameters"]["properties"]
    args = {}
    for param, value in re.findall(r"<parameter=([^>]+)>\n(.*?)\n</parameter>", call.group(1), re.S):
        kind = schema[param]["type"]
        args[param] = int(value) if kind == "integer" else value == "true" if kind == "boolean" \
            else json.loads(value) if kind == "array" else value
    return name, args


# What the user sees: no thinking, no tool call.
def visible(text: str) -> str:
    text = text.split("</think>")[-1]
    return re.sub(r"<tool_call>.*", "", text, flags=re.S).strip()


# Plays one case: every user message, with up to MAX_CALLS tool calls each. Returns the phone, the last
# answer and a log of everything the model wrote and every tool result.
def play(server: Server, case: dict, base: dict, tools: list[dict], think: bool, seed: int):
    now = datetime.fromisoformat(case.get("now", base["now"]))
    phone = sim.Phone(base["state"] | case.get("state", {}), now)
    tokens = list(server.tokens(prompt.system_turn(tools), True, first=True))  # a copy: the cache stays clean
    log, answer, pending, close = [], "", None, False

    def add(segments):
        for text, special in segments:
            tokens.extend(server.tokens(text, special))

    for message in case["messages"]:
        log.append({"user": message})
        phone.in_background = False
        add(prompt.user_turn(message, now, think, close, pending))
        pending, close, calls = None, True, 0
        while True:
            if len(tokens) >= CONTEXT_SIZE:
                log.append({"error": "context full"})
                return phone, "", log
            text, generated = server.generate(tokens, calls < MAX_CALLS, seed)
            tokens.extend(generated)
            log.append({"model": text})
            answer = visible(text)
            call = parse_call(text)
            if call is None:
                break
            name, args = call
            result = phone.run(name, args)
            calls += 1
            log.append({"tool": name, "args": args, "result": result})
            if sim.TOOLS[name]["hven"]["confirm"] and not result.startswith("Error:"):
                pending = result  # the app waits for the tap; the outcome goes to the next turn
                break
            add(prompt.tool_response(result, think))
    return phone, answer, log


# True if `actual` fits `expected`. Strings starting with "~" match by case-insensitive containment,
# {"<": n} means less than n, and dicts check only the keys they list.
def matches(expected, actual) -> bool:
    if isinstance(expected, str) and expected.startswith("~"):
        return isinstance(actual, str) and fold(expected[1:]) in fold(actual)  # "~lucia" matches "Lucía's birthday"
    if isinstance(expected, dict) and set(expected) == {"words"}:
        return isinstance(actual, str) and re.findall(r"\w+", fold(expected["words"])) == re.findall(r"\w+", fold(actual))
    if isinstance(expected, dict) and set(expected) == {"<"}:
        return isinstance(actual, (int, float)) and actual < expected["<"]
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(k in actual and matches(v, actual[k]) for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(expected) == len(actual) and all(map(matches, expected, actual))
    return expected == actual


# True if the answer says `part`. Numbers must stand alone ("7:00" isn't in "17:00", "1pm" isn't in "11pm",
# "28.2" is in "28.20", "28,2" in "28,20"); a phone number may also hide in the answer's digits ("+39 347-555 0101").
def in_answer(answer: str, part: str) -> bool:
    answer, part = answer.lower(), part.lower()
    if part.isdigit() and len(part) >= 7:
        return any(re.sub(r"\D", "", number) == part for number in re.findall(r"\+?\d(?:[ ().-]*\d)*", answer))
    pattern = re.escape(part)
    if part[:1].isdigit():
        pattern = r"(?<![\d:.,])" + pattern
    if part[-1:].isdigit():
        pattern += r"0*(?![\d:])" if "." in part or "," in part else r"(?![\d:])"
    return re.search(pattern, answer) is not None


# Check that the model saw the exact recipient or event ids before acting on them.
def grounded_actions(log: list) -> bool:
    numbers, emails, ids = set(), set(), set()
    for entry in log:
        if "tool" not in entry:
            continue
        name, args, result = entry["tool"], entry["args"], entry["result"]
        if name == "find_contact":
            for number, email in re.findall(r": phone ([^,\n]+), email ([^\n]+)", result):
                numbers.add(re.sub(r"[^0-9+]", "", number))
                if email != "none":
                    emails.add(email.lower())
        elif name == "find_events":
            ids.update(int(i) for i in re.findall(r"^id (\d+):", result, re.M))
        elif name in ("call", "draft_message") and args.get("number"):
            if re.sub(r"[^0-9+]", "", args["number"]) not in numbers:
                return False
        elif name == "draft_email" and args.get("to") and args["to"].lower() not in emails:
            return False
        elif name in ("edit_events", "delete_events") and not ids.issuperset(args["ids"]):
            return False
    return True


# A free/busy check must cover the whole day before booking an event.
def searched_day(calls: list, day: str) -> bool:
    for e in calls:
        if e["tool"] == "add_event":
            break
        if e["tool"] == "find_events" and not e["result"].startswith("Error:"):
            args = e["args"]
            if not args.get("query") and args["from_date"] <= day <= args.get("to_date", args["from_date"]):
                return True
    return False


# Confirmed actions end on their card. Every other completed turn needs a visible answer.
def ends_on_card(log: list) -> bool:
    return bool(log and "tool" in log[-1] and sim.TOOLS[log[-1]["tool"]]["hven"]["confirm"]
                and not log[-1]["result"].startswith("Error:"))


# Check the expected state, tool evidence and answer together.
def passed(case: dict, start: dict, phone: sim.Phone, answer: str, log: list) -> bool:
    if any("error" in e for e in log) or not (answer.strip() or ends_on_card(log)) or not grounded_actions(log):
        return False
    if "first_expect" in case:
        turns = [i for i, e in enumerate(log) if "user" in e]
        if len(turns) != 2:
            return False
        first = log[:turns[1]]
        interim = sim.Phone(start, phone.now)
        for e in first:
            if "tool" in e:
                interim.run(e["tool"], e["args"])
        first_answer = next((visible(e["model"]) for e in reversed(first) if "model" in e), "")
        if not passed({"expect": case["first_expect"]}, start, interim, first_answer, first):
            return False
    calls = [e for e in log if "tool" in e]
    for outcome in case["expect"]:
        if outcome.get("no_tool") and calls or not matches(start | outcome.get("changes", {}), phone.state):
            continue
        if "calendar_day" in outcome and not searched_day(calls, outcome["calendar_day"]):
            continue
        if not all(in_answer(answer, s) for s in outcome.get("answer_has", [])):
            continue
        if not all(any(in_answer(answer, s) for s in group) for group in outcome.get("answer_any", [])):
            continue
        if any(in_answer(answer, s) for s in outcome.get("answer_not", [])):
            continue
        if any(not any(e["tool"] == name and all(in_answer(e["result"], s) for s in facts)
                       for e in calls) for name, facts in outcome.get("reads", {}).items()):
            continue
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8080")
    parser.add_argument("--cases", default=str(HERE / "cases/core.jsonl"))
    parser.add_argument("--think", action="store_true", help="thinking on (off in the app today)")
    parser.add_argument("--runs", type=int, default=1, help="times each case runs, with different seeds")
    parser.add_argument("--only", default="", help="run only cases whose id starts with one of these (comma-separated)")
    parser.add_argument("--tools", default="", help="show the model only these tools (comma-separated)")
    args = parser.parse_args()

    grammar = GRAMMAR_FILE.read_text()
    if grammar != build():
        raise SystemExit("hven.gbnf is out of date: run `uv run python -m app.grammar`")
    tools = prompt.load_tools()
    if args.tools:
        tools = [t for t in tools if t["name"] in args.tools.split(",")]
        grammar = build(tools)
    server = Server(args.server, grammar, args.think)
    base = json.loads((HERE / "cases/phone.json").read_text())
    cases = [json.loads(line) for line in Path(args.cases).read_text().splitlines() if line.strip()]
    cases = [c for c in cases if c["id"].startswith(tuple(args.only.split(",")))]

    model = server.model_name()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = HERE / "out/results" / f"{stamp}-{model}{'-think' if args.think else ''}{'-' + str(len(tools)) + 'tools' if args.tools else ''}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    groups = {}
    with out_path.open("w") as out:
        for run in range(args.runs):
            for i, case in enumerate(cases):
                start = copy.deepcopy(sim.EMPTY_STATE) | base["state"] | case.get("state", {})
                began = time.monotonic()
                phone, answer, log = play(server, case, base, tools, args.think, seed=run * 1000 + i)
                ok = passed(case, start, phone, answer, log)
                seconds = time.monotonic() - began
                group = case["id"].split("/")[0]
                groups.setdefault(group, []).append(ok)
                print(f"{'PASS' if ok else 'FAIL'}  {case['id']:<26} {seconds:5.1f} s  {answer[:70]!r}", flush=True)
                out.write(json.dumps({"id": case["id"], "run": run, "passed": ok, "seconds": round(seconds, 1),
                                      "answer": answer, "log": log, "state": phone.state}) + "\n")

    print(f"\n{model}{' (thinking)' if args.think else ''}")
    for group, results in groups.items():
        print(f"  {group:<10} {sum(results)}/{len(results)}")
    total = [ok for results in groups.values() for ok in results]
    print(f"  {'total':<10} {sum(total)}/{len(total)}  ({100 * sum(total) / len(total):.0f}%)")
    print(f"details: {out_path}")


if __name__ == "__main__":
    main()
