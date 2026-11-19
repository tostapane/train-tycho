# Qwen3.5's prompt format, the Python twin of ModelConfig.kt: system turn, user turns, tool responses.
# A prompt is a list of segments (text, special): special=True means control markers like <|im_start|>
# are real tokens (our template); False means plain text (user messages, tool results).

import json
from datetime import datetime, timedelta
from pathlib import Path

TOOLS_FILE = Path(__file__).parent / "tools.json"

# <tool_call> is one token in Qwen3.5's vocabulary; the grammar switches on when it appears.
TOOL_CALL_TOKEN = 248058

SYSTEM_PROMPT = "You are Tycho, a private assistant that runs offline on the user's phone. Answer directly and briefly."

# Copied word for word from the chat template in the GGUF.
TOOL_INSTRUCTIONS = (
    "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call>\n\n"
    "<IMPORTANT>\nReminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
    "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
    "</IMPORTANT>"
)


# The tools in tools.json, with Hven's own fields ("hven": kind, confirm, opens_app).
def load_tools() -> list[dict]:
    return json.loads(TOOLS_FILE.read_text())


# One tool as the template prints it: Python's json.dumps of the schema, without Hven's fields.
def tool_line(tool: dict) -> str:
    schema = {"name": tool["name"], "description": tool["description"], "parameters": tool["parameters"]}
    return json.dumps({"type": "function", "function": schema})


# The system turn as the template builds it when tools are present: tools, instructions, our prompt.
# `formats` is how the user writes dates and times, e.g. ("day/month", "24-hour"); left out for the bare model,
# which misreads it (10 min became 10 hours), so it only goes in with training.
def system_turn(tools: list[dict], formats: tuple[str, str] | None = None) -> str:
    lines = "\n".join(tool_line(t) for t in tools)
    habits = f" The user writes dates as {formats[0]} and uses the {formats[1]} clock." if formats else ""
    return (
        "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n"
        + lines + "\n</tools>\n\n" + TOOL_INSTRUCTIONS + "\n\n" + SYSTEM_PROMPT + habits + "<|im_end|>\n"
    )


# Start of the model's answer. Thinking off is an empty <think> block, as in the app.
def assistant_start(think: bool) -> str:
    return "<|im_start|>assistant\n<think>\n" if think else "<|im_start|>assistant\n<think>\n\n</think>\n\n"


# Segments of a user turn. `close` ends the model's previous answer (the app never feeds its end token);
# `tool_result` reports a confirmed action from the previous turn, as the app does.
def user_turn(text: str, now: datetime, think: bool, close: bool, tool_result: str | None = None) -> list:
    segments = []
    start = "<|im_end|>\n" if close else ""
    if tool_result is not None:
        segments += [(start + "<|im_start|>user\n<tool_response>\n", True), (tool_result, False),
                     ("\n</tool_response><|im_end|>\n", True)]
        start = ""
    segments += [(start + "<|im_start|>user\n" + date_list(now) + "\n\n", True), (text, False),
                 ("<|im_end|>\n" + assistant_start(think), True)]
    return segments


# Segments that hand a tool's result back to the model in the middle of its answer.
def tool_response(result: str, think: bool) -> list:
    return [("<|im_end|>\n<|im_start|>user\n<tool_response>\n", True), (result, False),
            ("\n</tool_response><|im_end|>\n" + assistant_start(think), True)]


# Dates the model can look up, so it never does date arithmetic. Same text as ModelConfig.dateList:
#   Today: Wednesday 2026-11-18, 10:00
#   Tomorrow: Thursday 2026-11-19
#   Friday: 2026-11-20
#   ...
#   Next Wednesday: 2026-11-25
def date_list(now: datetime) -> str:
    today = now.date()

    def day(offset: int):
        d = today + timedelta(days=offset)
        return d.strftime("%A"), d.isoformat()

    lines = [f"Today: {day(0)[0]} {day(0)[1]}, {now.strftime('%H:%M')}", "Tomorrow: {} {}".format(*day(1))]
    lines += ["{}: {}".format(*day(offset)) for offset in range(2, 7)]
    lines.append("Next {}: {}".format(*day(7)))
    return "\n".join(lines)
