# Generates hven.gbnf from tools.json: the grammar for what follows <tool_call>, so the model
# can only write a call to one of our tools, with parameters in schema order and values of the right shape.
# Also a regex twin of it, to check calls written without the grammar (the teacher's).
# Run after changing tools.json: uv run python -m app.grammar

import json
import re
from pathlib import Path

from app.prompt import load_tools

GRAMMAR_FILE = Path(__file__).parent / "hven.gbnf"

# Value shapes. Ranges (hour <= 23) and real dates are checked by the tools, not here.
COMMON_RULES = """
integer ::= "-"? [0-9]{1,6}
text ::= [^<\\n]+
date ::= [0-9]{4} "-" [0-9]{2} "-" [0-9]{2}
time ::= [0-9]{2} ":" [0-9]{2}
boolean ::= "true" | "false"
"""


# A GBNF string literal.
def literal(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


# The rule for one parameter's value, from its JSON schema.
def value_rule(schema: dict) -> str:
    if schema["type"] == "array":
        items = schema["items"]
        item = "(" + " | ".join(literal(json.dumps(v)) for v in items["enum"]) + ")" if "enum" in items else value_rule(items)
        return f'"[" {item} ("," " "? {item})* "]"'
    if "enum" in schema:
        return "(" + " | ".join(literal(v) for v in schema["enum"]) + ")"
    if schema["type"] == "string":
        return schema.get("format", "text")
    return schema["type"]


# Every rule of the grammar, as GBNF text, for all tools or only the given ones.
def build(tools: list[dict] | None = None) -> str:
    tools = tools or load_tools()
    rules = []
    names = []
    for tool in tools:
        name = tool["name"].replace("_", "-")
        names.append(name)
        params = tool["parameters"]["properties"]
        required = tool["parameters"]["required"]
        parts = [literal(tool["name"] + ">\n")]
        for param, schema in params.items():
            rule = f"{name}-{param.replace('_', '-')}"
            opening = literal(f"<parameter={param}>\n")
            closing = literal("\n</parameter>\n")
            rules.append(f"{rule} ::= {opening} {value_rule(schema)} {closing}")
            parts.append(rule if param in required else rule + "?")
        rules.append(f"{name} ::= " + " ".join(parts))
    root = "root ::= " + literal("<tool_call>\n<function=") + " (" + " | ".join(names) + ") " + literal("</function>\n</tool_call>")
    return root + "\n\n" + "\n".join(rules) + "\n" + COMMON_RULES


# The same value shapes as regexes.
COMMON_PATTERNS = {"integer": r"-?[0-9]{1,6}", "text": r"[^<\n]+", "date": r"[0-9]{4}-[0-9]{2}-[0-9]{2}",
                   "time": r"[0-9]{2}:[0-9]{2}", "boolean": r"(?:true|false)"}


# The regex for one parameter's value: twin of value_rule.
def value_pattern(schema: dict) -> str:
    if schema["type"] == "array":
        items = schema["items"]
        item = "(?:" + "|".join(re.escape(json.dumps(v)) for v in items["enum"]) + ")" if "enum" in items else value_pattern(items)
        return rf"\[{item}(?:, ?{item})*\]"
    if "enum" in schema:
        return "(?:" + "|".join(re.escape(v) for v in schema["enum"]) + ")"
    return COMMON_PATTERNS[schema.get("format", "text") if schema["type"] == "string" else schema["type"]]


# A regex for exactly the calls the grammar allows: twin of build().
def call_pattern(tools: list[dict] | None = None) -> str:
    calls = []
    for tool in tools or load_tools():
        parts = [re.escape(tool["name"] + ">\n")]
        for param, schema in tool["parameters"]["properties"].items():
            part = re.escape(f"<parameter={param}>\n") + value_pattern(schema) + re.escape("\n</parameter>\n")
            parts.append(part if param in tool["parameters"]["required"] else f"(?:{part})?")
        calls.append("".join(parts))
    return re.escape("<tool_call>\n<function=") + "(?:" + "|".join(calls) + ")" + re.escape("</function>\n</tool_call>")


if __name__ == "__main__":
    GRAMMAR_FILE.write_text(build())
    print(f"wrote {GRAMMAR_FILE}")
