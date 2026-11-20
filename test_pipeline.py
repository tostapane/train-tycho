# Local regressions for the generation and evaluation checks. No teacher or GPU needed.
# Run with uv run python -m unittest.

import json
import unittest
from datetime import datetime
from unittest.mock import Mock

import eval
import generate
from app import sim
from gen import checks, recipes


# Return exactly the replies a real completions server would supply, including tool-call markup.
def teacher(*replies):
    model = Mock()
    outputs = []
    for reply in replies:
        if isinstance(reply, tuple):
            name, args = reply
            params = "".join(f"<parameter={key}>\n{value}\n</parameter>\n" for key, value in args.items())
            reply = f"<tool_call>\n<function={name}>\n{params}</function>\n"  # the server stops before </tool_call>
        outputs.append((reply, False))
    model.complete.side_effect = outputs
    return model


# Exercise the real parser, simulator, conversation loop and scorer.
def grade(case, *replies):
    case = case | {"messages": [case["task"]] + ([case["followup"]] if case["followup"] else [])}
    return generate.check(case, generate.live_run(teacher(*replies), case, 0))


class PipelineTests(unittest.TestCase):
    # Both timers belong in the final state, but only the first belongs in the first turn.
    def test_timer_turns(self):
        case = recipes.draw(34)
        first = ("set_timer", {"minutes": 41})
        second = ("set_timer", {"minutes": 42})
        end = "The 42-minute timer started. The old one is still running in Clock."
        self.assertIsNone(grade(case, first, "Started a timer for 41 minutes.", second, end))
        self.assertIsNotNone(grade(case, first, second, "Both timers started.", end))
        for message, followup, valid in [("Timer for 41 minutes", "make it 42", True),
                                         ("Timer for 42 minutes", "make it 41", False)]:
            reply = json.dumps({"message": message, "followup": followup})
            result = generate.write_messages(teacher(reply), case)
            self.assertEqual("dropped" not in result, valid)

    # A draft must preserve its content and use a recipient returned by an earlier lookup.
    def test_email(self):
        case = recipes.draw(149)
        target = case["expect"][0]["changes"]["opened"][0]
        name = case["say"][0]
        lookup = ("find_contact", {"name": name})
        draft = ("draft_email", {"to": target["to"], "body": "The slides are attached."})
        answer = f"Your email to {name} is ready."
        self.assertIsNone(grade(case, lookup, draft, answer))
        self.assertIsNotNone(grade(case, draft, answer))
        self.assertIsNotNone(grade(case, ("find_contact", {"name": "unknown"}), draft, answer))
        self.assertIsNotNone(grade(case, lookup, ("draft_email", {"to": target["to"], "body": "Bananas."}), answer))

    # A formatted complete number passes. A suffix, longer number or guessed answer fails.
    def test_contact_number(self):
        case = recipes.draw(82)
        number = case["read_has"][0]
        lookup = ("find_contact", {"name": "taxi"})
        self.assertIsNone(grade(case, lookup, number))
        self.assertIsNotNone(grade(case, number))
        for wrong in (recipes.digits(number)[-7:], number + "1"):
            self.assertIsNotNone(grade(case, lookup, wrong))

    # A disambiguating reply can reuse the previous turn's lookup.
    def test_contact_followup(self):
        case = recipes.draw(17)
        lookup = ("find_contact", {"name": case["first_turn"]["say"][0]})
        names = case["first_turn"]["read_has"]
        call = ("call", {"number": case["expect"][0]["changes"]["opened"][0]["number"]})
        self.assertIsNone(grade(case, lookup, f"{names[0]} or {names[1]}?", call))
        self.assertIsNotNone(grade(case, lookup, "Which one?", call))

    # An ordinal always resolves to its next occurrence. Units must stay attached to their value.
    def test_recipe_facts(self):
        case = recipes.draw(59)
        self.assertIn("3 July", case["followup"])
        self.assertNotIn("the 3rd", case["followup"])
        self.assertFalse(checks.says("5 hours and 30 seconds", "5 seconds"))
        self.assertTrue(checks.says("an hour and a half", "90 minutes"))
        self.assertTrue(generate.contains("open signal please", "open signal"))

    # Silence cannot pass a core case. An offline disclaimer followed by a forecast also fails.
    def test_core_answers(self):
        base = json.loads((eval.HERE / "cases/phone.json").read_text())
        for line in (eval.HERE / "cases/core.jsonl").read_text().splitlines():
            case = json.loads(line)
            phone = sim.Phone(base["state"] | case.get("state", {}), datetime.fromisoformat(base["now"]))
            self.assertFalse(eval.passed(case, phone.state, phone, "", []), case["id"])
            if case["id"] == "out/weather":
                self.assertTrue(eval.passed(case, phone.state, phone, "I can't check live weather offline.", []))
                self.assertFalse(eval.passed(case, phone.state, phone, "I'm offline but it will rain.", []))

    # Only an actual end token gets removed. A length-limited response keeps its last token.
    def test_token_history(self):
        server = eval.Server("unused", "", False)
        for stop, wanted in [("eos", [23]), ("limit", [23, 42])]:
            server.post = Mock(return_value={"tokens": [23, 42], "content": "text", "stop_type": stop})
            self.assertEqual(server.generate([], False, 1)[1], wanted)
