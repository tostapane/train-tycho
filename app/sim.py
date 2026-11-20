# The fake phone: calendar, contacts, notes, reminders, clock, media, and a log of what was opened.
# Each tool works like its Android version and returns the text the model reads back.
# Used to score the eval, check generated training data, and later as the RL reward.

import ast
import copy
import operator
import re
from datetime import date, datetime, time, timedelta

from app.prompt import load_tools

TOOLS = {t["name"]: t for t in load_tools()}

# A phone with nothing on it. A case's state replaces any of these keys.
EMPTY_STATE = {
    "events": [],      # {"id", "title", "date", "start": "HH:MM" or None (all day), "duration_minutes", "repeat"}
    "contacts": [],    # {"name", "phone", "email" or None}
    "notes": [],       # {"text"}
    "reminders": [],   # {"text", "date", "time"}
    "alarms": [],      # {"time": "HH:MM", "days": weekdays it repeats on, empty = once}
    "timers": [],      # seconds
    "media": {"playing": True, "track": 1, "volume": 50},
    "flashlight": False,
    "apps": ["Calendar", "Camera", "Clock", "Contacts", "Firefox", "Gallery", "Messages", "Organic Maps", "Phone", "Signal"],
    "opened": [],      # other apps Hven opened: {"tool", ...arguments}
}


class ToolError(Exception):
    pass


class Phone:
    def __init__(self, state: dict, now: datetime):
        self.state = copy.deepcopy(EMPTY_STATE) | copy.deepcopy(state)
        self.now = now
        # Once Hven opens another app it's in the background until the user's next message,
        # and Android won't let it open a second one.
        self.in_background = False

    # Runs one tool call and returns what the model reads back. Bad values come back as an error line.
    def run(self, name: str, args: dict) -> str:
        if self.in_background and TOOLS[name]["hven"]["opens_app"]:
            return "Error: Hven is in the background, so it can't open another app."
        if "number" in args:
            # Android dials "+1 (415) 555-0132" and "+14155550132" alike.
            args = args | {"number": re.sub(r"[^0-9+]", "", args["number"])}
        try:
            result = getattr(self, "tool_" + name)(**args)
        except ToolError as e:
            return f"Error: {e}"
        except TypeError:
            return "Error: missing or unknown parameters."
        if TOOLS[name]["hven"]["opens_app"]:
            self.state["opened"].append({"tool": name, **args})
            self.in_background = True
        return result

    def tool_set_timer(self, hours=0, minutes=0, seconds=0):
        total = hours * 3600 + minutes * 60 + seconds
        if not 1 <= total <= 86400:
            raise ToolError("a timer must last between 1 second and 24 hours.")
        self.state["timers"].append(total)
        return f"Timer started: {total // 3600} h {total % 3600 // 60} min {total % 60} s."

    def tool_set_alarm(self, hour, minute, days=None):
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ToolError("hour must be 0-23 and minute 0-59.")
        self.state["alarms"].append({"time": f"{hour:02}:{minute:02}", "days": days or []})
        return f"Alarm set for {hour:02}:{minute:02}" + (f", every {', '.join(days)}." if days else ".")

    # Events day by day, each day by start time (all-day first), as calendar apps list them.
    def tool_find_events(self, from_date, to_date=None):
        start, end = parse_date(from_date), parse_date(to_date or from_date)
        if (end - start).days > 400:
            raise ToolError("search at most 400 days at a time.")
        lines = []
        for offset in range((end - start).days + 1):
            day = start + timedelta(days=offset)
            on_day = sorted((e for e in self.state["events"] if occurs(e, day)), key=lambda e: e["start"] or "")
            lines += [event_line(e, day) for e in on_day]
        return "\n".join(lines) or "No events."

    def tool_add_event(self, title, date, start=None, duration_minutes=60, repeat=None):
        parse_date(date)
        if start is not None:
            parse_time(start)
        event = {"id": max((e["id"] for e in self.state["events"]), default=0) + 1, "title": title, "date": date,
                 "start": start, "duration_minutes": duration_minutes if start else None, "repeat": repeat}
        self.state["events"].append(event)
        return "Added: " + event_line(event)

    def tool_edit_events(self, ids, title=None, date=None, start=None, duration_minutes=None,
                         shift_days=0, shift_minutes=0):
        events = self.events_by_id(ids)
        for e in events:
            new_start = start or e["start"]
            moved = datetime.combine(parse_date(date or e["date"]), parse_time(new_start or "00:00"))
            moved += timedelta(days=shift_days, minutes=shift_minutes)
            e["date"], e["start"] = moved.date().isoformat(), moved.strftime("%H:%M") if new_start else None
            if title is not None:
                e["title"] = title
            if duration_minutes is not None:
                e["duration_minutes"] = duration_minutes
            if e["start"] and e["duration_minutes"] is None:
                e["duration_minutes"] = 60
        return "Changed:\n" + "\n".join(event_line(e) for e in events)

    def tool_delete_events(self, ids):
        events = self.events_by_id(ids)
        self.state["events"] = [e for e in self.state["events"] if e not in events]
        return "Deleted:\n" + "\n".join(event_line(e) for e in events)

    def tool_find_contact(self, name):
        words = name.lower().split()
        found = [c for c in self.state["contacts"] if all(w in c["name"].lower() for w in words)]
        lines = [f"{c['name']}: phone {c['phone']}, email {c['email'] or 'none'}" for c in found]
        return "\n".join(lines) or f"No contact matches '{name}'."

    def tool_call(self, number):
        return f"Calling {number}."

    def tool_draft_message(self, text, number=None, app="sms"):
        return f"Message ready in {app}, " + (f"on the chat with {number}." if number else "the user picks the chat.")

    def tool_draft_email(self, body, to=None, subject=None):
        return f"Email {'to ' + to if to else 'without a recipient'} ready in the email app."

    def tool_add_reminder(self, text, date, time):
        when = datetime.combine(parse_date(date), parse_time(time))
        if when <= self.now:
            raise ToolError("that time has already passed.")
        self.state["reminders"].append({"text": text, "date": date, "time": time})
        return f"Reminder set for {date} {time}."

    def tool_add_note(self, text):
        self.state["notes"].append({"text": text})
        return "Note saved."

    def tool_find_notes(self, query):
        words = query.lower().split()
        found = [n["text"] for n in self.state["notes"] if all(w in n["text"].lower() for w in words)]
        return "\n".join(found) or "No notes match."

    def tool_open_app(self, name):
        if not any(name.lower() == a.lower() for a in self.state["apps"]):
            raise ToolError(f"no app called '{name}'. Installed: {', '.join(self.state['apps'])}.")
        return f"Opened {name}."

    def tool_navigate(self, destination):
        return f"Directions to {destination} opened."

    def tool_flashlight(self, on):
        self.state["flashlight"] = on
        return f"Flashlight {'on' if on else 'off'}."

    def tool_open_settings(self, panel):
        return f"Opened the {panel} settings."

    def tool_media_control(self, action):
        media = self.state["media"]
        if action in ("play", "pause"):
            media["playing"] = action == "play"
        else:
            media["track"] = max(1, media["track"] + (1 if action == "next" else -1))
        return "Done."

    def tool_volume(self, direction=None, percent=None):
        media = self.state["media"]
        if percent is not None:
            media["volume"] = percent
        elif direction is not None:
            media["volume"] += 10 if direction == "up" else -10
        else:
            raise ToolError("give a direction or a percent.")
        media["volume"] = max(0, min(100, media["volume"]))
        return f"Volume {media['volume']}%."

    def tool_play_music(self, query):
        return f"Playing {query}."

    def tool_calculate(self, expression):
        value = evaluate(expression)
        return f"{expression} = {round(value, 6)}"

    def tool_days_until(self, date):
        days = (parse_date(date) - self.now.date()).days
        return f"{days} days from today to {date}."

    # The events with these ids, or an error naming the missing ones.
    def events_by_id(self, ids):
        events = [e for e in self.state["events"] if e["id"] in ids]
        missing = set(ids) - {e["id"] for e in events}
        if missing:
            raise ToolError(f"no event with id {', '.join(map(str, sorted(missing)))}.")
        return events


# One event on one day, e.g. "id 3: Dentist, Thursday 2026-11-19 10:00, 60 min"
# or "id 8: Anna's birthday, Thursday 2026-11-26, all day, repeats yearly".
def event_line(e: dict, day: date | None = None) -> str:
    day = day or parse_date(e["date"])
    when = f"{e['start']}, {e['duration_minutes']} min" if e["start"] else "all day"
    repeat = f", repeats {e['repeat']}" if e.get("repeat") else ""
    return f"id {e['id']}: {e['title']}, {day.strftime('%A')} {day.isoformat()} {when}{repeat}"


# True if the event happens on that day, counting repeats.
def occurs(e: dict, day: date) -> bool:
    first = parse_date(e["date"])
    repeat = e.get("repeat")
    if day < first or (not repeat and day != first):
        return False
    return (not repeat or repeat == "daily" or (repeat == "weekdays" and day.weekday() < 5)
            or (repeat == "weekly" and day.weekday() == first.weekday())
            or (repeat == "monthly" and day.day == first.day)
            or (repeat == "yearly" and (day.month, day.day) == (first.month, first.day)))


def parse_date(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise ToolError(f"'{text}' is not a real date.")


def parse_time(text: str) -> time:
    try:
        return time.fromisoformat(text)
    except ValueError:
        raise ToolError(f"'{text}' is not a real time.")


OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
             ast.Mod: operator.mod, ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}


# Arithmetic only: numbers, + - * / % ** and brackets. Anything else is an error, never executed.
def evaluate(expression: str) -> float:
    def walk(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            if isinstance(node.op, ast.Pow) and abs(walk(node.right)) > 100:
                raise ToolError("exponent too large.")
            return OPERATORS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in OPERATORS:
            return OPERATORS[type(node.op)](walk(node.operand))
        raise ToolError(f"can't compute '{expression}'.")

    try:
        return walk(ast.parse(expression, mode="eval").body)
    except (SyntaxError, ZeroDivisionError, OverflowError):
        raise ToolError(f"can't compute '{expression}'.")
