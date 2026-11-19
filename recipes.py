# Recipes for generated cases: code picks what the user wants and computes the right outcome; the teacher only
# writes the words. A recipe = phone + now + a task for the message writer + the words the message must contain
# + the expected outcome (eval format, plus answer patterns). Categories and weights follow AGENT-design.md.

import copy
import random
import re
from datetime import date, datetime, timedelta

from phones import INGREDIENTS, contact, is_person, random_now, random_phone
from sim import occurs

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
TEXTS = ["running late", "on my way", "call me when you can", "happy birthday!", "dinner at 8?", "see you soon",
         "can you pick up milk?", "thanks for today", "I'm outside", "the meeting moved to 3", "landed safely",
         "don't forget the keys"]
CHAT = ["thank the assistant", "say good morning", "say the last thing worked, nice", "ask how it is doing",
        "ask a general knowledge question (science, geography, language)", "ask for a word that means happy",
        "ask for a quick cooking tip", "ask what a word means"]
NEGATIONS = ["not to set an alarm because they'll wake up on their own", "not to add anything, they were just thinking aloud",
             "not to call anyone yet", "not to set a timer, they changed their mind", "to forget the last idea"]
MESSY = ["typos", "slang", "terse", "lowercase", "run-on"]
NOT_FOUND = ["can't", "cannot", "couldn't", "could not", "no contact", "not in your", "don't have", "isn't", "not find"]
NOT_INSTALLED = ["not installed", "isn't installed", "can't find", "couldn't find", "don't see", "no app", "not on your"]
NO_EMAIL = ["no email", "email address", "don't have", "doesn't have", "no address", "isn't saved", "not saved"]


# Normalizes a number the way the fake phone (and Android) does.
def digits(number: str) -> str:
    return re.sub(r"[^0-9+]", "", number)


# The whole state, unchanged: the expected outcome of a case where nothing should happen.
def nothing(answer_has=None, answer_any=None, answer_not=None) -> list[dict]:
    return [{"changes": {}, "answer_has": answer_has or [], "answer_any": answer_any or [], "answer_not": answer_not or []}]


def ordinal(n: int) -> str:
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


class Draw:
    """One recipe being drawn: a seeded random, a now, a phone, the user's date and clock habits."""

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.now = random_now(self.rng)
        self.phone = random_phone(self.rng, self.now)
        us = any(c["phone"].startswith("+1") for c in self.phone["contacts"][:3])
        self.formats = ("month/day", "12-hour") if us else (self.rng.choice(["day/month", "day/month", "month/day"]),
                                                           self.rng.choice(["24-hour", "24-hour", "12-hour"]))

    # A future day, and how the user refers to it ("tomorrow", "friday", "the 24th", "24/11").
    def day(self, max_offset: int = 20, allow_today: bool = True):
        low = 0 if allow_today and self.now.hour < 20 else 1
        offset = self.rng.randint(low, max_offset)
        day = self.now.date() + timedelta(days=offset)
        if offset == 0:
            words = "today" if self.now.hour < 15 else self.rng.choice(["today", "tonight"])
        elif offset == 1:
            words = "tomorrow"
        elif offset <= 6:
            words = WEEKDAYS[day.weekday()]
        else:
            numeric = f"{day.day}/{day.month}" if self.formats[0] == "day/month" else f"{day.month}/{day.day}"
            words = self.rng.choice([f"the {ordinal(day.day)}", f"{day.day} {day.strftime('%B')}", numeric])
        return day, words

    # A time inside `hours`, and how the user says it ("7pm", "19:00", or a bare "7" when it can only mean one thing).
    def time(self, hours: tuple[int, int], day=None):
        low = hours[0]
        if day == self.now.date():
            low = max(low, self.now.hour + 1)
        hour = self.rng.randint(min(low, hours[1]), hours[1])
        minute = self.rng.choice([0, 0, 0, 15, 30, 45])
        hhmm = f"{hour:02d}:{minute:02d}"
        bare = f"{(hour - 1) % 12 + 1}" + (f":{minute:02d}" if minute else "")
        if not hours[0] <= (hour + 12) % 24 <= hours[1] and self.rng.random() < 0.3:
            words = bare
        else:
            words = hhmm if self.formats[1] == "24-hour" and self.rng.random() < 0.6 else twelve(hhmm)
        return hhmm, words

    # How the answer may name a day: today/tonight, tomorrow, a weekday within a week, else day and month.
    def answer_day(self, day: date) -> list[str]:
        offset = (day - self.now.date()).days
        if offset == 0:
            return ["today", "tonight", "this evening", "this afternoon", "this morning"]
        if offset == 1:
            return ["tomorrow"]
        if offset <= 6:
            return [WEEKDAYS[day.weekday()]]
        month = day.strftime("%B").lower()
        return [f"{day.day} {month}", f"{ordinal(day.day)} {month}", f"{month} {day.day}", f"{month} {ordinal(day.day)}"]

    # How the answer must write a time, in the user's clock: "19:00" or "7pm".
    def answer_time(self, hhmm: str) -> list[str]:
        return [hhmm, hhmm.lstrip("0") if hhmm[1] != ":" else hhmm] if self.formats[1] == "24-hour" else [twelve(hhmm)]

    # A contact; `real` keeps only people (not "Mum" or "Dentist"), `with_email` filters on having an email.
    def person(self, with_email: bool | None = None, real: bool = False) -> dict:
        people = [c for c in self.phone["contacts"] if (with_email is None or bool(c["email"]) == with_email)
                  and (not real or is_person(c))]
        return self.rng.choice(people or self.phone["contacts"])

    def future_events(self, days: int = 14) -> list[dict]:
        end = self.now.date() + timedelta(days=days)
        return [e for e in self.phone["events"] if not e["repeat"] and e["start"]
                and self.now.date() < datetime.fromisoformat(e["date"]).date() <= end]

    # The finished recipe. `task` tells the writer what the user wants; `say` lists words the messages must contain.
    def recipe(self, category: str, kind: str, task: str, expect: list, say: list[str], followup: str | None = None):
        return {"category": category, "kind": kind, "now": self.now.isoformat(timespec="minutes"),
                "formats": list(self.formats), "phone": self.phone, "task": task, "followup": followup,
                "say": [s for s in say if s], "expect": expect}


# "19:30" → "7:30pm", "07:00" → "7am".
def twelve(hhmm: str) -> str:
    hour, minute = map(int, hhmm.split(":"))
    return f"{(hour - 1) % 12 + 1}" + (f":{minute:02d}" if minute else "") + ("am" if hour < 12 else "pm")


# How the user names a contact: "mum", "the dentist", "Sara", or "Sara Romano" when two share the first name.
def name_words(d: Draw, c: dict) -> str:
    if not is_person(c):
        return c["name"].lower() if c["name"] in INGREDIENTS["family"] else "the " + c["name"].lower()
    first = c["name"].split()[0]
    same = [x for x in d.phone["contacts"] if x["name"].split()[0] == first]
    return c["name"] if len(same) > 1 else d.rng.choice([first, first, c["name"]])


# The word that proves a note was found or saved: the last word of its value ("pinetree42", "B12").
def note_key(text: str) -> str:
    return text.split(":", 1)[-1].strip().split(",")[0].split()[-1].strip(".")


# One outcome with a changed state and the answer naming the day and/or time.
def done(changes: dict, d: Draw, day: date | None = None, hhmm: str | None = None, extra: list | None = None) -> list:
    groups = ([d.answer_day(day)] if day else []) + ([d.answer_time(hhmm)] if hhmm else []) + (extra or [])
    return [{"changes": changes, "answer_any": groups}]


# --- Look up, then act ---

def call_one(d: Draw):
    c = d.person()
    words = name_words(d, c)
    return d.recipe("lookup", "call_one", f"Call {words} (in contacts as \"{c['name']}\").",
                    [{"changes": {"opened": [{"tool": "call", "number": digits(c["phone"])}]}}], [words])


def message_one(d: Draw):
    c, text = d.person(), d.rng.choice(TEXTS)
    words = name_words(d, c)
    app = d.rng.choice(["sms"] * 3 + ["whatsapp", "telegram", "signal"])
    via = "" if app == "sms" else f" on {app.capitalize()} (name the app)"
    return d.recipe("lookup", "message_one", f"Text {words}{via} saying \"{text}\" (those words exactly; any typos go elsewhere).",
                    [{"changes": {"opened": [{"tool": "draft_message", "number": digits(c["phone"]),
                                              "text": "~" + text.rstrip("!?"), "app": app}]}}],
                    [words, text, "" if app == "sms" else app])


def email_one(d: Draw):
    c = d.person(with_email=True)
    if not c["email"]:
        return call_one(d)
    words = name_words(d, c)
    topic = d.rng.choice(["the report is ready", "the slides are attached", "I'll be late tomorrow", "thanks for the help"])
    return d.recipe("lookup", "email_one", f"Email {words} that {topic}.",
                    [{"changes": {"opened": [{"tool": "draft_email", "to": c["email"]}]}}], [words, "email"])


def no_email(d: Draw):
    c = d.person(with_email=False)
    if c["email"]:
        return email_one(d)
    words = name_words(d, c)
    return d.recipe("lookup", "no_email", f"Email {words} a short note (they have no email address saved).",
                    nothing(answer_any=[NO_EMAIL], answer_not=["ready", "sent"]), [words, "email"])


def contact_missing(d: Draw):
    names = {c["name"].split()[0] for c in d.phone["contacts"]}
    region = d.rng.choice(list(INGREDIENTS["regions"].values()))
    first = d.rng.choice([n for n in region["first"] if n not in names] or ["Zoltan"])
    return d.recipe("lookup", "contact_missing", f"Call {first} (there is no {first} in the contacts).",
                    nothing([first], [NOT_FOUND], ["calling"]), [first])


# An event with a title no other event has, so "cancel the yoga" means one thing. None if there's none.
def unique_event(d: Draw):
    events = [e for e in d.future_events() if sum(x["title"] == e["title"] for x in d.phone["events"]) == 1]
    return d.rng.choice(events) if events else None


def event_delete(d: Draw):
    e = unique_event(d)
    if not e:
        return call_one(d)
    day = datetime.fromisoformat(e["date"]).date()
    offset = (day - d.now.date()).days
    when = "tomorrow" if offset == 1 else WEEKDAYS[day.weekday()] if offset < 7 else ""
    rest = [x for x in d.phone["events"] if x["id"] != e["id"]]
    return d.recipe("lookup", "event_delete", f"Cancel \"{e['title']}\"" + (f" ({when})" if when else "") + ".",
                    [{"changes": {"events": rest}}], [e["title"], when])


def event_move(d: Draw):
    e = unique_event(d)
    if not e:
        return call_one(d)
    new_day, day_words = d.day(allow_today=False)
    hhmm, time_words = d.time((8, 20))
    moved = [x | ({"date": new_day.isoformat(), "start": hhmm} if x["id"] == e["id"] else {}) for x in d.phone["events"]]
    return d.recipe("lookup", "event_move", f"Move \"{e['title']}\" to {day_words} at {time_words}.",
                    done({"events": moved}, d, new_day, hhmm), [e["title"].split()[0], day_words, time_words])


# --- Dates, times, units ---

def event_add(d: Draw, category="dates"):
    kind = d.rng.choice(INGREDIENTS["events"])
    title = kind["title"].replace("{person}", d.person(real=True)["name"].split()[0]).replace("{city}", d.rng.choice(INGREDIENTS["cities"]))
    day, day_words = d.day()
    hhmm, time_words = d.time(tuple(kind["hours"]), day)
    minutes = d.rng.choice(kind["minutes"] + [None, None])
    length = f", for {minutes} minutes (say it naturally: 'for an hour', '90 min'…)" if minutes else ""
    key = title.split(" with ")[0].split(" to ")[0]
    new = {"title": "~" + key, "date": day.isoformat(), "start": hhmm}
    if minutes:
        new["duration_minutes"] = minutes
    return d.recipe(category, "event_add", f"Add \"{title}\" on {day_words} at {time_words}{length}.",
                    done({"events": d.phone["events"] + [new]}, d, day, hhmm), [key, day_words, time_words])


def reminder_add(d: Draw):
    task = d.rng.choice(INGREDIENTS["reminders"]).replace("{person}", d.person(real=True)["name"].split()[0])
    day, day_words = d.day()
    hhmm, time_words = d.time((7, 21), day)
    key = task.split()[-1].lower()
    return d.recipe("dates", "reminder_add", f"Be reminded to \"{task.lower()}\" {day_words} at {time_words}.",
                    done({"reminders": [{"text": "~" + key, "date": day.isoformat(), "time": hhmm}]}, d, day, hhmm),
                    [key, day_words, time_words])


def alarm_add(d: Draw):
    hhmm, time_words = d.time((4, 10))
    wake = d.rng.random() < 0.5
    task = f"Be woken up at {time_words}" if wake else f"An alarm at {time_words}"
    every = ""
    if d.rng.random() < 0.3:
        days = d.rng.choice([WEEKDAYS[:5], WEEKDAYS[5:], [d.rng.choice(WEEKDAYS)]])
        every = "weekday" if days == WEEKDAYS[:5] else "weekend" if days == WEEKDAYS[5:] else days[0]
        task += " every " + every
    else:
        days = []
        task += d.rng.choice(["", " tomorrow"])
    return d.recipe("dates", "alarm_add", task + ".",
                    done({"alarms": d.phone["alarms"] + [{"time": hhmm, "days": days}]}, d, hhmm=hhmm),
                    [time_words, "every " + every if every else ""])


def timer_units(d: Draw):
    seconds, words = d.rng.choice([(90, "90 seconds"), (5400, "an hour and a half"), (300, "5 minutes"),
                                   (2700, "45 minutes"), (7200, "2 hours"), (30, "30 seconds"), (1200, "20 minutes"),
                                   (4500, "an hour and 15 minutes"), (600, "10 minutes"), (900, "a quarter of an hour"),
                                   (180, "3 minutes"), (3600, "an hour")])
    return d.recipe("dates", "timer_units", f"A timer for {words} (say 'timer' or 'countdown').",
                    [{"changes": {"timers": [seconds]}}], [words])


def event_repeat(d: Draw):
    kind = d.rng.choice([k for k in INGREDIENTS["events"] if "{" not in k["title"]])
    weekday = d.rng.randint(0, 6)
    day = d.now.date() + timedelta(days=(weekday - d.now.weekday()) % 7 or 7)
    hhmm, time_words = d.time(tuple(kind["hours"]))
    new = {"title": "~" + kind["title"], "date": day.isoformat(), "start": hhmm, "repeat": "weekly"}
    return d.recipe("dates", "event_repeat", f"Add \"{kind['title']}\" every {WEEKDAYS[weekday]} at {time_words}, starting this coming one.",
                    done({"events": d.phone["events"] + [new]}, d, hhmm=hhmm, extra=[[WEEKDAYS[weekday]]]),
                    [kind["title"].split()[0], "every " + WEEKDAYS[weekday], time_words])


# --- Right tool among look-alikes ---

def reminder_in(d: Draw):
    task = d.rng.choice(INGREDIENTS["reminders"]).replace("{person}", d.person(real=True)["name"].split()[0])
    minutes = d.rng.choice([5, 10, 12, 15, 20, 30, 45, 60, 90, 120])
    when = d.now + timedelta(minutes=minutes)
    words = f"{minutes} minutes" if minutes < 60 else {60: "an hour", 90: "an hour and a half", 120: "2 hours"}[minutes]
    key = task.split()[-1].lower()
    hhmm = when.strftime("%H:%M")
    return d.recipe("lookalike", "reminder_in", f"Be reminded to \"{task.lower()}\" in {words} (use 'remind me').",
                    done({"reminders": [{"text": "~" + key, "date": when.date().isoformat(), "time": hhmm}]}, d,
                         extra=[d.answer_time(hhmm) + [words]]),
                    [key, str(minutes) if minutes < 60 else "hour"])


def note_find(d: Draw):
    note = d.rng.choice(d.phone["notes"])["text"]
    if ":" not in note:
        return note_add(d)
    topic = note.split(":", 1)[0].lower()
    return d.recipe("lookalike", "note_find", f"Ask what they saved about \"{topic}\" (a note), without saying 'note'.",
                    nothing([note_key(note)]), [topic.split()[-1]])


def note_add(d: Draw):
    text = d.rng.choice(INGREDIENTS["notes"])["text"].replace("{person}", d.person(real=True)["name"].split()[0])
    key = note_key(text)
    return d.recipe("lookalike", "note_add", f"Save this as a note: \"{text}\".",
                    [{"changes": {"notes": d.phone["notes"] + [{"text": "~" + key}]}}], [key])


def media(d: Draw):
    m = d.phone["media"]
    action = d.rng.choice(["next", "previous", "pause" if m["playing"] else "play"])
    words = {"next": "skip this song", "previous": "go back to the previous song", "pause": "pause the music", "play": "resume the music"}
    after = copy.deepcopy(m)
    if action in ("play", "pause"):
        after["playing"] = action == "play"
    else:
        after["track"] = max(1, m["track"] + (1 if action == "next" else -1))
    return d.recipe("lookalike", "media", f"{words[action].capitalize()}.", [{"changes": {"media": after}}], [])


def volume(d: Draw):
    v = d.phone["media"]["volume"]
    if d.rng.random() < 0.5:
        up = d.rng.random() < 0.5
        target = max(0, min(100, v + (10 if up else -10)))
        task, say = f"Turn the volume {'up' if up else 'down'} a bit (current volume {v}%, don't mention it).", []
    else:
        target = d.rng.choice([20, 30, 40, 60, 70, 80, 100])
        task, say = f"Set the volume to {target}%.", [str(target)]
    return d.recipe("lookalike", "volume", task, [{"changes": {"media": d.phone["media"] | {"volume": target}}}], say)


def phone_controls(d: Draw):
    choice = d.rng.choice(["flashlight", "settings", "navigate", "music", "app"])
    if choice == "flashlight":
        return d.recipe("lookalike", "flashlight", "Turn on the flashlight (or torch).", [{"changes": {"flashlight": True}}], [])
    if choice == "settings":
        panel = d.rng.choice(["wifi", "bluetooth", "dnd", "airplane", "location", "battery", "display", "sound"])
        return d.recipe("lookalike", "settings", f"Open the {panel} settings (do-not-disturb if dnd).",
                        [{"changes": {"opened": [{"tool": "open_settings", "panel": panel}]}}], [])
    if choice == "navigate":
        place = d.rng.choice(INGREDIENTS["places"])
        key = place.split()[-1].strip("'s")
        return d.recipe("lookalike", "navigate", f"Get directions to {place}.",
                        [{"changes": {"opened": [{"tool": "navigate", "destination": "~" + key}]}}], [key])
    if choice == "music":
        query = d.rng.choice(INGREDIENTS["music"])
        return d.recipe("lookalike", "music", f"Play {query}.",
                        [{"changes": {"opened": [{"tool": "play_music", "query": "~" + query.split()[-1]}]}}], [query.split()[-1]])
    app = d.rng.choice(d.phone["apps"])
    return d.recipe("lookalike", "open_app", f"Open {app}.", [{"changes": {"opened": [{"tool": "open_app", "name": "~" + app}]}}], [app])


def date_math(d: Draw):
    day, _ = d.day(max_offset=200, allow_today=False)
    days = (day - d.now.date()).days
    if d.rng.random() < 0.5:
        return d.recipe("lookalike", "days_until", f"Ask how many days until {day.day} {day.strftime('%B')}.",
                        nothing([str(days)]), [str(day.day), day.strftime("%B")])
    total, people = d.rng.choice([(84.6, 3), (120, 4), (57.5, 2), (230, 5), (99.9, 3), (45, 6)])
    each = f"{total / people:.2f}".rstrip("0").rstrip(".")
    return d.recipe("lookalike", "split", f"Ask to split {total} between {people} people.", nothing([each]),
                    [str(total), str(people)])


# A day in the next 6 with two or more one-off events (so its weekday name can't mean today), or None.
def busy_day(d: Draw):
    counts = {}
    for e in d.future_events(6):
        counts[e["date"]] = counts.get(e["date"], 0) + 1
    days = sorted(day for day, n in counts.items() if n >= 2)
    return date.fromisoformat(d.rng.choice(days)) if days else None


# --- Follow-ups (two messages) ---

def timer_change(d: Draw):
    a, b = d.rng.sample([5, 10, 15, 20, 25, 30, 40], 2)
    return d.recipe("followup", "timer_change", f"A timer for {a} minutes.",
                    [{"changes": {"timers": [a * 60, b * 60]}, "answer_has": ["clock"]}], [str(a), str(b)],
                    followup=f"Change it to {b} minutes, very short (\"make it {b}\").")


def event_change(d: Draw):
    kind = d.rng.choice([k for k in INGREDIENTS["events"] if "{" not in k["title"]])
    day, day_words = d.day(allow_today=False)
    first, first_words = d.time(tuple(kind["hours"]))
    second, second_words = d.time(tuple(kind["hours"]))
    if second == first:
        return event_add(d, "followup")
    new = {"title": "~" + kind["title"], "date": day.isoformat(), "start": second}
    return d.recipe("followup", "event_change", f"Add \"{kind['title']}\" on {day_words} at {first_words}.",
                    done({"events": d.phone["events"] + [new]}, d, hhmm=second),
                    [kind["title"].split()[0], day_words, first_words, second_words],
                    followup=f"Change its time to {second_words}, very short (\"actually make it {second_words}\").")


def pick_contact(d: Draw):
    a = d.person(real=True)
    b = contact(d.rng, "uk", first=a["name"].split()[0])
    if b["name"] == a["name"]:
        return call_one(d)
    d.phone["contacts"].append(b)
    target = d.rng.choice([a, b])
    first, last = target["name"].split(maxsplit=1)
    return d.recipe("followup", "pick_contact", f"Call {first} (two contacts share that first name).",
                    [{"changes": {"opened": [{"tool": "call", "number": digits(target["phone"])}]}}], [first, last],
                    followup=f"When asked which one, answer with just \"{last}\".")


def list_then_act(d: Draw):
    day = busy_day(d)
    if not day:
        return event_move(d)
    on_day = [e for e in d.phone["events"] if occurs(e, day)]
    if any(not e["start"] for e in on_day):
        return event_move(d)
    first = min(on_day, key=lambda e: e["start"])
    words = "tomorrow" if (day - d.now.date()).days == 1 else WEEKDAYS[day.weekday()]
    shift = d.rng.choice([30, 60, 120])
    moved_at = datetime.fromisoformat(f"{first['date']}T{first['start']}") + timedelta(minutes=shift)
    new_start = moved_at.strftime("%H:%M")
    moved = [x | ({"date": moved_at.date().isoformat(), "start": new_start} if x["id"] == first["id"] else {})
             for x in d.phone["events"]]
    later = {30: "half an hour later", 60: "an hour later", 120: "two hours later"}[shift]
    return d.recipe("followup", "list_then_act", f"Ask what's on the calendar {words}.",
                    done({"events": moved}, d, hhmm=new_start), [words, "first"],
                    followup=f"Move the first one {later} (\"move the first one {later}\").")


# --- No tool ---

def no_tool(d: Draw):
    kind = d.rng.random()
    if kind < 0.45:
        return d.recipe("notool", "chat", f"{d.rng.choice(CHAT).capitalize()}. No action wanted.", nothing(), [])
    if kind < 0.75:
        return d.recipe("notool", "negation", f"Tell the assistant {d.rng.choice(NEGATIONS)}.", nothing(), [])
    c = d.person()
    words = name_words(d, c)
    return d.recipe("notool", "later", f"Say they should {d.rng.choice(['call', 'text', 'email'])} {words} "
                    f"later, without asking for anything.", nothing(), [words, "later"])


# --- Ask or confirm only when needed ---

def missing_info(d: Draw):
    title = d.rng.choice([k["title"] for k in INGREDIENTS["events"] if "{" not in k["title"]])
    what, words = d.rng.choice([("an alarm, with no time", "alarm"), (f"\"{title}\" tomorrow, with no time", "tomorrow"),
                                ("a reminder, with no time or day", "remind"), ("a text to someone, with no recipient", "text"),
                                ("a timer, with no duration", "timer")])
    return d.recipe("ask", "missing", f"Ask for {what}. Leave that information out.",
                    nothing(["?"], answer_not=["done", "i've set", "i have set", "added", "started", "is ready"]), [words])


def two_matches_ask(d: Draw):
    r = pick_contact(d)
    if r["kind"] != "pick_contact":
        return r
    first = r["say"][0]
    names = [c["name"].split(maxsplit=1)[1] for c in d.phone["contacts"] if c["name"].split()[0] == first]
    return r | {"category": "ask", "kind": "two_matches", "followup": None, "say": [first],
                "expect": nothing(["?"] + names[:2])}


def clear_day(d: Draw):
    day = busy_day(d)
    if not day:
        return missing_info(d)
    gone = {e["id"] for e in d.phone["events"] if occurs(e, day)}
    words = "tomorrow" if (day - d.now.date()).days == 1 else WEEKDAYS[day.weekday()]
    return d.recipe("ask", "clear_day", f"Clear the whole calendar for {words}.",
                    [{"changes": {"events": [e for e in d.phone["events"] if e["id"] not in gone]}}], [words])


# --- One call then stop, clean arguments ---

def app_missing(d: Draw):
    missing = [a for a in INGREDIENTS["apps"] if a not in d.phone["apps"]]
    if not missing:
        return event_add(d, "stopclean")
    app = d.rng.choice(missing)
    return d.recipe("stopclean", "app_missing", f"Open {app} (it isn't installed).",
                    nothing([app], [NOT_INSTALLED], ["opened", "opening"]), [app, "open"])


CATEGORIES = {
    "lookup": (0.20, [call_one, call_one, message_one, message_one, email_one, no_email, contact_missing, event_delete, event_move]),
    "dates": (0.15, [event_add, event_add, reminder_add, alarm_add, timer_units, event_repeat]),
    "lookalike": (0.15, [reminder_in, note_find, note_add, media, volume, phone_controls, phone_controls, date_math]),
    "followup": (0.12, [timer_change, event_change, pick_contact, list_then_act]),
    "notool": (0.10, [no_tool]),
    "ask": (0.10, [missing_info, missing_info, two_matches_ask, clear_day]),
    "stopclean": (0.10, [lambda d: event_add(d, "stopclean"), message_one, app_missing]),
    "messy": (0.08, [call_one, message_one, event_add, alarm_add, reminder_in, note_find, media, timer_units]),
}


# Recipe number `seed`: picks a category by weight, a recipe in it, and a style ("messy" only uses messy styles).
# A recipe that doesn't fit the phone falls back to another kind; then a new phone is drawn, so shares stay right.
def draw(seed: int) -> dict:
    names = list(CATEGORIES)
    category = random.Random(seed).choices(names, weights=[CATEGORIES[n][0] for n in names])[0]
    for attempt in range(50):
        d = Draw(seed * 1000 + attempt)
        recipe = d.rng.choice(CATEGORIES[category][1])(d)
        if category in ("messy", "stopclean") or recipe["category"] == category:
            break
    recipe["category"] = category
    if category == "messy":
        recipe["style"] = d.rng.choice(MESSY)
    else:
        styles = ["plain", "plain", "casual", "casual", "terse", "polite", "lowercase"]
        recipe["style"] = d.rng.choice(styles + (["indirect"] if recipe["kind"] in ("reminder_add", "reminder_in") else []))
    recipe["seed"] = seed
    return recipe


if __name__ == "__main__":
    import json
    for seed in range(12):
        r = draw(seed)
        print(f"{r['category']:<10} {r['kind']:<16} {r['style']:<9} {r['now']}  {r['task']}" + (f"  THEN: {r['followup']}" if r["followup"] else ""))
        print("    say:", r["say"], " answer:", [{k: v for k, v in o.items() if k != "changes"} for o in r["expect"]])
