# Recipes for generated cases: code picks what the user wants and computes the right outcome; the teacher only
# writes the words. A recipe = phone + now + a task for the message writer + the words the message must contain
# + the expected outcome (eval format, plus answer patterns). Categories and weights follow AGENT-design.md.

import copy
import random
import re
from datetime import date, datetime, time, timedelta

from gen.phones import INGREDIENTS, contact, is_person, random_now, random_phone
from app.sim import fold, occurs

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
TEXTS = ["running late", "on my way", "call me when you can", "happy birthday!", "dinner at 8?", "see you soon",
         "can you pick up milk?", "thanks for today", "I'm outside", "the meeting moved to 3", "landed safely",
         "don't forget the keys", "I'll be there in ten", "can't talk now, will call later", "did you get my email?",
         "happy anniversary!", "where are you?", "I'm at the entrance", "traffic is bad", "I missed the bus",
         "good luck today!", "congratulations!", "let's meet at noon", "I'll bring dessert", "sorry, I can't make it",
         "I've got the tickets", "see you at the station", "the keys are under the mat", "call me back please",
         "thanks, that helped a lot", "I'm leaving now", "just landed", "happy new year!", "get well soon",
         "sounds good", "I'll be home late", "dinner is ready", "please bring the charger", "are we still on for tonight?",
         "I forgot my wallet", "I'm in the lobby", "can you send me the address?", "I'm stuck in the lift",
         "the kids are asleep", "lunch is on me", "don't wait for me", "all good here", "no worries, take your time"]
CHAT = ["thank the assistant", "say good morning", "say the last thing worked, nice", "ask how it is doing",
        "ask a general knowledge question (science, geography, language)", "ask for a word that means happy",
        "ask for a quick cooking tip", "ask what a word means", "ask how far the moon is",
        "ask for the capital of a country", "ask for a synonym of tired", "ask for a joke", "ask how to boil an egg",
        "ask who wrote a famous book", "ask what a haiku is", "ask for a fun fact about an animal",
        "ask what the speed of light is", "ask what photosynthesis is", "ask how to say thank you in another language",
        "ask how to convert Celsius to Fahrenheit", "ask for a short motivational quote", "ask how long to roast a chicken",
        "say they feel tired", "say they had a good day", "say goodbye", "say hello", "ask for a good name for a cat",
        "ask for a word that rhymes with another", "ask in which year a famous event happened",
        "ask how many days are in a leap year", "ask how to unclog a drain", "ask for a book recommendation"]
NEGATIONS = ["not to add anything, they were just thinking aloud",
             "not to call anyone yet", "not to set a timer, they changed their mind", "to forget the last idea",
             "to ignore the last message", "never mind, they'll do it themselves", "not to message anyone, they changed their mind",
             "not to add the event, it was cancelled", "not to remind them, they'll remember", "not to open any app",
             "to stop, they don't need anything now", "not to call the dentist, they're going in person",
             "not to play any music, they want quiet", "not to turn on the flashlight, it's fine", "not to book anything yet",
             "not to delete the event after all", "not to send the message, they'll do it later",
             "not to set a timer, the food is already done", "not to move the meeting, they decided to keep it",
             "to forget it, it was a mistake"]
CAN = ["send emails", "set alarms", "make calls", "read my calendar", "play music", "set timers", "work offline",
       "send WhatsApp messages", "find my contacts", "take notes", "do maths", "open apps"]
# Requests no tool can do, and a word the message must contain.
CANT = [("whether it will rain this weekend", "rain|weather"), ("for the latest news headlines", "news"),
        ("for last night's football score", "score|football"), ("for a big company's stock price", "stock|price|shares"),
        ("whether the pharmacy is open right now", "pharmacy"), ("how bad the traffic is on the way to work", "traffic"),
        ("whether a flight is on time", "flight"), ("where their package is", "package|parcel|delivery|order"),
        ("for their bank balance", "bank|balance"), ("what's on TV tonight", "tv|television"), ("{money}", "")]
# Currencies of the regions (one and many), for "{money}" above.
CURRENCIES = {"euro": "euros", "dollar": "dollars", "pound": "pounds", "rupee": "rupees", "yuan": "yuan", "yen": "yen",
              "won": "won", "naira": "naira", "dirham": "dirhams", "peso": "pesos", "real": "reais", "zloty": "zloty",
              "ruble": "rubles", "rupiah": "rupiah", "taka": "taka", "lira": "lira", "rand": "rand"}
CANT_SAY = ["can't", "cannot", "not able", "unable", "don't have", "no access", "offline", "no internet"]
MESSY = ["typos", "slang", "terse", "lowercase", "run-on"]
# How the user words the message, beyond the style. One is drawn per recipe (none about half of the time).
SHAPES = {"short": "very short: two to five words", "sentence": "one full sentence", "two": "two short sentences",
          "value first": "start with the day, the time or the name, then the action",
          "preface": 'start with a filler or a greeting ("hey", "ok so", "quick one")',
          "reason": "add a short reason or context (running late, in a meeting...)",
          "thanks": 'end with "thanks" or "please"'}
NOT_FOUND = ["can't", "cannot", "couldn't", "could not", "no contact", "not in your", "don't have", "isn't", "not find"]
NOT_INSTALLED = ["not installed", "isn't installed", "can't find", "couldn't find", "don't see", "no app", "not on your",
                 "not available", "isn't available", "unavailable", "don't have"]
CALL = "call|calling|phone|dial"  # a call request says one of these ("ring" is kept for alarms)
NO_NOTE = NOT_FOUND + ["no note", "nothing", "no saved"]
# Whole-day events: (title, the word its title must keep, what the message must say, may repeat yearly).
ALL_DAY = [("{person}'s birthday", "{person}", ["{person}", "birthday|bday"], True), ("Day off", "off", ["off"], False),
           ("Holiday", "holiday", ["holiday"], False), ("Bank holiday", "holiday", ["bank holiday|public holiday"], False),
           ("Wedding anniversary", "anniversary", ["anniversary"], True)]
# Words for "move all my meetings": an event title word and its plural.
GROUPS = {"meeting": "meetings", "lesson": "lessons", "class": "classes", "appointment": "appointments"}
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

    # A future day, and how the user refers to it ("tomorrow", "friday", "in 3 days", "the 24th", "24/11").
    def day(self, max_offset: int = 20, allow_today: bool = True):
        low = 0 if allow_today and self.now.hour < 20 else 1
        offset = self.rng.randint(low, max_offset)
        day = self.now.date() + timedelta(days=offset)
        if offset == 0:
            words = "today" if self.now.hour < 15 else self.rng.choice(["today", "tonight"])
        elif offset == 1:
            words = "tomorrow"
        elif offset <= 6:
            other = "the day after tomorrow" if offset == 2 else f"in {offset} days"
            words = self.rng.choice([WEEKDAYS[day.weekday()]] * 3 + [other])
        else:
            numeric = f"{day.day}/{day.month}" if self.formats[0] == "day/month" else f"{day.month}/{day.day}"
            words = self.rng.choice([f"the {ordinal(day.day)}", f"{day.day} {day.strftime('%B')}", numeric])
        if words == f"the {ordinal(day.day)}":
            if any((self.now.date() + timedelta(days=i)).day == day.day for i in range(1, offset)):
                words = day.strftime("%-d %B")
        return day, words

    # A time inside `hours`, and how the user says it ("7pm", "19:00", "half past 7 in the evening", or a bare "7" when
    # it can only mean one thing: on a 24-hour clock "8:45" means 08:45, but a whole "at 4" is said on both clocks).
    def time(self, hours: tuple[int, int], day=None):
        low = hours[0]
        if day == self.now.date():
            low = max(low, self.now.hour + 1)
        hour = self.rng.randint(min(low, hours[1]), hours[1])
        minute = self.rng.choice([0, 0, 0, 15, 30, 45])
        hhmm = f"{hour:02d}:{minute:02d}"
        bare = f"{(hour - 1) % 12 + 1}" + (f":{minute:02d}" if minute else "")
        twelve_hour = self.formats[1] == "12-hour"
        if (twelve_hour or not minute) and not hours[0] <= (hour + 12) % 24 <= hours[1] and self.rng.random() < 0.3:
            words = bare
        else:
            words = hhmm if self.formats[1] == "24-hour" and self.rng.random() < 0.6 else twelve(hhmm)
        if self.rng.random() < 0.2:
            words = spoken(hour, minute) or words
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
        return [hhmm, hhmm.removeprefix("0")] if self.formats[1] == "24-hour" else [twelve(hhmm)]

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
    # Its category is set by draw().
    def recipe(self, kind: str, task: str, expect: list, say: list[str], followup: str | None = None):
        return {"kind": kind, "now": self.now.isoformat(timespec="minutes"),
                "formats": list(self.formats), "phone": self.phone, "task": task, "followup": followup,
                "say": [s for s in say if s], "expect": expect}


# "19:30" → "7:30pm", "07:00" → "7am".
def twelve(hhmm: str) -> str:
    hour, minute = map(int, hhmm.split(":"))
    return f"{(hour - 1) % 12 + 1}" + (f":{minute:02d}" if minute else "") + ("am" if hour < 12 else "pm")


# A time as people say it, naming the part of the day: "noon", "10 in the morning", "half past 7 in the evening",
# "quarter to 9 in the morning". None around midnight and midday, where the part of the day would be unclear.
def spoken(hour: int, minute: int) -> str | None:
    if (hour, minute) == (12, 0):
        return "noon"
    named = hour + 1 if minute == 45 else hour  # "quarter to 9" names 9
    part = "morning" if 5 <= named < 12 else "afternoon" if 13 <= named < 18 else "evening" if 18 <= named < 23 else None
    if not part:
        return None
    lead = {0: "", 15: "quarter past ", 30: "half past ", 45: "quarter to "}[minute]
    return f"{lead}{(named - 1) % 12 + 1} in the {part}"


# How the user names a contact: "mum", "the dentist", "Sara", or "Sara Romano" when two share the first name.
def name_words(d: Draw, c: dict) -> str:
    if not is_person(c):
        return c["name"].lower() if c["name"] in INGREDIENTS["family"] else "the " + c["name"].lower()
    first = c["name"].split()[0]
    same = [x for x in d.phone["contacts"] if x["name"].split()[0] == first]
    return c["name"] if len(same) > 1 else d.rng.choice([first, first, c["name"]])


# The word that proves a note was found or saved: the last word of its value ("tulip1952", "B12").
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
    return d.recipe("call_one", f"Call {words} (in contacts as \"{c['name']}\").",
                    [{"changes": {"opened": [{"tool": "call", "number": digits(c["phone"])}]}}], [words, CALL])


def message_one(d: Draw):
    c, text = d.person(), d.rng.choice(TEXTS)
    words = name_words(d, c)
    app = d.rng.choice(["sms"] * 3 + ["whatsapp", "telegram", "signal"])
    via = "" if app == "sms" else f" on {app.capitalize()} (name the app)"
    return d.recipe("message_one", f"Text {words}{via} saying \"{text}\" (those words exactly; any typos go elsewhere).",
                    [{"changes": {"opened": [{"tool": "draft_message", "number": digits(c["phone"]),
                                              "text": {"words": text}, "app": app}]}}],
                    [words, text, "" if app == "sms" else app]) | {"verbatim": text}


def email_one(d: Draw):
    c = d.person(with_email=True)
    if not c["email"]:
        return None
    words = name_words(d, c)
    topic = d.rng.choice(["the report is ready", "the slides are attached", "I'll be late tomorrow", "thanks for the help"])
    return d.recipe("email_one", f'Email {words} saying "{topic}" (keep those words).',
                    [{"changes": {"opened": [{"tool": "draft_email", "to": c["email"], "body": {"words": topic}}]}}], [words, "email", topic]) | {"verbatim": topic}


def no_email(d: Draw):
    c = d.person(with_email=False)
    if c["email"]:
        return None
    words = name_words(d, c)
    return d.recipe("no_email", f"Email {words} a short note (they have no email address saved).",
                    nothing(answer_any=[NO_EMAIL], answer_not=["ready", "sent"]), [words, "email"]) | {"read_has": [c["name"] + ": phone " + c["phone"] + ", email none"]}


def contact_missing(d: Draw):
    names = [fold(c["name"]) for c in d.phone["contacts"]]
    region = d.rng.choice(list(INGREDIENTS["regions"].values()))
    missing = [n for n in region["first"] if all(fold(n) not in name for name in names)]
    if not missing:
        return None
    first = d.rng.choice(missing)
    return d.recipe("contact_missing", f"Call {first} (there is no {first} in the contacts).",
                    nothing([first], [NOT_FOUND], ["calling"]), [first, CALL]) | {"read_has": [f"No contact matches '{first}'"]}


# "What's Sara's number?": look it up and answer with the number (and the full name of a real person); never call.
def contact_info(d: Draw):
    c = d.person()
    words = name_words(d, c)
    number = re.sub(r"\D", "", c["phone"])
    return d.recipe("contact_info", f"Ask for {words}'s phone number (only the number, no call).",
                    nothing([number] + ([c["name"]] if is_person(c) else []), answer_not=["calling"]),
                    [words, "number|phone|digits"]) | {"read_has": [c["phone"]]}


# An event with a title no other event has, so "cancel the yoga" means one thing. None if there's none.
def unique_event(d: Draw):
    events = [e for e in d.future_events() if sum(x["title"] == e["title"] for x in d.phone["events"]) == 1]
    return d.rng.choice(events) if events else None


def event_delete(d: Draw):
    e = unique_event(d)
    if not e:
        return None
    day = datetime.fromisoformat(e["date"]).date()
    offset = (day - d.now.date()).days
    when = "tomorrow" if offset == 1 else WEEKDAYS[day.weekday()] if offset < 7 else ""
    rest = [x for x in d.phone["events"] if x["id"] != e["id"]]
    return d.recipe("event_delete", f"Cancel \"{e['title']}\"" + (f" ({when})" if when else "") + ".",
                    [{"changes": {"events": rest}}], [e["title"], when, "cancel|delete|remove|drop|scrap|clear|axe|nix"])


def event_move(d: Draw):
    e = unique_event(d)
    if not e:
        return None
    new_day, day_words = d.day(allow_today=False)
    hhmm, time_words = d.time((8, 20))
    moved = [x | ({"date": new_day.isoformat(), "start": hhmm} if x["id"] == e["id"] else {}) for x in d.phone["events"]]
    return d.recipe("event_move", f"Move \"{e['title']}\" to {day_words} at {time_words}.",
                    done({"events": moved}, d, new_day, hhmm),
                    [e["title"].split()[0], day_words, time_words, "move|reschedule|push|shift|change|postpone|bump"])


# "What do I have friday?": look the day up and answer with every event on it, repeating ones included.
def agenda(d: Draw):
    day = busy_day(d)
    if not day:
        return None
    titles = [e["title"].split(" with ")[0].split(" to ")[0] for e in d.phone["events"] if occurs(e, day)]
    words = "tomorrow" if (day - d.now.date()).days == 1 else WEEKDAYS[day.weekday()]
    return d.recipe("agenda", f"Ask what's on the calendar {words}.", nothing(titles),
                    [words, "what|whats|anything|agenda|schedule|calendar|plans|busy|have"]) | {"calendar_day": day.isoformat()}


# "When is my dentist?": find it by name and answer its day and time; change nothing.
def event_when(d: Draw):
    e = unique_event(d)
    if not e:
        return None
    return d.recipe("event_when", f"Ask when \"{e['title']}\" is (day and time).",
                    nothing(answer_any=[d.answer_day(date.fromisoformat(e["date"])), d.answer_time(e["start"])]),
                    [e["title"], "when|what time|what day|which day"]) | {"read_has": [f"id {e['id']}:"]}


# "Move all my meetings to next week": every event with that word moves, in one edit_events call (a confirm card).
def move_all(d: Draw):
    word = d.rng.choice(list(GROUPS))
    kinds = [k for k in INGREDIENTS["events"] if word in k["title"].lower()]
    d.phone["events"] = [e for e in d.phone["events"] if word not in e["title"].lower()]  # only the ones drawn here
    first_id = max((e["id"] for e in d.phone["events"]), default=0) + 1
    for i, offset in enumerate(sorted(d.rng.sample(range(1, 7), d.rng.randint(2, 3)))):
        kind = d.rng.choice(kinds)
        hhmm, _ = d.time(tuple(kind["hours"]))
        d.phone["events"].append({"id": first_id + i, "title": event_title(d, kind), "start": hhmm, "repeat": None,
                                  "date": (d.now.date() + timedelta(days=offset)).isoformat(), "duration_minutes": kind["minutes"][0]})
    shift, words = d.rng.choice([(7, "to next week"), (7, "to next week"), (1, "back by a day")])
    moved = [e | {"date": (date.fromisoformat(e["date"]) + timedelta(days=shift)).isoformat()} if word in e["title"].lower() else e
             for e in d.phone["events"]]
    return d.recipe("move_all", f"Move all their {GROUPS[word]} {words}.", [{"changes": {"events": moved}}],
                    ["all|every|everything", word, "week" if shift == 7 else "day", "move|push|shift|reschedule|postpone|bump"])


PARTS = [(5, "morning"), (12, "afternoon"), (18, "evening")]  # (first hour, part of the day)


# "Am I free thursday afternoon? if so book padel at 4": look first, add only if nothing is on at that hour.
# Half the time something is (moved there if needed): then nothing is added and the answer names it.
def if_free(d: Draw):
    kind = d.rng.choice(INGREDIENTS["events"])
    offset = d.rng.randint(0, 6)
    day = d.now.date() + timedelta(days=offset)
    hhmm, time_words = d.time(tuple(kind["hours"]), day)
    first_hour, part = max(p for p in PARTS if p[0] <= int(hhmm[:2]))
    if offset == 0 and d.now.hour >= first_hour:
        return None  # "this afternoon" has already started
    when = (f"this {part}" if offset == 0 else
            f"{d.rng.choice(['tomorrow', WEEKDAYS[day.weekday()]]) if offset == 1 else WEEKDAYS[day.weekday()]} {part}")
    start = datetime.combine(day, time.fromisoformat(hhmm))

    def clashes(e):  # with the hour the new event would take
        if not e["start"] or not occurs(e, day):
            return False
        begins = datetime.combine(day, time.fromisoformat(e["start"]))
        return begins < start + timedelta(minutes=60) and start < begins + timedelta(minutes=e["duration_minutes"])

    clash = next((e for e in d.phone["events"] if clashes(e)), None)
    busy = d.rng.random() < 0.5
    if clash and not busy:
        return None
    if busy and not clash:
        other = d.rng.choice([k for k in INGREDIENTS["events"] if "{" not in k["title"] and k != kind])
        begins = start + timedelta(minutes=d.rng.choice([-30, 0, 30]))
        clash = {"id": max((e["id"] for e in d.phone["events"]), default=0) + 1, "title": other["title"],
                 "date": day.isoformat(), "start": begins.strftime("%H:%M"), "duration_minutes": 60, "repeat": None}
        d.phone["events"].append(clash)
    title = event_title(d, kind)
    key = title.split(" with ")[0].split(" to ")[0]
    task = f"Ask if they're free {when}, and if so add \"{title}\" at {time_words}."
    say = [when.split()[0] if offset else "", part if part != "evening" else "evening|tonight", key, time_words,
           "if|then|whether", "free|busy|available|anything|nothing|plans|clear|open"]
    if busy:
        return d.recipe("if_free", task, nothing([clash["title"].split(" with ")[0].split(" to ")[0]]), say) | {"calendar_day": day.isoformat()}
    new = {"title": "~" + key, "date": day.isoformat(), "start": hhmm, "repeat": None}
    return d.recipe("if_free", task, done({"events": d.phone["events"] + [new]}, d, day, hhmm), say) | {"calendar_day": day.isoformat()}


# --- Dates, times, units ---

# An event's title with its person or city filled in: "Lunch with Sara", "Train to Lyon".
def event_title(d: Draw, kind: dict) -> str:
    return kind["title"].replace("{person}", d.person(real=True)["name"].split()[0]).replace("{city}", d.rng.choice(INGREDIENTS["cities"]))


def event_add(d: Draw):
    kind = d.rng.choice(INGREDIENTS["events"])
    title = event_title(d, kind)
    day, day_words = d.day(allow_today=d.now.hour < kind["hours"][1])  # today only if its hours haven't passed
    hhmm, time_words = d.time(tuple(kind["hours"]), day)
    minutes = d.rng.choice(kind["minutes"] + [None, None])
    length = f", for {minutes} minutes (say it naturally: 'for an hour', '90 min'…)" if minutes else ""
    key = title.split(" with ")[0].split(" to ")[0]
    new = {"title": "~" + key, "date": day.isoformat(), "start": hhmm, "repeat": None}
    if minutes:
        new["duration_minutes"] = minutes
    return d.recipe("event_add", f"Add \"{title}\" on {day_words} at {time_words}{length}.",
                    done({"events": d.phone["events"] + [new]}, d, day, hhmm),
                    [key, day_words, time_words, f"{minutes} minutes" if minutes else ""])


def reminder_add(d: Draw):
    task = d.rng.choice(INGREDIENTS["reminders"]).replace("{person}", d.person(real=True)["name"].split()[0])
    if d.now.hour < 20 and d.rng.random() < 0.25:
        day, day_words = d.now.date(), ""  # "remind me at 5 to call the bank" means today
    else:
        day, day_words = d.day()
    hhmm, time_words = d.time((7, 21), day)
    key = task.split()[-1].lower()
    when = f"{day_words} at {time_words}".strip() + ("" if day_words else " (no day: it's today)")
    return d.recipe("reminder_add", f"Be reminded to \"{task.lower()}\" {when}.",
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
        if hhmm <= d.now.strftime("%H:%M"):  # an alarm has no date: "tomorrow" only when that time has passed today
            task += d.rng.choice(["", " tomorrow"])
    return d.recipe("alarm_add", task + ".",
                    done({"alarms": d.phone["alarms"] + [{"time": hhmm, "days": days}]}, d, hhmm=hhmm),
                    [time_words, "every " + every if every else ""])


# A length of time up to 5 hours and how the user says it: "45 seconds", "12 minutes", "an hour and 20 minutes"...
def duration(d: Draw) -> tuple[int, str]:
    h, m, s = d.rng.randint(1, 5), d.rng.randint(2, 59), d.rng.randrange(10, 60, 5)
    hours = "an hour" if h == 1 else f"{h} hours"
    return d.rng.choice([(s, f"{s} seconds"), (m * 60, f"{m} minutes"), (h * 3600, hours),
                         (h * 3600 + m * 60, f"{hours} and {m} minutes"), (60 + s, f"a minute and {s} seconds"),
                         (1800, "half an hour"), (900, "a quarter of an hour"), (5400, "an hour and a half")])


def timer_units(d: Draw):
    seconds, words = duration(d)
    return d.recipe("timer_units", f"A timer for {words} (say 'timer' or 'countdown').",
                    [{"changes": {"timers": [seconds]}}], [words])


# A repeating event: mostly weekly, also daily, weekdays, monthly (yearly comes with birthdays). It starts on the first
# day it happens; when that's today and the time is still ahead, the next one is fine too.
def event_repeat(d: Draw):
    kind = d.rng.choice([k for k in INGREDIENTS["events"] if "{" not in k["title"]])
    hhmm, time_words = d.time(tuple(kind["hours"]))
    repeat = d.rng.choice(["weekly"] * 3 + ["daily", "weekdays", "monthly"])
    weekday, dd = d.rng.randint(0, 6), d.rng.randint(1, 28)
    fits, words, say, extra = {  # which days it happens, how the task says it, what the message and answer must say
        "weekly": (lambda day: day.weekday() == weekday, f"every {WEEKDAYS[weekday]}", ["every " + WEEKDAYS[weekday]], [[WEEKDAYS[weekday]]]),
        "daily": (lambda day: True, "every day", ["every day|daily"], [["every day", "daily", "each day"]]),
        "weekdays": (lambda day: day.weekday() < 5, "every weekday", ["weekday|weekdays"], [["weekday", "monday to friday"]]),
        "monthly": (lambda day: day.day == dd, f"on the {ordinal(dd)} of every month", [ordinal(dd), "month|monthly"],
                    [[ordinal(dd)], ["month"]]),
    }[repeat]
    ahead = hhmm > d.now.strftime("%H:%M")
    starts = [day for day in (d.now.date() + timedelta(days=i) for i in range(62)) if fits(day) and (day > d.now.date() or ahead)]
    starts = starts[:2] if starts[0] == d.now.date() else starts[:1]
    new = {"title": "~" + kind["title"], "start": hhmm, "repeat": repeat}
    return d.recipe("event_repeat", f"Add \"{kind['title']}\" {words} at {time_words}, starting with the next one.",
                    [o for day in starts for o in done({"events": d.phone["events"] + [new | {"date": day.isoformat()}]}, d,
                                                        hhmm=hhmm, extra=extra)],
                    [kind["title"].split()[0], *say, time_words])


# Something that lasts all day (birthday, holiday, day off): an event with no start. A birthday may repeat yearly.
def all_day(d: Draw):
    title, key, say, yearly = d.rng.choice(ALL_DAY)
    first = d.person(real=True)["name"].split()[0]
    title, key, say = title.replace("{person}", first), key.replace("{person}", first), [w.replace("{person}", first) for w in say]
    day, day_words = d.day(max_offset=60, allow_today=False)
    new = {"title": "~" + key, "date": day.isoformat(), "start": None, "repeat": None}
    return d.recipe("all_day", f"Add \"{title}\" {day_words} (all day, no time).",
                    [o for repeat in ([None, "yearly"] if yearly else [None])
                     for o in done({"events": d.phone["events"] + [new | {"repeat": repeat}]}, d, day)],
                    say + [day_words])


# Two requests in one message ("dentist friday at 3 and wake me at 7"): both done, one answer naming both.
# Only tools that act silently, so no confirm card or other app stops the second call.
def two_actions(d: Draw):
    a, b = (make(d) for make in d.rng.sample([event_add, reminder_add, alarm_add, timer_units], 2))
    first, second = a["expect"][0], b["expect"][0]
    return d.recipe("two_actions", f"Two requests in one message. First: {a['task']} Second: {b['task']}",
                    [{"changes": first["changes"] | second["changes"],
                      "answer_any": first.get("answer_any", []) + second.get("answer_any", [])}], a["say"] + b["say"])


# --- Right tool among look-alikes ---

def reminder_in(d: Draw):
    task = d.rng.choice(INGREDIENTS["reminders"]).replace("{person}", d.person(real=True)["name"].split()[0])
    minutes = d.rng.choice([5, 10, 12, 15, 20, 30, 45, 60, 90, 120])
    when = d.now + timedelta(minutes=minutes)
    words = f"{minutes} minutes" if minutes < 60 else {60: "an hour", 90: "an hour and a half", 120: "2 hours"}[minutes]
    key = task.split()[-1].lower()
    hhmm = when.strftime("%H:%M")
    return d.recipe("reminder_in", f"Be reminded to \"{task.lower()}\" in {words} (use 'remind me').",
                    done({"reminders": [{"text": "~" + key, "date": when.date().isoformat(), "time": hhmm}]}, d,
                         extra=[d.answer_time(hhmm) + [words]]),
                    [key, f"{minutes} minutes", "remind"])


def note_find(d: Draw):
    note = d.rng.choice(d.phone["notes"])["text"]
    if ":" not in note:
        return None
    topic = note.split(":", 1)[0].lower()
    return d.recipe("note_find", f"Ask what they saved about \"{topic}\" (a note), without saying 'note'.",
                    nothing([note_key(note)]),
                    [topic.split()[-1], "what|whats|where|which|find|search|look|show|saved|wrote|again|tell|check"]) | {"read_has": [note]}


# "What was the gate code?" when no note has it: search, then say nothing was found.
def note_missing(d: Draw):
    saved = " ".join(fold(n["text"]) for n in d.phone["notes"])
    topics = [n["text"].split(":", 1)[0] for n in INGREDIENTS["notes"] if ":" in n["text"] and "{" not in n["text"]]
    topics = [t for t in topics if not any(w in saved for w in fold(t).split() if len(w) > 3)]
    if not topics:
        return None
    topic = d.rng.choice(topics).lower()
    return d.recipe("note_missing", f"Ask what they saved about \"{topic}\" (a note), without saying 'note'. There is none.",
                    nothing(answer_any=[NO_NOTE]),
                    [topic.split()[-1], "what|whats|where|which|find|search|look|show|saved|wrote|again|tell|check"]) | {"note_topic": topic, "read_has": ["No notes match."]}


def note_add(d: Draw):
    text = d.rng.choice(INGREDIENTS["notes"])["text"].replace("{person}", d.person(real=True)["name"].split()[0])
    key = note_key(text)
    return d.recipe("note_add", f"Save this as a note: \"{text}\".",
                    [{"changes": {"notes": d.phone["notes"] + [{"text": "~" + key}]}}], [key])


def media(d: Draw):
    m = d.phone["media"]
    action = d.rng.choice(["next", "previous", "pause" if m["playing"] else "play"])
    words = {"next": "skip this song", "previous": "go back to the previous song", "pause": "pause the music", "play": "resume the music"}
    verbs = {"next": "skip|next", "previous": "previous|back|last|before", "pause": "pause|stop",
             "play": "resume|play|unpause|continue|start"}
    after = copy.deepcopy(m)
    if action in ("play", "pause"):
        after["playing"] = action == "play"
    else:
        after["track"] = max(1, m["track"] + (1 if action == "next" else -1))
    return d.recipe("media", f"{words[action].capitalize()}.", [{"changes": {"media": after}}], [verbs[action]])


def volume(d: Draw):
    v = d.phone["media"]["volume"]
    if d.rng.random() < 0.5:
        up = d.rng.random() < 0.5
        target = max(0, min(100, v + (10 if up else -10)))
        task = f"Turn the volume {'up' if up else 'down'} a bit (current volume {v}%, don't mention it)."
        say = ["up|louder|raise|increase|higher|boost" if up else "down|quieter|lower|softer|reduce|decrease"]
    else:
        target = d.rng.choice([20, 30, 40, 60, 70, 80, 100])
        task, say = f"Set the volume to {target}%.", [str(target)]
    return d.recipe("volume", task, [{"changes": {"media": d.phone["media"] | {"volume": target}}}], say)


def phone_controls(d: Draw):
    choice = d.rng.choice(["flashlight", "settings", "navigate", "music", "app"])
    if choice == "flashlight":
        on, d.phone["flashlight"] = d.rng.random() < 0.5, d.rng.random() < 0.5
        already = [["already"]] if d.phone["flashlight"] == on else []  # nothing changes: the answer says so
        return d.recipe("flashlight", f"Turn {'on' if on else 'off'} the flashlight (or torch).",
                        [{"changes": {"flashlight": on}, "answer_any": already}],
                        ["flashlight|torch|light"] + ([] if on else ["off"]))
    if choice == "settings":
        panels = {"wifi": "wifi|wi-fi|wireless", "bluetooth": "bluetooth", "dnd": "dnd|disturb",
                  "airplane": "airplane|aeroplane|flight|plane", "location": "location|gps", "battery": "battery",
                  "display": "display|screen|brightness", "sound": "sound|sounds|ringtone"}
        panel = d.rng.choice(list(panels))
        return d.recipe("settings", f"Open the {panel} settings (do-not-disturb if dnd).",
                        [{"changes": {"opened": [{"tool": "open_settings", "panel": panel}]}}], [panels[panel]])
    if choice == "navigate":
        place = d.rng.choice(INGREDIENTS["places"])
        key = place.split()[-1].removesuffix("'s")
        return d.recipe("navigate", f"Get directions to {place}.",
                        [{"changes": {"opened": [{"tool": "navigate", "destination": "~" + key}]}}], [key])
    if choice == "music":
        query = d.rng.choice(INGREDIENTS["music"])
        return d.recipe("music", f"Play {query}.",
                        [{"changes": {"opened": [{"tool": "play_music", "query": "~" + query.split()[-1]}]}}], [query.split()[-1]])
    app = d.rng.choice(d.phone["apps"])
    return d.recipe("open_app", f"Open {app}.", [{"changes": {"opened": [{"tool": "open_app", "name": "~" + app}]}}], [app])


def date_math(d: Draw):
    day, _ = d.day(max_offset=200, allow_today=False)
    days = (day - d.now.date()).days
    if d.rng.random() < 0.5:
        return d.recipe("days_until", f"Ask how many days until {day.day} {day.strftime('%B')}.",
                        nothing([str(days)]), [str(day.day), day.strftime("%B")])
    # a whole bill or one with cents ("84.65"; no trailing zero, so "84.6" can't be the same bill)
    cents, people = d.rng.choice([0, d.rng.choice([c for c in range(1, 100) if c % 10])]), d.rng.randint(2, 8)
    total = d.rng.randint(10, 300) + cents / 100
    text = f"{total:.2f}" if cents else str(int(total))
    if cents and d.rng.random() < 0.3:  # much of the world writes "84,65"
        text = text.replace(".", ",")
    each = f"{total / people:.2f}".rstrip("0").rstrip(".")
    return d.recipe("split", f"Ask to split {text} between {people} people.",
                    nothing(answer_any=[[each, each.replace(".", ",")]]), [text, str(people)])


# A day in the next 6 with two or more one-off events (so its weekday name can't mean today), or None.
def busy_day(d: Draw):
    counts = {}
    for e in d.future_events(6):
        counts[e["date"]] = counts.get(e["date"], 0) + 1
    days = sorted(day for day, n in counts.items() if n >= 2)
    return date.fromisoformat(d.rng.choice(days)) if days else None


# --- Follow-ups (two messages) ---

def timer_change(d: Draw):
    a, b = d.rng.sample(range(2, 61), 2)
    return d.recipe("timer_change", f"A timer for {a} minutes.",
                    [{"changes": {"timers": [a * 60, b * 60]}, "answer_has": ["clock"]}], [str(a), str(b)],
                    followup=f"Change it to {b} minutes, very short (\"make it {b}\").") | {
                        "first_turn": {"kind": "timer_units", "say": [f"{a} minutes", "timer"], "expect": [
                            {"changes": {"timers": [a * 60]}, "answer_has": [str(a)]}]},
                        "follow_say": [str(b)]}


def event_change(d: Draw):
    kind = d.rng.choice([k for k in INGREDIENTS["events"] if "{" not in k["title"]])
    day, day_words = d.day(allow_today=False)
    first, first_words = d.time(tuple(kind["hours"]))
    second, second_words = d.time(tuple(kind["hours"]))
    if second == first:
        return None
    new = {"title": "~" + kind["title"], "date": day.isoformat(), "start": second, "repeat": None}
    return d.recipe("event_change", f"Add \"{kind['title']}\" on {day_words} at {first_words}.",
                    done({"events": d.phone["events"] + [new]}, d, hhmm=second),
                    [kind["title"].split()[0], day_words, first_words, second_words],
                    followup=f"Change its time to {second_words}, very short (\"actually make it {second_words}\").") | {
                        "first_turn": {"kind": "event_add", "say": [kind["title"].split()[0], day_words, first_words],
                                       "expect": done({"events": d.phone["events"] + [new | {"start": first}]}, d, day, first)},
                        "follow_say": [second_words]}


def pick_contact(d: Draw):
    a = d.person(real=True)
    b = contact(d.rng, "uk", first=a["name"].split()[0])
    if b["name"] == a["name"]:
        return None
    d.phone["contacts"].append(b)
    target = d.rng.choice([a, b])
    first, last = target["name"].split(maxsplit=1)
    return d.recipe("pick_contact", f"Call {first} (two contacts share that first name).",
                    [{"changes": {"opened": [{"tool": "call", "number": digits(target["phone"])}]}}], [first, last],
                    followup=f"When asked which one, answer with just \"{last}\".") | {
                        "first_turn": {"kind": "two_matches", "say": [first, CALL], "expect": nothing(["?", a["name"], b["name"]]),
                                       "read_has": [a["name"], b["name"]]}, "follow_say": [last]}


def list_then_act(d: Draw):
    day = busy_day(d)
    if not day:
        return None
    on_day = [e for e in d.phone["events"] if occurs(e, day)]
    if any(not e["start"] or e["repeat"] for e in on_day):  # moving one occurrence isn't supported
        return None
    first = min(on_day, key=lambda e: e["start"])
    words = "tomorrow" if (day - d.now.date()).days == 1 else WEEKDAYS[day.weekday()]
    shift = d.rng.choice([30, 60, 120])
    moved_at = datetime.fromisoformat(f"{first['date']}T{first['start']}") + timedelta(minutes=shift)
    new_start = moved_at.strftime("%H:%M")
    moved = [x | ({"date": moved_at.date().isoformat(), "start": new_start} if x["id"] == first["id"] else {})
             for x in d.phone["events"]]
    later = {30: "half an hour later", 60: "an hour later", 120: "two hours later"}[shift]
    return d.recipe("list_then_act", f"Ask what's on the calendar {words}.",
                    done({"events": moved}, d, hhmm=new_start), [words, "first", f"{shift} minutes"],
                    followup=f"Move the first one {later} (\"move the first one {later}\").") | {
                        "first_turn": {"kind": "agenda", "say": [words], "expect": nothing([e["title"] for e in on_day]),
                                       "calendar_day": day.isoformat()}, "follow_say": ["first", f"{shift} minutes"]}


# --- No tool ---

def no_tool(d: Draw):
    kind = d.rng.random()
    if kind < 0.45:
        return d.recipe("chat", f"{d.rng.choice(CHAT).capitalize()}. No action wanted.", nothing(), [])
    if kind < 0.75:
        return d.recipe("negation", f"Tell the assistant {d.rng.choice(NEGATIONS)}.", nothing(), [])
    c = d.person()
    words = name_words(d, c)
    return d.recipe("later", f"Say they should {d.rng.choice(['call', 'text', 'email'])} {words} "
                    f"later, without asking for anything.", nothing(), [words, "later"])


# A remark that mentions something on the phone but asks for nothing: a past event, a past call, a past wake-up,
# or a question about what the assistant can do. Tool-like words, no tool.
def mention(d: Draw):
    past = [e for e in d.phone["events"] if not e["repeat"] and date.fromisoformat(e["date"]) < d.now.date()]
    c = d.person()
    words = name_words(d, c)
    hhmm, time_words = d.time((5, 9))
    forms = [(f"Remark that they {d.rng.choice(['called', 'texted', 'emailed'])} {words} yesterday", [words, "yesterday"]),
             (f"Complain that they had to get up at {time_words} this morning", [time_words]),
             (f"Ask whether the assistant can {d.rng.choice(CAN)}", [])]
    if past:
        e = d.rng.choice(past)
        forms.append((f"Remark how \"{e['title']}\" went (it's over)", [e["title"].split()[0]]))
    task, say = d.rng.choice(forms)
    return d.recipe("mention", task + ", asking for nothing to be done.", nothing(), say)


# A request no tool can do (weather, news, bank...): no call, and the answer says it can't.
def cant(d: Draw):
    task, key = d.rng.choice(CANT)
    say = [key]
    if task == "{money}":  # any two currencies; with an amount, calculate would need a rate nobody gave it
        (a, a_many), (b, b_many) = d.rng.sample(list(CURRENCIES.items()), 2)
        amount = d.rng.choice([10, 50, 100, 250, 1000])
        task = d.rng.choice([f"for today's {a} to {b} exchange rate", f"how much {amount} {a_many} is in {b_many} today"])
        say = [f"{a}|{a_many}", f"{b}|{b_many}"]
    d.phone["apps"] = [a for a in d.phone["apps"] if a != "Weather"]  # no app could answer it either
    return d.recipe("cant", f"Ask {task}.", nothing(answer_any=[CANT_SAY]), say)


# --- Ask or confirm only when needed ---

def missing_info(d: Draw):
    title = d.rng.choice([k["title"] for k in INGREDIENTS["events"] if "{" not in k["title"]])
    # what is asked for, a word the message must contain, and how the question asks for what's missing
    what, words, asks = d.rng.choice([
        ("an alarm, with no time", "alarm", "time|when"), (f"\"{title}\" tomorrow, with no time", "tomorrow", "time|when"),
        ("a reminder, with no time or day", "remind", "time|when|day"),
        ("a text to someone, with no recipient", "text", "who|whom|contact|recipient|number"),
        ("a timer, with no duration", "timer", "how long|how many|duration|minutes")])
    return d.recipe("missing", f"Ask for {what}. Leave that information out.",
                    nothing(["?"], [asks.split("|")], ["done", "i've set", "i have set", "added", "started", "is ready"]), [words])


def two_matches_ask(d: Draw):
    r = pick_contact(d)
    if r is None:
        return None
    first = r.pop("first_turn")
    r.pop("follow_say")
    return r | first | {"followup": None}


# "Cancel the yoga" when two events have that title: ask which one, naming both days.
def two_events_ask(d: Draw):
    e = unique_event(d)
    if not e:
        return None
    first = date.fromisoformat(e["date"])
    other, _ = d.day(max_offset=14, allow_today=False)
    if other == first:
        return None
    hhmm, _ = d.time((8, 20))
    d.phone["events"].append(e | {"id": max(x["id"] for x in d.phone["events"]) + 1, "date": other.isoformat(), "start": hhmm})
    return d.recipe("two_events", f"Cancel \"{e['title']}\", without saying which day (there are two).",
                    nothing(["?"], [d.answer_day(first), d.answer_day(other)]),
                    [e["title"], "cancel|delete|remove|drop|scrap|clear|axe|nix"]) | {"read_has": [f"id {e['id']}:", f"id {d.phone['events'][-1]['id']}:"]}


def clear_day(d: Draw):
    day = busy_day(d)
    if not day:
        return None
    # One-off events on that day go; repeating ones stay but skip that day.
    after = [e | {"skip": [day.isoformat()]} if e["repeat"] and occurs(e, day) else e
             for e in d.phone["events"] if e["repeat"] or not occurs(e, day)]
    words = "tomorrow" if (day - d.now.date()).days == 1 else WEEKDAYS[day.weekday()]
    return d.recipe("clear_day", f"Clear the whole calendar for {words}.", [{"changes": {"events": after}}],
                    [words, "clear|cancel|delete|wipe|free|remove|empty|scrap|drop"])


# --- One call then stop, clean arguments ---

def app_missing(d: Draw):
    missing = [a for a in INGREDIENTS["apps"] if a not in d.phone["apps"]]
    if not missing:
        return None
    app = d.rng.choice(missing)
    return d.recipe("app_missing", f"Open {app} (it isn't installed).",
                    nothing([app], [NOT_INSTALLED], ["opened", "opening"]), [app, "open"]) | {"missing_app": app}


CATEGORIES = {
    "lookup": (0.20, [call_one, call_one, message_one, message_one, email_one, no_email, contact_missing, contact_info,
                      event_delete, event_move, agenda, if_free, event_when, move_all]),
    "dates": (0.15, [event_add, event_add, reminder_add, alarm_add, timer_units, event_repeat, two_actions, all_day]),
    "lookalike": (0.15, [reminder_in, note_find, note_missing, note_add, media, volume, phone_controls, phone_controls, date_math]),
    "followup": (0.12, [timer_change, event_change, pick_contact, list_then_act]),
    "notool": (0.10, [no_tool, no_tool, no_tool, mention, cant]),
    "ask": (0.10, [missing_info, missing_info, two_matches_ask, two_events_ask, clear_day]),
    "stopclean": (0.10, [event_add, message_one, app_missing]),
    "messy": (0.08, [call_one, message_one, event_add, alarm_add, reminder_in, note_find, media, timer_units, event_move,
                     event_delete, agenda, clear_day]),
}

# The tools a correct run of each kind may call; any other call is a useless one and fails the run.
ALLOWED_TOOLS = {
    "call_one": ["find_contact", "call"],
    "message_one": ["find_contact", "draft_message"],
    "email_one": ["find_contact", "draft_email"],
    "no_email": ["find_contact"],
    "contact_missing": ["find_contact"],
    "contact_info": ["find_contact"],
    "event_delete": ["find_events", "delete_events"],
    "event_move": ["find_events", "edit_events"],
    "agenda": ["find_events"],
    "if_free": ["find_events", "add_event"],
    "event_when": ["find_events"],
    "move_all": ["find_events", "edit_events"],
    "all_day": ["find_events", "add_event"],
    "note_missing": ["find_notes"],
    "event_add": ["find_events", "add_event"],  # looking at the day first is fine
    "reminder_add": ["add_reminder"],
    "alarm_add": ["set_alarm"],
    "timer_units": ["set_timer"],
    "event_repeat": ["find_events", "add_event"],
    "two_actions": ["find_events", "add_event", "add_reminder", "set_alarm", "set_timer"],
    "reminder_in": ["add_reminder"],
    "note_find": ["find_notes"],
    "note_add": ["add_note"],
    "media": ["media_control"],
    "volume": ["volume"],
    "flashlight": ["flashlight"],
    "settings": ["open_settings"],
    "navigate": ["navigate"],
    "music": ["play_music"],
    "open_app": ["open_app"],
    "days_until": ["days_until"],
    "split": ["calculate"],
    "timer_change": ["set_timer"],
    "event_change": ["find_events", "edit_events"],
    "pick_contact": ["find_contact", "call"],
    "list_then_act": ["find_events", "edit_events"],
    "chat": [], "negation": [], "later": [], "mention": [], "cant": [], "missing": [],
    "two_matches": ["find_contact"],
    "two_events": ["find_events"],
    "clear_day": ["find_events", "delete_events"],
    "app_missing": ["open_app"],  # the error lists what is installed
}
# The tools a correct run must call: what the model can't know without looking (is a contact, an email, an app there?
# is the flashlight already on? is the volume already there? is the user free?).
NEEDED_TOOLS = {"if_free": ["find_events"], "contact_missing": ["find_contact"], "no_email": ["find_contact"],
                "app_missing": ["open_app"], "flashlight": ["flashlight"], "volume": ["volume"], "note_missing": ["find_notes"],
                "contact_info": ["find_contact"], "two_matches": ["find_contact"], "two_events": ["find_events"],
                "agenda": ["find_events"], "event_when": ["find_events"], "note_find": ["find_notes"], "media": ["media_control"]}


# Earlier requests for a chat that already has history: kinds that change other parts of the phone than most recipes
# and never change the phone they're drawn on.
EARLIER = [timer_units, alarm_add, media, volume, note_add, reminder_add]


# The app keeps one chat until "new chat": the main request comes after an earlier, unrelated one on the same phone.
# Only an earlier kind that changes other parts of the phone, so the two outcomes add up. None if none fits.
def with_earlier(d: Draw, main: dict) -> dict | None:
    keys = {k for o in main["expect"] for k in o["changes"]}
    for make in d.rng.sample(EARLIER, len(EARLIER)):
        first = make(d)
        if first["kind"] != main["kind"] and not keys & set(first["expect"][0]["changes"]):
            return main | {"task": first["task"], "followup": main["task"], "say": first["say"] + main["say"], "earlier": first["kind"],
                           "first_turn": {key: first[key] for key in ("kind", "expect", "say")}, "follow_say": main["say"],
                           "expect": [o | {"changes": first["expect"][0]["changes"] | o["changes"]} for o in main["expect"]]}
    return None


# Recipe number `seed`: picks a category by weight, a recipe in it, and a style ("messy" only uses messy styles).
# A recipe returns None when the phone doesn't fit it (no email, no busy day...); then a new phone is drawn.
def draw(seed: int) -> dict:
    rng = random.Random(seed)
    names = list(CATEGORIES)
    category = rng.choices(names, weights=[CATEGORIES[n][0] for n in names])[0]
    make = rng.choice(CATEGORIES[category][1])
    for attempt in range(100):
        d = Draw(seed * 1000 + attempt)
        recipe = make(d)
        if recipe:
            break
    else:
        raise ValueError(f"{make.__name__} fits none of 100 phones")
    recipe["category"] = category
    if not recipe["followup"] and rng.random() < 0.15:
        recipe = with_earlier(d, recipe) or recipe
    if category == "messy":
        recipe["style"] = d.rng.choice(MESSY)
    else:
        styles = ["plain", "plain", "casual", "casual", "terse", "polite", "lowercase"]
        recipe["style"] = d.rng.choice(styles + (["indirect"] if recipe["kind"] in ("reminder_add", "reminder_in") else []))
    recipe["seed"] = seed
    recipe["shape"] = draw_shape(seed, recipe)
    return recipe


# A shape for the message, or None. Terse styles only get the short ones; "short" needs room for the required words.
def draw_shape(seed: int, recipe: dict) -> str | None:
    rng = random.Random(seed + 0x5EED)
    options = ["short", "value first"] if recipe["style"] == "terse" else list(SHAPES)
    if len(recipe["say"]) > 2:
        options = [o for o in options if o != "short"]
    return rng.choice(options) if rng.random() < 0.5 and options else None


if __name__ == "__main__":
    for seed in range(12):
        r = draw(seed)
        print(f"{r['category']:<10} {r['kind']:<16} {r['style']:<9} {r['now']}  {r['task']}" + (f"  THEN: {r['followup']}" if r["followup"] else ""))
        print("    say:", r["say"], " answer:", [{k: v for k, v in o.items() if k != "changes"} for o in r["expect"]])
