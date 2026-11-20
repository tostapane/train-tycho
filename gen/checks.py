# Text matching for generated messages: does the user's message say what the recipe needs, whatever the wording
# (typos, short forms, number words, "7pm" or "19:00", "an hour and a half")? Also the wording filters.
# `says(text, phrase)` is the main entry; "a|b" in a phrase means either one.

import difflib
import re
import unicodedata

SHORT = {"tomorrow": ["tmrw", "tmr", "tmrow", "2moro", "tomo"], "tonight": ["tonite", "2nite"],
         "minutes": ["min", "mins"], "hours": ["hr", "hrs", "h"], "seconds": ["sec", "secs"],
         "minute": ["min", "mins"], "hour": ["hr", "hrs", "h"], "second": ["sec", "secs"], "with": ["w"], "you": ["u"],
         "monday": ["mon"], "tuesday": ["tue", "tues"], "wednesday": ["wed", "weds"], "thursday": ["thu", "thur", "thurs"],
         "friday": ["fri"], "saturday": ["sat"], "sunday": ["sun"], "please": ["pls", "plz"], "message": ["msg"], "mum": ["mom", "mother", "mama", "mummy", "mommy"],
         "text": ["txt", "texting", "sms", "message"], "open": ["launch", "start", "opening"], "timer": ["countdown"],
         "alarm": ["wake"], "every": ["each", "weekly", "mondays", "tuesdays", "wednesdays", "thursdays", "fridays", "saturdays", "sundays"]}
SHORT |= {m.lower(): [m[:3].lower()] for m in ["January", "February", "March", "April", "June", "July", "August",
                                               "September", "October", "November", "December"]}
SKIP = {"the", "and", "for", "you", "can"}
ASSISTANT = re.compile(r"^\W*(i'?m sorry|sorry,? i|sure\b|got it|of course|certainly|would you like|here'?s|here is|"
                       r"happy to|no problem|as an ai|i can help|i'?d be happy|okay, i'?ll|"
                       r"i can (?:set|add|create|start|make|open|send|play|turn|do)\b|i(?:'?ll| will) (?:set|add|create|start|make|open|send|play|turn)\b)", re.I)
# "ring" meaning "call" is out; it stays for alarms and timers.
RING = re.compile(r"\bring(?:ing)?\b", re.I)
RING_OK = {"alarm_add", "timer_units", "timer_change"}


ONES = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
                                   "sixteen seventeen eighteen nineteen".split())}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
ORDINALS = {"first": 1, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
            "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15, "sixteenth": 16,
            "seventeenth": 17, "eighteenth": 18, "nineteenth": 19, "twentieth": 20, "thirtieth": 30}
MONTHS = {m: i for i, m in enumerate("january february march april may june july august september october november december".split(), 1)}
MONTHS |= {m[:3]: i for m, i in list(MONTHS.items())}
UNITS = {"h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600, "m": 60, "min": 60, "mins": 60, "minute": 60,
         "minutes": 60, "s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1}
DURATION_WORDS = set(UNITS) | {"a", "an", "and", "of", "half", "quarter", "quarters"}
UNIT = "|".join(sorted(UNITS, key=len, reverse=True))


# Lowercase without accents ("Andrés" → "andres"), number words as digits ("twenty-five" → "25", "thirteenth" → "13").
def plain(text: str) -> str:
    text = "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))
    combos = "|".join(w for w in list(ONES)[1:10] + list(ORDINALS)[:8])
    text = re.sub(rf"\b({'|'.join(TENS)})[ -]({combos})\b",
                  lambda m: str(TENS[m[1]] + (ONES.get(m[2]) or ORDINALS[m[2]])), text)
    return re.sub(rf"\b({'|'.join(list(ONES) + list(TENS) + list(ORDINALS))})\b",
                  lambda m: str((ONES | TENS | ORDINALS)[m[1]]), text)


# Words of a text: times joined ("7:00 PM" → "7pm"), "13th" → "13", number words as digits, no leading zeros.
def words(text: str) -> list[str]:
    text = plain(text)
    text = re.sub(r"(\d{1,2}):00\s*(am|pm)\b", r"\1\2", text)
    text = re.sub(r"(\d)\s+(am|pm)\b", r"\1\2", text)
    text = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", r"\1", text)
    return [str(int(w)) if w.isdigit() else w for w in re.findall(r"[a-z0-9]+", text)]


# The lengths of time in a text, in seconds: "1.5 hours", "an hour and a half", "90 min", "1h15", "half an hour".
def durations(text: str) -> set[float]:
    t = plain(text).replace(",", ".")
    t = re.sub(rf"\b(?:an?|(\d+(?:\.\d+)?))\s*({UNIT})\s+and\s+(?:a\s+)?half\b",
               lambda m: f"{float(m[1] or 1) + 0.5} {m[2]}", t)
    t = re.sub(r"\bhalf\s+(?:an?\s+)?(hour|minute)\b", r"0.5 \1", t)
    t = re.sub(r"\b(?:(\d+)|an?)\s+quarters?\s+(?:of\s+an?\s+)?hour\b", lambda m: f"{int(m[1] or 1) * 0.25} hour", t)
    t = re.sub(r"\bquarter\s+(?:of\s+an?\s+)?hour\b", "0.25 hour", t)
    t = re.sub(r"\ban?\s+(hour|minute)\b", r"1 \1", t)
    t = re.sub(rf"\b(\d+)\s*(?:h|hr|hrs|hours?)\s*(\d+)\b(?!\s*(?:{UNIT})\b)", r"\1 hours \2 minutes", t)
    pair = rf"\d+(?:\.\d+)?\s*(?:{UNIT})\b"
    return {sum(float(n) * UNITS[u] for n, u in re.findall(rf"(\d+(?:\.\d+)?)\s*({UNIT})\b", run))
            for run in re.findall(rf"(?:{pair}(?:\s*,?\s*(?:and\s+)?)?)+", t)}


# The clock times in a text, each as the minutes-of-day it may mean ("7pm" → {1140}; a bare "at 7" → {420, 1140}).
def times(text: str) -> list[set[int]]:
    t, found = plain(text), []
    for h, m, mer in re.findall(r"\b(\d{1,2})(?:[:.h]?(\d{2}))?\s*(am|pm)\b", t):
        found.append({(int(h) % 12 + (12 if mer == "pm" else 0)) * 60 + int(m or 0)})
    for h, m, day in re.findall(r"\b(\d{1,2})(?::(\d{2}))?\s+in the (morning|afternoon|evening)\b", t):
        found.append({(int(h) % 12 + (0 if day == "morning" else 12)) * 60 + int(m or 0)})
    for h, m in re.findall(r"\b(\d{1,2}):(\d{2})\b(?!\s*(?:am|pm))", t):
        found.append({int(h) * 60 + int(m)} if int(h) > 12 or h.startswith("0") else {int(h) % 12 * 60 + int(m), (int(h) % 12 + 12) * 60 + int(m)})
    for h, m in re.findall(r"\b([01]\d|2[0-3])([0-5]\d)\b", t):
        found.append({int(h) * 60 + int(m)})
    for h, m in re.findall(r"\b(\d{1,2})h(\d{2})?\b", t):
        found.append({int(h) * 60 + int(m or 0)} if int(h) > 12 else {int(h) % 12 * 60 + int(m or 0), (int(h) % 12 + 12) * 60 + int(m or 0)})
    for h in re.findall(r"\b(?:at|@)\s+(\d{1,2})\b(?!\s*(?:am|pm|[:.]\d))", t):
        found.append({int(h) % 12 * 60, (int(h) % 12 + 12) * 60})
    return found


# The minute-of-day of a phrase that is a clock time ("7pm", "10:15am", "16:00"), else None.
def clock(phrase: str) -> int | None:
    m = re.fullmatch(r"\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*", phrase.lower())
    if not m or not (m[2] or m[3]):
        return None
    return (int(m[1]) % 12 + (12 if m[3] == "pm" else 0) if m[3] else int(m[1])) * 60 + int(m[2] or 0)


# True if `word` is in `found`, allowing short forms ("tues"), prefixes and typos, a month as a number ("14/2"), and
# the reverse, but never a different number.
def has_word(word: str, found: list[str]) -> bool:
    if word in found or any(s in found for s in SHORT.get(word, [])):
        return True
    if word in MONTHS:
        return str(MONTHS[word]) in found
    if word.isdigit():
        return any(MONTHS.get(f) == int(word) for f in found)
    if any(c.isdigit() for c in word):
        return False
    return any(len(f) >= max(3, len(word) // 2) and (word.startswith(f) or difflib.SequenceMatcher(None, word, f).ratio() >= 0.8)
               for f in found)


# True if the message says `phrase`: a time or a length of time by its value, else every word longer than two letters
# (and every number). "a|b" means either one.
def says(text: str, phrase: str) -> bool:
    if "|" in phrase:
        return any(says(text, option) for option in phrase.split("|"))
    found = words(text)
    if all(has_word(w, found) for w in words(phrase) if (len(w) > 2 or w.isdigit()) and w not in SKIP):
        return True
    if (minute := clock(phrase)) is not None:
        return any(minute in options for options in times(text))
    own = durations(phrase)
    return bool(own) and all(w.isdigit() or w in DURATION_WORDS for w in words(phrase)) and own <= durations(text)
