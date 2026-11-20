# Random phones for generated cases: contacts from many countries, a calendar, notes, alarms, and a "now".
# Everything is drawn from ingredients.json with a seeded random, so the same seed gives the same phone.

import json
import random
from datetime import date, datetime, time, timedelta
from pathlib import Path

from app.sim import fold

INGREDIENTS = json.loads((Path(__file__).parent / "ingredients.json").read_text())
CORE_APPS = ["Calendar", "Clock", "Contacts", "Messages", "Phone"]


# A phone number in the region's style: international ("+44 7700 900123") or local ("07700 900123").
def phone_number(rng: random.Random, region: dict) -> str:
    digits = "".join(str(rng.randint(0, 9)) if c == "#" else c for c in rng.choice(region["mobile"]))
    if rng.random() < 0.3 and region["local_prefix"] is not None:
        return region["local_prefix"] + digits
    return f"{region['code']} {digits}"


# One contact. Most come from the user's home region; `first` forces a first name (for "two Giulias").
def contact(rng: random.Random, home: str, first: str | None = None) -> dict:
    region_name = home if rng.random() < 0.7 else rng.choice(list(INGREDIENTS["regions"]))
    region = INGREDIENTS["regions"][region_name]
    first = first or rng.choice(region["first"])
    last = rng.choice(region["last"])
    email = None
    if rng.random() < 0.5:
        email = fold(f"{first}.{last}@{rng.choice(INGREDIENTS['email_domains'])}").encode("ascii", "ignore").decode()
    return {"name": f"{first} {last}", "phone": phone_number(rng, region), "email": email}


# Contacts: a few family members and services, the rest people. Names are unique unless forced later.
def contacts(rng: random.Random, home: str) -> list[dict]:
    region = INGREDIENTS["regions"][home]
    people = [{"name": n, "phone": phone_number(rng, region), "email": None}
              for n in rng.sample(INGREDIENTS["family"], rng.randint(1, 3))]
    people += [{"name": n, "phone": phone_number(rng, region), "email": None}
               for n in rng.sample(INGREDIENTS["services"], rng.randint(1, 3))]
    names = {p["name"].split()[0] for p in people}
    while len(people) < rng.randint(8, 14):
        c = contact(rng, home)
        if c["name"].split()[0] not in names:
            names.add(c["name"].split()[0])
            people.append(c)
    rng.shuffle(people)
    return people


# True for real people ("Sara Romano"), false for "Mum", "My sister" or "Dentist".
def is_person(c: dict) -> bool:
    return c["name"] not in INGREDIENTS["family"] + INGREDIENTS["services"]


# Fills {person} and {city} in a title from this phone's contacts.
def fill(rng: random.Random, text: str, people: list[dict]) -> str:
    first_names = [p["name"].split()[0] for p in people if is_person(p)]
    return text.replace("{person}", rng.choice(first_names)).replace("{city}", rng.choice(INGREDIENTS["cities"]))


# Events from a few days ago to three weeks ahead, plus a weekly one and a yearly all-day one. Ids follow date order.
def events(rng: random.Random, today: date, people: list[dict]) -> list[dict]:
    out = []
    for _ in range(rng.randint(5, 12)):
        kind = rng.choice(INGREDIENTS["events"])
        day = today + timedelta(days=rng.randint(-3, 21))
        start = time(rng.randint(*kind["hours"]), rng.choice([0, 15, 30, 45]))
        out.append({"title": fill(rng, kind["title"], people), "date": day.isoformat(),
                    "start": start.strftime("%H:%M"), "duration_minutes": rng.choice(kind["minutes"]), "repeat": None})
    weekly = rng.choice(INGREDIENTS["events"])
    out.append({"title": fill(rng, weekly["title"], people), "date": (today - timedelta(days=rng.randint(1, 60))).isoformat(),
                "start": time(rng.randint(*weekly["hours"]), 0).strftime("%H:%M"),
                "duration_minutes": rng.choice(weekly["minutes"]), "repeat": "weekly"})
    birthday = today + timedelta(days=rng.randint(1, 300))
    birthday = birthday.replace(day=28) if (birthday.month, birthday.day) == (2, 29) else birthday
    out.append({"title": fill(rng, "{person}'s birthday", people), "date": birthday.replace(year=1990).isoformat(),
                "start": None, "duration_minutes": None, "repeat": "yearly"})
    out.sort(key=lambda e: (e["date"], e["start"] or ""))
    return [{"id": i + 1} | e for i, e in enumerate(out)]


# A random moment: any day of 2026–2027, at an hour people use phones (late night included).
def random_now(rng: random.Random) -> datetime:
    day = date(2026, 1, 1) + timedelta(days=rng.randint(0, 729))
    return datetime.combine(day, time(rng.choice(INGREDIENTS["now"]["hours"]), rng.randint(0, 59)))


# A whole phone state, in the format of sim.EMPTY_STATE.
def random_phone(rng: random.Random, now: datetime) -> dict:
    home = rng.choice(list(INGREDIENTS["regions"]))
    people = contacts(rng, home)
    extra_apps = [a for a in INGREDIENTS["apps"] if a not in CORE_APPS]
    return {
        "events": events(rng, now.date(), people),
        "contacts": people,
        "notes": [{"text": fill(rng, n["text"], people)} for n in rng.sample(INGREDIENTS["notes"], rng.randint(2, 5))],
        "reminders": [],
        "alarms": [{"time": f"{rng.randint(5, 9):02d}:{rng.choice([0, 15, 30]):02d}",
                    "days": rng.choice([[], ["monday", "tuesday", "wednesday", "thursday", "friday"]])}
                   for _ in range(rng.randint(0, 2))],
        "timers": [],
        "media": {"playing": rng.random() < 0.5, "track": 1, "volume": rng.choice(range(20, 90, 10))},
        "flashlight": False,
        "apps": sorted(CORE_APPS + rng.sample(extra_apps, rng.randint(4, len(extra_apps)))),
        "opened": [],
    }


if __name__ == "__main__":
    rng = random.Random(1)
    now = random_now(rng)
    print(now.strftime("%A %Y-%m-%d %H:%M"))
    print(json.dumps(random_phone(rng, now), indent=1, ensure_ascii=False))
