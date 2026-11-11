"""Новые источники -> общая схема стартов: data/extra_starts.parquet.

MyFinish (протоколы), results.zone (полный каталог велогонок), sportident.online (заявочные списки, Самарская область).
Классификация дисциплины и региона — функциями проекта rvvr (только чтение).
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT.parent / "rvvr" / "src"))
from rvvr.depth import person_key  # noqa: E402
from rvvr.races import classify, region_of  # noqa: E402

RAW = ROOT / "data" / "raw"
# не любительские старты: сборные, паралимпийские дисциплины, детские лиги, смежные виды
SKIP = re.compile(r"cyprus|georgian|триатлон|дуатлон|акватлон|лыжеролл|бего|чемпионат\s+россии|первенств\w*\s+россии|всероссийск|спорту?\s+глухих|"
                  r"поражением\s+ОДА|слепых|детск|спартакиад|велопробег|вело-?парад|ГТО|самокат|беговел", re.I)
CITY = {"дубна": "Московская область", "волоколамск": "Московская область", "тула": "Тульская область", "суздаль": "Владимирская область",
        "ярославль": "Ярославская область", "углич": "Ярославская область", "калуга": "Калужская область", "тверь": "Тверская область",
        "завидово": "Тверская область", "рязань": "Рязанская область", "коломна": "Московская область", "руза": "Московская область",
        "верея": "Московская область", "бородино": "Московская область", "истра": "Московская область", "дмитров": "Московская область",
        "яхрома": "Московская область", "звенигород": "Московская область", "можайск": "Московская область", "зеленоград": "Москва",
        "крылатское": "Москва", "кисловодск": "Ставропольский край", "пятигорск": "Ставропольский край", "геленджик": "Краснодарский край",
        "анапа": "Краснодарский край", "красная поляна": "Краснодарский край", "ростов": "Ростовская область", "калининград": "Калининградская область",
        "светлогорск": "Калининградская область", "псков": "Псковская область", "выборг": "Ленинградская область", "исеть": "Свердловская область",
        "сысерть": "Свердловская область", "миасс": "Челябинская область", "кисегач": "Челябинская область", "белокуриха": "Алтайский край",
        "барнаул": "Алтайский край", "заринск": "Алтайский край", "бирюзовая катунь": "Алтайский край", "горно-алтайск": "Республика Алтай",
        "кемерово": "Кемеровская область", "инской": "Кемеровская область", "мариинск": "Кемеровская область", "лесная поляна": "Кемеровская область",
        "маслянино": "Новосибирская область", "томск": "Томская область", "северск": "Томская область", "курск": "Курская область",
        "лихославль": "Тверская область", "удомля": "Тверская область", "цвелодубово": "Ленинградская область", "тарко-сале": "Ямало-Ненецкий автономный округ",
        "спутник": "Пензенская область", "курган": "Курганская область", "мурманск": "Мурманская область", "сыктывкар": "Республика Коми",
        "альметьевск": "Республика Татарстан", "елабуга": "Республика Татарстан", "мо": "Московская область",
        "мичуринск": "Ленинградская область", "токсов": "Ленинградская область", "орехово": "Ленинградская область", "коробицыно": "Ленинградская область",
        "коробицино": "Ленинградская область", "золотая долина": "Ленинградская область", "sosnovo": "Ленинградская область", "игора": "Ленинградская область",
        "igora": "Ленинградская область", "приозерск": "Ленинградская область", "сортавала": "Республика Карелия", "spb": "Санкт-Петербург",
        "saint petersburg": "Санкт-Петербург", "новокузнецк": "Кемеровская область", "шерегеш": "Кемеровская область", "шерегешь": "Кемеровская область",
        "полдневой": "Свердловская область", "ревда": "Свердловская область", "новоуральск": "Свердловская область", "новоуральский": "Свердловская область",
        "киргишаны": "Свердловская область", "кадниково": "Свердловская область", "полевской": "Свердловская область", "верхняя салда": "Свердловская область",
        "коптяки": "Свердловская область", "хрустальная": "Свердловская область", "калиновка": "Свердловская область", "калиновский": "Свердловская область",
        "евразия": "Свердловская область", "ural mtb": "Свердловская область", "верхний уфалей": "Челябинская область", "большой камень": "Челябинская область",
        "троицк": "Челябинская область", "тургояк": "Челябинская область", "солнечная долина": "Челябинская область", "новососедово": "Новосибирская область",
        "березовка": "Алтайский край", "гришкино": "Тверская область", "анцифарово": "Тверская область", "шуклинка": "Курская область",
        "kursk": "Курская область", "курской": "Курская область", "kazan": "Республика Татарстан", "tula": "Тульская область", "поленово": "Тульская область",
        "серпухов": "Московская область", "serpukhov": "Московская область", "ногинск": "Московская область", "noginsk": "Московская область",
        "лотошино": "Московская область", "dmitrov": "Московская область", "сергиев посад": "Московская область", "пересвет": "Московская область",
        "клёново": "Москва", "кленово": "Москва", "садовое кольцо": "Москва", "крылатские": "Москва", "лужники": "Москва", "парк горького": "Москва",
        "коломенское": "Москва", "томилин": "Московская область", "ромашков": "Московская область", "мангазея": "Москва", "столица": "Москва",
        "олимпия": "Московская область", "две реки": "Московская область", "кремль": "Московская область", "спортмастер pro": "Москва"}


def region(*texts):
    r = region_of(*texts)
    if r:
        return r
    blob = " ".join(t for t in texts if isinstance(t, str)).lower().replace("ё", "е")
    for c, reg in CITY.items():
        if re.search(rf"(?<![а-яa-z]){re.escape(c.replace('ё', 'е'))}" + (r"(?![а-яa-z])" if len(c) <= 4 else ""), blob):
            return reg
    return None


def gender_of(group):
    g = (group or "").strip().lower()
    if re.match(r"(м|муж|юноши|юниоры|мальчики|men|male|m\d|m[ -])", g) or re.search(r"мужчин|юнош|юниоры|мальчик", g):
        return "m"
    if re.match(r"(ж|жен|девушки|юниорки|девочки|women|female|w\d|f\d)", g) or re.search(r"женщин|девуш|юниорки|девочк", g):
        return "f"
    return None


MALE_A = {"никита", "илья", "лука", "фома", "кузьма", "савва", "данила", "добрыня", "муса", "иса", "мустафа", "лёва", "лева"}
AMBIG = {"саша", "женя", "валя", "слава", "паша", "миша", "дима", "вова", "коля", "петя", "ваня", "лёша", "леша", "гоша", "сережа", "серёжа",
         "костя", "вася", "толя", "гриша", "рома", "тема", "тёма", "гена", "жора", "лёня", "леня", "сева", "стёпа", "степа", "федя", "юра", "витя",
         "боря", "вадя", "володя", "андрюша", "илюша", "антоша", "кирюша", "данила"}


def gender_by_name(name):
    """Пол по фамилии и имени, когда его нет в протоколе: женские окончания фамилии (-ова, -ина, -ская) или имени (-а, -я)."""
    t = [x for x in re.sub(r"[^а-яё\s-]", " ", str(name).lower()).split() if len(x) > 1]
    if len(t) < 2:
        return None
    sur, first = t[0], t[1]
    if first in AMBIG:
        return "f" if re.search(r"(ова|ева|ёва|ина|ына|ская|цкая|ая)$", sur) else ("m" if re.search(r"(ов|ев|ёв|ин|ын|ский|цкий|ой|ий)$", sur) else None)
    if first in MALE_A:
        return "m"
    if re.search(r"(ова|ева|ёва|ина|ына|ская|цкая)$", sur) and first[-1] in "ая":
        return "f"
    if re.search(r"(ов|ев|ёв|ин|ын|ский|цкий)$", sur) and first[-1] not in "ая":
        return "m"
    if first[-1] in "ая":
        return "f"
    return "m" if first[-1] in "йнрлвдмтсгкбпзхьш" else None


NOT_BIKE = re.compile(r"\brun\b|бег\b|забег|ходьб|лыж|пеш", re.I)


def kind(title, default):
    t = classify(title)
    if t != "unknown":
        return t
    if re.search(r"марафон|xc[omc]?\b|хс[ом]\b|кросс|mtb|мтб|трейл|эндуро|enduro|bike marathon", title, re.I):
        return "mtb"
    return default


def myfinish():
    out = []
    for l in (RAW / "myfinish/results.jsonl").open():
        e = json.loads(l)
        if SKIP.search(e["name"]):
            continue
        date = pd.to_datetime(e["ts"], unit="s") + pd.Timedelta(5, "h")  # метка времени — полночь по местному времени
        date = date.normalize()
        typ, reg, title = kind(e["name"], "road"), region(e.get("place"), e["name"]), re.sub(r"\s+", " ", e["name"]).strip()
        for g in e["groups"]:
            if NOT_BIKE.search(f"{g['group']} {g.get('descr') or ''}"):
                continue
            sex = gender_of(g["group"]) or gender_of(g.get("descr"))
            for r in g["rows"]:
                if r.get("status") in (8,) or not r.get("name"):
                    continue
                out.append(("myfinish", f"mf-{e['id']}", date, r["name"], sex or gender_by_name(r["name"]), r.get("birth") or None, reg, r.get("home"), typ, title, e.get("org")))
    return out


def resultszone():
    out = []
    for l in (RAW / "resultszone/results.jsonl").open():
        e = json.loads(l)
        if SKIP.search(e["event"] + " " + e["race"]):
            continue
        typ, reg = kind(e["event"] + " " + e["race"], "road"), region(e["event"], e["slug"].replace("-", " "))
        for r in e["rows"]:
            if r.get("status") == "DNS" or not r.get("name"):
                continue
            out.append(("results.zone", e["slug"], pd.Timestamp(e["date"]), r["name"], {"male": "m", "female": "f"}.get(r.get("gender")), None, reg,
                        r.get("city"), typ, e["event"], e.get("series")))
    return out


def sportident():
    out = []
    for l in (RAW / "sportident/participants.jsonl").open():
        e = json.loads(l)
        date = pd.to_datetime(e["date"], format="%d.%m.%Y", errors="coerce")
        reg, typ = region(e["place"]), kind(e["title"], "mtb")
        for r in e["rows"]:
            out.append(("sportident", f"si-{e['id']}", date, r["name"], gender_of(r["group"]) or gender_by_name(r["name"]), r.get("birth"), reg, r.get("city"), typ, e["title"], None))
    return out


def build():
    cols = ["src", "event", "date", "name", "gender", "birth", "region_event", "region_home", "type", "event_name", "organizer"]
    d = pd.DataFrame(myfinish() + resultszone() + sportident(), columns=cols)
    d["key"] = d.name.map(person_key)
    d = d.dropna(subset=["key", "date"])
    d["year"] = d.date.dt.year
    d = d[d.year.between(2014, 2026) & ~d.type.isin(["mass_ride", "online"])]
    d["birth"] = pd.to_numeric(d.birth, errors="coerce")
    d.loc[~d.birth.between(1930, 2022), "birth"] = None
    d = d.drop_duplicates(["src", "event", "key"])
    d.to_parquet(ROOT / "data" / "extra_starts.parquet", index=False)
    return d


if __name__ == "__main__":
    d = build()
    print(d.groupby("src").agg(rows=("key", "size"), people=("key", "nunique"), events=("event", "nunique"), no_region=("region_event", lambda s: s.isna().mean()),
                               no_gender=("gender", lambda s: s.isna().mean())))
    print(d.type.value_counts().to_dict())
    print(d[d.year >= 2023].groupby("region_event", dropna=False).agg(rows=("key", "size"), ev=("event", "nunique")).sort_values("rows", ascending=False).head(45).to_string())
    miss = d[d.region_event.isna()].groupby(["src", "event_name"]).size().sort_values(ascending=False).head(40)
    print(miss.to_string())
