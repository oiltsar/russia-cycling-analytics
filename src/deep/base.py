"""База для анализа: старты без дублей, события, серии, панель «гонщик × сезон».

Читает данные проекта rvvr только на чтение. Персональные данные остаются локально:
в data/ лежат таблицы с ключом гонщика, в git они не попадают.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RVVR = ROOT.parent / "rvvr"
DATA = ROOT / "data"

VOLGA = ["Самарская область", "Республика Татарстан", "Нижегородская область", "Саратовская область", "Ульяновская область",
         "Пензенская область", "Оренбургская область", "Республика Башкортостан", "Пермский край", "Удмуртская Республика",
         "Чувашская Республика", "Республика Марий Эл", "Республика Мордовия", "Кировская область"]
REGION_FIX = {"Ленинградская область": "Санкт-Петербург и Ленинградская область", "Московская область": "Москва и Московская область",
              "Москва": "Москва и Московская область", "Санкт-Петербург": "Санкт-Петербург и Ленинградская область"}
NOT_RACES = r"uci-|aalborg|chempionat-evropy|chempionat-rossii|detsk|lyzheroller|spidskeyt|cyprus|kipr|world-championship|pervenstvo-rossii"
SRC_PRIORITY = {"fatracing": 0, "results.zone": 1, "russiarunning": 2, "orgeo": 3}


def series_id(slug: str) -> str:
    """Серия из адреса события: убираем год, номер этапа и служебные хвосты."""
    s = slug.lower().split("|")[0]
    s = re.sub(r"(19|20)\d\d", "", s)
    s = re.sub(r"(^|[-_])(\d{1,2}|i{1,3}|iv|v|vi{0,3})[-_]?(etap|stage|tur|den)\b", r"\1", s)
    s = re.sub(r"\b(etap|stage)[-_]?\d{0,2}\b", "", s)
    s = re.sub(r"[-_]\d{1,3}$", "", s)
    s = re.sub(r"[-_]{2,}", "-", s).strip("-_ ")
    return s or slug


def parse_date(d, year):
    """Даты Orgeo записаны как «дд.мм.гггг» или диапазон; берём первый день."""
    t = pd.to_datetime(d, errors="coerce", format="%Y-%m-%d")
    if pd.notna(t):
        return t
    m = re.match(r"(\d{2})(?:-\d{2})?\.(\d{2})\.(\d{1,4})", str(d))
    if m:
        try:
            return pd.Timestamp(year=int(year), month=int(m.group(2)), day=int(m.group(1)))
        except ValueError:
            return pd.NaT
    return pd.NaT


def build():
    a = pd.read_parquet(RVVR / "data/processed/all_starts.parquet")
    a["date"] = [parse_date(d, y) for d, y in zip(a.date, a.year)]
    a = a.dropna(subset=["date", "key"])
    # не любительские велогонки России: зарубежные старты, детские лиги, лыжероллеры, чемпионаты сборных
    a = a[~a.event.str.contains(NOT_RACES, case=False, regex=True)]
    a["year"] = a.date.dt.year
    a["region"] = a.region_event.replace(REGION_FIX)
    # одно событие приходит из нескольких источников: оставляем один старт на человека в день
    a["prio"] = a.src.map(SRC_PRIORITY)
    a = a.sort_values("prio").drop_duplicates(["key", "date"]).drop(columns="prio")
    # события, сведённые между источниками: день + регион + дисциплина дают один и тот же старт
    a["event_id"] = a.src + ":" + a.event
    canon = (a.groupby(["date", "event_id"]).size().rename("n").reset_index()
             .sort_values("n", ascending=False))
    a["series"] = a.event.map(series_id)

    org = pd.read_parquet(RVVR / "data/processed/participations.parquet").drop_duplicates("slug").set_index("slug")
    a["organizer"] = a.event.map(org.organizer)
    a["event_name"] = a.event.map(org.event_name)
    titles = pd.read_parquet(RVVR / "data/processed/orgeo_events.parquet")
    titles["orgeo_id"] = titles.orgeo_id.astype(str)
    a.loc[a.src == "orgeo", "event_name"] = a.loc[a.src == "orgeo", "event"].map(titles.set_index("orgeo_id").title)

    g = a.groupby("key")
    a["first_date"] = g.date.transform("min")
    a["first_year"] = a.first_date.dt.year
    ev = a.groupby("event_id").agg(src=("src", "first"), event=("event", "first"), name=("event_name", "first"), series=("series", "first"),
                                   organizer=("organizer", "first"), date=("date", "min"), year=("year", "first"),
                                   type=("type", lambda s: s.mode().iat[0]), region=("region", "first"),
                                   riders=("key", "nunique"), women=("gender", lambda s: (s == "f").sum()),
                                   gender_known=("gender", lambda s: s.notna().sum()),
                                   debut=("first_date", lambda s: 0)).reset_index()
    deb = a[a.date == a.first_date].groupby("event_id").key.nunique()
    ev["debut"] = ev.event_id.map(deb).fillna(0).astype(int)
    ev["women_share"] = ev.women / ev.gender_known.replace(0, np.nan)
    ev["volga"] = ev.region.isin(VOLGA)

    DATA.mkdir(exist_ok=True)
    cols = ["key", "event_id", "series", "organizer", "date", "year", "type", "region", "gender", "birth", "region_home", "first_date", "first_year", "src"]
    a[cols].to_parquet(DATA / "starts.parquet", index=False)
    ev.to_parquet(DATA / "events.parquet", index=False)
    return a[cols], ev


def rider_seasons(starts: pd.DataFrame) -> pd.DataFrame:
    """Панель «гонщик × сезон» с признаками первого старта сезона и фактом возврата на следующий год."""
    s = starts.sort_values(["key", "date"])
    g = s.groupby(["key", "year"])
    p = g.agg(starts=("date", "size"), events=("event_id", "nunique"), series=("series", "nunique"), types=("type", "nunique"),
              regions=("region", "nunique"), first=("date", "min"), last=("date", "max"), entry_event=("event_id", "first"),
              entry_type=("type", "first"), entry_region=("region", "first"), entry_series=("series", "first"),
              gender=("gender", "first"), birth=("birth", "max"), first_year=("first_year", "first")).reset_index()
    p["second"] = g.date.nth(1).reset_index(drop=True) if False else np.nan
    second = s[s.groupby(["key", "year"]).cumcount() == 1].set_index(["key", "year"]).date
    p["second"] = pd.MultiIndex.from_frame(p[["key", "year"]]).map(second)
    p["days_to_second"] = (p.second - p["first"]).dt.days
    p["span_days"] = (p["last"] - p["first"]).dt.days
    p["is_new"] = p.year == p.first_year
    nxt = set(zip(p.key, p.year - 1))
    p["returned"] = [(k, y) in nxt for k, y in zip(p.key, p.year)]
    p["age"] = p.year - p.birth
    return p


if __name__ == "__main__":
    starts, ev = build()
    p = rider_seasons(starts)
    p.to_parquet(DATA / "rider_seasons.parquet", index=False)
    print("стартов", len(starts), "гонщиков", starts.key.nunique(), "событий", len(ev), "серий", starts.series.nunique())
    print(starts.groupby("year").agg(starts=("key", "size"), riders=("key", "nunique"), events=("event_id", "nunique")).tail(6))
    print(p[p.year.between(2023, 2025)].groupby("year").agg(riders=("key", "size"), new=("is_new", "mean"), ret=("returned", "mean"), two=("starts", lambda x: (x >= 2).mean())))
    print(ev.sort_values("riders", ascending=False).head(8)[["event_id", "series", "organizer", "date", "type", "region", "riders", "debut"]].to_string())
    print(starts.series.value_counts().head(25).to_dict())
