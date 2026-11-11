"""Признаки новичка: что было вокруг его первой гонки."""
import numpy as np
import pandas as pd

from deep.base import DATA, VOLGA


def region_group(r):
    if r == "Москва и Московская область":
        return "Москва"
    if r == "Санкт-Петербург и Ленинградская область":
        return "Петербург"
    if r == "Приморский край":
        return "Приморье"
    if r in VOLGA:
        return "Поволжье"
    return "нет данных" if pd.isna(r) else "другие регионы"


def newcomers() -> pd.DataFrame:
    p = pd.read_parquet(DATA / "rider_seasons.parquet")
    ev = pd.read_parquet(DATA / "events.parquet")
    new = p[p.is_new & p.year.between(2023, 2025)].copy()
    e = ev.set_index("event_id")
    new["entry_riders"] = new.entry_event.map(e.riders)
    new["entry_women_share"] = new.entry_event.map(e.women_share)
    new["entry_organizer"] = new.entry_event.map(e.organizer)

    # календарь вокруг первой гонки: когда следующая гонка той же дисциплины в том же регионе
    reg = ev.dropna(subset=["region"]).sort_values("date")
    gap, near60, season_left = {}, {}, {}
    for eid, row in e.loc[new.entry_event.unique()].iterrows():
        if pd.isna(row.region):
            continue
        c = reg[(reg.region == row.region) & (reg.type == row.type) & (reg.date > row.date) & (reg.year == row.year)]
        gap[eid] = (c.date.min() - row.date).days if len(c) else np.nan
        near60[eid] = int(((c.date - row.date).dt.days <= 60).sum())
        season_left[eid] = len(c)
    new["next_gap"] = new.entry_event.map(gap)
    new["events_60d"] = new.entry_event.map(near60)
    new["events_left"] = new.entry_event.map(season_left)

    per_series = ev.groupby(["series", "year"]).event_id.nunique().rename("series_events")
    new = new.merge(per_series, left_on=["entry_series", "year"], right_index=True, how="left")
    per_org = ev.dropna(subset=["organizer"]).groupby(["organizer", "year"]).event_id.nunique().rename("organizer_events")
    new = new.merge(per_org, left_on=["entry_organizer", "year"], right_index=True, how="left")

    new["entry_month"] = new["first"].dt.month
    new["region_group"] = new.entry_region.map(region_group)
    new["two"] = new.starts >= 2
    new["two_60d"] = new.days_to_second.le(60)
    new["female"] = (new.gender == "f").astype(float).where(new.gender.notna())
    return new


if __name__ == "__main__":
    n = newcomers()
    n.to_parquet(DATA / "newcomers.parquet", index=False)
    print(len(n), n[["two", "two_60d", "returned"]].mean().round(3).to_dict())
    print(n.groupby(pd.cut(n.series_events, [0, 1, 2, 4, 8, 100])).agg(n=("key", "size"), two=("two", "mean"), ret=("returned", "mean")).round(3))
    print(n.groupby("region_group").agg(n=("key", "size"), two=("two", "mean"), ret=("returned", "mean"), gap=("next_gap", "median")).round(3))
