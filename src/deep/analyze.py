"""Все расчёты исследования. Результат — обезличенные таблицы в out/ (имён и ключей гонщиков в них нет)."""
import json
import warnings

import community as louvain
import lightgbm as lgb
import networkx as nx
import numpy as np
import pandas as pd
import shap
import statsmodels.formula.api as smf
from lifelines import KaplanMeierFitter
from sklearn.model_selection import cross_val_score

from deep.base import DATA, ROOT, RVVR, VOLGA

warnings.filterwarnings("ignore")
OUT = ROOT / "out"
TYPE_RU = {"road": "шоссе", "mtb": "МТБ", "gravel": "гревел", "track": "трек", "cyclocross": "велокросс", "unknown": "не указана"}


def rate_table(d, by, name):
    t = d.groupby(by, observed=True).agg(newcomers=("key", "size"), second_start=("two", "mean"), returned=("returned", "mean")).reset_index()
    t.to_csv(OUT / f"{name}.csv", index=False)
    return t


def calendar(n):
    """Как календарь вокруг первой гонки связан со вторым стартом и возвратом."""
    d = n[n.entry_region.notna()].copy()
    d["gap"] = pd.cut(d.next_gap, [0, 14, 30, 60, 400], labels=["до 2 недель", "2–4 недели", "1–2 месяца", "больше 2 месяцев"]).astype(object)
    d["gap"] = d.gap.where(d.next_gap.notna(), "до конца сезона гонок нет")
    order = ["до 2 недель", "2–4 недели", "1–2 месяца", "больше 2 месяцев", "до конца сезона гонок нет"]
    t = rate_table(d, "gap", "calendar_gap").set_index("gap").loc[order].reset_index()
    t.to_csv(OUT / "calendar_gap.csv", index=False)
    rate_table(n, "entry_month", "entry_month")
    n["size"] = pd.cut(n.entry_riders, [0, 50, 100, 200, 500, 1000, 5000], labels=["до 50", "50–100", "100–200", "200–500", "500–1000", "1000+"])
    rate_table(n, "size", "entry_size")
    n["entry_type_ru"] = n.entry_type.map(TYPE_RU)
    rate_table(n, "entry_type_ru", "entry_type")
    g = n.groupby("region_group").agg(newcomers=("key", "size"), second_start=("two", "mean"), returned=("returned", "mean"),
                                      median_gap=("next_gap", "median"), no_next=("next_gap", lambda s: s.isna().mean())).reset_index()
    g.to_csv(OUT / "region_groups.csv", index=False)
    sec = n[n.two].copy()
    sec["when"] = pd.cut(sec.days_to_second, [-1, 7, 30, 60, 400], labels=["в ту же неделю", "8–30 дней", "31–60 дней", "позже 60 дней"])
    sec.groupby("when", observed=True).agg(riders=("key", "size"), returned=("returned", "mean")).reset_index().to_csv(OUT / "second_timing.csv", index=False)
    return t, g


def drivers(n):
    """Что предсказывает возврат новичка: градиентный бустинг и SHAP."""
    d = n[n.entry_region.notna()].copy()
    feats = ["entry_riders", "entry_month", "next_gap", "events_60d", "events_left", "series_events", "entry_women_share", "female", "starts", "types", "span_days"]
    x = d[feats].copy()
    x["next_gap"] = x.next_gap.fillna(400)
    for c in ("entry_type", "region_group"):
        x[c] = d[c].astype("category")
    y = d.returned.astype(int)
    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.04, num_leaves=24, min_child_samples=60, subsample=0.8, colsample_bytree=0.8, verbose=-1)
    auc = cross_val_score(model, x, y, cv=5, scoring="roc_auc").mean()
    base = cross_val_score(lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=16, verbose=-1), x.drop(columns=["starts", "types", "span_days"]), y, cv=5, scoring="roc_auc").mean()
    model.fit(x, y)
    sv = shap.TreeExplainer(model).shap_values(x)
    sv = sv[1] if isinstance(sv, list) else sv
    names = {"entry_riders": "размер первой гонки", "entry_month": "месяц первой гонки", "next_gap": "дней до следующей гонки в регионе",
             "events_60d": "гонок в регионе за 60 дней после первой", "events_left": "гонок до конца сезона", "series_events": "этапов в серии",
             "entry_women_share": "доля женщин на первой гонке", "female": "пол", "starts": "стартов в первом сезоне", "types": "число дисциплин",
             "span_days": "длина первого сезона, дней", "entry_type": "дисциплина", "region_group": "регион"}
    imp = pd.DataFrame({"feature": [names[c] for c in x.columns], "mean_abs_shap": np.abs(sv).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(OUT / "shap_importance.csv", index=False)
    return dict(auc_full=float(auc), auc_entry_only=float(base), n=len(d)), imp


def series_ranking(n):
    """Какие серии лучше всего удерживают новичков: эмпирический Байес (бета-биномиальное сглаживание)."""
    d = n[~n.entry_type.isin(["track"])].copy()
    g = d.groupby("entry_series").agg(newcomers=("key", "size"), ret=("returned", "sum"), second_start=("two", "mean"),
                                      type=("entry_type", lambda s: TYPE_RU[s.mode().iat[0]]), region=("entry_region", "first"),
                                      field=("entry_riders", "median"), organizer=("entry_organizer", "first")).reset_index()
    big = g[g.newcomers >= 50]
    m = g.ret.sum() / g.newcomers.sum()
    v = (big.ret / big.newcomers).var()
    k = m * (1 - m) / v - 1
    g["returned_raw"] = g.ret / g.newcomers
    g["returned_eb"] = (g.ret + m * k) / (g.newcomers + k)
    g = g[g.newcomers >= 60].sort_values("returned_eb", ascending=False).drop(columns="ret")
    g.to_csv(OUT / "series_ranking.csv", index=False)
    o = n.dropna(subset=["entry_organizer"]).groupby("entry_organizer").agg(newcomers=("key", "size"), ret=("returned", "sum"), second_start=("two", "mean")).reset_index()
    o["returned_eb"] = (o.ret + m * k) / (o.newcomers + k)
    o[o.newcomers >= 100].drop(columns="ret").sort_values("returned_eb", ascending=False).to_csv(OUT / "organizer_ranking.csv", index=False)
    return dict(prior=float(m), strength=float(k)), g


def gravel(s):
    """Откуда приходят в гревел и куда уходят из других дисциплин."""
    known = s[(s.type != "unknown") & s.year.between(2023, 2026)]
    main = known.groupby(["key", "year"]).type.agg(lambda x: x.mode().iat[0]).reset_index()
    nxt = main.merge(main.assign(year=main.year - 1), on=["key", "year"], how="left", suffixes=("", "_next"))
    nxt = nxt[nxt.year < 2026]
    nxt["type_next"] = nxt.type_next.fillna("не вернулся")
    tr = pd.crosstab(nxt.type.map(TYPE_RU), nxt.type_next.map(lambda v: TYPE_RU.get(v, v)), normalize="index")
    tr.to_csv(OUT / "discipline_transitions.csv")
    first_gravel = s[s.type == "gravel"].sort_values("date").drop_duplicates("key")[["key", "date", "year"]].rename(columns={"date": "gdate", "year": "gyear"})
    prev = s.merge(first_gravel, on="key")
    prev = prev[(prev.date < prev.gdate) & (prev.type != "unknown")]
    origin = prev.groupby("key").type.agg(lambda x: x.mode().iat[0])
    first_gravel["origin"] = first_gravel.key.map(origin).map(TYPE_RU).fillna("первая гонка в жизни")
    og = first_gravel[first_gravel.gyear >= 2024].groupby(["gyear", "origin"]).size().unstack(fill_value=0)
    og.to_csv(OUT / "gravel_origin.csv")
    t26 = s[s.year == 2026].groupby("key").type.agg(set)
    gk = t26[t26.apply(lambda x: "gravel" in x)]
    return tr, og, dict(gravel_riders_2026=len(gk), only_gravel=float(gk.apply(lambda x: x == {"gravel"}).mean()),
                        also_road=float(gk.apply(lambda x: "road" in x).mean()), also_mtb=float(gk.apply(lambda x: "mtb" in x).mean()))


def survival(p):
    """Сколько сезонов гонщик остаётся в гонках: кривые Каплана — Мейера по дисциплине первого сезона."""
    first = p[p.is_new & p.year.between(2023, 2025)][["key", "year", "entry_type", "starts"]]
    last = p.groupby("key").year.max().rename("last_year")
    d = first.merge(last, on="key")
    d["seasons"] = d.last_year - d.year + 1
    d["churned"] = d.last_year < 2026
    rows = []
    for name, grp in [("все", d)] + [(TYPE_RU[t], d[d.entry_type == t]) for t in ("road", "mtb", "gravel")] + \
                     [("1 старт в первом сезоне", d[d.starts == 1]), ("2 старта", d[d.starts == 2]), ("3 и больше", d[d.starts >= 3])]:
        km = KaplanMeierFitter().fit(grp.seasons, grp.churned)
        for t in (1, 2, 3):
            rows.append(dict(group=name, seasons=t, alive=float(km.predict(t)), n=len(grp)))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "survival.csv", index=False)
    return out


def network(s, ev):
    """Сеть гонок сезона 2025–2026: ребро — общие участники. Сообщества ищем методом Лувена."""
    d = s[s.year.isin([2025, 2026])]
    series_size = d.groupby("series").key.nunique()
    keep = series_size[series_size >= 60].index
    d = d[d.series.isin(keep)].drop_duplicates(["key", "series"])
    pairs = d.merge(d, on="key")
    pairs = pairs[pairs.series_x < pairs.series_y].groupby(["series_x", "series_y"]).size().rename("shared").reset_index()
    pairs["jaccard"] = pairs.shared / (pairs.series_x.map(series_size) + pairs.series_y.map(series_size) - pairs.shared)
    edges = pairs[(pairs.shared >= 15) & (pairs.jaccard >= 0.03)]
    g = nx.Graph()
    g.add_weighted_edges_from(edges[["series_x", "series_y", "jaccard"]].itertuples(index=False))
    part = louvain.best_partition(g, weight="weight", random_state=1)
    meta = ev[ev.year.isin([2025, 2026])].groupby("series").agg(type=("type", lambda x: TYPE_RU[x.mode().iat[0]]), region=("region", "first"), name=("name", "first"))
    nodes = pd.DataFrame({"series": list(part), "community": list(part.values())})
    nodes["riders"] = nodes.series.map(series_size)
    nodes["degree"] = nodes.series.map(dict(g.degree(weight="weight")))
    nodes = nodes.join(meta, on="series")
    nodes.to_csv(OUT / "network_nodes.csv", index=False)
    edges.to_csv(OUT / "network_edges.csv", index=False)
    comm = nodes.groupby("community").agg(series=("series", "size"), riders=("riders", "sum"),
                                          types=("type", lambda x: ", ".join(f"{k} {v}" for k, v in x.value_counts().head(3).items())),
                                          regions=("region", lambda x: ", ".join(f"{k} {v}" for k, v in x.fillna("нет данных").value_counts().head(2).items())),
                                          biggest=("name", lambda x: str(x.iloc[nodes.loc[x.index, "riders"].argmax()])[:60])).sort_values("riders", ascending=False)
    comm.to_csv(OUT / "network_communities.csv")
    return dict(nodes=g.number_of_nodes(), edges=g.number_of_edges(), communities=len(comm),
                modularity=float(louvain.modularity(part, g, weight="weight"))), comm


def regions(ev):
    """Потенциал регионов: сколько гонок ожидаем при таком населении и беговой базе и сколько есть на деле."""
    races = pd.read_csv(RVVR / "outputs/summary_regions_2023_2025.csv").rename(columns={"reg": "region", "median_n": "median_field"})
    run = pd.read_csv(RVVR / "outputs/region_summary.csv").rename(columns={"agg": "region"})[["region", "orgeo_running_reg_per_100k_year", "orgeo_reg_athletics"]]
    alias = {"Москва и МО": "Москва и Московская область", "СПб и ЛО": "Санкт-Петербург и Ленинградская область"}
    races["region_full"] = races.region.replace(alias)
    # гонки из новых источников (MyFinish, results.zone, sportident), которых нет в каталоге rvvr: сверяем по региону и дате
    cat = pd.read_parquet(RVVR / "data/processed/races_all.parquet")
    cat = cat[cat.year.between(2023, 2025)]
    cat["region_full"] = cat.region.replace({"Москва": alias["Москва и МО"], "Московская область": alias["Москва и МО"],
                                             "Санкт-Петербург": alias["СПб и ЛО"], "Ленинградская область": alias["СПб и ЛО"]})
    known = set(zip(cat.region_full, pd.to_datetime(cat.date).dt.date))
    new = ev[ev.src.isin(["myfinish", "results.zone", "sportident"]) & ev.year.between(2023, 2025) & ev.region.notna()]
    new = new[[(r, d.date()) not in known for r, d in zip(new.region, new.date)]].drop_duplicates(["region", "date"])
    add = new.groupby("region").size()
    races["races_catalog"] = races.races
    races["races"] = races.races + races.region_full.map(add).fillna(0).astype(int)
    races["races_per_mln_year"] = races.races / 3 / races.population * 1e6
    e25 = ev[ev.year.between(2023, 2025) & ev.region.notna() & ev.date.dt.month.between(4, 10)]
    gap = e25.groupby(["region", "year"]).date.apply(lambda d: d.drop_duplicates().sort_values().diff().dt.days.median()).groupby("region").median().rename("median_gap_days")
    d = races.merge(gap, left_on="region_full", right_index=True, how="left").merge(run, on="region", how="left")
    d = d[d.population > 0]
    d["log_pop"], d["log_races"] = np.log(d.population), np.log(d.races + 1)
    fit = smf.ols("log_races ~ log_pop", d).fit()
    d["expected_races"] = np.exp(fit.predict(d)) - 1
    d["ratio"] = d.races / d.expected_races
    d["volga"] = d.region.isin(VOLGA)
    cols = ["region", "population", "races", "expected_races", "ratio", "median_field", "races_per_mln_year", "median_gap_days", "volga"]
    d[cols].sort_values("ratio", ascending=False).to_csv(OUT / "regions.csv", index=False)
    return dict(elasticity=float(fit.params["log_pop"]), r2=float(fit.rsquared), n=len(d)), d[cols].sort_values("ratio", ascending=False)


def scenario(n, newcomers_per_season=300):
    """Что даёт плотный календарь: когортная модель на три сезона.

    Доли возврата взяты из данных: новичок с одним стартом, новичок с двумя и больше, гонщик второго сезона.
    """
    p_one = float(n[~n.two].returned.mean())
    p_two = float(n[n.two].returned.mean())
    keep = 0.62  # возврат гонщика, который уже пережил первый сезон
    rows = []
    for name, share_two in (("редкий календарь", 0.10), ("как в среднем по выборке", float(n.two.mean())), ("плотный календарь", 0.30),
                            ("цель: каждый второй новичок со вторым стартом", 0.50)):
        ret = share_two * p_two + (1 - share_two) * p_one
        s1 = newcomers_per_season
        s2 = newcomers_per_season + s1 * ret
        s3 = newcomers_per_season + newcomers_per_season * ret + s1 * ret * keep
        rows.append(dict(scenario=name, share_second=share_two, newcomer_return=ret, season1=s1, season2=s2, season3=s3))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "scenario.csv", index=False)
    return out, dict(p_one=p_one, p_two=p_two, keep=keep)


def main():
    OUT.mkdir(exist_ok=True)
    s = pd.read_parquet(DATA / "starts.parquet")
    ev = pd.read_parquet(DATA / "events.parquet")
    p = pd.read_parquet(DATA / "rider_seasons.parquet")
    n = pd.read_parquet(DATA / "newcomers.parquet")
    res = {"base": dict(starts=len(s), riders=int(s.key.nunique()), events=len(ev), newcomers_2023_2025=len(n),
                        returned=float(n.returned.mean()), second_start=float(n.two.mean()))}
    cal, groups = calendar(n)
    print(cal.round(3).to_string(), "\n", groups.round(3).to_string())
    res["drivers"], imp = drivers(n)
    print(res["drivers"], "\n", imp.round(4).to_string())
    res["series"], rank = series_ranking(n)
    print(rank.head(14).round(3).to_string(), "\n", rank.tail(8).round(3).to_string())
    tr, og, res["gravel"] = gravel(s)
    print((tr * 100).round(1).to_string(), "\n", og.to_string(), res["gravel"])
    print(survival(p).pivot(index="group", columns="seasons", values="alive").round(3).to_string())
    res["network"], comm = network(s, ev)
    print(res["network"], "\n", comm.head(12).to_string())
    res["regions"], reg = regions(ev)
    print(res["regions"], "\n", reg.head(12).round(2).to_string(), "\n", reg[reg.volga].round(2).to_string(), "\n", reg.tail(10).round(2).to_string())
    sc, res["scenario"] = scenario(n)
    print(sc.round(2).to_string(), res["scenario"])
    res["causal"] = json.load(open(OUT / "causal.json"))
    json.dump(res, open(OUT / "results.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
