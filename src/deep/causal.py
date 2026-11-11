"""Причинный эффект второго старта на возврат в следующем сезоне.

Три оценки: наивная разница, дважды устойчивая оценка (AIPW с перекрёстной подгонкой)
и инструментальная переменная — была ли в регионе следующая гонка в ближайшие 30 дней.
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold
import statsmodels.formula.api as smf

from deep.base import DATA, ROOT

X_NUM = ["entry_riders", "entry_month", "events_60d", "events_left", "series_events", "entry_women_share", "female", "year"]
X_CAT = ["entry_type", "region_group"]


def design(d):
    x = d[X_NUM].copy()
    x["next_gap"] = d.next_gap.fillna(999)
    return pd.concat([x, pd.get_dummies(d[X_CAT], dtype=float)], axis=1)


def aipw(d, treat="two_60d", outcome="returned", folds=5, seed=0):
    x, t, y = design(d).to_numpy(), d[treat].to_numpy().astype(int), d[outcome].to_numpy().astype(int)
    ps, m1, m0 = np.zeros(len(d)), np.zeros(len(d)), np.zeros(len(d))
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(x):
        ps[te] = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300).fit(x[tr], t[tr]).predict_proba(x[te])[:, 1]
        for arm, out in ((1, m1), (0, m0)):
            idx = tr[t[tr] == arm]
            out[te] = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300).fit(x[idx], y[idx]).predict_proba(x[te])[:, 1]
    ps = ps.clip(0.02, 0.98)
    psi = m1 - m0 + t * (y - m1) / ps - (1 - t) * (y - m0) / (1 - ps)
    return dict(ate=psi.mean(), se=psi.std(ddof=1) / np.sqrt(len(psi)), overlap=[float(np.quantile(ps, q)) for q in (0.01, 0.5, 0.99)])


def main():
    d = pd.read_parquet(DATA / "newcomers.parquet")
    d = d[d.entry_region.notna()].copy()
    # у новичка должно оставаться время на второй старт: первая гонка не позже 31 августа
    d = d[d.entry_month.between(4, 8)]
    res = {"n": len(d), "share_second_60d": float(d.two_60d.mean())}
    a, b = d[d.two_60d], d[~d.two_60d]
    res["naive"] = dict(treated=float(a.returned.mean()), control=float(b.returned.mean()), diff=float(a.returned.mean() - b.returned.mean()))
    r = aipw(d)
    res["aipw"] = dict(ate=r["ate"], lo=r["ate"] - 1.96 * r["se"], hi=r["ate"] + 1.96 * r["se"], overlap=r["overlap"])
    # инструмент: следующая гонка той же дисциплины в регионе не позже чем через 30 дней
    d["z"] = d.next_gap.le(30).astype(int)
    d["t"], d["y"] = d.two_60d.astype(int), d.returned.astype(int)
    d["cell"] = d.entry_region + "|" + d.entry_type + "|" + d.year.astype(str)
    d = d[d.groupby("cell").z.transform("nunique") > 1]  # в ячейке должны быть и близкие, и далёкие гонки
    fs = smf.ols("t ~ z + C(cell) + C(entry_month) + np.log(entry_riders)", d).fit(cov_type="cluster", cov_kwds={"groups": d.entry_event})
    rf = smf.ols("y ~ z + C(cell) + C(entry_month) + np.log(entry_riders)", d).fit(cov_type="cluster", cov_kwds={"groups": d.entry_event})
    res["iv"] = dict(n=len(d), cells=int(d.cell.nunique()), first_stage=float(fs.params["z"]), first_stage_se=float(fs.bse["z"]),
                     reduced_form=float(rf.params["z"]), reduced_form_se=float(rf.bse["z"]),
                     wald=float(rf.params["z"] / fs.params["z"]), f_stat=float((fs.params["z"] / fs.bse["z"]) ** 2))
    (ROOT / "out").mkdir(exist_ok=True)
    json.dump(res, open(ROOT / "out" / "causal.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
