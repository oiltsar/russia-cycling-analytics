"""Интерактивный дашборд: одна HTML-страница на Plotly. Запуск: python dashboard/build_dashboard.py -> docs/index.html"""
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
INK, INK2, GREY, LIGHT, BG, ACCENT, SECOND = "#151917", "#4f5753", "#b9beba", "#e1e4df", "#fcfcfb", "#e8590c", "#1f5c4a"
res = json.load(open(OUT / "results.json"))


def layout(fig, height=380, **kw):
    kw.setdefault("margin", dict(l=10, r=20, t=10, b=10))
    fig.update_layout(height=height, paper_bgcolor=BG, plot_bgcolor=BG, showlegend=False,
                      font=dict(family="Golos Text, PT Sans, Arial, sans-serif", size=13, color=INK), hoverlabel=dict(bgcolor="white", font_size=13), **kw)
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=LIGHT, tickfont=dict(color=INK2))
    fig.update_yaxes(showgrid=False, zeroline=False, linecolor=LIGHT, tickfont=dict(color=INK2))
    return fig


def html(fig):
    return fig.to_html(full_html=False, include_plotlyjs=False, config=dict(displayModeBar=False, responsive=True))


def fig_calendar():
    c = pd.read_csv(OUT / "calendar_gap.csv")
    fig = go.Figure()
    for col, vis in (("second_start", True), ("returned", False)):
        v = c[col] * 100
        fig.add_bar(x=c.gap, y=v, visible=vis, marker_color=[ACCENT if i in (0, 4) else GREY for i in range(len(c))], text=[f"{x:.0f}%" for x in v],
                    textposition="outside", customdata=c.newcomers, hovertemplate="%{x}<br>%{y:.1f}%<br>новичков: %{customdata:,}<extra></extra>")
    fig.update_layout(updatemenus=[dict(type="buttons", direction="right", x=0, y=1.18, xanchor="left", showactive=True, bgcolor="white", bordercolor=LIGHT,
                                        buttons=[dict(label="Второй старт в сезоне", method="update", args=[{"visible": [True, False]}]),
                                                 dict(label="Возврат на следующий год", method="update", args=[{"visible": [False, True]}])])])
    fig.update_yaxes(visible=False, range=[0, 50])
    return layout(fig, 400, margin=dict(l=10, r=20, t=60, b=10))


def fig_month():
    m = pd.read_csv(OUT / "entry_month.csv").query("4 <= entry_month <= 10")
    names = ["апрель", "май", "июнь", "июль", "август", "сентябрь", "октябрь"]
    fig = go.Figure(go.Bar(x=names, y=m.second_start * 100, marker_color=[ACCENT if i in (1, 5) else GREY for i in range(7)], text=[f"{v:.0f}%" for v in m.second_start * 100],
                           textposition="outside", customdata=np.c_[m.newcomers, m.returned * 100], hovertemplate="%{x}<br>второй старт: %{y:.0f}%<br>вернулись: %{customdata[1]:.0f}%<br>новичков: %{customdata[0]:,}<extra></extra>"))
    fig.update_yaxes(visible=False, range=[0, 55])
    return layout(fig, 340)


def fig_regions():
    r = pd.read_csv(OUT / "regions.csv").sort_values("ratio")
    color = [ACCENT if v < 0.5 else SECOND if v > 2 else GREY for v in r.ratio]
    fig = go.Figure(go.Bar(y=r.region, x=r.ratio, orientation="h", marker_color=color, customdata=np.c_[r.races, r.expected_races, r.population / 1e6, r.median_field.fillna(0)],
                           hovertemplate="<b>%{y}</b><br>найдено гонок: %{customdata[0]:.0f}<br>ожидается при таком населении: %{customdata[1]:.0f}<br>население: %{customdata[2]:.1f} млн<br>медиана участников: %{customdata[3]:.0f}<extra></extra>"))
    fig.add_vline(x=1, line=dict(color=INK2, width=1, dash="dot"))
    fig.update_xaxes(type="log", title="гонок найдено к ожидаемому числу (лог. шкала)", tickvals=[0.1, 0.25, 0.5, 1, 2, 5, 10, 20], ticktext=["×0,1", "×0,25", "×0,5", "×1", "×2", "×5", "×10", "×20"])
    return layout(fig, 1150)


def fig_series():
    s = pd.read_csv(OUT / "series_ranking.csv")
    s["label"] = s.organizer.fillna(s.entry_series)
    fig = go.Figure()
    types = ["все дисциплины"] + sorted(s.type.unique())
    for i, t in enumerate(types):
        d = s if i == 0 else s[s.type == t]
        fig.add_scatter(x=d.second_start * 100, y=d.returned_eb * 100, mode="markers", visible=i == 0,
                        marker=dict(size=np.sqrt(d.newcomers) * 1.6 + 4, color=[ACCENT if x == "гревел" else GREY for x in d.type], opacity=0.8, line=dict(color="white", width=1)),
                        customdata=np.c_[d.label, d.type, d.region.fillna("регион не указан"), d.newcomers, d.field],
                        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]} · %{customdata[2]}<br>новичков: %{customdata[3]}<br>второй старт: %{x:.0f}%<br>вернулись: %{y:.0f}%<br>участников на гонке: %{customdata[4]:.0f}<extra></extra>")
    fig.update_layout(updatemenus=[dict(x=0, y=1.14, xanchor="left", bgcolor="white", bordercolor=LIGHT,
                                        buttons=[dict(label=t, method="update", args=[{"visible": [j == i for j in range(len(types))]}]) for i, t in enumerate(types)])])
    fig.update_xaxes(title="новичков со вторым стартом в сезоне, %", showgrid=True, gridcolor=LIGHT)
    fig.update_yaxes(title="новичков вернулось на следующий год, %", showgrid=True, gridcolor=LIGHT)
    return layout(fig, 520, margin=dict(l=10, r=20, t=60, b=10))


def fig_network():
    nodes, edges = pd.read_csv(OUT / "network_nodes.csv"), pd.read_csv(OUT / "network_edges.csv")
    g = nx.Graph(); g.add_weighted_edges_from(edges[["series_x", "series_y", "jaccard"]].itertuples(index=False))
    pos = nx.spring_layout(g, weight="weight", k=0.32, seed=4, iterations=120)
    nd = nodes.set_index(nodes.series.astype(str))
    gravel = nodes[nodes.type == "гревел"].community.mode().iat[0]
    ex, ey = [], []
    for a, b in g.edges():
        ex += [pos[a][0], pos[b][0], None]; ey += [pos[a][1], pos[b][1], None]
    fig = go.Figure(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color=LIGHT, width=0.6), hoverinfo="skip"))
    keys = [str(n) for n in g]
    fig.add_scatter(x=[pos[n][0] for n in g], y=[pos[n][1] for n in g], mode="markers",
                    marker=dict(size=[6 + np.sqrt(nd.riders.get(k, 60)) / 2.2 for k in keys], color=GREY, line=dict(color="white", width=0.6)),
                    customdata=np.c_[[str(nd.name.get(k)) if pd.notna(nd.name.get(k)) else k for k in keys], [nd.type.get(k, "") for k in keys],
                                     [nd.region.get(k) if pd.notna(nd.region.get(k)) else "регион не указан" for k in keys], [nd.riders.get(k, 0) for k in keys]],
                    hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]} · %{customdata[2]}<br>гонщиков: %{customdata[3]}<extra></extra>")
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    return layout(fig, 620)


def fig_survival():
    s = pd.read_csv(OUT / "survival.csv")
    fig = go.Figure()
    for name, label, color, w in (("1 старт в первом сезоне", "один старт", GREY, 2.5), ("2 старта", "два старта", INK2, 2.5), ("3 и больше", "три и больше", ACCENT, 4)):
        d = s[s.group == name]
        y = [100] + list(d.alive * 100)
        fig.add_scatter(x=["первый сезон", "через год", "через два", "через три"], y=y, mode="lines+markers+text", line=dict(color=color, width=w), text=[""] * 3 + [f"  {label} {y[-1]:.0f}%"],
                        textposition="middle right", textfont=dict(color=INK), hovertemplate=label + ": %{y:.0f}%<extra></extra>", cliponaxis=False)
    fig.update_yaxes(visible=False, range=[0, 105]); fig.update_xaxes(range=[-0.2, 4.3])
    return layout(fig, 400)


def fig_origin():
    o = pd.read_csv(OUT / "gravel_origin.csv").set_index("gyear")
    sh = o.div(o.sum(axis=1), axis=0) * 100
    fig = go.Figure()
    for col, color in (("первая гонка в жизни", ACCENT), ("шоссе", GREY), ("МТБ", INK2), ("велокросс", LIGHT)):
        fig.add_bar(y=sh.index.astype(str), x=sh[col], orientation="h", marker_color=color, text=[f"{col} {v:.0f}%" if v > 9 else "" for v in sh[col]], textposition="inside",
                    insidetextanchor="middle", customdata=o[col], hovertemplate=col + ": %{x:.0f}% (%{customdata} чел.)<extra></extra>")
    fig.update_layout(barmode="stack"); fig.update_xaxes(visible=False); fig.update_yaxes(autorange="reversed")
    return layout(fig, 260)


def fig_transitions():
    t = pd.read_csv(OUT / "discipline_transitions.csv", index_col=0) * 100
    cols = ["шоссе", "МТБ", "гревел", "велокросс", "трек", "не вернулся"]
    t = t.loc[["шоссе", "МТБ", "гревел", "велокросс", "трек"], cols]
    fig = go.Figure(go.Heatmap(z=t.values, x=cols, y=t.index, colorscale=[[0, BG], [1, ACCENT]], showscale=False, text=[[f"{v:.0f}%" for v in row] for row in t.values],
                               texttemplate="%{text}", hovertemplate="из «%{y}» в «%{x}»: %{z:.1f}%<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title="дисциплина в этом году"); fig.update_xaxes(side="top", title="дисциплина на следующий год")
    return layout(fig, 330)


b, ca = res["base"], res["causal"]
sections = [
    ("Удержание", "Чем больше стартов в первом сезоне, тем дольше гонщик остаётся", "Доля новичков, которые продолжают выходить на старт, по числу стартов в первом сезоне (кривые Каплана — Мейера).", fig_survival()),
    ("Дисциплины", "Между дисциплинами почти не переходят", "Куда гонщик уходит на следующий год, в процентах от строки. 2023–2025 годы.", fig_transitions()),
    ("Сеть гонок", "Гонки России — это острова", f"{res['network']['nodes']} серии 2025–2026 годов, связь — общие участники. Наведите на точку.", fig_network()),
]
kpis = [(f"{b['starts'] // 1000} тыс.", "стартов, 2023–2026"), (f"{b['riders'] / 1000:.1f}".replace(".", ",") + " тыс.", "гонщиков"), (f"{b['events']:,}".replace(",", " "), "событий"),
        (f"{b['returned'] * 100:.0f}%", "новичков возвращаются на следующий год"), (f"+{ca['aipw']['ate'] * 100:.0f} п.п.", "к возврату даёт второй старт за 60 дней")]
body = "".join(f'<section><div class="eyebrow">{e}</div><h2>{t}</h2><p class="sub">{s}</p>{html(f)}</section>' for e, t, s, f in sections)
page = f"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Велогонки России: что удерживает гонщиков</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link href="https://fonts.googleapis.com/css2?family=Golos+Text:wght@400;600&family=PT+Sans+Narrow:wght@700&family=IBM+Plex+Mono&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root {{ --bg:#f4f5f2; --surface:{BG}; --ink:{INK}; --ink2:{INK2}; --line:{LIGHT}; --accent:{ACCENT}; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--ink); font-family:"Golos Text","PT Sans",Arial,sans-serif; }}
main {{ max-width:1080px; margin:0 auto; padding:40px 16px 80px; }}
h1 {{ font-family:"PT Sans Narrow","Arial Narrow",sans-serif; text-transform:uppercase; font-size:clamp(38px,7vw,76px); line-height:.95; margin:8px 0 18px; }}
h1 span {{ color:var(--accent); }} h2 {{ font-family:"PT Sans Narrow","Arial Narrow",sans-serif; text-transform:uppercase; font-size:clamp(24px,4vw,34px); line-height:1; margin:4px 0 8px; }}
.eyebrow {{ font-family:"IBM Plex Mono",monospace; font-size:12px; letter-spacing:.08em; text-transform:uppercase; color:#848b87; }}
.lead, .sub {{ color:var(--ink2); line-height:1.45; max-width:760px; }} .lead {{ font-size:18px; }} .sub {{ font-size:15px; margin:0 0 10px; }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin:28px 0 8px; }}
.kpi {{ background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:16px; }}
.kpi b {{ display:block; font-family:"PT Sans Narrow",sans-serif; font-size:40px; line-height:1; }} .kpi span {{ font-size:13.5px; color:var(--ink2); }}
section {{ background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:22px 18px 10px; margin-top:22px; overflow:hidden; }}
footer {{ margin-top:28px; font-size:13.5px; color:var(--ink2); line-height:1.5; }} a {{ color:var(--ink); }}
</style></head><body><main>
<div class="eyebrow">Исследование · открытые протоколы 2023–2026</div>
<h1>Велогонки России:<br><span>что удерживает гонщиков</span></h1>
<p class="lead">Почему одни участники любительских велогонок возвращаются на следующий сезон, а другие нет. Проект в работе: здесь показана обобщённая часть результатов.</p>
<div class="kpis">{"".join(f'<div class="kpi"><b>{v}</b><span>{l}</span></div>' for v, l in kpis)}</div>
{body}
<footer>Данные: открытые протоколы и календари гонок (fatracing.ru, results.zone, RussiaRunning, Orgeo). Учтены только гонки с открытыми списками участников, все числа — нижняя граница; 2026 год неполный. На странице только обезличенные агрегаты.
Автор: Александр Царегородский · <a href="https://github.com/oiltsar/russia-cycling-analytics">код и методы на GitHub</a></footer>
</main></body></html>"""
(ROOT / "docs").mkdir(exist_ok=True)
(ROOT / "docs" / "index.html").write_text(page, encoding="utf-8")
print("ok", len(page) // 1024, "KB")
