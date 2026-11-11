"""results.zone: полный каталог событий и протоколы велогонок.

Каталог: главная страница, ?page=N, список в window.general.events (25 событий на страницу).
Протокол: /<slug>/races/<id>/results.json?page=N (50 строк на страницу, сервер ограничивает частоту — паузы).
-> data/raw/resultszone/catalog.jsonl, data/raw/resultszone/results.jsonl (одна строка на заезд). Персональные данные остаются локально.
"""
import json
import time
from pathlib import Path

import httpx

OUT = Path(__file__).resolve().parents[3] / "data" / "raw" / "resultszone"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) research-crawler"}
KEEP = ["name", "gender", "city", "country", "team", "club", "category", "cluster", "status", "result_time", "rank_abs"]
PAUSE = 1.3


def get(c, url):
    for a in range(6):
        time.sleep(PAUSE * (a + 1))
        try:
            r = c.get(url)
            if r.status_code == 200 and r.text.strip():
                return r.text
            if r.status_code == 404:
                return ""
        except httpx.HTTPError:
            pass
    return ""


def catalog(c):
    path = OUT / "catalog.jsonl"
    if path.exists():
        return [json.loads(l) for l in path.open()]
    out, seen, page = [], set(), 1
    while True:
        h = get(c, f"https://results.zone/?page={page}")
        i = h.find("general.events=")
        arr = json.JSONDecoder().raw_decode(h[i + 15:])[0] if i >= 0 else []
        new = [e for e in arr if e["slug"] not in seen]
        if not new:
            break
        for e in new:
            seen.add(e["slug"])
            out.append({k: e.get(k) for k in ("slug", "name", "series", "date_start", "sport", "timezone")}
                       | {"races": [{k: r.get(k) for k in ("id", "name", "distance", "start_time", "race_type")} for r in e.get("races") or []]})
        if page % 10 == 0:
            print("каталог, страница", page, len(out), flush=True)
        page += 1
    with path.open("w") as f:
        for e in out:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return out


def crawl():
    OUT.mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers=UA, timeout=40, follow_redirects=True) as c:
        cat = catalog(c)
        bike = [e for e in cat if e["sport"] == "cycling"]
        path = OUT / "results.jsonl"
        done = {json.loads(l)["race_id"] for l in path.open()} if path.exists() else set()
        print("в каталоге", len(cat), "вело", len(bike), "заездов готово", len(done), flush=True)
        with path.open("a") as f:
            for n, e in enumerate(bike, 1):
                for r in e["races"]:
                    if r["id"] in done:
                        continue
                    items, page = [], 1
                    while True:
                        t = get(c, f"https://results.zone/{e['slug']}/races/{r['id']}/results.json?page={page}")
                        try:
                            d = json.loads(t) if t else {}
                        except ValueError:
                            d = {}
                        items += [{k: it.get(k) for k in KEEP} for it in d.get("items") or []]
                        pi = d.get("page_info") or {}
                        if not d.get("items") or page >= (pi.get("totalPages") or 1):
                            break
                        page += 1
                    f.write(json.dumps(dict(slug=e["slug"], event=e["name"], series=e["series"], date=e["date_start"], race_id=r["id"], race=r["name"],
                                            distance=r["distance"], rows=items), ensure_ascii=False) + "\n")
                    f.flush()
                if n % 20 == 0:
                    print(n, len(bike), flush=True)
    print("готово", flush=True)


if __name__ == "__main__":
    crawl()
