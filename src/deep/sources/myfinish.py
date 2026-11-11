"""MyFinish.info (AltTiming): каталог соревнований и открытые протоколы.

Каталог: php/ajax_gate.php?do=compUsrList&year=<год>. Протокол: php/gate_online.php?evid=<id>&d=2 (категории), d=10+<категория> (строки).
-> data/raw/myfinish/catalog.jsonl, data/raw/myfinish/results.jsonl (одна строка на событие). Персональные данные остаются локально.
"""
import json
import re
import time
from pathlib import Path

import httpx

OUT = Path(__file__).resolve().parents[3] / "data" / "raw" / "myfinish"
BIKE = re.compile(r"вело|mtb|мтб|bike|байк|gran ?fondo|гран ?фондо|критериум|cyclo|кросс-кантри|\bxc[omc]?\b|гревел|gravel|eurasia hard", re.I)
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) research-crawler"}


def fetch(c, url):
    for a in range(4):
        try:
            r = c.get(url)
            if r.status_code == 200:
                return r.json() if r.text.strip() else {}
        except (httpx.HTTPError, ValueError):
            pass
        time.sleep(2 * (a + 1))
    return None


def crawl(years=range(2014, 2027), delay=0.6):
    OUT.mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers=UA, timeout=40) as c:
        cat = []
        for y in years:
            d = fetch(c, f"https://myfinish.info/php/ajax_gate.php?do=compUsrList&year={y}") or {}
            cat += d.get("comp") or []
            time.sleep(delay)
        with open(OUT / "catalog.jsonl", "w") as f:
            for x in cat:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        todo = [x for x in cat if BIKE.search(x["name"])]
        path = OUT / "results.jsonl"
        done = {json.loads(l)["id"] for l in path.open()} if path.exists() else set()
        print("в каталоге", len(cat), "вело", len(todo), "уже есть", len(done), flush=True)
        with path.open("a") as f:
            for i, x in enumerate(todo, 1):
                if x["id"] in done:
                    continue
                head = fetch(c, f"https://myfinish.info/php/gate_online.php?evid={x['id']}&d=2") or {}
                time.sleep(delay)
                groups = head.get("g") or []
                real = [g for g in groups if g.get("name") != "Сводная"] or groups
                out = []
                for g in real:
                    res = fetch(c, f"https://myfinish.info/php/gate_online.php?evid={x['id']}&d={10 + int(g['id'])}") or {}
                    time.sleep(delay)
                    rows = [dict(name=r.get("i"), club=r.get("k"), home=r.get("reg"), birth=r.get("y"), status=r.get("status"), result=r.get("r"))
                            for r in res.get("r") or []]
                    out.append(dict(group=g.get("name"), descr=g.get("descr"), rows=rows))
                f.write(json.dumps(dict(id=x["id"], ts=x["date"], name=x["name"], org=x.get("org"), place=x.get("place"), date=head.get("d"), groups=out),
                                   ensure_ascii=False) + "\n")
                f.flush()
                if i % 25 == 0:
                    print(i, len(todo), flush=True)
    print("готово", flush=True)


if __name__ == "__main__":
    crawl()
