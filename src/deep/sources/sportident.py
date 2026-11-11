"""sportident.online: списки заявленных на велособытия (в основном Самарская область).

Идентификаторы событий берём из уже собранного rvvr/data/raw/sportident/events.jsonl (только чтение).
-> data/raw/sportident/participants.jsonl. Персональные данные остаются локально.
"""
import json
import re
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "raw" / "sportident"
SRC = ROOT.parent / "rvvr" / "data" / "raw" / "sportident" / "events.jsonl"


def crawl():
    OUT.mkdir(parents=True, exist_ok=True)
    ev = [json.loads(l) for l in SRC.open()]
    with httpx.Client(headers={"User-Agent": "Mozilla/5.0 research-crawler"}, timeout=40) as c, open(OUT / "participants.jsonl", "w") as f:
        for e in ev:
            h = c.get(f"https://sportident.online/entry/?id={e['id']}&lis").text
            time.sleep(1)
            rows = []
            # список разбит по категориям: каждая открывается своей ссылкой
            for g, gname in re.findall(r"href='\?lis&id=\d+&g=(\d+)'>([^<(]+)\(", h):
                part = c.get(f"https://sportident.online/entry/?lis&id={e['id']}&g={g}").text
                time.sleep(1)
                for m in re.finditer(r"<tr class='hoverRow'><td[^>]*>\d+</td><td[^>]*>([^<]*)</td><td[^>]*>(\d{4})?</td><td[^>]*>[^<]*</td><td[^>]*>([^<]*)</td><td[^>]*>([^<]*)</td>", part):
                    rows.append(dict(group=gname.strip(), name=m[1].strip(), birth=int(m[2]) if m[2] else None, club=m[3].strip(), city=m[4].strip()))
            f.write(json.dumps(dict(id=e["id"], title=re.sub(r"^Заявиться Список участников Информация\s*", "", e["title"] or ""), date=e["date"],
                                    place=e["place"], rows=rows), ensure_ascii=False) + "\n")
            print(e["id"], e["date"], len(rows), flush=True)


if __name__ == "__main__":
    crawl()
