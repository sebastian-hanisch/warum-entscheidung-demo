"""Rechnet das Raster vor und friert es ein: data/wde_vorgerechnet.json (alle Ensemble-Zahlen der App stammen von hier).

    python tools/wde_make_vorgerechnet.py --workers 3     # alle Zellen in 5 Ziehungen rechnen und data/wde_vorgerechnet.json schreiben
    python tools/wde_make_vorgerechnet.py --check         # Raster gegen den Quelltext prüfen und je eine kleine Zelle neu rechnen (alle Zähler exakt gleich)

Rechenzeit: einige Minuten (n = 40 mit Zahlenskala 10 braucht je Instanz die Statustabelle über alle Kapazitäten). Die Messung steht in wde_ensemble.py; sie rechnet in
Ganzzahlen und mit SplitMix64, die Zähler sind deshalb auf jeder Plattform exakt gleich (Mediane und Mittel sind daraus abgeleitet)."""

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import wde_constants as C  # noqa: E402
import wde_ensemble as M  # noqa: E402

GRID = ROOT / C.GRID_DATEI
QUELLEN = ("wde_ensemble.py", "wde_estimates.py", "wde_knapsack.py", "wde_rng.py", "wde_rules.py", "wde_schedule.py")
STAND = "2026-11-19"


def quelltext_hash():
    """Hash über den Quelltext der Messung (Zeilenenden vereinheitlicht): ändert sich der Code, ist das Raster veraltet."""
    h = hashlib.sha256()
    for name in QUELLEN:
        h.update(name.encode())
        h.update((ROOT / name).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def key_zelle(n, s, f):
    return f"{n}|{s}|{f}"


def job(args):
    art, block, par = args
    if art == "zelle":
        return args, M.messe_zelle(block, *par)
    if art == "schaetz":
        return args, M.messe_schaetz(block, *par)
    return args, M.messe_reihenfolge(block, *par)


def aufgaben():
    t = []
    for b in range(C.BLOECKE):
        for n in C.N_STUFEN:
            for s in C.SKALEN:
                for f in C.FAMILIEN:
                    t.append(("zelle", b, (n, s, f)))
        for n in C.N_SCHAETZ:
            for s in C.SKALEN:
                for f in C.FAMILIEN:
                    for dl in C.RAUSCHEN:
                        t.append(("schaetz", b, (n, s, f, dl)))
        for rmax in C.RMAX_STUFEN:
            t.append(("reihenfolge", b, (rmax,)))
    return t


def rechne(workers):
    t0 = time.time()
    tasks = sorted(aufgaben(), key=lambda a: -(a[2][0] if a[0] != "reihenfolge" else 0))      # große Zellen zuerst
    zellen, schaetz, reihe = {}, {}, {}
    with ProcessPoolExecutor(workers) as ex:
        for i, (args, res) in enumerate(ex.map(job, tasks, chunksize=1)):
            art, block, par = args
            if art == "zelle":
                zellen.setdefault(key_zelle(*par), {})[block] = res
            elif art == "schaetz":
                schaetz.setdefault(f"{par[0]}|{par[1]}|{par[2]}|{par[3]:g}", {})[block] = res
            else:
                reihe.setdefault(str(par[0]), {})[block] = res
            if i % 20 == 0:
                print(f"[{time.time() - t0:5.0f} s] {i + 1}/{len(tasks)}", flush=True)
    sort = lambda d: {k: [v[b] for b in sorted(v)] for k, v in sorted(d.items())}      # noqa: E731
    meta = dict(stand=STAND, quelltext_hash=quelltext_hash(), bloecke=C.BLOECKE, k_je_kap=C.K_JE_KAP, k_seq=C.K_SEQ, n_stoerung=C.N_STOERUNG,
                quelle="tools/wde_make_vorgerechnet.py (wde_ensemble), erfundenes Rucksack-Vehikel, kein Modell, keine reale Datenquelle")
    GRID.parent.mkdir(exist_ok=True)
    GRID.write_text(json.dumps(dict(meta=meta, zellen=sort(zellen), schaetz=sort(schaetz), reihenfolge=sort(reihe)), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{GRID}: {GRID.stat().st_size / 1e6:.2f} MB, {len(tasks)} Aufgaben in {time.time() - t0:.0f} s")


def gleich(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(gleich(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(gleich(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))
    return a == b


def pruefe():
    roh = json.loads(GRID.read_text(encoding="utf-8"))
    fehler = []
    if roh["meta"]["quelltext_hash"] != quelltext_hash():
        fehler.append("Quelltext-Hash weicht ab: Raster neu rechnen (tools/wde_make_vorgerechnet.py)")
    proben = [("zelle", 0, (8, 1, "unkorreliert"), roh["zellen"][key_zelle(8, 1, "unkorreliert")][0]),
              ("zelle", 3, (12, 10, "stark korreliert"), roh["zellen"][key_zelle(12, 10, "stark korreliert")][3]),
              ("schaetz", 1, (12, 1, "unkorreliert", 0.05), roh["schaetz"]["12|1|unkorreliert|0.05"][1]),
              ("reihenfolge", 2, (10,), roh["reihenfolge"]["10"][2])]
    for art, block, par, alt in proben:
        _, neu = job((art, block, par))
        if not gleich(json.loads(json.dumps(neu)), alt):
            fehler.append(f"{art} {par} Ziehung {block}: Zähler weichen ab")
    for f in fehler:
        print("FEHLER:", f)
    print("Raster in Ordnung" if not fehler else f"{len(fehler)} Abweichungen")
    return 1 if fehler else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        sys.exit(pruefe())
    rechne(a.workers)


if __name__ == "__main__":
    main()
