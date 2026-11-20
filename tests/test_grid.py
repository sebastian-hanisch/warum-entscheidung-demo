"""Das eingefrorene Raster (data/wde_vorgerechnet.json): Vollständigkeit, Invarianten der Zähler, Quelltext-Hash und exaktes Neu-Rechnen einzelner Zellen.

Alle Zähler sind Ganzzahlen aus SplitMix64 und exakter Arithmetik: das Neu-Rechnen muss auf jeder Plattform dieselben Zähler geben (kein Band nötig); Mediane und Mittel sind abgeleitet."""

import importlib.util
import json
from pathlib import Path

import pytest

import wde_constants as C
import wde_ensemble as M
import wde_evaluation as ev

R = ev.lade_raster()


def _tool():
    spec = importlib.util.spec_from_file_location("wde_make", Path(__file__).resolve().parent.parent / "tools" / "wde_make_vorgerechnet.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_raster_ist_vollstaendig_fuenf_ziehungen_je_zelle():
    assert len(R["zellen"]) == 16 and len(R["schaetz"]) == 16 and set(R["reihenfolge"]) == {"0", "10", "25"}
    for z in list(R["zellen"].values()) + list(R["schaetz"].values()) + list(R["reihenfolge"].values()):
        assert len(z) == C.BLOECKE == 5
    for n in C.N_STUFEN:
        for s in C.SKALEN:
            for f in C.FAMILIEN:
                assert ev.instanzen_je_ziehung(ev.zelle(R, n, s, f)) == (150 if n == 40 else 300)
    m = R["meta"]
    assert (m["bloecke"], m["k_je_kap"], m["k_seq"], m["n_stoerung"]) == (5, 100, 200, 5) and "kein Modell" in m["quelle"]


def test_zaehler_sind_in_sich_stimmig():
    for key, zeilen in R["zellen"].items():
        n = int(key.split("|")[0])
        for z in zeilen:
            assert 0 <= z["tie_i"] <= z["ninst"] and z["tie_p"] <= z["pak"] == n * z["ninst"]
            assert z["tie_i"] <= z["tie_p"] or z["tie_p"] == 0                                    # jede Instanz mit Gleichstand hat mindestens ein Paket im Gleichstand
            assert z["drin"] <= z["sicher"] <= z["pak"] and z["kipp5"] <= z["kipp10"] <= z["sicher"]
            assert z["uniq"] == z["ninst"] - z["tie_i"] and z["g_set"] <= z["uniq"] and z["s_set"] <= z["uniq"]
            assert z["g_it"] <= z["sicher"] and z["s_it"] <= z["sicher"] and z["g_opt"] <= z["ninst"]
            assert z["nah5"] <= z["nah10"] <= z["nah25"] <= z["sicher"]
            assert 0 <= z["kipp_median"] <= 1.5


def test_greedy_trifft_den_optimalwert_mindestens_so_oft_wie_regel_s():
    """Greedy enthält die Menge der Regel S (gleiche Reihenfolge, nur weiter gefüllt) und hat deshalb nie einen kleineren Wert: trifft S den Optimalwert, trifft ihn auch Greedy.
    Für die Einzelentscheidungen und die ganze Menge gilt das nicht in jeder Zelle (Gegenprobe: Zahl der Ausnahmen steht unten)."""
    ausnahmen_pakete = ausnahmen_menge = 0
    for zeilen in R["zellen"].values():
        for z in zeilen:
            assert z["s_opt"] <= z["g_opt"]
            ausnahmen_pakete += int(z["s_it"] > z["g_it"])
            ausnahmen_menge += int(z["s_set"] > z["g_set"])
    assert (ausnahmen_pakete, ausnahmen_menge) == (8, 0)             # S ist bei den Einzelentscheidungen in 8 von 80 Zellen-Ziehungen besser, bei der ganzen Menge nie


def test_schaetz_zaehler_monoton_in_der_sicherheit_und_keine_fehlschlaege_bei_abs8():
    for zeilen in R["schaetz"].values():
        for z in zeilen:
            c = z["cnt"]
            assert c["rel0"] <= c["rel25"] <= c["rel50"] <= c["rel100"] <= z["den"]
            assert c["abs2"] <= c["abs4"] <= c["abs8"] <= z["den"]
            assert z["fehlschlaege_abs8"] == z["den"] - c["abs8"] == 0                              # Gegenprobe zu „volle Sicherheit gibt es nicht“: 0 Fehlschläge
    # Gegenprobe: der relative Aufschlag hat Fehlschläge (stark korreliert, n = 40, 10 % Rauschen, Verdopplung)
    assert sum(z["den"] - z["cnt"]["rel100"] for z in R["schaetz"]["40|1|stark korreliert|0.1"]) > 1000


def test_reihenfolge_zaehler():
    for rm, zeilen in R["reihenfolge"].items():
        for z in zeilen:
            assert z["S_opt"] <= z["ni"] and z["D_opt"] <= z["ni"] and z["S_pn"] <= z["pd"] and z["D_pn"] <= z["pd"] and z["pair_den"] == 42 * z["ni"]
            if rm == "0":
                assert z["S_opt"] == z["D_opt"] == z["ni"] and z["S_pn"] == z["pd"] and z["D_pn"] == z["pd"]          # ohne Freigaben exakt
                assert z["n_paare"] > 3000
            else:
                assert z["S_opt"] < 0.2 * z["ni"]                                                                  # Gegenprobe: mit Freigaben verfehlt die statische Regel meist


def test_quelltext_hash_des_rasters_gleicht_dem_aktuellen_quelltext():
    assert R["meta"]["quelltext_hash"] == _tool().quelltext_hash(), "Quelltext der Messung geändert: Raster neu rechnen (tools/wde_make_vorgerechnet.py)"


@pytest.mark.parametrize("art,block,par,pfad", [
    ("zelle", 0, (8, 1, "unkorreliert"), lambda r: r["zellen"]["8|1|unkorreliert"][0]),
    ("zelle", 3, (12, 10, "stark korreliert"), lambda r: r["zellen"]["12|10|stark korreliert"][3]),
    ("schaetz", 1, (12, 1, "unkorreliert", 0.05), lambda r: r["schaetz"]["12|1|unkorreliert|0.05"][1]),
    ("reihenfolge", 2, (10,), lambda r: r["reihenfolge"]["10"][2]),
])
def test_neu_rechnen_gibt_exakt_dieselben_zaehler(art, block, par, pfad):
    f = {"zelle": M.messe_zelle, "schaetz": M.messe_schaetz, "reihenfolge": M.messe_reihenfolge}[art]
    neu = json.loads(json.dumps(f(block, *par)))
    assert _tool().gleich(neu, pfad(R))


def test_raster_ist_klein_und_gueltig():
    pfad = ev.ROOT / C.GRID_DATEI
    assert pfad.stat().st_size < 500_000 and json.loads(pfad.read_text(encoding="utf-8"))["meta"]["stand"]
