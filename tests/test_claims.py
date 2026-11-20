"""Claims-Tests: jede Zahl der README steht hier, aus dem eingefrorenen Raster (data/wde_vorgerechnet.json, fünf Ziehungen je Zelle) neu aufgerechnet und in der README gesucht.
Jede Absolutaussage hat eine Gegenprobe (Zählung über alle Zellen). Die Zahl-Test-Tabelle der README nennt die Tests unten; das Neu-Rechnen einzelner Zellen steht in test_grid.py."""

from pathlib import Path

import numpy as np
import pytest

import wde_constants as C
import wde_evaluation as ev

README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
R = ev.lade_raster()
bt = ev.band_text
FAM = {"unk": "unkorreliert", "stark": "stark korreliert"}


def im_readme(text):
    assert text in README, f"nicht in der README: {text!r}"


def d1(x):
    return ev.de(100 * x, 1)


def test_gleichstand():
    erwartet = {("unk", 1): ["4,3 % [2,3 bis 6,0]", "6,4 % [4,3 bis 9,0]", "10,8 % [9,3 bis 13,0]", "19,9 % [13,3 bis 28,7]"],
                ("unk", 10): ["0,5", "0,8", "1,4", "1,9"], ("stark", 1): ["41,7", "74,5", "88,8", "96,0"], ("stark", 10): ["6,4", "21,2", "56,7", "86,1"]}
    for (f, s), werte in erwartet.items():
        for n, w in zip(C.N_STUFEN, werte):
            b = ev.gleichstand(R, n, s, FAM[f])["instanzen"]
            assert bt(b).startswith(w) if "[" in w else d1(b[0]) == w, (f, s, n)
    im_readme("4,3 % [2,3 bis 6,0] / 6,4 % [4,3 bis 9,0] / 10,8 % [9,3 bis 13,0] / 19,9 % [13,3 bis 28,7]")
    im_readme("mit s = 10: 0,5 / 0,8 / 1,4 / 1,9 %")
    im_readme("**Stark, s = 1: 41,7 / 74,5 / 88,8 / 96,0 %**, s = 10: 6,4 / 21,2 / 56,7 / 86,1 % |")
    # Gegenprobe zu „Nein“: in jeder Zelle der stark korrelierten Familie mit n >= 12 und s = 1 haben mehr als die Hälfte der Instanzen mehrere optimale Mengen; bei unkorrelierten Zahlen nie
    assert all(ev.gleichstand(R, n, 1, "stark korreliert")["instanzen"][1] > 0.5 for n in (12, 20, 40))
    assert all(ev.gleichstand(R, n, s, "unkorreliert")["instanzen"][2] < 0.3 for n in C.N_STUFEN for s in C.SKALEN)
    # Mischwert: die vier Zellen n = 8, 12, s = 1, beide Familien, je Ziehung gemittelt
    mix = [np.mean([R["zellen"][f"{n}|1|{f}"][b]["tie_i"] / R["zellen"][f"{n}|1|{f}"][b]["ninst"] for n in (8, 12) for f in C.FAMILIEN]) for b in range(5)]
    assert (d1(np.mean(mix)), d1(min(mix)), d1(max(mix))) == ("31,7", "30,6", "33,2")
    im_readme("31,7 % [30,6 bis 33,2] der Instanzen")
    zellen = [ev.gleichstand(R, n, 1, f)["instanzen"][0] for n in (8, 12) for f in C.FAMILIEN]
    assert (d1(min(zellen)), d1(max(zellen))) == ("4,3", "74,5")
    im_readme("(die Zellen gehen von 4,3 bis 74,5 % auseinander)")


def test_kippabstand():
    unk = [ev.kippabstand(R, n, s, "unkorreliert") for n in C.N_STUFEN for s in C.SKALEN]
    stark = [ev.kippabstand(R, n, s, "stark korreliert") for n in C.N_STUFEN for s in C.SKALEN]
    assert (ev.de(min(k["median"][0] for k in unk), 2), ev.de(max(k["median"][0] for k in unk), 2)) == ("0,47", "0,57")
    assert (d1(min(k["le5"][0] for k in unk)), d1(max(k["le5"][0] for k in unk))) == ("3,2", "5,4")
    assert (ev.de(min(k["median"][0] for k in stark), 2), ev.de(max(k["median"][0] for k in stark), 2)) == ("0,04", "0,43")
    assert (d1(min(k["le5"][0] for k in stark)), d1(max(k["le5"][0] for k in stark))) == ("1,7", "53,2")
    im_readme("stark: 0,04 bis 0,43 beziehungsweise 1,7 bis 53,2 %")
    im_readme("unk: 0,47 bis 0,57 des eigenen Werts über alle acht Zellen; Anteil mit Abstand höchstens 5 %: 3,2 bis 5,4 %")
    # Gegenprobe: der Median unkorrelierter Pakete liegt in allen acht Zellen über dem der stark korrelierten in derselben Zelle
    assert all(a["median"][0] > b["median"][0] for a, b in zip(unk, stark))


def test_kapazitaet():
    nm = [ev.kapazitaet(R, n, 1, "unkorreliert")["nichtmonoton"][0] for n in C.N_STUFEN]
    assert [d1(x) for x in nm] == ["73,9", "83,6", "91,3", "96,8"]
    im_readme("unk, s = 1: 73,9 / 83,6 / 91,3 / 96,8 % (n = 8 bis 40; der Bereich wächst mit n)")
    n5u = [ev.kapazitaet(R, n, s, "unkorreliert")["nah5"][0] for n in C.N_STUFEN for s in C.SKALEN]
    n5s = [ev.kapazitaet(R, n, s, "stark korreliert")["nah5"][0] for n in C.N_STUFEN for s in C.SKALEN]
    assert (d1(min(n5u)), d1(max(n5u))) == ("20,7", "29,1") and (d1(min(n5s)), d1(max(n5s))) == ("69,2", "99,4")
    im_readme("unk 20,7 bis 29,1 % der Pakete, stark 69,2 bis 99,4 %")
    # Gegenprobe zur Existenzaussage: bei unkorrelierten Zahlen fliegt in allen acht Zellen mehr als die Hälfte der Pakete irgendwo hinaus; bei n = 40, s = 1, stark nur ein Viertel
    assert all(ev.kapazitaet(R, n, s, "unkorreliert")["nichtmonoton"][0] > 0.7 for n in C.N_STUFEN for s in C.SKALEN)
    assert d1(ev.kapazitaet(R, 40, 1, "stark korreliert")["nichtmonoton"][0]) == "24,8"


def test_regeln():
    r = [ev.regeln(R, n, 1, "unkorreliert") for n in C.N_STUFEN]
    assert [d1(x["g_pakete"][0]) for x in r] == ["91,0", "91,1", "93,6", "96,1"]
    assert [d1(x["g_menge"][0]) for x in r] == ["68,3", "55,3", "46,4", "32,6"] and [d1(x["g_opt"][0]) for x in r] == ["68,4", "55,1", "46,1", "33,2"]
    assert (d1(min(x["basis"][0] for x in r)), d1(max(x["basis"][0] for x in r))) == ("56,6", "60,2")
    im_readme("Greedy: Pakete **richtig** 91,0 / 91,1 / 93,6 / 96,1 %, **ganze Menge gleich** (Instanzen mit eindeutigem Optimum) 68,3 / 55,3 / 46,4 / 32,6 %, Optimalwert getroffen 68,4 / 55,1 / 46,1 / 33,2 %; "
              "Basis „Pakete richtig“ (das häufigere Status raten) 56,6 bis 60,2 %")
    st = [ev.regeln(R, n, s, "stark korreliert") for n in C.N_STUFEN for s in C.SKALEN]
    assert (d1(min(x["g_opt"][0] for x in st)), d1(max(x["g_opt"][0] for x in st))) == ("2,4", "12,7")
    assert (d1(min(x["g_pakete"][0] for x in st)), d1(max(x["g_pakete"][0] for x in st))) == ("72,2", "99,8")
    assert ev.regeln(R, 40, 1, "stark korreliert")["eindeutig"] == [5, 8, 6, 5, 6] and ev.regeln(R, 40, 1, "stark korreliert")["g_menge"][0] > 0.6
    im_readme("Greedy trifft den Optimalwert nur in 2,4 bis 12,7 % der Instanzen (alle acht Zellen), Pakete richtig 72,2 bis 99,8 %; bei n = 40, s = 1 gibt es nur 5 bis 8 Instanzen mit eindeutigem Optimum je Ziehung")
    # Gegenprobe zu „Greedy ist je Paket klar besser als Raten“: bei unkorrelierten Zahlen in allen acht Zellen, bei stark korrelierten nicht überall deutlich
    assert all(ev.regeln(R, n, s, "unkorreliert")["g_pakete"][0] > ev.regeln(R, n, s, "unkorreliert")["basis"][0] + 0.25 for n in C.N_STUFEN for s in C.SKALEN)
    mix = [np.mean([R["zellen"][f"{n}|1|{f}"][b]["g_opt"] / R["zellen"][f"{n}|1|{f}"][b]["ninst"] for n in (8, 12) for f in C.FAMILIEN]) for b in range(5)]
    assert (d1(np.mean(mix)), d1(min(mix)), d1(max(mix))) == ("37,2", "35,7", "39,8")
    im_readme("37,2 % [35,7 bis 39,8]")
    for n in C.N_STUFEN:       # Basisrate-Test des Plans (Punkt d): Greedy je Paket besser als Raten, auch bei n = 20 und 40
        assert ev.regeln(R, n, 1, "unkorreliert")["g_pakete"][1] > ev.regeln(R, n, 1, "unkorreliert")["basis"][2]


def test_reihenfolge():
    r0, r10, r25 = ev.reihe(R, 0), ev.reihe(R, 10), ev.reihe(R, 25)
    assert bt(r0["mehrere"]) == "58,8 % [56,0 bis 61,0]" and bt(r0["s_opt"]) == bt(r0["d_opt"]) == bt(r0["s_paare"]) == "100,0 % [100,0 bis 100,0]"
    assert (bt(r10["s_opt"]), bt(r10["d_opt"])) == ("8,6 % [7,5 bis 9,0]", "61,1 % [59,5 bis 63,0]")
    assert (bt(r25["s_opt"]), bt(r25["d_opt"])) == ("1,2 % [0,5 bis 2,0]", "54,0 % [51,0 bis 60,5]")
    assert (bt(r10["mehrere"]), bt(r25["mehrere"])) == ("35,7 % [32,0 bis 40,5]", "21,5 % [19,5 bis 24,5]")
    for text in ("mehrere optimale Reihenfolgen trotzdem in 58,8 % [56,0 bis 61,0]", "statisch trifft den Optimalwert in 8,6 % [7,5 bis 9,0], Dispatch in 61,1 % [59,5 bis 63,0]",
                 "bis 25: 1,2 % [0,5 bis 2,0] gegen 54,0 % [51,0 bis 60,5]", "58,8 [56,0 bis 61,0] / 35,7 [32,0 bis 40,5] / 21,5 [19,5 bis 24,5] %"):
        im_readme(text)
    assert (ev.de(r0["dw_median"], 2), d1(r0["dw_le_third"]), ev.de(r0["dp_median"], 2), r0["nicht_kippbar_p"], r0["paare"]) == ("0,75", "14,3", "1,45", 645, 20090)
    im_readme("im Median 0,75; Anteil höchstens 1/3: 14,3 %; kleinste Erhöhung von p_A relativ im Median 1,5 (Mittel der fünf Ziehungsmediane 1,45); 645 von 20 090 sicheren Paaren")
    z0 = R["reihenfolge"]["0"][0]                                                           # Ziehung 0 benutzt die Zufalls-Seeds des Plans
    assert (z0["tie_i"], z0["ni"], z0["n_paare"], z0["nonk_p"], z0["dw_median"]) == (122, 200, 4012, 152, 0.75) and d1(z0["dw_le_third"]) == "15,1"
    im_readme("122 von 200 Instanzen ohne Freigaben mit mehreren optimalen Reihenfolgen (61,0 %), 4 012 sichere Paare, 152 davon nicht kippbar bis +29, Median der Senkung von w_A 0,75, Anteil höchstens 1/3: 15,1 %")
    # Gegenprobe: ohne Freigaben ist die statische Regel in jeder Ziehung exakt, mit Freigaben in keiner Ziehung bei mehr als 10 % der Instanzen
    assert all(z["S_opt"] == z["ni"] for z in R["reihenfolge"]["0"]) and all(z["S_opt"] <= 0.1 * z["ni"] for rm in ("10", "25") for z in R["reihenfolge"][rm])


def test_stoerung():
    def f(n, s, fam, d):
        return d1(ev.stoerung(R, n, s, fam, d)[0])
    assert [f(n, 1, "unkorreliert", 0.05) for n in (12, 20, 40)] == ["5,9", "9,9", "18,6"] and [f(n, 1, "unkorreliert", 0.1) for n in (12, 20, 40)] == ["14,7", "22,8", "38,7"]
    assert [f(n, 10, "stark korreliert", 0.01) for n in (12, 20, 40)] == ["26,0", "28,9", "31,5"]
    im_readme("5,9 / 9,9 / 18,6 % und 14,7 / 22,8 / 38,7 %; stark, s = 10, schon bei ±1 %: 26,0 / 28,9 / 31,5 %")
    # Gegenprobe: bei s = 1 und unkorrelierten Zahlen ändert eine Störung von 1 % die Menge praktisch nie (höchstens 0,4 %)
    assert all(ev.stoerung(R, n, 1, "unkorreliert", 0.01)[0] <= 0.005 for n in (12, 20, 40))


def test_schaetzung():
    def zelle(k, kind):
        return {key: ev.schaetzung(R, int(key.split("|")[0]), int(key.split("|")[1]), key.split("|")[2], float(key.split("|")[3]), k)["gueltig"][0] for key in R["schaetz"] if kind(key)}
    unk = lambda key: "|unkorreliert|" in key           # noqa: E731
    st12 = lambda key: key.startswith("12|") and "stark" in key       # noqa: E731
    st40 = lambda key: key.startswith("40|") and "stark" in key       # noqa: E731
    rel0, rel100 = zelle("rel0", lambda k: True), zelle("rel100", lambda k: True)
    assert (d1(min(v for k, v in rel0.items() if unk(k))), d1(max(v for k, v in rel0.items() if unk(k)))) == ("51,2", "63,1")
    assert (d1(min(v for k, v in rel0.items() if st12(k))), d1(max(v for k, v in rel0.items() if st12(k)))) == ("59,5", "68,8")
    assert (d1(min(v for k, v in rel0.items() if st40(k))), d1(max(v for k, v in rel0.items() if st40(k)))) == ("20,2", "35,1")
    assert (d1(min(v for k, v in rel100.items() if unk(k))), d1(max(v for k, v in rel100.items() if unk(k)))) == ("95,2", "99,2")
    assert (d1(min(v for k, v in rel100.items() if st40(k))), d1(max(v for k, v in rel100.items() if st40(k)))) == ("69,0", "90,2")
    for text in ("in 51,2 bis 63,1 % der Störungen (acht Zellen), bei stark mit n = 12 in 59,5 bis 68,8 %, **bei stark mit n = 40 nur in 20,2 bis 35,1 %**", "unk 95,2 bis 99,2 %; **stark mit n = 40: 69,0 bis 90,2 %**"):
        im_readme(text)
    cells = [ev.schaetzung(R, int(k.split("|")[0]), int(k.split("|")[1]), k.split("|")[2], float(k.split("|")[3]), "abs8") for k in R["schaetz"]]
    assert all(c["fehlschlaege"] == 0 and c["gueltig"][0] == 1.0 for c in cells) and len(cells) == 16
    assert (min(c["stoerungen"] for c in cells), max(c["stoerungen"] for c in cells)) == (14569, 118187)
    im_readme("**8 Rauscheinheiten: in allen 16 Zellen kein einziger Fehlschlag** (14 569 bis 118 187 Störungen je Zelle aus fünf Ziehungen)")
    aw = [ev.aufwand(R, int(k.split("|")[0]), int(k.split("|")[1]), k.split("|")[2], float(k.split("|")[3]), "abs8")[0] for k in R["schaetz"]]
    assert ev.de(max(aw), 1) == "12,1"
    im_readme("der Aufwand reicht bis zum 12,1-Fachen")
    aw100 = [ev.aufwand(R, int(k.split("|")[0]), int(k.split("|")[1]), k.split("|")[2], float(k.split("|")[3]), "rel100")[0] for k in R["schaetz"]]
    assert ev.de(max(aw100), 1) == "2,6"
    im_readme("Aufwand bis zum 2,6-Fachen")
    # Gegenprobe zu „volle Sicherheit gibt es nicht“: der relative Aufschlag 100 % liegt in 6 von 16 Zellen unter 95 % (alle stark korreliert), der absolute Abstand 8 in keiner
    fehl = [k for k in R["schaetz"] if ev.schaetzung(R, int(k.split("|")[0]), int(k.split("|")[1]), k.split("|")[2], float(k.split("|")[3]), "rel100")["gueltig"][0] < 0.95]
    assert len(fehl) == 6 and all("stark" in k for k in fehl)
    im_readme("unter 95 % liegt sie in 6 von 16 Zellen, alle stark korreliert")


def test_oracle_zahlen_der_readme():
    im_readme("in 48 Instanzen (n = 6, 8, beide Familien und Skalen, drei Kapazitäten)")
    im_readme("in 62 von 62 Paketen")
    im_readme("in 64 von 64 Paketen")


def test_handbeispiel_zahlen_der_readme():
    for text in ("Optimum 18 mit {3, 4}, eindeutig", "seine Wertschwelle ist 2 (B = 18, A = 7 + 9 = 16)", "**Gleichstand mit drei optimalen Mengen** {3, 4}, {2, 3}, {2, 4} (je 18)",
                 "Greedy nimmt {2, 3} mit Wert 16 gegen 18", "**Paket 2 ist bei Kapazität 6 und 7 drin, bei 8 und 9 draußen, bei 10 wieder drin:** der Optimalwert steigt von 16 auf 18",
                 "geschätzte Schwelle 1 gegen wahre 2"):
        im_readme(text)


def test_zaehleinheiten_und_instanzzahlen():
    assert C.K_JE_KAP == 100 and C.BLOECKE == 5 and C.K_SEQ == 200 and C.N_STOERUNG == 5 and len(C.KAPAZITAETEN) == 3
    for text in ("je Ziehung 3 Kapazitätsanteile × 100 = 300; bei n = 40: 150", "**fünf getrennten Ziehungen**", "5 Schätzungen je Instanz", "175 Aufgaben"):
        im_readme(text)


@pytest.mark.parametrize("wort", ["Saatwert", "Schwanz", " trägt ", " tragen ", "erklärt"])
def test_verbotene_woerter_fehlen_in_readme_und_app(wort):
    app = (Path(__file__).resolve().parent.parent / "app.py").read_text(encoding="utf-8")
    assert wort not in README and wort not in app
