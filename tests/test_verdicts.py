"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Schwelle und jede Zahl im Text steht hier. Ein Verdikt ist eine reine Funktion (Art, Text)."""

import pytest

import wde_evaluation as ev


def test_paket_sicher_draussen_nennt_schwelle_und_preis_der_erzwingung():
    kind, text = ev.verdict_paket(2, -1, -2, 7, 3, 8, True)                      # Handbeispiel: Paket 2, Wert 7, Wertschwelle 2
    assert kind == "success" and "Paket 2 (Wert 7) liegt in **keiner**" in text and "um 2 auf 9 steigen (28,6 % seines Werts)" in text and "kostet genau 2 Wertpunkte" in text


def test_paket_sicher_drin_nennt_fall_des_werts():
    kind, text = ev.verdict_paket(4, 1, 2, 9, 5, 8, True)
    assert kind == "success" and "liegt in **jeder** optimalen Menge" in text and "um 2 auf 7 fallen (22,2 % seines Werts)" in text and "auszuschließen kostet genau 2" in text


def test_paket_gleichstand_und_passt_nie():
    kind, text = ev.verdict_paket(3, 0, 0, 9, 3, 8, True)
    assert kind == "info" and "**Gleichstand**" in text and "drei verschiedene Fragen" in text
    kind, text = ev.verdict_paket(1, -1, 0, 5, 20, 8, False)
    assert kind == "warning" and "wiegt 20, mehr als die Kapazität 8: es passt nie" in text


def _g(m, lo=0.0, hi=1.0):
    return dict(instanzen=(m, lo, hi), pakete=(m / 3, 0, 1), ninst=300)


def test_gleichstand_schwellen_halb_und_ein_fuenftel_sind_inklusiv():
    assert ev.verdict_gleichstand(_g(0.5), 12, 1, "stark korreliert")[0] == "warning"
    assert ev.verdict_gleichstand(_g(0.4999), 12, 1, "stark korreliert")[0] == "info"
    assert ev.verdict_gleichstand(_g(0.2), 12, 1, "unkorreliert")[0] == "info"
    kind, text = ev.verdict_gleichstand(_g(0.1999, 0.15, 0.25), 8, 1, "unkorreliert")
    assert kind == "success" and "20,0 % [15,0 bis 25,0]" in text and "300 Instanzen" in text and "selten, aber nicht null" in text


def test_gleichstand_text_nennt_zelle_und_spanne():
    kind, text = ev.verdict_gleichstand(_g(0.745, 0.71, 0.79), 12, 1, "stark korreliert")
    assert "n = 12, Zahlenskala 1, stark korreliert" in text and "74,5 % [71,0 bis 79,0]" in text and "mehrere gleich gute Antworten" in text


def test_kapazitaet_nennt_beide_zahlen_und_die_existenzaussage():
    k = dict(nichtmonoton=(0.739, 0.718, 0.76), nah5=(0.209, 0.197, 0.221))
    kind, text = ev.verdict_kapazitaet(k, 8, 1, "unkorreliert")
    assert kind == "info" and "73,9 % [71,8 bis 76,0]" in text and "20,9 % [19,7 bis 22,1]" in text and "hebt den Optimalwert nie" in text


def test_regel_warnt_wenn_ganze_menge_oder_optimalwert_unter_halb_liegt():
    r = dict(g_pakete=(0.91, 0.89, 0.92), s_pakete=(0.89, 0.88, 0.9), g_menge=(0.683, 0.621, 0.73), s_menge=(0.52, 0.46, 0.56), g_opt=(0.684, 0.62, 0.737), s_opt=(0.5, 0.4, 0.6), basis=(0.566, 0.56, 0.57))
    kind, text = ev.verdict_regel(r, "greedy", 8, 1, "unkorreliert")
    assert kind == "info" and "je Paket **91,0 % [89,0 bis 92,0]** richtig" in text and "**68,3 % [62,1 bis 73,0]** der Instanzen mit eindeutigem Optimum" in text and "56,6 %" in text
    r["g_menge"] = (0.499, 0.4, 0.6)
    assert ev.verdict_regel(r, "greedy", 8, 1, "unkorreliert")[0] == "warning"
    r["g_menge"], r["g_opt"] = (0.9, 0.8, 1.0), (0.499, 0.4, 0.6)
    assert ev.verdict_regel(r, "greedy", 8, 1, "unkorreliert")[0] == "warning"
    r["g_menge"] = None
    kind, text = ev.verdict_regel(r, "greedy", 40, 1, "stark korreliert")
    assert "nicht messbar" in text
    assert ev.verdict_regel(r, "schwelle", 8, 1, "unkorreliert")[1].startswith("Regel S bei n = 8")


def _s(gueltig, fehl=0, stoer=100_000):
    return dict(gueltig=gueltig, fehlschlaege=fehl, stoerungen=stoer)


def test_schaetzung_ohne_fehlschlag_nennt_voraussetzung_nicht_sicher():
    kind, text = ev.verdict_schaetz(_s((1.0, 1.0, 1.0)), "abs8", 40, 1, "stark korreliert", 0.1)
    assert kind == "success" and "kein einziger Fehlschlag in 100000 Störungen" in text and "nicht „sicher“" in text and "bekanntes Rauschniveau" in text and "Mit der Sicherheit „absoluter Abstand 8 Rauscheinheiten“ (Rauschen 10 %, n = 40, Zahlenskala 1, stark korreliert)" in text


def test_schaetzung_fehlschlaege_schwelle_neun_zehntel():
    kind, text = ev.verdict_schaetz(_s((0.69, 0.656, 0.723), 9000, 29277), "rel100", 40, 1, "stark korreliert", 0.1)
    assert kind == "warning" and "69,0 % [65,6 bis 72,3]" in text and "9000 Fehlschläge in 29277 Störungen" in text and "Rauschen 10 %" in text
    assert ev.verdict_schaetz(_s((0.9, 0.89, 0.91), 10), "rel100", 40, 1, "unkorreliert", 0.05)[0] == "info"
    assert ev.verdict_schaetz(_s((0.8999, 0.89, 0.91), 10), "rel100", 40, 1, "unkorreliert", 0.05)[0] == "warning"


def _r():
    return dict(ni=200, s_opt=(0.086, 0.075, 0.09), d_opt=(0.611, 0.595, 0.63), s_paare=(0.854, 0.847, 0.859), mehrere=(0.588, 0.56, 0.61))


def test_reihenfolge_ohne_freigaben_ist_exakt_mit_freigaben_je_regel():
    r = dict(_r(), s_opt=(1.0, 1.0, 1.0), s_paare=(1.0, 1.0, 1.0))
    kind, text = ev.verdict_reihenfolge(r, 0, "statisch")
    assert kind == "success" and "exakt" in text and "100,0 % [100,0 bis 100,0]" in text and "58,8 % [56,0 bis 61,0]" in text
    kind, text = ev.verdict_reihenfolge(_r(), 10, "statisch")
    assert kind == "warning" and "Regel „statische Smith-Reihenfolge“ den Optimalwert in **8,6 % [7,5 bis 9,0]**" in text and "61,1 %" in text
    kind, text = ev.verdict_reihenfolge(_r(), 10, "dispatch")
    assert kind == "info" and "Regel „Dispatch“ den Optimalwert in **61,1 %" in text
    assert ev.verdict_reihenfolge(dict(_r(), d_opt=(0.4999, 0, 1)), 10, "dispatch")[0] == "warning"


def test_zahlenformate():
    assert ev.de(0.5, 2) == "0,50" and ev.pct(0.1234) == "12,3 %" and ev.band_text((0.5, 0.25, 0.75)) == "50,0 % [25,0 bis 75,0]"
    assert ev.band([{"a": 1, "b": 2}, {"a": 3, "b": 4}], "a", "b") == pytest.approx((0.625, 0.5, 0.75))
