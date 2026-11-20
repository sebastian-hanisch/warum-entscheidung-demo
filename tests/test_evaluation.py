"""Aggregation über die Ziehungen an von Hand gesetzten Zählern: Mittel und Spanne der Anteile (nicht der Zähler), Schlüssel, Randfälle."""

import pytest

import wde_evaluation as ev


def test_band_ist_mittel_der_anteile_je_ziehung_mit_kleinstem_und_groesstem():
    zeilen = [{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 0, "b": 10}]
    m, lo, hi = ev.band(zeilen, "a", "b")
    assert m == pytest.approx((0.5 + 0.75 + 0.0) / 3) and lo == 0.0 and hi == 0.75           # Mittel der Anteile, nicht Verhältnis der Summen (4/16)


def test_band_ueberspringt_ziehungen_ohne_nenner_und_nimmt_funktionen():
    m, lo, hi = ev.band([{"a": 1, "b": 0}, {"a": 2, "b": 4}], "a", "b")
    assert (m, lo, hi) == (0.5, 0.5, 0.5)
    m, _, _ = ev.band([{"x": 3, "y": 1, "n": 4}], lambda z: max(z["x"], z["y"]), "n")
    assert m == 0.75


def test_band_text_und_zahlenformate():
    assert ev.band_text((0.043, 0.023, 0.06)) == "4,3 % [2,3 bis 6,0]" and ev.band_text((0.5, 0.25, 0.75), 2) == "50,00 % [25,00 bis 75,00]"
    assert ev.de(0.5, 2) == "0,50" and ev.pct(0.1234) == "12,3 %" and ev.pct(1.0, 0) == "100 %"


def _raster():
    z = {"ninst": 10, "tie_i": 2, "tie_p": 3, "pak": 80, "sicher": 60, "drin": 20, "kipp5": 6, "kipp10": 9, "kipp_median": 0.4, "nm": 40, "nah5": 12, "nah10": 20, "nah25": 30,
         "g_it": 54, "s_it": 50, "g_opt": 5, "s_opt": 4, "g_gap": 0.5, "uniq": 8, "g_set": 4, "s_set": 2, "noise": {"0.01": [0, 0], "0.05": [3, 30], "0.1": [6, 30]}}
    return {"zellen": {"8|1|unkorreliert": [dict(z), dict(z, tie_i=4)]}}


def test_gleichstand_regeln_und_basis_aus_gesetzten_zaehlern():
    R = _raster()
    g = ev.gleichstand(R, 8, 1, "unkorreliert")
    assert g["instanzen"] == pytest.approx((0.3, 0.2, 0.4)) and g["pakete"] == pytest.approx((3 / 80, 3 / 80, 3 / 80)) and g["ninst"] == 10
    r = ev.regeln(R, 8, 1, "unkorreliert")
    assert r["g_pakete"][0] == pytest.approx(0.9) and r["s_pakete"][0] == pytest.approx(50 / 60) and r["basis"][0] == pytest.approx(40 / 60)         # Basis: max(drin, draußen) / sicher
    assert r["g_menge"][0] == pytest.approx(0.5) and r["s_menge"][0] == pytest.approx(0.25) and r["eindeutig"] == [8, 8]
    k = ev.kippabstand(R, 8, 1, "unkorreliert")
    assert k["median"] == (0.4, 0.4, 0.4) and k["le5"][0] == pytest.approx(0.1)


def test_menge_gleich_nicht_messbar_wenn_eine_ziehung_unter_fuenf_eindeutige_hat():
    R = _raster()
    R["zellen"]["8|1|unkorreliert"][1]["uniq"] = 4
    r = ev.regeln(R, 8, 1, "unkorreliert")
    assert r["g_menge"] is None and r["s_menge"] is None


def test_stoerung_nur_wenn_alle_ziehungen_loesungen_haben():
    R = _raster()
    assert ev.stoerung(R, 8, 1, "unkorreliert", 0.05)[0] == pytest.approx(0.1)
    assert ev.stoerung(R, 8, 1, "unkorreliert", 0.01) is None                       # 0 Neulösungen: keine Zahl statt 0 / 0


def test_schaetzung_und_aufwand_zaehlen_fehlschlaege_und_nehmen_mittel_der_mediane():
    R = {"schaetz": {"12|1|unkorreliert|0.1": [
        {"den": 100, "cnt": {k: 90 for k in ("rel0", "rel25", "rel50", "rel100", "abs2", "abs4", "abs8")}, "aufwand_median": {"rel0": 0.5, "rel100": 1.0, "abs8": 2.0}},
        {"den": 200, "cnt": {k: 200 for k in ("rel0", "rel25", "rel50", "rel100", "abs2", "abs4", "abs8")}, "aufwand_median": {"rel0": 0.7, "rel100": 1.2, "abs8": 3.0}}]}}
    s = ev.schaetzung(R, 12, 1, "unkorreliert", 0.1, "rel0")
    assert s["gueltig"] == pytest.approx((0.95, 0.9, 1.0)) and s["fehlschlaege"] == 10 and s["stoerungen"] == 300 and s["je_ziehung"] == 150
    assert ev.aufwand(R, 12, 1, "unkorreliert", 0.1, "abs8") == pytest.approx((2.5, 2.0, 3.0))
    assert ev.aufwand(R, 12, 1, "unkorreliert", 0.1, "rel25") is None               # für diese Stufe wird kein Aufwand gespeichert


def test_schluessel_der_zellen():
    R = {"zellen": {"12|10|stark korreliert": [1]}, "schaetz": {"12|10|stark korreliert|0.05": [2]}, "reihenfolge": {"10": [3]}}
    assert ev.zelle(R, 12, 10, "stark korreliert") == [1] and ev.schaetz_zelle(R, 12, 10, "stark korreliert", 0.05) == [2] and ev.reihenfolge(R, 10) == [3]
