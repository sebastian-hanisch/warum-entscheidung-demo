"""Eigenschaftstests und Randfälle des Rucksacks (feste Zufalls-Seeds, 20 bis 50 Instanzen je Eigenschaft)."""

import numpy as np
import pytest

import wde_constants as C
import wde_knapsack as K
import wde_rules as RU
import wde_estimates as E
from wde_rng import SplitMix64


def _inst(seed, n=8, fam="unkorreliert", s=1, kap=0.5):
    return K.instanz(seed, n, fam, s, kap)


def test_instanz_bereiche_wiederholbarkeit_und_eingefrorene_werte():
    for fam in C.FAMILIEN:
        for s in C.SKALEN:
            v, w, cap = _inst(12345, 12, fam, s)
            assert w.min() >= 5 * s and w.max() <= 30 * s and cap == int(np.floor(0.5 * int(w.sum())))
            if fam == "stark korreliert":
                assert (v == w + 10 * s).all()
            else:
                assert v.min() >= 10 * s and v.max() <= 60 * s
            v2, w2, cap2 = _inst(12345, 12, fam, s)
            assert (v == v2).all() and (w == w2).all() and cap == cap2
    v, w, cap = _inst(12345, 12)                                                       # eingefroren: SplitMix64 gibt auf jeder Plattform dieselbe Instanz
    assert w[:4].tolist() == [19, 6, 30, 7] and v[:4].tolist() == [23, 51, 23, 28] and cap == 102     # zugleich gleich dem Erzeuger der Vormessung (ex_p10b_lib)
    with pytest.raises(ValueError):
        K.instanz(1, 4, "xyz")


def test_werte_mit_c_multipliziert_gleicher_status_schwellen_skalieren():
    for seed in range(20):
        v, w, cap = _inst(seed)
        st, d, fits = K.status_schnell(v, w, cap)
        st3, d3, _ = K.status_schnell(3 * v, w, cap)
        assert (st == st3).all() and (d[fits] * 3 == d3[fits]).all()


def test_gewichte_und_kapazitaet_mit_k_multipliziert_gleicher_status():
    for seed in range(20):
        v, w, cap = _inst(seed)
        st, d, fits = K.status_schnell(v, w, cap)
        st2, d2, fits2 = K.status_schnell(v, 2 * w, 2 * cap)
        assert (st == st2).all() and (d == d2).all() and (fits == fits2).all()


def test_pakete_umnummeriert_status_permutiert():
    rng = np.random.default_rng(0)
    for seed in range(20):
        v, w, cap = _inst(seed)
        perm = rng.permutation(len(v))
        st, d, _ = K.status_schnell(v, w, cap)
        st2, d2, _ = K.status_schnell(v[perm], w[perm], cap)
        assert (st[perm] == st2).all() and (d[perm] == d2).all()


def test_zwei_identische_pakete_haben_denselben_status():
    for seed in range(30):
        v, w, cap = _inst(seed)
        v, w = np.append(v, v[2]), np.append(w, w[2])
        st, d, fits = K.status_schnell(v, w, cap)
        assert st[2] == st[-1] and d[2] == d[-1]


def test_zusaetzliches_paket_verschlechtert_den_optimalwert_nie_und_mehr_kapazitaet_ebenso():
    for seed in range(30):
        v, w, cap = _inst(seed)
        base = int(K.dp_best(v, w, cap)[cap])
        assert int(K.dp_best(np.append(v, 5), np.append(w, 7), cap)[cap]) >= base
        opt = K.dp_best(v, w, cap + 20)
        assert all(a <= b for a, b in zip(opt, opt[1:]))


def test_status_ist_in_der_kapazitaet_nicht_monoton_gegenbeispiel(hb):
    v, w, _ = hb
    row = np.sign(K.diff_table(v, w, 16)[1])
    assert row[7] == 1 and row[8] == -1                                                # bei 7 drin, bei 8 draußen, obwohl der Optimalwert steigt
    assert K.nichtmonoton(row)


def test_kippabstand_ist_nicht_negativ_und_null_genau_im_gleichstand():
    for seed in range(30):
        v, w, cap = _inst(seed, 8, "stark korreliert", 1)
        st, d, fits = K.status_schnell(v, w, cap)
        assert ((np.abs(d[fits]) == 0) == (st[fits] == 0)).all() and (np.abs(d[fits]) >= 0).all()


def test_schaetzung_gleich_wahren_werten_macht_jede_anhebung_bei_aufschlag_0_gueltig():
    for seed in range(20):
        v, w, cap = _inst(seed, 12)
        a = E.auswertung(v, w, cap, v.copy(), 0.1)
        assert a["rel0"].all() and a["rel100"].all() and (np.asarray(a["m_hat"]) == np.asarray(a["m_wahr"])).all()


def test_absoluter_abstand_wirkt_monoton():
    rng = SplitMix64(5)
    for seed in range(20):
        v, w, cap = _inst(seed, 12, "stark korreliert", 1)
        a = E.auswertung(v, w, cap, E.schaetzung(v, 0.1, rng), 0.1)
        assert (a["abs2"] <= a["abs4"]).all() and (a["abs4"] <= a["abs8"]).all() and (a["rel0"] <= a["rel25"]).all() and (a["rel50"] <= a["rel100"]).all()


# ------------------------------------------------------------------------------------------------------------------------------------- Randfälle

def test_kapazitaet_null_kein_paket_passt():
    v, w = np.array([5, 6]), np.array([3, 4])
    st, d, fits = K.status_schnell(v, w, 0)
    assert not fits.any() and (st == -1).all() and (d == K.NIE).all()                  # nichts passt: Status -1 nur formal, `fits` ist falsch (die Anzeige sagt „passt nie“)
    assert K.dp_best(v, w, 0).tolist() == [0]


def test_paket_ueber_der_kapazitaet_passt_nie_und_zaehlt_nicht_als_sicher_draussen():
    v, w = np.array([10, 4, 3]), np.array([20, 2, 2])
    st, d, fits = K.status_schnell(v, w, 5)
    assert fits.tolist() == [False, True, True] and st.tolist() == [-1, 1, 1]          # Paket 1 passt nie (Gewicht 20 über Kapazität 5); die beiden anderen passen zusammen (Gewicht 4)


def test_ein_paket_und_alle_passen_alle_sicher_drin():
    st, d, fits = K.status_schnell(np.array([7]), np.array([3]), 5)
    assert st.tolist() == [1] and d.tolist() == [7]
    v, w = np.array([4, 5, 6]), np.array([1, 1, 1])
    st, d, fits = K.status_schnell(v, w, 10)
    assert st.tolist() == [1, 1, 1] and d.tolist() == [4, 5, 6]


def test_alle_werte_gleich_und_wert_null_gleichstand():
    st, d, fits = K.status_schnell(np.array([5, 5, 5]), np.array([2, 2, 2]), 4)           # zwei von drei gleich guten Paketen: alle im Gleichstand
    assert st.tolist() == [0, 0, 0] and d.tolist() == [0, 0, 0]
    st, d, _ = K.status_schnell(np.array([0, 9]), np.array([3, 3]), 3)
    assert st.tolist() == [-1, 1]
    st, d, _ = K.status_schnell(np.array([0, 0]), np.array([1, 1]), 5)
    assert st.tolist() == [0, 0]


def test_gleichstand_zwischen_allen_mengen():
    st, d, fits = K.status_schnell(np.array([3, 3, 3, 3]), np.array([1, 1, 1, 1]), 2)
    assert (st == 0).all()
    vm, ste, nopt, sub = K.enum_status(np.array([3, 3, 3, 3]), np.array([1, 1, 1, 1]), 2)
    assert nopt == 6 and vm == 6 and (ste == 0).all()


def test_greedy_und_regel_s_unterscheiden_sich_auf_mindestens_einer_instanz_und_sind_zulaessig():
    verschieden = 0
    for seed in range(60):
        v, w, cap = _inst(seed, 8, "unkorreliert", 1)
        xg, xs = RU.greedy_regel(v, w, cap), RU.schwellenregel(v, w, cap)
        assert int(w @ xg) <= cap and int(w @ xs) <= cap and (xs <= xg).all()
        verschieden += int((xg != xs).any())
        assert int(v @ xg) <= int(K.dp_best(v, w, cap)[cap])
    assert verschieden >= 5                                                            # kein toter Zweig: Regel S weicht von Greedy ab
    with pytest.raises(ValueError):
        RU.regel("xyz", v, w, cap)


def test_nahbereich_und_nichtmonoton_randfaelle():
    assert K.nahbereich(np.array([1, 1, 1, -1, -1]), 1) == pytest.approx(2.0)       # erste Änderung bei Kapazität 3: |3 - 1| / 1 = 2
    assert K.nahbereich(np.array([1, 1, 1]), 1) == float("inf")
    assert not K.nichtmonoton(np.array([-1, -1, 1, 1])) and K.nichtmonoton(np.array([-1, 1, 0, -1]))
