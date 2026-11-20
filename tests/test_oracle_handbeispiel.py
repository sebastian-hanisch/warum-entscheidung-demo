"""Orakel-Test Handbeispiel: vier Pakete, Gewichte 4, 3, 3, 5, Werte 6, 7, 9, 9, Kapazität 8. Alle Zahlen von Hand gerechnet, die Demo-Funktionen dagegen geprüft.

Handrechnung (Plan, Handbeispiel Teil 1 bis 5, 7 und 9):
 1. Zehn zulässige Mengen (Gewicht <= 8): leer 0, {1} 6, {2} 7, {3} 9, {4} 9, {1,2} 13, {1,3} 15, {2,3} 16, {2,4} 16, {3,4} 18. Optimum 18 mit {3, 4}, eindeutig, Status (-1, -1, +1, +1).
 2. Paket 2: B = 18 (bester Wert ohne Paket 2), A0 = 9 (ohne Paket 2 bei Kapazität 5), A = 7 + 9 = 16, Wertschwelle B - A = 2; bei Wert 8 bleibt es draußen, bei 9 Gleichstand mit drei optimalen
    Mengen {3,4}, {2,3}, {2,4} (je 18), bei 10 in jeder optimalen Menge (19). Paket 4: A - B = 18 - 16 = 2.
 3. Greedy nach Verhältnis (3, 7/3, 9/5, 3/2): nimmt 3 und 2, findet 16 gegen 18.
 5. Kapazitäten: Optimalwerte 0, 0, 0, 9, 9, 9, 16, 16, 18, 18, 22, 25, 25, 25, 25, 31, 31 für C = 0 bis 16; Paket 2 drin bei C = 6 und 7, draußen bei 8 und 9, drin bei 10.
 9. Schätzung (6, 8, 9, 9): geschätzte Schwelle 1 gegen wahre 2; gültig ab Aufschlag 100 %."""

import itertools
from fractions import Fraction as F

import numpy as np

import wde_estimates as E
import wde_knapsack as K
import wde_rules as RU

V, W, CAP = (6, 7, 9, 9), (4, 3, 3, 5), 8


def _opt(v, w, cap):
    zul = [x for x in itertools.product((0, 1), repeat=len(v)) if sum(a * b for a, b in zip(x, w)) <= cap]
    vm = max(sum(a * b for a, b in zip(x, v)) for x in zul)
    return vm, [x for x in zul if sum(a * b for a, b in zip(x, v)) == vm]


def _status(mengen, i):
    ins = [x[i] for x in mengen]
    return 1 if all(ins) else (-1 if not any(ins) else 0)


def test_zehn_zulaessige_mengen_und_optimum(hb):
    v, w, cap = hb
    zul = {x: sum(a * b for a, b in zip(x, V)) for x in itertools.product((0, 1), repeat=4) if sum(a * b for a, b in zip(x, W)) <= CAP}
    assert len(zul) == 10 and sorted(zul.values()) == [0, 6, 7, 9, 9, 13, 15, 16, 16, 18]
    vm, ste, nopt, sub = K.enum_status(v, w, cap)
    assert (vm, nopt, sub.tolist(), ste.tolist()) == (18, 1, [[0, 0, 1, 1]], [-1, -1, 1, 1])
    st, d, fits = K.status_schnell(v, w, cap)
    assert st.tolist() == [-1, -1, 1, 1] and fits.all()


def test_wertschwelle_paket_2_und_4_von_hand(hb):
    v, w, cap = hb
    st, d, _ = K.status_schnell(v, w, cap)
    assert d[1] == 16 - 18 and d[3] == 18 - 16                           # A - B: Paket 2 (-2: sicher draußen), Paket 4 (+2: sicher drin)
    B, A0 = K.thresholds_fast(v, w, cap)
    assert (int(B[1]), int(A0[1])) == (18, 9) and int(v[1]) + int(A0[1]) == 16
    for wert, erwartet in ((8, (-1, 18, 1)), (9, (0, 18, 3)), (10, (1, 19, 2))):
        v2 = list(V)
        v2[1] = wert
        vm, mengen = _opt(v2, W, CAP)
        assert (_status(mengen, 1), vm, len(mengen)) == erwartet
    v2 = list(V)
    v2[1] = 9
    assert sorted(_opt(v2, W, CAP)[1]) == [(0, 0, 1, 1), (0, 1, 0, 1), (0, 1, 1, 0)]       # {3,4}, {2,4}, {2,3}
    for wert, erwartet in ((8, (1, 17, 1)), (7, (0, 16, 2)), (6, (-1, 16, 1))):          # Paket 4 fällt von 9
        v2 = list(V)
        v2[3] = wert
        vm, mengen = _opt(v2, W, CAP)
        assert (_status(mengen, 3), vm, len(mengen)) == erwartet


def test_preis_der_erzwingung_und_relativer_kippabstand(hb):
    v, w, cap = hb
    vm, _ = _opt(V, W, CAP)
    mit2 = max(sum(a * b for a, b in zip(x, V)) for x in itertools.product((0, 1), repeat=4) if x[1] == 1 and sum(a * b for a, b in zip(x, W)) <= CAP)
    mit1 = max(sum(a * b for a, b in zip(x, V)) for x in itertools.product((0, 1), repeat=4) if x[0] == 1 and sum(a * b for a, b in zip(x, W)) <= CAP)
    assert (vm - mit2, vm - mit1) == (2, 3)
    d = K.status_schnell(v, w, cap)[1]
    assert (-d[1], -d[0]) == (2, 3) and F(int(-d[1]), int(v[1])) == F(2, 7)


def test_greedy_und_regel_s(hb):
    v, w, cap = hb
    xg = RU.greedy_regel(v, w, cap)
    assert xg.tolist() == [0, 1, 1, 0] and int(v @ xg) == 16                                   # 16 gegen 18
    assert RU.schwellenregel(v, w, cap).tolist() == [0, 1, 1, 0]                               # kritisches Paket ist 4: dieselbe Menge
    st = K.status_schnell(v, w, cap)[0]
    assert int(((xg == 1) == (st == 1)).sum()) == 2                                            # Pakete 1 und 3 richtig, 2 und 4 verfehlt
    assert sorted(range(4), key=lambda j: (-F(V[j], W[j]), j)) == [2, 1, 3, 0]                 # Verhältnisse 3, 7/3, 9/5, 3/2


def test_kapazitaetsfolge_und_nichtmonotonie_von_paket_2(hb):
    v, w, _ = hb
    strf, opt = K.kapazitaetsstreifen(v, w, 16)
    assert opt.tolist() == [0, 0, 0, 9, 9, 9, 16, 16, 18, 18, 22, 25, 25, 25, 25, 31, 31]
    assert [int(strf[1][c]) for c in (6, 7, 8, 9, 10)] == [1, 1, -1, -1, 1]                    # drin, drin, draußen, draußen, drin
    assert [_status(_opt(V, W, c)[1], 1) for c in (6, 7, 8, 9, 10)] == [1, 1, -1, -1, 1]       # Aufzählung je Kapazität
    assert int(strf[2][5]) == 0 and int(strf[3][5]) == 0 and (_opt(V, W, 5)[0], sorted(_opt(V, W, 5)[1])) == (9, [(0, 0, 0, 1), (0, 0, 1, 0)])   # C = 5: Gleichstand {3} und {4}
    assert opt[8] > opt[7] and K.nichtmonoton(np.sign(K.diff_table(v, w, 16)[1]))               # der Optimalwert steigt, Paket 2 fliegt trotzdem hinaus
    assert all(a <= b for a, b in zip(opt, opt[1:]))                                           # der Optimalwert fällt in C nie


def test_schaetzwerte_von_hand(hb):
    v, w, cap = hb
    vh = np.array([6, 8, 9, 9])
    assert int(E.schwellen(vh, w, cap)[1]) == 1 and int(E.schwellen(v, w, cap)[1]) == 2        # geschätzte Schwelle 1 gegen wahre 2
    m_hat, m_wahr = 1, 2
    assert [E.gueltig(m_hat * (1 + a), m_wahr) for a in (0, 0.5, 1)] == [False, False, True]   # gültig genau ab Aufschlag 100 %
    v2 = [F(x) for x in V]
    v2[1] += m_wahr                                                                            # genau um die wahre Schwelle angehoben: nur Gleichstand
    assert _status(_opt([int(2 * x) for x in v2], [int(x) for x in W], CAP)[1], 1) == 0        # Werte verdoppelt, damit ganzzahlig: gleicher Status
    a = E.auswertung(v, w, cap, vh, 0.1)
    assert a["index"].tolist() == [0, 1]                                                       # Pakete 1 und 2 sind in Schätzung und Wahrheit sicher draußen
    j = a["index"].tolist().index(1)
    assert (a["m_hat"][j], a["m_wahr"][j], bool(a["rel0"][j]), bool(a["rel100"][j])) == (1, 2, False, True)
