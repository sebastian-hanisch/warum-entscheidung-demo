"""Orakel-Test Reihenfolge (Tab): Aufzählung aller 5 040 Reihenfolgen gegen Smith-Regel, geschlossene Gegenfakten, Dispatch und ein Big-M-Programm (scipy, nur im Test).

Handrechnung (Handbeispiel Teil 7): p = (1, 2, 3), w = (3, 2, 1): Smith-Reihenfolge nach p/w (1/3, 1, 3) = 1, 2, 3, Zielwert 3*1 + 2*3 + 1*6 = 15, eindeutig. Senkung von w_2 um
d_w = 2 - floor(2*1/3) = 2; Erhöhung von p_2 um d_p = ceil(3*2/1) - 2 = 4 (p_2 = 5 sicher, p_2 = 6 Gleichstand, beide Quotienten 3). Mit Freigaben p = (2, 1), w = (1, 5), r = (0, 10):
statische Smith-Reihenfolge (2 zuerst) kostet 5*11 + 1*13 = 68, Auftrag 1 zuerst 2 + 5*11 = 57."""

import itertools
from fractions import Fraction as F

import numpy as np
import pytest

import wde_schedule as S


def _kosten(perm, p, w, r):
    t, tot = 0, 0
    for j in perm:
        t = max(t, int(r[j])) + int(p[j])
        tot += int(w[j]) * t
    return tot


def _opt_klein(p, w, r):
    perms = list(itertools.permutations(range(len(p))))
    best = min(_kosten(q, p, w, r) for q in perms)
    return best, [q for q in perms if _kosten(q, p, w, r) == best]


def test_handbeispiel_ohne_freigaben_von_hand():
    p, w, r = (1, 2, 3), (3, 2, 1), (0, 0, 0)
    best, perms = _opt_klein(p, w, r)
    assert best == 15 and perms == [(0, 1, 2)]
    assert sorted(range(3), key=lambda j: (F(p[j], w[j]), j)) == [0, 1, 2]
    d_w, d_p = S.gegenfaktum_geschlossen(np.array(p), np.array(w), 1, 2)
    assert (d_w, d_p) == (2, 4)
    for wa, erwartet in ((2, True), (1, True), (0, False)):
        w2 = list(w)
        w2[1] = wa
        _, q = _opt_klein(p, w2, r)
        assert all(x.index(1) < x.index(2) for x in q) == erwartet                       # mit w_2 = 1 noch sicher vor 3, mit w_2 = 0 nicht mehr
    for pa, status in ((5, "sicher"), (6, "gleich")):
        p2 = list(p)
        p2[1] = pa
        _, q = _opt_klein(p2, w, r)
        vor = [x.index(1) < x.index(2) for x in q]
        assert (all(vor) if status == "sicher" else (any(vor) and not all(vor)))


def test_handbeispiel_mit_freigaben():
    p, w, r = (2, 1), (1, 5), (0, 10)
    best, perms = _opt_klein(p, w, r)
    assert best == 57 and perms == [(0, 1)]
    assert S.smith_statisch(np.array(p), np.array(w)) == [1, 0] and _kosten([1, 0], p, w, r) == 68
    assert S.dispatch(np.array(p), np.array(w), np.array(r)) == [0, 1] and _kosten([0, 1], p, w, r) == 57          # der Dispatch wartet nicht auf Auftrag 2


def test_aufzaehlung_gegen_smith_und_dispatch_ohne_freigaben():
    for k in range(40):
        p, w, r = S.instanz(50_000_000 + k, 0)
        Cm = S.fertigstellung(p, r)
        om, mask = S.optimum(Cm, w)
        assert S.zielwert(S.smith_statisch(p, w), p, w, r) == om == S.zielwert(S.dispatch(p, w, r), p, w, r)
        # Reihenfolgen aus der Python-Schleife (unabhängig von der numpy-Aufzählung) für drei Perms
        j = int(np.nonzero(mask)[0][0])
        assert _kosten(S.PERMS[j], p, w, r) == om


def test_paarstatus_gleich_smith_vergleich_der_quotienten():
    for k in range(40):
        p, w, r = S.instanz(50_000_000 + k, 0)
        om, mask = S.optimum(S.fertigstellung(p, r), w)
        for a in range(7):
            for b in range(7):
                if a != b:
                    assert (S.paarstatus(mask, a, b) == 0) == (p[a] * w[b] == p[b] * w[a])          # im Gleichstand genau dann, wenn die Quotienten gleich sind
                    if p[a] * w[b] < p[b] * w[a]:
                        assert S.paarstatus(mask, a, b) == 1


def test_geschlossene_gegenfakten_gleich_aufzaehlung():
    geprueft = 0
    for k in range(6):
        p, w, r = S.instanz(50_000_000 + k, 0)
        om, mask = S.optimum(S.fertigstellung(p, r), w)
        for a in range(7):
            for b in range(7):
                if a == b or S.paarstatus(mask, a, b) != 1:
                    continue
                d_w, d_p = S.gegenfaktum_geschlossen(p, w, a, b)
                e_w, e_p = S.gegenfaktum_aufzaehlung(p, w, r, a, b)
                assert d_w == e_w and (d_p if d_p <= 29 else None) == e_p, (k, a, b)
                geprueft += 1
    assert geprueft > 100


def test_big_m_programm_gleich_aufzaehlung():
    pytest.importorskip("scipy.optimize")
    from scipy.optimize import Bounds, LinearConstraint, milp
    n = 7
    for rmax in (0, 25):
        for k in range(1):
            p, w, r = S.instanz(50_000_000 + 1000 * rmax + k, rmax)
            om, _ = S.optimum(S.fertigstellung(p, r), w)
            M = int(r.max() + p.sum())
            pairs = list(itertools.combinations(range(n), 2))
            nv = n + len(pairs)
            rows, lo = [], []
            for idx, (j, q) in enumerate(pairs):
                a = np.zeros(nv)
                a[q], a[j], a[n + idx] = 1, -1, -M
                rows.append(a)
                lo.append(p[q] - M)
                a = np.zeros(nv)
                a[j], a[q], a[n + idx] = 1, -1, M
                rows.append(a)
                lo.append(p[j])
            res = milp(np.concatenate([w.astype(float), np.zeros(len(pairs))]), constraints=LinearConstraint(np.array(rows), lo, np.inf),
                       bounds=Bounds(np.concatenate([r + p, np.zeros(len(pairs))]).astype(float), np.concatenate([np.full(n, np.inf), np.ones(len(pairs))])),
                       integrality=np.concatenate([np.zeros(n), np.ones(len(pairs))]))
            assert round(res.fun) == om, (rmax, k)


def test_eigenschaften_umnummerieren_und_gewichte_skalieren():
    p, w, r = S.instanz(777, 10)
    om, mask = S.optimum(S.fertigstellung(p, r), w)
    om2, _ = S.optimum(S.fertigstellung(p, r), 4 * w)
    assert om2 == 4 * om                                                                # alle Gewichte mit c > 0 multipliziert: Zielwert skaliert, Reihenfolge bleibt
    assert (S.PERMS[mask] == S.PERMS[S.optimum(S.fertigstellung(p, r), 4 * w)[1]]).all()
    perm = np.array([3, 1, 0, 6, 2, 5, 4])
    om3, _ = S.optimum(S.fertigstellung(p[perm], r[perm]), w[perm])
    assert om3 == om                                                                    # Umnummerieren ändert nur die Namen


def test_randfall_dispatch_wartet_bis_zur_naechsten_freigabe_und_mit_freigaben_nicht_exakt():
    seq = S.dispatch(np.array([2, 2]), np.array([1, 1]), np.array([5, 0]))
    assert seq == [1, 0]
    schlechter = 0
    for k in range(40):
        p, w, r = S.instanz(50_000_000 + 10_000 + k, 25)
        om, _ = S.optimum(S.fertigstellung(p, r), w)
        schlechter += int(S.zielwert(S.smith_statisch(p, w), p, w, r) > om)
        assert S.zielwert(S.dispatch(p, w, r), p, w, r) >= om
    assert schlechter >= 20                                                             # mit Freigaben verfehlt die statische Regel den Optimalwert meistens
