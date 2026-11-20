"""Orakel-Test Rucksack: DP-Status und Wertschwelle gegen unabhängige Rechenwege.

(a) Aufzählung aller 2^n Mengen (reine Python-Schleife über itertools, nicht die numpy-Aufzählung der Demo), (b) Wertschwelle: Status bei d-1 / d / d+1 geänderten Werten durch Aufzählung,
(c) Preis der Erzwingung durch Aufzählung nur der Mengen mit dem Paket, (d) schnelle Schwellen gegen die Statustabelle über alle Kapazitäten und gegen Aufzählung je Kapazität,
(e) ganzzahliges Programm (scipy milp) im Testlauf, nicht in der App. Alle auf festen Zufalls-Seeds, kleine Instanzen."""

import itertools

import numpy as np
import pytest

import wde_knapsack as K


def _enum(v, w, cap):
    """(Optimalwert, optimale Mengen als Tupel) durch Aufzählung mit Python-Schleifen."""
    best, mengen = -1, []
    for x in itertools.product((0, 1), repeat=len(v)):
        if sum(a * b for a, b in zip(x, w)) > cap:
            continue
        val = sum(a * b for a, b in zip(x, v))
        if val > best:
            best, mengen = val, [x]
        elif val == best:
            mengen.append(x)
    return best, mengen


def _status(mengen, i):
    ins = [x[i] for x in mengen]
    return 1 if all(ins) else (-1 if not any(ins) else 0)


INSTANZEN = [(seed, n, fam, s, kap) for seed in range(6) for n in (6, 8) for fam in ("unkorreliert", "stark korreliert") for s in (1, 10) for kap in (0.3, 0.5, 0.7)]


@pytest.mark.parametrize("seed,n,fam,s,kap", INSTANZEN[::3])
def test_dp_status_gleich_aufzaehlung(seed, n, fam, s, kap):
    v, w, cap = K.instanz(seed, n, fam, s, kap)
    vm, mengen = _enum([int(a) for a in v], [int(a) for a in w], cap)
    st, d, fits = K.status_schnell(v, w, cap)
    assert int(K.dp_best(v, w, cap)[cap]) == vm
    for i in range(n):
        if fits[i]:
            assert st[i] == _status(mengen, i), (i, st[i])
    vm2, ste, nopt, sub = K.enum_status(v, w, cap)                # die numpy-Aufzählung der Demo gegen die Python-Schleife
    assert vm2 == vm and nopt == len(mengen) and sorted(map(tuple, sub.tolist())) == sorted(mengen)


def test_wertschwelle_status_bei_d_minus_1_d_und_d_plus_1():
    geprueft = 0
    for seed in range(8):
        v, w, cap = K.instanz(seed, 8, "unkorreliert", 1, 0.5)
        st, d, fits = K.status_schnell(v, w, cap)
        for i in range(8):
            if not fits[i] or st[i] == 0:
                continue
            betrag, sgn = abs(int(d[i])), (-1 if st[i] == 1 else 1)
            res = []
            for dd in (betrag - 1, betrag, betrag + 1):
                v2 = [int(a) for a in v]
                v2[i] += sgn * dd
                _, mengen = _enum(v2, [int(a) for a in w], cap)
                res.append(_status(mengen, i))
            assert betrag >= 1 and tuple(res) == (st[i], 0, -st[i]), (seed, i)
            geprueft += 1
    assert geprueft == 62                                                       # 62 Pakete mit eindeutigem Status in 8 Instanzen (n = 8, unkorreliert, Skala 1, Kapazität 50 %)


def test_preis_der_erzwingung_ist_gleich_der_wertschwelle():
    for seed in range(8):
        v, w, cap = K.instanz(seed, 8, "stark korreliert", 1, 0.5)
        vm, _ = _enum([int(a) for a in v], [int(a) for a in w], cap)
        st, d, fits = K.status_schnell(v, w, cap)
        for i in range(8):
            if not fits[i]:
                continue
            mit = max(sum(a * b for a, b in zip(x, v)) for x in itertools.product((0, 1), repeat=8) if x[i] == 1 and sum(a * b for a, b in zip(x, w)) <= cap)
            assert vm - mit == max(0, -int(d[i])), (seed, i)                  # Erzwingen kostet B - A = -d (0 bei drin oder Gleichstand)


def test_schnelle_schwellen_gleich_statustabelle_und_aufzaehlung_je_kapazitaet():
    for seed in range(5):
        v, w, cap = K.instanz(seed, 8, "unkorreliert", 1, 0.7)
        tab = K.diff_table(v, w, int(w.sum()))
        for c in (0, 5, cap // 2, cap, int(w.sum())):
            st, d, fits = K.status_schnell(v, w, c) if c > 0 else (None, None, None)
            if c == 0:
                continue
            assert (d == tab[:, c]).all()
            vm, mengen = _enum([int(a) for a in v], [int(a) for a in w], c)
            for i in range(8):
                if fits[i]:
                    assert st[i] == _status(mengen, i), (seed, c, i)


def test_dp_gegen_ganzzahliges_programm_milp():
    opt = pytest.importorskip("scipy.optimize")
    from scipy.optimize import Bounds, LinearConstraint, milp
    for seed in range(4):
        for n, s in ((12, 1), (20, 10)):
            v, w, cap = K.instanz(seed, n, "unkorreliert", s, 0.5)
            res = milp(-v.astype(float), constraints=LinearConstraint(w.astype(float)[None, :], -np.inf, cap), bounds=Bounds(np.zeros(n), np.ones(n)), integrality=np.ones(n))
            assert int(round(-res.fun)) == int(K.dp_best(v, w, cap)[cap])
            st, d, fits = K.status_schnell(v, w, cap)
            for i in range(0, n, 4):                                           # Paket erzwungen und verboten
                lb, ub = np.zeros(n), np.ones(n)
                lb[i] = 1
                mit = milp(-v.astype(float), constraints=LinearConstraint(w.astype(float)[None, :], -np.inf, cap), bounds=Bounds(lb, ub), integrality=np.ones(n))
                lb[i], ub[i] = 0, 0
                ohne = milp(-v.astype(float), constraints=LinearConstraint(w.astype(float)[None, :], -np.inf, cap), bounds=Bounds(lb, ub), integrality=np.ones(n))
                a = int(round(-mit.fun)) if mit.status == 0 else None
                b = int(round(-ohne.fun))
                if fits[i]:
                    assert np.sign(a - b) == st[i], (seed, n, i)
                    assert a - b == d[i]


def test_dp_set_gibt_eine_optimale_menge():
    for seed in range(6):
        v, w, cap = K.instanz(seed, 12, "unkorreliert", 1, 0.5)
        x, val = K.dp_set(v, w, cap)
        assert int(w @ x) <= cap and int(v @ x) == int(val) == int(K.dp_best(v, w, cap)[cap])
