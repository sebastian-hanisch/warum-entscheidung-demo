"""Reihenfolge-Entscheidung (Tab): Einmaschinen-Problem 1 | r_j | Summe w_j C_j mit n = 7 Aufträgen (5 040 Reihenfolgen), erfunden.

Auftrag j hat Dauer p_j (1 bis 9), Gewicht w_j (1 bis 5) und Freigabe r_j (0 bis rmax); Ziel: Summe w_j C_j. Frage „Warum Auftrag A vor B?“: A steht in JEDER optimalen Reihenfolge vor B (+1),
in keiner (-1), in mancher (0).
Smith-Regel (Lehrbuch, Vertauschungsargument), ohne Freigaben: A steht in jeder optimalen Reihenfolge vor B genau dann, wenn p_A / w_A < p_B / w_B. Daraus die geschlossenen Gegenfakten:
  kleinste ganzzahlige Senkung des Gewichts w_A, nach der A nicht mehr sicher vor B steht:  d_w = w_A - floor(p_A w_B / p_B);
  kleinste ganzzahlige Erhöhung der Dauer p_A:                                              d_p = ceil(p_B w_A / w_B) - p_A.
Mit Freigaben gibt es keine einfache Formel: die Aufzählung bleibt die Wahrheit je Frage, gemessen werden die statische Smith-Reihenfolge und der Dispatch."""

import itertools

import numpy as np

from wde_knapsack import uni
from wde_rng import SplitMix64

N = 7
PERMS = np.array(list(itertools.permutations(range(N))), dtype=np.int64)
POS = np.argsort(PERMS, axis=1)
ROWS = np.arange(len(PERMS))


def instanz(seed, rmax, n=N):
    """(p, w, r) aus dem Zufalls-Seed: p in 1..9, w in 1..5, r in 0..rmax (rmax = 0: alle 0)."""
    rng = SplitMix64(seed)
    p = np.array([uni(rng, 1, 9) for _ in range(n)], dtype=np.int64)
    w = np.array([uni(rng, 1, 5) for _ in range(n)], dtype=np.int64)
    r = np.array([uni(rng, 0, rmax) for _ in range(n)], dtype=np.int64) if rmax > 0 else np.zeros(n, dtype=np.int64)
    return p, w, r


def fertigstellung(p, r):
    """Fertigstellungszeiten C_j je Reihenfolge (5 040 x n)."""
    t = np.zeros(len(PERMS), dtype=np.int64)
    Cm = np.zeros((len(PERMS), len(p)), dtype=np.int64)
    for k in range(len(p)):
        j = PERMS[:, k]
        s = np.maximum(t, r[j])
        t = s + p[j]
        Cm[ROWS, j] = t
    return Cm


def optimum(Cm, w):
    """(Optimalwert, Maske der optimalen Reihenfolgen)."""
    obj = Cm @ w
    om = obj.min()
    return int(om), obj == om


def paarstatus(mask, a, b):
    """+1: A steht in jeder optimalen Reihenfolge vor B; -1: in keiner; 0: in mancher."""
    vor = POS[mask, a] < POS[mask, b]
    return 1 if vor.all() else (-1 if (~vor).all() else 0)


def zielwert(seq, p, w, r):
    t, tot = 0, 0
    for j in seq:
        t = max(t, int(r[j])) + int(p[j])
        tot += int(w[j]) * t
    return tot


def smith_statisch(p, w):
    """Reihenfolge nach p/w aufsteigend (Gleichstand: kleinerer Index)."""
    return sorted(range(len(p)), key=lambda j: (p[j] / w[j], j))


def dispatch(p, w, r):
    """Immer den freigegebenen Auftrag mit kleinstem p/w; ist keiner frei, bis zur nächsten Freigabe warten."""
    n = len(p)
    t, fertig, seq = 0, set(), []
    while len(fertig) < n:
        frei = [j for j in range(n) if j not in fertig and r[j] <= t]
        if not frei:
            t = min(int(r[j]) for j in range(n) if j not in fertig)
            continue
        j = min(frei, key=lambda q: (p[q] / w[q], q))
        seq.append(j)
        fertig.add(j)
        t += int(p[j])
    return seq


def paare_richtig(seq, sicher):
    """Zahl der sicheren geordneten Paare (A, B), die die Reihenfolge `seq` wie die optimalen Reihenfolgen ordnet (A vor B)."""
    pos = np.empty(len(seq), dtype=int)
    pos[list(seq)] = np.arange(len(seq))
    return sum(int(pos[a] < pos[b]) for (a, b) in sicher)


def gegenfaktum_geschlossen(p, w, a, b):
    """(d_w, d_p) für ein sicheres Paar (A vor B) ohne Freigaben: kleinste ganzzahlige Senkung von w_A und Erhöhung von p_A, nach der A nicht mehr sicher vor B steht."""
    d_w = int(w[a]) - (int(p[a]) * int(w[b])) // int(p[b])
    d_p = -(-(int(p[b]) * int(w[a])) // int(w[b])) - int(p[a])
    return d_w, d_p


def gegenfaktum_aufzaehlung(p, w, r, a, b, dmax_p=29):
    """Dasselbe durch Aufzählung (Orakel): kleinste Senkung von w_A (bis w_A) und kleinste Erhöhung von p_A (bis dmax_p), nach der Paar A vor B nicht mehr sicher ist; None, wenn nie."""
    Cm = fertigstellung(p, r)
    d_w = None
    for d in range(1, int(w[a]) + 1):
        w2 = w.copy()
        w2[a] -= d
        _, m2 = optimum(Cm, w2)
        if paarstatus(m2, a, b) != 1:
            d_w = d
            break
    d_p = None
    for d in range(1, dmax_p + 1):
        p2 = p.copy()
        p2[a] += d
        _, m2 = optimum(fertigstellung(p2, r), w)
        if paarstatus(m2, a, b) != 1:
            d_p = d
            break
    return d_w, d_p
