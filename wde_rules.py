"""Einfache Regeln für den Rucksack: Greedy nach Verhältnis v/w (die „lesbare“ Regel) und Regel S (bis zum kritischen Paket).

Beide sortieren nach v/w absteigend (Gleichstand: kleinerer Index). Greedy überspringt Nichtpassendes und füllt Lücken, Regel S hört beim ersten Paket auf, das nicht mehr passt
(kritisches Paket der LP-Relaxation), und lässt dieses und alle weiteren draußen."""

import numpy as np


def _ordnung(v, w):
    return sorted(range(len(v)), key=lambda j: (-(v[j] / w[j]), j))


def greedy_regel(v, w, C_):
    x = np.zeros(len(v), dtype=np.int64)
    rest = C_
    for j in _ordnung(v, w):
        if w[j] <= rest:
            x[j] = 1
            rest -= int(w[j])
    return x


def schwellenregel(v, w, C_):
    x = np.zeros(len(v), dtype=np.int64)
    rest = C_
    for j in _ordnung(v, w):
        if w[j] <= rest:
            x[j] = 1
            rest -= int(w[j])
        else:
            break
    return x


def regel(name, v, w, C_):
    if name == "greedy":
        return greedy_regel(v, w, C_)
    if name == "schwelle":
        return schwellenregel(v, w, C_)
    raise ValueError(name)
