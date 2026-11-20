"""Rucksack „Fuhre packen“ (erfunden): Instanz, dynamische Programmierung, Status je Paket, Wertschwelle, Statustabelle über alle Kapazitäten, Aufzählung.

n Pakete mit ganzzahligem Gewicht w in [5s, 30s] und Wert v in [10s, 60s] („unkorreliert“) oder v = w + 10s („stark korreliert“); Kapazität C = floor(Anteil * Summe w);
s ist die Zahlenskala (1 grob, 10 fein). Alles ganzzahlig, deshalb sind Gleichstände zählbar. Die Familien sind nicht wortgleich mit denen anderer Demos der Rucksack-Linie.

Status eines Pakets bei Kapazität C: +1 sicher drin (in JEDER optimalen Menge), -1 sicher draußen (in KEINER), 0 Gleichstand (in mancher).
Wertschwelle (Kippabstand): d = |A - B| mit A = bester Wert mit Paket i, B = bester Wert ohne Paket i (beide bei Kapazität C); B - A > 0 heißt sicher draußen, A - B > 0 sicher drin,
0 Gleichstand. Der Preis der Erzwingung („Paket i zwingend packen“) ist B - A und gleich der Wertschwelle (Identität, weil A als bester Wert mit Paket i erklärt ist)."""

import numpy as np

import wde_constants as C
from wde_rng import SplitMix64

NIE = -(10 ** 12)  # „passt nie“: Gewicht größer als die Kapazität


def uni(rng, lo, hi):
    """Ganzzahl gleichverteilt in lo..hi (einschließlich): lo + below(hi - lo + 1)."""
    return lo + rng.below(hi - lo + 1)


def instanz(seed, n, fam, skala=1, kap=0.5):
    """Instanz aus dem Zufalls-Seed: Werte v, Gewichte w (numpy-Arrays, ganzzahlig) und Kapazität C."""
    rng = SplitMix64(seed)
    w = np.array([uni(rng, 5 * skala, 30 * skala) for _ in range(n)], dtype=np.int64)
    if fam == "unkorreliert":
        v = np.array([uni(rng, 10 * skala, 60 * skala) for _ in range(n)], dtype=np.int64)
    elif fam == "stark korreliert":
        v = w + 10 * skala
    else:
        raise ValueError(fam)
    return v, w, int(np.floor(kap * int(w.sum())))


def dp_best(v, w, C_, skip=None):
    """f(c), c = 0..C_: bester Wert aller Pakete außer `skip` bei Kapazität höchstens c (0/1-Rucksack)."""
    f = np.zeros(C_ + 1, dtype=np.asarray(v).dtype)
    for j in range(len(v)):
        if j == skip or w[j] > C_:
            continue
        wj = int(w[j])
        f[wj:] = np.maximum(f[wj:], f[: C_ + 1 - wj] + v[j])
    return f


def diff_table(v, w, cmax):
    """diff[i, c] = (bester Wert bei Kapazität c mit Paket i) - (ohne Paket i): Vorzeichen = Status, Betrag = Wertschwelle. Passt Paket i bei c nicht, steht NIE."""
    n = len(v)
    cs = np.arange(cmax + 1)
    out = np.empty((n, cmax + 1), dtype=np.int64)
    for i in range(n):
        f = dp_best(v, w, cmax, skip=i)
        wi = int(w[i])
        a = np.full(cmax + 1, NIE, dtype=np.int64)
        if wi <= cmax:
            a[wi:] = v[i] + f[cs[wi:] - wi]
        out[i] = a - f
    return out


def thresholds_fast(v, w, C_):
    """B_i = f_{-i}(C), A0_i = f_{-i}(C - w_i) (NIE, wenn w_i > C) für alle i über Vorwärts- und Rückwärtstabellen (Zeit n * C statt n^2 * C)."""
    n = len(v)
    pre = np.zeros((n + 1, C_ + 1), dtype=np.asarray(v).dtype)
    suf = np.zeros((n + 1, C_ + 1), dtype=np.asarray(v).dtype)
    for j in range(n):
        pre[j + 1] = pre[j]
        if w[j] <= C_:
            wj = int(w[j])
            pre[j + 1][wj:] = np.maximum(pre[j][wj:], pre[j][: C_ + 1 - wj] + v[j])
    for j in range(n - 1, -1, -1):
        suf[j] = suf[j + 1]
        if w[j] <= C_:
            wj = int(w[j])
            suf[j][wj:] = np.maximum(suf[j + 1][wj:], suf[j + 1][: C_ + 1 - wj] + v[j])
    B = np.zeros(n, dtype=np.asarray(v).dtype)
    A0 = np.zeros(n, dtype=np.asarray(v).dtype)
    for i in range(n):
        p, s = pre[i], suf[i + 1]
        B[i] = (p + s[::-1]).max()
        if w[i] <= C_:
            c = C_ - int(w[i])
            A0[i] = (p[: c + 1] + s[c::-1]).max()
        else:
            A0[i] = NIE
    return B, A0


def status_schnell(v, w, C_):
    """(Status, d, fits): Status +1/-1/0 je Paket, d = A - B (Vorzeichen = Status). Ein Paket mit w > C passt nie: sein A ist NIE, der Status -1 und `fits` False
    (die Anzeige sagt dann „passt nie“, nicht „sicher draußen“; gezählt werden nur Pakete mit `fits`)."""
    v, w = np.asarray(v), np.asarray(w)
    fits = w <= C_
    B, A0 = thresholds_fast(v, w, C_)
    A = np.where(fits, v + A0, NIE)
    d = A - B
    return np.sign(d), d, fits


def dp_set(v, w, C_):
    """Eine optimale Menge und ihr Wert (bei Gleichstand wird das Paket nicht genommen); v darf Gleitkommazahlen enthalten."""
    n = len(v)
    T = np.zeros((n + 1, C_ + 1), dtype=np.float64)
    for j in range(n):
        T[j + 1] = T[j]
        if w[j] <= C_:
            wj = int(w[j])
            T[j + 1][wj:] = np.maximum(T[j][wj:], T[j][: C_ + 1 - wj] + v[j])
    x = np.zeros(n, dtype=np.int64)
    c = C_
    for j in range(n - 1, -1, -1):
        if T[j + 1][c] != T[j][c]:
            x[j] = 1
            c -= int(w[j])
    return x, float(T[n][C_])


_SUBS = {}


def subsets(n):
    """Alle 2^n Mengen als 0/1-Matrix (Zeile k = Bitmaske k)."""
    if n not in _SUBS:
        idx = np.arange(2 ** n)
        _SUBS[n] = ((idx[:, None] >> np.arange(n)) & 1).astype(np.int64)
    return _SUBS[n]


def enum_status(v, w, C_):
    """Aufzählung aller 2^n Mengen: (Optimalwert, Status je Paket, Zahl optimaler Mengen, Matrix der optimalen Mengen)."""
    n = len(v)
    S = subsets(n)
    W, V = S @ np.asarray(w), S @ np.asarray(v)
    feas = W <= C_
    vm = V[feas].max()
    opt = feas & (V == vm)
    sub = S[opt]
    st = np.where(sub.min(0) == 1, 1, np.where(sub.max(0) == 0, -1, 0))
    return int(vm), st, int(opt.sum()), sub


def kapazitaetsstreifen(v, w, cmax):
    """Status je Paket über alle Kapazitäten 0..cmax: +1 drin, -1 draußen, 0 Gleichstand, 2 passt nicht; dazu der Optimalwert je Kapazität."""
    tab = diff_table(v, w, cmax)
    streifen = np.sign(tab)
    for i in range(len(v)):
        streifen[i, : int(w[i])] = 2
    return streifen, dp_best(v, w, cmax)


def nahbereich(st_row, c_):
    """Kleinste relative Kapazitätsänderung |C' - C| / C, bei der sich der Status (Gleichstand zählt als Änderung) gegenüber C ändert; unendlich, wenn nie."""
    row = np.asarray(st_row)
    dif = np.nonzero(row != row[c_])[0]
    return float(np.abs(dif - c_).min() / c_) if len(dif) and c_ > 0 else float("inf")


def nichtmonoton(st_row):
    """True, wenn das Paket bei einer Kapazität sicher drin und bei einer größeren sicher draußen ist (Vorzeichen der Tabellenzeile, NIE zählt als draußen nur über passt-nicht ausgeschlossen)."""
    row = np.asarray(st_row)
    fi = np.nonzero(row == 1)[0]
    return bool(len(fi) and (row[fi[0] + 1:] == -1).any())
