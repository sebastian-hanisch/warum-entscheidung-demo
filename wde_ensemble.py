"""Die Ensemble-Messung (erzeugt das eingefrorene Raster data/wde_vorgerechnet.json; `tools/wde_make_vorgerechnet.py` ruft sie auf).

Zähleinheiten: Instanzen (je Ziehung und Zelle 3 Kapazitätsanteile x 100 = 300; bei n = 40 je 50 = 150), Pakete (Instanz x Paket mit eindeutigem Status), Neulösungen
(Instanz x Störung), Störungen (Instanz x Paket x Schätzung), Reihenfolgen. Jede Zelle (n, Skala, Familie) wird in BLOECKE = 5 getrennten Ziehungen gemessen (eigene Zufalls-Seeds);
das Raster speichert die Zähler je Ziehung, die App zeigt Mittel und Spanne über die Ziehungen. Alle Rechenwege sind exakt (Ganzzahlen); der Zufall ist SplitMix64.

Zufalls-Seeds: Gleichstand, Regeln, Kapazität, Störung: 80 000 000 + 1 000 000 * Ziehung + 100 000 * Zeilenindex + 1 000 * Kapazitätsindex + k (Zeilenindex aus n, Skala, Familie);
Schätzwerte: 90 000 000 + 1 000 000 * Ziehung + 100 000 * (4 [n = 40] + 2 [Skala 10] + 1 [stark korreliert]) + 10 000 * Kapazitätsindex + k, Schätzungen aus seed + 9 000 000;
Reihenfolge: 50 000 000 + 100 000 * Ziehung + 1 000 * rmax + k, Störungen aus 60 000 000 + 100 000 * Ziehung + 1 000 * rmax + k."""

import numpy as np

import wde_constants as C
import wde_estimates as E
import wde_knapsack as K
import wde_rules as RU
import wde_schedule as S
from wde_rng import SplitMix64


def zeilenindex(n, skala, fam):
    return 1 + C.N_STUFEN.index(n) * 4 + C.SKALEN.index(skala) * 2 + C.FAMILIEN.index(fam)


def _median(x):
    x = np.asarray(x, dtype=float)
    return float(np.median(x)) if len(x) else None


def messe_zelle(block, n, skala, fam, k_je_kap=C.K_JE_KAP):
    """Zähler einer Ziehung für Experimente 2 (Gleichstand), 3 (Kapazität, Kippabstand), 4 (Regeln) und die Störung aller Werte."""
    z = zeilenindex(n, skala, fam)
    R = dict(ninst=0, tie_i=0, tie_p=0, pak=0, sicher=0, drin=0, kipp5=0, kipp10=0, kipp_rel=[], nm=0, nah5=0, nah10=0, nah25=0,
             g_it=0, s_it=0, g_opt=0, s_opt=0, g_gap=0.0, uniq=0, g_set=0, s_set=0, noise={0.01: [0, 0], 0.05: [0, 0], 0.10: [0, 0]})
    for ki, kap in enumerate(C.KAPAZITAETEN):
        for k in range(k_je_kap if n < 40 else k_je_kap // 2):
            seed = 80_000_000 + 1_000_000 * block + 100_000 * z + 1000 * ki + k
            v, w, C_ = K.instanz(seed, n, fam, skala, kap)
            st, dC, fits = K.status_schnell(v, w, C_)
            vopt = int(K.dp_best(v, w, C_)[C_])
            R["ninst"] += 1
            ti = int((st == 0).any())
            R["tie_i"] += ti
            R["tie_p"] += int((st == 0).sum())
            R["pak"] += n
            sicher = fits & (st != 0)
            ns = int(sicher.sum())
            R["sicher"] += ns
            R["drin"] += int((st[sicher] == 1).sum())
            rel = np.abs(dC[sicher]) / v[sicher]
            R["kipp_rel"].extend(rel.tolist())
            R["kipp5"] += int((rel <= 0.05).sum())
            R["kipp10"] += int((rel <= 0.10).sum())
            # Kapazität: Nichtmonotonie über 0..Summe w und Nahbereich um C (Zeilen der Statustabelle als Vorzeichen; passt nicht zählt als draußen)
            tab = K.diff_table(v, w, int(w.sum()))
            for i in range(n):
                row = np.sign(tab[i])
                R["nm"] += int(K.nichtmonoton(row))
                if sicher[i]:
                    dd = K.nahbereich(row, C_)
                    R["nah5"] += int(dd <= 0.05)
                    R["nah10"] += int(dd <= 0.10)
                    R["nah25"] += int(dd <= 0.25)
            # Regeln
            xg, xs = RU.greedy_regel(v, w, C_), RU.schwellenregel(v, w, C_)
            R["g_it"] += int(((xg == 1) == (st == 1))[sicher].sum())
            R["s_it"] += int(((xs == 1) == (st == 1))[sicher].sum())
            vg, vs = int(v @ xg), int(v @ xs)
            R["g_opt"] += int(vg == vopt)
            R["s_opt"] += int(vs == vopt)
            R["g_gap"] += (vopt - vg) / vopt
            if ti == 0:
                xo = (st == 1).astype(np.int64)
                R["uniq"] += 1
                R["g_set"] += int((xo == xg).all())
                R["s_set"] += int((xo == xs).all())
                if n >= 12:      # Störung aller Werte (nur Instanzen mit eindeutigem Optimum): 10 Störungen je Instanz
                    rng = SplitMix64(seed + 7_000_000)
                    for dl in (0.01, 0.05, 0.10):
                        ch = 0
                        for _ in range(10):
                            v2 = v * (1 + dl * np.array([2 * rng.random() - 1 for _ in range(n)]))
                            x2, _ = K.dp_set(v2, w, C_)
                            ch += int((x2 != xo).any())
                        R["noise"][dl][0] += ch
                        R["noise"][dl][1] += 10
    R["kipp_median"] = _median(R.pop("kipp_rel"))
    R["noise"] = {f"{d:g}": c for d, c in R["noise"].items()}
    return R


def messe_schaetz(block, n, skala, fam, delta, k_je_kap=C.K_JE_KAP):
    """Zähler einer Ziehung für Experiment 5: Anteil gültiger Anhebungen je Sicherheitsstufe (Zähleinheit Störungen), Mediane des Aufwands."""
    zz = (n == 40) * 4 + (skala == 10) * 2 + (fam != "unkorreliert")
    cnt = {k: 0 for k in C.SICHERHEITEN}
    auf = {k: [] for k in ("rel0", "rel100", "abs8")}
    den = 0
    for ki, kap in enumerate(C.KAPAZITAETEN):
        for k in range(k_je_kap):
            seed = 90_000_000 + 1_000_000 * block + 100_000 * zz + 10_000 * ki + k
            v, w, C_ = K.instanz(seed, n, fam, skala, kap)
            rng = SplitMix64(seed + 9_000_000)
            for _ in range(C.N_STOERUNG):
                v_hat = E.schaetzung(v, delta, rng)
                a = E.auswertung(v, w, C_, v_hat, delta)
                den += len(a["index"])
                for key in cnt:
                    cnt[key] += int(a[key].sum())
                for key in auf:
                    auf[key].extend(a[key + "_aufwand"].tolist())
    return dict(den=den, cnt=cnt, aufwand_median={k: _median(x) for k, x in auf.items()}, fehlschlaege_abs8=den - cnt["abs8"])


def messe_reihenfolge(block, rmax, ni=C.K_SEQ):
    """Zähler einer Ziehung für den Tab Reihenfolge (n = 7): mehrere optimale Reihenfolgen, Regeln gegen die Aufzählung, Störung der Dauern, Gegenfaktum (nur rmax = 0, geschlossen)."""
    n = S.N
    R = dict(ni=ni, tie_i=0, pair_tie=0, pair_den=0, S_opt=0, D_opt=0, S_gap=0.0, D_gap=0.0, S_pn=0, D_pn=0, pd=0, noise={1: [0, 0], 2: [0, 0]}, dw=[], dp=[], nonk_p=0)
    for k in range(ni):
        p, w, r = S.instanz(50_000_000 + 100_000 * block + 1000 * rmax + k, rmax)
        Cm = S.fertigstellung(p, r)
        om, mask = S.optimum(Cm, w)
        sicher, ties = [], 0
        for a in range(n):
            for b in range(n):
                if a == b:
                    continue
                s = S.paarstatus(mask, a, b)
                ties += int(s == 0)
                if s == 1:
                    sicher.append((a, b))
        R["tie_i"] += int(mask.sum() > 1)
        R["pair_tie"] += ties
        R["pair_den"] += n * (n - 1)
        seq = {"S": S.smith_statisch(p, w), "D": S.dispatch(p, w, r)}
        for key, sq in seq.items():
            val = S.zielwert(sq, p, w, r)
            R[key + "_opt"] += int(val == om)
            R[key + "_gap"] += (val - om) / om
            R[key + "_pn"] += S.paare_richtig(sq, sicher)
        R["pd"] += len(sicher)
        if rmax == 0:
            for (a, b) in sicher:
                d_w, d_p = S.gegenfaktum_geschlossen(p, w, a, b)
                R["dw"].append(d_w / w[a])
                if d_p <= 29:
                    R["dp"].append(d_p / p[a])
                else:
                    R["nonk_p"] += 1
        if mask.sum() == 1:
            base = S.PERMS[mask][0]
            rng = SplitMix64(60_000_000 + 100_000 * block + 1000 * rmax + k)
            for amp in (1, 2):
                ch = 0
                for _ in range(10):
                    p2 = np.maximum(1, p + np.array([K_uni(rng, -amp, amp) for _ in range(n)], dtype=np.int64))
                    _, m2 = S.optimum(S.fertigstellung(p2, r), w)
                    ch += int(not (m2.sum() == 1 and (S.PERMS[m2][0] == base).all()))
                R["noise"][amp][0] += ch
                R["noise"][amp][1] += 10
    dw, dp = np.array(R.pop("dw")), np.array(R.pop("dp"))
    R["dw_median"] = _median(dw)
    R["dw_le_third"] = float((dw <= 1 / 3 + 1e-9).mean()) if len(dw) else None
    R["dp_median"] = _median(dp)
    R["noise"] = {str(a): c for a, c in R["noise"].items()}
    R["n_paare"] = int(len(dw))
    return R


def K_uni(rng, lo, hi):
    return K.uni(rng, lo, hi)
