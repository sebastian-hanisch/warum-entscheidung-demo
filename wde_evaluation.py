"""Kennzahlen und Verdikte aus dem Raster (data/wde_vorgerechnet.json) und aus der Einzelinstanz. Verdikte sind reine Funktionen (Art, Text).

Zähleinheit überall: Anteile über Instanzen oder Pakete einer Zelle (n, Zahlenskala, Familie), je Ziehung gerechnet und als Mittel mit Spanne (kleinste bis größte Ziehung) über die fünf
getrennten Ziehungen gezeigt; die Instanzmenge (300 Instanzen je Ziehung, bei n = 40: 150) steht daneben. Nie die Punktzahl einer festen Ziehung."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

import wde_constants as C

ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def lade_raster():
    return json.loads((ROOT / C.GRID_DATEI).read_text(encoding="utf-8"))


def de(x, nd=1):
    return f"{x:.{nd}f}".replace(".", ",")


def pct(x, nd=1):
    return de(100 * x, nd) + " %"


def band(zeilen, zaehler, nenner):
    """Mittel [kleinste, größte] des Anteils zaehler / nenner über die Ziehungen (zeilen = Liste von Wörterbüchern je Ziehung; `zaehler`, `nenner` Schlüssel oder Funktionen)."""
    f = (lambda z, k: z[k] if isinstance(k, str) else k(z))
    werte = np.array([f(z, zaehler) / f(z, nenner) for z in zeilen if f(z, nenner)], dtype=float)
    return float(werte.mean()), float(werte.min()), float(werte.max())


def band_text(b, nd=1):
    """„34,0 % [30,1 bis 38,2]“ (Mittel mit Spanne über die Ziehungen)."""
    return f"{de(100 * b[0], nd)} % [{de(100 * b[1], nd)} bis {de(100 * b[2], nd)}]"


def zelle(raster, n, skala, fam):
    return raster["zellen"][f"{n}|{skala}|{fam}"]


def schaetz_zelle(raster, n, skala, fam, delta):
    return raster["schaetz"][f"{n}|{skala}|{fam}|{delta:g}"]


def reihenfolge(raster, rmax):
    return raster["reihenfolge"][str(rmax)]


def instanzen_je_ziehung(zeilen):
    return int(zeilen[0]["ninst"])


# ---------------------------------------------------------------- Zusammenfassungen je Zelle

def gleichstand(raster, n, skala, fam):
    z = zelle(raster, n, skala, fam)
    return dict(instanzen=band(z, "tie_i", "ninst"), pakete=band(z, "tie_p", "pak"), ninst=instanzen_je_ziehung(z))


def kapazitaet(raster, n, skala, fam):
    z = zelle(raster, n, skala, fam)
    return dict(nichtmonoton=band(z, "nm", "pak"), nah5=band(z, "nah5", "sicher"), nah10=band(z, "nah10", "sicher"), nah25=band(z, "nah25", "sicher"), ninst=instanzen_je_ziehung(z))


def kippabstand(raster, n, skala, fam):
    z = zelle(raster, n, skala, fam)
    med = np.array([r["kipp_median"] for r in z], dtype=float)
    return dict(median=(float(med.mean()), float(med.min()), float(med.max())), le5=band(z, "kipp5", "sicher"), le10=band(z, "kipp10", "sicher"))


def regeln(raster, n, skala, fam):
    z = zelle(raster, n, skala, fam)
    out = dict(g_pakete=band(z, "g_it", "sicher"), s_pakete=band(z, "s_it", "sicher"), g_opt=band(z, "g_opt", "ninst"), s_opt=band(z, "s_opt", "ninst"),
               g_luecke=band(z, "g_gap", "ninst"), basis=band(z, lambda r: max(r["drin"], r["sicher"] - r["drin"]), "sicher"), ninst=instanzen_je_ziehung(z),
               eindeutig=[r["uniq"] for r in z])
    zz = [r for r in z if r["uniq"] >= 5]
    out["g_menge"] = band(zz, "g_set", "uniq") if len(zz) == len(z) else None
    out["s_menge"] = band(zz, "s_set", "uniq") if len(zz) == len(z) else None
    return out


def stoerung(raster, n, skala, fam, delta):
    z = zelle(raster, n, skala, fam)
    zz = [r for r in z if r["noise"][f"{delta:g}"][1] > 0]
    return band([{"a": r["noise"][f"{delta:g}"][0], "b": r["noise"][f"{delta:g}"][1]} for r in zz], "a", "b") if zz and len(zz) == len(z) else None


def schaetzung(raster, n, skala, fam, delta, sicherheit):
    z = schaetz_zelle(raster, n, skala, fam, delta)
    gueltig = band([{"a": r["cnt"][sicherheit], "b": r["den"]} for r in z], "a", "b")
    fehl = int(sum(r["den"] - r["cnt"][sicherheit] for r in z))
    return dict(gueltig=gueltig, stoerungen=int(sum(r["den"] for r in z)), fehlschlaege=fehl, je_ziehung=int(np.mean([r["den"] for r in z])))


def aufwand(raster, n, skala, fam, delta, sicherheit):
    z = schaetz_zelle(raster, n, skala, fam, delta)
    key = sicherheit if sicherheit in ("rel0", "rel100", "abs8") else None
    if key is None:
        return None
    med = [r["aufwand_median"][key] for r in z if r["aufwand_median"][key] is not None]
    return (float(np.mean(med)), float(np.min(med)), float(np.max(med))) if med else None


def reihe(raster, rmax):
    z = reihenfolge(raster, rmax)
    ni = int(z[0]["ni"])
    out = dict(ni=ni, mehrere=band(z, "tie_i", "ni"), paare_gleich=band(z, "pair_tie", "pair_den"), s_opt=band(z, "S_opt", "ni"), d_opt=band(z, "D_opt", "ni"),
               s_luecke=band(z, "S_gap", "ni"), d_luecke=band(z, "D_gap", "ni"), s_paare=band(z, "S_pn", "pd"), d_paare=band(z, "D_pn", "pd"))
    out["stoerung1"] = band([{"a": r["noise"]["1"][0], "b": r["noise"]["1"][1]} for r in z], "a", "b")
    out["stoerung2"] = band([{"a": r["noise"]["2"][0], "b": r["noise"]["2"][1]} for r in z], "a", "b")
    if rmax == 0:
        out["dw_median"] = float(np.mean([r["dw_median"] for r in z]))
        out["dp_median"] = float(np.mean([r["dp_median"] for r in z]))
        out["dw_le_third"] = float(np.mean([r["dw_le_third"] for r in z]))
        out["nicht_kippbar_p"] = int(sum(r["nonk_p"] for r in z))
        out["paare"] = int(sum(r["n_paare"] for r in z))
    return out


# ---------------------------------------------------------------- Verdikte (reine Funktionen)

def verdict_paket(i, status, d, v, w, kap, fits):
    """Experiment 1: Begründung für Paket i (Nummer ab 1). status +1 drin, -1 draußen, 0 Gleichstand; d = A - B (Wertschwelle, Vorzeichen wie der Status)."""
    if not fits:
        return "warning", f"Paket {i} wiegt {w}, mehr als die Kapazität {kap}: es passt nie. Das ist keine Entscheidung der Optimierung, sondern eine Bedingung."
    if status == 0:
        return "info", (f"Paket {i} (Wert {v}) ist im **Gleichstand**: es liegt in manchen optimalen Mengen und in anderen nicht. Die Frage „Warum drin?“ hat keine eindeutige Antwort; "
                        "„in jeder“, „in keiner“ und „in mancher“ sind drei verschiedene Fragen.")
    betrag = abs(d)
    rel = betrag / v if v else float("inf")
    if status < 0:
        return "success", (f"Paket {i} (Wert {v}) liegt in **keiner** optimalen Menge. Sein Wert müsste um {betrag} auf {v + betrag} steigen ({pct(rel)} seines Werts), dann ist die Entscheidung "
                           f"nicht mehr eindeutig (Gleichstand); das Paket zu erzwingen kostet genau {betrag} Wertpunkte (Wertschwelle B − A = {betrag}).")
    return "success", (f"Paket {i} (Wert {v}) liegt in **jeder** optimalen Menge. Sein Wert müsste um {betrag} auf {v - betrag} fallen ({pct(rel)} seines Werts), dann ist die Entscheidung "
                       f"nicht mehr eindeutig (Gleichstand); das Paket auszuschließen kostet genau {betrag} Wertpunkte (Wertschwelle A − B = {betrag}).")


def verdict_gleichstand(g, n, skala, fam):
    """Experiment 2: Anteil der Instanzen mit mehreren optimalen Mengen (Mittel mit Spanne über die Ziehungen)."""
    m = g["instanzen"][0]
    zelle_text = f"n = {n}, Zahlenskala {skala}, {fam}"
    if m >= 0.5:
        return "warning", (f"Bei {zelle_text} haben im Mittel über fünf Ziehungen **{band_text(g['instanzen'])}** der Instanzen mehrere optimale Mengen (je Ziehung {g['ninst']} Instanzen): "
                           "„Warum A statt B?“ hat hier meist mehrere gleich gute Antworten.")
    if m >= 0.2:
        return "info", (f"Bei {zelle_text} haben **{band_text(g['instanzen'])}** der Instanzen mehrere optimale Mengen (fünf Ziehungen zu je {g['ninst']} Instanzen): jede fünfte oder öfter.")
    return "success", (f"Bei {zelle_text} haben nur **{band_text(g['instanzen'])}** der Instanzen mehrere optimale Mengen (fünf Ziehungen zu je {g['ninst']} Instanzen): selten, aber nicht null.")


def verdict_kapazitaet(k, n, skala, fam):
    """Experiment 3: Anteil Pakete, die bei mehr Kapazität irgendwo im Bereich 0 bis Σw hinausfliegen, und im Nahbereich (höchstens 5 % Kapazitätsänderung)."""
    return "info", (f"Bei n = {n}, Zahlenskala {skala}, {fam}: **{band_text(k['nichtmonoton'])}** der Pakete sind bei einer Kapazität sicher drin und bei einer größeren sicher draußen "
                    f"(irgendwo im Bereich 0 bis Σw, der mit n wächst); bei höchstens 5 % Kapazitätsänderung ändert sich der Status von **{band_text(k['nah5'])}** der Pakete mit eindeutigem Status. "
                    "Mehr Kapazität hebt den Optimalwert nie, kann aber ein Paket hinauswerfen.")


def verdict_regel(r, name, n, skala, fam):
    """Experiment 4: die einfache Regel trifft die meisten Einzelentscheidungen, nicht die ganze Lösung."""
    pk = r["g_pakete"] if name == "greedy" else r["s_pakete"]
    menge = r["g_menge"] if name == "greedy" else r["s_menge"]
    opt = r["g_opt"] if name == "greedy" else r["s_opt"]
    teil = f"je Paket **{band_text(pk)}** richtig"
    if menge is None:
        teil2 = "die ganze Menge ist bei zu wenigen Instanzen mit eindeutigem Optimum nicht messbar"
    else:
        teil2 = f"die ganze Menge stimmt nur in **{band_text(menge)}** der Instanzen mit eindeutigem Optimum"
    kind = "warning" if (menge is not None and menge[0] < 0.5) or opt[0] < 0.5 else "info"
    return kind, (f"{C.REGEL_LABEL[name].split(' (')[0]} bei n = {n}, Zahlenskala {skala}, {fam}: {teil}, {teil2}; den Optimalwert trifft sie in {band_text(opt)} der Instanzen "
                  f"(Basis: ein Paket gleich dem häufigeren Status zu raten ergäbe {band_text(r['basis'])}).")


def verdict_schaetz(s, sicherheit, n, skala, fam, delta):
    """Experiment 5: Anteil gültiger Anhebungen auf Schätzwerten."""
    name = C.SICHERHEIT_LABEL[sicherheit]
    kopf = f"Mit der Sicherheit „{name}“ (Rauschen {pct(delta, 0)}, n = {n}, Zahlenskala {skala}, {fam}) gilt die auf den Schätzwerten berechnete Anhebung an den wahren Werten in **{band_text(s['gueltig'])}** der Störungen"
    if s["fehlschlaege"] == 0:
        return "success", (f"{kopf} (kein einziger Fehlschlag in {s['stoerungen']} Störungen aus fünf Ziehungen; das heißt nicht „sicher“, sondern „nie ein Fehlschlag in dieser Messung“, und setzt ein "
                           "bekanntes Rauschniveau voraus).")
    if s["gueltig"][0] < 0.9:
        return "warning", f"{kopf} ({s['fehlschlaege']} Fehlschläge in {s['stoerungen']} Störungen aus fünf Ziehungen): die Begründung gilt an den wahren Werten nicht verlässlich."
    return "info", f"{kopf} ({s['fehlschlaege']} Fehlschläge in {s['stoerungen']} Störungen aus fünf Ziehungen)."


def verdict_reihenfolge(r, rmax, regel):
    """Tab Reihenfolge: statische Smith-Regel und Dispatch gegen die Aufzählung aller 5 040 Reihenfolgen."""
    if rmax == 0:
        return "success", (f"Ohne Freigaben ist die statische Smith-Reihenfolge exakt: sie trifft den Optimalwert in {band_text(r['s_opt'])} der Instanzen (je Ziehung {r['ni']}), und jeder sichere "
                           f"Paarvergleich stimmt ({band_text(r['s_paare'])}); mehrere optimale Reihenfolgen gibt es trotzdem in {band_text(r['mehrere'])}.")
    opt = r["s_opt"] if regel == "statisch" else r["d_opt"]
    name = C.SEQ_REGEL_LABEL[regel].split(" (")[0]
    kind = "warning" if opt[0] < 0.5 else "info"
    return kind, (f"Mit Freigaben bis {rmax} trifft die Regel „{name}“ den Optimalwert in **{band_text(opt)}** der Instanzen (je Ziehung {r['ni']}); die andere Regel in "
                  f"{band_text(r['d_opt'] if regel == 'statisch' else r['s_opt'])}.")
