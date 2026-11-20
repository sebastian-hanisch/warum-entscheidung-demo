"""Schätzwert-Falle: eine Begründung, auf geschätzten Werten berechnet, an den wahren Werten geprüft.

Die „Schätzung“ ist ein Rauschmodell, kein Lernmodell: v_hat = rint(v * (1 + delta * U(-1, 1))) je Paket unabhängig; Gewichte und Kapazität sind bekannt. Für ein Paket, das in der
Schätzung und in der Wahrheit sicher draußen ist, hat die geschätzte Wertschwelle m_hat = B_hat - A_hat (Anhebung des geschätzten Werts, ab der das Paket in der Schätzung
nicht mehr sicher draußen steht) und die wahre m_wahr. Aktion: den wahren Wert um m_aktion anheben, mit
  relativ:  m_aktion = m_hat * (1 + Aufschlag),
  absolut:  m_aktion = m_hat + kappa * delta * (mittlerer geschätzter Wert)   (kappa Rauscheinheiten).
Gültig, wenn die Anhebung knapp über m_aktion liegt und m_aktion >= m_wahr ist (genau um m_wahr angehoben wäre es nur ein Gleichstand). Aufwand = m_aktion / (m_wahr + 1).
Das setzt ein bekanntes Rauschniveau delta voraus (hier erfunden bekannt); Rauschen, das mit dem Wert wächst oder verzerrt ist, ist nicht gemessen."""

import numpy as np

from wde_knapsack import thresholds_fast


def schaetzung(v, delta, rng):
    """Geschätzte Werte v_hat (ganzzahlig) aus einem SplitMix64-Strom."""
    u = np.array([2 * rng.random() - 1 for _ in range(len(v))])
    return np.rint(np.asarray(v) * (1 + delta * u)).astype(np.int64)


def schwellen(v, w, C_):
    """m = B - A0 - v je Paket (> 0: sicher draußen; passt nie: sehr groß, wird über `fits` ausgeschlossen)."""
    B, A0 = thresholds_fast(v, w, C_)
    return B - A0 - v


def aktionen(m_hat, v_hat, delta):
    """Anhebungen der sechs Sicherheitsstufen (relativ 0, 25, 50, 100 %; absolut 2, 4, 8 Rauscheinheiten) je Paket."""
    einheit = delta * float(np.mean(v_hat))
    return {"rel0": m_hat * 1.0, "rel25": m_hat * 1.25, "rel50": m_hat * 1.5, "rel100": m_hat * 2.0,
            "abs2": m_hat + 2 * einheit, "abs4": m_hat + 4 * einheit, "abs8": m_hat + 8 * einheit}


def gueltig(m_aktion, m_wahr):
    return m_aktion >= m_wahr


def auswertung(v, w, C_, v_hat, delta):
    """Je Paket (nur dort, wo es in Schätzung und Wahrheit sicher draußen und passend ist): m_hat, m_wahr und je Sicherheitsstufe Gültigkeit und Aufwand."""
    fits = w <= C_
    m_wahr = schwellen(v, w, C_)
    m_hat = schwellen(v_hat, w, C_)
    m = fits & (m_hat > 0) & (m_wahr > 0)
    akt = aktionen(m_hat[m].astype(float), v_hat, delta)
    out = {"index": np.nonzero(m)[0], "m_hat": m_hat[m], "m_wahr": m_wahr[m]}
    for k, a in akt.items():
        out[k] = a >= m_wahr[m]
        out[k + "_aufwand"] = a / (m_wahr[m] + 1.0)
    return out
