"""Konstanten: Vehikel (erfundenes Rucksack-Vehikel „Fuhre packen“), Messraster, Regler und Presets.

Es gibt keine reale Datenquelle und kein Modell: alle Pakete, Gewichte und Werte sind erfunden und ganzzahlig."""

# --- Vehikel -----------------------------------------------------------------------------------------------------------------
FAMILIEN = ("unkorreliert", "stark korreliert")
FAMILIE_LABEL = {"unkorreliert": "unkorreliert (Wert unabhängig vom Gewicht)", "stark korreliert": "stark korreliert (Wert = Gewicht + 10 s)"}
SKALEN = (1, 10)
SKALA_LABEL = {1: "1 (grob: Gewichte 5 bis 30, Werte 10 bis 60)", 10: "10 (fein: zehnmal feinere Zahlen)"}
KAPAZITAETEN = (0.3, 0.5, 0.7)
N_STUFEN = (8, 12, 20, 40)
N_LIVE = N_STUFEN                          # wählbar; die Aufzählung läuft live nur bis n = 12 (4 096 Mengen)
N_ENUM_MAX = 12
N_SEQ = 7                                   # Reihenfolge: n = 7 Aufträge, 5 040 Reihenfolgen
RMAX_STUFEN = (0, 10, 25)
SEQ_REGELN = ("statisch", "dispatch")
SEQ_REGEL_LABEL = {"statisch": "statische Smith-Reihenfolge (p/w aufsteigend)", "dispatch": "Dispatch (freigegebener Auftrag mit kleinstem p/w)"}
REGELN = ("greedy", "schwelle")
REGEL_LABEL = {"greedy": "Greedy nach Verhältnis (füllt Lücken)", "schwelle": "Regel S (bis zum kritischen Paket)"}

# --- Messraster (Ziehungen: getrennte Zufalls-Seeds; Anteile sind Mittel und Spanne über die Ziehungen) ------------------------------
BLOECKE = 5                                 # Ziehungen je Zelle
K_JE_KAP = 100                              # Instanzen je Kapazitätsanteil und Ziehung: 300 (n = 40: 150)
K_SEQ = 200                                 # Instanzen je Freigabestufe und Ziehung (Reihenfolge)
RAUSCHEN = (0.05, 0.10)
N_SCHAETZ = (12, 40)
N_STOERUNG = 5                              # Schätzungen je Instanz (Experiment 5)
SICHERHEITEN = ("rel0", "rel25", "rel50", "rel100", "abs2", "abs4", "abs8")
SICHERHEIT_LABEL = {"rel0": "relativer Aufschlag 0 %", "rel25": "relativer Aufschlag 25 %", "rel50": "relativer Aufschlag 50 %", "rel100": "relativer Aufschlag 100 %",
                    "abs2": "absoluter Abstand 2 Rauscheinheiten", "abs4": "absoluter Abstand 4 Rauscheinheiten", "abs8": "absoluter Abstand 8 Rauscheinheiten"}
GRID_DATEI = "data/wde_vorgerechnet.json"

# --- Seitenleiste ------------------------------------------------------------------------------------------------------------------
DEFAULT = dict(n=12, familie="unkorreliert", skala=1, kap=0.5, seed=7, hb_wert=7, hb_kap=8, regel="greedy", rauschen=0.10, sicherheit="rel0", rmax=10, seq_regel="dispatch")
SEED_MAX = 999999
HB_WEIGHTS = (4, 3, 3, 5)
HB_VALUES = (6, 7, 9, 9)
HB_PAKET = 1                                # Handbeispiel: Wert von Paket 2 (Index 1) ist regelbar
HB_WERT_MIN, HB_WERT_MAX = 4, 13
HB_KAP_MIN, HB_KAP_MAX = 3, 16
COLOR_DRIN = "#2F6B65"
COLOR_RAUS = "#b24c4c"
COLOR_GLEICH = "#d6a84b"
COLOR_NIE = "rgba(150,150,150,0.55)"

_S = dict(DEFAULT)
PRESETS = {
    "Fuhre mit 12 Paketen (Voreinstellung)": dict(_S),
    "Handbeispiel: Paket 2 auf Wert 9 (Gleichstand)": {**_S, "hb_wert": 9},
    "Stark korreliert (Gleichstand häufig)": {**_S, "familie": "stark korreliert"},
    "Kapazität +1 wirft Paket 2 raus": {**_S, "hb_kap": 7},
    "Greedy verfehlt (Regel S)": {**_S, "regel": "schwelle", "n": 8},
    "Schätzwerte: stark korreliert, 40 Pakete": {**_S, "familie": "stark korreliert", "n": 40, "sicherheit": "rel100", "rauschen": 0.10},
}
PRESET_HELP = {
    "Fuhre mit 12 Paketen (Voreinstellung)": "12 unkorrelierte Pakete, Kapazität 50 % der Gewichtssumme, Zahlenskala 1: die Voreinstellung aller fünf Experimente.",
    "Handbeispiel: Paket 2 auf Wert 9 (Gleichstand)": "Im Handbeispiel (4 Pakete, Kapazität 8) steigt der Wert von Paket 2 von 7 auf 9: Gleichstand mit drei optimalen Mengen (Experiment 1).",
    "Stark korreliert (Gleichstand häufig)": "Wert = Gewicht + 10: viele Pakete haben dasselbe Verhältnis, die optimale Menge ist oft nicht eindeutig (Experiment 2).",
    "Kapazität +1 wirft Paket 2 raus": "Handbeispiel: bei Kapazität 7 liegt Paket 2 in der optimalen Menge, bei 8 nicht, obwohl der Optimalwert steigt (Experiment 3).",
    "Greedy verfehlt (Regel S)": "8 Pakete, Regel S (bis zum kritischen Paket): die einfache Regel trifft die meisten Einzelentscheidungen, nicht die ganze Menge (Experiment 4).",
    "Schätzwerte: stark korreliert, 40 Pakete": "40 stark korrelierte Pakete, Rauschen 10 %, relativer Aufschlag 100 %: auch Verdopplung reicht nicht immer (Experiment 5).",
}
