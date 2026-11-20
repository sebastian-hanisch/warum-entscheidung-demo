"""Warum ist dieses Paket drin? Eine Packentscheidung begründen - interaktive Konzept-Demo (Stück der Linie „Beitrag und Beleg“)
Sebastian Hanisch - Operations Research und Machine Learning

Die Einzelinstanz wird live gerechnet (Aufzählung bis 12 Pakete, dynamische Programmierung bis 40); die Ensemble-Zahlen (Anteile über 300 Instanzen in fünf Ziehungen) stehen im
vorgerechneten Raster data/wde_vorgerechnet.json (tools/wde_make_vorgerechnet.py). Es gibt kein Modell und keine reale Datenquelle: die Pakete sind erfunden.

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import wde_constants as C
import wde_estimates as E
import wde_evaluation as ev
import wde_knapsack as K
import wde_rules as RU
import wde_schedule as S
import wde_visualization as V
from wde_layout import dunkles_design, kopf_css
from wde_presets import init_session_state_defaults, load_permalink_settings, randomize_seed, sidebar_presets, sync_query_params
from wde_rng import SplitMix64

st.set_page_config(page_title="Warum ist dieses Paket drin? – Sebastian Hanisch", layout="wide")

STATUS_TEXT = {1: "drin (in jeder optimalen Menge)", -1: "draußen (in keiner)", 0: "Gleichstand (in mancher)"}


@st.cache_data(show_spinner=False)
def _raster():
    return ev.lade_raster()


@st.cache_data(show_spinner=False)
def _instanz(seed, n, fam, skala, kap):
    v, w, cap = K.instanz(seed, n, fam, skala, kap)
    st_, d, fits = K.status_schnell(v, w, cap)
    vopt = int(K.dp_best(v, w, cap)[cap])
    out = dict(v=v, w=w, cap=cap, st=st_, d=d, fits=fits, vopt=vopt, enum=None)
    if n <= C.N_ENUM_MAX:
        vm, ste, nopt, sub = K.enum_status(v, w, cap)
        out["enum"] = dict(vm=vm, st=ste, nopt=nopt, sub=sub, gleich=bool(vm == vopt and (ste[fits] == st_[fits]).all()))
    return out


@st.cache_data(show_spinner=False)
def _streifen(seed, n, fam, skala, kap):
    v, w, cap = K.instanz(seed, n, fam, skala, kap)
    return K.kapazitaetsstreifen(v, w, int(w.sum()))


def _zeigen(kind_text):
    getattr(st, kind_text[0])(kind_text[1])


def _menge_text(x):
    return "{" + ", ".join(str(i + 1) for i in range(len(x)) if x[i]) + "}"


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in wde_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("📦 Warum ist dieses Paket drin?")
st.markdown(
    """
Eine **Packentscheidung** ist das Ergebnis einer exakten Optimierung, kein Modell. Trotzdem fragt man: **Warum liegt Paket A in der Fuhre und Paket B nicht, und was müsste anders sein?**
Die Demo begründet die Entscheidung **exakt** (Aufzählung aller Mengen, dynamische Programmierung) und zählt, **wo die Begründung bricht**: bei Gleichstand, bei mehr Kapazität, bei einer einfachen
Regel und bei nur geschätzten Werten. Die Pakete sind **erfunden**; es gibt **keine reale Datenquelle** und kein Modell.

Weiter unten: fünf **🔬 Experimente**, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So begründet man eine Packentscheidung", expanded=True):
    st.markdown(
        """
1. **Lesart festlegen:** Eine Frage wie „Warum ist Paket A drin?“ hat nur mit einer Lesart eine Antwort: **sicher drin** (in jeder optimalen Menge), **sicher draußen** (in keiner) oder **Gleichstand** (in mancher).
2. **Wertschwelle:** A ist der beste Wert mit Paket i, B der beste ohne Paket i (beide bei derselben Kapazität). Das Paket wechselt seinen Status, wenn sein Wert um **|A − B|** steigt oder fällt;
   für ein sicher draußen liegendes Paket ist dieselbe Zahl der **Preis der Erzwingung**: Paket i zwingend zu packen kostet B − A Wertpunkte.
3. **Orakel:** Die Aufzählung aller 2^n Mengen (bis 12 Pakete live), die dynamische Programmierung (bis 40) und ein ganzzahliges Programm im Test geben dieselben Status.
        """
    )
st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Instanz (Fuhre packen)**")
    n = st.select_slider("Pakete n", options=C.N_LIVE, key="n_sel", help="Die Aufzählung aller Mengen läuft live bis 12 Pakete; darüber rechnet nur die dynamische Programmierung.")
    fam = st.selectbox("Familie", C.FAMILIEN, key="familie_sel", format_func=C.FAMILIE_LABEL.get, help="Unkorreliert: Wert unabhängig vom Gewicht. Stark korreliert: Wert = Gewicht + 10 s (viele gleiche Verhältnisse).")
    skala = st.radio("Zahlenskala s", C.SKALEN, key="skala_sel", format_func=C.SKALA_LABEL.get, help="Feinere Zahlen machen Gleichstände seltener.")
    kap = st.radio("Kapazität", C.KAPAZITAETEN, key="kap_sel", format_func=lambda k: f"{int(100 * k)} % der Gewichtssumme", horizontal=True)
    seed = st.number_input("Zufalls-Seed der Instanz", 0, C.SEED_MAX, key="seed_input", step=1)
    st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed)
    st.markdown("**Handbeispiel (4 Pakete)**")
    hb_wert = st.slider("Wert von Paket 2", C.HB_WERT_MIN, C.HB_WERT_MAX, key="hb_wert", help="Im Handbeispiel hat Paket 2 den Wert 7.")
    hb_kap = st.slider("Kapazität des Handbeispiels", C.HB_KAP_MIN, C.HB_KAP_MAX, key="hb_kap", help="Im Handbeispiel ist die Kapazität 8.")
    st.markdown("**Einfache Regel (Experiment 4)**")
    regel = st.radio("Regel", C.REGELN, key="regel_sel", format_func=C.REGEL_LABEL.get)
    st.markdown("**Geschätzte Werte (Experiment 5)**")
    rauschen = st.radio("Rauschen der Schätzung", C.RAUSCHEN, key="rauschen_sel", format_func=lambda d: f"{int(100 * d)} %", horizontal=True)
    sicherheit = st.selectbox("Sicherheit der Anhebung", C.SICHERHEITEN, key="sicherheit_sel", format_func=C.SICHERHEIT_LABEL.get)
    st.markdown("**Reihenfolge (Tab in Experiment 4)**")
    rmax = st.radio("Freigaben bis", C.RMAX_STUFEN, key="rmax_sel", horizontal=True, help="Freigabezeit der sieben Aufträge, gleichverteilt von 0 bis zu diesem Wert.")
    seq_regel = st.radio("Regel", C.SEQ_REGELN, key="seq_regel_sel", format_func=C.SEQ_REGEL_LABEL.get)

sync_query_params({"n_sel": n, "familie_sel": fam, "skala_sel": skala, "kap_sel": kap, "seed_input": int(seed), "hb_wert": hb_wert, "hb_kap": hb_kap, "regel_sel": regel,
                   "rauschen_sel": rauschen, "sicherheit_sel": sicherheit, "rmax_sel": rmax, "seq_regel_sel": seq_regel})

raster = _raster()
I = _instanz(int(seed), n, fam, skala, kap)
v, w, cap, stat, dd, fits = I["v"], I["w"], I["cap"], I["st"], I["d"], I["fits"]
zelltext = f"n = {n}, Zahlenskala {skala}, {fam}"

# --- Ergebnis ---------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
st.caption(f"Instanz: {n} Pakete, {fam}, Zahlenskala {skala}, Kapazität {cap} ({int(100 * kap)} % von {int(w.sum())}), Zufalls-Seed {int(seed)}.")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Optimalwert", f"{I['vopt']}", delta=f"Kapazität {cap}", delta_color="off")
if I["enum"] is not None:
    m2.metric("Optimale Mengen", f"{I['enum']['nopt']}", delta="durch Aufzählung aller 2^n Mengen", delta_color="off")
else:
    m2.metric("Optimale Mengen", "–", delta="Aufzählung nur bis 12 Pakete", delta_color="off")
sicher = fits & (stat != 0)
m3.metric("Pakete im Gleichstand", f"{int((fits & (stat == 0)).sum())} von {n}", delta="in manchen optimalen Mengen", delta_color="off")
m4.metric("Pakete, die nie passen", f"{int((~fits).sum())}", delta="Gewicht über der Kapazität", delta_color="off")

st.markdown("## ❓ Kernfragen")
st.markdown("1. **Warum ist Paket A drin und B nicht?** (Experiment 1) · 2. **Ist die Antwort eindeutig?** (2) · 3. **Was ändert mehr Kapazität?** (3) · 4. **Reicht eine einfache Regel?** (4) · 5. **Gilt die Begründung, wenn die Werte nur geschätzt sind?** (5)")
gl = ev.gleichstand(raster, n, skala, fam)
_zeigen(ev.verdict_gleichstand(gl, n, skala, fam))

st.markdown("---")

# --- Experiment 1 -----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Warum ist Paket A drin, Paket B nicht?")
tab1, tab1b = st.tabs(["Die Instanz", "Handbeispiel (4 Pakete)"])
with tab1:
    st.plotly_chart(V.build_pakete(v, w, stat, fits), width="stretch", key="chart_pakete")
    zeilen = []
    for i in range(n):
        if not fits[i]:
            zeilen.append({"Paket": i + 1, "Gewicht": int(w[i]), "Wert": int(v[i]), "Status": "passt nie", "Kippabstand absolut": "–", "relativ": "–", "Preis der Erzwingung": "–"})
            continue
        status_text = STATUS_TEXT[int(stat[i])]
        betrag = abs(int(dd[i]))
        zeilen.append({"Paket": i + 1, "Gewicht": int(w[i]), "Wert": int(v[i]), "Status": status_text, "Kippabstand absolut": str(betrag), "relativ": ev.pct(betrag / v[i]) if stat[i] != 0 else "0,0 %",
                       "Preis der Erzwingung": (str(betrag) if stat[i] < 0 else ("0" if stat[i] == 0 else "–"))})
    st.dataframe(zeilen, hide_index=True, width="stretch")
    paket = st.selectbox("Paket für die Begründung", list(range(1, n + 1)), key="paket_sel")
    i = paket - 1
    _zeigen(ev.verdict_paket(paket, int(stat[i]), int(dd[i]), int(v[i]), int(w[i]), cap, bool(fits[i])))
    if I["enum"] is not None:
        e = I["enum"]
        mengen = [_menge_text(x) for x in e["sub"][:8]]
        st.caption(f"Optimale Mengen (Wert {e['vm']}): {', '.join(mengen)}{' …' if e['nopt'] > 8 else ''}. Orakelzeile: **Aufzählung = dynamische Programmierung "
                   f"{'✓' if e['gleich'] else '✗'}** (Optimalwert und Status aller {int(fits.sum())} passenden Pakete).")
    else:
        x, vx = K.dp_set(v, w, cap)
        st.caption(f"Bei {n} Paketen läuft nur die dynamische Programmierung (eine optimale Menge: {_menge_text(x)}, Wert {int(vx)}); die Aufzählung aller 2^{n} Mengen steht nur bis 12 Pakete live. "
                   "Dass beide Wege übereinstimmen, prüfen die Tests.")
    st.caption("Kippabstand = Wertschwelle |A − B|: um so viele Wertpunkte müsste sich der Wert des Pakets ändern, bis die Entscheidung nicht mehr eindeutig ist; relativ = Anteil am eigenen Wert. "
               "Der Preis der Erzwingung eines sicher draußen liegenden Pakets ist **dieselbe** Zahl (Identität, in den Tests gerechnet).")
with tab1b:
    hv = list(C.HB_VALUES)
    hv[C.HB_PAKET] = hb_wert
    hv = np.array(hv)
    hw = np.array(C.HB_WEIGHTS)
    hs, hd, hf = K.status_schnell(hv, hw, hb_kap)
    hvm, hse, hnopt, hsub = K.enum_status(hv, hw, hb_kap)
    st.markdown(f"Vier Pakete mit Gewichten {list(C.HB_WEIGHTS)} und Werten {[int(a) for a in hv]} (im Handbeispiel 6, 7, 9, 9), Kapazität {hb_kap}. Die Seitenleiste ändert den Wert von Paket 2 und die Kapazität.")
    st.dataframe([{"Paket": j + 1, "Gewicht": int(hw[j]), "Wert": int(hv[j]), "Status": ("passt nie" if not hf[j] else STATUS_TEXT[int(hse[j])]),
                   "Kippabstand": ("–" if not hf[j] else str(abs(int(hd[j]))))} for j in range(4)], hide_index=True, width="stretch")
    st.plotly_chart(V.build_pakete(hv, hw, hse, hf, "Handbeispiel: Wert, Farbe = Status"), width="stretch", key="chart_hb")
    st.caption(f"Optimalwert {hvm}; optimale Mengen: {', '.join(_menge_text(x) for x in hsub)}. Aufzählung = dynamische Programmierung: "
               f"{'✓' if (hse[hf] == hs[hf]).all() else '✗'}.")
    if hb_kap == 8 and hb_wert == 7:
        st.success("Paket 2 (Wert 7) bleibt draußen, bis sein Wert 9 erreicht: dort **Gleichstand mit drei optimalen Mengen** {3, 4}, {2, 3}, {2, 4} (je 18); ab 10 liegt es in jeder optimalen Menge. "
                   "Das Paket zu erzwingen kostet genau 2 Wertpunkte (Wertschwelle).")
    elif hb_kap == 8 and hb_wert == 9:
        st.info("Wert 9: **Gleichstand mit drei optimalen Mengen** {3, 4}, {2, 3}, {2, 4}, je 18. Paket 2 liegt in manchen, aber nicht in jeder optimalen Menge.")

st.markdown("---")

# --- Experiment 2 -----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Ist die Antwort eindeutig? (Gleichstand)")
st.caption("Anteil der Instanzen mit mehreren optimalen Mengen, Mittel mit Spanne (kleinste bis größte Ziehung) über fünf getrennte Ziehungen; je Ziehung 3 Kapazitätsanteile × 100 Instanzen "
           "(bei n = 40: 150 Instanzen).")
spalten = [f"s = {s}, {f}" for s in C.SKALEN for f in C.FAMILIEN]
matrix = [[ev.gleichstand(raster, nn, s, f)["instanzen"][0] for s in C.SKALEN for f in C.FAMILIEN] for nn in C.N_STUFEN]
c1, c2 = st.columns([1.1, 1])
c1.plotly_chart(V.build_matrix(matrix, [f"n = {nn}" for nn in C.N_STUFEN], spalten, "Instanzen mit mehreren optimalen Mengen (Mittel über fünf Ziehungen)"), width="stretch", key="chart_matrix")
c2.plotly_chart(V.build_band(["Instanzen mit Gleichstand", "Pakete im Gleichstand"], [gl["instanzen"], gl["pakete"]], f"Gewählte Zelle: {zelltext}", farben=[C.COLOR_GLEICH, C.COLOR_RAUS]),
                width="stretch", key="chart_gleich")
_zeigen(ev.verdict_gleichstand(gl, n, skala, fam))
st.dataframe([{"Zelle": f"n = {nn}, s = {s}, {f}", "Instanzen mit Gleichstand (Mittel [Spanne])": ev.band_text(ev.gleichstand(raster, nn, s, f)["instanzen"]),
               "Pakete im Gleichstand": ev.band_text(ev.gleichstand(raster, nn, s, f)["pakete"])} for nn in C.N_STUFEN for s in C.SKALEN for f in C.FAMILIEN], hide_index=True, width="stretch")
if I["enum"] is not None and I["enum"]["nopt"] > 1:
    st.caption(f"Die gewählte Instanz hat {I['enum']['nopt']} optimale Mengen: {', '.join(_menge_text(x) for x in I['enum']['sub'][:8])}{' …' if I['enum']['nopt'] > 8 else ''} (Rashomon im Kleinen).")
elif I["enum"] is not None:
    st.caption("Die gewählte Instanz hat genau eine optimale Menge.")
st.caption("Die Gleichstände wachsen mit n, fallen bei unkorrelierten Zahlen mit feinerer Skala und kaum bei stark korrelierten: einzelne Zahlen früherer Messungen, die zwei Familien und zwei Größen "
           "mischen, sind nicht lesbar. Bei stark korrelierten Zahlen und s = 1 beruht ein erheblicher Teil der Gleichstände auf dem Vertauschen identischer Pakete (gleiches Gewicht und gleicher Wert); "
           "gezählt sind alle Instanzen mit mehreren optimalen Mengen, auch wenn sich die Mengen nur darin unterscheiden.")

st.markdown("---")

# --- Experiment 3 -----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Was ändert mehr Kapazität?")
tab3, tab3b = st.tabs(["Die Instanz", "Handbeispiel (4 Pakete)"])
with tab3:
    strf, optw = _streifen(int(seed), n, fam, skala, kap)
    st.plotly_chart(V.build_streifen(strf, cap), width="stretch", key="chart_streifen")
    st.plotly_chart(V.build_optwerte(optw, cap), width="stretch", key="chart_optw")
    tabelle = np.sign(K.diff_table(v, w, int(w.sum()))) if n <= 12 else None
    out = [j + 1 for j in range(n) if K.nichtmonoton(tabelle[j])] if tabelle is not None else None
    if out is not None:
        st.caption(f"Pakete der gewählten Instanz, die bei einer größeren Kapazität hinausfliegen, nachdem sie bei einer kleineren sicher drin waren: {out if out else 'keines'}.")
    kz = ev.kapazitaet(raster, n, skala, fam)
    _zeigen(ev.verdict_kapazitaet(kz, n, skala, fam))
    st.caption("Streifen: grün drin, rot draußen, gelb Gleichstand, grau passt nicht; gestrichelt die gewählte Kapazität. Der Optimalwert steigt mit der Kapazität nie; der Status ist nicht monoton.")
with tab3b:
    hv0 = np.array(C.HB_VALUES)
    hv0[C.HB_PAKET] = hb_wert
    hstr, hopt = K.kapazitaetsstreifen(hv0, np.array(C.HB_WEIGHTS), 16)
    st.plotly_chart(V.build_streifen(hstr, hb_kap), width="stretch", key="chart_hb_streifen")
    st.markdown(f"Optimalwerte des Handbeispiels für C = 0 bis 16: {', '.join(str(int(a)) for a in hopt)}.")
    st.caption("Paket 2 ist bei C = 6 und 7 drin, bei C = 8 und 9 draußen und bei C = 10 wieder drin: der Optimalwert steigt von 16 auf 18, trotzdem fliegt Paket 2 hinaus, weil {3, 4} "
               "(Gewicht 8) zulässig wird.")

st.markdown("---")

# --- Experiment 4 -----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Reicht eine einfache Regel?")
tab4, tab4b = st.tabs(["Regel gegen exakte Lösung", "Reihenfolge"])
with tab4:
    x = RU.regel(regel, v, w, cap)
    st.markdown(f"**{C.REGEL_LABEL[regel]}** auf der gewählten Instanz: Menge {_menge_text(x)}, Wert {int(v @ x)}, Gewicht {int(w @ x)} (Kapazität {cap}); der Optimalwert ist {I['vopt']}.")
    richtig = int(((x == 1) == (stat == 1))[sicher].sum())
    st.caption(f"Einzelentscheidungen (Pakete mit eindeutigem Status) richtig: {richtig} von {int(sicher.sum())}. "
               + ("Die Regel trifft den Optimalwert." if int(v @ x) == I["vopt"] else f"Die Regel verfehlt den Optimalwert um {I['vopt'] - int(v @ x)} Wertpunkte."))
    rg = ev.regeln(raster, n, skala, fam)
    namen = ["Pakete richtig", "ganze Menge gleich", "Optimalwert getroffen"]
    pk, mg, op = (rg["g_pakete"], rg["g_menge"], rg["g_opt"]) if regel == "greedy" else (rg["s_pakete"], rg["s_menge"], rg["s_opt"])
    bander, namen2 = [pk], [namen[0]]
    if mg is not None:
        bander.append(mg)
        namen2.append(namen[1])
    bander.append(op)
    namen2.append(namen[2])
    st.plotly_chart(V.build_band(namen2, bander, f"{C.REGEL_LABEL[regel].split(' (')[0]}: {zelltext} (Mittel und Spanne über fünf Ziehungen)"), width="stretch", key="chart_regel")
    _zeigen(ev.verdict_regel(rg, regel, n, skala, fam))
    if mg is None:
        st.caption("Bei dieser Zelle gibt es in mindestens einer Ziehung zu wenige Instanzen mit eindeutigem Optimum (unter 5): „ganze Menge gleich“ ist nicht messbar.")
    st.caption(f"Mittlere Wertlücke der Greedy-Menge zum Optimum: {ev.band_text(rg['g_luecke'], 2)}. Basis „Pakete richtig“: Raten des häufigeren Status ({ev.band_text(rg['basis'])}). "
               "Eine Regel trifft die meisten Einzelentscheidungen, nicht die Lösung.")
with tab4b:
    rr = ev.reihe(raster, rmax)
    st.markdown("Einmaschinen-Reihenfolge, **n = 7 Aufträge** (5 040 Reihenfolgen), Ziel Summe w · C; die Frage heißt „Warum Auftrag A vor B?“. Smith-Regel: A steht in jeder optimalen Reihenfolge "
                "vor B genau dann, wenn p_A / w_A < p_B / w_B (ohne Freigaben).")
    p, wq, r = S.instanz(int(seed), rmax)
    Cm = S.fertigstellung(p, r)
    om, mask = S.optimum(Cm, wq)
    seqs = S.smith_statisch(p, wq), S.dispatch(p, wq, r)
    best = S.PERMS[mask][0]
    st.dataframe([{"Auftrag": j + 1, "Dauer p": int(p[j]), "Gewicht w": int(wq[j]), "Freigabe r": int(r[j]), "p/w": ev.de(p[j] / wq[j], 2)} for j in range(7)], hide_index=True, width="stretch")
    st.caption(f"Instanz aus dem Zufalls-Seed {int(seed)}: Optimum {om} (optimale Reihenfolgen: {int(mask.sum())}; eine davon: {' → '.join(str(j + 1) for j in best)}); statische Smith-Reihenfolge "
               f"{' → '.join(str(j + 1) for j in seqs[0])} mit {S.zielwert(seqs[0], p, wq, r)}; Dispatch {' → '.join(str(j + 1) for j in seqs[1])} mit {S.zielwert(seqs[1], p, wq, r)}.")
    if rmax == 0:
        a = st.selectbox("Auftrag A", list(range(1, 8)), key="seq_a")
        b = st.selectbox("Auftrag B", [j for j in range(1, 8) if j != a], key="seq_b")
        stp = S.paarstatus(mask, a - 1, b - 1)
        if stp == 1:
            dw, dp = S.gegenfaktum_geschlossen(p, wq, a - 1, b - 1)
            st.success(f"Auftrag {a} steht in **jeder** optimalen Reihenfolge vor {b} (p/w {ev.de(p[a - 1] / wq[a - 1], 2)} gegen {ev.de(p[b - 1] / wq[b - 1], 2)}). Gegenfaktum (geschlossen): das Gewicht von "
                       f"{a} müsste um {dw} sinken (auf {int(wq[a - 1]) - dw}) oder seine Dauer um {dp} steigen (auf {int(p[a - 1]) + dp}), bis das nicht mehr sicher gilt.")
        elif stp == -1:
            st.info(f"Auftrag {a} steht in **keiner** optimalen Reihenfolge vor {b}: umgekehrt ist es sicher.")
        else:
            st.info(f"Auftrag {a} und {b} sind im **Gleichstand** (gleiches p/w): in mancher optimalen Reihenfolge steht A vor B, in mancher B vor A.")
    _zeigen(ev.verdict_reihenfolge(rr, rmax, seq_regel))
    st.dataframe([{"Freigaben bis": rm, "mehrere optimale Reihenfolgen": ev.band_text(ev.reihe(raster, rm)["mehrere"]), "Smith statisch trifft den Optimalwert": ev.band_text(ev.reihe(raster, rm)["s_opt"]),
                   "Dispatch trifft den Optimalwert": ev.band_text(ev.reihe(raster, rm)["d_opt"])} for rm in C.RMAX_STUFEN], hide_index=True, width="stretch")
    st.caption(f"Je Ziehung {rr['ni']} Instanzen, fünf Ziehungen. Eine um ±1 gestörte Dauer ändert die ganze optimale Reihenfolge in {ev.band_text(rr['stoerung1'])} der Neulösungen (Freigaben bis {rmax}).")

st.markdown("---")

# --- Experiment 5 -----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 5: Gilt die Begründung, wenn die Werte nur geschätzt sind?")
n5 = n if n in C.N_SCHAETZ else 12
if n5 != n:
    st.info(f"Das Raster zu Experiment 5 gibt es für n = 12 und n = 40; für n = {n} wird n = {n5} gezeigt (das Beispiel unten rechnet live mit der gewählten Instanz).")
st.markdown("Die „Schätzung“ ist ein Rauschmodell (kein Lernmodell): jeder Wert wird mit einem gleichverteilten Fehler bis ± Rauschen gestört. Auf den Schätzwerten wird die **Mindestanhebung** eines "
            "sicher draußen liegenden Pakets berechnet; geprüft wird, ob die Anhebung an den **wahren** Werten ausreicht.")
sz = ev.schaetzung(raster, n5, skala, fam, rauschen, sicherheit)
st.plotly_chart(V.build_band([C.SICHERHEIT_LABEL[k].replace("relativer Aufschlag ", "rel. ").replace("absoluter Abstand ", "abs. ").replace(" Rauscheinheiten", " RE") for k in C.SICHERHEITEN],
                             [ev.schaetzung(raster, n5, skala, fam, rauschen, k)["gueltig"] for k in C.SICHERHEITEN], f"Anteil gültiger Anhebungen: n = {n5}, s = {skala}, {fam}, Rauschen {int(100 * rauschen)} %",
                             farben=[C.COLOR_GLEICH] * 4 + [C.COLOR_DRIN] * 3), width="stretch", key="chart_schaetz")
_zeigen(ev.verdict_schaetz(sz, sicherheit, n5, skala, fam, rauschen))
aw = ev.aufwand(raster, n5, skala, fam, rauschen, sicherheit)
if aw is not None:
    st.caption(f"Aufwand (angehobener Betrag geteilt durch die wahre Schwelle plus 1, Median je Ziehung): {ev.de(aw[0], 2)} [{ev.de(aw[1], 2)} bis {ev.de(aw[2], 2)}]. Der Aufwand „bei 100 % etwa das Doppelte“ "
               "ist per Konstruktion (1 + Aufschlag) und kein Befund.")
st.markdown("**Beispiel mit der gewählten Instanz (live):**")
rng = SplitMix64(int(seed) + 9_000_000)
vh = E.schaetzung(v, rauschen, rng)
au = E.auswertung(v, w, cap, vh, rauschen)
if len(au["index"]):
    j = int(np.argmax(au["m_wahr"]))
    pk = int(au["index"][j])
    st.markdown(f"Paket {pk + 1}: Wert {int(v[pk])}, geschätzt {int(vh[pk])}; wahre Schwelle {int(au['m_wahr'][j])}, auf den Schätzwerten berechnete Schwelle {int(au['m_hat'][j])}. "
                f"Gültig bei: {', '.join(C.SICHERHEIT_LABEL[k] for k in C.SICHERHEITEN if au[k][j]) or 'keiner der Stufen'}; "
                f"ungültig bei: {', '.join(C.SICHERHEIT_LABEL[k] for k in C.SICHERHEITEN if not au[k][j]) or 'keiner'}.")
else:
    st.caption("In dieser Instanz und Schätzung gibt es kein Paket, das in Schätzung und Wahrheit sicher draußen und passend ist.")
st.caption("Gilt für dieses Rauschmodell (multiplikativ, gleichverteilt, bekanntes Niveau); der absolute Abstand setzt voraus, dass das Rauschniveau bekannt ist. „Volle Sicherheit gibt es nicht“ gilt für den "
           "relativen Aufschlag bei stark korrelierten Zahlen mit 40 Paketen, nicht für einen Abstand in Rauscheinheiten.")

st.markdown("---")

# --- Grenzen ----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme oder Grenze | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Wenn die Daten stimmen, hat jede Entscheidung genau eine Begründung, und sie gilt auch bei kleiner Änderung** | Gleichstand (Experiment 2), Hinauswurf bei mehr Kapazität (3), Regel trifft die Lösung nicht (4), Schätzfehler (5). | diese Demo |
| **Kein Modell, erfundene Zahlen** | Die Pakete sind erfunden, es gibt keine reale Datenquelle; nichts über Nutzen oder Gefahr in einem Betrieb, in der Logistik oder im Recht. | – |
| **Ein Problem (Rucksack), ein Tab (Reihenfolge)** | n bis 40, Zahlenskala 1 und 10, zwei Familien; Reihenfolge nur mit 7 Aufträgen. Größere Instanzen und andere Ziele (Verspätung) sind nicht gemessen. | offen |
| **Bekanntes Rauschniveau** | Der absolute Abstand in Rauscheinheiten hielt in der Messung jede Anhebung, setzt aber das Niveau voraus; Rauschen, das mit dem Wert wächst oder verzerrt ist, ist nicht gemessen. | offen |
| **Lesbarkeit für Menschen** | Ob solche Begründungen verstanden werden oder bessere Entscheidungen liefern, ist nicht gemessen; Menschenstudien sind nicht Teil dieser Demo. | – |
| **Lineare Programme** | Schattenpreise und Bereiche sind nicht Teil dieser Demo (siehe `lp-dualitaet-demo`); Kontext- und Unzulässigkeitsbegründungen (nächster Kontext, irreduzible unzulässige Teilsysteme) sind nicht gebaut. | offen |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind $n$ Pakete mit Gewicht $w_j$ und Wert $v_j$ und eine Kapazität $C$; gesucht ist eine Menge, die den Wert maximiert, und zu jedem Paket die Frage, ob es in
jeder, in keiner oder in mancher optimalen Menge liegt.

| Symbol | Bedeutung |
|---|---|
| $x_j \in \{0,1\}$ | Paket $j$ wird gepackt |
| $w_j$, $v_j$, $C$ | Gewicht, Wert, Kapazität (ganze Zahlen) |
| $f(c)$ | bester Wert bei Kapazität höchstens $c$ |
| $f_{-i}(c)$ | bester Wert ohne Paket $i$ bei Kapazität höchstens $c$ |
| $A_i$, $B_i$ | bester Wert mit beziehungsweise ohne Paket $i$ bei Kapazität $C$ |
| $d_i$ | Wertschwelle (Kippabstand) $\lvert A_i - B_i\rvert$ |

**Modell.** $\max \sum_{j=1}^{n} v_j x_j$ unter $\sum_{j=1}^{n} w_j x_j \le C$, $x_j \in \{0,1\}$.

**Dynamische Programmierung.** $f_{-i}(c) = \max\{f_{-i}(c),\; f_{-i}(c - w_j) + v_j\}$ für alle $j \ne i$ in der Reihenfolge der Pakete, mit $f_{-i}(c) = 0$ am Anfang und $c = C, \dots, w_j$ absteigend.

**Wertschwelle.** $A_i = v_i + f_{-i}(C - w_i)$ (für $w_i \le C$), $B_i = f_{-i}(C)$. Paket $i$ liegt in jeder optimalen Menge, wenn $A_i > B_i$, in keiner, wenn $A_i < B_i$, und im
Gleichstand, wenn $A_i = B_i$. Ändert man $v_i$ um $\lvert A_i - B_i\rvert$ in Richtung des Gleichstands, so entsteht genau der Gleichstand; eine weitere ganzzahlige Änderung kehrt den Status um.
**Preis der Erzwingung:** Paket $i$ zwingend zu packen senkt den Optimalwert um $B_i - A_i$ (für $A_i \le B_i$); das ist dieselbe Zahl.

**Einmaschinen-Reihenfolge.** $\min \sum_j w_j C_j$ unter Freigaben $r_j$. Ohne Freigaben steht $A$ in jeder optimalen Reihenfolge vor $B$ genau dann, wenn $p_A/w_A < p_B/w_B$ (Smith-Regel,
Vertauschungsargument). Gegenfakten: $d_w = w_A - \lfloor p_A w_B / p_B \rfloor$ und $d_p = \lceil p_B w_A / w_B \rceil - p_A$.

**Schätzwerte.** $\hat v_j = \operatorname{rint}\bigl(v_j (1 + \delta u_j)\bigr)$ mit $u_j \sim U(-1,1)$. Aktion: den wahren Wert um $m_{\text{Aktion}}$ anheben, relativ $m_{\text{Aktion}} = \hat m (1 + a)$,
absolut $\hat m + \kappa\,\delta\,\bar{\hat v}$; gültig, wenn $m_{\text{Aktion}} \ge m_{\text{wahr}}$.

**Quellen (Stufe).** Forel, Parmentier, Vidal (ICML 2023), Kurtz, Birbil, den Hertog (2024), Korikov, Beck (CP 2021) und Lera-Leri u. a. (X-MILP 2025): nur Zusammenfassungen über einen Quellenbericht gelesen
(Stufe A); die Rechnungen dieser Demo sind eigene und keine Nachrechnung dieser Arbeiten. Čyras u. a. (AAAI 2019): Stufe S. Smith-Regel und Rucksack-Tabelle sind Lehrbuch. **Keine Neuheitsaussage:**
Sensitivitätsanalyse und Gegenfaktum sind Lehrbuch und Forschung; neu ist hier nur die nachrechenbare Vorführung mit Zähleinheit und Orakeln.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
