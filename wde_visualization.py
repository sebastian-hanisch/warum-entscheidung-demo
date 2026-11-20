"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import wde_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=320, legend=False):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="rgba(0,0,0,0)", showlegend=legend, legend=dict(orientation="h", y=-0.25))
    return lock_axes(fig)


def farbe_status(status, fits):
    if not fits:
        return C.COLOR_NIE
    return C.COLOR_DRIN if status > 0 else (C.COLOR_RAUS if status < 0 else C.COLOR_GLEICH)


def build_pakete(v, w, status, fits, titel="Pakete: Wert, Farbe = Status"):
    """Ein Balken je Paket (Höhe = Wert, Beschriftung = Gewicht); grün drin in jeder optimalen Menge, rot in keiner, gelb Gleichstand, grau passt nie."""
    x = [f"P{i + 1}" for i in range(len(v))]
    fig = go.Figure(go.Bar(x=x, y=[int(a) for a in v], marker_color=[farbe_status(status[i], fits[i]) for i in range(len(v))], text=[f"w={int(a)}" for a in w], textposition="outside",
                           hovertemplate="%{x}: Wert %{y}<extra></extra>"))
    fig.update_xaxes(type="category")
    fig.update_yaxes(title_text="Wert")
    fig.update_layout(title=dict(text=titel, font=dict(size=13)))
    return _base(fig, height=300)


def build_streifen(streifen, cap, max_spalten=400):
    """Statusstreifen je Paket über der Kapazität (0 = nie bis cmax). Werte: 1 drin, -1 draußen, 0 Gleichstand, 2 passt nicht. Marke bei der gewählten Kapazität."""
    cmax = streifen.shape[1] - 1
    schritt = max(1, int(np.ceil((cmax + 1) / max_spalten)))
    cols = np.arange(0, cmax + 1, schritt)
    code = {-1: 0, 0: 1, 1: 2, 2: 3}
    z = np.vectorize(code.get)(streifen[:, cols])
    farben = [[0, C.COLOR_RAUS], [0.25, C.COLOR_RAUS], [0.25, C.COLOR_GLEICH], [0.5, C.COLOR_GLEICH], [0.5, C.COLOR_DRIN], [0.75, C.COLOR_DRIN], [0.75, C.COLOR_NIE], [1, C.COLOR_NIE]]
    fig = go.Figure(go.Heatmap(z=z, x=cols, y=[f"P{i + 1}" for i in range(len(z))], colorscale=farben, zmin=0, zmax=3, showscale=False, xgap=0, ygap=1,
                               hovertemplate="Kapazität %{x}, %{y}: Status-Code %{z}<extra></extra>"))
    fig.add_vline(x=cap, line_width=2, line_dash="dash", line_color="rgba(60,60,60,0.9)")
    fig.update_xaxes(title_text="Kapazität C′" + (f" (jede {schritt}. gezeigt)" if schritt > 1 else ""))
    fig.update_yaxes(type="category", autorange="reversed")
    return _base(fig, height=max(220, 26 * len(z) + 90))


def build_optwerte(opt, cap, max_punkte=400):
    cmax = len(opt) - 1
    schritt = max(1, int(np.ceil((cmax + 1) / max_punkte)))
    cols = np.arange(0, cmax + 1, schritt)
    fig = go.Figure(go.Scatter(x=cols, y=np.asarray(opt)[cols], mode="lines", line=dict(color=C.COLOR_DRIN, shape="hv")))
    fig.add_vline(x=cap, line_width=2, line_dash="dash", line_color="rgba(60,60,60,0.9)")
    fig.update_xaxes(title_text="Kapazität C′")
    fig.update_yaxes(title_text="Optimalwert")
    return _base(fig, height=220)


def build_band(namen, bander, titel, farben=None, achse="Anteil"):
    """Balken mit Spanne: bander = Liste (Mittel, kleinste Ziehung, größte Ziehung) je Name; Anteile 0 bis 1."""
    mitte = [b[0] for b in bander]
    fig = go.Figure(go.Bar(x=namen, y=mitte, marker_color=farben or C.COLOR_DRIN,
                           error_y=dict(type="data", symmetric=False, array=[b[2] - b[0] for b in bander], arrayminus=[b[0] - b[1] for b in bander], visible=True),
                           text=[f"{100 * m:.1f} %".replace(".", ",") for m in mitte], textposition="outside", hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    fig.update_xaxes(type="category")
    fig.update_yaxes(title_text=achse, range=[0, 1.12])
    fig.update_layout(title=dict(text=titel, font=dict(size=13)))
    return _base(fig)


def build_matrix(werte, zeilen, spalten, titel):
    """Tabelle der Anteile (Mittel) als Heatmap mit Zahlen; werte[zeile][spalte] in 0 bis 1."""
    z = np.array(werte, dtype=float)
    fig = go.Figure(go.Heatmap(z=z, x=spalten, y=zeilen, zmin=0, zmax=1, colorscale=[[0, "#eaf3f1"], [1, C.COLOR_DRIN]], showscale=False,
                               text=[[f"{100 * a:.1f} %".replace(".", ",") for a in r] for r in z], texttemplate="%{text}", hovertemplate="%{y}, %{x}: %{z:.3f}<extra></extra>"))
    fig.update_xaxes(type="category")
    fig.update_yaxes(type="category", autorange="reversed")
    fig.update_layout(title=dict(text=titel, font=dict(size=13)))
    return _base(fig, height=300)
