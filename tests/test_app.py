"""End-to-end Smoke-Test via Streamlits AppTest: lädt app.py mit den Standardeinstellungen, klickt jeden Button, stellt jeden Regler auf jede Option (Zahlenregler an die Grenzen) und prüft, dass
kein Python-Fehler auftritt (insbesondere StreamlitDuplicateElementId bei mehreren plotly_chart-Aufrufen ohne eindeutiges key=). Dazu die Kernaussagen der Seite."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import wde_constants as C
import wde_presets as P

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))
REGLER_ARTEN = ("selectbox", "radio", "select_slider", "slider", "number_input")


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _regler(at, key):
    for art in REGLER_ARTEN:
        try:
            return getattr(at.sidebar, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(key)


def _text(at):
    teile = [m.value for m in at.markdown] + [m.value for m in at.info] + [m.value for m in at.success] + [m.value for m in at.warning] + [c.value for c in at.caption]
    return "\n".join(teile)


def test_app_loads_without_exception():
    at = _fresh_app()
    assert "keine reale Datenquelle" in _text(at)


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


@pytest.mark.parametrize("state_key", list(P.SETTING_SPECS))
def test_every_option_or_bound_of_every_control_does_not_raise(state_key):
    spec = P.SETTING_SPECS[state_key]
    werte = spec.options if spec.options is not None else (spec.lo, spec.hi)
    for option in werte:
        at = _fresh_app()
        _regler(at, state_key).set_value(option).run(timeout=TIMEOUT)
        assert not at.exception, (state_key, option, [str(e) for e in at.exception])


def test_voreinstellung_zeigt_fuenf_experimente_handbeispiel_grenzen_und_formeln():
    at = _fresh_app()
    sub = [s.value for s in at.subheader]
    assert [s.split(":")[0] for s in sub if s.startswith("🔬 Experiment")] == [f"🔬 Experiment {i}" for i in range(1, 6)]
    assert "🚧 Wo die Annahmen enden" in sub and any(e.label == "📐 Mathematische Formulierung" for e in at.expander)
    assert any("Optimale Mengen" == m.label for m in at.metric)


def test_handbeispiel_standard_nennt_paket_2_schwelle_2():
    at = _fresh_app()
    _regler(at, "hb_wert").set_value(7).run(timeout=TIMEOUT)
    assert any("Paket 2 (Wert 7) bleibt draußen" in s.value for s in at.success)
    _regler(at, "hb_wert").set_value(9).run(timeout=TIMEOUT)
    assert any("Gleichstand mit drei optimalen Mengen" in s.value for s in at.info)
    assert not any("Paket 2 (Wert 7) bleibt draußen" in s.value for s in at.success)


def test_jedes_preset_laeuft_ohne_fehler_und_zeigt_verdikte():
    for name in C.PRESETS:
        at = _fresh_app()
        at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        assert len(at.success) + len(at.info) + len(at.warning) >= 5, name


def test_stark_korreliertes_preset_meldet_nicht_gleichstand_selten():
    at = _fresh_app()
    at.sidebar.button(key="preset_Stark korreliert (Gleichstand häufig)").click().run(timeout=TIMEOUT)
    texte = [w.value for w in at.warning] + [i.value for i in at.info] + [s.value for s in at.success]
    treffer = [t for t in texte if t.startswith("Bei n = 12, Zahlenskala 1, stark korreliert")]
    assert treffer and "mehrere gleich gute Antworten" in treffer[0]                     # Gleichstand ist hier die Regel: Verdikt „warning“, nicht „selten“
    assert not any("selten, aber nicht null" in t for t in treffer)
