"""Permalink (auf die erlaubten Werte begrenzt), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben) und der Zufalls-Seed-Button."""

import random
from dataclasses import dataclass

import streamlit as st

import wde_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    options: tuple | None = None
    lo: float | None = None
    hi: float | None = None


D = C.DEFAULT
SETTING_SPECS = {
    "n_sel": SettingSpec("n", int, D["n"], options=C.N_LIVE),
    "familie_sel": SettingSpec("fam", str, D["familie"], options=C.FAMILIEN),
    "skala_sel": SettingSpec("skala", int, D["skala"], options=C.SKALEN),
    "kap_sel": SettingSpec("kap", float, D["kap"], options=C.KAPAZITAETEN),
    "seed_input": SettingSpec("seed", int, D["seed"], lo=0, hi=C.SEED_MAX),
    "hb_wert": SettingSpec("hbw", int, D["hb_wert"], lo=C.HB_WERT_MIN, hi=C.HB_WERT_MAX),
    "hb_kap": SettingSpec("hbk", int, D["hb_kap"], lo=C.HB_KAP_MIN, hi=C.HB_KAP_MAX),
    "regel_sel": SettingSpec("regel", str, D["regel"], options=C.REGELN),
    "rauschen_sel": SettingSpec("rausch", float, D["rauschen"], options=C.RAUSCHEN),
    "sicherheit_sel": SettingSpec("sich", str, D["sicherheit"], options=C.SICHERHEITEN),
    "rmax_sel": SettingSpec("rmax", int, D["rmax"], options=C.RMAX_STUFEN),
    "seq_regel_sel": SettingSpec("seqregel", str, D["seq_regel"], options=C.SEQ_REGELN),
}
PRESET_KEYS = {"n": "n_sel", "familie": "familie_sel", "skala": "skala_sel", "kap": "kap_sel", "seed": "seed_input", "hb_wert": "hb_wert", "hb_kap": "hb_kap", "regel": "regel_sel",
               "rauschen": "rauschen_sel", "sicherheit": "sicherheit_sel", "rmax": "rmax_sel", "seq_regel": "seq_regel_sel"}


def zulaessig(spec, value):
    if spec.options is not None:
        return value in spec.options
    return (spec.lo is None or value >= spec.lo) and (spec.hi is None or value <= spec.hi)


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, soweit sie zu den Optionen beziehungsweise Grenzen passen."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            if zulaessig(spec, value):
                st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None (zustandslos)."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(st.session_state.get(state_key, spec.default) == preset.get(preset_key_of.get(state_key), spec.default) for state_key, spec in SETTING_SPECS.items()):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}", type="primary" if aktiv == name else "secondary")


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
