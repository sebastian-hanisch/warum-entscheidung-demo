"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, jede wählbare Zelle ist im Raster vorgerechnet."""

import wde_constants as C
import wde_evaluation as ev
import wde_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_among_the_options_or_inside_the_bounds():
    for name, preset in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            assert P.zulaessig(P.SETTING_SPECS[state_key], preset[key]), (name, key, preset[key])


def test_every_preset_differs_from_every_other_in_at_least_one_setting():
    werte = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(werte)) == len(werte)


def test_default_preset_equals_the_default_settings():
    erster = C.PRESETS[next(iter(C.PRESETS))]
    for key, state_key in P.PRESET_KEYS.items():
        assert erster[key] == P.SETTING_SPECS[state_key].default


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_zulaessig_randfaelle():
    spec = P.SETTING_SPECS["hb_wert"]
    assert P.zulaessig(spec, spec.lo) and P.zulaessig(spec, spec.hi) and not P.zulaessig(spec, spec.lo - 1) and not P.zulaessig(spec, spec.hi + 1)
    assert not P.zulaessig(P.SETTING_SPECS["n_sel"], 7) and P.zulaessig(P.SETTING_SPECS["n_sel"], 40)


def test_jede_waehlbare_zelle_ist_im_raster_vorgerechnet():
    R = ev.lade_raster()
    for n in C.N_LIVE:
        for s in C.SKALEN:
            for f in C.FAMILIEN:
                assert f"{n}|{s}|{f}" in R["zellen"]
    for n in C.N_SCHAETZ:
        for s in C.SKALEN:
            for f in C.FAMILIEN:
                for d in C.RAUSCHEN:
                    assert f"{n}|{s}|{f}|{d:g}" in R["schaetz"]
    assert set(map(str, C.RMAX_STUFEN)) == set(R["reihenfolge"]) and set(C.SICHERHEITEN) == set(R["schaetz"]["12|1|unkorreliert|0.05"][0]["cnt"])
