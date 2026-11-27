"""DAW-agnostic parameter decoders for Avid stock AAX plugin states (``Complete Controls State``)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
from typing import Any

from .models import PtxInsert

_SLOPE_DB_PER_OCT: dict[int, int] = {0: 6, 1: 12, 2: 18, 3: 24}

_EQ1_FILTER_TYPES: dict[int, str] = {
    0: "high_pass",
    1: "low_shelf",
    2: "notch",
    3: "peak",
    4: "high_shelf",
    5: "low_pass",
}


@dataclass(frozen=True)
class DecodedPluginParameters:
    """Human-readable decoded parameter set for a recognized AAX insert plugin."""

    plugin_name: str
    fourcc: str
    parameters: dict[str, Any]
    raw_controls: dict[str, int | float]


def _r_hz(val: float | None) -> float | None:
    if val is None:
        return None
    return round(val, 0) if val >= 100.0 else round(val, 1)


def _r_db(val: float | None) -> float | None:
    return round(val, 1) if val is not None else None


def _r_q(val: float | None) -> float | None:
    return round(val, 2) if val is not None else None


def _r_ms(val: float | None) -> float | None:
    if val is None:
        return None
    mag = abs(val)
    if mag < 1.0:
        return round(val, 2)
    if mag < 100.0:
        return round(val, 1)
    return round(val, 0)


def _num(controls: Mapping[str, int | float], key: str, default: float = 0.0) -> float:
    val = controls.get(key)
    if val is None:
        return default
    fval = float(val)
    return fval if math.isfinite(fval) else default


def _decode_eq3_7band(controls: Mapping[str, int | float]) -> dict[str, Any] | None:
    if not ("frq1" in controls and "frq7" in controls and "gai2" in controls):
        return None
    bands: list[dict[str, Any]] = []
    hpf_is_pass = int(_num(controls, "hprg", 1)) == 1
    bands.append({
        "band": "HPF",
        "enabled": int(_num(controls, "in 1", 1)) != 0,
        "type": "high_pass" if hpf_is_pass else "notch",
        "freq_hz": _r_hz(_num(controls, "frq1")),
        "slope_db_per_oct": _SLOPE_DB_PER_OCT.get(int(_num(controls, "hpfs", 1)), 12) if hpf_is_pass else None,
        "q": None if hpf_is_pass else _r_q(_num(controls, "hpfq", 0.71)),
    })
    mid_names = {2: "LF", 3: "LMF", 4: "MF", 5: "HMF", 6: "HF"}
    for idx, label in mid_names.items():
        btype = "peak"
        if idx == 2 and int(_num(controls, "lfrg", 1)) == 1:
            btype = "low_shelf"
        elif idx == 6 and int(_num(controls, "hfrg", 1)) == 1:
            btype = "high_shelf"
        bands.append({
            "band": label,
            "enabled": int(_num(controls, f"in {idx}", 1)) != 0,
            "type": btype,
            "freq_hz": _r_hz(_num(controls, f"frq{idx}")),
            "gain_db": _r_db(_num(controls, f"gai{idx}", 0.0)),
            "q": _r_q(_num(controls, f"bnd{idx}", 1.0)),
        })
    lpf_is_pass = int(_num(controls, "lprg", 1)) == 1
    bands.append({
        "band": "LPF",
        "enabled": int(_num(controls, "in 7", 1)) != 0,
        "type": "low_pass" if lpf_is_pass else "notch",
        "freq_hz": _r_hz(_num(controls, "frq7")),
        "slope_db_per_oct": _SLOPE_DB_PER_OCT.get(int(_num(controls, "lpfs", 1)), 12) if lpf_is_pass else None,
        "q": None if lpf_is_pass else _r_q(_num(controls, "lpfq", 0.71)),
    })
    return {
        "bands": bands,
        "input_gain_db": _r_db(_num(controls, "inpt", 0.0)),
        "output_gain_db": _r_db(_num(controls, "outp", 0.0)),
        "phase_invert": int(_num(controls, "Phse", 0)) != 0,
    }


def _decode_eq3_1band(controls: Mapping[str, int | float]) -> dict[str, Any] | None:
    if not ("ctrf" in controls and "type" in controls):
        return None
    type_code = int(_num(controls, "type", 3))
    ftype = _EQ1_FILTER_TYPES.get(type_code, f"type_{type_code}")
    is_pass = ftype in ("high_pass", "low_pass")
    return {
        "type": ftype,
        "type_code": type_code,
        "freq_hz": _r_hz(_num(controls, "ctrf")),
        "gain_db": 0.0 if is_pass else _r_db(_num(controls, "gain", 0.0)),
        "q": None if is_pass else _r_q(_num(controls, "bndw", 1.0)),
        "slope_db_per_oct": _SLOPE_DB_PER_OCT.get(int(_num(controls, "slop", 1)), 12) if is_pass else None,
        "trim_db": _r_db(_num(controls, "trim", 0.0)),
        "phase_invert": int(_num(controls, "Phse", 0)) != 0,
    }


def _decode_dyn3_comp(controls: Mapping[str, int | float]) -> dict[str, Any] | None:
    if not ("sldC" in controls and "sld1" in controls and "sld2" in controls):
        return None
    return {
        "threshold_db": _r_db(_num(controls, "sldC")),
        "ratio": round(_num(controls, "sld2"), 1),
        "attack_ms": _r_ms(_num(controls, "sld1", 0.0) * 1000.0),
        "release_ms": _r_ms(_num(controls, "sld3", 0.0) * 1000.0),
        "knee_db": _r_db(_num(controls, "sldG", 0.0)),
        "makeup_gain_db": _r_db(_num(controls, "sldD", 0.0)),
        "key_hpf_hz": _r_hz(_num(controls, "LoFF")) if int(_num(controls, "LoFE", 0)) != 0 else None,
        "key_lpf_hz": _r_hz(_num(controls, "HiFF")) if int(_num(controls, "HiFE", 0)) != 0 else None,
        "phase_invert": int(_num(controls, "Phse", 0)) != 0,
    }


def _decode_dyn3_deesser(controls: Mapping[str, int | float]) -> dict[str, Any] | None:
    if not ("DFrq" in controls and "DRng" in controls):
        return None
    return {
        "freq_hz": _r_hz(_num(controls, "DFrq")),
        "range_db": _r_db(_num(controls, "DRng")),
        "hf_only": int(_num(controls, "HFOn", 0)) != 0,
    }


def _decode_dyn3_expgate(controls: Mapping[str, int | float]) -> dict[str, Any] | None:
    if not ("sldC" in controls and "sld2" in controls and "sldG" in controls):
        return None
    return {
        "threshold_db": _r_db(_num(controls, "sldC")),
        "ratio": round(_num(controls, "sld2"), 1),
        "range_db": _r_db(_num(controls, "sldG")),
        "attack_ms": _r_ms(_num(controls, "sld1", 0.0) * 1000.0),
        "hold_ms": _r_ms(_num(controls, "sldD", 0.0) * 1000.0),
        "release_ms": _r_ms(_num(controls, "sld3", 0.0) * 1000.0),
        "key_hpf_hz": _r_hz(_num(controls, "LoFF")) if int(_num(controls, "LoFE", 0)) != 0 else None,
        "key_lpf_hz": _r_hz(_num(controls, "HiFF")) if int(_num(controls, "HiFE", 0)) != 0 else None,
        "lookahead": int(_num(controls, "LkAh", 0)) != 0,
        "phase_invert": int(_num(controls, "Phse", 0)) != 0,
    }


_PLUGIN_DECODERS: dict[str, Callable[[Mapping[str, int | float]], dict[str, Any] | None]] = {
    "EQ3 7-Band": _decode_eq3_7band,
    "EQ3 1-Band": _decode_eq3_1band,
    "Dyn3 Compressor/Limiter": _decode_dyn3_comp,
    "Dyn3 De-Esser": _decode_dyn3_deesser,
    "Dyn3 Expander/Gate": _decode_dyn3_expgate,
}

_PRODUCT_4CC_TO_PLUGIN: dict[str, str] = {
    "EQ7B": "EQ3 7-Band",
    "EQ1B": "EQ3 1-Band",
    "CpLm": "Dyn3 Compressor/Limiter",
    "DEss": "Dyn3 De-Esser",
    "ExpG": "Dyn3 Expander/Gate",
}


def decode_insert_parameters(insert: PtxInsert) -> DecodedPluginParameters | None:
    """Decode human-readable parameters for a supported stock AAX plugin insert, or ``None``."""
    if insert.state is None or not insert.state.controls:
        return None
    canonical_name = insert.plugin_name
    decoder = _PLUGIN_DECODERS.get(canonical_name)
    if decoder is None and insert.fourcc:
        parts = insert.fourcc.split("/")
        prod4 = parts[1] if len(parts) >= 2 else ""
        mapped = _PRODUCT_4CC_TO_PLUGIN.get(prod4)
        if mapped is not None:
            canonical_name = mapped
            decoder = _PLUGIN_DECODERS.get(mapped)
    if decoder is None:
        return None
    params = decoder(insert.state.controls)
    if params is None:
        return None
    return DecodedPluginParameters(
        plugin_name=canonical_name,
        fourcc=insert.fourcc,
        parameters=params,
        raw_controls=dict(insert.state.controls),
    )


def _lin(lo: float, hi: float) -> Callable[[float], float]:
    return lambda r: lo + (r + 0.5) * (hi - lo)


def _log(lo: float, hi: float) -> Callable[[float], float]:
    return lambda r: lo * ((hi / lo) ** (r + 0.5))


_AUTO_TAPERS: dict[str, dict[str, Callable[[float], float]]] = {
    "EQ3 7-Band": {
        "gai2": _lin(-18.0, 18.0), "gai3": _lin(-18.0, 18.0), "gai4": _lin(-18.0, 18.0),
        "gai5": _lin(-18.0, 18.0), "gai6": _lin(-18.0, 18.0),
        "frq2": _log(20.0, 500.0), "frq3": _log(40.0, 1000.0), "frq4": _log(125.0, 8000.0),
        "frq5": _log(200.0, 18000.0), "frq6": _log(1800.0, 20000.0),
        "bnd2": _log(0.1, 2.0), "bnd3": _log(0.1, 10.0), "bnd4": _log(0.1, 10.0),
        "bnd5": _log(0.1, 10.0), "bnd6": _log(0.1, 2.0),
    },
    "EQ3 1-Band": {"gain": _lin(-18.0, 18.0)},
    "Dyn3 Compressor/Limiter": {
        "sldC": _lin(-60.0, 0.0), "sld2": _log(1.0, 100.0), "sld1": _log(0.01, 300.0),
        "sld3": _log(5.0, 4000.0), "sldD": _lin(0.0, 40.0), "sldG": _lin(0.0, 30.0),
    },
    "Dyn3 De-Esser": {"DFrq": _log(500.0, 16000.0)},
    "Dyn3 Expander/Gate": {
        "sldC": _lin(-60.0, 0.0), "sld2": _log(1.0, 100.0), "sldG": _lin(-80.0, 0.0),
        "sld1": _log(0.01, 300.0), "sldD": _log(5.0, 4000.0), "sld3": _log(5.0, 4000.0),
    },
}


def decode_plugin_automation_value(
    plugin_name: str,
    parameter_tag: str,
    raw_normalized: float,
    *,
    parameter_name: str = "",
) -> float | None:
    """Map a decoded ``0x260b`` plugin automation value to physical units (Hz/dB/Q/ms/ratio/0..1).

    Note: Follows Pro Tools 12.5 fixed-point ``[-0.5, +0.5]`` taper conventions (clamped to
    ``[-0.5, +0.5]``) for supported Avid stock EQ3/Dyn3 tags and ``0.0``/``1.0`` for
    ``Master Bypass``. Non-finite values return ``None``.
    """
    val = float(raw_normalized)
    if not math.isfinite(val):
        return None
    if parameter_name == "Master Bypass":
        return 1.0 if val != 0.0 else 0.0
    taper = _AUTO_TAPERS.get(plugin_name, {}).get(parameter_tag)
    if taper is None:
        return None
    return taper(max(-0.5, min(0.5, val)))
