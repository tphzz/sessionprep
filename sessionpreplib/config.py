from __future__ import annotations

import json
import math
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .audio import legacy_patterns_to_regex, migrate_legacy_override as migrate_legacy_override
from .models import ConfigError as ConfigError, ParamSpec
from .paths import (
    DEFAULT_STEREO_WINDOW_MS as DEFAULT_STEREO_WINDOW_MS,
    get_app_dir as get_app_dir,
)

PRESET_SCHEMA_VERSION = "1.0"

# Keys that are internal/CLI-only and should not be saved in presets
_INTERNAL_KEYS = {
    "execute",
    "overwrite",
    "output_folder",
    "backup",
    "report",
    "json",
    "_source_dir",
}


@dataclass
class ConfigFieldError:
    """A single validation error for one configuration field.

    Attributes:
        key:     The config key that failed validation.
        value:   The offending value.
        message: Human-readable explanation of what is wrong.
    """

    key: str
    value: Any
    message: str


@dataclass(frozen=True)
class ConfigChangeImpact:
    """Lifecycle impact of changing one structured config preset."""

    phase1_keys: frozenset[str] = frozenset()
    phase2_keys: frozenset[str] = frozenset()
    presentation_keys: frozenset[str] = frozenset()
    daw_keys: frozenset[str] = frozenset()
    repair_keys: frozenset[str] = frozenset()

    @property
    def changed(self) -> bool:
        return bool(
            self.phase1_keys
            or self.phase2_keys
            or self.presentation_keys
            or self.daw_keys
            or self.repair_keys
        )

    @property
    def requires_phase1(self) -> bool:
        return bool(self.phase1_keys)

    @property
    def requires_phase2(self) -> bool:
        return bool(self.phase2_keys)

    @property
    def presentation_only(self) -> bool:
        return self.changed and not (
            self.phase1_keys or self.phase2_keys or self.daw_keys or self.repair_keys
        )

    @property
    def daw_only(self) -> bool:
        return self.changed and not (
            self.phase1_keys or self.phase2_keys or self.presentation_keys or self.repair_keys
        )


def default_config() -> dict[str, Any]:
    """Returns the built-in default configuration derived from ParamSpec definitions."""
    from .detectors import default_detectors
    from .processors import default_processors

    cfg: dict[str, Any] = {}
    for spec in (*ANALYSIS_PARAMS, *PRESENTATION_PARAMS, *REPAIR_PARAMS):
        cfg[spec.key] = deepcopy(spec.default)
    for component in (*default_detectors(), *default_processors()):
        for spec in component.config_params():
            cfg[spec.key] = deepcopy(spec.default)
    cfg.update({
        "group": [],
        "anchor": None,
        "execute": False,
        "overwrite": False,
        "output_folder": "processed",
        "backup": "_originals",
        "report": "sessionprep.txt",
        "json": "sessionprep.json",
    })
    return cfg


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    """Merge multiple config dicts left-to-right (later overrides earlier).

    List values (``group``) are concatenated.
    """
    _LIST_KEYS = {"group"}
    result: dict[str, Any] = {}
    for cfg in configs:
        for k, v in cfg.items():
            if (
                k in _LIST_KEYS
                and k in result
                and isinstance(result[k], list)
                and isinstance(v, list)
            ):
                result[k] = result[k] + v
            else:
                result[k] = v
    return result


_RETIRED_FLAT_KEYS = frozenset({
    "audio_classifier_enabled",
    "crest_threshold",
    "decay_lookahead_ms",
    "decay_db_threshold",
    "sparse_density_threshold",
    "force_sustained",
})


def migrate_legacy_config(raw: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Migrate retired 3-metric classifier keys in flat or structured config dicts."""
    cfg = deepcopy(raw)
    notices: list[str] = []

    if isinstance(cfg.get("detectors"), dict) and "audio_classifier" in cfg["detectors"]:
        cfg["detectors"].pop("audio_classifier", None)
        notices.append("Removed retired 'detectors.audio_classifier' configuration.")

    procs = cfg.get("processors")
    if isinstance(procs, dict) and isinstance(procs.get("bimodal_normalize"), dict):
        bn = procs["bimodal_normalize"]
        if "force_transient" in bn:
            pattern = legacy_patterns_to_regex(bn.pop("force_transient"))
            if pattern and not bn.get("force_peak"):
                bn["force_peak"] = pattern
            notices.append(f"Migrated legacy 'processors.bimodal_normalize.force_transient' to 'force_peak' = {pattern!r}.")
        for retired in ("force_sustained", "crest_threshold"):
            if retired in bn:
                bn.pop(retired, None)
                notices.append(f"Removed retired key 'processors.bimodal_normalize.{retired}'.")

    if "force_transient" in cfg:
        ft = cfg.pop("force_transient")
        pattern = legacy_patterns_to_regex(ft)
        if pattern:
            if isinstance(procs, dict):
                bn = procs.setdefault("bimodal_normalize", {})
                if isinstance(bn, dict) and not bn.get("force_peak"):
                    bn["force_peak"] = pattern
            elif not cfg.get("force_peak"):
                cfg["force_peak"] = pattern
            notices.append(f"Migrated legacy 'force_transient' to 'force_peak' = {pattern!r}.")

    for key in sorted(_RETIRED_FLAT_KEYS):
        if key in cfg:
            cfg.pop(key, None)
            notices.append(f"Removed retired key '{key}'.")

    return cfg, notices


def load_preset(path: str) -> dict[str, Any]:
    """
    Load a JSON preset file. Returns a partial config dict (migrated if legacy).
    Raises ConfigError if the file cannot be read or parsed.
    """
    if not os.path.isfile(path):
        raise ConfigError(f"Preset file not found: {path}")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON in preset file {path}: {e}") from e
    except OSError as e:
        raise ConfigError(f"Cannot read preset file {path}: {e}") from e

    if not isinstance(data, dict):
        raise ConfigError(
            f"Preset file must contain a JSON object, got {type(data).__name__}"
        )

    # Strip metadata keys — they are informational, not config
    preset = {
        k: v for k, v in data.items() if k not in ("schema_version", "_description")
    }
    migrated, _ = migrate_legacy_config(preset)
    return migrated


def save_preset(
    config: dict[str, Any], path: str, *, description: str | None = None
) -> None:
    """
    Save a config dict as a JSON preset file.
    Internal/CLI-only keys are excluded automatically.
    """
    preset: dict[str, Any] = {"schema_version": PRESET_SCHEMA_VERSION}
    if description:
        preset["_description"] = description

    defaults = default_config()
    for k, v in config.items():
        if k in _INTERNAL_KEYS:
            continue
        if k.startswith("_"):
            continue
        # Only save values that differ from defaults
        if k in defaults and defaults[k] == v:
            continue
        preset[k] = v

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(preset, f, indent=4, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Shared parameter sections
# ---------------------------------------------------------------------------

ANALYSIS_PARAMS: list[ParamSpec] = [
    ParamSpec(
        key="window",
        type=int,
        default=400,
        min=1,
        label="RMS window size (ms)",
        description="Unweighted RMS analysis window length in milliseconds.",
    ),
    ParamSpec(
        key="stereo_mode",
        type=str,
        default="avg",
        choices=["avg", "sum"],
        label="Stereo RMS mode",
        description="How left/right channels are combined for RMS.",
    ),
    ParamSpec(
        key="rms_anchor",
        type=str,
        default="percentile",
        choices=["percentile", "max"],
        label="RMS anchor strategy",
        description=(
            "How to pick the representative RMS level from the distribution of "
            "momentary RMS windows. 'percentile' (default) takes the Nth "
            "percentile of gated windows — robust to outliers like breath pops "
            "or bleed spikes, tracks the chorus-level loudness that drives your "
            "insert processing. 'max' takes the single loudest window — useful "
            "for very short files (single hits) but fragile for longer material "
            "where one anomalous moment can skew the gain decision."
        ),
    ),
    ParamSpec(
        key="rms_percentile",
        type=(int, float),
        default=95.0,
        min=0.0,
        max=100.0,
        min_exclusive=True,
        max_exclusive=True,
        label="RMS percentile",
        description=(
            "Which percentile of the gated RMS window distribution to use as "
            "the anchor (only applies when anchor = percentile). P95 means "
            "95% of active windows are at or below the anchor — in practice, "
            "this represents 'what the loud sections typically sound like'. "
            "Lower values (e.g. 90) produce a lower anchor and more aggressive "
            "gain. Higher values approach the max window."
        ),
    ),
    ParamSpec(
        key="gate_relative_db",
        type=(int, float),
        default=40.0,
        min=0.0,
        label="Relative gate (dB)",
        description=(
            "RMS windows more than this many dB below the loudest window are "
            "excluded before computing the anchor and tail statistics. This is "
            "relative to the loudest window, not an absolute dBFS value. "
            "Critical for sparse tracks (FX hits, vocal doubles) where most "
            "windows are near-silent — without gating, the percentile anchor "
            "would be dominated by silence."
        ),
    ),
    ParamSpec(
        key="dbfs_convention",
        type=str,
        default="standard",
        choices=["standard", "aes17"],
        presentation_only=True,
        label="dBFS convention",
        description=(
            "Standard: 0 dBFS = full-scale digital. "
            "AES17: +3.01 dB offset applied to RMS readouts/targets only (sample peaks stay unshifted)."
        ),
    ),
    # -- Global processing defaults ------------------------------------------
    ParamSpec(
        key="fader_headroom_db",
        type=(int, float),
        default=8.0,
        min=0.0,
        label="Fader headroom (dB)",
        description=(
            "Minimum dB of headroom to reserve at the top of the DAW fader "
            "range for mixing moves. Fader offsets are shifted so the highest "
            "fader stays this many dB below the DAW's fader ceiling. "
            "Set to 0 to disable rebalancing."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Presentation parameters  (config-preset-scoped, not analysis-affecting)
# ---------------------------------------------------------------------------

PRESENTATION_PARAMS: list[ParamSpec] = [
    ParamSpec(
        key="show_clean_detectors",
        type=bool,
        default=False,
        presentation_only=True,
        label="Show clean detector results",
        description=(
            "When enabled, detectors that found no issues (OK) are "
            "shown in the file detail view and summary. Disable to "
            "reduce clutter and focus on problems, warnings, and "
            "informational findings only."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Validation  (ParamSpec-driven)
# ---------------------------------------------------------------------------


def _validate_numeric_range(
    spec: ParamSpec,
    value: int | float,
) -> ConfigFieldError | None:
    if isinstance(value, float) and not math.isfinite(value):
        return ConfigFieldError(spec.key, value, f"{spec.label} must be finite.")
    if spec.min is not None:
        if spec.min_exclusive and value <= spec.min:
            return ConfigFieldError(
                spec.key, value, f"{spec.label} must be greater than {spec.min}."
            )
        if not spec.min_exclusive and value < spec.min:
            return ConfigFieldError(
                spec.key, value, f"{spec.label} must be at least {spec.min}."
            )
    if spec.max is not None:
        if spec.max_exclusive and value >= spec.max:
            return ConfigFieldError(
                spec.key, value, f"{spec.label} must be less than {spec.max}."
            )
        if not spec.max_exclusive and value > spec.max:
            return ConfigFieldError(
                spec.key, value, f"{spec.label} must be at most {spec.max}."
            )
    return None


def _validate_single_param(
    spec: ParamSpec,
    value: Any,
) -> ConfigFieldError | None:
    if value is None:
        if spec.nullable:
            return None
        return ConfigFieldError(spec.key, value, f"{spec.label} must not be empty.")

    expected = spec.type
    if expected is not bool and isinstance(value, bool):
        return ConfigFieldError(
            spec.key,
            value,
            f"{spec.label} must be {_type_label(expected)}, got boolean.",
        )
    if not isinstance(value, expected):
        return ConfigFieldError(
            spec.key,
            value,
            f"{spec.label} must be {_type_label(expected)}, got {type(value).__name__}.",
        )

    if spec.choices is not None and value not in spec.choices:
        opts = ", ".join(repr(c) for c in spec.choices)
        return ConfigFieldError(spec.key, value, f"{spec.label} must be one of {opts}.")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        err = _validate_numeric_range(spec, value)
        if err is not None:
            return err

    if spec.item_type is not None and isinstance(value, list):
        for i, item in enumerate(value):
            if not isinstance(item, spec.item_type):
                return ConfigFieldError(
                    spec.key,
                    value,
                    f"{spec.label}[{i}] must be {spec.item_type.__name__}, "
                    f"got {type(item).__name__}.",
                )
    return None


def validate_param_values(
    params: list[ParamSpec],
    values: dict[str, Any],
) -> list[ConfigFieldError]:
    """Validate *values* against a list of :class:`ParamSpec` definitions.

    Returns a (possibly empty) list of :class:`ConfigFieldError` objects.
    Only keys present in *values* are checked; missing keys are not errors
    (they will receive their default).
    """
    errors: list[ConfigFieldError] = []
    for spec in params:
        if spec.key not in values:
            continue
        err = _validate_single_param(spec, values[spec.key])
        if err is not None:
            errors.append(err)
    return errors


REPAIR_PARAMS = [
    ParamSpec("repair_auto_dc", bool, False, "DC removal",
              "Automatically select 2 Hz DC removal when a track has a persistent signed bias."),
    ParamSpec("repair_auto_low_cut", bool, False, "Subsonic filter",
              "Automatically select the low cut for sustained infrasonic energy after DC treatment."),
    ParamSpec("repair_auto_ultrasonic", bool, False, "Ultrasonic filter",
              "Automatically select filtering for sustained ultrasonic content on high-rate sources."),
    ParamSpec("repair_low_cut_hz", (int, float), 10.0, "Low-cut cutoff (Hz)",
              "Whole-track high-pass corner for subsonic filtering.", min=5, max=40),
]


def _all_param_specs() -> list[ParamSpec]:
    """Collect every :class:`ParamSpec` from analysis, detectors,
    and processors.  Used by the flat-config validators."""
    from .detectors import default_detectors
    from .processors import default_processors

    from .daw_processors import default_daw_processors

    specs = [*ANALYSIS_PARAMS, *REPAIR_PARAMS]
    for det in default_detectors():
        specs.extend(det.config_params())
    for proc in default_processors():
        specs.extend(proc.config_params())
    for dp in default_daw_processors():
        specs.extend(dp.config_params())
    return specs


def _validate_normalization_invariants(
    values: dict[str, Any],
    *,
    prefix: str = "",
) -> list[ConfigFieldError]:
    """Check cross-field Peak-Capped RMS invariants (target_rms <= target_peak, valid regex)."""
    errors: list[ConfigFieldError] = []
    t_rms = values.get("target_rms")
    t_peak = values.get("target_peak")
    if (
        isinstance(t_rms, (int, float)) and not isinstance(t_rms, bool)
        and isinstance(t_peak, (int, float)) and not isinstance(t_peak, bool)
        and math.isfinite(float(t_rms)) and math.isfinite(float(t_peak))
        and float(t_rms) > float(t_peak)
    ):
        errors.append(
            ConfigFieldError(
                f"{prefix}target_rms",
                t_rms,
                f"Target RMS ({t_rms} dBFS) must not exceed Peak ceiling ({t_peak} dBFS).",
            )
        )
    pattern = values.get("force_peak")
    if isinstance(pattern, str) and pattern.strip():
        try:
            re.compile(pattern)
        except re.error as exc:
            errors.append(
                ConfigFieldError(
                    f"{prefix}force_peak",
                    pattern,
                    f"Force peak pattern is not a valid regular expression: {exc}.",
                )
            )
    return errors


def validate_config_fields(config: dict[str, Any]) -> list[ConfigFieldError]:
    """Validate a **flat** config dict against all known :class:`ParamSpec`
    definitions (analysis + every detector + every processor).

    Returns structured errors.  Never raises.
    """
    errors = validate_param_values(_all_param_specs(), config)
    errors.extend(_validate_normalization_invariants(config))
    return errors


def validate_config(config: dict[str, Any]) -> None:
    """Validate a flat config dict.

    Raises :class:`ConfigError` listing every invalid field.
    Backward-compatible wrapper around :func:`validate_config_fields`.
    """
    errors = validate_config_fields(config)
    if errors:
        lines = [e.message for e in errors]
        raise ConfigError(
            "Configuration has invalid values:\n  • " + "\n  • ".join(lines)
        )


# ---------------------------------------------------------------------------
# Structured config  (GUI config file format)
# ---------------------------------------------------------------------------


def build_structured_defaults() -> dict[str, Any]:
    """Build a structured config dict with all defaults, organized by section.

    Returns::

        {
            "analysis": { ... },
            "detectors": {
                "<detector_id>": { ... },
                ...
            },
            "processors": {
                "<processor_id>": { ... },
                ...
            },
        }
    """
    from .detectors import default_detectors
    from .processors import default_processors
    from .daw_processors import default_daw_processors

    structured: dict[str, Any] = {
        "analysis": {p.key: p.default for p in ANALYSIS_PARAMS},
        "detectors": {},
        "processors": {},
        "daw_processors": {},
    }

    for det in default_detectors():
        params = det.config_params()
        if params:
            structured["detectors"][det.id] = {p.key: p.default for p in params}

    for proc in default_processors():
        params = proc.config_params()
        if params:
            structured["processors"][proc.id] = {p.key: p.default for p in params}
    structured["processors"]["corrective"] = {p.key: p.default for p in REPAIR_PARAMS}

    for dp in default_daw_processors():
        params = dp.config_params()
        if params:
            structured["daw_processors"][dp.id] = {p.key: p.default for p in params}
        # DAWProject: include templates list default
        if dp.id == "dawproject":
            structured["daw_processors"].setdefault(dp.id, {})
            structured["daw_processors"][dp.id].setdefault("dawproject_templates", [])
        # Pro Tools: include templates list default
        elif dp.id == "protools":
            structured["daw_processors"].setdefault(dp.id, {})
            structured["daw_processors"][dp.id].setdefault("protools_templates", [])

    return structured


def strip_presentation_keys(structured: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of *structured* with all ``presentation_only`` keys removed.

    Used to compare configs for analysis-affecting changes only.
    Presentation-only keys (e.g. ``report_as``) are tagged via
    :attr:`ParamSpec.presentation_only`.
    """
    import copy
    from .detectors import default_detectors
    from .processors import default_processors

    # Collect presentation-only keys per (section_type, component_id)
    pres_keys: dict[tuple[str, str], set[str]] = {}
    for det in default_detectors():
        keys = {p.key for p in det.config_params() if p.presentation_only}
        if keys:
            pres_keys[("detectors", det.id)] = keys
    for proc in default_processors():
        keys = {p.key for p in proc.config_params() if p.presentation_only}
        if keys:
            pres_keys[("processors", proc.id)] = keys

    stripped = copy.deepcopy(structured)
    for spec in ANALYSIS_PARAMS:
        if spec.presentation_only and isinstance(stripped.get("analysis"), dict):
            stripped["analysis"].pop(spec.key, None)
    if isinstance(stripped.get("presentation"), dict):
        for spec in PRESENTATION_PARAMS:
            if spec.presentation_only:
                stripped["presentation"].pop(spec.key, None)
    for (section_type, comp_id), keys in pres_keys.items():
        section = stripped.get(section_type, {}).get(comp_id)
        if isinstance(section, dict):
            for k in keys:
                section.pop(k, None)
    return stripped


_FLAT_PRESET_SECTIONS = ("analysis", "presentation")
_COMPONENT_PRESET_SECTIONS = ("detectors", "processors", "daw_processors")


def _structured_config_values(config: dict[str, Any]) -> dict[str, Any]:
    """Index preset fields by the same qualified keys used for change impacts."""
    values: dict[str, Any] = {}
    for section in _FLAT_PRESET_SECTIONS:
        fields = config.get(section)
        if isinstance(fields, dict):
            values.update((f"{section}.{key}", value) for key, value in fields.items())
    for section in _COMPONENT_PRESET_SECTIONS:
        components = config.get(section)
        if isinstance(components, dict):
            for component, fields in components.items():
                if isinstance(fields, dict):
                    values.update(
                        (f"{section}.{component}.{key}", value)
                        for key, value in fields.items()
                    )
    return values


def resolve_session_config(
    preset: dict[str, Any], overrides: dict[str, Any],
) -> dict[str, Any]:
    """Combine a preset and explicit per-field overrides without sharing state."""
    effective = deepcopy(preset)
    for path, value in overrides.items():
        section, _, field = path.partition(".")
        if section in _FLAT_PRESET_SECTIONS and field:
            effective.setdefault(section, {})[field] = deepcopy(value)
        elif section in _COMPONENT_PRESET_SECTIONS:
            component, _, key = field.rpartition(".")
            if not component or not key:
                continue
            effective.setdefault(section, {}).setdefault(component, {})[key] = deepcopy(value)
    return effective


def infer_session_overrides(
    preset: dict[str, Any], saved_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Preserve saved differences when a legacy session lacks override metadata."""
    if not isinstance(saved_config, dict):
        return {}
    defaults = _structured_config_values(preset)
    return {
        path: deepcopy(value)
        for path, value in _structured_config_values(saved_config).items()
        if path not in defaults or value != defaults[path]
    }


def _known_config_impacts() -> tuple[dict[tuple[str, str | None, str], str], dict[str, str]]:
    from .daw_processors import default_daw_processors
    from .detectors import default_detectors
    from .models import LifecyclePhase
    from .processors import default_processors

    known: dict[tuple[str, str | None, str], str] = {}
    for spec in ANALYSIS_PARAMS:
        known[("analysis", None, spec.key)] = "presentation" if spec.presentation_only else "phase2"
    for spec in PRESENTATION_PARAMS:
        known[("presentation", None, spec.key)] = "presentation"
    for spec in REPAIR_PARAMS:
        known[("processors", "corrective", spec.key)] = "repair"

    detector_phase: dict[str, str] = {}
    for det in default_detectors():
        impact = "phase1" if getattr(det, "phase", LifecyclePhase.PHASE2) == LifecyclePhase.PHASE1 else "phase2"
        detector_phase[det.id] = impact
        for spec in det.config_params():
            known[("detectors", det.id, spec.key)] = "presentation" if spec.presentation_only else impact
    for proc in default_processors():
        for spec in proc.config_params():
            known[("processors", proc.id, spec.key)] = "presentation" if spec.presentation_only else "phase2"
    for dp in default_daw_processors():
        for spec in dp.config_params():
            known[("daw_processors", dp.id, spec.key)] = "presentation" if spec.presentation_only else "daw"
    return known, detector_phase


def classify_structured_config_change(
    old: dict[str, Any],
    new: dict[str, Any],
) -> ConfigChangeImpact:
    """Classify how changing one structured config preset affects analysis."""
    known, detector_phase = _known_config_impacts()

    def _impact_for(section: str, component: str | None, key: str) -> str:
        if (hit := known.get((section, component, key))) is not None:
            return hit
        if section == "presentation":
            return "presentation"
        if section == "detectors" and component:
            return detector_phase.get(component, "phase2")
        return "daw" if section == "daw_processors" else "phase2"

    changed: dict[str, set[str]] = {k: set() for k in ("phase1", "phase2", "presentation", "daw", "repair")}
    old_values = _structured_config_values(old)
    new_values = _structured_config_values(new)
    for path in old_values.keys() | new_values.keys():
        if old_values.get(path) != new_values.get(path):
            section, _, key = path.partition(".")
            component = None
            if section in _COMPONENT_PRESET_SECTIONS:
                component, _, key = key.rpartition(".")
            changed[_impact_for(section, component, key)].add(path)

    return ConfigChangeImpact(
        phase1_keys=frozenset(changed["phase1"]),
        phase2_keys=frozenset(changed["phase2"]),
        presentation_keys=frozenset(changed["presentation"]),
        daw_keys=frozenset(changed["daw"]),
        repair_keys=frozenset(changed["repair"]),
    )


def flatten_structured_config(structured: dict[str, Any]) -> dict[str, Any]:
    """Flatten a structured config into a flat key-value dict for the pipeline.

    Merges all sections into a single dict.  The pipeline, detectors, and
    processors read from this flat dict via ``config.get(key, default)``.
    """
    flat: dict[str, Any] = {}
    flat.update(structured.get("analysis", {}))
    for section in structured.get("detectors", {}).values():
        if isinstance(section, dict):
            flat.update(section)
    for section in structured.get("processors", {}).values():
        if isinstance(section, dict):
            flat.update(section)
    for section in structured.get("daw_processors", {}).values():
        if isinstance(section, dict):
            flat.update(section)
    return flat


def _validate_daw_templates(dp_id: str, section: dict[str, Any]) -> list[ConfigFieldError]:
    key = f"{dp_id}_templates"
    path = f"daw_processors.{dp_id}.{key}"
    templates = section.get(key, [])
    if not isinstance(templates, list):
        return [ConfigFieldError(path, templates, "Expected a list of templates.")]
    errors = []
    fields = ("name", "group", "ptxt_path", "path") if dp_id == "protools" else ("name", "template_path")
    for index, template in enumerate(templates):
        if not isinstance(template, dict) or any(not isinstance(template.get(f, ""), str) for f in fields):
            errors.append(ConfigFieldError(f"{path}[{index}]", template, "Invalid template row."))
            continue
        if dp_id == "dawproject":
            ceiling = template.get("fader_ceiling_db", 6.0)
            try:
                valid = not isinstance(ceiling, bool) and math.isfinite(float(ceiling))
            except (TypeError, ValueError, OverflowError):
                valid = False
            if not valid:
                errors.append(ConfigFieldError(f"{path}[{index}].fader_ceiling_db", ceiling,
                                               "Expected a finite fader ceiling in dB."))
    return errors


def _prefix_errors(prefix: str, errs: list[ConfigFieldError]) -> list[ConfigFieldError]:
    return [ConfigFieldError(f"{prefix}.{e.key}", e.value, e.message) for e in errs]


def validate_structured_config(
    structured: dict[str, Any],
) -> list[ConfigFieldError]:
    """Validate a structured config dict section by section."""
    from .detectors import default_detectors
    from .processors import default_processors
    from .daw_processors import default_daw_processors

    errors: list[ConfigFieldError] = validate_param_values(
        ANALYSIS_PARAMS, structured.get("analysis", {}),
    )
    det_map = {d.id: d for d in default_detectors()}
    for det_id, section in structured.get("detectors", {}).items():
        det = det_map.get(det_id)
        if det is not None and isinstance(section, dict):
            errors.extend(_prefix_errors(f"detectors.{det_id}", validate_param_values(det.config_params(), section)))

    proc_map = {p.id: p for p in default_processors()}
    for proc_id, section in structured.get("processors", {}).items():
        if not isinstance(section, dict):
            continue
        specs = REPAIR_PARAMS if proc_id == "corrective" else (proc_map[proc_id].config_params() if proc_id in proc_map else None)
        if specs is None:
            continue
        errors.extend(_prefix_errors(f"processors.{proc_id}", validate_param_values(specs, section)))
        if proc_id == "bimodal_normalize":
            errors.extend(_validate_normalization_invariants(section, prefix="processors.bimodal_normalize."))

    dp_map = {dp.id: dp for dp in default_daw_processors()}
    for dp_id, section in structured.get("daw_processors", {}).items():
        dp = dp_map.get(dp_id)
        if dp is None or not isinstance(section, dict):
            continue
        if dp_id in ("protools", "dawproject"):
            errors.extend(_validate_daw_templates(dp_id, section))
        errors.extend(_prefix_errors(f"daw_processors.{dp_id}", validate_param_values(dp.config_params(), section)))
    return errors


def _type_label(t) -> str:
    """Human-readable label for an expected type or tuple of types."""
    if isinstance(t, tuple):
        return " or ".join(x.__name__ for x in t)
    return t.__name__
