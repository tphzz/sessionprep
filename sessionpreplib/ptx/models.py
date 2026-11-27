"""Data models and binary constants for Pro Tools ``.ptx`` / ``.ptxt`` files (v10+)."""

from __future__ import annotations

import base64
import bisect
from dataclasses import dataclass, field
from typing import Protocol

PTX_MAGIC = b"\x030010111100101011"
UUID_PREFIX = b"\x00\x00\x00\x00\x2a\x00\x00\x00"
MAX_GROUP_ID = 104  # 4 Pro Tools group banks * 26 letters (a..z, 2a..2z, 3a..3z, 4a..4z)

ZERO_TICKS = 1_000_000_000_000  # 0xE8D4A51000 timeline tick origin
TICKS_PER_QUARTER = 960_000
TICK_MASK_56 = 0x00FF_FFFF_FFFF_FFFF

TRACK_TYPE_MAP: dict[int, str] = {
    0: "Audio",
    1: "Midi",
    2: "Aux",  # Becomes "RoutingFolder" when is_folder=True
    3: "Video",  # Disambiguated against Vca via mixer block on older v12 sessions
    5: "Master",
    7: "Instrument",
    8: "Video",  # Older Pro Tools v12 Video track code
    9: "Vca",
    11: "BasicFolder",
}

IO_KIND_MAP: dict[int, str] = {
    0: "input",
    1: "output",
    2: "bus",
}

GROUP_TYPE_MAP: dict[int, str] = {
    0: "Edit",
    1: "Mix",
    2: "Mix/Edit",
}

AUTOMATION_MODE_MAP: dict[int, str] = {
    0: "Off",
    1: "Read",
    2: "Touch",
    3: "Latch",
    4: "Write",
    5: "Touch/Latch",
}

MARKER_TIME_PROPS: dict[int, str] = {
    1: "Selection",
    2: "None",
    3: "Marker",
}

FADE_TYPE_MAP: dict[int, str] = {
    1: "crossfade",
    2: "out",
    3: "in",
}

FADE_SHAPE_MAP: dict[int, str] = {
    0: "standard",
    1: "s_curve",
    2: "preset",
}

CROSSFADE_LINK_MAP: dict[int, str] = {
    1: "equal_power",
    2: "equal_gain",
    3: "none",
}

MIDI_CONTROL_LANE_TYPE_MAP: dict[int, str] = {
    1: "pitch_bend",
    2: "cc",
    3: "program_change",
}

# Timecode frame-rate codes observed in chunk 0x204d:
# code -> (real_fps, nominal_fps, label, is_drop_frame)
FPS_CODE_MAP: dict[int, tuple[float, int, str, bool]] = {
    1: (24.0, 24, "24", False),
    2: (25.0, 25, "25", False),
    3: (30000.0 / 1001.0, 30, "29.97 non-drop", False),
    5: (30000.0 / 1001.0, 30, "29.97 drop", True),
    7: (30.0, 30, "30", False),
    8: (30.0, 30, "30 drop", True),
    9: (24000.0 / 1001.0, 24, "23.976", False),
}

# Pro Tools 12 track colour palette (69 swatches: 3 rows of 23).
PT12_COLOR_PALETTE: dict[int, tuple[int, int, int]] = {
    25: (44, 0, 252), 26: (86, 0, 252), 27: (136, 0, 252), 28: (191, 0, 252),
    29: (190, 0, 192), 30: (189, 0, 136), 31: (189, 0, 84), 32: (188, 0, 13),
    33: (189, 30, 13), 34: (189, 82, 14), 35: (190, 137, 17), 36: (192, 197, 20),
    37: (137, 197, 17), 38: (87, 198, 16), 39: (46, 198, 15), 40: (28, 198, 14),
    41: (30, 198, 84), 42: (32, 196, 136), 43: (35, 195, 193), 44: (39, 193, 253),
    45: (33, 132, 252), 46: (28, 74, 252), 47: (25, 0, 252),
    49: (30, 0, 163), 50: (55, 0, 163), 51: (85, 0, 163), 52: (116, 0, 164),
    53: (124, 0, 137), 54: (123, 0, 102), 55: (122, 0, 70), 56: (122, 0, 11),
    57: (122, 18, 11), 58: (122, 49, 12), 59: (123, 81, 13), 60: (137, 128, 16),
    61: (102, 128, 14), 62: (72, 128, 13), 63: (45, 128, 12), 64: (24, 128, 12),
    65: (21, 128, 51), 66: (22, 127, 81), 67: (26, 140, 126), 68: (29, 141, 164),
    69: (25, 105, 164), 70: (22, 70, 163), 71: (20, 35, 163),
    73: (20, 0, 95), 74: (33, 0, 95), 75: (49, 0, 95), 76: (65, 0, 95),
    77: (75, 0, 87), 78: (71, 0, 66), 79: (71, 0, 49), 80: (71, 0, 11),
    81: (71, 12, 11), 82: (71, 28, 11), 83: (71, 44, 12), 84: (87, 77, 15),
    85: (66, 74, 12), 86: (50, 74, 12), 87: (35, 74, 12), 88: (21, 75, 11),
    89: (15, 74, 29), 90: (15, 74, 44), 91: (20, 89, 76), 92: (22, 89, 95),
    93: (20, 71, 95), 94: (19, 53, 95), 95: (17, 34, 95),
}

SEND_SLOTS = "ABCDEFGHIJ"
SLOT_LETTERS_LOWER = "abcdefghij"


@dataclass
class PtxChunk:
    """Single Z-chunk node in a decrypted ``.ptx`` buffer."""

    offset: int
    version: int
    size: int
    tag: int
    raw: bytes = field(repr=False)
    children: list[PtxChunk] = field(default_factory=list, repr=False)

    @property
    def body(self) -> bytes:
        return self.raw[9:]


@dataclass
class PtxSessionInfo:
    """Session-level metadata parsed from header and timing chunks."""

    file_path: str | None = None
    xor_delta: int = 0
    product_name: str = ""
    protools_version: str = ""
    build_type: str = ""
    os_name: str = ""
    sample_rate: int = 0
    bit_depth: int = 0
    audio_format_code: int = 0
    tempo_bpm: float | None = None
    session_id_code: str = ""
    saved_session_path: str = ""
    source_template_path: str = ""
    file_size: int = 0
    session_root: str = ""
    fps_code: int | None = None
    frame_rate: float | None = None
    frame_rate_label: str = ""
    is_drop_frame: bool = False
    start_frames: int | None = None
    start_seconds: float | None = None
    start_timecode: str = ""


@dataclass
class PtxTempoEvent:
    """Single tempo map event from ``0x2028``."""

    tick: int
    bpm: float
    seconds: float = 0.0
    is_ramp: bool = False


@dataclass
class PtxMeterEvent:
    """Single meter / time-signature event from ``0x2029``."""

    tick: int
    numerator: int
    denominator: int
    click_value: int = 0


@dataclass
class PtxMarker:
    """Memory location / marker entry from ``0x2030`` -> ``0x2077``."""

    number: int
    name: str
    start_sample: int
    end_sample: int
    time_properties: str = "Marker"
    reference: str = "Absolute"
    comments: str = ""
    start_seconds: float | None = None
    end_seconds: float | None = None


@dataclass
class PtxAutomationPoint:
    """Single breakpoint in a decoded Pro Tools ``0x01 0x46`` control curve."""

    position: int  # Samples (timebase == "Samples") or ticks relative to ZERO_TICKS
    value: float


def _points_are_automated(points: list[PtxAutomationPoint]) -> bool:
    if len(points) <= 1:
        return False
    first = points[0].value
    return any(abs(pt.value - first) > 1e-9 for pt in points[1:])


def _eval_automation_points(
    points: list[PtxAutomationPoint], position: int, *, stepped: bool = False
) -> float | None:
    if not points:
        return None
    pts = sorted(points, key=lambda p: p.position)
    if position < pts[0].position:
        return pts[0].value
    if position >= pts[-1].position:
        return pts[-1].value
    positions = [p.position for p in pts]
    idx = bisect.bisect_right(positions, position) - 1
    a, b = pts[idx], pts[idx + 1]
    if stepped or position == a.position or b.position == a.position:
        return a.value
    frac = (position - a.position) / float(b.position - a.position)
    return a.value + frac * (b.value - a.value)


class _AutomationCurveMixin:
    """Shared evaluation helpers for sample- and tick-based automation lanes."""

    timebase: str
    points: list[PtxAutomationPoint]

    @property
    def is_automated(self) -> bool:
        """Return True when the lane contains more than one point with a non-constant value."""
        return _points_are_automated(self.points)

    @property
    def initial_value(self) -> float | None:
        if not self.points:
            return None
        return min(self.points, key=lambda p: p.position).value

    def value_at(self, position: int, *, stepped: bool = False) -> float | None:
        """Evaluate the curve at *position* (samples or ticks) via linear or stepped interpolation."""
        return _eval_automation_points(self.points, position, stepped=stepped)

    def value_at_seconds(
        self, seconds: float, timeline: PtxTimeConverter, *, stepped: bool = False
    ) -> float | None:
        """Evaluate the curve at *seconds*, converting via *timeline* according to ``timebase``."""
        pos = timeline.seconds_to_ticks(seconds) if self.timebase == "Ticks" else timeline.seconds_to_samples(seconds)
        return self.value_at(pos, stepped=stepped)

    def points_in_seconds(self, timeline: PtxTimeConverter) -> list[tuple[float, float]]:
        """Return ``(seconds, value)`` pairs for all breakpoints in chronological order."""
        pts = sorted(self.points, key=lambda p: p.position)
        if self.timebase == "Ticks":
            return [(timeline.ticks_to_seconds(p.position), p.value) for p in pts]
        return [(timeline.samples_to_seconds(p.position), p.value) for p in pts]


@dataclass
class PtxAutomationLane(_AutomationCurveMixin):
    """Multi-point automation curve for a track, send, VCA, or plugin parameter."""

    parameter: str
    timebase: str = "Samples"
    points: list[PtxAutomationPoint] = field(default_factory=list)


@dataclass
class PtxSignalPath:
    """Signal path entry from ``0x2603`` (routable) or ``0x1022`` (I/O setup)."""

    index: int
    name: str
    kind: str
    is_stereo: bool
    channel_count: int
    uuid_hex: str
    uuid_braced: str
    source_chunk: str
    bus_channels: list[int] = field(default_factory=list)
    is_active: bool = True
    is_subpath: bool = False
    parent_io_index: int | None = None
    parent_name: str = ""
    parent_uuid_hex: str = ""
    subpaths: list[str] = field(default_factory=list)
    mapped_to_output: bool = False
    output_mappings: list[str] = field(default_factory=list)


@dataclass
class PtxSignalPathRef:
    """Reference to a resolved signal path (bus, input, or hardware output)."""

    name: str
    kind: str
    uuid_hex: str
    ptsl_index: int | None = None


@dataclass
class PtxSend:
    """Per-track send assignment (slots A..J)."""

    slot: str
    slot_index: int
    destination: PtxSignalPathRef
    volume_db: float | None = None
    is_muted: bool = False
    is_pre_fader: bool = False
    lfe_db: float | None = None
    pan_left: int | None = None
    pan_right: int | None = None
    is_active: bool = True
    volume_automation: PtxAutomationLane | None = None
    mute_automation: PtxAutomationLane | None = None
    lfe_or_trim_automation: PtxAutomationLane | None = None
    pan_left_automation: PtxAutomationLane | None = None
    pan_right_automation: PtxAutomationLane | None = None


@dataclass
class PtxPluginInfo:
    """Plugin catalog entry from ``0x1018``."""

    index: int
    name: str
    manufacturer_4cc: str = ""
    product_4cc: str = ""
    type_4cc: str = ""
    bundle_id: str = ""


@dataclass
class PtxAraPlugin:
    """Registered ARA2 plugin descriptor and optional document state from ``0x4823`` / ``0x4807``."""

    index: int
    name: str
    manufacturer_4cc: str = ""
    product_4cc: str = ""
    type_4cc: str = ""
    has_state: bool = False
    state_size: int = 0
    state_label: str = ""
    state_b64: str = ""

    @property
    def raw_state_bytes(self) -> bytes:
        return base64.b64decode(self.state_b64) if self.state_b64 else b""


@dataclass
class PtxAraBinding:
    """Decoded ARA2 document / audio-source / clip-modification binding from ``0x4820`` -> ``0x480c``."""

    chunk_id: str = ""
    plugin_name: str = ""
    manufacturer_4cc: str = ""
    product_4cc: str = ""
    type_4cc: str = ""
    audio_source_ids: list[str] = field(default_factory=list)
    audio_modification_uuids: list[str] = field(default_factory=list)
    audio_modification_ids: list[str] = field(default_factory=list)
    document_state_ids: list[str] = field(default_factory=list)


@dataclass
class PtxPluginState:
    """Decoded AAX plugin preset/state payload from insert chunk ``0x1038``."""

    manufacturer_4cc: str = ""
    product_4cc: str = ""
    plugin_type_4cc: str = ""
    state_type_4cc: str = ""
    state_label: str = ""
    payload_size: int = 0
    payload_b64: str = ""
    controls: dict[str, int | float] = field(default_factory=dict)

    @property
    def raw_bytes(self) -> bytes:
        return base64.b64decode(self.payload_b64) if self.payload_b64 else b""


@dataclass
class PtxPluginAutomationLane(_AutomationCurveMixin):
    """Plugin parameter automation lane from ``0x2614`` -> ``0x260f`` -> ``0x260b``."""

    parameter_tag: str
    parameter_name: str
    timebase: str = "Samples"
    points: list[PtxAutomationPoint] = field(default_factory=list)


@dataclass(frozen=True)
class SlotATrimInfo:
    """Validated Insert Slot A gain-trim plugin descriptor and current gain state."""

    plugin_id: str
    plugin_name: str
    fourcc: str
    bundle_id: str
    gain_db: float
    min_db: float
    max_db: float
    is_active: bool = True
    is_bypassed: bool = False
    is_clean_donor: bool = False


@dataclass
class PtxInsert:
    """Per-track insert slot assignment (slots A..J)."""

    slot: str
    slot_index: int
    plugin_index: int | None = None
    plugin_name: str = ""
    bundle_id: str = ""
    fourcc: str = ""
    is_active: bool = True
    is_bypassed: bool = False
    sidechain: PtxSignalPathRef | None = None
    state: PtxPluginState | None = None
    automation: list[PtxPluginAutomationLane] = field(default_factory=list)

    def control_value(self, tag: str) -> int | float | None:
        """Return a raw ``Complete Controls State`` parameter value by 4-6 char *tag*."""
        return self.state.controls.get(tag) if self.state is not None else None

    def automation_by_tag(self, tag: str) -> PtxPluginAutomationLane | None:
        return next((lane for lane in self.automation if lane.parameter_tag == tag), None)

    def automation_by_name(self, name: str) -> PtxPluginAutomationLane | None:
        low = name.lower()
        return next((lane for lane in self.automation if lane.parameter_name.lower() == low), None)


@dataclass
class PtxGroupAttributes:
    """Group attribute flags from a 62-byte ``0x258e`` block."""

    main_volume: bool = False
    main_mute: bool = False
    main_pan: bool = False
    main_lfe: bool = False
    follow_globals: bool = False
    record_enable: bool = False
    input_monitoring: bool = False
    solos: bool = False
    automation_mode: bool = False
    include_rulers: bool = False
    insert_controls: list[str] = field(default_factory=list)
    insert_bypass: list[str] = field(default_factory=list)
    send_level: list[str] = field(default_factory=list)
    send_mute: list[str] = field(default_factory=list)
    send_pan: list[str] = field(default_factory=list)
    send_lfe: list[str] = field(default_factory=list)
    extra_mix_flags: dict[int, int] = field(default_factory=dict)


@dataclass
class PtxGroup:
    """Edit / Mix / Mix-Edit group entry from ``0x202b``."""

    index: int
    name: str
    kind: str
    group_type: str
    id_number: int
    id_letter: str
    is_edit_active: bool = False
    is_mix_active: bool = False
    color_index: int = 0xFFFE
    vca_track_index: int | None = None
    vca_track_name: str | None = None
    member_indices: list[int] = field(default_factory=list)
    member_names: list[str] = field(default_factory=list)
    follow_globals: bool = False
    include_rulers: bool = False
    attributes: PtxGroupAttributes | None = None
    effective_attributes: PtxGroupAttributes | None = None


@dataclass
class PtxAudioFile:
    """Audio file descriptor from ``0x1004``."""

    index: int
    file_number: int
    filename: str
    directory: str = ""
    sample_rate: int = 0
    channels: int = 0
    bit_depth: int = 0
    length_samples: int = 0
    duration_seconds: float | None = None
    format_code: int = 0
    path: str = ""
    relative_path: str | None = None
    file_type_4cc: str = ""


@dataclass
class PtxClipGainBreakpoint:
    """Single clip-gain breakpoint from top-level chunk ``0x2637``."""

    sample: int
    gain_db: float
    seconds: float | None = None


@dataclass
class PtxClipGainCurve:
    """Clip-gain curve entry from top-level chunk ``0x2637``."""

    index: int
    flags: int = 1
    timebase: str = "Samples"
    breakpoints: list[PtxClipGainBreakpoint] = field(default_factory=list)

    @property
    def is_dynamic(self) -> bool:
        """Return True when the clip-gain curve varies across multiple breakpoints."""
        if len(self.breakpoints) <= 1:
            return False
        first = self.breakpoints[0].gain_db
        return any(abs(bp.gain_db - first) > 1e-6 for bp in self.breakpoints[1:])

    @property
    def initial_gain_db(self) -> float:
        if not self.breakpoints:
            return 0.0
        return min(self.breakpoints, key=lambda b: b.sample).gain_db

    def gain_db_at(self, sample: int) -> float:
        """Evaluate clip gain (dB) at *sample* via right-continuous linear interpolation."""
        if not self.breakpoints:
            return 0.0
        bps = sorted(self.breakpoints, key=lambda b: b.sample)
        if sample < bps[0].sample:
            return bps[0].gain_db
        if sample >= bps[-1].sample:
            return bps[-1].gain_db
        positions = [b.sample for b in bps]
        idx = bisect.bisect_right(positions, sample) - 1
        a, b = bps[idx], bps[idx + 1]
        if sample == a.sample or b.sample == a.sample:
            return a.gain_db
        frac = (sample - a.sample) / float(b.sample - a.sample)
        return a.gain_db + frac * (b.gain_db - a.gain_db)


@dataclass
class PtxWarpMarker:
    """Single Elastic Audio warp marker mapping source sample to timeline sample (``0x2636``)."""

    source_sample: int
    timeline_sample: int


@dataclass
class PtxFade:
    """Fade-in, fade-out, or crossfade definition from ``0x2630`` -> ``0x262f``."""

    index: int
    fade_type: str  # "in", "out", "crossfade", "unknown"
    pre_samples: int = 0
    total_samples: int = 0
    shape_code: int | None = None
    shape_name: str = ""
    curve_amount: int | None = None
    crossfade_link_code: int | None = None
    crossfade_link_name: str = ""
    in_shape_code: int | None = None
    in_curve_amount: int | None = None
    out_shape_code: int | None = None
    out_curve_amount: int | None = None


@dataclass
class PtxFadePlacement:
    """Placed fade event on a track playlist lane from ``0x1054`` -> ``0x1050`` -> ``0x104f``."""

    channel: int
    fade_index: int
    start_sample: int
    fade_type: str = "unknown"
    pre_samples: int = 0
    total_samples: int = 0
    flags: int = 0


@dataclass
class PtxCompoundClipMember:
    """Member clip entry inside a compound clip / clip group (``0x2428``)."""

    clip_index: int
    start_sample: int
    flags: int = 0
    is_muted: bool = False
    is_covered: bool = False


@dataclass
class PtxCompoundClip:
    """Compound clip / clip group definition from ``0x262c`` & ``0x2428``."""

    index: int
    name: str
    offset_samples: int = 0
    length_samples: int = 0
    members: list[PtxCompoundClipMember] = field(default_factory=list)


@dataclass
class PtxClip:
    """Clip definition from ``0x262a`` -> ``0x2629`` -> ``0x2628``."""

    index: int
    name: str
    clip_id: str
    audio_file_index: int | None = None
    audio_file_name: str = ""
    channel_index: int = 0
    start_sample: int = 0
    length_samples: int = 0
    gain_linear: float = 1.0
    gain_db: float = 0.0
    clip_gain_curve_index: int = -1
    sync_point_samples: int = 0
    original_timestamp_samples: int = 0
    user_timestamp_samples: int = 0
    has_clip_gain_automation: bool = False
    clip_gain_breakpoints: list[PtxClipGainBreakpoint] = field(default_factory=list)
    warp_markers: list[PtxWarpMarker] = field(default_factory=list)
    timeline_length_samples: int = 0
    playback_rate: float = 1.0
    ara_plugin_name: str = ""
    ara_modification_uuid: str = ""
    clip_gain_chunk_size: int = 0
    clip_gain_raw_b64: str = ""


@dataclass
class PtxClipPlacement:
    """Clip placement on a track playlist lane from ``0x1054`` -> ``0x104f``."""

    channel: int
    clip_index: int
    clip_name: str
    audio_file_name: str
    start_sample: int
    length_samples: int
    end_sample: int
    color_index: int = 0xFFFE
    start_seconds: float | None = None
    duration_seconds: float | None = None
    flags: int = 0
    is_muted: bool = False
    is_covered: bool = False
    is_compound: bool = False
    loop_count: int | None = None
    has_clip_gain_automation: bool = False
    clip_gain_breakpoints: list[PtxClipGainBreakpoint] = field(default_factory=list)
    ara_plugin_name: str = ""
    ara_modification_uuid: str = ""


@dataclass
class PtxPlaylist:
    """Playlist entry from ``0x1015``."""

    index: int
    name: str
    playlist_id: str
    channel_count: int
    lane_indices: list[int] = field(default_factory=list)
    placements: list[PtxClipPlacement] = field(default_factory=list)
    fade_placements: list[PtxFadePlacement] = field(default_factory=list)
    is_alternate: bool = False
    track_name: str = ""


@dataclass
class PtxMidiNote:
    """Single MIDI note event from ``0x2000`` ``MdNLB``."""

    tick: int  # Ticks relative to ZERO_TICKS
    pitch: int
    duration_ticks: int
    velocity: int


@dataclass
class PtxMidiControlEvent:
    """Single MIDI controller / pitch-bend / program-change event from ``0x2000`` ``MdCDS``."""

    tick: int  # Ticks relative to ZERO_TICKS
    value: int


@dataclass
class PtxMidiControlLane:
    """MIDI controller lane from ``0x2000`` ``MdCDS``."""

    lane_type_code: int
    lane_type: str  # "pitch_bend", "cc", "program_change", or "type_N"
    controller_number: int
    events: list[PtxMidiControlEvent] = field(default_factory=list)


@dataclass
class PtxMidiChunk:
    """Decoded ``MdChun`` block from ``0x2000``."""

    index: int
    notes: list[PtxMidiNote] = field(default_factory=list)
    control_lanes: list[PtxMidiControlLane] = field(default_factory=list)


@dataclass
class PtxMidiClip:
    """MIDI clip definition from ``0x2634`` -> ``0x2633``."""

    index: int
    name: str
    start_tick: int  # Ticks relative to ZERO_TICKS
    length_ticks: int
    chunk_index: int


@dataclass
class PtxMidiPlacement:
    """MIDI clip placement on a track from ``0x1058`` -> ``0x1057`` -> ``0x104f``."""

    clip_index: int
    clip_name: str
    start_tick: int  # Ticks relative to ZERO_TICKS
    length_ticks: int = 0
    flags: int = 0
    is_muted: bool = False
    start_seconds: float | None = None
    duration_seconds: float | None = None


@dataclass
class PtxVideoClip:
    """Video clip definition from ``0x262e`` -> ``0x262d``."""

    index: int
    name: str
    offset_frames: int = 0
    length_frames: int = 0


@dataclass
class PtxVideoPlacement:
    """Video clip placement on a video track from ``0x1055`` -> ``0x1053`` -> ``0x104f``."""

    clip_index: int
    clip_name: str
    start_sample: int
    start_seconds: float | None = None


@dataclass
class PtxTrack:
    """Decoded track combining directory, folder tree, mixer, automation, group, and clip data."""

    index: int
    name: str
    uuid_hex: str
    uuid_braced: str
    track_type: str
    format: str
    is_folder: bool = False
    is_elastic_or_ara: bool = False
    color_index: int = 0xFFFE
    is_hidden: bool = False
    is_inactive: bool = False
    timebase: str = "Samples"
    parent_folder_name: str = ""
    parent_folder_uuid: str = ""
    child_track_names: list[str] = field(default_factory=list)
    input_path: PtxSignalPathRef | None = None
    main_output: PtxSignalPathRef | None = None
    volume_db: float | None = None
    is_muted: bool = False
    is_soloed: bool = False
    is_record_enabled: bool = False
    is_record_safe: bool = False
    is_input_monitor: bool = False
    automation_mode: str = "Read"
    automation_mode_code: int = 1
    lfe_db: float | None = None
    pan_left: int | None = None
    pan_right: int | None = None
    sends: list[PtxSend] = field(default_factory=list)
    inserts: list[PtxInsert] = field(default_factory=list)
    slot_a_trim: SlotATrimInfo | None = None
    groups: list[str] = field(default_factory=list)
    vca_group_name: str | None = None
    vca_controlled_tracks: list[str] = field(default_factory=list)
    clips: list[PtxClipPlacement] = field(default_factory=list)
    is_solo_safe: bool = False
    comment: str = ""
    color_rgb: tuple[int, int, int] | None = None
    is_output_active: bool = True
    input_path_raw_index: int | None = None
    ara_plugin: PtxAraPlugin | None = None
    volume_automation: PtxAutomationLane | None = None
    mute_automation: PtxAutomationLane | None = None
    lfe_or_trim_automation: PtxAutomationLane | None = None
    pan_left_automation: PtxAutomationLane | None = None
    pan_right_automation: PtxAutomationLane | None = None
    fade_placements: list[PtxFadePlacement] = field(default_factory=list)
    alternate_playlists: list[PtxPlaylist] = field(default_factory=list)
    midi_clips: list[PtxMidiPlacement] = field(default_factory=list)
    video_clips: list[PtxVideoPlacement] = field(default_factory=list)

    def insert_by_slot(self, slot: str) -> PtxInsert | None:
        """Return the insert assigned to *slot* (``'A'..'J'``, case-insensitive), or ``None``."""
        target = slot.strip().upper()
        return next((ins for ins in self.inserts if ins.slot.upper() == target), None)

    def send_by_slot(self, slot: str) -> PtxSend | None:
        """Return the send assigned to *slot* (``'A'..'J'``, case-insensitive), or ``None``."""
        target = slot.strip().upper()
        return next((snd for snd in self.sends if snd.slot.upper() == target), None)


class PtxTimeConverter(Protocol):
    """Structural protocol for time-domain conversion (implemented by ``PtxTimeline``)."""

    def samples_to_seconds(self, samples: int) -> float: ...
    def seconds_to_samples(self, seconds: float) -> int: ...
    def ticks_to_seconds(self, tick: int) -> float: ...
    def seconds_to_ticks(self, seconds: float) -> int: ...


class PtxSessionLike(Protocol):
    """Read-only structural protocol satisfied by ``PtxSession`` (keeps module DAG strictly acyclic)."""

    info: PtxSessionInfo
    warnings: list[str]
    tempo_events: list[PtxTempoEvent]
    has_unparsed_tempo_ramps: bool
    meter_events: list[PtxMeterEvent]
    markers: list[PtxMarker]
    tracks: list[PtxTrack]
    groups: list[PtxGroup]
    audio_files: list[PtxAudioFile]
    clip_gain_curves: list[PtxClipGainCurve]
    clips: list[PtxClip]
    fades: list[PtxFade]
    compound_clips: list[PtxCompoundClip]
    playlists: list[PtxPlaylist]
    midi_chunks: list[PtxMidiChunk]
    midi_clips: list[PtxMidiClip]
    midi_playlists: dict[str, list[PtxMidiPlacement]]
    video_clips: list[PtxVideoClip]
    video_playlists: dict[str, list[PtxVideoPlacement]]

    @property
    def timeline(self) -> PtxTimeConverter: ...
    def track_by_name(self, name: str) -> PtxTrack | None: ...
    def group_by_name(self, name: str, kind: str = "edit") -> PtxGroup | None: ...
