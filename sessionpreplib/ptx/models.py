"""Data models and binary constants for Pro Tools ``.ptx`` / ``.ptxt`` files."""

from __future__ import annotations

from dataclasses import dataclass, field

PTX_MAGIC = b"\x030010111100101011"
UUID_PREFIX = b"\x00\x00\x00\x00\x2a\x00\x00\x00"
MAX_GROUP_ID = 104  # 4 Pro Tools group banks * 26 letters (a..z, 2a..2z, 3a..3z, 4a..4z)

TRACK_TYPE_MAP: dict[int, str] = {
    0: "Audio",
    1: "Midi",
    2: "Aux",  # Becomes "RoutingFolder" when is_folder=True
    3: "Video",
    5: "Master",
    7: "Instrument",
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

SEND_SLOTS = "ABCDEFGHIJ"
SLOT_LETTERS_LOWER = "abcdefghij"


@dataclass
class PtxChunk:
    """Single top-level Z-chunk in a decrypted ``.ptx`` buffer."""

    offset: int
    version: int
    size: int
    tag: int
    raw: bytes

    @property
    def body(self) -> bytes:
        return self.raw[9:]


@dataclass
class PtxSessionInfo:
    """Session-level metadata parsed from header chunks."""

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


@dataclass
class PtxPluginInfo:
    """Plugin catalog entry from ``0x1018``."""

    index: int
    name: str
    manufacturer_4cc: str = ""
    product_4cc: str = ""
    type_4cc: str = ""
    bundle_id: str = ""


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


@dataclass
class PtxClip:
    """Clip definition from ``0x262a`` -> ``0x2629``."""

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


@dataclass
class PtxPlaylist:
    """Playlist entry from ``0x1015``."""

    index: int
    name: str
    playlist_id: str
    channel_count: int
    lane_indices: list[int] = field(default_factory=list)
    placements: list[PtxClipPlacement] = field(default_factory=list)


@dataclass
class PtxTrack:
    """Decoded track combining directory, folder tree, mixer, group, and clip data."""

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
