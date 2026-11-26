import os
import sys
import argparse
from io import TextIOWrapper
from pathlib import Path

from sessionpreplib.logging_setup import setup_logging

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
    from rich import box
except ImportError:
    print("Error: The 'rich' library is required for the CLI but not installed.", file=sys.stderr)
    print("Please install it with: pip install sessionprep[cli] (or uv sync --extra cli)", file=sys.stderr)
    sys.exit(1)

from sessionpreplib import __version__
from sessionpreplib.pipeline import Pipeline, load_session
from sessionpreplib.detectors import default_detectors
from sessionpreplib.processors import default_processors
from sessionpreplib.config import default_config, merge_configs, validate_config_fields
from sessionpreplib.rendering import build_diagnostic_summary, render_diagnostic_summary_text
from sessionpreplib.reports import build_warnings, format_boundary_settling, generate_report, save_json
from sessionpreplib.utils import protools_sort_key
from sessionpreplib.audio import AUDIO_EXTENSIONS, discover_audio_files
from sessionpreplib.events import EventBus
from sessionpreplib.file_io import BACKUP_RECEIPT_SUFFIX

console = Console()


def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return ivalue


def _add_detection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--clip_consecutive", type=int, default=3,
                        help="Minimum consecutive near-full-scale samples to flag as clipped")
    parser.add_argument("--clip_report_max_ranges", type=positive_int, default=10,
                        help="Max clipped sample ranges to include in reports per file")
    parser.add_argument("--dc_offset_warn_db", type=float, default=-40.0,
                        help="Warn if DC offset exceeds this level (dBFS)")
    parser.add_argument("--corr_warn", type=float, default=-0.3,
                        help="Warn if stereo L/R correlation is below this value")
    parser.add_argument("--dual_mono_eps", type=float, default=1e-5,
                        help="Warn if stereo file appears dual-mono (max |L-R| <= eps)")
    parser.add_argument("--mono_loss_warn_db", type=float, default=6.0,
                        help="Warn if stereo file loses more than this many dB when folded to mono")
    parser.add_argument("--one_sided_silence_db", type=float, default=-80.0,
                        help="Warn if a stereo file has one channel at or below this RMS level (dBFS) while the other is not")
    parser.add_argument("--subsonic_hz", type=float, default=30.0,
                        help="Subsonic detector cutoff frequency (Hz)")
    parser.add_argument("--subsonic_warn_ratio_db", type=float, default=-20.0,
                        help="Warn if subsonic power ratio (<= cutoff) exceeds this level (dB relative to full-band power)")
    parser.add_argument("--fix", action="append", choices=["dc", "subsonic", "ultrasonic"], default=[],
                        help="Select this repair on matching tracks. May be repeated.")
    parser.add_argument("--low-cut-hz", type=float, default=10.0, metavar="HZ",
                        help="High-pass corner when subsonic filtering is selected (5–40 Hz)")


def _add_analysis_and_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--window", type=positive_int, default=400, help="RMS analysis window (ms)")
    parser.add_argument("--stereo_mode", type=str, choices=["avg", "sum"], default="avg",
                        help="Stereo RMS calculation mode")
    parser.add_argument("--rms_anchor", type=str, choices=["percentile", "max"], default="percentile",
                        help="Momentary RMS anchor strategy used for gain calculations")
    parser.add_argument("--rms_percentile", type=float, default=95.0,
                        help="Percentile used when --rms_anchor percentile (0-100)")
    parser.add_argument("--gate_relative_db", type=float, default=40.0,
                        help="Relative gate for momentary RMS windows. Ignore windows more than this many dB below the loudest RMS window (e.g. 40 means keep windows within 40 dB of the loudest).")
    parser.add_argument("--tail_max_regions", type=positive_int, default=20,
                        help="Max number of upper-tail regions to include in the report per file")
    parser.add_argument("--tail_min_exceed_db", type=float, default=3.0,
                        help="Only report tail regions exceeding the anchor by at least this many dB")
    parser.add_argument("--tail_hop_ms", type=positive_int, default=10,
                        help="Hop size for tail region reporting (ms). Larger values reduce the number of reported regions.")
    parser.add_argument("--anchor", type=str, default=None, help="Anchor track filename (fader stays at 0dB)")
    parser.add_argument("--fader_headroom_db", type=float, default=8.0,
                        help="Minimum dB of headroom at top of DAW fader range (default: 8.0, 0=disable)")
    parser.add_argument("--force_peak", type=str, default="",
                        help="Case-insensitive regex pattern to force Peak-target mode (e.g. 'kick|snare|tom')")
    parser.add_argument("--force_transient", nargs='+', default=[], action='extend',
                        help=argparse.SUPPRESS)
    parser.add_argument("--force_sustained", nargs='+', default=[], action='extend',
                        help=argparse.SUPPRESS)
    parser.add_argument("--group", action="append", default=[],
                        help="Named gain-linked group. Syntax: Name:pattern1,pattern2  (e.g. --group Kick:kick,kick_sub). Patterns support substring, glob (*/?), or exact match (suffix $). First match wins; overlaps produce a warning.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite source WAV files in-place (creates backups)")
    parser.add_argument("-x", "--execute", action="store_true",
                        help="Execute processing (write processed WAVs and reports). Without -x this runs in analysis-only mode.")
    parser.add_argument("--output_folder", type=str, default="processed", help="Subfolder name for processed files")
    parser.add_argument("--backup", type=str, default="_originals",
                        help="Backup folder name (Only used if overwriting files)")
    parser.add_argument("--report", type=str, default="sessionprep.txt", help="Output report filename")
    parser.add_argument("--json", type=str, default="sessionprep.json", help="Output JSON filename for automation")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="SessionPrep",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"sessionprep {__version__}")
    parser.add_argument("directory", type=str, help="Source directory containing audio tracks (.wav, .aif, .aiff)")
    parser.add_argument("--target_rms", type=float, default=-18.0, help="Target RMS level (dBFS)")
    parser.add_argument("--target_peak", type=float, default=-6.0, help="Peak ceiling / Peak-target level (dBFS)")
    parser.add_argument("--max_gain_db", type=float, default=24.0,
                        help="Maximum automatic boost cap in dB (0.0 to 36.0)")
    parser.add_argument("--crest_threshold", type=float, default=None, help=argparse.SUPPRESS)
    _add_detection_args(parser)
    _add_analysis_and_output_args(parser)

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    if not 0.0 < args.rms_percentile < 100.0:
        parser.error("--rms_percentile must be between 0 and 100 (exclusive)")
    if args.gate_relative_db < 0.0:
        parser.error("--gate_relative_db must be >= 0")
    if args.subsonic_hz <= 0.0:
        parser.error("--subsonic_hz must be > 0")
    if not 5 <= args.low_cut_hz <= 40:
        parser.error("--low-cut-hz must be between 5 and 40 Hz")
    if args.one_sided_silence_db > 0.0:
        parser.error("--one_sided_silence_db must be <= 0")
    return args


# ---------------------------------------------------------------------------
# Rich console rendering (CLI-only, not in the library)
# ---------------------------------------------------------------------------

def print_diagnostic_summary(summary):
    problems = summary.get("problems") or []
    attention = summary.get("attention") or []
    information = summary.get("information") or []
    clean = summary.get("clean") or []

    def item_count(groups):
        return sum(len(g.get("items") or []) for g in groups)

    def print_groups(groups, color, compact=False):
        any_printed = False
        for g in groups:
            title = g.get("title")
            hint = g.get("hint")
            items = g.get("items") or []
            if not items and not g.get("standalone"):
                continue
            header = f"{title}"
            if hint:
                header = f"{header} -> {hint}"
            console.print(f"  [{color}]-[/] {header}")
            for item in items:
                console.print(f"    [dim]* {item}[/]")
            if not compact:
                console.print("")
            any_printed = True
        return any_printed

    console.print("")
    console.print(f"[bold red]\U0001f534 PROBLEMS ({item_count(problems)})[/]")
    if not print_groups(problems, "red"):
        console.print("  [green]-[/] None")

    console.print("")
    console.print(f"[bold yellow]\U0001f7e1 ATTENTION ({item_count(attention)})[/]")
    if not print_groups(attention, "yellow"):
        console.print("  [green]-[/] None")

    console.print("")
    console.print(f"[bold blue]\U0001f535 INFORMATION ({item_count(information)})[/]")
    if not print_groups(information, "blue"):
        console.print("  [green]-[/] None")

    console.print("")
    console.print("[bold green]\U0001f7e2 CLEAN[/]")
    if not print_groups(clean, "green", compact=True):
        console.print("  [green]-[/] None")





# ---------------------------------------------------------------------------
# Main process_files() — thin wrapper around the sessionpreplib pipeline
# ---------------------------------------------------------------------------

def _prepare_cli_destinations(source, output, backup, filenames, report, json_report):
    """Check every requested write against the input audio, rendered audio and backups."""
    source, output = Path(source).resolve(), Path(output).resolve()
    backup = Path(backup).resolve() if backup is not None else None
    if backup is None and output == source:
        raise ValueError("Choose a different output folder or use --overwrite for in-place processing.")
    if backup == source:
        raise ValueError("The backup folder must be separate from the input folder.")

    # Existing hard links share a file ID even when their resolved names differ.
    claimed = {}

    def reserve(path, role):
        if os.name == "nt" and os.path.isreserved(path):
            raise ValueError(f"The {role} path contains a reserved Windows name. {path}")
        resolved = path.resolve()
        keys = [resolved]
        try:
            info = resolved.stat()
        except FileNotFoundError:
            pass
        else:
            if info.st_ino:
                keys.append((info.st_dev, info.st_ino))
        for key in keys:
            other = claimed.get(key)
            if other and not (other == role == "input audio"):
                raise ValueError(f"The {role} path conflicts with {other}. {path}")
            claimed[key] = role

    for name in filenames:
        reserve(source / name, "input audio")
    for name in filenames:
        if backup is None:
            reserve(output / name, "rendered audio")
        else:
            path = backup / name
            if path.exists() and not path.is_file():
                raise ValueError(f"The backup must name a file. {path}")
            reserve(path, "backup audio")
            reserve(path.with_name(path.name + BACKUP_RECEIPT_SUFFIX), "backup receipt")
    reports = [output / report, output / json_report]
    for path, role in zip(reports, ("text report", "JSON report")):
        if any(p.name.casefold().endswith(BACKUP_RECEIPT_SUFFIX) for p in (path, path.resolve())):
            raise ValueError(f"The {role} would replace a backup receipt. {path}")
        if path.exists() and not path.is_file():
            raise ValueError(f"The {role} must name a file. {path}")
        reserve(path, role)
        if path.is_file() and path.resolve().suffix.lower() in AUDIO_EXTENSIONS:
            raise ValueError(f"The {role} would replace an existing audio file. {path}")

    directories = [output, *(path.parent for path in reports)]
    if backup is not None:
        directories.append(backup)
    for directory in directories:
        resolved = directory.resolve()
        if any(parent in claimed for parent in (resolved, *resolved.parents)):
            raise ValueError(f"A file destination is also needed as a folder. {directory}")
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def _resolve_cli_force_peak(args) -> str:
    """Resolve --force_peak and emit deprecation warnings for legacy flags."""
    from sessionpreplib.audio import legacy_patterns_to_regex

    if getattr(args, "crest_threshold", None) is not None:
        print("Warning: --crest_threshold is deprecated and ignored under Peak-Capped RMS.", file=sys.stderr)
    if getattr(args, "force_sustained", None):
        print("Warning: --force_sustained is deprecated and ignored (Auto uses Peak-Capped RMS).", file=sys.stderr)
    force_peak = str(getattr(args, "force_peak", "") or "")
    legacy_trans = list(getattr(args, "force_transient", None) or [])
    if legacy_trans:
        print("Warning: --force_transient is deprecated; use --force_peak <regex> instead.", file=sys.stderr)
        if not force_peak:
            force_peak = legacy_patterns_to_regex(legacy_trans)
    return force_peak


def _build_cli_config(args, source_dir: str) -> dict:
    cli_overrides = {
        "target_rms": args.target_rms,
        "target_peak": args.target_peak,
        "max_gain_db": args.max_gain_db,
        "force_peak": _resolve_cli_force_peak(args),
        "clip_consecutive": args.clip_consecutive,
        "clip_report_max_ranges": args.clip_report_max_ranges,
        "dc_offset_warn_db": args.dc_offset_warn_db,
        "repair_auto_dc": "dc" in args.fix,
        "repair_auto_low_cut": "subsonic" in args.fix,
        "repair_auto_ultrasonic": "ultrasonic" in args.fix,
        "repair_low_cut_hz": args.low_cut_hz,
        "corr_warn": args.corr_warn,
        "dual_mono_eps": args.dual_mono_eps,
        "mono_loss_warn_db": args.mono_loss_warn_db,
        "one_sided_silence_db": args.one_sided_silence_db,
        "subsonic_hz": args.subsonic_hz,
        "subsonic_warn_ratio_db": args.subsonic_warn_ratio_db,
        "window": args.window,
        "stereo_mode": args.stereo_mode,
        "rms_anchor": args.rms_anchor,
        "rms_percentile": args.rms_percentile,
        "gate_relative_db": args.gate_relative_db,
        "tail_max_regions": args.tail_max_regions,
        "tail_min_exceed_db": args.tail_min_exceed_db,
        "tail_hop_ms": args.tail_hop_ms,
        "group": args.group,
        "anchor": args.anchor,
        "fader_headroom_db": args.fader_headroom_db,
        "execute": args.execute,
        "overwrite": args.overwrite,
        "output_folder": args.output_folder,
        "backup": args.backup,
        "report": args.report,
        "json": args.json,
        "_source_dir": source_dir,
    }
    return merge_configs(default_config(), cli_overrides)


def _print_header_and_groups(args, is_overwriting: bool, session=None) -> None:
    if session is None:
        rms_anchor_label = f"{args.rms_anchor}" + (
            f" P{args.rms_percentile:g}" if args.rms_anchor == "percentile" else ""
        )
        mode_label = "EXECUTE" if args.execute else "DRY-RUN"
        output_label = "(overwrite)" if is_overwriting else f"{args.output_folder}/"
        console.print(Panel.fit(
            f"[bold]SessionPrep[/]\n"
            f"Mode: [cyan]{mode_label}[/]\n"
            f"Target: [cyan]{args.target_rms} dB RMS[/] | [cyan]{args.target_peak} dB Peak[/]\n"
            f"RMS Anchor: [cyan]{rms_anchor_label}[/] | Window: [cyan]{args.window} ms[/]\n"
            f"Output: [green]{output_label}[/]",
            title="Configuration",
        ))
        return

    if not session.groups:
        return
    grouped: dict[str, list] = {}
    for t in session.tracks:
        if t.group is not None:
            grouped.setdefault(t.group, []).append(t)
    console.print("")
    group_table = Table(box=box.ROUNDED, title="Gain-Linked Groups", title_justify="left")
    group_table.add_column("Group", style="bold cyan")
    group_table.add_column("Gain", justify="right", style="bold green")
    group_table.add_column("Members", style="dim")
    from sessionpreplib.audio import INACTIVE_NORMALIZATION_CLASSIFICATIONS

    for gname in sorted(grouped):
        members = grouped[gname]
        active_pr = next(
            (
                p for m in members
                if (p := m.processor_results.get("bimodal_normalize")) is not None
                and not p.error
                and p.classification not in INACTIVE_NORMALIZATION_CLASSIFICATIONS
                and "bimodal_normalize" not in m.processor_skip
            ),
            None,
        )
        gain_str = f"{active_pr.gain_db:+.1f} dB" if active_pr is not None else "—"
        group_table.add_row(gname, gain_str, ", ".join(m.filename for m in members))
    console.print(group_table)
    console.print("")


def _load_and_analyze_cli_session(
    args, source_dir: str, output_dir: str, backup_dir: str | None, config: dict, pipeline: Pipeline, event_bus: EventBus,
):
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TextColumn("{task.completed}/{task.total}"), console=console,
    ) as progress:
        try:
            wav_files = discover_audio_files(source_dir)
        except OSError as exc:
            console.print(str(exc), style="bold red", markup=False)
            return None
        if not wav_files:
            console.print(f"No audio files found in {source_dir}", style="red", markup=False)
            return None

        task_id = progress.add_task("[cyan]Loading & analyzing tracks...", total=len(wav_files) * 2)
        on_done = lambda **data: progress.advance(task_id)  # noqa: E731
        event_bus.subscribe("track.analyze_complete", on_done)
        try:
            session = load_session(source_dir, config, event_bus=event_bus)
            if not session.tracks:
                raise ValueError(f"No audio files found in {source_dir}")
            if args.execute:
                _prepare_cli_destinations(
                    source_dir, output_dir, backup_dir,
                    [track.filename for track in session.tracks], args.report, args.json,
                )
        except (OSError, ValueError) as exc:
            console.print(str(exc), style="bold red", markup=False)
            return None

        session = pipeline.analyze_phase1(session)
        session = pipeline.analyze_phase2(session)
        event_bus.unsubscribe("track.analyze_complete", on_done)
    return session


def _print_repair_notices(session) -> None:
    flags = {"dc_remove": "dc", "low_cut": "subsonic", "ultrasonic": "ultrasonic"}
    for track in session.tracks:
        if track.repair_recipe.active:
            console.print("Repairs", track.filename, "·", ", ".join(track.repair_recipe.applied_ids()), markup=False)
            for warning in format_boundary_settling(track.repair_measurements.get("boundary_settling", [])):
                console.print(warning, style="yellow", markup=False)
        suggestions = track.repair_measurements.get("suggested", {})
        available = [
            f"--fix {flag}" for key, flag in flags.items()
            if suggestions.get(key) and key not in track.repair_recipe.applied_ids()
        ]
        if available:
            console.print("Suggested repairs", track.filename, "·", ", ".join(available), markup=False)


def _print_fader_table(session, write_errors: dict[str, str]) -> None:
    table = Table(box=box.ROUNDED, title="Fader Offsets")
    table.add_column("Track", style="cyan", max_width=30)
    table.add_column("Format", style="dim")
    table.add_column("Mode", justify="center")
    table.add_column("Gain", justify="right")
    table.add_column("Fader", justify="right", style="bold green")
    table.add_column("Status", justify="right")
    receipts = session.execution_receipts.executed_files or session.config.get("_executed_files", {})
    for t in session.tracks:
        if t.status != "OK" or t.filename in write_errors:
            table.add_row(t.filename, "Error", "\u2014", "\u2014", "\u2014", "[red]ERR[/]")
            continue
        fmt_str = f"{t.samplerate / 1000:.0f}k/{t.bitdepth}"
        pr = t.processor_results.get("bimodal_normalize")
        if pr and pr.classification == "Silent":
            table.add_row(t.filename, fmt_str, "Silent", "0.0 dB", "0.0 dB", "[yellow]SILENT[/]")
            continue
        classification = pr.classification if pr else "Unknown"
        receipt = receipts[t.filename]
        clip_r = t.detector_results.get("clipping")
        is_clipped = bool(clip_r.data.get("is_clipped")) if clip_r else False
        if "Peak" in classification:
            type_color = "magenta"
        elif classification == "Noise Floor":
            type_color = "yellow"
        elif classification == "Skip":
            type_color = "dim"
        else:
            type_color = "cyan"
        table.add_row(
            t.filename, fmt_str, f"[{type_color}]{classification}[/]",
            f"{receipt['gain_db']:+.1f} dB", f"{receipt['fader_db']:+.1f} dB",
            "[red]CLIP[/]" if is_clipped else "[green]OK[/]",
        )
    console.print(table)


def _execute_cli_session(args, session, pipeline: Pipeline, event_bus: EventBus, output_dir: str, backup_dir: str | None, is_overwriting: bool, config: dict, diagnostic_summary: dict) -> int:
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TextColumn("{task.completed}/{task.total}"), console=console,
    ) as progress:
        task_id = progress.add_task("[cyan]Processing tracks...", total=len(session.tracks))
        on_write = lambda **data: progress.advance(task_id)  # noqa: E731
        event_bus.subscribe("track.write_complete", on_write)
        session = pipeline.execute(session, output_dir=output_dir, backup_dir=backup_dir, is_overwriting=is_overwriting)
        event_bus.unsubscribe("track.write_complete", on_write)

    write_errors = dict(session.execution_receipts.execute_errors or session.config.get("_execute_errors", []))
    for filename, reason in write_errors.items():
        console.print(f"{filename}: {reason}", style="red", markup=False)

    print_diagnostic_summary(diagnostic_summary)
    _print_fader_table(session, write_errors)

    warnings = build_warnings(session, config)
    diagnostic_summary_text = render_diagnostic_summary_text(diagnostic_summary)
    report_path = os.path.join(output_dir, args.report)
    json_path = os.path.join(output_dir, args.json)
    try:
        generate_report(session, config, report_path, warnings, diagnostic_summary_text=diagnostic_summary_text)
        save_json(session, config, json_path, warnings)
    except OSError as exc:
        console.print(f"Could not save the processing reports. {exc}", style="bold red", markup=False)
        return 1
    console.print(f"\n[dim]Report saved to: {report_path}[/]")
    return 1 if write_errors else 0


def process_files():
    if (isinstance(sys.stdout, TextIOWrapper) and not sys.stdout.isatty()
            and sys.stdout.errors in {"strict", "surrogateescape", "surrogatepass"}):
        sys.stdout.reconfigure(errors="backslashreplace")
    args = parse_arguments()
    setup_logging()
    source_dir = args.directory
    if not os.path.isdir(source_dir):
        console.print(f"Input directory does not exist. {source_dir}", style="bold red", markup=False)
        return 1

    is_overwriting = args.execute and args.overwrite
    output_dir = source_dir if is_overwriting else os.path.join(source_dir, args.output_folder)
    backup_dir = os.path.join(source_dir, args.backup) if is_overwriting else None
    config = _build_cli_config(args, source_dir)
    errors = validate_config_fields(config)
    if errors:
        for error in errors:
            print(f"{error.key}: {error.message}", file=sys.stderr)
        return 2

    _print_header_and_groups(args, is_overwriting)
    event_bus = EventBus()
    pipeline = Pipeline(detectors=default_detectors(), audio_processors=default_processors(), config=config, event_bus=event_bus)
    session = _load_and_analyze_cli_session(args, source_dir, output_dir, backup_dir, config, pipeline, event_bus)
    if session is None:
        return 1

    session = pipeline.plan(session)
    session.tracks.sort(key=lambda t: protools_sort_key(t.filename))
    _print_header_and_groups(args, is_overwriting, session)
    diagnostic_summary = build_diagnostic_summary(session)
    _print_repair_notices(session)

    if not args.execute:
        print_diagnostic_summary(diagnostic_summary)
        failed = False
        for track in session.tracks:
            error = track.status if track.status != "OK" else track._cache.get("repair_error")
            if not error:
                error = next((r.error for r in track.processor_results.values() if r.error), None)
            if error:
                console.print(f"{track.filename}: {error}", style="red", markup=False)
                failed = True
        return int(failed)

    return _execute_cli_session(
        args, session, pipeline, event_bus, output_dir, backup_dir, is_overwriting, config, diagnostic_summary,
    )


if __name__ == "__main__":
    sys.exit(process_files())
