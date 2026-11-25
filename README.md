# SessionPrep

Batch audio analyzer, bimodal normalizer, and multi-stage DAW gain-staging tool for mix session preparation.

SessionPrep audits and prepares raw audio tracks for mix sessions.
It performs the mechanical, non-creative labor of checking files for problems,
normalizing track levels, and staging DAW routing/bus gain structure so you can
focus on mixing.

This is a **preflight heuristic** and a **starting-point generator**, not a fully
automatic mix prep tool. 

## Screenshots

<table>
  <tr>
    <td width="33%">
      <a href="https://github.com/user-attachments/assets/5cc16b4a-a43b-44c8-8716-ecd771a61972">
        <img width="1600" height="982" alt="image" src="https://github.com/user-attachments/assets/5cc16b4a-a43b-44c8-8716-ecd771a61972" />
      </a>
    </td>
    <td width="33%">
      <a href="https://github.com/user-attachments/assets/87af84e9-b704-418a-9258-ff3f4d8ca38a">
        <img width="1602" height="979" alt="image" src="https://github.com/user-attachments/assets/87af84e9-b704-418a-9258-ff3f4d8ca38a" />
      </a>
    </td>
    <td width="33%">
      <a href="https://github.com/user-attachments/assets/f6632479-fc35-4b0d-9098-edc937e2767e">
        <img width="1599" height="975" alt="image" src="https://github.com/user-attachments/assets/f6632479-fc35-4b0d-9098-edc937e2767e" />
      </a>
    </td>
  </tr>
  <tr>
    <td width="33%">
      <a href="https://github.com/user-attachments/assets/690c7a0c-0943-4d3c-8bac-e4329a392fdb">
        <img width="1602" height="982" alt="image" src="https://github.com/user-attachments/assets/690c7a0c-0943-4d3c-8bac-e4329a392fdb" />
      </a>
    </td>
    <td width="33%">
      <a href="https://github.com/user-attachments/assets/b629009f-1b9e-4650-8036-2940be348a1c">
        <img width="1601" height="981" alt="image" src="https://github.com/user-attachments/assets/b629009f-1b9e-4650-8036-2940be348a1c" />
      </a>
    </td>
    <td width="33%">
      <a href="https://github.com/user-attachments/assets/54a0bcaf-4241-4d65-a661-168c96b9dcb8">
        <img width="1147" height="732" alt="image" src="https://github.com/user-attachments/assets/54a0bcaf-4241-4d65-a661-168c96b9dcb8" />
      </a>
    </td>    
  </tr>
</table>

---

## Quick Start

**1. Download** `sessionprep-gui` (GUI) or `sessionprep` (CLI) from the
releases page — no Python required.

**2. Run the GUI** — open a session directory and review the analysis:

```
sessionprep-gui
```

The GUI provides an interactive file table, per-track detail view, waveform
and spectrogram display with issue overlays, and audio playback.  Select
multiple tracks (Shift-click or Ctrl-click) and hold **Alt+Shift** while
clicking a dropdown (RMS Anchor, Classification) to apply the change to all
selected tracks at once — mirroring the Pro Tools convention for batch control
changes.  Re-analysis runs asynchronously with a progress bar.

On first launch it creates a `sessionprep.config.json` in your user
preferences directory with all default thresholds. A rotating log file
(`sessionprep.log`) is written to the same directory. Set the environment
variable `SP_LOG_LEVEL` to `DEBUG`, `WARNING`, or `NONE` to control verbosity
(default: `INFO`).

Use **File → Preferences** to customize settings through a two-tab dialog:

- **Global** — General (HiDPI scale, default project directory), Colors
  (palette editor), and Groups (named group presets with color, gain-linking,
  and DAW target columns).
- **Config Presets** — named presets for Analysis, Detector, Processor, and
  DAW Processor parameters. Add, duplicate, rename, or delete presets. Each
  parameter shows a description, expected type, valid range, and default
  value. A reset button next to each widget restores the default.

Preferences are saved to `sessionprep.config.json`. In Phase 2, use **Config
Preset** to select the project's defaults and edit individual fields to set
session overrides. Untouched fields follow that preset when it changes,
including edits saved from Preferences in Phase 3. Explicit overrides stay
with the project when it switches presets. A field's reset arrow clears that
override and restores the selected preset value. **Revert to Preset** clears
all Config overrides. The blue Config markers show where effective values
differ from the selected preset.

DAW and report display changes apply immediately. Changes to analysis
settings take effect when the relevant phase runs again.

**3. Or use the CLI** for scripting and batch workflows:

```bash
sessionprep /path/to/tracks          # analyze (safe, read-only)
sessionprep /path/to/tracks -x       # analyze + process (writes to processed/)
```

In the GUI, **Summary** lists each finding, its affected file and the next step.
Use **Review file** to inspect it. Each detector row starts with its severity;
expand the finding for its full result. Under **Actions**, **Visualize** shows
only that detector's locations, opens the preview and fits the whole file while
retaining RMS curves and Peak/RMS markers. **Help** opens its explanation in a
right-side panel with its own scroll area. Click another Help link to switch
explanations, or close the panel with **X** or **Esc** while it has focus.
The local **Waveform preview** header lets
you hide or reopen audio inspection without losing the current view. When
collapsed, the header sits at the bottom and releases the space to the report. The chevron
points up to open and down to close; keyboard focus remains visible.
Table Gain fields share the neighboring selectors’ gray surface, with clear focus
and muted disabled values.

**4. Import** tracks into your DAW. **DAW Transfer** builds self-contained
DAWproject archives (with embedded audio, fader offsets, and empty-folder
cleanup), and for Pro Tools automates session setup via PTSL and validated
`.ptx` updates: routing inserted tracks to their parent Routing Folders,
clearing newly created audio track inputs to `No Input`, removing empty
template folders, unused VCAs, and causally orphaned downstream Aux busses
(enabled by default), creating per-folder Groups linked to template VCAs,
applying two-level Clip Gain + inverse Track Fader staging, and running
**recursive multi-stage bimodal gain staging** across the template's bus
hierarchy with an offline **Signal Flow** inspector.

---

## Installation

### Standalone executables (recommended, no Python required)

Download from the releases page:

| Executable        | Description                                                  |
|-------------------|--------------------------------------------------------------|
| `sessionprep-gui` | GUI application (interactive analysis + waveform + playback) |
| `sessionprep`     | Command-line tool (scripting, batch workflows, CI)           |

The All Platforms workflow produces one Windows installer that selects native
x64 or ARM64 files, separate macOS ARM64/Intel DMGs, and Linux packages for both
architectures. Individual build runs retain their own platform artifacts. Builds
are manually triggered; release and preview drafts remain unpublished until
reviewed. See [build and release details](packaging/github/README.sessionprep.md).

### From source

If you prefer to run the Python scripts directly:

```bash
git clone <repo-url>
cd sessionprep
uv run python sessionprep-gui.py                        # run the GUI
uv run --extra cli python sessionprep.py /path/to/tracks # run the CLI
```

The GUI command installs the optional desktop dependencies on first use if
needed.

See [DEVELOPMENT.md](DEVELOPMENT.md) for full development setup, building, and
distribution instructions.

---

## How It Works

SessionPrep operates in three phases:

| Phase | Name                   | What happens                                                                                                                                                                                                                                                                                                                                                                                                              | When                           |
|-------|------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------|
| **1** | Track Layout           | Define how source tracks map to output files: channel routing, reordering, splitting, merging. Automatically optimizes layouts based on Phase 1 diagnostics (e.g. dropping silent files, extracting active channels from dual-mono and one-sided silence). Drag-and-drop between input/output trees with visual insert-position indicator. Optional recursive subfolder scanning. Output written to `sp_01_tracklayout/`. Pre-computed peak cache files written to `sp_peaks/` for fast waveform rendering. | GUI Phase 1 (always available) |
| **2** | Analysis & Preparation | Format checks, clipping, DC offset, stereo compatibility, silence, subsonic, peak/RMS measurement, classification, tail exceedance. Bimodal normalization (clip gain adjustment) via Prepare. Output written to `sp_02_prepared/`.                                                                                                                                                                                        | GUI Phase 2 / CLI              |
| **3** | DAW Transfer           | Transfer tracks into DAW sessions with per-track naming and folder assignment. Duplicate entries for multi-track scenarios (same clip on different tracks). DAWproject exports self-contained archives with embedded audio, fader offsets, and empty-folder cleanup. Pro Tools transfers route tracks to parent Routing Folders, clear audio track inputs to `No Input`, clean up empty folders, unused VCAs, and orphaned downstream Aux busses, create VCA-linked folder Groups, apply two-level Clip Gain + Track Fader staging, and run recursive multi-stage bimodal bus gain staging via validated Slot A trim plugins (with an offline Signal Flow inspector and per-bus fader overrides). Supports unattended batch processing of multiple songs. | GUI Phase 3 |


In Track Layout, **Unassigned** marks an input channel with no output routes. Collapse the input file to see how many of its channels are unassigned.

Reopening a project restores your edited routes. **Reset to Default** rebuilds the recommended layout from the original input files. When an older project lacks source findings, or a source file or detection threshold has changed, opening Track Layout checks the affected inputs in the background.

### Diagnostic categories

The output is organized into four categories:

- **PROBLEMS** — typically require fixing before mixing (clipping, format mismatches)
- **ATTENTION** — may be intentional, worth checking (DC offset, subsonic, silence)
- **INFORMATION** — useful context (stereo compatibility, dual-mono)
- **CLEAN** — explicit "No ... detected" lines when checks pass

For details on every detector, see [REFERENCE.md](REFERENCE.md).

### Correcting a finding

In Phase 2, **Track repairs…** appears in File and the track’s Processing menu when the track has a repairable finding, an existing repair setting, or an output that needs recovery. Expand a DC, subsonic or ultrasonic finding and choose **Review repair** to open the relevant control. Choose **On** to apply it to the whole track or **Use preset** to follow the selected preset. Switch between **Original** and **Corrected** in the shared waveform or spectrogram to audition the change before normalization. **Back to findings** or **Alt+Left** returns to your place in the report and any open detector Help.

File shows **✓ Correction applied** when a detector rechecks the repaired audio and clears a finding. Expand the row for the before-and-after readings. The plots follow the selected source, and Summary counts the remaining findings.

### Processing: Bimodal normalization

Most normalizers treat a kick drum and a synth pad the same. SessionPrep
classifies tracks using three metrics — crest factor, envelope decay rate,
and content density — then normalizes them differently:

- **Transient** (high crest + fast decay, or sparse percussion): peak-normalized to `--target_peak` (-6 dBFS)
- **Sustained** (low crest + slow decay): RMS-normalized to `--target_rms` (-18 dBFS), peak-limited

The three-metric approach resolves common misclassifications: compressed drums
(low crest but fast decay → correctly Transient), plucked instruments (high
crest but slow decay → correctly Sustained), and sparse percussion like toms
and crashes (caught by density even with ambiguous crest/decay values).

### Multi-stage bus gain staging (Pro Tools)

Calibrating individual tracks does not keep summed subgroups, stems, or the mix
bus at target level: a few quiet tracks summed onto a subgroup can under-drive
bus inserts, while many tracks summed onto a drum bus or mix bus can overload
them.

When transferring to a Pro Tools template, SessionPrep reconstructs the
template's signal-flow graph (Audio → Routing Folders → Subgroup/Stem Auxes →
Mix Bus → Master), sums the time-aligned, gain-adjusted audio in topological
order, and applies **recursive multi-stage bimodal gain staging**:

- **Intermediate insert-bearing busses** use a validated Insert Slot A trim
  plugin (**DMG Audio TrackControl**, **Avid Trim** in default `+6 dB` mode, or
  **Process.Audio Protility**) set to `+g` pre-insert, paired with an inverse
  `-g` post-insert Bus Fader offset (in balance-preserving modes) so downstream
  bus inserts receive the bimodal target level while the rough-mix balance stays
  intact across unequal bus depths.
- **Terminal Mix Bus** applies a single attenuate-only pre-insert Slot A trim
  with its fader at unity (`0.0 dB`) so downstream serial Mix Busses and Master
  stay below target without overloading.
- **Missing or blocked Slot A handling:** On templates using Slot A trim
  plugins, if a required staged bus has an empty Slot A and needs non-zero
  staging, SessionPrep clones the template's most-used clean matching-format
  (`Mono`/`Stereo`) Slot A trim plugin into Slot A automatically; if Slot A is
  occupied by a non-trim plugin, transfer stops with an error rather than
  silently falling back to post-insert faders. Pure summing folders, quiet
  `0.0 dB` busses, post-terminal Mix Busses, and Master are exempt.

For the engineering rationale, see [TECHNICAL.md](TECHNICAL.md). For full Pro
Tools transfer details, see [REFERENCE.md §9](REFERENCE.md#9-pro-tools-transfer-automation-ptx--ptsl).

---

## Usage Examples (CLI)

### Basic analysis (dry-run)
```bash
uv run python sessionprep.py .
```

### Execute processing
```bash
uv run python sessionprep.py . -x
```

### Apply repairs to matching tracks

```bash
uv run python sessionprep.py . --fix dc --fix subsonic --low-cut-hz 10 --execute
```

Omit `--execute` to review selections first. Add `--fix ultrasonic` for eligible high-rate files.

### Force classification overrides
```bash
uv run python sessionprep.py . -x --force_transient 808 snare --force_sustained pad
```

### Hotter calibration (modern plugins)
```bash
uv run python sessionprep.py . -x --target_rms -16
```

### Safety-first RMS anchor (strict max window)
```bash
uv run python sessionprep.py . -x --rms_anchor max
```

### Tune the relative gate (sparse tracks)
```bash
uv run python sessionprep.py . -x --gate_relative_db 30
```

### Anchor track (keep one fader at 0 dB)
```bash
uv run python sessionprep.py . -x --anchor "Kick"
```

### Custom report filenames
```bash
uv run python sessionprep.py . -x --report my_report.txt --json my_report.json
```

### Group related tracks (identical gain)
```bash
uv run python sessionprep.py . -x --group Kick:kick,kick_sub --group OH:overhead,oh --group Toms:tom
```

Groups use `Name:pattern1,pattern2` syntax. Patterns support substring, glob
(`*`/`?`), or exact match (suffix `$`). First match wins; if a file matches
multiple groups a warning is printed.

---

## Documentation

| Document                         | Contents                                                         |
|----------------------------------|------------------------------------------------------------------|
| [README.md](README.md)           | This file — overview, installation, quick start, usage           |
| [REFERENCE.md](REFERENCE.md)     | Detector reference, analysis metrics, processing details         |
| [TECHNICAL.md](TECHNICAL.md)     | Audio engineering background, normalization theory, signal chain |
| [DEVELOPMENT.md](DEVELOPMENT.md) | Development setup, building, library architecture                |
| [KANBAN.md](KANBAN.md)           | Backlog and planned features                                     |

---

## License

LGPL-3.0-or-later. See [LICENSE](LICENSE).
