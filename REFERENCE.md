# SessionPrep — Analysis & Processing Reference

This document is the detailed reference for every detector, analysis metric,
and processing stage in SessionPrep.

For the technical rationale, see [TECHNICAL.md](TECHNICAL.md).
For usage and quick start, see [README.md](README.md).

---

## Table of Contents

1. [Diagnostic Categories](#1-diagnostic-categories)
2. [Detectors](#2-detectors)
3. [Analysis Metrics (Stage B)](#3-analysis-metrics-stage-b)
4. [Processing (Stage C)](#4-processing-stage-c)
5. [Fader Restoration (Stage D)](#5-fader-restoration-stage-d)
6. [GUI Waveform Controls](#6-gui-waveform-controls)
7. [GUI Track Table Controls](#7-gui-track-table-controls)
8. [Pro Tools Utils Window](#8-pro-tools-utils-window)
9. [Pro Tools Transfer Automation (`.ptx` / PTSL)](#9-pro-tools-transfer-automation-ptx--ptsl)

---

## 1. Diagnostic Categories

The terminal output and `sessionprep.txt` are organized into four high-level
categories:
  - **PROBLEMS**: typically require fixing before mixing.
  - **ATTENTION**: may be intentional, but worth checking.
  - **INFORMATION**: useful context that generally does not require client fixes and typically does not change processing decisions.
  - **CLEAN**: explicit "No ... detected" lines when checks pass.

### What makes a detector worth having?

A detector earns its place if it meets at least one of these criteria:
  - Catches something you can't easily hear
  - Catches something that wastes significant time if discovered late
  - Catches something the client/label needs to fix
  - Changes your processing decisions

---

## 2. Detectors

### 2.1 File errors

- **What it means:** a file could not be read/analyzed.
- **Why it matters:** missing/failed files break prep and invalidate the session overview.
- **Categorization:**
  - PROBLEMS: any file errors were encountered.

### 2.2 Session format consistency

- **What it means:** whether all analyzed files share the same sample rate / bit depth.
- **Why it matters:** inconsistent session formats create import problems, SRC surprises, and alignment risk.
- **Categorization:**
  - CLEAN: `No inconsistent session formats` (all files match the most common format).
  - PROBLEMS: per-file `Format mismatches` when any file differs from the most common format (the header includes the most common/reference format summary).

### 2.3 File length consistency

- **What it means:** whether all analyzed files have the same timeline length (normalized to the most common sample rate).
- **Why it matters:** mismatched track lengths can cause misalignment, missing tails, or timing drift when importing.
- **Categorization:**
  - CLEAN: `No inconsistent file lengths` (includes the most common length in samples and minutes:seconds).
  - PROBLEMS: per-file `Length mismatches` when any file differs from the most common length (the header includes the most common/reference length summary).

### 2.4 Digital clipping

- **What it means:** consecutive samples reach a decoded magnitude of 0.9999 or higher on the same channel. For signed or unsigned 8-bit PCM, the positive rail is 127/128 and the negative rail is −1. Corrected audio is checked before output encoding.
- **Why it matters:** clipped prints are often unrecoverable and will distort downstream processing.
- **Controls:** `--clip_consecutive`, `--clip_report_max_ranges` (the report threshold is the run length: how many consecutive near-full-scale samples are required to count as clipping)
- **Categorization:**
  - CLEAN: `No digital clipping detected`.
  - PROBLEMS: `Digital clipping` with per-file details.

### 2.5 DC offset

- **What it means:** the file has a measurable DC component above `--dc_offset_warn_db`.
- **Why it matters:** DC can reduce headroom, bias compressors/saturators, and skew metering.
- **Controls:** `--dc_offset_warn_db`
- **Categorization:**
  - CLEAN: `No DC offset issues detected`.
  - ATTENTION: `DC offset` with per-file details.

### 2.6 Stereo compatibility

- **What it means:** one or more stereo-compatibility warnings were detected:
  - correlation below `--corr_warn` and/or
  - mono fold-down loss above `--mono_loss_warn_db`
- **Why it matters:** low or negative correlation indicates phase differences between channels that cause level loss or cancellation in mono playback (phone speakers, mono PA systems).
- **Controls:** `--corr_warn`, `--mono_loss_warn_db`, `--corr_windowed`, `--corr_window_ms` (default 250 ms), `--corr_max_regions`
- **Windowed analysis:** when enabled, per-window Pearson correlation and mono folddown loss are computed. Contiguous windows exceeding either threshold are merged into regions with waveform overlays.
- Whole-file readings use all frames with the channel means removed over the entire track. Changing the local window or disabling localization preserves those readings. A channel with no AC signal makes correlation unavailable. A hard-panned signal still has a measurable 3.01 dB mono loss.
- Complete cancellation in mono is reported as `∞ dB` loss. The JSON export records `mono_fully_cancelled: true` and `mono_loss_db: null` for this case. Finite losses keep their numeric dB value.
- **Categorization:**
  - ATTENTION: localized regions with poor stereo compatibility detected.
  - INFORMATION: whole-file correlation or mono loss exceeds threshold (no localized regions).
  - CLEAN: `No stereo compatibility issues detected`.

### 2.7 Dual-mono (identical L/R)

- **What it means:** the largest absolute difference between corresponding left and right samples in one stereo file is at or below the configured tolerance. Every frame is checked. A pair of separate mono files remains two inputs.
- **Why it matters:** usually a valid delivery choice and often intentional.
- **Controls:** `--dual_mono_eps` defaults to `0.00001` of full scale. Even perfectly correlated channels can differ by more than this if one has a different gain.
- **Track Layout:** Recommends keeping the left channel as a mono output. A completely silent file follows the Silent Files rule.
- **Categorization:**
  - INFORMATION: `Dual-mono (identical L/R)` with per-file details.

### 2.8 Silent files

- **What it means:** the file is all zeros (or effectively empty).
- **Why it matters:** may be intentional (placeholder) but is often an export issue.
- **Track Layout:** Triggers a layout recommendation (`topology_action: drop`) to skip the file in the session layout outputs entirely.
- **Categorization:**
  - CLEAN: `No silent files detected`.
  - ATTENTION: `Silent files` with per-file details.

### 2.9 One-sided silence

- **What it means:** exactly one channel's RMS is at or below the configured level after subtracting that channel's mean. Both RMS readings use the entire file.
- **Why it matters:** often an export/cabling/routing mistake; importing it as stereo can cause unexpected balance issues.
- **Controls:** `--one_sided_silence_db`
- **Track Layout:** Triggers a layout recommendation (`topology_action: extract_channel`) to extract specifically the active, non-silent channel into a mono output track.
- **Categorization:**
  - CLEAN: `No one-sided silent stereo files detected`.
  - ATTENTION: `One-sided silence` with per-file details.

### 2.10 Subsonic content

- **What it means:** the file has significant energy below a cutoff frequency
  (default `30 Hz`).
- **Why it matters:** subsonic rumble can eat headroom, trigger
  compressors/limiters, and cause translation issues on smaller speakers.
- **Controls:** `--subsonic_hz`, `--subsonic_warn_ratio_db`,
  `subsonic_windowed`, `subsonic_window_ms`, `subsonic_max_regions`
- **Categorization:**
  - CLEAN: `No significant subsonic content detected`.
  - INFORMATION: `Subsonic content` with per-file details (consider an HPF).

**How detection works:**

The detector subtracts each channel's whole-file mean before Hann windowing,
padding or gating. The whole-file ratio compares energy in positive Fourier
bins at or below the cutoff with total AC energy. Bin spacing is the inverse
of the recording duration. A constant offset alone cannot trigger the advisory.
A ratio of −20 dB means 1 % of the signal's AC energy is subsonic, the default
threshold for an Information finding. Local Hann windows locate evidence on
flagged channels. Changing their length leaves the whole-file ratio intact.

**Per-channel analysis (always active for stereo+):**

For multi-channel files, each channel is analyzed independently.  The combined
ratio is the **maximum** (worst) of all per-channel ratios — more conservative
than the previous mono-downmix approach, which could mask subsonic content in
one channel through phase cancellation.  If only one channel exceeds the
threshold, the issue is reported for that specific channel; if all channels
exceed it, a whole-file issue is reported.

Subsonic issues carry frequency bounds (`freq_min_hz=0`, `freq_max_hz=cutoff`)
for frequency-bounded overlays in spectrogram mode.

**Windowed analysis (default: on):**

When `subsonic_windowed` is enabled, the detector also splits each channel into
short windows (default 500 ms via `subsonic_window_ms`) and computes the
subsonic ratio per window. Contiguous windows exceeding the threshold are
merged into regions with precise sample ranges, visible as waveform overlays.

This serves two purposes:
  1. **Localization** — shows *where* subsonic content is concentrated (bass
     drops, HVAC bleed in quiet sections) instead of painting the entire file.
  2. **Absolute subsonic power gate** — each window's absolute subsonic energy
     level is checked (`window_rms_db + ratio_db`).  Windows where this is
     below −40 dBFS are suppressed.  This prevents amp hum or noise in quiet
     gaps from producing false positives — a high subsonic *ratio* is
     meaningless when the absolute energy is too quiet to waste headroom.

**Threshold relaxation for windowed analysis:**

Individual 500 ms windows have less frequency resolution than a very long FFT.
When the whole-file ratio is borderline (e.g. −18 dB vs −20 dB threshold), no
single window may cross the same threshold even though the aggregate clearly
does.  To handle this, the windowed analysis uses a relaxed threshold (6 dB
below the configured threshold).  This ensures that windows where subsonic
energy is concentrated still produce visible regions.

**Fallback for diffuse subsonic content:**

If no windows exceed the relaxed threshold (the subsonic energy is spread
evenly across the file rather than concentrated), the detector falls back to
marking **active-signal regions** — windows whose RMS is within 20 dB of the
loudest AC window on the flagged channel. These fallback locations identify
active audio. Only measured band locations report a local subsonic ratio.

If even the active-signal approach produces no regions, a whole-file overlay
is shown as a last resort — an Information result always has at least one
visible issue.

### 2.11 Tail regions exceeded anchor

- **What it means:** contiguous regions where momentary RMS exceeds the file's anchor by more than `--tail_min_exceed_db`.
- **Why it matters:** highlights "quiet overall but occasionally very loud" tracks that may need section-based rides.
- **Controls:** `--tail_min_exceed_db`, `--tail_hop_ms`, `--tail_max_regions`
- Each marked region covers consecutive, touching measurement windows above the threshold. A gated-out interval separates the passages on either side of it.
- **Categorization:**
  - ATTENTION: `Tail regions exceeded anchor` with per-file details.

### 2.12 Grouping overlaps

- **What it means:** a file matched multiple `--group` specs.
- **Why it matters:** indicates ambiguous patterns; grouping may not apply as intended.
- **Controls:** `--group Name:pattern1,pattern2` (first match wins; overlaps produce a warning)
- **Categorization:**
  - ATTENTION: `Grouping overlaps` with overlap details.

---

### 2.13 Ultrasonic content

Sources at 50 kHz or higher are eligible. Five consecutive 100 ms windows with 50% overlap must exceed both −60 dBFS RMS and −35 dB relative to program energy in the band from 24.1 kHz to Nyquist. Each channel is evaluated separately.

The local meter uses the available samples on clips shorter than 100 ms. Only full windows count toward a suggestion. Reported spans cover the qualifying windows. Manually enabling the filter provides both the original and corrected readings.

## 3. Analysis Metrics (Stage B)

Stage B is where SessionPrep extracts level and dynamic information from each
file using windowed measurements. This is meant to be more representative than
a single "whole-file average", especially for tracks with intros, dropouts, and
sparse events.

### 3.1 Peak amplitude (dBFS)

**What it is:**
  - The maximum absolute sample value converted to dBFS (sample peak).

**Why it matters:**
  - A fast reality check for headroom and accidental full-scale content.
  - Useful context for how hot a track is feeding inserts in a DAW.

**Relevance for mix engineers:**
  - Quickly reveals "too hot to mix comfortably" tracks (even if the perceived
    loudness is low).

### 3.2 Momentary RMS windows (short-time energy)

**What it is:**
  - RMS is computed over sliding windows (default `--window 400` ms) across the
    file, creating a distribution of momentary loudness.

**Why it matters:**
  - Windowing avoids being dominated by long silences or by a single loud hit.

**Relevance for mix engineers:**
  - Better matches how you perceive a track's "working level" when you start
    inserting compressors, saturators, and channel strips.

### 3.3 Anchor selection (percentile vs. max)

**What it is:**

The RMS analysis produces many short-time RMS values — one per window (default
400 ms). After relative gating removes near-silent windows (see §3.4), we have
a distribution of momentary RMS levels representing the "active" parts of the
track. The **anchor** is the single representative value chosen from this
distribution, used as the reference level for gain decisions and tail
exceedance reporting.

Two strategies are available:

  - **`percentile`** (default, `--rms_anchor percentile`):
    Takes the Nth percentile of the gated window distribution
    (default P95 via `--rms_percentile 95`). This means 95 % of the active
    windows are at or below the anchor — in practice, the anchor represents
    "what the loud sections of this track typically sound like."

  - **`max`**:
    Takes the single loudest gated window. The anchor is the absolute peak of
    the RMS distribution.

**Why percentile is usually better than max:**

Consider a vocal track with a consistent verse at −20 dBFS RMS, a loud chorus
at −16 dBFS, and one isolated shout that hits −12 dBFS for a single 400 ms
window:

  - **Max anchor** = −12 dBFS. Gain is computed against that one shout, which
    means the verse and chorus are treated as "quieter than the track's level."
    The tail exceedance report is clean (nothing exceeds the max), but the gain
    decision is driven by a moment that may not represent the working level of
    the track at all.

  - **P95 anchor** ≈ −16 dBFS (roughly the chorus level). Gain is computed
    against the chorus — the part that matters most when feeding insert
    processing. The shout shows up as a tail exceedance, flagging it for
    manual clip-gain attention.

In general, **max** is fragile because a single anomalous window (a breath
pop, a drum bleed spike, a momentary feedback ring) can pull the anchor away
from the track's true working level. **Percentile** is robust to these
outliers while still tracking the loud sections faithfully.

**When max is useful:**

  - Very short, punchy files (a single hit, a sound effect) where there is no
    meaningful "distribution" — every window matters equally.
  - When you explicitly want the gain decision anchored to the absolute loudest
    moment in the file.

**How P95 works step by step:**

  1. Compute momentary RMS for each 400 ms window across the file.
  2. Gate: discard windows more than `--gate_relative_db` (default 40 dB) below
     the loudest window. This removes silence and very quiet sections.
  3. Sort the remaining "active" windows by RMS value.
  4. Pick the value at the 95th percentile of that sorted list.
  5. Convert to dBFS → this is `rms_anchor_db`.

The anchor is then used to:
  - Compute sustained-material gain: `gain = target_rms − anchor`
    (capped by `target_peak − peak`).
  - Identify tail exceedances: windows that exceed the anchor by more than
    `--tail_min_exceed_db` are reported as regions needing manual attention.

**Relevance for mix engineers:**
  - Percentile anchoring reflects "the part of the track that matters" — the
    loud sections that will drive your insert processing — rather than a
    momentary spike that may not represent the track's working level.
  - If you find that gain decisions are too conservative (track ends up quieter
    than expected), lower the percentile (e.g., `--rms_percentile 90`).
  - If gain decisions are too aggressive (track ends up too hot), raise the
    percentile or switch to `--rms_anchor max`.

### 3.4 Relative gating for sparse tracks

**What it is:**
  - Before anchor/max/tail statistics are computed, momentary windows that are
    far below the loudest window are ignored.
  - The gate is *relative* to the loudest window:
    `threshold_db = max_window_db - gate_relative_db`

**Why it matters:**
  - Sparse tracks (mostly silence with a few hits/phrases) can otherwise produce a
    misleadingly low percentile anchor dominated by silence.

**Relevance for mix engineers:**
  - Prevents "false alarm" tail exceedances and keeps analysis anchored to the
    actual musical content.

**Parameter intuition:**
  - `--gate_relative_db 40` means "keep windows within 40 dB of the loudest RMS
    window." It is not an absolute dBFS value (so it is not `-40`).

### 3.5 Peak-Capped RMS Normalization & Governing Constraint

**What it is:**
  - Every non-silent track above the `-70.0 dBFS` noise-floor hold is normalized
    with one continuous rule:
    `gain = min(target_rms - rms_anchor, target_peak - sample_peak, max_gain_db)`.
  - At the default targets (`-18.0 dBFS` unweighted RMS / `-6.0 dBFS` Peak
    ceiling), tracks with effective crest `≥ 12.0 dB` are governed by **Peak**
    (`Auto (Peak)`), while tracks with effective crest `< 12.0 dB` (sustained
    sources and compressed drum loops) are governed by **RMS** (`Auto (RMS)`).
  - Tracks at or below `-70.0 dBFS` peak hold at `0.0 dB` (**Noise Floor**) to
    prevent room-tone or dither wind-up. Automatic boost is capped by
    `--max_gain_db` (default `+24.0 dB`, range `0–36 dB`).
  - **Governing & Next-Highest Peak:** Reports the clickable timestamp of the
    global sample peak alongside the next-highest distinct peak outside a
    stereo-linked `±100 ms` exclusion window (`—` when no non-zero sample lies
    outside `±100 ms`).
  - **Lazy True-Peak Headroom Warnings:** When post-gain sample peak exceeds
    `-5.5 dBFS` (for example under an `RMS` override or manual gain edit),
    SessionPrep computes oversampled true peak (`dBTP`) on the post-repair audio
    and flags **Amber** (`> −1.0 dBFS/dBTP`) or **Red** (`> 0.0 dBFS/dBTP`).
  - **CLI Flags & Legacy Compatibility:**
    - `--target_rms` (default `-18.0`), `--target_peak` (default `-6.0`, must
      satisfy `target_rms <= target_peak`), `--max_gain_db` (default `24.0`),
      `--force_peak <regex>` (force Peak mode on matching filenames).
    - Deprecated flags `--force_transient` (mapped to `--force_peak`),
      `--crest_threshold`, and `--force_sustained` are accepted with deprecation
      warnings to stderr. Legacy `.spsession` (v7 → v8) and preset files migrate
      automatically (`Transient → Peak`, `Sustained → Auto`, `Skip → Skip`;
      prepared outputs with auto-shifted tracks are marked stale).

### 3.6 Tail exceedance report (significant regions above anchor)

**What it is:**
  - When using percentile anchoring, the analysis also reports contiguous regions
    where momentary RMS significantly exceeds the anchor.

**Why it matters:**
  - Highlights tracks that have "quiet overall, but occasionally very loud"
    sections, which can affect gain staging and compression decisions.

**Relevance for mix engineers:**
  - Points you to the exact time ranges that may need manual attention (clip gain
    rides, automation, or section-specific treatment).

---

## 4. Processing (Stage C)

In execute mode (`-x/--execute`), based on the analysis, SessionPrep writes a
COPY of your audio to a `processed/` folder.

With `--overwrite` and a backup directory, keep each backup WAV together with
its adjacent `.sessionprep-backup.json` file. The receipt lets later runs verify the
first-generation copy and the current source. An older backup with identical
bytes can be adopted. If the files differ and there is no receipt, choose a
fresh backup directory to preserve both versions.

It applies a specific gain amount to put tracks into a predictable starting
range (default `--target_rms -18` for sustained material or `--target_peak -6`
for transient material). This primarily affects the level feeding the first
insert on each channel.

### Optional corrective processing

**DC removal** uses a first-order 2 Hz high-pass. **Subsonic filter** uses a second-order high-pass with a 10 Hz default and a 5–40 Hz range. Selecting both applies the stronger high-pass once. Automatic low-cut selection looks for sustained infrasonic energy on individual channels after any selected DC removal. The broader subsonic advisory can also describe intentional bass. **Ultrasonic filter** preserves the band through 20 kHz and attenuates frequencies from 24 kHz by at least 80 dB on eligible sources. Its FIR delay is compensated. All three preserve the sample count and channel layout.

Automatic low-cut selection uses three consecutive 1 s windows at 50% overlap and considers AC energy up to the smaller of 20 Hz and the configured subsonic frequency. The repair's local meter subtracts the frame mean before applying a Hann window. An isolated stationary tone at or inside a band edge contributes its full windowed RMS, including the energy that tapering a very low tone places in the zero-frequency bin. Other energy is assigned by local frequency estimates. Only confidently in-band energy can qualify an automatic repair when components overlap at the edge.

Use **Preferences → Config Presets → Processors → Corrective processing** for automatic selection defaults. Phase 2 Config can override these for one project. File and the track’s Processing menu offer **Track repairs…** when a detector finds a matching issue, a track has an applied repair or saved repair setting, or an output needs recovery. Expand a finding and choose **Review repair** to open its controls. If DC removal also clears Subsonic Content, both findings point to DC removal. Remaining local rumble can still offer the subsonic filter. **Use preset** releases a track override. A Stereo Compatibility finding on its own offers **Visualize** and **Help**.

Phase 2 checks enabled signal detectors on the repaired audio before normalization. **✓ Correction applied** means the detector flagged the original and passed on the repaired track. Expand the finding to compare measurements, channels and location counts. **New after repair** identifies a finding first detected on the repaired audio. The Analysis column and Summary count the remaining findings for either audition source. While a check is pending or unavailable, they retain the original warnings.

In **Review repair**, read the **Original** and **Corrected** columns across each row. For the subsonic filter, the first table compares whole-track energy with the detector threshold. When local rumble was found, the next table shows its RMS level, region count and affected channels for each source. A whole-track check can pass while local regions remain. A dash means that reading is unavailable.

Waveform and Spectrogram show the locations measured on the selected source. **Show original locations** returns to the source evidence at the current playhead and zoom. The project saves completed checks. Reopening an older project with active repairs starts verification in the background.

Expand a repair to inspect its before-and-after measurements or edit its settings. **Back to findings** or **Alt+Left** restores the expanded findings, report position, open Help passage and keyboard focus. The waveform and spectrogram share the same audition while you move between findings and repairs. Selecting another track opens that track’s File report.

Corrections run before gain calculation. **Prepare** checks the final, routed samples against the destination encoding. If the level is too high, **Adjust Gain** offers the highest safe output gain you can select in 0.1 dB steps. Confirm the change and retry. The CLI reports the necessary reduction in 0.1 dB steps from the current gain. Files are replaced after a successful render. A failed retry retains the previous file and marks it stale.

Check the repair measurements for **Boundary settling** when a filter rings into an edit gap. For each channel with at least 100 ms of exact source silence, SessionPrep reports the first and last affected sample and the peak if it exceeds −120 dBFS before gain. The text and JSON delivery reports include the same measurements.

When several normalized sources share an output, they must share a DAW fader offset. If Prepare reports different offsets, link the source gains or route them to separate outputs and retry.

---

## 5. Fader Restoration (Stage D)

The script calculates the inverse fader offsets for the gain applied in Stage C.
Applying those offsets preserves the incoming file-level balance exactly (which
matches the producer's rough-mix balance when multitracks were exported
post-fader; when tracks were exported pre-fader at unity, inverse faders simply
reproduce the raw unity import while track inserts still receive calibrated
levels).

Linked tracks use the lowest individual gain among the active members.
Headroom can shift all eligible faders down together. A named anchor then
shifts them together again so the first matching active track reads 0 dB.
This anchor takes priority over the headroom target. Other faders can end
up above the DAW's maximum when the anchor requires a large upward shift.
Changing a group recalculates these offsets from its current members.
In the normalization comparison, targets and measurements use the same
selected dBFS convention. Switching the display convention preserves the audio gain.

Example:
  - Input: A quiet synth pad (-24 dB).
  - Action: Script adds +6 dB Clip Gain to hit the -18 dB target.
  - Output: Script reports a Fader Offset of -6 dB.

In execute mode (`-x/--execute`), the script generates a text report and a
machine-readable JSON file (`sessionprep.json`) for applying fader offsets.
In dry-run mode (no `-x`), it prints a session overview to the terminal.

Applying the offsets preserves the incoming file-level balance (rough-mix
balance when multitracks were printed post-fader) while keeping the prepared
levels ahead of your inserts.

- **Pro Tools** — **DAW Transfer** imports the audio and applies static Clip
  Gain, Track/Bus Faders, and validated Slot A Trim plugin adjustments in a
  single atomic post-close `.ptx` pass (see [§9](#9-pro-tools-transfer-automation-ptx--ptsl));
  only legacy PTSL volume-automation breakpoint writes remain disabled.
- **DAWproject** — fader volumes are written directly into the generated
  `.dawproject` file.

The `sessionprep.json` export remains available for custom workflows and
third-party tools.

In JSON schema `2.0`, use only tracks with `processing.status == "written"` for delivery automation. Planned and failed tracks have null applied gain and fader values. Each written track records the repairs and gain applied in that execution. Manual trim with normalization bypassed carries a 0 dB DAW fader offset.

---

## 6. GUI Waveform Controls

### 6.1 Mouse

| Action                   | Effect                                                                          |
|--------------------------|---------------------------------------------------------------------------------|
| **Click**                | Set playback cursor position                                                    |
| **Hover**                | Crosshair guide with dBFS readout (waveform) or frequency readout (spectrogram) |
| **Ctrl + wheel**         | Horizontal zoom (centered on pointer)                                           |
| **Ctrl + Shift + wheel** | Vertical zoom (amplitude in waveform, frequency range in spectrogram)           |
| **Shift + Alt + wheel**  | Scroll up / down (frequency pan, spectrogram mode)                              |
| **Shift + wheel**        | Scroll left / right                                                             |

### 6.2 Keyboard Shortcuts

| Key   | Effect                                                        |
|-------|---------------------------------------------------------------|
| **R** | Zoom in (centered on mouse guide, or cursor if not hovering)  |
| **T** | Zoom out (centered on mouse guide, or cursor if not hovering) |

> The waveform must have keyboard focus (click it first) for keyboard
> shortcuts to work.

### 6.3 Preview Header and View Settings

In Phase 2, **Original** and **Corrected** select the routed source or its repaired audio before normalization and output gain. The filled segment identifies the active source. Left and Right switch at the current sample, and Space activates the focused choice. A 10 ms crossfade follows each selection, including a reversal during a fade. Starting, stopping and seeking also use short fades. The common monitor pad keeps device samples within full scale for both choices. Direct playback in Phase 1 also attenuates over-range samples.

Full, Low (0–100 Hz) and High (20 kHz to Nyquist) share a sine-peak dBFS color scale. The display measures both ends of each band, including at pull-up and pull-down sample rates. Near DC and Nyquist, adjacent audio can confirm the frequency of a stationary tone that spans less than a cycle in the measurement window. When that frequency is uncertain, the display uses the observed local spectrum. Each screen cell keeps its strongest measured event as you resize or zoom. The RMS peak marker and curve share the same sample positions. For clips shorter than the RMS window, the waveform reports the RMS of the available samples.

**Waveform preview** is a local disclosure in both Track Layout and File detail.
Track Layout starts collapsed. The header stays visible while the preview is
hidden at the bottom of the pane, releasing the preview's space to the content
above. Reopening preserves its waveform, cursor and zoom, and restores the
splitter allocation proportionally after window resizing.
The chevron points up to open and down to close. Mouse clicks do not leave a
selection outline; Tab/Backtab and keyboard activation show a blue focus outline.
Fit, horizontal zoom and vertical scale use named icon buttons in this header.
Display settings and playback controls remain available when expanded.
 Buttons

| Button                       | Effect                                                                                            |
|------------------------------|---------------------------------------------------------------------------------------------------|
| **Waveform / Spectrogram ▾** | Switch between waveform and spectrogram display mode                                              |
| **Display ▾**                | Spectrogram settings: FFT Size, Window, Color Theme, dB Floor, dB Ceiling (spectrogram mode only) |
| **Detector Overlays ▾**      | Toggle visibility of individual detector overlays (both modes)                                    |
| **Peak / RMS Max**           | Toggle peak ("P") and max-RMS ("R") markers (waveform mode, off by default)                       |
| **RMS L/R**                  | Toggle per-channel RMS envelope (yellow, waveform mode)                                           |
| **RMS AVG**                  | Toggle combined RMS envelope (orange, waveform mode)                                              |
| **Fit waveform**             | Reset zoom to show entire file                                                                    |
| **Zoom in at cursor**        | Zoom in at cursor                                                                                 |
| **Zoom out at cursor**       | Zoom out at cursor                                                                                |
| **Scale up**                 | Scale up (amplitude in waveform, frequency range in spectrogram)                                  |
| **Scale down**               | Scale down (amplitude in waveform, frequency range in spectrogram)                                |

### 6.4 Playback Controls

| Button     | Effect                                                                                                                                                       |
|------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **▶ Play** | Start playback from cursor position                                                                                                                          |
| **■ Stop** | Stop playback and return cursor to start position                                                                                                            |
| **M**      | Toggle mono playback — folds stereo to mono via (L+R)/2 for auditioning mono compatibility. Latched: toggle once, then play/stop freely. Orange when active. |
| **Space**  | Toggle play/stop (global shortcut)                                                                                                                           |

---

## 7. GUI Track Table Controls

**Summary** separates file health counts from finding counts and gives each
finding an affected file, next step and **Review file** action. Review selects
the corresponding file even after sorting the track table.

In **File**, detector rows start with severity before the finding chevron,
followed by available locations and guidance. Click the finding to expand its
full result independently of the **Actions** beside it. **Visualize** is offered
when a finding has valid waveform locations. It selects only that detector's
overlays, expands the preview and fits the whole file; RMS curves, Peak/RMS Max
markers, the report and any open Help remain unchanged. Manual choices in
**Detector Overlays** still allow multiple detectors. If audio is still loading,
only the latest Visualize request for the current file is applied. If audio is
unavailable, the status bar explains that selecting the file again retries loading.

**Help** opens the complete explanation in a
right-side panel. Its title and **X** stay visible while the explanation scrolls
separately from the report. Another Help link switches the explanation and starts
it at the top; reopening the same explanation keeps its reading position.
Opening Help focuses its reading area. Tab/Backtab can reach the centered **X**
with a visible keyboard focus outline.
Close with **X**, or **Esc** while Help has focus, to restore report width and
return focus to the caller while retaining the report's reading position.
Changing file/session or removing the active finding closes Help. Long findings
reflow at narrower widths. The waveform keeps its own persistent header outside
the report. These controls change presentation; they do not repair audio or
alter analysis.

### 7.1 Analysis Column

The **Analysis** column displays per-track severity counts instead of a single
worst-severity label.  Each severity level is color-coded and only non-zero
counts are shown:

| Display    | Meaning                                                      |
|------------|--------------------------------------------------------------|
| `2P 1A 5I` | 2 Problems (red), 1 Attention (orange), 5 Information (blue) |
| `1A`       | 1 Attention only                                             |
| `OK`       | All detectors clean (green)                                  |
| `Error`    | File could not be read (red)                                 |

The column is sortable — tracks with problems sort first, then attention, then
clean.  Within the same worst severity, tracks with more total issues sort
higher.

Edit **Gain** directly or use its up/down buttons in 0.1 dB steps. The gray
numeric editor matches the adjacent table selectors; a blue border identifies
focus. Disabled Gain values (for example, **Skip**) remain muted, including in
selected rows. The mouse wheel scrolls the table without changing the value.

### 7.2 Selection

Standard Extended Selection applies to the track table:

| Action            | Effect                                          |
|-------------------|-------------------------------------------------|
| **Click**         | Select single row                               |
| **Shift + click** | Extend selection to contiguous range            |
| **Ctrl + click**  | Toggle individual rows (non-adjacent selection) |

### 7.3 Batch Editing

Hold **Alt+Shift** and click a dropdown in any selected row to apply the
chosen value to **all** selected rows.  This mirrors the Pro Tools convention
where Alt-clicking a control applies it across the track selection.

- The multi-row selection is preserved while the dropdown is open (the table
  overrides Qt's default behaviour of clearing the selection on cell-widget
  focus).
- Re-selecting the same value that is already shown still triggers the batch
  action — useful for normalising all selected rows to a common setting.
- Re-analysis of affected tracks runs **asynchronously** with a progress bar.
  The selection is restored after re-analysis completes, even if sorting
  reorders the rows.

Supported batch dropdowns:

| Dropdown           | Column | Effect                                                                            |
|--------------------|--------|-----------------------------------------------------------------------------------|
| **RMS Anchor**     | 5      | Override per-track RMS anchor; triggers full re-analysis (detectors + processors) |
| **Mode**           | 3      | Override per-track normalization mode (`Auto`, `Peak`, `RMS`, `Skip`)             |

### 7.4 RMS Anchor Override

Per-track dropdown overriding the global `rms_anchor` analysis setting.

| Label   | Override value | Meaning                                  |
|---------|----------------|------------------------------------------|
| Default | *(none)*       | Use the session's effective RMS anchor setting |
| Max     | `max`          | Loudest gated RMS window                 |
| P99     | `p99`          | 99th percentile of gated RMS windows     |
| P95     | `p95`          | 95th percentile (default global setting) |
| P90     | `p90`          | 90th percentile                          |
| P85     | `p85`          | 85th percentile                          |

Changing the anchor re-runs all detectors and processors for the affected
track(s), since the anchor value influences both tail exceedance detection and
gain calculation.

### 7.5 Mode Override (`Auto`, `Peak`, `RMS`, `Skip`)

Per-track **Mode** dropdown in Phase 2 controlling how Peak-Capped RMS normalizes each track (manual overrides display a blue change marker; grouped tracks limited by a sibling display `Auto (Grp)`, `Peak (Grp)`, or `RMS (Grp)` with a tooltip naming the governing track):

| Label       | Effect                                                                               |
|-------------|--------------------------------------------------------------------------------------|
| Auto        | Peak-Capped RMS (`Auto (RMS)` or `Auto (Peak)`, or `Noise Floor` at `≤ −70.0 dBFS`)  |
| Peak        | Ignore `target_rms` and normalize directly to `target_peak` (capped by `max_gain_db`) |
| RMS         | Ignore `target_peak` and normalize to `target_rms` (capped by `max_gain_db`, with lazy `dBTP` warnings) |
| Skip        | Exclude track from normalization (bit-exact `0.0 dB` gain, `0.0 dB` fader offset)     |

Changing the Mode re-runs processors only (no detector re-analysis needed) and clears any prior manual gain override on that track.


## 8. Pro Tools Utils Window

Pro Tools Utils uses a narrower initial width on ultrawide displays (a full-screen
aspect ratio of at least 2:1). Standard displays retain the existing initial width.
The window remains centered and resizable; its height continues to fit the active tool.

---

## 9. Pro Tools & DAWproject Transfer Automation (`.ptx` / PTSL / `.dawproject`)

When creating a Pro Tools session from a template, SessionPrep combines PTSL
commands with validated, atomic `.ptx` session-file updates (applied in a single
post-close pass with automatic rollback if Pro Tools fails to reopen the session):

- **Template Paths, Offline Fetch & Transfer Mode (`protools_transfer_backend`, default `offline_ptx`):**
  Each entry in **Pro Tools Templates** accepts a `Template Group` + `Template Name`
  (auto-resolving `~/Documents/Pro Tools/Session Templates/<group>/<name>.ptxt` on
  Windows and macOS) and an optional explicit `.ptxt` file path (working on Windows,
  macOS, Linux, and custom template folders). When a valid local `.ptxt` file is
  present, Phase 3 **Fetch** reads folder topology offline in milliseconds without
  launching Pro Tools. **Transfer Mode** defaults to **Offline .ptx Writer**
  (`offline_ptx`), with **Live PTSL + .ptx Hybrid** (`ptsl_hybrid`) available as an
  option. Offline mode disables PTSL connection controls in Preferences, validates local template/project
  readiness offline in Phase 3 without PTSL prompts, stages BWF/UMID-linked WAVs into
  `<project_dir>/<session_name>/Audio Files/` (resampling/converting if needed), and
  writes complete Pro Tools `.ptx` sessions directly from `.ptxt`/`.ptx` templates
  (including templates with pre-existing audio tracks, clips, and playlists) —
  including folder placement, `No Input`, Routing Folder bus output assignment, track
  colors, whole-file clips spotted at sample 0, Clip Gain, Slot A Trim, Track/Bus
  Faders, VCA-linked Folder Groups, and empty folder/VCA/orphaned Aux cleanup — with
  no running Pro Tools instance required.
- **Automatic Folder Output Routing (`protools_auto_route_folders`, default `true`), Empty Folder/VCA & Orphaned Aux Cleanup (`protools_remove_unused_aux_tracks`, default `true`), and Audio Track `No Input` Clearing:**
  Routes each newly created audio track's Main Output to its parent Routing
  Folder's input bus (walking up through nested Basic Folders when necessary),
  removes empty template folders, unused pre-created VCAs (including cascading
  super-VCAs that no longer control any surviving tracks), and — when
  `protools_remove_unused_aux_tracks` is enabled — causally orphaned downstream
  Aux busses whose upstream feeders were all pruned (plus any folders/VCAs
  emptied as a result, while keeping FX return busses, live parallel auxes,
  hardware-input auxes, active bus folders, and the Mix Bus chain intact),
  and clears each newly created audio track's auto-assigned hardware input to
  `No Input` during the atomic post-close `.ptx` pass. DAWproject export
  likewise prunes empty template folder tracks (and their dangling lanes) while
  preserving folders with pre-existing tracks, externally routed busses, and
  standalone submix/master channels.
- **Two-Level Gain Staging (`protools_gain_mode`, default `clip_gain_and_fader`):**
  - `clip_gain_and_fader`: On un-normalized source audio, writes Peak-Capped RMS
    normalization gain (`+gain_db`) to Pro Tools Clip Gain (pre-insert, mono and
    stereo `.L`/`.R` clips) and the inverse offset (`fader_offset`) to the Track
    Fader (post-insert), preserving incoming file-level balance without requiring
    gain-baked WAV files. On already-prepared WAVs, Clip Gain stays `0.0 dB` so
    gain is never double-applied (displayed as `0.0 dB (baked)` in Phase 3).
  - `clip_gain_unity`: Writes `+gain_db` to Clip Gain on un-normalized audio and
    leaves Track Faders at `0.0 dB`.
  - `fader_only`: Leaves Clip Gain at `0.0 dB` and applies `fader_offset` only
    when transferring gain-baked prepared WAVs.
  - `none`: Leaves both Clip Gain and Track Faders untouched.
- **Automatic Per-Folder Groups & Template VCA Linking (`protools_auto_group_folders`, `protools_link_template_vcas`, defaults `true`; `protools_group_type`, default `Mix/Edit`):**
  Creates or updates a Mix/Edit group per populated folder and links each group
  to its matching template VCA master (supporting up to 104 groups across all
  four Pro Tools banks).
- **Signal-Flow Bus Gain Staging, Slot A Trim Plugins & Phase 3 Inspector (`protools_stage_bus_faders`, default `true`; `protools_pan_law_db`, default `-3.0`):**
  Reconstructs the template's audio routing graph, sums the time-aligned,
  gain-adjusted audio tracks in topological order to measure summed Peak dBFS,
  active RMS dBFS, and the governing Peak-Capped RMS constraint (**Governs**:
  `Peak` / `RMS` on staged busses, `—` on non-staged summing folders and
  `Master`). When template busses carry a validated Insert Slot A gain-trim
  plugin (**DMG Audio TrackControl**, **Avid Trim** in default `+6 dB` mode, or
  **Process.Audio Protility**), SessionPrep applies recursive multi-stage
  Peak-Capped RMS gain staging: pure summing folders stay at `0.0 dB` unity, intermediate
  insert-bearing busses receive pre-insert Slot A Trim gain (`+g`, activating an
  inactive Slot A Trim automatically while respecting bypassed inserts) paired
  with an inverse post-insert Bus Fader offset (`-g` relative to the template
  fader) so downstream analog-emulated insert chains hit their initial linear
  target sweet spot while the incoming file-level balance stays intact across
  unequal bus depths (or Slot A Trim only with unity faders in
  `clip_gain_unity` mode), and the terminal convergence Mix Bus applies a
  single attenuate-only headroom trim protecting Master while keeping its fader
  at unity (sparse arrangements below target are left untouched at `0.0 dB`
  rather than boosted). Bus levels and Slot A trims are **initial linear
  estimates** computed assuming unity gain for unsimulated non-trim inserts,
  initial fader positions, and excluded post-fader sends; they serve as a
  calibrated starting point before mix-time fader moves or insert gain changes.
  If a required staged bus (such as the terminal Mix Bus or an active-insert
  intermediate Aux needing non-zero staging) has an empty Insert Slot A,
  SessionPrep automatically inserts the template's most-used clean
  matching-format (`Mono`/`Stereo`) Slot A Trim plugin; if Slot A is occupied
  by a non-trim or unvalidated plugin, SessionPrep stops with an error rather
  than silently falling back to post-insert fader staging (post-terminal serial
  Mix Busses, quiet `0.0 dB` busses, and `Master` are exempt). On templates or
  branches without Slot A Trim plugins, SessionPrep stages Routing Folders and
  multi-input Subgroup/Stem Auxes via bus faders while leaving single-input
  insert Auxes, Master Faders, and `-∞ dB` parallel auxes untouched. In the GUI
  Phase 3 right pane, **Routing & Folders** previews each folder's input bus,
  planned group/VCA, and output routing chain directly from the `.ptxt`
  template, while the **Signal Flow** tab provides both a calculation **Table**
  and an auto-routed **Graph** DAG view with path-equation tracing, a
  balance-validation banner (reporting path spread, global leaf-fader
  convergence shifts, pre-fader send caveats, and unsimulated active inserts),
  per-bus manual fader overrides (`Reset Overrides` or per-row reset), and
  exports the annotated graph (including `recommended_slot_a_trims`,
  `balance_validation`, and `leaf_fader_shift_db`) under `daw_signal_flow` in
  JSON reports.
- **DAWproject Mix Templates, VST3 Slot 1 Trim & Signal Flow (`dawproject_gain_mode`, `dawproject_insert_track_trim`, `dawproject_apply_trim_gain`, `dawproject_stage_bus_faders`, `dawproject_pan_law_db`, `dawproject_remove_unused_buses`):**
  DAWproject always exports atomically from a pre-existing `.dawproject` template
  (preserving the template untouched). In Phase 3, **Routing & Folders** shows each
  folder's bus channel, VST3 Slot 1 Trim donor status (excluding VST2 plugins), and
  output routing chain, while **Signal Flow** provides the same **Table** and
  **Graph** inspector capped at each template's `fader_ceiling_db` (default `+6.0 dB`).
  Verified VST3 Trim donors (**DMG Audio TrackControl** / **Process.Audio Protility**)
  clone into Slot 1 at unity (`0.0 dB`) by default; experimental `.vstpreset` gain
  patching (`dawproject_apply_trim_gain`, default `false`) stays opt-in until
  calibrated against your target DAW.
