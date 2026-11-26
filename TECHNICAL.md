# SessionPrep — Technical Background

This document explains the audio engineering concepts behind SessionPrep:
why it exists, what problem it solves, and how its normalization strategy works.
Some of the reasoning below is opinionated and reflects one practical approach
to session preparation — not the only valid one.

For detector and analysis reference, see [REFERENCE.md](REFERENCE.md).
For development setup and architecture, see [DEVELOPMENT.md](DEVELOPMENT.md).

---

## Table of Contents

1. [Physics vs. Flow](#1-physics-vs-flow)
2. [Peak-Capped RMS Normalization](#2-peak-capped-rms-normalization)
3. [Signal Chain & Workflow](#3-signal-chain--workflow)
4. [Why This Is Not Circular Logic](#4-why-this-is-not-circular-logic)
5. [What Happens Next (The Manual Mix Phase)](#5-what-happens-next-the-manual-mix-phase)
6. [Core Concepts & Practical Notes](#6-core-concepts--practical-notes)

---

## 1. Physics vs. Flow

In a professional mixing environment, the engineer is trapped between two
opposing requirements: the (possibly) rigorous physics of software emulation and the
delicate artistic balance of the rough mix.

### Goal A: The Physics (Predictable headroom and drive)

Some analog modelled and drive-sensitive plugins (channels, tape, transformer/tube
saturation, some compressors) can respond differently depending on input level.

The `-18 dBFS` reference is the default (`--target_rms -18`) but can be adjusted
to match your workflow. It is one conservative starting point when you want
predictable headroom and "sane" starting levels at the top of the channel.
The reference level is ultimately a matter of choice — some engineers calibrate
to `-14 dBFS` or higher to deliberately drive plugins harder and make more use
of their saturation and harmonic characteristics.
Many modern plugins (clean EQs, reverbs, delays, modern dynamics, and many recent
analog emulations) are far more forgiving and/or auto-calibrated, so this is not
presented as required for plugins to work correctly.

Even when a plugin does not "need" this, a conservative starting level is usually
low-risk: it tends to preserve headroom, reduce surprise overs, and keep drive/
threshold controls in a familiar range.

Practical limitation:
  - Insert chains have multiple stages, and each stage can have its own "best"
    operating level. SessionPrep standardizes the level entering the first
    insert on each track — and, for Pro Tools templates, recursively at each
    downstream insert-bearing bus stage (see §3.1) — while you still trim
    between serial plugins inside a single insert chain as needed.

1. **NON-LINEAR RESPONSE:**
   Analog gear (and its software clones like SSL channels, 1176s, Tape Machines)
   does not react linearly to gain.
   - Many designs are calibrated around a nominal reference level
     (often discussed as ~0 VU; the exact dBFS-to-VU mapping varies by
     plugin — commonly -18, -20, or -14 dBFS = 0 VU).
   - As you push past -10 dBFS, transformers saturate, tubes compress, and
     op-amps clip.

2. **THE "CALIBRATION GAP":**
   If a producer sends you a modern synth track peaking at -0.1 dBFS, and you
   insert a modeled Neve EQ:
   - Some older or more strictly calibrated models may "hear" a signal much
     hotter than the hardware was designed for.
   - Result: The tone/dynamics may change in a way you did not intend.
   - The Fix: Use the plugin input trim (or a trim plugin) where needed.

3. **THE DYNAMIC BEHAVIOR:**
   Compressor attack and release curves are often program-dependent. Feeding
   a compressor a signal 20dB lower than expected results in sluggish detection
   and weak, transparent compression instead of the "grab" you expect.

### Goal B: The Art (The "Rough Mix" Approval)

The client, producer, and A&R have been listening to the "Rough Mix" for weeks.
They have emotionally bonded to the relative balance: the kick is dominating,
the backing vocals are buried, the shaker is loud.

1. **THE "DEMOITIS" DANGER:**
   If you normalize every track to a technical standard (-18 dBFS), you flatten
   the mix. The kick and shaker are now equal volume. The balance is destroyed.

2. **THE PSYCHOLOGICAL COST:**
   If you start your mix with a flat balance, you spend the first 3 hours just
   trying to recreate the vibe the client already had. You are working backward,
   not forward.

### The Conflict & The Solution

The Conflict:
  - If you preserve the file levels (Art), drive-sensitive plugins may exhibit calibration-dependent behavior.
  - If you normalize the file levels (Physics), your balance is lost.

Practical note:
  - A conservative gain staging approach that keeps analog modelled/drive-sensitive
    plugins in a predictable operating range will still work fine with most modern
    wide-headroom plugins. The targets here are chosen to be compatible with both.

The "Manual" Solution (The Slog):
  Professional mixers solve this by manually adjusting Clip Gain on every single
  region to hit the sweet spot, then counter-adjusting the Fader to match the
  rough mix volume. This takes hours of non-creative clicking.

The Script's Solution:
  It follows an analysis-first workflow:
  - Dry-run (default): analyze the session and print a session health/level overview.
  - Execute (`-x`): optionally write processed tracks and export fader offsets.

  In execute mode — or directly in Pro Tools via non-destructive `.ptx` Clip
  Gain + inverse Track Fader staging — it sets a consistent starting level into
  the first insert (Physics) and applies the inverse fader offset to preserve
  the incoming file-level balance (Art — matching the producer's rough mix when
  multitracks were exported post-fader; when tracks were printed pre-fader at
  unity, inverse faders simply reproduce a raw unity import while track inserts
  still receive calibrated levels). On Pro Tools templates with bus routing, it
  extends the same two-level principle recursively down the bus hierarchy (see
  §3.1). DAWproject exports embed audio and fader offsets into self-contained
  archives, and `sessionprep.json` remains available for custom workflows.

---

## 2. Peak-Capped RMS Normalization

Most normalizers are "dumb" — they treat a Kick drum and a Synth Pad the same.
If you normalize an acoustic Kick drum to -18 dB RMS, you can severely compromise it (the peak
may exceed 0 dBFS and clip, because the body is so thin relative to the transient).

### Calibration Note

`-18 dBFS RMS` is a common *conservative* reference point for gain staging into
analog-modeled / drive-sensitive processing, but it is not universal.

Practical reality:
  - Many modern plugins (FabFilter, Valhalla, most stock DAW plugins) are largely
    headroom-agnostic and behave consistently across a wide range of input levels.
  - Some "analog-modeled" plugins auto-calibrate internally (for example certain
    Plugin Alliance / TMT-style designs), so the exact input level is less critical.
  - Other analog-modeled chains (channel strips, tape, transformers/tubes, some
    compressors) can still be noticeably level-dependent.

For that reason, target levels are intentionally adjustable via CLI or Preferences:
  - `--target_rms` (default `-18.0 dBFS`): operating unweighted RMS target (anchored to gated loud sections)
  - `--target_peak` (default `-6.0 dBFS`): sample-peak ceiling (`target_rms <= target_peak`)
  - `--max_gain_db` (default `+24.0 dB`, range `0–36 dB`): automatic boost cap
  - `--force_peak <regex>`: optional regex for tracks that always target the peak ceiling (e.g. `kick|snare|tom`)

If your chain is mostly modern, wide-headroom plugins, you can run hotter by
raising `--target_rms` (e.g. `-16` or `-14`). If your chain is heavy on
saturation/tape/legacy calibrated plugins, `-18` remains a safe default.

Example (modern/hotter chain):
  - `--target_rms -14 --target_peak -3`

Even with the correct target, edge-case outliers (or section-to-section macro
dynamics) may still require manual clip gain adjustments in the DAW.

### The Continuous Peak-Capped RMS Law

Instead of sorting tracks into discrete buckets with heuristic decay/density
classifiers, SessionPrep applies one continuous, monotonic rule to every track:

```text
gain_db = min(target_rms - rms_anchor_db, target_peak - sample_peak_db, max_gain_db)
```

At the default targets (`-18 dBFS` RMS / `-6 dBFS` Peak), the natural crossover
sits at **12 dB effective crest** (`peak − rms_anchor`):

  - **Peak-governed (`Auto (Peak)`):** High-crest sources (uncompressed kicks,
    snares, percussion, isolated spikes; effective crest ≥ 12 dB) automatically
    stop at `--target_peak` (`-6 dBFS`), preserving clean transient headroom.
    Exact ties within `1e-9 dB` resolve deterministically to `Peak`.
  - **RMS-governed (`Auto (RMS)`):** Sustained instruments (vocals, bass, 808s,
    guitars, synths, pads) *and* dense compressed drum loops (effective crest
    < 12 dB) target `--target_rms` (`-18 dBFS`), delivering predictable electrical
    drive into compressors and saturators without pushing compressed loops 4–6 dB
    over target.
  - **Noise Floor Hold (`≤ −70.0 dBFS` peak):** Silent stems, room tone, and
    16/24-bit dither hold at `0.0 dB` (`Noise Floor`) so quiet tracks never wind up.
  - **Explicit Overrides:** Per-track **Peak** mode (or `--force_peak`) targets
    the peak ceiling when peak-sensitive drum processing (gates, transient
    designers, fast 1176s) expects fixed peak levels; per-track **RMS** mode
    bypasses the peak ceiling (for example, when one isolated plosive caps an
    otherwise quiet vocal) and triggers lazy oversampled true-peak (`dBTP`)
    warnings whenever post-gain sample peak exceeds `-5.5 dBFS`.

### Governing Peak & Next-Highest Peak (`±100 ms`)

To help you immediately distinguish an isolated vocal plosive from a consistently
hot kick drum, SessionPrep reports both the global governing peak (with clickable
timestamp) and the **next-highest distinct peak** outside a stereo-linked
`±100 ms` exclusion window (`—` when no non-zero sample falls outside `±100 ms`).
A large gap (e.g. `-3.6 dBFS` vs next `-12.4 dBFS`) points to a single outlier
worth fixing with clip gain or an `RMS` override; a tight gap confirms that
every hit sits near the ceiling.

---

## 3. Signal Chain & Workflow

This section summarizes the signal flow this tool is designed around in a typical
DAW (Pro Tools, Logic, Cubase, Reaper, Ableton).

This tool is intentionally two-phase:
1) Analysis-first (default): it prints a session overview so you can decide what
   to fix/request before creating the DAW session.
2) Processing second (optional, `-x`): it writes processed tracks and exports
   automation data for restoring the rough balance.

The signal path is not a single volume knob. It is a chain:

```
[SOURCE FILE]
      |
[CLIP GAIN]    <--- STEP 1: OPTIMIZATION (THE SCRIPT)
      |             (Optional, execute mode) We modify the actual file gain
      |             (non-destructively) to hit a chosen target calibration.
      |
[INSERTS]      <--- STEP 2: THE "SWEET SPOT" (THE PHYSICS)
      |             This is where your plugin chain lives.
      |             The first insert receives a predictable input level.
      |             You still gain-stage between inserts as needed.
      |
[FADER]        <--- STEP 3: RESTORATION (THE REPORT)
      |             We counteract the gain change using the Fader.
      |             This restores the Producer's intended balance.
      |             The GUI's DAW Transfer applies these fader offsets
      |             automatically (Pro Tools via PTSL + .ptx patching,
      |             DAWproject via file generation). sessionprep.json remains
      |             for custom workflows.
      |
[MIX BUS]
```

### 3.1 Recursive Multi-Stage Bus Gain Staging (Pro Tools)

Normalizing leaf tracks solves gain structure at the top of individual audio
channels, but summing multiple tracks into subgroups, stems, and a mix bus
shifts the combined level again:

- A quiet two-track bass subgroup can still arrive well below `-18 dBFS RMS`,
  under-driving analog-modeled bus compressors and saturators.
- A dense drum stem or full mix bus summing many channels can easily exceed
  `-6 dBFS peak` or `0 dBFS` pre-fader, over-driving or clipping bus inserts.

Conceptually, this works a bit like **Automatic Delay Compensation (ADC) for
static gain structure**: each track and insert-bearing bus gets the pre-insert
trim its plugins expect (e.g., to reach a target level per bus), paired with an
inverse post-insert fader so every path from leaf track to Mix Bus accumulates
the exact same net linear gain. Two key differences from ADC make offline
host-level execution necessary:

1. **Signed local cancellation:** Unlike latency (which is always positive and
   must be padded on parallel branches), gain in dB can be inverted locally
   (`+g` pre-insert, `-g` at the bus fader = `0.0 dB` net per intermediate bus).
2. **Waveform-dependent summing:** Combined Peak and RMS at a bus depend on
   actual time-domain waveform overlap and phase, not static per-track metadata,
   and standard plugins (even ARA2) have neither visibility into the bus DAG nor
   write access to native Clip Gain and Track/Bus Faders.

For Pro Tools templates, SessionPrep reconstructs the template's directed
acyclic signal-flow graph (Audio → Routing Folders → Subgroup/Stem Auxes →
Mix Bus → Master), sums the time-aligned, gain-adjusted Main Output audio in
topological order (respecting Mono/Stereo pan geometry, mutes, and VCAs), and
applies the same Peak-Capped RMS rule and two-level gain-staging principle
recursively at each bus stage:

```
[SUMMED BUS INPUT]
        |
[SLOT A TRIM (+g)]  <--- Pre-insert gain stage (TrackControl / Avid Trim / Protility)
        |                Calibrates summed signal to Peak-Capped RMS target (-18 RMS / -6 pk cap)
[BUS INSERTS]       <--- Bus compressors, EQs, tape/console models hit nominal drive
        |
[BUS FADER (-g)]    <--- Inverse post-insert offset cancels +g so downstream
        |                balance stays identical across unequal bus depths
[NEXT BUS / MIX BUS]
        |
[TERMINAL SLOT A TRIM (attenuate-only)]  <--- Single headroom trim; Mix Bus fader stays at 0.0 dB
        |
[POST-TERMINAL BUSSES / MASTER]
```

Key invariants:

- **Balance preservation across unequal bus depths & headroom convergence:**
  Because every intermediate bus satisfies
  `(slot_a_trim - active_template_trim) + (bus_fader - template_fader) == 0`
  in balance-preserving modes (`clip_gain_and_fader` / `fader_only` / `none`),
  parallel branches of different depths (for example, a direct vocal bus
  alongside a deeper drum chain) arrive at the terminal Mix Bus with their
  exact relative incoming file-level balance intact (matching the rough mix when
  multitracks were exported post-fader, or the unity tracking sum when exported
  pre-fader). If an intermediate bus would exceed the `+12.0 dB` inverse-fader
  ceiling, the solver applies a shared global leaf-fader shift
  (`leaf_fader_shift_db`) equally across all leaf tracks so both headroom and
  incoming balance converge. In `clip_gain_unity` mode, Slot A trims are applied
  with unity faders.
- **Initial linear estimate & model boundaries:** While algebraic balance
  invariance (`Δ_bus = 0.0 dB`) is exact at session open, bus levels and Slot A
  trims are **initial linear estimates** rather than permanent physical targets:
  they assume unity pass-through for unsimulated active track/bus inserts
  (compressors, saturators, EQs — indicated explicitly in the Signal Flow UI),
  exclude post-fader FX sends, and reflect initial fader positions (moving track
  faders or engaging non-unity plugins during mixing will shift downstream bus
  levels by several dB). Active **pre-fader sends** tap after Clip Gain or
  Slot A Trim and before the inverse fader and are flagged explicitly in the
  Signal Flow balance validator (post-fader sends remain invariant).
- **Pure summing folders vs. insert-bearing busses:** Zero-insert summing
  Routing Folders feeding downstream Slot A trim busses remain at `0.0 dB`
  unity; post-terminal serial Mix Busses (e.g. metering/mastering chains) and
  `Master` remain untouched.
- **Validated Slot A trim plugins & strict safety:** SessionPrep supports
  **DMG Audio TrackControl**, **Avid Trim** (default `+6 dB` mode), and
  **Process.Audio Protility** in Insert Slot A. If a required staged bus needs
  non-zero staging and has an empty Slot A, SessionPrep automatically clones
  the template's most-used clean matching-format (`Mono`/`Stereo`) Slot A trim
  plugin into Slot A; if Slot A is occupied by a non-trim or unvalidated
  plugin, transfer stops with an error rather than silently falling back to a
  post-insert fader. Templates without Slot A trim plugins fall back to bus
  fader staging on Routing Folders and multi-input subgroup Auxes.

---

## 4. Why This Is Not Circular Logic

A common doubt among engineers:
"If you boost the file +4dB (Clip Gain) and cut the fader -4dB, haven't you
just ended up exactly where you started?"

THE ANSWER: NO. You have changed the signal hitting the INSERTS.

Consider a quiet synth pad sitting at -30 dBFS:
1.  **Without this script:** The signal hits your Analog Compressor plugin at
    -30 dB. The threshold knob is at noon, but the compressor may not engage
    meaningfully. The tubes don't saturate. It may sound thin. You have to
    crank the "Input" knob 12dB just to wake it up.
2.  **With this tool (execute mode):** The signal hits the Compressor at a more
    predictable operating level. That can make thresholds/drive controls behave
    more consistently for calibrated or drive-sensitive plugins.
3.  **The Fader:** The fader (applied manually or via automation) restores the
    intended rough balance.

You can end up with the same perceived mix balance, but with a different internal
gain structure feeding the inserts.

---

## 5. What Happens Next (The Manual Mix Phase)

This script does not replace the Mix Engineer. It simply sets the table.
Once you have created a fresh session and imported the prepared tracks (and applied
the fader offsets manually or via automation), the real work begins.

This is where manual clip gain riding still matters: if a verse is 6 dB quieter
than the chorus (or an intro is sparse but the outro is dense), you will still
shape those section-to-section macro dynamics by hand.

You are now in a superior starting position:

1.  **High-Resolution Fader Throws:**
    This tool optimizes the level hitting inserts; the required fader offsets may
    be small or large depending on the source material and the chosen targets.
    If your resulting fader positions end up near Unity (0 dB), you benefit from
    higher-resolution moves around 0 dB. If the offsets push faders far from unity,
    you still gain the primary benefit: predictable insert drive.

2.  **Dynamic Clip Gain (The Human Touch):**
    The script sets a "Global Anchor" (e.g., maximizing the loudest Chorus).
    Now, YOU do the musical work:
    - Grab the quiet Verse regions and clip-gain them up manually.
    - Spot a harsh "S" or pop and clip-gain it down.
    - The script handled the macro-physics; you handle the micro-dynamics.

3.  **Creative Automation:**
    The calculated fader positions are just a static starting point to match
    the rough mix. As you mix, you will (and should) automate faders, push
    into compressors, and change balances. You are simply starting from a
    grid of calibrated, healthy signals rather than a chaotic mess of files.

---

## 6. Core Concepts & Practical Notes

### Unweighted RMS vs. LUFS & Stereo Mode (`avg` vs. `sum`)

LUFS is an excellent standard for broadcast delivery and perceived loudness, but
its K-weighting filter rolls off low bass and boosts upper midrange (~9.5 dB
divergence between 40 Hz and 4 kHz). For gain-staging analog-modeled and
drive-sensitive processing on Pop/Rock/R&B stems (808 sub bass, kick drums,
guitars, vocals), **unweighted momentary RMS** matches electrical voltage drive
directly rather than ear sensitivity.

For stereo tracks, `--stereo_mode avg` (default, half-power mean across L/R)
keeps centered stereo tracks consistent with mono tracks; hard-panned mono
material inside a stereo file reads 3.01 dB lower than in `--stereo_mode sum`
(full-power sum). Both modes are strictly L↔R swap symmetric.

### AES17 Scale Convention (`dbfs_convention`)

`standard` (default) displays full-scale square-wave `0 dBFS` (`-3.01 dBFS` for
a full-scale sine). `aes17` adds `+3.0103 dB` (`10 · log10(2)`) strictly as a
**presentation-only offset on RMS readouts and RMS targets** so a full-scale sine
reads `0.0 dBFS RMS`. Sample Peak, Peak Ceiling, Next-Highest Peak, `dBTP`, clip
warnings, and DC offset always remain unshifted digital dBFS, and toggling
`standard` ↔ `aes17` never alters gains or rendered samples.

### Internal gain structure (why clip gain and fader offsets both exist)

Changing clip gain and applying the inverse fader offset can preserve the
incoming file-level balance (matching the rough mix on post-fader exports) while
still changing the level feeding inserts. This can matter for calibrated plugins
and non-linear stages where drive changes tone/dynamics. It can also improve
fader control resolution in cases where the resulting fader positions end up
closer to unity, but that is not guaranteed.

### Explicit Overrides (`Peak`, `RMS`, `--force_peak`)

Peak-Capped RMS removes heuristic threshold failures, while keeping two targeted
manual overrides when an engineer wants different behavior:
  - **`Peak` / `--force_peak <regex>`**: ignores `--target_rms` and targets
    `--target_peak` directly (useful for compressed drums feeding peak-sensitive
    gates, transient designers, or fast 1176 chains).
  - **`RMS`**: ignores `--target_peak` (still bounded by `--max_gain_db`) to
    prevent one isolated transient or vocal plosive from under-gaining the whole
    track. Post-gain peaks above `-5.5 dBFS` trigger lazy oversampled true-peak
    checks (`amber > -1.0`, `red > 0.0` dBFS/dBTP).

### Multi-mic sources (phase vs non-linearity)

Constant gain does not change time alignment, so phase relationships are
preserved. The practical risk is tonal divergence if each mic is driven into
non-linear per-channel processing differently. If you care about matched behavior,
prefer bus processing, or keep heavy saturation/compression on the group bus.

Track grouping (multi-mic and bundles):
If a set of tracks should behave as a single instrument (e.g. kick in/out/sub,
snare top/bottom, BV stacks), use `--group Name:pattern1,pattern2` so those
files get identical applied gain. This preserves internal balance and avoids
changing how the bundle hits a bus compressor.

Example:
  ```
  --group Kick:kick,kick_sub --group OH:overhead,oh --group Toms:tom
  ```

Patterns support substring, glob (`*`/`?`), or exact match (suffix `$`).
First match wins — if a file matches multiple groups, it is assigned to the
first matching group and a warning is printed.

### Windowing and anchor strategy

Sustained-material gain is computed against a single representative RMS value
called the **anchor**. Finding this anchor is a multi-step process:

1. **Window**: Slice the track into overlapping unweighted short-time RMS windows
   (default 400 ms, `round(window_ms * sr / 1000)` samples) computed via
   cancellation-free block suffix+prefix summation so quiet passages after loud
   sections keep full `float64` precision. Each window produces one momentary
   RMS value. Together they form a distribution of momentary loudness across
   the file.

2. **Gate**: Discard windows that are far below the loudest window
   (`--gate_relative_db`, default 40 dB). This removes silence and very quiet
   passages, leaving only "active" content. Critical for sparse tracks (FX
   hits, vocal doubles, breakdown elements) where most windows are near-silent.

3. **Select anchor**:
   - **`percentile`** (default): Take the Nth percentile of the gated
     distribution (default P95). This represents "what the loud sections
     typically sound like" while ignoring rare spikes — a single anomalous
     window (breath pop, drum bleed, feedback ring) cannot pull the anchor
     away from the track's true working level.
   - **`max`**: Take the single loudest gated window. Useful for very short
     files (single hits, sound effects) but fragile for longer material.

4. **Use**: The anchor drives the sustained gain calculation
   (`gain = target_rms − anchor`, capped by `target_peak − peak`) and defines
   the baseline for tail exceedance reporting.

Why P95 is the default: most real-world tracks have occasional moments louder
than their "working level" (a vocalist leaning in, a dynamic fill, an
arrangement accent). Max anchoring treats those moments as the reference,
under-gaining the rest of the file. Percentile anchoring tracks the chorus-
level loudness that will actually drive your insert processing, and flags the
louder moments as tail exceedances for manual clip-gain attention.

See [REFERENCE.md §3.3](REFERENCE.md) for a detailed walkthrough with examples.

### Tail (significant exceedances)

The tail report is meant to be actionable for a mix engineer, not an exhaustive
dump of every window above the anchor. By default it reports only significant
exceedances above the anchor:
  - `--tail_min_exceed_db` defaults to 3.0 dB
  - `--tail_hop_ms` defaults to 10 ms to reduce noise and region spam
  - `--tail_max_regions` caps the number of regions per file

### True peaks / inter-sample peaks (ISP) and headroom

Under default `Auto` or `Peak` targeting (`--target_peak -6`), signals retain at
least 6 dB of sample-peak margin, so oversampled true-peak measurement is skipped
lazily for speed. Whenever an `RMS` override, manual gain edit, or elevated
ceiling pushes the post-gain sample peak above `-5.5 dBFS`, SessionPrep computes
bounded-memory oversampled true peak (`4×` at `≤ 48 kHz`, `2×` at `< 192 kHz`
including `176.4 kHz`, `1×` at `≥ 192 kHz`, keeping oversampled rate `≥ 192 kHz`)
on the exact post-repair audio:

- **Amber notice:** `post_peak > −1.0 dBFS` or `post_true_peak > −1.0 dBTP`
- **Red warning:** `post_peak > 0.0 dBFS` (over 0 dBFS into first insert; Prepare
  to fixed-point PCM would clip) or `post_true_peak > 0.0 dBTP` (inter-sample
  over into oversampled inserts or DAC reconstruction)

### Pre-session workflow and automation

The default mode is analysis-first: it prints a session overview so you can fix
format issues, clipping, DC offset, correlation concerns, and other problems
before creating a DAW session. Execute mode (`-x`) is optional: it writes processed
tracks and exports `sessionprep.json`. In the GUI, fader offsets are applied
automatically via DAW Transfer (Pro Tools via PTSL + `.ptx` patching,
DAWproject via file generation).

### Heuristic limitation

The momentary RMS anchor (percentile or max) sets a single global reference per
file. It does not replace musical judgment or section-based leveling. Expect to
ride clip gain (or use automation) to manage macro dynamics between song sections.
