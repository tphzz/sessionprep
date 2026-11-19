"""Sample-aligned, unity-gain audition of whole-track repairs."""
from __future__ import annotations

import numpy as np

from .monitor import MonitorPad, monitor_fade
from .repairs import RepairRecipe, render_corrected
from .spectrum import inspection_spectrogram as inspection_spectrogram


class RepairAudition:
    """The original and corrected share one time base and one monitor gain."""

    def __init__(self, original: np.ndarray, sample_rate: int, recipe: RepairRecipe):
        self.original = np.asarray(original)
        self.corrected = render_corrected(self.original, sample_rate, recipe)
        self.sample_rate = sample_rate
        self.monitor_pad = MonitorPad(self.original, self.corrected)

    @property
    def monitor_gain_db(self):
        return self.monitor_pad.decibels

    def playback_samples(self, corrected: bool) -> np.ndarray:
        audio = self.corrected if corrected else self.original
        return audio if self.monitor_gain_db == 0 else self.monitor_pad.apply(audio)

    def switch_samples(self, sample: int, from_corrected: bool, to_corrected: bool,
                       frames: int, *, fade_offset: int = 0) -> np.ndarray:
        """Return a sample-aligned excerpt of an uninterrupted source transition."""
        if min(sample, frames, fade_offset) < 0:
            raise ValueError("Playback position, frame count and fade offset must be nonnegative")
        dest = self.corrected if to_corrected else self.original
        result = self.monitor_pad.apply(dest[sample:sample + frames])
        if from_corrected != to_corrected:
            curve = monitor_fade(self.sample_rate)
            wet = curve[fade_offset:fade_offset + len(result)]
            count = len(wet)
            if result.ndim == 2:
                wet = wet[:, None]
            start = self.corrected if from_corrected else self.original
            dry = self.monitor_pad.apply(start[sample:sample + count])
            result[:count] = dry * (1 - wet) + result[:count] * wet
        return result
