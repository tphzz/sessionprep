"""Sample-aligned file monitoring with bounded storage for device callbacks."""
from functools import lru_cache
import math

import numpy as np


_DEVICE_MIN_NORMAL = float(np.finfo(np.float32).tiny)


class MonitorPad:
    """Common peak-based attenuation for both sources before device conversion."""

    def __init__(self, *sources):
        peak = 0.
        for source in sources:
            low = float(np.min(source, initial=0.))
            high = float(np.max(source, initial=0.))
            if not (math.isfinite(low) and math.isfinite(high)):
                raise ValueError("Preview contains non-finite samples.")
            peak = max(peak, -low, high)
        self.peak = peak
        self.decibels = -math.ceil(200 * math.log10(peak)) / 10 if peak > 1. else 0.
        # Half the dB attenuation applied twice stays normal even for a
        # maximum-finite DOUBLE source on a device that flushes denormals.
        self.half_gain = 10 ** (self.decibels / 40)

    def apply(self, samples, out=None):
        if out is None:
            out = np.array(samples, dtype=np.float64, copy=True)
        else:
            np.copyto(out, samples)
        if self.half_gain != 1.:
            out *= self.half_gain
            out *= self.half_gain
        return out


@lru_cache(maxsize=16)
def monitor_fade(sample_rate):
    """Unit step of a 10 ms Blackman-windowed smoothing kernel."""
    if not math.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("Sample rate must be positive and finite.")
    t = np.linspace(0., 1., max(2, round(.01 * sample_rate)))
    phase = (2 * np.pi) * t
    curve = t - (.5 * np.sin(phase) - .04 * np.sin(2 * phase)) / (.42 * 2 * np.pi)
    curve[0], curve[-1] = 0., 1.
    curve.setflags(write=False)
    return curve


def _routes(channels, mode, channel):
    """Input lanes and mixing divisor for each device output."""
    if channel is not None and 0 <= channel < channels:
        if mode == "channel_mono":
            return [((channel,), 1)]
        if mode == "channel_as_is":
            return [((channel,) if side == channel % 2 else (), len(range(side, channels, 2)))
                    for side in range(min(2, channels))]
    if mode == "mono":
        return [(range(channels), channels)]
    return [(range(side, channels, 2), len(range(side, channels, 2)))
            for side in range(min(2, channels))]


class MonitorPlayback:
    """One stream's audio state. Only render() advances the playhead.

    The UI sets selected and stopping. The device thread reads them at block
    boundaries. All audio scratch is allocated before the device opens.
    """

    _CHUNK = 256

    def __init__(self, original, rate, start=0, mode="as_is", channel=None, *,
                 audition=None, corrected=False):
        source = audition.original if audition is not None else original
        self.original = source.reshape(len(source), -1)
        self.corrected = (audition.corrected.reshape(self.original.shape)
                          if audition is not None else self.original)
        self.sample_rate = rate
        self.start = start if 0 <= start < len(source) else 0
        self.position = self.start
        self.selected = self._target = corrected
        self.stopping = False
        self._stop_offset = 0
        self._routes = _routes(self.original.shape[1], mode, channel)
        self.channels = len(self._routes)
        self.monitor_pad = audition.monitor_pad if audition is not None else MonitorPad(source)
        self._curve = monitor_fade(rate)
        self._release = self._curve[::-1]
        self._blend = np.empty_like(self._curve)
        self._next_blend = np.empty_like(self._curve)
        self._blend_offset = len(self._curve)
        self._lane, self._other, self._addend = np.empty((3, self._CHUNK))
        self._invalid = np.empty(self._CHUNK, dtype=bool)

    def _select(self, target):
        # Superpose each target step on the remaining response. This is a
        # convolution of 0/1 selections with a positive unity-sum kernel:
        # arbitrarily fast reversals retain the preceding slope and timing.
        length = len(self._curve)
        remaining = length - self._blend_offset
        self._next_blend[:remaining] = self._blend[self._blend_offset:]
        self._next_blend[remaining:] = self._target
        if target:
            np.add(self._next_blend, self._curve, out=self._next_blend)
        else:
            np.subtract(self._next_blend, self._curve, out=self._next_blend)
        self._blend, self._next_blend = self._next_blend, self._blend
        self._blend_offset = 0
        self._target = target

    def _route(self, source, indices, divisor, lane):
        if not indices:
            lane.fill(0.)
            return
        stop = self.position + len(lane)
        self.monitor_pad.apply(source[self.position:stop, indices[0]], out=lane)
        for index in indices[1:]:
            addend = self._addend[:len(lane)]
            self.monitor_pad.apply(source[self.position:stop, index], out=addend)
            np.add(lane, addend, out=lane)
        if divisor > 1:
            lane /= divisor

    def render(self, out):
        """Fill the entire device buffer, returning True after the last frame."""
        target = self.selected
        if target != self._target:
            self._select(target)
        stopping = self.stopping
        length = len(self._curve)
        available = min(len(out), len(self.original) - self.position)
        if stopping:
            available = min(available, length - self._stop_offset)
        written = 0
        while written < available:
            count = min(self._CHUNK, available - written)
            blending = self._blend_offset < length
            if blending:
                count = min(count, length - self._blend_offset)
                weights = self._blend[self._blend_offset:self._blend_offset + count]
            elapsed = self.position - self.start
            head = min(count, max(0, length - elapsed))
            remaining = len(self.original) - (self.position + count)
            tail = min(count, max(0, length - remaining))
            lane = self._lane[:count]
            for side, (indices, divisor) in enumerate(self._routes):
                if blending:
                    other = self._other[:count]
                    self._route(self.original, indices, divisor, lane)
                    self._route(self.corrected, indices, divisor, other)
                    np.subtract(other, lane, out=other)
                    np.multiply(other, weights, out=other)
                    np.add(lane, other, out=lane)
                else:
                    self._route(self.corrected if target else self.original, indices, divisor, lane)
                if head:
                    lane[:head] *= self._curve[elapsed:elapsed + head]
                if tail:
                    lane[-tail:] *= self._curve[remaining:remaining + tail][::-1]
                if stopping:
                    lane *= self._release[self._stop_offset:self._stop_offset + count]
                # Guard the device even if a source is inadvertently mutated
                # after preflight. Never narrow a non-finite sample to float32.
                invalid = self._invalid[:count]
                np.isfinite(lane, out=invalid)
                np.logical_not(invalid, out=invalid)
                np.copyto(lane, 0., where=invalid)
                np.maximum(lane, -1., out=lane)
                np.minimum(lane, 1., out=lane)
                # Values below the float32 normal range are silent at the DAC.
                # Flush in float64, before the device conversion can denormalize.
                magnitude = self._addend[:count]
                np.absolute(lane, out=magnitude)
                np.less(magnitude, _DEVICE_MIN_NORMAL, out=invalid)
                np.copyto(lane, 0., where=invalid)
                out[written:written + count, side] = lane
            written += count
            self.position += count
            if blending:
                self._blend_offset += count
            if stopping:
                self._stop_offset += count
        out[written:] = 0.
        return self.position == len(self.original) or (stopping and self._stop_offset == length)
