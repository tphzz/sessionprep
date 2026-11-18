"""Audio playback controller using sounddevice."""

from __future__ import annotations

import sounddevice as sd
import numpy as np

from PySide6.QtCore import QObject, Qt, QTimer, Signal, Slot


def _route(block, out, mode, channel):
    """Write a block into the device buffer, using the same routing on both sides."""
    data = block.reshape(len(block), -1)
    channels = data.shape[1]
    if channel is not None and 0 <= channel < channels:
        if mode == "channel_mono":
            out[:, 0] = data[:, channel]
            return
        if mode == "channel_as_is":
            out.fill(0)
            if channels <= 2:
                out[:, channel] = data[:, channel]
            else:
                side = channel % 2
                out[:, side] = data[:, channel] / data[:, side::2].shape[1]
            return
    if mode == "mono":
        np.mean(data, axis=1, out=out[:, 0])
    elif channels > 2:
        np.mean(data[:, ::2], axis=1, out=out[:, 0])
        np.mean(data[:, 1::2], axis=1, out=out[:, 1])
    else:
        out[:] = data


class PlaybackController(QObject):
    """Manages audio playback state and sounddevice OutputStream lifecycle.

    Signals:
        cursor_updated(int): Emitted ~30fps with the current sample position.
        playback_finished(): Emitted when playback reaches the end of audio.
        error(str): Emitted on playback errors.
    """

    cursor_updated = Signal(int)
    playback_finished = Signal()
    error = Signal(str)
    _stream_finished = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stream: sd.OutputStream | None = None
        self._play_start_sample: int = 0
        self._play_frame_count: list[int] = [0]
        self._generation = 0
        self._corrected = False
        self._stream_finished.connect(self._on_finished_main, Qt.QueuedConnection)

        self._timer = QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._on_timer)

    @property
    def is_playing(self) -> bool:
        return self._stream is not None and self._stream.active

    @property
    def play_start_sample(self) -> int:
        return self._play_start_sample

    def play(self, audio_data, samplerate: int, start_sample: int = 0,
             mode: str = "as_is", channel: int | None = None, *, audition=None,
             corrected: bool = False):
        # pylint: disable=too-many-positional-arguments
        """Start playback from the given sample position.

        Parameters
        ----------
        mode : str
            ``"as_is"`` — play all channels unchanged.
            ``"mono"`` — fold all channels to mono (sum / N).
            ``"channel_as_is"`` — solo *channel* in its original speaker
            position (other channels silenced).
            ``"channel_mono"`` — extract *channel* as a 1-channel stream
            routed equally to all speakers.
        channel : int or None
            Channel index for ``channel_as_is`` / ``channel_mono`` modes.
        """
        self.stop()

        if audio_data is None or audio_data.size == 0:
            return

        audio = audio_data
        n_ch = audio.shape[1] if audio.ndim == 2 else 1
        if channel is not None and not 0 <= channel < n_ch:
            channel = None
        mono = mode == "mono" or (mode == "channel_mono" and channel is not None)
        output_channels = 1 if mono else min(n_ch, 2)

        if start_sample >= audio.shape[0]:
            start_sample = 0

        self._play_start_sample = start_sample
        frame_count = self._play_frame_count = [0]
        generation = self._generation
        self._corrected = corrected
        target = corrected
        fade_length = max(2, round(.01 * samplerate))
        elapsed = fade_length
        monitor_scale = 10 ** (audition.monitor_gain_db / 20) if audition else 1.

        def callback(outdata, frames, time_info, status):
            nonlocal target, elapsed
            pos = start_sample + frame_count[0]
            count = min(frames, len(audio) - pos)
            if count:
                selected = self._corrected
                if audition and selected != target:
                    # A reversal starts at the last blend heard, even mid-fade.
                    elapsed = max(0, fade_length - elapsed)
                    target = selected
                if audition and elapsed < fade_length:
                    block = audition.switch_samples(pos, not target, target, count, fade_offset=elapsed)
                    elapsed = min(fade_length, elapsed + count)
                    _route(block, outdata[:count], mode, channel)
                else:
                    source = audition.corrected if audition and target else audio
                    block = source[pos:pos + count]
                    if monitor_scale != 1.:
                        block = np.multiply(block, monitor_scale, dtype=np.float64)
                    _route(block, outdata[:count], mode, channel)
                frame_count[0] += count
            if count < frames:
                outdata[count:] = 0
                raise sd.CallbackStop()

        def finished():
            self._stream_finished.emit(generation)

        try:
            self._stream = sd.OutputStream(
                samplerate=samplerate,
                channels=output_channels,
                callback=callback,
                finished_callback=finished,
            )
            self._stream.start()
            self._timer.start()
        except Exception as e:
            self.stop()
            self.error.emit(str(e))

    def set_audition_source(self, corrected: bool):
        self._corrected = corrected

    def stop(self):
        """Stop playback. Returns the start sample for cursor restoration."""
        self._timer.stop()
        self._generation += 1
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def current_sample(self) -> int:
        """Return the current playback sample position."""
        return self._play_start_sample + self._play_frame_count[0]

    @Slot(int)
    def _on_finished_main(self, generation):
        """Handle playback completion on the main thread."""
        if generation != self._generation:
            return
        self.stop()
        self.playback_finished.emit()

    @Slot()
    def _on_timer(self):
        """Emit cursor position updates during playback."""
        if self._stream is not None:
            pos = self._play_start_sample + self._play_frame_count[0]
            self.cursor_updated.emit(pos)
