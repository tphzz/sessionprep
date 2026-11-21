"""Audio playback controller using sounddevice."""
from __future__ import annotations

import time
from typing import Any

from PySide6.QtCore import QObject, Qt, QTimer, Signal, Slot

from sessionpreplib.monitor import MonitorPlayback


class _LazySoundDevice:
    """Defer importing ``sounddevice`` (and PortAudio initialization) until playback."""

    _module: Any = None

    def _load(self) -> Any:
        if self._module is None:
            import sounddevice as sounddevice_mod
            self._module = sounddevice_mod
        return self._module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._load(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_module":
            super().__setattr__(name, value)
        else:
            setattr(self._load(), name, value)

    def __delattr__(self, name: str) -> None:
        if name == "_module":
            super().__delattr__(name)
        else:
            delattr(self._load(), name)


sd: Any = _LazySoundDevice()


class PlaybackController(QObject):
    """Own the device on the GUI thread. Its callback only renders samples."""

    cursor_updated = Signal(int)
    playback_finished = Signal()
    error = Signal(str)
    _stream_finished = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stream: Any = None
        self._monitor: MonitorPlayback | None = None
        self._pending: MonitorPlayback | None = None
        self._play_start_sample = 0
        self._last_sample = 0
        self._generation = 0
        self._release_deadline = 0.
        self._stream_finished.connect(self._on_finished_main, Qt.QueuedConnection)
        self._timer = QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._on_timer)

    @property
    def is_playing(self) -> bool:
        return (self._pending is not None
                or (self._stream is not None and not self._monitor.stopping and self._stream.active))

    @property
    def play_start_sample(self) -> int:
        return self._play_start_sample

    def play(self, audio_data, samplerate: int, start_sample: int = 0,
             mode: str = "as_is", channel: int | None = None, *, audition=None,
             corrected: bool = False):
        """Play all channels, a mono fold, or a solo channel at the selected cursor."""
        self.stop()
        if audio_data is None or not audio_data.size:
            return
        try:
            monitor = MonitorPlayback(audio_data, samplerate, start_sample, mode, channel,
                                      audition=audition, corrected=corrected)
        except ValueError as exc:
            self.error.emit(str(exc))
            return
        self._play_start_sample = monitor.start
        if self._stream is None:
            self._open(monitor)
        else:
            # An exclusive device must finish the old cursor's release before
            # its replacement acquires the device and attacks at the new cursor.
            self._pending = monitor

    def _open(self, monitor):
        self._generation += 1
        generation = self._generation
        self._monitor = monitor

        def finished():
            self._stream_finished.emit(generation)

        try:
            callback_stop = sd.CallbackStop

            def callback(outdata, frames, time_info, status):
                if monitor.render(outdata):
                    raise callback_stop()

            self._stream = sd.OutputStream(
                samplerate=monitor.sample_rate, channels=monitor.channels,
                dtype="float32", callback=callback, finished_callback=finished)
            self._stream.start()
            self._timer.start()
        except Exception as exc:
            self._close_stream()
            self.error.emit(str(exc))
            return False
        return True

    def set_audition_source(self, corrected: bool):
        monitor = self._pending or self._monitor
        if monitor is not None and not monitor.stopping:
            monitor.selected = corrected

    def stop(self):
        """Request a release on the next callback, draining its samples before close."""
        self._pending = None
        if self._stream is None:
            return
        if not self._monitor.stopping:
            self._monitor.stopping = True
            # A disconnected device may never call back. Give a working stream
            # its release and host latency before reclaiming it on the GUI thread.
            self._release_deadline = time.monotonic() + max(1., 3 * getattr(self._stream, "latency", 0.))
        if not self._stream.active:
            self._close_stream()

    def close(self):
        """Drain during window shutdown when queued Qt completion can no longer run."""
        self.stop()
        while (self._stream is not None and self._stream.active
               and time.monotonic() < self._release_deadline):
            time.sleep(.005)
        self._close_stream()

    def _close_stream(self):
        self._timer.stop()
        if self._monitor is not None:
            self._last_sample = self._monitor.position
            self._monitor = None
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.close()
            except sd.PortAudioError as exc:
                self.error.emit(str(exc))

    def current_sample(self) -> int:
        monitor = self._pending or self._monitor
        return monitor.position if monitor is not None else self._last_sample

    @Slot(int)
    def _on_finished_main(self, generation):
        if self._stream is None or generation != self._generation:
            return
        user_stopped = self._monitor.stopping
        self._close_stream()
        pending, self._pending = self._pending, None
        if pending is not None:
            if not self._open(pending):
                self.playback_finished.emit()
        elif not user_stopped:
            self.playback_finished.emit()

    @Slot()
    def _on_timer(self):
        if self._stream is not None:
            if not self._monitor.stopping:
                self.cursor_updated.emit(self._monitor.position)
            elif time.monotonic() >= self._release_deadline:
                self._on_finished_main(self._generation)
