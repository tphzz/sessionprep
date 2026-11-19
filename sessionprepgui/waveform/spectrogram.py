"""Spectrogram renderer: mel image, frequency scale, freq zoom, recompute worker."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter, QPen

from ..theme import COLORS
from .compute import (
    SPECTROGRAM_COLORMAPS, SpectrogramRecomputeWorker,
    _SPEC_DB_FLOOR, _SPEC_F_MAX, _SPEC_F_MIN, _SPEC_N_FFT,
    _hz_to_mel, _mel_to_hz,
)


@dataclass
class SpecRenderCtx:
    """All data SpectrogramRenderer needs — snapshot from WaveformWidget."""
    x0: int
    draw_w: int
    draw_h: int
    view_start: int
    view_end: int
    total_samples: int
    samplerate: int


def _peak_pixels(values, centers, edges, axis):
    """Hold the maximum of every source cell intersecting each pixel interval."""
    boundaries = (centers[:-1] + centers[1:]) / 2
    first = np.searchsorted(boundaries, edges[:-1], side="right")
    last = np.searchsorted(boundaries, edges[1:], side="left")
    values = np.moveaxis(values, axis, 0)
    # A pixel edge can bisect a cell. Include that cell on both sides;
    # repeated start indices naturally extend a cell when zooming in.
    reduced = np.maximum.reduceat(values[:last[-1] + 1], first, axis=0)
    np.maximum(reduced, values[last], out=reduced)
    return np.moveaxis(reduced, 0, axis)


class SpectrogramRenderer:
    """Renders the mel spectrogram image, frequency scale, and frequency guide.

    Owns all spectrogram-specific state: image cache, FFT settings, mel view
    range, and the recompute worker.
    """

    def __init__(self):
        self._spec_db: np.ndarray | None = None
        self._spec_image: QImage | None = None
        self._spec_image_data = None          # prevent GC of numpy buffer
        self._spec_cache_key: tuple = ()
        self._spec_recompute_worker: SpectrogramRecomputeWorker | None = None
        self._spec_n_fft: int = _SPEC_N_FFT
        self._spec_window: str = "hann"
        self._spec_db_floor: float = _SPEC_DB_FLOOR
        self._spec_db_ceil: float = 0.0
        self._colormap: str = "magma"
        self._mel_view_min: float = _hz_to_mel(_SPEC_F_MIN)
        self._mel_view_max: float = _hz_to_mel(_SPEC_F_MAX)
        self._on_done_callback = None
        self.band = "full"
        self._native_range = False
        self._frequencies = None
        self._column_times = None

    # ── Public API ──────────────────────────────────────────────────────────

    def reset(self, samplerate: int = 44100):
        """Clear spec data on track unload.  Resets mel view to full range."""
        if self._spec_recompute_worker is not None:
            self._spec_recompute_worker.cancel()
            self._spec_recompute_worker.finished.disconnect()
            self._spec_recompute_worker = None
        self._spec_db = None
        self._spec_image = None
        self._spec_image_data = None
        self._spec_cache_key = ()
        self.band = "full"
        self._native_range = False
        self._frequencies = None
        self._column_times = None
        self._mel_view_min = _hz_to_mel(_SPEC_F_MIN)
        self._mel_view_max = _hz_to_mel(min(_SPEC_F_MAX, samplerate / 2.0))

    def set_spec_data(self, spec_db):
        """Set new spectrogram data and invalidate image cache."""
        self._spec_db = spec_db
        self._spec_image = None
        self._spec_image_data = None
        self._spec_cache_key = ()

    def set_inspection(self, result, preserve=False):
        band = result.get("inspection_band", "full")
        same = preserve and self.band == band and self._native_range
        self.band = band
        self._native_range = True
        self._frequencies = result.get("spec_freq")
        self._column_times = result.get("spec_times")
        if not same:
            self.reset_freq_view(result["samplerate"])
        self.set_spec_data(result.get("spec_db"))

    def _axis_bounds(self, samplerate):
        if self.band != "full" and self._frequencies is not None:
            return float(self._frequencies[0]), float(self._frequencies[-1])
        upper = samplerate / 2 if self._native_range else min(_SPEC_F_MAX, samplerate / 2)
        return _hz_to_mel(_SPEC_F_MIN), _hz_to_mel(upper)

    def hz_for_axis(self, value):
        return _mel_to_hz(value) if self.band == "full" else value

    def _band_label(self, frequency):
        return f"{frequency / 1000:.4g} kHz" if self.band == "high" else f"{frequency:.3g} Hz"

    def axis_margin(self):
        if self.band == "full":
            return 0
        fm = QFontMetrics(QFont("Consolas", 7))
        return max(fm.horizontalAdvance(self._band_label(hz)) for hz in
                   (self._mel_view_min, self._mel_view_max, (self._mel_view_min + self._mel_view_max) / 2)) + 7

    def invalidate(self):
        """Force image rebuild on next paint (e.g. resize)."""
        self._spec_image = None
        self._spec_image_data = None
        self._spec_cache_key = ()

    def paint(self, painter: QPainter, ctx: SpecRenderCtx):
        """Paint spectrogram image and frequency scale."""
        if self._spec_db is None:
            painter.setPen(QPen(QColor(COLORS["dim"])))
            painter.drawText(ctx.x0, int(ctx.draw_h / 2),
                             "Spectrogram not available (audio too short)")
            return
        cache_key = (ctx.view_start, ctx.view_end, ctx.draw_w,
                     int(ctx.draw_h), self._colormap,
                     self._mel_view_min, self._mel_view_max,
                     self._spec_db_floor, self._spec_db_ceil)
        if self._spec_cache_key != cache_key or self._spec_image is None:
            self._build_spec_image(ctx)
            self._spec_cache_key = cache_key
        if self._spec_image is not None:
            painter.drawImage(ctx.x0, 0, self._spec_image)
        self._draw_freq_scale(painter, ctx)

    def draw_freq_guide(self, painter: QPainter, ctx: SpecRenderCtx, my: float):
        """Draw frequency readout at mouse position (called from paintEvent)."""
        if self._spec_db is None or ctx.draw_h <= 0:
            return
        mel_range = self._mel_view_max - self._mel_view_min
        if mel_range <= 0:
            return
        frac = max(0.0, min(1.0 - my / ctx.draw_h, 1.0))
        freq = self.hz_for_axis(self._mel_view_min + frac * mel_range)
        freq_label = f"{freq / 1000:.1f} kHz" if freq >= 1000 else f"{freq:.0f} Hz"
        painter.setFont(QFont("Consolas", 7))
        label_color = QColor(180, 180, 180, 120)
        painter.setPen(label_color)
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(freq_label)
        label_y = int(my) + fm.ascent() // 2
        painter.drawText(ctx.x0 + 4, label_y, freq_label)
        painter.drawText(ctx.x0 + ctx.draw_w - tw - 4, label_y, freq_label)

    def freq_zoom(self, factor: float, anchor_mel: float | None,
                  samplerate: int = 44100):
        """Zoom the mel frequency range by factor around anchor_mel."""
        mel_range = self._mel_view_max - self._mel_view_min
        mel_full_min, mel_full_max = self._axis_bounds(samplerate)
        if anchor_mel is not None:
            anchor = max(self._mel_view_min, min(anchor_mel, self._mel_view_max))
            frac = (anchor - self._mel_view_min) / mel_range if mel_range > 0 else 0.5
        else:
            anchor = (self._mel_view_min + self._mel_view_max) / 2.0
            frac = 0.5
        minimum = 1. if self.band == "low" else 50.
        new_range = min(mel_full_max - mel_full_min, max(mel_range * factor, minimum))
        new_min = anchor - frac * new_range
        new_max = anchor + (1.0 - frac) * new_range
        if new_min < mel_full_min:
            new_min = mel_full_min
            new_max = new_min + new_range
        if new_max > mel_full_max:
            new_max = mel_full_max
            new_min = new_max - new_range
        self._mel_view_min = max(new_min, mel_full_min)
        self._mel_view_max = min(new_max, mel_full_max)
        self._spec_image = None
        self._spec_cache_key = ()

    def scroll_freq(self, delta_mel: float, samplerate: int = 44100):
        """Pan the frequency view by delta_mel mels."""
        mel_full_min, mel_full_max = self._axis_bounds(samplerate)
        mel_range = self._mel_view_max - self._mel_view_min
        new_min = self._mel_view_min + delta_mel
        new_max = self._mel_view_max + delta_mel
        if new_min < mel_full_min:
            new_min = mel_full_min
            new_max = new_min + mel_range
        if new_max > mel_full_max:
            new_max = mel_full_max
            new_min = new_max - mel_range
        self._mel_view_min = max(new_min, mel_full_min)
        self._mel_view_max = min(new_max, mel_full_max)
        self._spec_image = None
        self._spec_cache_key = ()

    def reset_freq_view(self, samplerate: int):
        """Reset mel frequency view to full range (used by zoom_fit)."""
        self._mel_view_min, self._mel_view_max = self._axis_bounds(samplerate)
        self._spec_image = None
        self._spec_cache_key = ()

    def recompute(self, channels: list, sr: int, *, on_done, parent=None):
        """Launch background spectrogram recompute.  Calls on_done() when done."""
        if self._spec_recompute_worker is not None:
            self._spec_recompute_worker.cancel()
            self._spec_recompute_worker.finished.disconnect()
            self._spec_recompute_worker = None
        self._spec_db = None
        self._spec_image = None
        self._spec_cache_key = ()
        self._on_done_callback = on_done
        worker = SpectrogramRecomputeWorker(
            channels, sr,
            n_fft=self._spec_n_fft, window=self._spec_window,
            parent=parent,
        )
        worker.finished.connect(self._on_spec_recomputed)
        self._spec_recompute_worker = worker
        worker.start()

    def set_colormap(self, name: str):
        if name not in SPECTROGRAM_COLORMAPS:
            return
        self._colormap = name
        self._spec_image = None
        self._spec_cache_key = ()

    def set_n_fft(self, n_fft: int):
        self._spec_n_fft = n_fft

    def set_window(self, window: str):
        self._spec_window = window

    def set_db_floor(self, val: float):
        self._spec_db_floor = val
        self._spec_image = None
        self._spec_cache_key = ()

    def set_db_ceil(self, val: float):
        self._spec_db_ceil = val
        self._spec_image = None
        self._spec_cache_key = ()

    @property
    def mel_view_min(self) -> float:
        return self._mel_view_min

    @property
    def mel_view_max(self) -> float:
        return self._mel_view_max

    @property
    def spec_n_fft(self) -> int:
        return self._spec_n_fft

    @property
    def spec_window(self) -> str:
        return self._spec_window

    # ── Internal helpers ────────────────────────────────────────────────────

    def _on_spec_recomputed(self, spec_db):
        self._spec_db = spec_db
        self._spec_image = None
        self._spec_cache_key = ()
        self._spec_recompute_worker = None
        if self._on_done_callback is not None:
            self._on_done_callback()

    def _build_spec_image(self, ctx: SpecRenderCtx):
        """Render the visible portion of the spectrogram to a cached QImage."""
        spec = self._spec_db
        if spec is None or ctx.draw_w <= 0 or ctx.draw_h <= 0:
            self._spec_image = None
            return
        rows, columns = spec.shape
        if not rows or not columns:
            self._spec_image = None
            return
        full_min, full_max = self._axis_bounds(ctx.samplerate)
        centers = self._column_times
        if centers is None:
            centers = (np.arange(columns) + 0.5) * ctx.total_samples / (columns * ctx.samplerate)
        time_edges = np.linspace(ctx.view_start, ctx.view_end, ctx.draw_w + 1) / ctx.samplerate
        view_spec = _peak_pixels(spec, centers, time_edges, axis=1)
        freq_centers = (self._frequencies if self.band != "full" and self._frequencies is not None
                        else np.linspace(full_min, full_max, rows))
        freq_edges = np.linspace(self._mel_view_min, self._mel_view_max, ctx.draw_h + 1)
        view_spec = _peak_pixels(view_spec, freq_centers, freq_edges, axis=0)
        db_floor = self._spec_db_floor
        db_ceil = self._spec_db_ceil
        norm = np.clip((view_spec - db_floor) / max(db_ceil - db_floor, 1.0), 0.0, 1.0)
        norm = norm[::-1, :]  # low freq at bottom
        lut = SPECTROGRAM_COLORMAPS.get(self._colormap)
        if lut is None:
            lut = SPECTROGRAM_COLORMAPS.get("magma", np.zeros((256, 4), np.uint8))
        indices = (norm * 255).astype(np.uint8)
        rgba = np.ascontiguousarray(lut[indices])
        self._spec_image_data = rgba
        self._spec_image = QImage(rgba.data, ctx.draw_w, ctx.draw_h, rgba.strides[0],
                                  QImage.Format.Format_RGBA8888)

    def _draw_freq_scale(self, painter: QPainter, ctx: SpecRenderCtx):
        """Draw frequency scale on left/right margins for spectrogram mode."""
        if self._spec_db is None or ctx.draw_h <= 0:
            return
        ticks = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 40000, 80000]
        _MIN_TICK_SPACING = 20
        mel_min = self._mel_view_min
        mel_max = self._mel_view_max
        mel_range = mel_max - mel_min
        if mel_range <= 0:
            return
        scale_font = QFont("Consolas", 7)
        painter.setFont(scale_font)
        fm = painter.fontMetrics()
        text_h = fm.height()
        label_color = QColor(COLORS["dim"])
        tick_pen = QPen(label_color, 1)
        grid_color = QColor(COLORS["accent"])
        grid_color.setAlpha(35)
        grid_pen = QPen(grid_color, 1, Qt.DotLine)
        used_ys: list[int] = []
        if self.band != "full":
            ticks = np.linspace(mel_min, mel_max, 3)
        for freq in ticks:
            mel = _hz_to_mel(freq) if self.band == "full" else freq
            if mel < mel_min or mel > mel_max:
                continue
            frac = (mel - mel_min) / mel_range
            y = int(ctx.draw_h * (1.0 - frac))
            if self.band != "full":
                y = max(text_h // 2 + 1, min(y, ctx.draw_h - text_h // 2 - 1))
            elif y < text_h or y > ctx.draw_h - text_h:
                continue
            if any(abs(uy - y) < _MIN_TICK_SPACING for uy in used_ys):
                continue
            used_ys.append(y)
            if self.band == "full":
                label = f"{freq / 1000:g}k" if freq >= 1000 else f"{freq:g}"
            else:
                label = self._band_label(freq)
            painter.setPen(grid_pen)
            painter.drawLine(ctx.x0, y, ctx.x0 + ctx.draw_w, y)
            painter.setPen(tick_pen)
            tw = fm.horizontalAdvance(label)
            painter.drawText(ctx.x0 - 5 - tw, y + fm.ascent() // 2, label)
            painter.drawText(ctx.x0 + ctx.draw_w + 5, y + fm.ascent() // 2, label)
            painter.drawLine(ctx.x0 - 3, y, ctx.x0, y)
            painter.drawLine(ctx.x0 + ctx.draw_w, y, ctx.x0 + ctx.draw_w + 3, y)
