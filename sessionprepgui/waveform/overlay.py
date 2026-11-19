"""Stateless overlay drawing helpers: issue overlays and time scale."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen

from ..theme import COLORS
from .compute import _hz_to_mel

_SEVERITY_OVERLAY = {
    "problem": QColor(255, 68, 68, 55),
    "attention": QColor(255, 170, 0, 45),
    "information": QColor(68, 153, 255, 40),
    "info": QColor(68, 153, 255, 40),
}
_SEVERITY_BORDER = {
    "problem": QColor(255, 68, 68, 140),
    "attention": QColor(255, 170, 0, 120),
    "information": QColor(68, 153, 255, 100),
    "info": QColor(68, 153, 255, 100),
}


def issue_y_bounds(issue, height, mode, channels, axis_min, axis_max, linear):
    """Return the visible half-open pixel interval for painting and hit testing."""
    if mode == "waveform" and issue.channel is not None:
        if not 0 <= issue.channel < channels:
            return 0, 0
        return int(height * issue.channel / channels), int(height * (issue.channel + 1) / channels)
    if mode == "spectrogram" and axis_max > axis_min:
        def edge(hz, fallback):
            if hz is None:
                return fallback
            axis_value = hz if linear else _hz_to_mel(hz)
            return max(0, min(int(height), int(height * (axis_max - axis_value) / (axis_max - axis_min))))
        return edge(issue.freq_max_hz, 0), edge(issue.freq_min_hz, int(height))
    return 0, int(height)


def draw_issue_overlays(
    painter: QPainter,
    x0: int,
    draw_w: int,
    draw_h: float,
    view_start: int,
    view_end: int,
    total_samples: int,
    issues: list,
    enabled_overlays: set,
    display_mode: str,
    num_channels: int,
    mel_view_min: float,
    mel_view_max: float,
    linear_frequency: bool = False,
):  # pylint: disable=too-many-positional-arguments
    """Draw detector issue overlays.  Works in both waveform and spectrogram modes."""
    if not issues or not enabled_overlays:
        return

    view_len = view_end - view_start
    if view_len <= 0 or total_samples <= 0:
        return

    def _sample_to_x(sample: int) -> int:
        return x0 + int((sample - view_start) / view_len * draw_w)

    for issue in issues:
        if issue.label not in enabled_overlays:
            continue
        sev_val = issue.severity.value if hasattr(issue.severity, "value") else str(issue.severity)
        fill = _SEVERITY_OVERLAY.get(sev_val, QColor(255, 255, 255, 30))
        border = _SEVERITY_BORDER.get(sev_val, QColor(255, 255, 255, 60))

        ix1 = _sample_to_x(issue.sample_start)
        ix2 = (_sample_to_x(issue.sample_end + 1)
               if issue.sample_end is not None else ix1)
        rx = ix1
        rw = max(ix2 - ix1, 2)

        ry, bottom = issue_y_bounds(issue, draw_h, display_mode, num_channels,
                                    mel_view_min, mel_view_max, linear_frequency)
        rh = bottom - ry
        if rh <= 0:
            continue

        painter.fillRect(rx, ry, rw, rh, fill)
        painter.setPen(QPen(border, 1))
        painter.drawRect(rx, ry, rw, rh)


def draw_time_scale(
    painter: QPainter,
    x0: int,
    draw_w: int,
    draw_h: float,
    view_start: int,
    view_end: int,
    samplerate: int,
):  # pylint: disable=too-many-positional-arguments
    """Draw horizontal time axis with adaptive tick labels below the waveform."""
    if samplerate <= 0 or draw_w <= 0:
        return

    view_start_sec = view_start / samplerate
    view_end_sec = view_end / samplerate
    visible_dur = view_end_sec - view_start_sec
    if visible_dur <= 0:
        return

    _NICE_INTERVALS = [
        0.001, 0.002, 0.005,
        0.01, 0.02, 0.05,
        0.1, 0.2, 0.5,
        1, 2, 5, 10, 15, 30,
        60, 120, 300, 600, 1800, 3600,
    ]
    _MIN_TICK_PX = 60

    interval = _NICE_INTERVALS[-1]
    for ni in _NICE_INTERVALS:
        if ni / visible_dur * draw_w >= _MIN_TICK_PX:
            interval = ni
            break

    if interval >= 1.0:
        def _fmt(t):
            m = int(t) // 60
            s = int(t) % 60
            return f"{m}:{s:02d}"
    elif interval >= 0.1:
        def _fmt(t):
            m = int(t) // 60
            s = t - m * 60
            return f"{m}:{s:04.1f}"
    elif interval >= 0.01:
        def _fmt(t):
            m = int(t) // 60
            s = t - m * 60
            return f"{m}:{s:05.2f}"
    else:
        def _fmt(t):
            m = int(t) // 60
            s = t - m * 60
            return f"{m}:{s:06.3f}"

    scale_font = QFont("Consolas", 7)
    painter.setFont(scale_font)
    fm = painter.fontMetrics()

    label_color = QColor(COLORS["dim"])
    tick_pen = QPen(label_color, 1)
    grid_color = QColor(COLORS["accent"])
    grid_color.setAlpha(25)
    grid_pen = QPen(grid_color, 1, Qt.DotLine)

    first_tick = (int(view_start_sec / interval) + 1) * interval
    if abs(view_start_sec / interval - round(view_start_sec / interval)) < 1e-9:
        first_tick = round(view_start_sec / interval) * interval

    t = first_tick
    bottom_y = int(draw_h)
    while t <= view_end_sec + interval * 0.01:
        frac = (t - view_start_sec) / visible_dur
        px = x0 + int(frac * draw_w)

        if px < x0 or px > x0 + draw_w:
            t += interval
            continue

        painter.setPen(grid_pen)
        painter.drawLine(px, 0, px, bottom_y)

        painter.setPen(tick_pen)
        painter.drawLine(px, bottom_y, px, bottom_y + 4)

        label = _fmt(t)
        tw = fm.horizontalAdvance(label)
        lx = px - tw // 2
        ly = bottom_y + 4 + fm.ascent()
        painter.drawText(int(lx), int(ly), label)

        t += interval
