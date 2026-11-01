"""Shared vector artwork for desktop controls."""
from functools import lru_cache
from pathlib import Path

from PySide6.QtGui import QIcon

_NAMES = frozenset((
    "waveform", "fit", "zoom-in", "zoom-out", "scale-up", "scale-down",
    "play", "stop", "reset", "chevron-down", "chevron-right", "chevron-up",
    "plus", "minus", "refresh", "busy", "close",
))


def icon_path(name: str) -> Path:
    """Return a packaged icon path, independent of the working directory."""
    if name not in _NAMES:
        raise ValueError(f"Unknown desktop icon: {name}")
    return Path(__file__).with_name("res") / f"ui-{name}.svg"


@lru_cache(maxsize=17)
def _cached_icon(name: str) -> QIcon:
    return QIcon(str(icon_path(name)))


def icon(name: str) -> QIcon:
    """Return an independent, implicitly shared icon."""
    return QIcon(_cached_icon(name))
