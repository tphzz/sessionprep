from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QRect, QSize

from sessionprepgui.window_geometry import (
    STARTUP_SCREEN_HEIGHT_FRACTION,
    STARTUP_SCREEN_MARGIN,
    STARTUP_SCREEN_WIDTH_FRACTION,
    calculate_startup_geometry,
)


def test_startup_geometry_uses_wider_five_sixths_1920_by_1080_shape():
    available = QRect(0, 0, 1920, 1080)

    geometry = calculate_startup_geometry(available, QSize(0, 0))

    assert geometry.width() == round(1920 * STARTUP_SCREEN_WIDTH_FRACTION)
    assert geometry.height() == round(1080 * STARTUP_SCREEN_HEIGHT_FRACTION)
    assert geometry.height() == 900
    assert geometry.center() == available.center()


def test_startup_geometry_stays_inside_small_available_area():
    available = QRect(0, 23, 1366, 721)

    geometry = calculate_startup_geometry(available, QSize(0, 0))

    assert available.contains(geometry)
    assert geometry.width() == round(1366 * STARTUP_SCREEN_WIDTH_FRACTION)
    assert geometry.height() == round(721 * STARTUP_SCREEN_HEIGHT_FRACTION)


def test_startup_geometry_scales_on_large_displays_without_fixed_1600_cap():
    available = QRect(0, 0, 3840, 2160)

    geometry = calculate_startup_geometry(available, QSize(0, 0))

    assert geometry.width() == round(3840 * STARTUP_SCREEN_WIDTH_FRACTION)
    assert geometry.height() == round(2160 * STARTUP_SCREEN_HEIGHT_FRACTION)
    assert geometry.width() > 1600


def test_startup_geometry_uses_layout_minimum_width_when_it_fits():
    available = QRect(0, 0, 1920, 1080)

    geometry = calculate_startup_geometry(available, QSize(1700, 950))

    assert geometry.width() == 1700
    assert geometry.height() == round(1080 * STARTUP_SCREEN_HEIGHT_FRACTION)
    assert available.contains(geometry)


def test_startup_geometry_does_not_expand_height_to_layout_hint():
    available = QRect(0, 0, 1920, 1080)

    geometry = calculate_startup_geometry(available, QSize(704, 948))

    assert geometry.width() == round(1920 * STARTUP_SCREEN_WIDTH_FRACTION)
    assert geometry.height() == round(1080 * STARTUP_SCREEN_HEIGHT_FRACTION)
    assert available.contains(geometry)


def test_startup_geometry_clamps_oversized_minimum_width_to_available_area():
    available = QRect(0, 0, 1920, 1080)

    geometry = calculate_startup_geometry(available, QSize(3000, 2000))

    assert geometry.width() == 1920 - STARTUP_SCREEN_MARGIN * 2
    assert geometry.height() == round(1080 * STARTUP_SCREEN_HEIGHT_FRACTION)
    assert available.contains(geometry)


def test_startup_geometry_centers_on_screens_with_nonzero_origins():
    available = QRect(-1920, 100, 1920, 1080)

    geometry = calculate_startup_geometry(available, QSize(0, 0))

    assert available.contains(geometry)
    assert geometry.center() == available.center()
