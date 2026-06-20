from __future__ import annotations

import re

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QTextBrowser

from sessionprepgui.detail.report import render_summary_html, render_track_detail_html
from sessionprepgui.detail.report_style import (
    configure_report_browser,
    report_font_scale,
    wrap_report_html,
)
from sessionpreplib.detectors.silence import SilenceDetector
from sessionpreplib.models import (
    DetectorResult,
    ProcessorResult,
    Severity,
    TrackContext,
)
from sessionpreplib.processors.bimodal_normalize import BimodalNormalizeProcessor


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _assert_no_fixed_report_typography(html: str) -> None:
    assert "Consolas" not in html
    assert "monospace" not in html
    assert not re.search(r"font-size\s*:\s*[^;\"']*pt\b", html)


class _FakeScreen:
    def __init__(self, dpr: float, logical_dpi: float):
        self._dpr = dpr
        self._logical_dpi = logical_dpi

    def devicePixelRatio(self) -> float:
        return self._dpr

    def logicalDotsPerInch(self) -> float:
        return self._logical_dpi


@pytest.mark.parametrize(
    ("screen", "expected"),
    [
        (_FakeScreen(1.0, 96.0), 1.0),
        (_FakeScreen(1.25, 96.0), 1.08),
        (_FakeScreen(1.0, 144.0), 1.08),
        (_FakeScreen(2.0, 96.0), 1.15),
        (_FakeScreen(1.0, 192.0), 1.15),
    ],
)
def test_report_font_scale_buckets(screen, expected):
    assert report_font_scale(screen) == expected


def test_report_browser_uses_application_font(qapp):
    browser = QTextBrowser()

    configure_report_browser(browser, _FakeScreen(1.0, 96.0))

    assert browser.font().family() == qapp.font().family()
    assert browser.document().defaultFont().family() == qapp.font().family()
    assert browser.document().defaultFont().pointSize() == qapp.font().pointSize()


def test_report_browser_applies_high_dpi_boost(qapp):
    browser = QTextBrowser()
    screen = _FakeScreen(2.0, 96.0)

    configure_report_browser(browser, screen)

    expected_size = qapp.font().pointSizeF() * 1.15
    assert browser.font().family() == qapp.font().family()
    assert browser.document().defaultFont().pointSizeF() == pytest.approx(
        expected_size
    )


def test_report_wrapper_does_not_override_platform_font_or_size():
    html = wrap_report_html("<p>Body</p>")

    _assert_no_fixed_report_typography(html)
    assert "font-family" not in html
    assert "font-size" not in html


def test_summary_report_uses_relative_typography():
    summary = {
        "problems": [{"title": "Clipping", "hint": None, "items": ["Kick.wav"]}],
        "attention": [],
        "information": [],
        "clean": [],
        "clean_count": 3,
        "total_ok": 4,
        "overview": {"most_common_sr": 48000, "most_common_bd": "24-bit"},
    }

    html = render_summary_html(summary)

    _assert_no_fixed_report_typography(html)
    assert "font-size:1.3em" in html
    assert "font-size:1.15em" in html


def test_track_detail_and_report_fragments_use_relative_typography():
    track = TrackContext(
        filename="Kick.wav",
        filepath="Kick.wav",
        audio_data=None,
        samplerate=48000,
        channels=1,
        total_samples=48000,
        bitdepth="24-bit",
        subtype="PCM_24",
        duration_sec=1.0,
    )

    html = render_track_detail_html(track)
    _assert_no_fixed_report_typography(html)

    detector = SilenceDetector()
    detector.configure({})
    det_html = detector.render_html(
        DetectorResult(
            detector_id="silence",
            severity=Severity.ATTENTION,
            summary="silent",
            data={"is_silent": True},
        ),
        track,
    )
    _assert_no_fixed_report_typography(det_html)
    assert "font-size:0.82em" in det_html

    processor = BimodalNormalizeProcessor()
    processor.configure({})
    proc_html = processor.render_html(
        ProcessorResult(
            processor_id="bimodal_normalize",
            gain_db=-3.0,
            classification="Transient",
            method="Peak \u2192 -6 dB",
            data={
                "detected_peak_db": -3.0,
                "detected_rms_db": -15.0,
                "target_peak": -6.0,
                "target_rms": -18.0,
                "rms_anchor_label": "p95",
                "gain_for_peak": -3.0,
                "gain_for_rms": -3.0,
            },
        ),
        track,
        verbose=True,
    )
    _assert_no_fixed_report_typography(proc_html)
    assert "font-size:0.9em" in proc_html
