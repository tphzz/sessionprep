from __future__ import annotations

import re

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QTextBrowser

from sessionprepgui.detail.report import render_summary_html, render_track_detail_html
from sessionprepgui.detail.report_style import (
    configure_report_browser,
    REPORT_CONTENT_SCALE,
    report_font_scale,
    wrap_report_html,
)
from sessionpreplib.detectors.silence import SilenceDetector
from sessionpreplib.detectors.stereo_compat import StereoCompatDetector
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


def _assert_detector_badge_does_not_wrap(html: str) -> None:
    assert 'width="118"' in html
    assert "min-width:118px" in html
    assert "white-space:nowrap" in html


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

    expected_size = qapp.font().pointSizeF() * REPORT_CONTENT_SCALE
    assert browser.font().family() == qapp.font().family()
    assert browser.document().defaultFont().family() == qapp.font().family()
    assert browser.document().defaultFont().pointSizeF() == pytest.approx(
        expected_size
    )


def test_report_browser_applies_high_dpi_boost(qapp):
    browser = QTextBrowser()
    screen = _FakeScreen(2.0, 96.0)

    configure_report_browser(browser, screen)

    expected_size = qapp.font().pointSizeF() * REPORT_CONTENT_SCALE * 1.15
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
    _assert_detector_badge_does_not_wrap(det_html)
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


def test_stereo_compat_report_badge_does_not_wrap():
    detector = StereoCompatDetector()
    detector.configure({})
    html = detector.render_html(
        DetectorResult(
            detector_id="stereo_compat",
            severity=Severity.ATTENTION,
            summary="stereo warning",
            data={
                "lr_corr": 0.43,
                "mono_loss_db": 1.5,
                "windowed_regions": [{"start": 0.0, "end": 1.0}],
            },
        )
    )

    _assert_no_fixed_report_typography(html)
    _assert_detector_badge_does_not_wrap(html)
    assert "ATTENTION" in html


def test_fallback_detector_report_uses_compact_row():
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
    track.detector_results["unknown_detector"] = DetectorResult(
        detector_id="unknown_detector",
        severity=Severity.ATTENTION,
        summary="unknown warning",
        data={},
    )

    html = render_track_detail_html(track)

    _assert_no_fixed_report_typography(html)
    assert "unknown_detector" in html
    assert "Attention" in html
    assert "report-details:unknown_detector" in html


@pytest.mark.parametrize("screen_scale", [1.0, 2.0])
def test_finding_ui_role_matches_application_size_and_preserves_report_header(qapp, screen_scale):
    from sessionprepgui.detail.browser import ReportBrowser

    current = TrackContext("UI role.wav", "missing.wav", None, 48000, 1, 96000,
                           "24-bit", "PCM_24", 2.0)
    current.detector_results["example"] = DetectorResult(
        "example", Severity.ATTENTION, "Review this finding", {})
    browser = ReportBrowser({})
    browser.refresh_font(_FakeScreen(screen_scale, 96.0))
    browser.present_track(current, None)
    finding = browser.document().find("example")
    header = browser.document().find("UI role.wav")
    expected = qapp.font().pointSizeF() * report_font_scale(_FakeScreen(screen_scale, 96.0))
    assert finding.charFormat().font().pointSizeF() == pytest.approx(expected)
    assert header.charFormat().font().pointSizeF() == pytest.approx(expected * REPORT_CONTENT_SCALE)
    assert finding.charFormat().fontWeight() >= 600


def test_custom_pixel_and_point_font_sizes_survive_finding_role_and_refresh(qapp):
    from sessionprepgui.detail.browser import ReportBrowser
    from sessionpreplib.models import SessionContext

    class Detector:
        id = "format"
        name = "Custom formatting"
        html_help = ""
        def effective_severity(self, result):
            return result.severity
        def render_html(self, result, track):
            return ("<tr><td><span style='font-size:20px'>PIXEL</span> "
                    "<span style='font-size:17pt; font-family:Consolas; font-style:italic'>POINT</span>"
                    "</td></tr>")

    current = TrackContext("Formats.wav", "missing.wav", None, 48000, 1, 96000,
                           "24-bit", "PCM_24", 2.0)
    current.detector_results["format"] = DetectorResult(
        "format", Severity.ATTENTION, "Formatted detail", {})
    session = SessionContext([current], {}, detectors=[Detector()])
    browser = ReportBrowser({})
    browser.present_track(current, session)
    from PySide6.QtCore import QUrl
    browser.activate_report_link(QUrl("report-details:format"))
    for scale in (1.0, 2.0, 1.0):
        browser.refresh_font(_FakeScreen(scale, 96.0))
        pixel = browser.document().find("PIXEL").charFormat()
        point = browser.document().find("POINT").charFormat()
        assert pixel.font().pixelSize() == 20
        assert point.fontPointSize() == pytest.approx(17.0)
        assert point.fontItalic()
        assert point.font().family() == "Consolas"
