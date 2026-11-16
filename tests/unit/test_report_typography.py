from __future__ import annotations

import re

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QTextBrowser

from sessionprepgui.detail.browser import ReportBrowser
from sessionprepgui.detail.report import render_summary_html, render_track_detail_html
from sessionprepgui.detail.report_style import (
    configure_report_browser,
    REPORT_CONTENT_SCALE,
    report_font_scale,
    wrap_report_html,
)
from sessionprepgui.theme import COLORS, REPORT_INSET
from sessionpreplib.detectors.silence import SilenceDetector
from sessionpreplib.detectors.stereo_compat import StereoCompatDetector
from sessionpreplib.models import (
    DetectorResult,
    ProcessorResult,
    SessionContext,
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
    # Title role uses a CSS keyword Qt honours; category headings keep their size.
    assert "font-size:x-large" in html
    assert "font-size:medium" in html


def _report_cursor(browser, text):
    cursor = browser.document().find(text)
    assert not cursor.isNull(), text
    return cursor


def _header_band_height(browser, text):
    position = browser.cursorRect(_report_cursor(browser, text)).center()
    image = browser.viewport().grab().toImage()
    dpr = image.devicePixelRatio()
    x = round((REPORT_INSET + 3) * dpr)
    y = round(position.y() * dpr)
    column = [image.pixelColor(x, row).name() for row in range(image.height())]
    surface = COLORS["surface"]
    assert column[y] == surface
    first = y
    while first > 0 and column[first - 1] == surface:
        first -= 1
    last = y
    while last + 1 < len(column) and column[last + 1] == surface:
        last += 1
    return last - first + 1


@pytest.mark.parametrize("width", [1200, 620])
def test_summary_and_file_share_finding_font_and_header_height(form_font, qapp, width):
    track = TrackContext(
        "Bass direct input alternate full take with performance notes.wav",
        "missing.wav", None, 48000, 1, 96000, "24-bit", "PCM_24", 2.0,
    )
    long_detail = "The alternate print has a long result that can wrap across several lines."
    long_guidance = (
        "Check the entire affected recording and compare the original export "
        "with the alternate print before preparing this session for transfer."
    )
    track.detector_results = {
        "Early finding": DetectorResult("Early finding", Severity.PROBLEM, "First result", {}),
        "Later finding": DetectorResult("Later finding", Severity.INFO, long_detail, {},
                                        hint=long_guidance),
    }
    session = SessionContext([track], {})
    summary = {
        "clean_count": 0,
        "total_ok": 1,
        "problems": [{
            "title": "Early finding",
            "items": ["First result"],
            "item_refs": [{"filename": track.filename, "detail": "First result"}],
        }],
        "information": [{
            "title": "Later finding",
            "items": [long_detail],
            "item_refs": [{"filename": track.filename, "detail": long_detail}],
            "hint": long_guidance,
        }],
    }
    summary_view, file_view = ReportBrowser({}), ReportBrowser({})
    try:
        for view in (summary_view, file_view):
            view.resize(width, 560)
            view.show()
        summary_view.present_summary(summary, session)
        file_view.present_track(track, session)
        for scale in (1.0, 1.15, 1.5, 1.75, 1.0):
            screen = _FakeScreen(scale, 96.0)
            for view in (summary_view, file_view):
                view.refresh_font(screen)
            qapp.processEvents()

            body_size = qapp.font().pointSizeF() * report_font_scale(screen)
            for view, labels in (
                (summary_view, ("Affected file", "Early finding", "Later finding",
                                track.filename, long_detail, long_guidance, "Review file")),
                (file_view, ("Severity", "Early finding", "Later finding", long_guidance)),
            ):
                for label in labels:
                    fmt = _report_cursor(view, label).charFormat()
                    assert fmt.font().pointSizeF() == pytest.approx(body_size), label
            for view, labels in (
                (summary_view, ("Problems · 1 finding", "Information · 1 finding",
                                "0 of 1 files clean")),
                (file_view, ("Detector findings", "48000 Hz")),
            ):
                for label in labels:
                    fmt = _report_cursor(view, label).charFormat()
                    assert fmt.font().pointSizeF() == pytest.approx(body_size * REPORT_CONTENT_SCALE)

            summary_height = _header_band_height(summary_view, "Affected file")
            file_height = _header_band_height(file_view, "Severity")
            tolerance = 0 if summary_view.devicePixelRatioF().is_integer() else 1
            assert abs(summary_height - file_height) <= tolerance
            assert _report_cursor(summary_view, long_guidance).block().layout().lineCount() > 1
    finally:
        summary_view.close()
        file_view.close()


def test_empty_summary_and_file_share_finding_font(qapp):
    track = TrackContext("Clean.wav", "missing.wav", None, 48000, 1, 96000,
                         "24-bit", "PCM_24", 2.0)
    session = SessionContext([track], {})
    summary = {"clean": [{"title": "Safe levels", "items": ["Clean.wav"]}],
               "clean_count": 1, "total_ok": 1}
    summary_view, file_view = ReportBrowser({}), ReportBrowser({})
    try:
        for view in (summary_view, file_view):
            view.refresh_font(_FakeScreen(1.0, 96.0))
        summary_view.present_summary(summary, session, show_clean=False)
        file_view.present_track(track, session, show_clean=False)
        for view in (summary_view, file_view):
            empty = _report_cursor(view, "No findings to show.").charFormat()
            assert empty.font().pointSizeF() == pytest.approx(qapp.font().pointSizeF())
    finally:
        summary_view.close()
        file_view.close()


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
    # Chosen state is named in text; normalization uses no status colours.
    assert "Chosen" in proc_html
    assert "#44cc44" not in proc_html and "#cc8844" not in proc_html


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
    # The file name uses the x-large title role (1.5x the report body).
    assert header.charFormat().font().pointSizeF() == pytest.approx(expected * REPORT_CONTENT_SCALE * 1.5)
    section = browser.document().find("Detector findings")
    assert section.charFormat().font().pointSizeF() == pytest.approx(expected * REPORT_CONTENT_SCALE)
    assert finding.charFormat().fontWeight() >= 600


def test_custom_pixel_and_point_font_sizes_survive_finding_role_and_refresh(qapp):
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
