from __future__ import annotations

import re

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QTextBrowser

from sessionprepgui.detail.report import render_summary_html, render_track_detail_html
from sessionprepgui.detail.report_style import (
    configure_report_browser,
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


def test_report_browser_uses_application_font(qapp):
    browser = QTextBrowser()

    configure_report_browser(browser)

    assert browser.font().family() == qapp.font().family()
    assert browser.document().defaultFont().family() == qapp.font().family()
    assert browser.document().defaultFont().pointSize() == qapp.font().pointSize()


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
