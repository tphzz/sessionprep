"""The narrow File report: stacked findings, narrow normalization, Help stacking."""
import re
from urllib.parse import quote

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QApplication

from sessionprepgui.detail import report
from sessionprepgui.detail.browser import NARROW_BELOW, NARROW_EXIT_AT, ReportBrowser
from sessionprepgui.detail.help_panel import FileReportPane, stack_two_column_rows
from sessionprepgui.detail.report import prepare_track_report, render_prepared_track_report
from sessionprepgui.theme import COLORS
from sessionpreplib.models import DetectorResult, ProcessorResult, SessionContext, Severity, TrackContext
from sessionpreplib.processors.bimodal_normalize import BimodalNormalizeProcessor


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    return app


@pytest.fixture(autouse=True)
def _no_chunks(monkeypatch):
    monkeypatch.setattr(report, "read_chunks", lambda path: ("WAVE", []))


HELP = ("<h3>Description</h3><p>Help body.</p>"
        "<table style='margin:4px 0;'>"
        "<tr><td><b>&minus;10 dB</b></td><td style='padding-left:8px;'>Severe</td></tr>"
        "<tr><td><b>&minus;20 dB</b></td><td style='padding-left:8px;'>Default</td></tr>"
        "</table>")


class _Detector:
    id = "tail"
    name = "Tail Regions Exceeded Anchor"

    def effective_severity(self, result):
        return result.severity

    def render_html(self, result, current):
        return "<tr><td>Rich result</td></tr>"

    def html_help(self):
        return HELP


def _session(classification="Sustained", method="RMS", **data):
    processor = BimodalNormalizeProcessor()
    processor.configure({})
    values = {"detected_peak_db": -9.1, "detected_rms_db": -22.0, "rms_anchor_label": "p95",
              "target_peak": -6.0, "target_rms": -16.0, "gain_for_peak": 3.1,
              "gain_for_rms": 6.0, "fader_offset": -1.0}
    values.update(data)
    track = TrackContext("Lead Vocal.wav", "missing.wav", None, 48000, 1, 96000,
                         "24-bit", "PCM_24", 2.0)
    track.processor_results[processor.id] = ProcessorResult(
        processor.id, 3.1, classification, method, values)
    detector = _Detector()
    track.detector_results[detector.id] = DetectorResult(
        detector.id, Severity.ATTENTION, "5 tail regions exceed anchor", {},
        hint="Check for section-based riding")
    return track, SessionContext([track], {}, processors=[processor], detectors=[detector])


# ── Normalization ────────────────────────────────────────────────────────

def _normalization(**kw):
    processor = BimodalNormalizeProcessor()
    processor.configure({})
    track, session = _session(**kw)
    result = track.processor_results[processor.id]
    return processor, result, track


def test_narrow_normalization_has_a_legend_and_one_entry_per_measure():
    processor, result, track = _normalization(classification="Sustained", method="Peak Limited",
                                              fader_rebalance_shift=2.0)
    html = processor.render_html(result, track, narrow=True)
    assert "Measure<br>Detected &rarr; Target &middot; Gain" in html
    # One single-cell row per measure: legend + Peak + RMS.
    assert html.count("<tr>") == 3 and "<td width" not in html
    entries = html.split("<tr>")
    peak = next(e for e in entries if "Peak</span>" in e)
    assert "Chosen &middot; limiting" in peak
    assert ("<nobr>&minus;9.1 dBFS</nobr> &rarr; <nobr>&minus;6.0 dBFS</nobr>"
            " &middot; <nobr>+3.1 dB</nobr>") in peak
    assert "font-weight:bold" in peak  # the chosen measure, also named in text
    rms = next(e for e in entries if "RMS (p95)</span>" in e)
    assert "Would exceed peak" in rms and "font-weight:bold" not in rms
    assert "Fader rebalanced" in html


def test_narrow_normalization_keeps_non_finite_signs_and_skip_states():
    processor, result, track = _normalization(detected_rms_db=float("-inf"),
                                              gain_for_rms=float("inf"))
    html = processor.render_html(result, track, narrow=True)
    assert "<nobr>&minus;&infin; dBFS</nobr>" in html and "<nobr>+&infin; dB</nobr>" in html
    processor, result, track = _normalization(classification="Silent")
    assert "<table" not in processor.render_html(result, track, narrow=True)


def test_prepared_report_carries_a_narrow_header():
    track, session = _session()
    data = prepare_track_report(track, session)
    assert "Detected &rarr; Target &middot; Gain" in data["header_narrow"]
    assert "Detected &rarr; Target" not in data["header_compact"]
    assert data["header_narrow"] in render_prepared_track_report(data, narrow=True)


def test_processors_without_narrow_fall_back_to_their_compact_fragment():
    class CompactOnly(BimodalNormalizeProcessor):
        def render_html(self, result, track=None, *, verbose=False, compact=False):
            return "<div>COMPACT</div>" if compact else "<div>WIDE</div>"

    class Legacy(BimodalNormalizeProcessor):
        def render_html(self, result, track=None, *, verbose=False):
            return "<div>LEGACY</div>"

    for cls, expected in ((CompactOnly, "COMPACT"), (Legacy, "LEGACY")):
        processor = cls()
        processor.configure({})
        track, _ = _session()
        session = SessionContext([track], {}, processors=[processor])
        data = prepare_track_report(track, session)
        assert expected in data["header_narrow"]


# ── Stacked findings ─────────────────────────────────────────────────────

def test_narrow_findings_stack_in_one_column():
    track, session = _session()
    data = prepare_track_report(track, session)
    html = render_prepared_track_report(data, narrow=True)
    findings = html[html.index("report-findings"):]
    assert "Finding, location and next step" not in findings  # no column header band
    start = findings.index("finding-tail")
    outer = findings.rindex("<td", 0, findings.rindex("<table", 0, start))
    row = findings[outer:]
    # Severity, title band, location + next step, then inline actions, in order.
    order = [row.index(t) for t in ("Attention", "Tail Regions Exceeded Anchor",
                                    "Check for section-based riding", "report-help:tail")]
    assert order == sorted(order)
    assert "padding:4px 12px 4px 12px" in row[:row.index(">")]  # 12 px row inset
    # 6 px padding centres the title and the actions in their 32 px bands.
    assert re.search(r'<td style="padding:6px 0;">[^<]*<a href="report-details:tail"', row)
    assert re.search(r'<td style="padding:6px 0;"><table.*?report-help:tail', row, re.S)
    assert '<td style="padding-left:24px;' in row
    # No paragraphs/divs between the rows: Qt would add stray empty blocks.
    body = row[:row.index("</table></td></tr>")]
    assert "<p" not in body and "<div" not in body


def test_narrow_details_indent_and_active_help_surface():
    track, session = _session()
    data = prepare_track_report(track, session)
    html = render_prepared_track_report(data, narrow=True, expanded={"tail"}, active_help="tail")
    assert '<td style="padding:4px 12px 4px 36px;"><b>Result</b>' in html
    start = html.index("finding-tail")
    cell = html[html.rindex("<td", 0, html.rindex("<table", 0, start)):start]
    assert f'background-color:{COLORS["surface"]}' in cell[:cell.index(">")]


# ── Browser layout switching ─────────────────────────────────────────────

def _browser(qapp):
    track, session = _session()
    browser = ReportBrowser({})
    browser.resize(900, 4000)  # tall: no vertical scroll bar to move the viewport
    browser.show()
    browser.present_track(track, session)
    qapp.processEvents()
    return browser


def _set_viewport(qapp, browser, width):
    browser.resize(width + browser.width() - browser.viewport().width(), browser.height())
    qapp.processEvents()
    assert browser.viewport().width() == width


def test_browser_switches_to_narrow_with_hysteresis(qapp, monkeypatch):
    browser = _browser(qapp)
    # Isolate the viewport thresholds from font-dependent header measurement.
    monkeypatch.setattr(browser, "_header_width", lambda header: 0)
    _set_viewport(qapp, browser, NARROW_BELOW)
    assert browser._layout == "compact"
    _set_viewport(qapp, browser, NARROW_BELOW - 1)
    assert browser._layout == "narrow"
    assert "Detected → Target" in browser.toPlainText()
    _set_viewport(qapp, browser, NARROW_EXIT_AT - 1)
    assert browser._layout == "narrow"  # stays until the exit width
    _set_viewport(qapp, browser, NARROW_EXIT_AT)
    assert browser._layout == "compact"
    browser.close()


def test_browser_goes_narrow_when_the_compact_header_cannot_fit(qapp, monkeypatch):
    browser = _browser(qapp)
    compact = browser._prepared["header_compact"]
    monkeypatch.setattr(browser, "_header_width",
                        lambda header: 10_000 if header == compact else 0)
    _set_viewport(qapp, browser, 600)
    assert browser._layout == "narrow"
    browser.close()


def test_minimum_header_width_measures_the_narrow_header(qapp):
    browser = _browser(qapp)
    prepared = browser._prepared
    assert browser.minimum_header_width() == browser._header_width(prepared["header_narrow"])
    assert browser.minimum_header_width() <= browser._header_width(prepared["header_compact"])
    browser.close()


# ── Help dock at the pane minimum ───────────────────────────────────────

def test_two_column_help_rows_stack_and_keep_other_rows():
    stacked = stack_two_column_rows(HELP)
    assert stacked.count("<tr>") == 4
    assert "padding-left" not in stacked
    assert stacked.index("&minus;10 dB") < stacked.index("Severe") < stacked.index("&minus;20 dB")
    nested = "<table><tr><td>a</td><td><table><tr><td>x</td></tr></table></td></tr></table>"
    assert stack_two_column_rows(nested) == nested
    three = "<tr><td>a</td><td>b</td><td>c</td></tr>"
    assert stack_two_column_rows(three) == three


@pytest.fixture
def pane(qapp):
    track, session = _session()
    view = FileReportPane(ReportBrowser({}))
    # Like the File splitter's explicit minimum: the pane may go below the
    # layout's hint (report + gap + the current Help width).
    view.setMinimumSize(1, 1)
    view.resize(1069, 620)
    view.show()
    view.report.present_track(track, session)
    qapp.processEvents()
    yield view
    view.close()


def _help_rows(pane):
    return pane.help_browser.toHtml().count("<tr>")


def test_help_tables_stack_only_in_a_narrow_body(qapp, pane, monkeypatch):
    monkeypatch.setattr(pane.report, "minimum_header_width", lambda: 0)
    pane.report.activate_report_link(QUrl("report-help:" + quote("tail", safe="")))
    qapp.processEvents()
    assert pane._help_panel.width() == 360 and _help_rows(pane) == 2
    pane.resize(400, 620)
    qapp.processEvents()
    # 400 px pane: Help min(360, max(140, 160)) = 160; the body is under 240.
    assert pane._help_panel.width() == 160 and _help_rows(pane) == 4
    assert pane.report.width() == 400 - 8 - 160
    assert pane.report._layout == "narrow"
    pane.resize(1069, 620)
    qapp.processEvents()
    assert _help_rows(pane) == 2


def test_narrow_count_leads_the_metadata_instead_of_wrapping_the_title():
    track, session = _session()
    track.detector_results["tail"].data["runs"] = 5
    data = prepare_track_report(track, session)
    html = render_prepared_track_report(data, narrow=True)
    row = html[html.index("finding-tail"):]
    title = row.index("report-details:tail")
    band_end = row.index("</td></tr>", title)  # the 32 px title band
    assert "5 ranges" not in row[:band_end]
    details = row[row.index('<td style="padding-left:24px;'):]
    assert details.index("<nobr>5 ranges</nobr><br>") < details.index("Check for section-based riding")


def test_narrow_rows_leave_out_an_absent_location():
    track, session = _session()  # the tail finding has no sample location
    data = prepare_track_report(track, session)
    assert data["rows"][0]["location"] == "—"
    html = render_prepared_track_report(data, narrow=True)
    row = html[html.index("finding-tail"):]
    assert "—" not in row and "&mdash;" not in row


def _first_visible_text(pane):
    block = pane.help_browser.cursorForPosition(pane.help_browser.rect().topLeft()).block()
    while block.isValid() and not block.text().strip():
        block = block.next()
    return block.text()


def test_restacking_help_keeps_the_reading_passage(qapp, pane, monkeypatch):
    # Repeated, distinct paragraphs: the anchor must find the right copy.
    long_help = "".join(f"<p>Passage {i}: " + "word " * 120 + "</p>" + HELP for i in range(6))
    monkeypatch.setattr(pane.report, "minimum_header_width", lambda: 0)
    row = pane.report.help_row("tail")
    monkeypatch.setattr(pane.report, "help_row",
                        lambda key: dict(row, help=long_help) if key == "tail" else None)
    pane.resize(1069, 300)
    pane.report.activate_report_link(QUrl("report-help:" + quote("tail", safe="")))
    qapp.processEvents()
    browser = pane.help_browser
    document = browser.document()
    target = next(b for b in (document.findBlockByNumber(i) for i in range(document.blockCount()))
                  if b.text().startswith("Passage 3"))
    cursor = browser.textCursor()
    cursor.setPosition(target.position())
    bar = browser.verticalScrollBar()
    bar.setValue(bar.value() + browser.cursorRect(cursor).top())
    qapp.processEvents()
    assert _first_visible_text(pane).startswith("Passage 3")
    for width, stacked in ((400, True), (1069, False)):
        pane.resize(width, 300)
        for _ in range(3):
            qapp.processEvents()
        assert pane._help_stacked is stacked
        assert _first_visible_text(pane).startswith("Passage 3")


def test_kwargs_processor_gets_its_compact_fragment_in_the_narrow_layout():
    class KwargsOnly(BimodalNormalizeProcessor):
        def render_html(self, result, track=None, **kw):
            return "<div>COMPACT</div>" if kw.get("compact") else "<div>WIDE</div>"

    processor = KwargsOnly()
    processor.configure({})
    track, _ = _session()
    data = prepare_track_report(track, SessionContext([track], {}, processors=[processor]))
    assert "COMPACT" in data["header_narrow"] and "WIDE" in data["header"]


def test_failing_layout_variants_fall_back_to_the_wider_fragment():
    class Fragile(BimodalNormalizeProcessor):
        def render_html(self, result, track=None, *, verbose=False, compact=False, narrow=False):
            if narrow:
                raise ValueError("narrow renderer bug")
            return "<div>COMPACT</div>" if compact else "<div>WIDE</div>"

    processor = Fragile()
    processor.configure({})
    track, _ = _session()
    data = prepare_track_report(track, SessionContext([track], {}, processors=[processor]))
    assert "COMPACT" in data["header_compact"] and "COMPACT" in data["header_narrow"]
