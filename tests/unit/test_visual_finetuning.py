"""Change markers, report row density, heading roles and normalization layout."""
import math
import re

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QIcon
from PySide6.QtWidgets import QApplication, QPushButton, QTabWidget, QTreeWidget, QTreeWidgetItem, QWidget

from sessionpreplib.models import DetectorResult, ProcessorResult, SessionContext, Severity, TrackContext
from sessionpreplib.processors.bimodal_normalize import BimodalNormalizeProcessor
from sessionprepgui.change_marker import (
    MARKER_COLUMN_WIDTH, MARKER_DOT, MARKER_GAP, PARENT_TOOLTIP, ChangeMarkerSlot, change_marker_icon,
    install_button_marker, install_tab_marker, is_button_marked, is_tab_marked,
    set_button_marker, set_item_marker, set_tab_marker, use_marker_tab_bar)
from sessionprepgui.detail import report
from sessionprepgui.detail.report import (
    prepare_track_report, render_prepared_track_report, render_summary_html)
from sessionprepgui.detail.report_style import REPORT_SECTION_HEADING
from sessionprepgui.theme import COLORS, STYLESHEET


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    return app


def _rgb(image, x, y):
    color = QColor(image.pixelColor(x, y))
    return color.red(), color.green(), color.blue(), color.alpha()


# ── Change marker ────────────────────────────────────────────────────────

@pytest.mark.parametrize("scale", [1.0, 1.15, 1.5, 2.0])
def test_marker_icon_is_painted_at_the_device_pixel_ratio(qapp, scale):
    pixmap = change_marker_icon().pixmap(QSize(16, 16), scale)
    assert pixmap.devicePixelRatio() == pytest.approx(scale)
    assert pixmap.width() == round(16 * scale)
    image = pixmap.toImage()
    centre = image.width() // 2
    r, g, b, a = _rgb(image, centre, centre)
    assert (r, g, b) == QColor(COLORS["information"]).getRgb()[:3] and a == 255
    assert _rgb(image, 0, 0)[3] == 0  # transparent outside the 7/16 dot
    edge = round(image.width() * 4.5 / 16) - 1  # just outside the dot's left edge
    assert _rgb(image, edge, centre)[3] < 128


def test_marker_icon_turns_white_on_selection(qapp):
    image = change_marker_icon().pixmap(QSize(16, 16), 1.0, QIcon.Selected).toImage()
    assert _rgb(image, 8, 8)[:3] == QColor(COLORS["heading"]).getRgb()[:3]


def test_tree_items_mark_pages_and_explain_parents(qapp):
    tree = QTreeWidget()
    tree.setColumnCount(2)
    parent = QTreeWidgetItem(tree, ["Detectors", ""])
    child = QTreeWidgetItem(parent, ["Silent Files", ""])
    set_item_marker(child, 1, True, "Changed from preset “Default”")
    set_item_marker(parent, 1, True, "Changed from preset “Default”")
    assert not child.icon(1).isNull()
    assert child.toolTip(1) == "Changed from preset “Default”"
    assert child.data(1, Qt.AccessibleDescriptionRole) == child.toolTip(1)
    assert parent.toolTip(1) == PARENT_TOOLTIP
    set_item_marker(child, 1, False, "ignored")
    assert child.icon(1).isNull() and child.toolTip(1) == ""
    assert child.data(1, Qt.AccessibleDescriptionRole) is None
    assert MARKER_COLUMN_WIDTH == 24


def _themed_tabs():
    tabs = QTabWidget()
    tabs.setStyleSheet(STYLESHEET)  # widget-local: the shared tab padding applies
    use_marker_tab_bar(tabs)
    tabs.setDocumentMode(True)
    tabs.addTab(QWidget(), "Groups")
    tabs.addTab(QWidget(), "Config")
    return tabs


def test_tab_marker_needs_the_marker_tab_bar(qapp):
    tabs = QTabWidget()
    tabs.addTab(QWidget(), "Config")
    with pytest.raises(RuntimeError):
        install_tab_marker(tabs, 0)
    with pytest.raises(RuntimeError):
        use_marker_tab_bar(tabs)


def test_tab_marker_dot_sits_right_after_the_label(qapp):
    tabs = _themed_tabs()
    install_tab_marker(tabs, 1)
    tabs.resize(500, 120)
    tabs.show()
    qapp.processEvents()
    bar = tabs.tabBar()
    label = bar.fontMetrics().size(Qt.TextShowMnemonic, "Config").width()
    plain = bar.tabRect(0).width() - bar.fontMetrics().size(Qt.TextShowMnemonic, "Groups").width()
    # A markable but clean tab keeps its normal width.
    assert bar.tabRect(1).width() - label == plain
    set_tab_marker(tabs, 1, True, "Changed")
    qapp.processEvents()
    # Marked tab = normal tab padding + label + 4 px gap + 7 px dot.
    assert bar.tabRect(1).width() - label == plain + MARKER_GAP + MARKER_DOT
    image = bar.grab().toImage()
    rect = bar.tabRect(1)
    ratio = image.devicePixelRatio()
    dot_x = rect.x() + 16 + label + MARKER_GAP + MARKER_DOT / 2
    colour = QColor(image.pixelColor(round(dot_x * ratio), round(rect.center().y() * ratio)))
    assert colour.name() == COLORS["information"]
    tabs.close()


def test_tab_marker_widens_only_while_shown_and_explains_itself(qapp):
    tabs = _themed_tabs()
    install_tab_marker(tabs, 1)
    tabs.show()
    qapp.processEvents()
    width = tabs.tabBar().tabRect(1).width()
    set_tab_marker(tabs, 1, True, "Changed from preset “Default”")
    qapp.processEvents()
    assert is_tab_marked(tabs, 1)
    assert tabs.tabText(1) == "Config"
    assert tabs.tabToolTip(1) == "Changed from preset “Default”"
    assert tabs.tabBar().tabRect(1).width() == width + MARKER_GAP + MARKER_DOT
    set_tab_marker(tabs, 1, True, "again")  # idempotent: no second widening
    qapp.processEvents()
    assert tabs.tabBar().tabRect(1).width() == width + MARKER_GAP + MARKER_DOT
    set_tab_marker(tabs, 1, False, "x")
    qapp.processEvents()
    assert not is_tab_marked(tabs, 1) and tabs.tabToolTip(1) == ""
    assert tabs.tabBar().tabRect(1).width() == width  # back to normal
    tabs.setTabEnabled(1, False)
    set_tab_marker(tabs, 1, True, "x")
    assert not is_tab_marked(tabs, 1)  # disabled tabs never show the marker
    tabs.setTabEnabled(1, True)
    set_tab_marker(tabs, 0, True, "x")  # tabs that were never made markable are ignored
    assert not is_tab_marked(tabs, 0)
    tabs.close()


def test_button_marker_widens_only_while_shown_and_turns_white_when_pressed(qapp):
    button = QPushButton("Save Preferences")
    button.setStyleSheet("QPushButton { color: #123456; }")  # the button's own sheet
    slot = install_button_marker(button)
    button.show()
    qapp.processEvents()
    width = button.sizeHint().width()
    assert not slot.isVisible()
    set_button_marker(button, True, "Unsaved changes")
    qapp.processEvents()
    assert is_button_marked(button)
    assert button.text() == "Save Preferences"
    assert button.toolTip() == "Unsaved changes" == button.accessibleDescription()
    assert button.sizeHint().width() == width + MARKER_GAP + MARKER_DOT
    assert "color: #123456" in button.styleSheet()  # own styling is kept
    assert slot.geometry().right() < button.width() and slot.isVisible()
    assert not slot._on_selection()
    button.setDown(True)
    assert slot._on_selection()
    button.setDown(False)
    set_button_marker(button, False, "Unsaved changes")
    qapp.processEvents()
    assert not is_button_marked(button) and button.toolTip() == ""
    assert button.styleSheet() == "QPushButton { color: #123456; }"
    assert button.sizeHint().width() == width and not slot.isVisible()
    button.close()


def test_marker_slot_paints_nothing_when_clean_or_disabled(qapp):
    slot = ChangeMarkerSlot()
    clean = slot.grab().toImage()
    slot.set_marked(True)
    marked = slot.grab().toImage()
    slot.setEnabled(False)
    disabled = slot.grab().toImage()
    centre = (MARKER_GAP + 3, 8)
    assert clean.pixelColor(*centre) == disabled.pixelColor(*centre)
    assert marked.pixelColor(*centre) != clean.pixelColor(*centre)


# ── Report rows and headings ─────────────────────────────────────────────

def _track(**kw):
    track = TrackContext("Lead Vocal.wav", "missing.wav", None, 48000, 1, 96000,
                         "24-bit", "PCM_24", 2.0)
    for key, value in kw.items():
        setattr(track, key, value)
    return track


def _normalized_session(classification="Sustained", method="RMS → -16 dB", **data):
    processor = BimodalNormalizeProcessor()
    processor.configure({})
    values = {"detected_peak_db": -15.0, "detected_rms_db": -21.0, "rms_anchor_label": "p95",
              "target_peak": -6.0, "target_rms": -16.0, "gain_for_peak": 9.0,
              "gain_for_rms": 5.0, "fader_offset": -3.0}
    values.update(data)
    track = _track()
    track.processor_results[processor.id] = ProcessorResult(
        processor.id, 5.0, classification, method, values)
    track.detector_results["clipping"] = DetectorResult(
        "clipping", Severity.PROBLEM, "clipping detected", {}, hint="Check limiting")
    return processor, track, SessionContext([track], {}, processors=[processor])


def test_summary_and_file_rows_use_four_pixel_padding_and_32_px_lines(monkeypatch):
    monkeypatch.setattr(report, "read_chunks", lambda path: ("WAVE", []))
    summary = {"problems": [{"title": "Clipping", "hint": None, "items": ["Kick.wav"]}],
               "attention": [], "information": [], "clean": [],
               "clean_count": 0, "total_ok": 1, "overview": {}}
    html = render_summary_html(summary)
    assert not re.search(r"padding:8px", html)
    assert "padding:4px 6px 4px 12px" in html
    _processor, track, session = _normalized_session()
    data = prepare_track_report(track, session)
    for compact in (False, True):
        file_html = render_prepared_track_report(data, compact=compact)
        assert 'height="40"' not in file_html
        assert 'table height="32"' in file_html
        assert not re.search(r"padding:8px \d+px 8px", file_html)
    expanded = render_prepared_track_report(data, expanded={"clipping"})
    assert "padding:4px 12px 4px 136px" in expanded


def test_file_title_and_sections_use_shared_roles(monkeypatch):
    monkeypatch.setattr(report, "read_chunks", lambda path: ("WAVE", []))
    _processor, track, session = _normalized_session()
    data = prepare_track_report(track, session)
    html = render_prepared_track_report(data)
    assert "font-size:x-large; font-weight:600;" in html  # title role
    headings = re.findall(rf'<p style="{re.escape(REPORT_SECTION_HEADING)}[^"]*">(?:<a name="[^"]+"></a>)?([^<]+)</p>', html)
    assert headings == ["Bimodal Normalization", "Detector findings"]
    assert "margin-left:8px" not in data["header"]


def test_report_browser_keeps_heading_roles_and_finding_body_size(qapp, monkeypatch):
    from sessionprepgui.detail.browser import ReportBrowser

    monkeypatch.setattr(report, "read_chunks", lambda path: ("WAVE", []))
    _processor, track, session = _normalized_session()
    browser = ReportBrowser({})
    browser.present_track(track, session)
    size = lambda text: browser.document().find(text).charFormat().font().pointSizeF()
    body = browser.font().pointSizeF()
    assert size("Lead Vocal.wav") == pytest.approx(body * 1.5)
    assert size("Bimodal Normalization") == pytest.approx(body)
    assert size("Detector findings") == pytest.approx(body)
    # Normalization content uses the finding body role, like the findings table.
    assert not browser.document().find("Problem").isNull()
    assert size("Chosen") == pytest.approx(size("Problem"))
    assert size("Chosen") < body


# ── Normalization ────────────────────────────────────────

def _render(classification, method, compact=False, **data):
    processor, track, _session = _normalized_session(classification, method, **data)
    return processor.render_html(track.processor_results[processor.id], track,
                                 verbose=False, compact=compact)


def test_normalization_names_the_chosen_measure_in_text():
    html = _render("Sustained", "RMS → -16 dB", fader_rebalance_shift=2.0)
    assert "RMS &rarr; &minus;16.0 dBFS" in html
    assert re.search(r"RMS \(p95\)</td>.*Chosen", html)
    for value in ("&minus;15.0 dBFS", "&minus;6.0 dBFS", "+9.0 dB", "&minus;21.0 dBFS", "&minus;16.0 dBFS", "+5.0 dB"):
        assert value in html
    assert "background-color:#2d2d2d" in html  # findings-table header band
    assert "Fader rebalanced: +2.0 dB shift &rarr; fader offset &minus;3.0 dB" in html
    assert "#cc8844" not in html and "#44cc44" not in html
    assert 'color:#44cccc; font-weight:600;">Chosen' in html


def test_normalization_transient_and_peak_limited_notes():
    transient = _render("Transient", "Peak → -6 dB")
    assert "Peak &rarr; &minus;6.0 dBFS" in transient
    assert re.search(r"Peak</td>.*Chosen</span>", transient)
    limited = _render("Sustained", "Peak Limited")
    assert "Peak (limited) &rarr; &minus;6.0 dBFS" in limited
    assert "Chosen &middot; limiting" in limited and "Would exceed peak" in limited


@pytest.mark.parametrize("classification", ["Skip", "Silent", "Unknown"])
def test_normalization_skip_states_are_one_line(classification):
    html = _render(classification, "None")
    assert html.count("<div") == 1 and "<table" not in html
    assert classification in html and "No normalization" in html


def test_normalization_non_finite_values_keep_their_sign():
    html = _render("Transient", "Peak → -6 dB", detected_rms_db=-math.inf,
                   gain_for_rms=math.inf)
    assert "&minus;&infin; dBFS" in html
    assert "+&infin; dB" in html


def test_compact_normalization_uses_percentage_columns_and_note_under_measure(monkeypatch):
    html = _render("Sustained", "Peak Limited", compact=True)
    assert 'table width="100%"' in html
    assert 'width="30%"' in html and 'width="150"' not in html
    assert re.search(r"Peak<br><span[^>]*>Chosen &middot; limiting", html)
    monkeypatch.setattr(report, "read_chunks", lambda path: ("WAVE", []))
    _processor, track, session = _normalized_session(method="Peak Limited")
    data = prepare_track_report(track, session)
    assert 'width="100%"' in data["header_compact"] and 'width="100%"' not in data["header"]
    assert data["header_compact"] in render_prepared_track_report(data, compact=True)
    assert data["header"] in render_prepared_track_report(data, compact=False)


def test_modified_icon_copies_stay_valid_after_collection(qapp):
    import gc
    from PySide6.QtGui import QPixmap

    copy = QIcon(change_marker_icon())
    extra = QPixmap(16, 16)
    extra.fill(Qt.transparent)
    copy.addPixmap(extra)  # detaches: Qt clones the engine
    gc.collect()
    assert not copy.pixmap(QSize(16, 16)).isNull()
    assert not change_marker_icon().pixmap(QSize(16, 16), 2.0).isNull()


def test_disabled_marker_is_dimmed(qapp):
    image = change_marker_icon().pixmap(QSize(16, 16), 1.0, QIcon.Disabled).toImage()
    assert _rgb(image, 8, 8)[:3] == QColor(COLORS["dim"]).getRgb()[:3]


def test_tab_mark_moves_with_its_tab(qapp):
    tabs = _themed_tabs()
    install_tab_marker(tabs, 1)
    set_tab_marker(tabs, 1, True, "Changed")
    tabs.insertTab(0, QWidget(), "Summary")
    assert not is_tab_marked(tabs, 1)
    assert is_tab_marked(tabs, 2) and tabs.tabText(2) == "Config"


def test_legacy_processor_without_compact_keeps_the_file_report(monkeypatch):
    from sessionpreplib.processor import AudioProcessor

    class Legacy(BimodalNormalizeProcessor):
        def render_html(self, result, track=None, *, verbose=False):  # pre-compact signature
            return "<div>LEGACY</div>"

    monkeypatch.setattr(report, "read_chunks", lambda path: ("WAVE", []))
    processor = Legacy()
    processor.configure({})
    assert isinstance(processor, AudioProcessor)
    track = _track()
    track.processor_results[processor.id] = ProcessorResult(processor.id, 0.0, "Sustained", "RMS", {})
    data = prepare_track_report(track, SessionContext([track], {}, processors=[processor]))
    assert "LEGACY" in data["header"] and "LEGACY" in data["header_compact"]


def test_group_line_uses_metadata_role(monkeypatch):
    monkeypatch.setattr(report, "read_chunks", lambda path: ("WAVE", []))
    _processor, track, session = _normalized_session()
    track.group = "Vox"
    data = prepare_track_report(track, session)
    assert f'<div style="color:{COLORS["metadata"]}; margin-top:8px;">Group: <b>Vox</b>' in data["header"]


def test_section_tables_carry_the_heading_gap(monkeypatch):
    # Qt drops a heading's bottom margin before a table, so tables carry 8 px.
    monkeypatch.setattr(report, "read_chunks", lambda path: ("WAVE", []))
    summary = {"problems": [{"title": "Clipping", "hint": None, "items": ["Kick.wav"]}],
               "attention": [], "information": [], "clean": [],
               "clean_count": 0, "total_ok": 1, "overview": {}}
    assert 'style="margin-top:8px; border-collapse:collapse;"' in render_summary_html(summary)
    _processor, track, session = _normalized_session()
    html = render_prepared_track_report(prepare_track_report(track, session))
    findings = html[html.index("Detector findings"):]
    assert findings.index('margin-top:8px; border-collapse:collapse;') < findings.index("Severity")


def test_report_browser_gap_below_findings_heading(qapp, monkeypatch):
    from sessionprepgui.detail.browser import ReportBrowser

    monkeypatch.setattr(report, "read_chunks", lambda path: ("WAVE", []))
    _processor, track, session = _normalized_session()
    browser = ReportBrowser({})
    browser.resize(1000, 700)
    browser.show()
    browser.present_track(track, session)
    qapp.processEvents()
    rect = lambda text: browser.cursorRect(browser.document().find(text))
    heading_gap = rect("Severity").top() - rect("Detector findings").bottom()
    summary_gap = rect("Measure").top() - rect("Sustained").bottom()
    assert heading_gap == summary_gap >= 8 + 4  # 8 px table margin + 4 px cell padding
    browser.close()


def test_sample_ranges_stay_together_and_compact_details_align(monkeypatch):
    monkeypatch.setattr(report, "read_chunks", lambda path: ("WAVE", []))
    _processor, track, session = _normalized_session()
    track.detector_results["clipping"].data["ranges"] = [(1000, 1004, 0)]
    data = prepare_track_report(track, session)
    for compact in (False, True):
        html = render_prepared_track_report(data, compact=compact, expanded={"clipping"})
        assert "Samples <nobr>1000–1004</nobr>" in html
        assert "padding:4px 12px 4px 136px;" in html


def test_normalization_cells_are_never_empty():
    # Empty cells keep the larger report line height and would enlarge the band.
    for compact in (False, True):
        html = _render("Transient", "Peak → -6 dB", compact=compact)
        assert not re.search(r'<td[^>]*></td>', html)


@pytest.mark.parametrize("css", [None, "", "QTabBar::tab { padding: 6px 24px; margin-right: 6px; }"])
def test_tab_marker_geometry_follows_the_style(qapp, css):
    tabs = QTabWidget()
    if css is not None:  # None = plain Fusion without the app stylesheet
        tabs.setStyleSheet(STYLESHEET + "\n" + css)
    use_marker_tab_bar(tabs)
    tabs.setDocumentMode(True)
    tabs.addTab(QWidget(), "Config")
    tabs.addTab(QWidget(), "Config")  # same label, unreserved: the plain tab
    install_tab_marker(tabs, 0)
    set_tab_marker(tabs, 0, True, "Changed")
    tabs.resize(500, 120)
    tabs.show()
    qapp.processEvents()
    bar = tabs.tabBar()
    assert bar.tabRect(0).width() == bar.tabRect(1).width() + MARKER_GAP + MARKER_DOT
    tabs.close()


def test_large_report_fonts_switch_to_compact_instead_of_scrolling(qapp, monkeypatch):
    from PySide6.QtGui import QFont
    from sessionprepgui.detail.browser import ReportBrowser

    monkeypatch.setattr(report, "read_chunks", lambda path: ("WAVE", []))
    _processor, track, session = _normalized_session(method="Peak Limited")
    browser = ReportBrowser({})
    browser.resize(900, 600)
    browser.show()
    # Grow the font until the wide header genuinely needs more than the widget
    # (font-independent: platform fonts differ between Windows and CI).
    for points in range(12, 80, 2):
        font = QFont(browser.font())
        font.setPointSizeF(float(points))
        browser.setFont(font)
        browser.document().setDefaultFont(font)
        browser.present_track(track, session)
        qapp.processEvents()
        if browser._header_width(browser._prepared["header"]) > browser.width():
            break
    else:
        pytest.skip("the wide header never exceeded the widget width")
    assert browser.viewport().width() >= 850  # the fixed threshold alone would stay wide
    assert browser._layout in ("compact", "narrow")
    if browser.minimum_header_width() <= browser.width():
        assert browser.horizontalScrollBar().maximum() == 0
    browser.close()


def test_normal_report_font_keeps_the_wide_layout_at_900_px(qapp, monkeypatch):
    from sessionprepgui.detail.browser import ReportBrowser

    monkeypatch.setattr(report, "read_chunks", lambda path: ("WAVE", []))
    _processor, track, session = _normalized_session()
    browser = ReportBrowser({})
    browser.resize(900, 600)
    browser.show()
    browser.present_track(track, session)
    qapp.processEvents()
    assert browser._layout == "wide"
    browser.resize(800, 600)
    qapp.processEvents()
    assert browser._layout == "compact"  # the existing 850 px viewport threshold still applies
    browser.close()


def test_pressed_save_marker_is_painted_white(qapp):
    button = QPushButton("Save Preferences")
    button.setStyleSheet(STYLESHEET)
    slot = install_button_marker(button)
    button.show()
    set_button_marker(button, True, "Unsaved changes")
    qapp.processEvents()
    centre = (MARKER_GAP + 3, 8)
    idle = slot.grab().toImage()
    assert QColor(idle.pixelColor(*centre)).name() == COLORS["information"]
    button.setDown(True)
    pressed = slot.grab().toImage()
    assert QColor(pressed.pixelColor(*centre)).name() == COLORS["heading"]
    button.setDown(False)
    button.close()


@pytest.mark.parametrize("css", [None, ""])
def test_marking_a_visible_tab_relayouts_to_the_exact_width(qapp, css):
    # Regression: the spacer is resized after Qt laid the tab out with its
    # initial width; the bar must lay the tabs out again.
    tabs = QTabWidget()
    if css is not None:
        tabs.setStyleSheet(STYLESHEET + css)
    use_marker_tab_bar(tabs)
    tabs.setDocumentMode(True)
    for label in ("Summary", "File", "Groups", "Config"):
        tabs.addTab(QWidget(), label)
    install_tab_marker(tabs, 3)
    tabs.resize(600, 120)
    tabs.show()
    qapp.processEvents()
    bar = tabs.tabBar()
    width = bar.tabRect(3).width()
    set_tab_marker(tabs, 3, True, "Changed")
    assert bar.tabRect(3).width() == width + MARKER_GAP + MARKER_DOT
    qapp.processEvents()
    assert bar.tabRect(3).width() == width + MARKER_GAP + MARKER_DOT
    tabs.close()


def test_zebra_rows_use_the_subtle_alt_row_role(qapp):
    from PySide6.QtGui import QPalette
    from sessionprepgui.theme import apply_dark_theme

    saved = QPalette(qapp.palette())
    try:
        apply_dark_theme(QWidget())
        palette = qapp.palette()
        assert COLORS["alt_row"] == "#2c2c2c"
        assert palette.color(QPalette.AlternateBase).name() == COLORS["alt_row"]
        assert palette.color(QPalette.Base).name() == COLORS["bg_alt"]
        # Accent keeps hover/control surfaces; it is no longer the zebra colour.
        assert palette.color(QPalette.Button).name() == COLORS["accent"] != COLORS["alt_row"]
    finally:
        qapp.setPalette(saved)


def test_unmarking_keeps_stylesheet_changes_made_after_install(qapp):
    button = QPushButton("Save Preferences")
    button.setStyleSheet("QPushButton { color: red; }")
    install_button_marker(button)
    button.setStyleSheet("QPushButton { color: blue; }")  # changed after install
    set_button_marker(button, True, "Unsaved changes")
    assert "color: blue" in button.styleSheet() and "padding-right" in button.styleSheet()
    set_button_marker(button, False, "")
    assert button.styleSheet() == "QPushButton { color: blue; }"
    bare = QPushButton("Save")
    install_button_marker(bare)
    set_button_marker(bare, True, "x")
    set_button_marker(bare, False, "")
    assert bare.styleSheet() == ""
