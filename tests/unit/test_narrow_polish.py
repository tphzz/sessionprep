"""Narrow transport, placeholder painting, field widths, File column, Help tables."""
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QTableWidgetItem

from sessionprepgui.detail.help_panel import normalize_help_tables, stack_two_column_rows
from sessionprepgui.prefs.param_form import LINE_EDIT_MAX_MIN_WIDTH, LINE_EDIT_MIN_WIDTH, _build_widget
from sessionprepgui.tracks.table_widgets import _DraggableTrackTable
from sessionprepgui.waveform.panel import WaveformPanel
from sessionpreplib.config import ParamSpec


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    return app


# ── Transport ────────────────────────────────────────────────────────────

def test_play_and_stop_become_icon_only_when_the_transport_does_not_fit(qapp):
    panel = WaveformPanel(analysis_mode=True)
    # Like the pane in its splitter: the panel may be narrower than its hint.
    panel.setMinimumSize(1, 1)
    panel.resize(1100, 260)
    panel.show()
    qapp.processEvents()
    assert not panel._transport_compact and panel.play_btn.text() == "Play"
    assert panel.play_btn.toolTip() == "Play" and panel.stop_btn.toolTip() == "Stop"
    assert panel._play_mode_combo.minimumWidth() == 120
    full = panel._transport_full_width()
    panel.resize(full - 20, 260)
    qapp.processEvents()
    assert panel._transport_compact
    assert panel.play_btn.text() == "" and panel.play_btn.width() == 32
    assert panel.play_btn.accessibleName() == "Play" and panel.play_btn.toolTip() == "Play"
    # Nothing overlaps the time readout.
    combo, time = panel._play_mode_combo, panel.time_label
    assert combo.geometry().right() < time.geometry().left()
    panel.resize(full + 4, 260)   # inside the hysteresis band: stays compact
    qapp.processEvents()
    assert panel._transport_compact
    panel.resize(full + 40, 260)
    qapp.processEvents()
    assert not panel._transport_compact and panel.stop_btn.text() == "Stop"
    panel.close()


# ── Placeholder ──────────────────────────────────────────────────────────

def test_placeholder_is_painted_at_the_current_size_while_resizing(qapp):
    panel = WaveformPanel(analysis_mode=True)
    panel.resize(800, 260)
    panel.show()
    qapp.processEvents()
    widget = panel.waveform
    reference = widget.grab().toImage()
    # Simulate an in-progress resize with a stale image of another size.
    stale = QPixmap(200, 50)
    stale.fill(Qt.red)
    widget._is_resizing = True
    widget._stale_pixmap = stale
    during = widget.grab().toImage()
    widget._is_resizing = False
    widget._stale_pixmap = None
    assert during == reference   # no stretched cached image, same text
    panel.close()


# ── Field widths ─────────────────────────────────────────────────────────

def test_string_fields_fit_their_value(qapp):
    spec = ParamSpec(key="phase1_output_folder", type=str, default="sp_01_tracklayout",
                     label="Phase 1: track layout folder name", description="")
    field = _build_widget(spec, "sp_01_tracklayout")
    text = field.fontMetrics().horizontalAdvance("sp_01_tracklayout")
    assert field.minimumWidth() >= max(LINE_EDIT_MIN_WIDTH, text + 24)
    long_value = "x" * 200
    long_field = _build_widget(spec, long_value)
    assert long_field.minimumWidth() == LINE_EDIT_MAX_MIN_WIDTH   # scrolls inside, no huge dialog


# ── File column ──────────────────────────────────────────────────────────

def _table(qapp, width):
    table = _DraggableTrackTable()
    table.setColumnCount(8)
    header = table.horizontalHeader()
    for col, size in enumerate((0, 30, 150, 120, 90, 100, 140, 130)):
        if col:
            header.resizeSection(col, size)
    table.setRowCount(2)
    table.setItem(0, 0, QTableWidgetItem("LeadVocal_Comp_Final_v12.wav"))
    table.resize(width, 300)
    table.show()
    qapp.processEvents()
    table.fit_file_column()
    qapp.processEvents()
    return table


def test_file_column_takes_the_free_width(qapp):
    table = _table(qapp, 1400)
    header = table.horizontalHeader()
    others = sum(header.sectionSize(c) for c in range(1, 8))
    assert header.sectionSize(0) == table.viewport().width() - others
    assert table.horizontalScrollBar().maximum() == 0
    table.close()


def test_narrow_pane_keeps_160_px_names_and_scrolls(qapp):
    table = _table(qapp, 500)
    assert table.horizontalHeader().sectionSize(0) == 160
    assert table.horizontalScrollBar().maximum() > 0
    assert table.textElideMode() == Qt.ElideMiddle   # extension stays visible
    table.close()


# ── Help tables ──────────────────────────────────────────────────────────

HELP = ("<b>Sensitivity guide</b><br/>Intro:<br/>"
        "<table style='margin:4px 0; font-size:8pt;'>"
        "<tr><td><b>-10 dB</b></td><td style='padding-left:8px;'>Severe</td></tr>"
        "<tr><td><b>-20 dB</b></td><td style='padding-left:8px;'>Default</td></tr>"
        "</table><br/><b>Per-channel analysis</b><br/>Text")


def test_help_tables_get_even_spacing_and_row_separators():
    html = normalize_help_tables(HELP)
    assert "<br/><table" not in html and "</table><br/>" not in html
    assert "font-size:8pt" not in html
    assert 'style="margin-top:8px; margin-bottom:8px; border-collapse:collapse;"' in html
    assert html.count("border-bottom:1px solid") == 4          # every cell of both rows
    assert "padding-left:8px" in html                           # meaning indent kept


def test_stacked_help_rows_have_one_separator_per_pair():
    stacked = stack_two_column_rows(normalize_help_tables(HELP))
    assert stacked.count("<tr>") == 4
    assert stacked.count("border-bottom:1px solid") == 2


def test_help_cells_replace_existing_separators_and_cover_header_cells():
    html = normalize_help_tables(
        "<table><tr><th>Term</th><td style=\"border-bottom:2px solid red; color:#fff\">x</td></tr></table>")
    assert html.count("border-bottom:") == 2            # one each for th and td, red one replaced
    assert "2px solid red" not in html and "color:#fff" in html
    quoted = normalize_help_tables("<table><tr><td style='font-family:\"Segoe UI\"'>x</td></tr></table>")
    assert "style='font-family:\"Segoe UI\"; border-bottom" in quoted


def test_user_cannot_drag_the_auto_fitted_file_column_out_of_sync(qapp):
    from PySide6.QtWidgets import QHeaderView
    from sessionprepgui.tracks.table_layout import apply_track_table_layout
    table = _table(qapp, 1400)
    apply_track_table_layout(table)
    assert table.horizontalHeader().sectionResizeMode(0) == QHeaderView.Fixed
    table.close()
