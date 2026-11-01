"""Behavior of the shared desktop action and preview controls."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QToolBar

from sessionprepgui.controls import bind_primary_action, set_action_busy
from sessionprepgui.icons import icon


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    return app


def test_busy_action_stays_disabled_and_restores_label(qapp):
    toolbar = QToolBar()
    action = QAction("Apply", toolbar)
    toolbar.addAction(action)
    button = bind_primary_action(toolbar, action)
    calls = []
    action.triggered.connect(lambda: calls.append(True))
    toolbar.show()
    set_action_busy(action, True, "Applying…")
    assert not action.isEnabled()
    assert button.text() == "Applying…"
    assert button.icon().isNull()
    assert action.icon().isNull()
    assert button.toolButtonStyle() == Qt.ToolButtonTextBesideIcon
    action.setEnabled(True)  # eligibility refresh must not re-enable a worker
    assert not action.isEnabled()
    QTest.mouseClick(button, Qt.LeftButton)
    assert calls == []
    set_action_busy(action, False)
    assert action.text() == "Apply"
    assert not action.isEnabled()  # lifecycle code owns eligibility
    action.setEnabled(True)
    QTest.mouseClick(button, Qt.LeftButton)
    assert calls == [True]
    toolbar.close()



@pytest.mark.parametrize("idle,running", [
    ("Apply", "Applying…"), ("Prepare", "Preparing…"), ("Create", "Creating…"),
])
def test_busy_action_restores_icon_across_updates_and_cycles(qapp, idle, running):
    toolbar = QToolBar()
    action = QAction(icon("play"), idle, toolbar)
    toolbar.addAction(action)
    button = bind_primary_action(toolbar, action)
    original = action.icon().pixmap(16, 16).toImage()
    set_action_busy(action, True, running)
    set_action_busy(action, True, running)
    assert action.icon().isNull() and button.icon().isNull()
    assert button.text() == running
    assert button.property("primaryBusy")
    set_action_busy(action, False)
    assert action.text() == idle
    assert action.icon().pixmap(16, 16).toImage() == original
    assert not button.property("primaryBusy")
    action.setIcon(icon("stop"))
    second = action.icon().pixmap(16, 16).toImage()
    set_action_busy(action, True, running)
    set_action_busy(action, False)
    assert action.icon().pixmap(16, 16).toImage() == second
    assert not action.isEnabled()


def test_busy_binding_refreshes_when_text_and_icon_do_not_change(qapp):
    toolbar = QToolBar()
    action = QAction("Apply", toolbar)
    action.setEnabled(False)
    toolbar.addAction(action)
    button = bind_primary_action(toolbar, action)
    set_action_busy(action, True)
    assert button.text() == "Apply"
    assert action.icon().isNull() and button.icon().isNull()
    assert button.property("primaryBusy")
    set_action_busy(action, False)
    assert not button.property("primaryBusy")
    assert not action.isEnabled()


def test_primary_action_resizes_for_label_and_keeps_focus(qapp):
    toolbar = QToolBar()
    action = QAction("Create", toolbar)
    toolbar.addAction(action)
    button = bind_primary_action(toolbar, action)
    first_width = button.width()
    action.setText("Enqueue >>")
    assert button.width() > first_width
    assert button.width() >= button.fontMetrics().horizontalAdvance("Enqueue >>") + 16
    assert button.height() == 32
    assert button.focusPolicy() == Qt.StrongFocus
    assert button.accessibleName() == "Enqueue >>"


@pytest.mark.parametrize("name", [
    "waveform", "fit", "zoom-in", "zoom-out", "scale-up", "scale-down",
    "play", "stop", "reset", "chevron-down", "chevron-right", "chevron-up",
    "plus", "minus", "refresh", "busy", "spin-up", "spin-down",
])
def test_vector_icons_are_available(qapp, name):
    assert not icon(name).isNull()
    assert not icon(name).pixmap(20, 20).isNull()


def test_preview_collapse_preserves_widget_and_audio_state(qapp, monkeypatch):
    from sessionprepgui.waveform.panel import WaveformPanel
    panel = WaveformPanel(analysis_mode=False)
    waveform = panel.waveform
    monkeypatch.setattr(waveform, "set_audio", lambda *a: pytest.fail("visibility reset audio"))
    monkeypatch.setattr(waveform, "zoom_fit", lambda: pytest.fail("visibility reset viewport"))
    panel.show()
    panel.set_expanded(False)
    assert panel.disclosure_button.isVisible()
    assert not panel._body.isVisible()
    panel.set_content_available(True)  # late worker completion
    assert not panel.is_expanded()
    panel.set_expanded(True)
    assert panel.waveform is waveform
    assert panel._body.isVisible()
    panel.close()


def test_preview_reopens_with_same_splitter_sizes(qapp):
    from PySide6.QtWidgets import QSplitter, QWidget
    from sessionprepgui.waveform.panel import WaveformPanel
    splitter = QSplitter(Qt.Vertical)
    splitter.addWidget(QWidget())
    panel = WaveformPanel()
    splitter.addWidget(panel)
    splitter.resize(800, 700)
    splitter.show()
    splitter.setSizes([420, 280])
    qapp.processEvents()
    before = splitter.sizes()
    panel.set_expanded(False)
    qapp.processEvents()
    assert panel.height() <= panel.preview_header.height()
    panel.set_expanded(True)
    qapp.processEvents()
    assert splitter.sizes() == before
    splitter.close()


def test_primary_busy_is_cleared_when_workspace_cancels_worker(qapp, monkeypatch):
    from unittest.mock import MagicMock
    from sessionprepgui import mainwindow
    from sessionprepgui.settings import build_defaults
    monkeypatch.setattr(mainwindow, "load_config", build_defaults)
    window = mainwindow.SessionPrepWindow()
    try:
        worker = MagicMock()
        worker.isRunning.return_value = False
        window._prepare_worker = worker
        set_action_busy(window._prepare_action, True, "Preparing…")
        window._clear_workspace()
        assert not window._prepare_action.property("primaryBusy")
        window._prepare_action.setEnabled(True)
        assert window._prepare_action.isEnabled()
    finally:
        window.close()


def test_table_selectors_reserve_label_gap_and_chevron(qapp):
    from sessionprepgui.widgets import TableCellComboBox, BatchToolButton
    combo = TableCellComboBox()
    combo.addItem("Default")
    button = BatchToolButton()
    button.setText("Default")
    for control in (combo, button):
        required = control.fontMetrics().horizontalAdvance("Default") + 42
        assert control.sizeHint().width() >= required


def test_selected_metadata_keeps_focus_and_restores_rich_status(qapp):
    from PySide6.QtGui import QColor, QPalette
    from PySide6.QtWidgets import QTableWidgetItem, QStyle, QStyleOptionViewItem
    from sessionprepgui.widgets import BatchEditTableWidget
    from sessionprepgui.tracks.table_widgets import _make_analysis_cell
    table = BatchEditTableWidget()
    table.setColumnCount(2)
    table.setRowCount(1)
    item = QTableWidgetItem("48000")
    item.setForeground(QColor("#aaaaaa"))
    table.setItem(0, 0, item)
    html = '<span style="color:#ffaa00">1A</span>'
    label, hidden = _make_analysis_cell(html, 1)
    table.setItem(0, 1, hidden)
    table.setCellWidget(0, 1, label)
    table.show()
    table.selectRow(0)
    qapp.processEvents()
    option = QStyleOptionViewItem()
    option.state = QStyle.State_Selected | QStyle.State_HasFocus
    table.itemDelegate().initStyleOption(option, table.model().index(0, 0))
    assert option.palette.color(QPalette.Text) == QColor("#ffffff")
    assert option.state & QStyle.State_HasFocus
    assert label.text() == "1A"
    assert label.palette().color(QPalette.WindowText) == QColor("#ffffff")
    table.clearSelection()
    qapp.processEvents()
    assert label.text() == html
    assert item.foreground().color() == QColor("#aaaaaa")
    table.close()


def test_splitter_cannot_hide_the_preview_disclosure(qapp):
    from PySide6.QtWidgets import QSplitter, QWidget
    from sessionprepgui.waveform.panel import WaveformPanel
    splitter = QSplitter(Qt.Vertical)
    splitter.addWidget(QWidget())
    panel = WaveformPanel()
    splitter.addWidget(panel)
    splitter.resize(800, 700)
    splitter.show()
    panel.set_expanded(False)
    splitter.setSizes([700, 0])
    qapp.processEvents()
    assert not splitter.isCollapsible(1)
    assert panel.height() >= 32
    assert panel.disclosure_button.isVisible()
    splitter.close()


def test_selector_label_fits_reserved_chevron_after_theme_polish(qapp):
    from PySide6.QtWidgets import QWidget, QStyle, QStyleOptionComboBox
    from sessionprepgui.widgets import TableCellComboBox
    from sessionprepgui.theme import apply_dark_theme
    parent = QWidget()
    apply_dark_theme(parent)
    combo = TableCellComboBox(parent)
    combo.addItem("Transient")
    combo.setStyleSheet("QComboBox { font-weight:bold; }")
    combo.ensurePolished()
    combo.resize(combo.sizeHint())
    option = QStyleOptionComboBox()
    combo.initStyleOption(option)
    rect = combo.style().subControlRect(
        QStyle.CC_ComboBox, option, QStyle.SC_ComboBoxEditField, combo)
    assert rect.width() >= combo.fontMetrics().horizontalAdvance(combo.currentText())


def test_selected_status_tracks_identity_through_sort_and_reselection(qapp):
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import QTableWidgetItem
    from sessionprepgui.widgets import BatchEditTableWidget
    from sessionprepgui.tracks.table_widgets import _make_analysis_cell
    table = BatchEditTableWidget()
    table.setColumnCount(2)
    table.setRowCount(3)
    html = '<span style="color:#ffaa00">1A</span>'
    for row, name in enumerate(("A", "B", "C")):
        table.setItem(row, 0, QTableWidgetItem(name))
        label, item = _make_analysis_cell(html, 1)
        table.setItem(row, 1, item)
        table.setCellWidget(row, 1, label)
        table.apply_row_color(row, QColor("#334433"))
    table.selectRow(0)
    table.setSortingEnabled(True)
    table.sortItems(0, Qt.DescendingOrder)
    assert table.item(2, 0).text() == "A"
    assert table.cellWidget(2, 1).property("selectedRow") is True
    table.selectRow(1)
    assert table.cellWidget(2, 1).text() == html
    assert table.cellWidget(2, 1).property("selectedRow") is False
    assert "background-color" in table.cellWidget(2, 1).styleSheet()
    assert table.cellWidget(1, 1).text() == "1A"
    table.close()


def test_selected_disabled_gain_preserves_foreground_and_numeric_state(qapp):
    from PySide6.QtCore import QLocale
    from PySide6.QtGui import QColor, QPalette
    from PySide6.QtWidgets import QStyle, QStyleOptionSpinBox, QTableWidgetItem
    from sessionprepgui.theme import apply_dark_theme
    from sessionprepgui.widgets import BatchEditTableWidget, TableCellDoubleSpinBox

    table = BatchEditTableWidget()
    table.setColumnCount(2)
    table.setRowCount(2)
    table.setItem(0, 0, QTableWidgetItem("B"))
    table.setItem(1, 0, QTableWidgetItem("A"))
    spin = TableCellDoubleSpinBox()
    spin.setRange(-60.0, 60.0)
    spin.setDecimals(1)
    spin.setSingleStep(0.1)
    spin.setSuffix(" dB")
    spin.setLocale(QLocale(QLocale.German, QLocale.Germany))
    spin.setValue(-2.0)
    table.setCellWidget(0, 1, spin)
    apply_dark_theme(table)
    changes = []
    spin.valueChanged.connect(changes.append)
    table.show()
    try:
        qapp.processEvents()
        table.clearSelection()
        initial_text = spin.palette().color(QPalette.Active, QPalette.Text)
        spin.setEnabled(False)
        qapp.processEvents()
        disabled_text = spin.palette().color(QPalette.Disabled, QPalette.Text)
        table.selectRow(0)
        qapp.processEvents()
        assert spin.palette().color(QPalette.Disabled, QPalette.Text) == disabled_text
        spin.setEnabled(True)
        qapp.processEvents()
        selected_text = table.palette().color(
            QPalette.Active, QPalette.HighlightedText)
        assert spin.palette().color(QPalette.Active, QPalette.Text) == selected_text
        spin.setEnabled(False)
        for tint in (QColor("#334433"), QColor("#443333"), None):
            table.apply_row_color(0, tint)
            qapp.processEvents()
            assert spin.palette().color(QPalette.Disabled, QPalette.Text) == disabled_text
        table.sortItems(0, Qt.AscendingOrder)
        qapp.processEvents()
        assert table.cellWidget(1, 1) is spin
        assert spin.palette().color(QPalette.Disabled, QPalette.Text) == disabled_text
        table.clearSelection()
        spin.setEnabled(True)
        qapp.processEvents()
        assert spin.palette().color(QPalette.Active, QPalette.Text) == initial_text
        assert spin.text() == "-2,0 dB"
        assert changes == []
        spin.setFocus(Qt.TabFocusReason)
        option = QStyleOptionSpinBox()
        spin.initStyleOption(option)
        up = spin.style().subControlRect(QStyle.CC_SpinBox, option, QStyle.SC_SpinBoxUp, spin)
        QTest.mouseClick(spin, Qt.LeftButton, pos=up.center())
        assert spin.value() == pytest.approx(-1.9)
        QTest.keyClick(spin, Qt.Key_Up)
        assert spin.value() == pytest.approx(-1.8)
        assert len(changes) == 2
    finally:
        table.close()
