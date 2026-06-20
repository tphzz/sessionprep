from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QHeaderView, QTableWidgetItem

from sessionprepgui.tracks.table_layout import (
    TRACK_TABLE_FILE_MIN_WIDTH,
    apply_track_table_layout,
    calculate_track_table_layout,
    track_table_extra_width,
)
from sessionprepgui.widgets import (
    BatchComboBox,
    BatchEditTableWidget,
    BatchToolButton,
    TableCellDoubleSpinBox,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _track_table() -> BatchEditTableWidget:
    table = BatchEditTableWidget()
    table.setColumnCount(8)
    table.setHorizontalHeaderLabels([
        "File", "Ch", "Analysis", "Classification", "Gain",
        "RMS Anchor", "Group", "Processing",
    ])
    table.setRowCount(2)
    table.verticalHeader().setVisible(False)
    table.setItem(0, 0, QTableWidgetItem("01_Kick1.wav"))
    table.setItem(1, 0, QTableWidgetItem("32_Vox14.wav"))
    table.setItem(0, 1, QTableWidgetItem("1"))
    table.setItem(1, 1, QTableWidgetItem("2"))
    table.setItem(0, 2, QTableWidgetItem("OK"))
    table.setItem(1, 2, QTableWidgetItem("1A"))

    cls_combo = BatchComboBox()
    cls_combo.addItems(["Transient", "Sustained", "Skip"])
    cls_combo.setCurrentText("Sustained")
    table.setCellWidget(0, 3, cls_combo)

    gain_spin = TableCellDoubleSpinBox()
    gain_spin.setRange(-60.0, 60.0)
    gain_spin.setSuffix(" dB")
    gain_spin.setValue(2.9)
    table.setCellWidget(0, 4, gain_spin)

    anchor_combo = BatchComboBox()
    anchor_combo.addItems(["Default", "Max", "P99", "P95", "P90", "P85"])
    table.setCellWidget(0, 5, anchor_combo)

    group_combo = BatchComboBox()
    group_combo.addItems(["(None)", "Wide Background Vocals"])
    group_combo.setCurrentText("Wide Background Vocals")
    table.setCellWidget(0, 6, group_combo)

    processing = BatchToolButton()
    processing.setText("Default")
    table.setCellWidget(0, 7, processing)
    return table


def test_track_table_layout_fits_headers_widgets_and_rows(qapp):
    table = _track_table()

    layout = calculate_track_table_layout(table)

    assert layout.column_widths[0] >= TRACK_TABLE_FILE_MIN_WIDTH
    for col in range(table.columnCount()):
        header_text = table.horizontalHeaderItem(col).text()
        header_width = (
            table.horizontalHeader().fontMetrics().horizontalAdvance(header_text)
            + 22
        )
        assert layout.column_widths[col] >= header_width
    for col in (3, 4, 5, 6, 7):
        widget = table.cellWidget(0, col)
        assert layout.column_widths[col] >= widget.sizeHint().width()
        assert layout.row_height >= widget.sizeHint().height()
    assert layout.table_width == layout.content_width + track_table_extra_width(table)


def test_apply_track_table_layout_keeps_file_stretch_and_controls_sized(qapp):
    table = _track_table()

    layout = apply_track_table_layout(table)
    header = table.horizontalHeader()

    assert header.sectionResizeMode(0) == QHeaderView.Stretch
    for col in range(1, table.columnCount()):
        assert header.sectionResizeMode(col) == QHeaderView.Interactive
        assert header.sectionSize(col) == layout.column_widths[col]
    assert table.verticalHeader().defaultSectionSize() == layout.row_height
    assert table.verticalHeader().minimumSectionSize() == layout.row_height
