from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication, QTableWidgetItem

from sessionprepgui.widgets import (
    BatchComboBox,
    BatchEditTableWidget,
    TableCellComboBox,
    TableCellDoubleSpinBox,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _wheel_event(delta: int = 120) -> QWheelEvent:
    return QWheelEvent(
        QPointF(8, 8),
        QPointF(8, 8),
        QPoint(0, 0),
        QPoint(0, delta),
        Qt.NoButton,
        Qt.NoModifier,
        Qt.ScrollUpdate,
        False,
    )


def _table_with_selected_first_row():
    table = BatchEditTableWidget()
    table.setColumnCount(2)
    table.setRowCount(2)
    table.setSelectionBehavior(BatchEditTableWidget.SelectRows)
    table.setSelectionMode(BatchEditTableWidget.ExtendedSelection)
    for row in range(2):
        table.setItem(row, 0, QTableWidgetItem(f"Track {row + 1}"))
    table.selectRow(0)
    table.setCurrentCell(0, 0)
    return table


@pytest.mark.parametrize("combo_cls", [TableCellComboBox, BatchComboBox])
def test_closed_table_cell_combo_ignores_wheel_without_changing_index(
    qapp, combo_cls
):
    combo = combo_cls()
    combo.addItems(["A", "B", "C"])
    combo.setCurrentIndex(1)

    event = _wheel_event()
    QApplication.sendEvent(combo, event)

    assert combo.currentIndex() == 1
    assert not event.isAccepted()
    assert combo.focusPolicy() == Qt.ClickFocus


def test_table_cell_double_spin_box_ignores_wheel_without_changing_value(qapp):
    spin = TableCellDoubleSpinBox()
    spin.setRange(-10.0, 10.0)
    spin.setSingleStep(1.0)
    spin.setValue(3.0)

    event = _wheel_event()
    QApplication.sendEvent(spin, event)

    assert spin.value() == 3.0
    assert not event.isAccepted()
    assert spin.focusPolicy() == Qt.ClickFocus


@pytest.mark.parametrize("combo_cls", [TableCellComboBox, BatchComboBox])
def test_wheel_over_table_cell_combo_does_not_move_table_selection(
    qapp, combo_cls
):
    table = _table_with_selected_first_row()
    combo = combo_cls()
    combo.addItems(["A", "B", "C"])
    table.setCellWidget(1, 1, combo)

    event = _wheel_event()
    QApplication.sendEvent(combo, event)

    assert table.currentRow() == 0
    assert [idx.row() for idx in table.selectionModel().selectedRows()] == [0]


def test_wheel_over_table_cell_spin_does_not_move_table_selection(qapp):
    table = _table_with_selected_first_row()
    spin = TableCellDoubleSpinBox()
    table.setCellWidget(1, 1, spin)

    event = _wheel_event()
    QApplication.sendEvent(spin, event)

    assert table.currentRow() == 0
    assert [idx.row() for idx in table.selectionModel().selectedRows()] == [0]
