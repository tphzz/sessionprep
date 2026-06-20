"""Layout helpers for the Phase 2 track table."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import QHeaderView, QStyle, QTableWidget


TRACK_TABLE_FILE_MIN_WIDTH = 150
TRACK_TABLE_CELL_HPAD = 18
TRACK_TABLE_HEADER_HPAD = 22
TRACK_TABLE_ROW_VPAD = 6
TRACK_TABLE_RIGHT_MIN_WIDTH = 400


@dataclass(frozen=True)
class TrackTableLayout:
    column_widths: dict[int, int]
    row_height: int
    content_width: int
    table_width: int


def track_table_extra_width(table: QTableWidget) -> int:
    """Return frame/header/scrollbar padding not included in column widths."""
    vertical = table.verticalHeader()
    vheader = vertical.width() if vertical.isVisible() else 0
    scrollbar = table.style().pixelMetric(QStyle.PM_ScrollBarExtent)
    frame = table.frameWidth() * 2
    return vheader + scrollbar + frame + 4


def calculate_track_table_layout(table: QTableWidget) -> TrackTableLayout:
    """Calculate stable Phase 2 track table column and row sizes."""
    header = table.horizontalHeader()
    header_fm = header.fontMetrics()
    table_fm = table.fontMetrics()
    row_height = max(
        table.verticalHeader().minimumSectionSize(),
        table_fm.height() + TRACK_TABLE_ROW_VPAD,
    )
    widths: dict[int, int] = {}

    for col in range(table.columnCount()):
        label = table.horizontalHeaderItem(col)
        header_text = label.text() if label else ""
        width = header_fm.horizontalAdvance(header_text) + TRACK_TABLE_HEADER_HPAD

        for row in range(table.rowCount()):
            widget = table.cellWidget(row, col)
            if widget is not None:
                hint = widget.sizeHint()
                min_hint = widget.minimumSizeHint()
                width = max(width, hint.width(), min_hint.width())
                row_height = max(row_height, hint.height(), min_hint.height())
                continue

            item = table.item(row, col)
            if item is not None:
                width = max(
                    width,
                    table_fm.horizontalAdvance(item.text()) + TRACK_TABLE_CELL_HPAD,
                )

        widths[col] = width

    if 0 in widths:
        widths[0] = max(widths[0], TRACK_TABLE_FILE_MIN_WIDTH)

    row_height += 2
    content_width = sum(widths.values())
    return TrackTableLayout(
        column_widths=widths,
        row_height=row_height,
        content_width=content_width,
        table_width=content_width + track_table_extra_width(table),
    )


def apply_track_table_layout(table: QTableWidget) -> TrackTableLayout:
    """Apply deterministic Phase 2 track table sizing."""
    layout = calculate_track_table_layout(table)
    header = table.horizontalHeader()

    for col, width in layout.column_widths.items():
        if col == 0:
            continue
        header.setSectionResizeMode(col, QHeaderView.Interactive)
        header.resizeSection(col, width)

    if table.columnCount() > 0:
        header.setMinimumSectionSize(24)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.resizeSection(0, layout.column_widths.get(0, TRACK_TABLE_FILE_MIN_WIDTH))

    vertical = table.verticalHeader()
    vertical.setMinimumSectionSize(layout.row_height)
    vertical.setDefaultSectionSize(layout.row_height)
    return layout
