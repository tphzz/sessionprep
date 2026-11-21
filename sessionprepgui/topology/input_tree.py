"""Input-tracks tree widget for Phase 1 topology.

Read-only QTreeWidget that displays source audio files with expandable
channel children.  Supports drag of individual channels or whole files.
"""

from __future__ import annotations

import json
from html import escape
from typing import TYPE_CHECKING

from PySide6.QtCore import QByteArray, QMimeData, QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QDrag, QFont, QFontMetrics, QPalette, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTreeWidget,
    QTreeWidgetItem,
)

from ..theme import COLORS, FILE_COLOR_OK, FILE_COLOR_WARNING, UNASSIGNED_INPUT_TEXT
from .operations import channel_label, used_channels

if TYPE_CHECKING:
    from sessionpreplib.models import TrackContext
    from sessionpreplib.topology import TopologyMapping

# MIME type used for cross-tree drag-and-drop
MIME_CHANNEL = "application/x-sessionprep-channel"

# Column indices
COL_NAME = 0
COL_CH = 1
COL_SR = 2
COL_BIT = 3
COL_DUR = 4

_DIM = QColor(COLORS["metadata"])
_ROUTING_STATUS_ROLE = Qt.UserRole + 1
_ROUTING_STATUS_COLOR = QColor(COLORS["clean"])
_ROUTING_STATUS_GAP = 8


def _format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def _set_routing_status(item: QTreeWidgetItem, status: str) -> None:
    name = item.text(COL_NAME)
    item.setData(COL_NAME, _ROUTING_STATUS_ROLE, status)
    item.setData(COL_NAME, Qt.AccessibleTextRole, f"{name}, {status}")
    tooltip = f"<p>{escape(name)}<br>{escape(status)}</p>"
    for column in range(COL_DUR + 1):
        item.setToolTip(column, tooltip)


def _status_font(font: QFont) -> QFont:
    font = QFont(font)
    font.setWeight(QFont.DemiBold)
    return font


class _RoutingStatusDelegate(QStyledItemDelegate):
    """Show source availability within the File cell."""

    def _status(self, index):
        if not index.parent().isValid() and self.parent().isExpanded(index):
            return None
        return index.data(_ROUTING_STATUS_ROLE)

    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        status = self._status(index)
        if status:
            opt = QStyleOptionViewItem(option)
            self.initStyleOption(opt, index)
            metrics = QFontMetrics(_status_font(opt.font))
            size.setWidth(size.width() + _ROUTING_STATUS_GAP + metrics.horizontalAdvance(status))
        return size

    def paint(self, painter, option, index):
        status = self._status(index)
        if not status:
            super().paint(painter, option, index)
            return

        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        widget = self.parent()
        style = widget.style()
        text_rect = style.subElementRect(QStyle.SE_ItemViewItemText, opt, widget)
        margin = style.pixelMetric(QStyle.PM_FocusFrameHMargin, opt, widget) + 1
        text_rect.adjust(margin, 0, -margin, 0)

        font = _status_font(opt.font)
        status_width = QFontMetrics(font).horizontalAdvance(status)
        name_width = text_rect.width() - _ROUTING_STATUS_GAP - status_width
        if name_width < opt.fontMetrics.horizontalAdvance("…"):
            super().paint(painter, option, index)
            return

        opt.text = opt.fontMetrics.elidedText(opt.text, opt.textElideMode, name_width)
        style.drawControl(QStyle.CE_ItemViewItem, opt, painter, widget)

        start = text_rect.left() + opt.fontMetrics.horizontalAdvance(opt.text)
        status_rect = QRect(start + _ROUTING_STATUS_GAP, text_rect.top(),
                            status_width, text_rect.height())
        status_rect = QStyle.visualRect(opt.direction, text_rect, status_rect)
        if opt.state & QStyle.State_Selected:
            color = opt.palette.color(QPalette.HighlightedText)
        elif opt.state & QStyle.State_Enabled:
            color = _ROUTING_STATUS_COLOR
        else:
            color = opt.palette.color(QPalette.Disabled, QPalette.Text)
        painter.save()
        painter.setClipRect(opt.rect, Qt.IntersectClip)
        painter.setFont(font)
        painter.setPen(color)
        painter.drawText(status_rect, Qt.AlignLeft | Qt.AlignVCenter, status)
        painter.restore()


class InputTree(QTreeWidget):
    """Read-only tree of source tracks with draggable channel children."""

    # Emitted on right-click: (filename, selected_filenames, global_pos)
    context_menu_requested = Signal(str, list, QPoint)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._source_dir: str | None = None

        self.setColumnCount(5)
        self.setHeaderLabels(["File", "Ch", "SR", "Bit", "Duration"])
        self.setItemDelegateForColumn(COL_NAME, _RoutingStatusDelegate(self))
        self.header().setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        h = self.header()
        h.setSectionsMovable(True)
        h.setSectionResizeMode(COL_NAME, QHeaderView.Interactive)
        self.setColumnWidth(COL_NAME, 380)
        for col in (COL_CH, COL_SR, COL_BIT, COL_DUR):
            h.setSectionResizeMode(col, QHeaderView.Interactive)

        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setAlternatingRowColors(True)
        self.setRootIsDecorated(True)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragOnly)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_context_menu)

        self.itemSelectionChanged.connect(self._enforce_single_level)
        self._enforcing = False

    def set_source_dir(self, path: str | None):
        self._source_dir = path

    # ------------------------------------------------------------------
    # Single-level selection
    # ------------------------------------------------------------------

    def _enforce_single_level(self):
        """Deselect items at a different tree depth than the clicked item."""
        if self._enforcing:
            return
        items = self.selectedItems()
        if len(items) <= 1:
            return
        # Use the current (clicked) item's depth as the anchor level
        anchor = self.currentItem()
        if anchor is None:
            return
        anchor_depth = self._item_depth(anchor)
        wrong = [it for it in items if self._item_depth(it) != anchor_depth]
        if not wrong:
            return
        self._enforcing = True
        self.blockSignals(True)
        for it in wrong:
            it.setSelected(False)
        self.blockSignals(False)
        self._enforcing = False

    @staticmethod
    def _item_depth(item) -> int:
        depth = 0
        p = item.parent()
        while p:
            depth += 1
            p = p.parent()
        return depth

    # ------------------------------------------------------------------
    # Populate
    # ------------------------------------------------------------------

    def populate(
        self,
        source_tracks: list[TrackContext],
        topology: TopologyMapping | None = None,
    ) -> None:
        """Rebuild the tree from *source_tracks*."""
        state = self._save_state()
        self.clear()
        ok_tracks = [t for t in source_tracks if t.status == "OK"]
        used = used_channels(topology) if topology else set()

        for track in ok_tracks:
            unassigned = {
                ch
                for ch in range(track.channels)
                if (track.filename, ch) not in used
            }

            # Check for Phase 1 detector warnings/suggestions
            p1_warnings = []
            for _det_id, res in track.detector_results.items():
                if res.severity and res.severity.name != "CLEAN":
                    p1_warnings.append(res.summary)

            file_color = FILE_COLOR_OK if unassigned else _DIM
            if p1_warnings:
                file_color = FILE_COLOR_WARNING

            file_item = QTreeWidgetItem()

            display_name = track.filename
            if p1_warnings:
                display_name += f"  [{', '.join(p1_warnings)}]"
                file_item.setToolTip(COL_NAME, "\\n".join(p1_warnings))

            file_item.setText(COL_NAME, display_name)
            file_item.setForeground(COL_NAME, file_color)
            file_item.setText(COL_CH, str(track.channels))
            file_item.setForeground(COL_CH, _DIM)
            file_item.setText(COL_SR, str(track.samplerate))
            file_item.setForeground(COL_SR, _DIM)
            file_item.setText(COL_BIT, track.bitdepth or "")
            file_item.setForeground(COL_BIT, _DIM)
            file_item.setText(COL_DUR, _format_duration(track.duration_sec))
            file_item.setForeground(COL_DUR, _DIM)
            file_item.setData(
                COL_NAME, Qt.UserRole, ("file", track.filename))
            file_item.setFlags(
                file_item.flags() | Qt.ItemIsDragEnabled)
            if unassigned:
                _set_routing_status(file_item, f"{len(unassigned)} unassigned")

            # Channel children
            for ch in range(track.channels):
                label = channel_label(ch, track.channels)
                ch_item = QTreeWidgetItem()
                ch_item.setText(COL_NAME, f"{ch} ({label}): {track.filename}")
                if ch in unassigned:
                    ch_item.setForeground(COL_NAME, UNASSIGNED_INPUT_TEXT)
                    _set_routing_status(ch_item, "Unassigned")
                else:
                    ch_item.setForeground(COL_NAME, _DIM)
                ch_item.setData(
                    COL_NAME, Qt.UserRole,
                    ("channel", track.filename, ch))
                ch_item.setFlags(
                    ch_item.flags() | Qt.ItemIsDragEnabled)
                file_item.addChild(ch_item)

            self.addTopLevelItem(file_item)

        # Expand all by default so channels are visible
        self.expandAll()
        self._restore_state(state)

    # ------------------------------------------------------------------
    # State save / restore
    # ------------------------------------------------------------------

    def _save_state(self) -> dict:
        """Capture scroll, selection, and expanded state before rebuild."""
        scroll_val = self.verticalScrollBar().value()
        selected_keys: list[tuple] = []
        for item in self.selectedItems():
            data = item.data(COL_NAME, Qt.UserRole)
            if data:
                selected_keys.append(data)
        collapsed: set[str] = set()
        for i in range(self.topLevelItemCount()):
            item = self.topLevelItem(i)
            if not item.isExpanded():
                data = item.data(COL_NAME, Qt.UserRole)
                if data and data[0] == "file":
                    collapsed.add(data[1])
        return {"scroll": scroll_val, "selected": selected_keys,
                "collapsed": collapsed}

    def _restore_state(self, state: dict) -> None:
        """Reapply scroll, selection, and expanded state after rebuild."""
        collapsed = state.get("collapsed", set())
        for i in range(self.topLevelItemCount()):
            item = self.topLevelItem(i)
            data = item.data(COL_NAME, Qt.UserRole)
            if data and data[0] == "file" and data[1] in collapsed:
                item.setExpanded(False)

        selected_keys = state.get("selected", [])
        if selected_keys:
            key_set = set(selected_keys)
            self.blockSignals(True)
            for i in range(self.topLevelItemCount()):
                item = self.topLevelItem(i)
                data = item.data(COL_NAME, Qt.UserRole)
                if data and data in key_set:
                    item.setSelected(True)
                for j in range(item.childCount()):
                    child = item.child(j)
                    cd = child.data(COL_NAME, Qt.UserRole)
                    if cd and cd in key_set:
                        child.setSelected(True)
            self.blockSignals(False)

        self.verticalScrollBar().setValue(state.get("scroll", 0))

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def find_item(self, filename: str) -> QTreeWidgetItem | None:
        """Find the top-level file item matching `filename`."""
        for i in range(self.topLevelItemCount()):
            item = self.topLevelItem(i)
            data = item.data(COL_NAME, Qt.UserRole)
            if data and data[0] == "file" and data[1] == filename:
                return item
        return None

    # ------------------------------------------------------------------
    # Drag support
    # ------------------------------------------------------------------

    def mimeTypes(self):
        return [MIME_CHANNEL, "text/uri-list"]

    def mimeData(self, items):
        """Encode dragged items as JSON list of channel descriptors, and text/uri-list."""
        payload = []
        filenames = set()
        for item in items:
            data = item.data(COL_NAME, Qt.UserRole)
            if not data:
                continue
            if data[0] == "channel":
                _, filename, ch = data
                payload.append({
                    "input_filename": filename,
                    "source_channel": ch,
                    "drag_type": "channel",
                })
                filenames.add(filename)
            elif data[0] == "file":
                _, filename = data
                filenames.add(filename)
                # Encode all channels of this file
                for i in range(item.childCount()):
                    child = item.child(i)
                    cd = child.data(COL_NAME, Qt.UserRole)
                    if cd and cd[0] == "channel":
                        payload.append({
                            "input_filename": cd[1],
                            "source_channel": cd[2],
                            "drag_type": "file",
                        })
        if not payload and not filenames:
            return None

        mime = QMimeData()

        if payload:
            mime.setData(MIME_CHANNEL,
                         QByteArray(json.dumps(payload).encode("utf-8")))

        if filenames and self._source_dir:
            import os
            from PySide6.QtCore import QUrl
            urls = [QUrl.fromLocalFile(os.path.join(self._source_dir, f))
                    for f in filenames]
            mime.setUrls(urls)

        return mime

    def startDrag(self, supportedActions):
        """Start drag with a semi-transparent pixmap."""
        items = self.selectedItems()
        if not items:
            return
        mime = self.mimeData(items)
        if mime is None:
            return
        drag = QDrag(self)
        drag.setMimeData(mime)
        # Render the first item's row as a pixmap at 50% opacity
        rect = self.visualItemRect(items[0])
        pixmap = QPixmap(self.viewport().size())
        pixmap.fill(Qt.transparent)
        self.viewport().render(pixmap, rect.topLeft(), rect)
        pixmap = pixmap.copy(0, 0, rect.width(), rect.height())
        # Apply 50% transparency
        faded = QPixmap(pixmap.size())
        faded.fill(Qt.transparent)
        from PySide6.QtGui import QPainter
        painter = QPainter(faded)
        painter.setOpacity(0.5)
        painter.drawPixmap(0, 0, pixmap)
        painter.end()
        drag.setPixmap(faded)
        drag.setHotSpot(QPoint(rect.width() // 2, rect.height() // 2))
        drag.exec(Qt.CopyAction)

    def supportedDragActions(self):
        return Qt.CopyAction

    # ------------------------------------------------------------------
    # Context menu
    # ------------------------------------------------------------------

    def _on_context_menu(self, pos):
        item = self.itemAt(pos)
        if not item:
            return
        data = item.data(COL_NAME, Qt.UserRole)
        if not data or data[0] != "file":
            return
        filename = data[1]
        # Collect all selected filenames (file-level only)
        selected = []
        for sel_item in self.selectedItems():
            sd = sel_item.data(COL_NAME, Qt.UserRole)
            if sd and sd[0] == "file":
                selected.append(sd[1])
        global_pos = self.viewport().mapToGlobal(pos)
        self.context_menu_requested.emit(filename, selected, global_pos)
