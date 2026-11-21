"""Track Layout (Phase 1) controller: trees, waveform preview, and Apply worker."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
import subprocess
import sys
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from PySide6.QtCore import QObject, QThread, QTimer, Qt, QUrl, Signal, Slot
from PySide6.QtGui import QAction, QDesktopServices
from shiboken6 import isValid
from PySide6.QtWidgets import (
    QAbstractItemView,
    QMenu,
    QMessageBox,
    QTreeWidget,
    QTreeWidgetItem,
)

from sessionpreplib.audio import format_duration
from sessionpreplib.topology import build_default_topology

from ..controls import set_action_busy
from ..settings import save_config
from ..shell import MainWindowShell
from ..state import WorkspaceState
from ..theme import COLORS
from ..tracks.table_widgets import _PHASE_ANALYSIS, _PHASE_SETUP, _PHASE_TOPOLOGY
from ..waveform.compute import WaveformLoadWorker
from ..waveform.peakcache import peak_source_key
from . import operations as ops
from .dialogs import add_output_tracks_dialog
from .page import TopologyPage

if TYPE_CHECKING:
    from ..detail.playback import PlaybackController

log = logging.getLogger(__name__)


class TopologyController(QObject):
    """Owns Phase 1 (Track Layout) page, workers, preview audio, and routing actions."""

    open_folder_requested = Signal()
    reanalyze_requested = Signal()
    topology_edited = Signal()
    layout_applied = Signal(str)  # output_dir
    peak_prioritize_requested = Signal(str)  # source_key

    def __init__(
        self,
        state: WorkspaceState,
        shell: MainWindowShell,
        playback: PlaybackController,
        *,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._shell = shell
        self._playback = playback

        self.page = TopologyPage(recursive_scan=state.recursive_scan)
        self._shell.register_phase_progress_layout(_PHASE_TOPOLOGY, self.page.main_layout)

        # Selection & scroll state
        self._sel_side: str | None = None
        self._selected_side: str | None = None
        self._syncing_scroll = False
        self._wf_expanded = False
        self._sel_timer = QTimer(self.page)
        self._sel_timer.setSingleShot(True)
        self._sel_timer.setInterval(150)
        self._sel_timer.timeout.connect(self._do_selection_changed)

        # Owned worker handles & preview cache
        self.apply_worker: Any = None
        self._pending_overwrite: tuple[str, dict[str, str]] | None = None
        self._overwrite_prompt_gen: int = 0
        self._apply_worker_gen: int = 0
        self.pending_sources: list[Any] = []
        self.audio_worker: Any = None
        self.wf_worker: Any = None
        self.resolve_worker: Any = None
        self.multi_worker: Any = None
        self.cached_audio: tuple[str, Any, Any, int] | None = None
        self.cached_labels: list[str] | None = None
        self.pending_labels: list[str] | None = None
        self.wf_request_id: int = 0
        self.wf_request_kind: str | None = None
        self.wf_peak_path: str | None = None
        self.wf_request_description: str = ""

        self._connect_page_signals()

    def _connect_page_signals(self) -> None:
        p = self.page
        p.open_action.triggered.connect(self.open_folder_requested.emit)
        p.recursive_cb.toggled.connect(self._on_recursive_toggled)
        p.reanalyze_action.triggered.connect(self.reanalyze_requested.emit)
        p.reset_action.triggered.connect(self.reset_topology)
        p.add_output_action.triggered.connect(self.add_output_tracks)
        p.remove_empty_action.triggered.connect(self.remove_empty_outputs)
        p.collapse_action.triggered.connect(lambda *_: p.output_tree.collapseAll())
        p.expand_action.triggered.connect(lambda *_: p.output_tree.expandAll())
        p.apply_action.triggered.connect(self.apply_topology)

        p.input_tree.context_menu_requested.connect(self._on_input_context_menu)
        p.output_tree.topology_modified.connect(self.notify_topology_changed)
        p.input_tree.selectionModel().selectionChanged.connect(
            lambda _sel, _desel: self._on_selection_changed("input")
        )
        p.output_tree.selectionModel().selectionChanged.connect(
            lambda _sel, _desel: self._on_selection_changed("output")
        )
        p.input_tree.verticalScrollBar().valueChanged.connect(self._on_input_scroll)

        p.wf_toggle.toggled.connect(self._on_wf_toggle)
        p.wf_panel.play_clicked.connect(self.play_preview)
        p.wf_panel.stop_clicked.connect(self.stop_preview)
        p.wf_panel.position_clicked.connect(self._on_wf_seek)
        p.wf_panel.display_mode_changed.connect(self._on_display_mode_changed)
        p.wf_panel.fft_group.triggered.connect(self._on_spec_fft_changed)
        p.wf_panel.win_group.triggered.connect(self._on_spec_window_changed)
        p.wf_panel.cmap_group.triggered.connect(self._on_spec_cmap_changed)
        p.wf_panel.floor_group.triggered.connect(self._on_spec_floor_changed)
        p.wf_panel.ceil_group.triggered.connect(self._on_spec_ceil_changed)

    # ── Lifecycle & workspace clearing ───────────────────────────────────

    def dismiss_overwrite_prompt(
        self, *, status_message: str | None = None, restore_focus: bool = False
    ) -> None:
        """Clear any pending unverified-layout overwrite prompt and invalidate in-flight checks."""
        self._overwrite_prompt_gen += 1
        self._pending_overwrite = None
        self._shell.dismiss_status_prompt(
            status_message=status_message,
            restore_focus=restore_focus,
        )

    def clear_workspace(self) -> None:
        """Stop in-flight layout/preview workers and reset Phase 1 UI."""
        self.dismiss_overwrite_prompt()
        if self.apply_worker is not None:
            worker: Any = self.apply_worker
            self.apply_worker = None
            if isinstance(worker, QObject) and isValid(worker):
                try:
                    worker.disconnect(self)
                except RuntimeError:
                    pass
                w_any: Any = worker
                w_any.cancel()
                w_any.finished.connect(w_any.deleteLater)
                if not w_any.isRunning():
                    w_any.deleteLater()
            elif hasattr(worker, "cancel"):
                worker.cancel()
        set_action_busy(self.page.apply_action, False)
        self.begin_waveform_request("none")
        self.cancel_preview_workers()
        self.cached_audio = None
        self.pending_sources = []
        self.page.wf_panel.waveform.set_audio(None, 44100)
        self.page.wf_panel.play_btn.setEnabled(False)
        self.page.input_tree.clear()
        self.page.input_tree.set_source_dir(None)
        self.page.output_tree.clear()
        self.page.status_label.clear()
        self.page.apply_action.setEnabled(False)
        self.page.reanalyze_action.setEnabled(False)
        self.page.reset_action.setEnabled(False)

    def dispose(self) -> None:
        """Stop all Phase 1 background threads cleanly before window close."""
        self.cancel_preview_workers()
        if isinstance(self.apply_worker, QThread) and isValid(self.apply_worker) and self.apply_worker.isRunning():
            cancel_apply = getattr(self.apply_worker, "cancel", None)
            if callable(cancel_apply):
                cancel_apply()
            else:
                self.apply_worker.requestInterruption()
            self.apply_worker.wait()
        self.apply_worker = None
        for worker in self.findChildren(QThread):
            if isValid(worker) and worker.isRunning():
                cancel_fn = getattr(worker, "cancel", None)
                if callable(cancel_fn):
                    cancel_fn()
                else:
                    worker.requestInterruption()
                worker.wait()

    @Slot(bool)
    def _on_recursive_toggled(self, checked: bool) -> None:
        self._state.recursive_scan = checked
        self._state.config.setdefault("app", {})["recursive_scan"] = checked
        save_config(self._state.config)

    # ── Populate & topology mutations ────────────────────────────────────

    def track_map(self) -> dict[str, Any]:
        """Return {filename: TrackContext} for OK original source tracks."""
        return {t.filename: t for t in self._state.topo_source_tracks if t.status == "OK"}

    def populate(self) -> None:
        """Refresh both trees from source tracks and topology."""
        if not self._state.session:
            return
        if self._state.topo_topology is None and not self.pending_sources:
            ok = [t for t in self._state.topo_source_tracks if t.status == "OK"]
            if ok:
                self._state.topo_topology = build_default_topology(self._state.topo_source_tracks)

        tmap = self.track_map()
        self.page.input_tree.populate(self._state.topo_source_tracks, self._state.topo_topology)
        self.page.output_tree.populate(self._state.topo_topology, tmap)

        n_in = sum(1 for t in self._state.topo_source_tracks if t.status == "OK")
        n_out = len(self._state.topo_topology.entries) if self._state.topo_topology else 0
        self.page.status_label.setText(f"{n_in} input \u2192 {n_out} output tracks")
        self.page.apply_action.setEnabled(n_out > 0 and self.apply_worker is None)

    def notify_topology_changed(self) -> None:
        """Refresh UI and invalidate downstream phases after a topology edit."""
        self.dismiss_overwrite_prompt()
        self.populate()
        if self._state.topology_dir is not None:
            self._state.topology_dir = None
            self._shell.set_phase_enabled(_PHASE_ANALYSIS, False)
            self._shell.set_phase_enabled(_PHASE_SETUP, False)
        self.topology_edited.emit()

        if self._selected_side == "input":
            file_items, channel_items = _split_tree_items(self.page.input_tree.selectedItems())
            self._update_output_highlights(file_items, channel_items)
        elif self._selected_side == "output":
            file_items, channel_items = _split_tree_items(self.page.output_tree.selectedItems())
            if file_items or channel_items:
                self._load_output_from_items(file_items, channel_items)
            else:
                self.cancel_preview_workers()
                self.cached_audio = None
                self.page.wf_panel.waveform.set_audio(None, 44100)
                self.page.wf_panel.play_btn.setEnabled(False)
                self._selected_side = None

    # ── Apply topology ───────────────────────────────────────────────────

    def _layout_output_dir(self) -> tuple[str, str]:
        assert self._state.source_dir is not None
        app_cfg = self._state.config.get("app", {})
        output_folder = app_cfg.get("phase1_output_folder", "sp_01_tracklayout")
        output_dir = os.path.join(self._state.source_dir, output_folder)
        peaks_dir = os.path.join(output_dir, app_cfg.get("peak_cache_folder", "sp_peaks"))
        return output_dir, peaks_dir

    @Slot()
    def apply_topology(self) -> None:
        """Write channel-rerouted files to the track layout output folder."""
        if not self._state.session or not self._state.source_dir or self.apply_worker is not None:
            return
        self.dismiss_overwrite_prompt()
        self._start_apply_worker(allowed_unverified=None)

    def _start_apply_worker(self, *, allowed_unverified: dict[str, str] | None) -> None:
        assert self._state.session is not None and self._state.source_dir is not None
        n_entries = len(self._state.topo_topology.entries) if self._state.topo_topology else 0
        log.info("Apply topology: %d entries", n_entries)

        from ..analysis.worker import TopologyApplyWorker

        output_dir, peaks_dir = self._layout_output_dir()
        set_action_busy(self.page.apply_action, True, "Applying…")
        self.page.apply_action.setEnabled(False)
        self.page.reset_action.setEnabled(False)
        self._shell.progress_start("Applying topology\u2026")
        self._apply_worker_gen = self._overwrite_prompt_gen

        self._state.session.topology = self._state.topo_topology
        worker = TopologyApplyWorker(
            self._state.session,
            output_dir,
            source_dir=self._state.source_dir,
            peaks_dir=peaks_dir,
            source_tracks=self._state.topo_source_tracks,
            allowed_unverified=allowed_unverified,
            prompt_on_unverified=allowed_unverified is None,
        )
        worker.setParent(self)
        worker.finished.connect(worker.deleteLater)
        worker.progress.connect(self._on_apply_progress)
        worker.progress_value.connect(self._on_apply_progress_value)
        worker.overwrite_confirmation_required.connect(self._on_overwrite_confirmation_required)
        worker.apply_finished.connect(self._on_apply_done)
        worker.error.connect(self._on_apply_error)
        self.apply_worker = worker
        worker.start()

    @Slot(object, str, object)
    def _on_overwrite_confirmation_required(
        self, worker: Any, output_dir: str, unverified: Any
    ) -> None:
        if worker is not self.apply_worker:
            return
        worker.wait()
        self.apply_worker = None
        set_action_busy(self.page.apply_action, False)
        self.page.apply_action.setEnabled(True)
        self.page.reset_action.setEnabled(not self.pending_sources)
        self._shell.clear_progress()

        unverified_map = dict(unverified)
        if (
            not unverified_map
            or self._apply_worker_gen != self._overwrite_prompt_gen
            or self._shell.current_phase() != _PHASE_TOPOLOGY
            or not self._state.source_dir
        ):
            return
        current_dir, _peaks_dir = self._layout_output_dir()
        if os.path.realpath(current_dir) != os.path.realpath(output_dir):
            return
        self._pending_overwrite = (output_dir, unverified_map)
        prefix, highlight, suffix, tooltip, accept_desc = _format_unverified_prompt(
            output_dir, unverified_map
        )
        apply_btn = self.page.toolbar.widgetForAction(self.page.apply_action)
        self._shell.show_status_prompt(
            prefix,
            highlight,
            suffix,
            tooltip=tooltip,
            on_accept=self._confirm_overwrite_apply,
            on_cancel=self._cancel_overwrite_apply,
            accept_label="Overwrite",
            cancel_label="Cancel",
            accept_description=accept_desc,
            cancel_description="Cancel Apply and leave existing layout files untouched",
            return_focus_widget=apply_btn,
        )

    def _confirm_overwrite_apply(self) -> None:
        pending = self._pending_overwrite
        self._pending_overwrite = None
        if (
            pending is None
            or not self._state.session
            or not self._state.source_dir
            or self.apply_worker is not None
        ):
            return
        output_dir, _peaks_dir = self._layout_output_dir()
        if os.path.realpath(output_dir) != os.path.realpath(pending[0]):
            self._shell.show_status("Layout output directory changed — click Apply to re-check.")
            return
        self._start_apply_worker(allowed_unverified=pending[1])

    def _cancel_overwrite_apply(self) -> None:
        self._pending_overwrite = None
        self._shell.show_status("Apply cancelled — existing layout files were left untouched.")

    @Slot(object, str)
    def _on_apply_progress(self, worker: Any, message: str) -> None:
        if worker is not self.apply_worker:
            return
        log.debug("Apply topology: %s", message)
        self._shell.progress_message(message)

    @Slot(object, int, int)
    def _on_apply_progress_value(self, worker: Any, current: int, total: int) -> None:
        if worker is not self.apply_worker:
            return
        self._shell.progress_value(current, total)

    @Slot(object)
    def _on_apply_done(self, worker: Any) -> None:
        if worker is not self.apply_worker:
            return
        worker.wait()
        log.info("Apply topology complete")
        self.apply_worker = None
        set_action_busy(self.page.apply_action, False)
        self.page.apply_action.setEnabled(True)
        self.page.reset_action.setEnabled(not self.pending_sources)

        session = self._state.session
        assert session is not None
        errors = session.config.get("_topology_apply_errors", [])
        n_out = len(session.output_tracks)
        if errors:
            msg = f"Topology applied: {n_out} file(s) written, {len(errors)} error(s)"
            self._shell.progress_finish(msg)
            detail = "\n".join(f"\u2022 {fn}: {err}" for fn, err in errors)
            QMessageBox.warning(
                self._shell.dialog_parent(), "Apply Topology \u2014 errors", f"{msg}\n\n{detail}"
            )
        else:
            msg = f"Topology applied: {n_out} file(s) written"
            self._shell.progress_finish(msg)
        self._shell.show_status(msg)

        output_folder = self._state.config.get("app", {}).get(
            "phase1_output_folder", "sp_01_tracklayout"
        )
        assert self._state.source_dir is not None
        output_dir = os.path.join(self._state.source_dir, output_folder)
        self._state.topology_dir = output_dir

        self._shell.set_phase_enabled(_PHASE_ANALYSIS, True)
        self._shell.switch_to_phase(_PHASE_ANALYSIS)
        self.layout_applied.emit(output_dir)

    @Slot(object, str)
    def _on_apply_error(self, worker: Any, message: str) -> None:
        if worker is not self.apply_worker:
            return
        worker.wait()
        log.error("Apply topology failed: %s", message)
        self.apply_worker = None
        set_action_busy(self.page.apply_action, False)
        self.page.apply_action.setEnabled(True)
        self.page.reset_action.setEnabled(not self.pending_sources)
        self._shell.progress_fail(message)
        self._shell.show_status(f"Apply topology error: {message}")

    # ── Toolbar actions & context menu ───────────────────────────────────

    @Slot()
    def reset_topology(self) -> None:
        """Rebuild the recommended layout from the original source findings."""
        if not self._state.session or self.pending_sources:
            return
        self._state.topo_topology = build_default_topology(self._state.topo_source_tracks)
        self.notify_topology_changed()

    @Slot()
    def remove_empty_outputs(self) -> None:
        """Remove output entries that have no wired source channels."""
        if not self._state.topo_topology:
            return
        removed = ops.remove_empty_outputs(self._state.topo_topology)
        if removed:
            self.notify_topology_changed()
            self._shell.show_status(f"Removed {removed} empty output track(s)")
        else:
            self._shell.show_status("No empty output tracks to remove")

    @Slot()
    def add_output_tracks(self) -> None:
        """Create one or more empty output tracks via dialog."""
        topo = self._state.topo_topology
        if not topo:
            return
        tracks = add_output_tracks_dialog(self._shell.dialog_parent(), topo, COLORS)
        if not tracks:
            return
        for name_hint, n_channels in tracks:
            parts = name_hint.rpartition(".")
            stem = parts[0] or parts[2]
            ext = f".{parts[2]}" if parts[0] else ".wav"
            fname = ops.unique_output_name(topo, stem, ext)
            ops.new_output_file(topo, fname, n_channels)
        self.notify_topology_changed()

    @Slot(str, list, object)
    def _on_input_context_menu(self, filename: str, selected: list[str], global_pos: Any = None) -> None:
        topo = self._state.topo_topology
        if not self._state.session or not topo:
            return
        tmap = self.track_map()
        track = tmap.get(filename)
        if not track:
            return

        is_excluded = not any(
            src.input_filename == filename for entry in topo.entries for src in entry.sources
        )
        menu = QMenu(self._shell.dialog_parent())
        act_open = menu.addAction("Open Folder")
        act_open.triggered.connect(lambda _c=False, fn=filename: self._open_folder_for_file(fn))
        menu.addSeparator()

        if is_excluded:
            act = menu.addAction("Include in Session")
            act.triggered.connect(lambda _c=False, fn=filename: self._input_action(ops.include_input, fn))
        else:
            self._populate_included_input_menu(menu, filename, track, selected, tmap)

        if menu.actions() and global_pos is not None:
            menu.exec(global_pos)

    def _populate_included_input_menu(
        self, menu: QMenu, filename: str, track: Any, selected: list[str], tmap: dict[str, Any]
    ) -> None:
        act_reset = menu.addAction("Reset to Passthrough")
        act_reset.triggered.connect(
            lambda _c=False, fn=filename: self._input_action(ops.reset_to_passthrough, fn)
        )
        if track.channels >= 2:
            act_split = menu.addAction("Split to L/R Mono")
            act_split.triggered.connect(
                lambda _c=False, fn=filename: self._input_action(ops.split_stereo, fn)
            )
            dm_menu = menu.addMenu("Downmix to Mono")
            for ch in range(track.channels):
                label = {0: "Keep Left", 1: "Keep Right"}.get(ch, f"Keep Ch {ch}")
                act = dm_menu.addAction(label)
                act.triggered.connect(
                    lambda _c=False, fn=filename, c=ch: self._input_action_ch(ops.extract_channel, fn, c)
                )
            sum_act = dm_menu.addAction("Sum All Channels")
            sum_act.triggered.connect(
                lambda _c=False, fn=filename: self._input_action(ops.sum_to_mono, fn)
            )
        if len(selected) == 2:
            t_a, t_b = tmap.get(selected[0]), tmap.get(selected[1])
            if t_a and t_b and t_a.channels == 1 and t_b.channels == 1:
                menu.addSeparator()
                act_merge = menu.addAction("Merge to Stereo")
                act_merge.triggered.connect(
                    lambda _c=False, a=selected[0], b=selected[1]: self._input_action_merge(a, b)
                )
        menu.addSeparator()
        act_excl = menu.addAction("Exclude from Session")
        act_excl.triggered.connect(lambda _c=False, fn=filename: self._input_action_exclude(fn))

    def _open_folder_for_file(self, filename: str) -> None:
        if not self._state.source_dir:
            return
        filepath = os.path.join(self._state.source_dir, filename)
        if not os.path.exists(filepath):
            return
        if sys.platform == "win32":
            subprocess.Popen(["explorer", "/select,", os.path.normpath(filepath)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", filepath])
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(filepath)))

    def _input_action(self, op_fn: Callable[..., Any], filename: str) -> None:
        if not self._state.topo_topology:
            return
        op_fn(self._state.topo_topology, self.track_map(), filename)
        self.notify_topology_changed()

    def _input_action_ch(self, op_fn: Callable[..., Any], filename: str, channel: int) -> None:
        if not self._state.topo_topology:
            return
        op_fn(self._state.topo_topology, self.track_map(), filename, channel)
        self.notify_topology_changed()

    def _input_action_merge(self, left: str, right: str) -> None:
        if not self._state.topo_topology:
            return
        ops.merge_stereo(self._state.topo_topology, self.track_map(), left, right)
        self.notify_topology_changed()

    def _input_action_exclude(self, filename: str) -> None:
        if not self._state.topo_topology:
            return
        ops.exclude_input(self._state.topo_topology, filename)
        self.notify_topology_changed()

    # ── Synchronized scrolling & cross-tree selection ────────────────────

    @staticmethod
    def _get_item_at_center(tree: QTreeWidget) -> QTreeWidgetItem | None:
        viewport = tree.viewport()
        return tree.itemAt(viewport.width() // 2, viewport.height() // 2)

    @Slot()
    def _on_input_scroll(self) -> None:
        if self._syncing_scroll or not self._state.topo_topology:
            return
        item = self._get_item_at_center(self.page.input_tree)
        if not item:
            return
        data = item.data(0, Qt.ItemDataRole.UserRole)
        input_filename = data[1] if data and data[0] in ("file", "channel") else None
        if not input_filename:
            return
        target_item = self.page.output_tree.find_item_for_source(input_filename)
        if target_item:
            self._syncing_scroll = True
            self.page.output_tree.scrollToItem(target_item, QAbstractItemView.ScrollHint.PositionAtCenter)
            self._syncing_scroll = False

    def _on_selection_changed(self, side: str) -> None:
        other = self.page.output_tree if side == "input" else self.page.input_tree
        if self._selected_side != side:
            other.blockSignals(True)
            other.clearSelection()
            other.blockSignals(False)
        self._selected_side = side
        self._sel_side = side
        self._sel_timer.start()

    def _do_selection_changed(self) -> None:
        side = self._sel_side
        if not side:
            return
        tree = self.page.input_tree if side == "input" else self.page.output_tree
        items = tree.selectedItems()
        if not items:
            return
        file_items, channel_items = _split_tree_items(items)
        if side == "input":
            self._update_output_highlights(file_items, channel_items)
            self._load_input_from_items(file_items, channel_items)
        else:
            self.page.output_tree.clear_highlights()
            self._load_output_from_items(file_items, channel_items)

    def _on_wf_toggle(self, checked: bool) -> None:
        self._wf_expanded = checked
        self.page.wf_panel.set_expanded(checked)

    def _update_output_highlights(self, file_items: list[Any], channel_items: list[Any]) -> None:
        if channel_items:
            data = channel_items[0].data(0, Qt.ItemDataRole.UserRole)
            if data and data[0] == "channel":
                self.page.output_tree.highlight_usages(data[1], data[2])
                return
        if file_items:
            data = file_items[0].data(0, Qt.ItemDataRole.UserRole)
            if data and data[0] == "file":
                self.page.output_tree.highlight_usages(data[1])
                return
        self.page.output_tree.clear_highlights()

    # ── Waveform preview workers & request guards ────────────────────────

    def cancel_preview_workers(self) -> None:
        """Cancel all in-flight Phase 1 topology preview workers."""
        for attr in ("audio_worker", "wf_worker", "resolve_worker", "multi_worker"):
            worker = getattr(self, attr, None)
            setattr(self, attr, None)
            if worker is None or (isinstance(worker, QObject) and not isValid(worker)):
                continue
            cancel_fn = getattr(worker, "cancel", None)
            if callable(cancel_fn):
                cancel_fn()
            for sig_name in ("completed", "error", "failed"):
                sig = getattr(worker, sig_name, None)
                if sig is not None:
                    try:
                        sig.disconnect()
                    except RuntimeError:
                        pass

    def begin_waveform_request(
        self,
        kind: str,
        *,
        peak_path: str | None = None,
        file_count: int = 0,
        channel_count: int = 0,
        labels_count: int = 0,
        description: str = "",
    ) -> int:
        """Start a new Phase 1 waveform request and return its generation id."""
        self.wf_request_id += 1
        self.wf_request_kind = kind
        self.wf_peak_path = peak_source_key(peak_path)
        self.wf_request_description = description
        log.debug(
            "Phase 1 waveform request #%d: kind=%s peak=%r files=%d channels=%d labels=%d %s",
            self.wf_request_id, kind, peak_path, file_count, channel_count, labels_count, description,
        )
        return self.wf_request_id

    def is_current_waveform_request(self, request_id: int | None) -> bool:
        """Return True if *request_id* still matches the active preview."""
        return request_id is None or request_id == self.wf_request_id

    def on_peak_file_done(self, source_key: str, peak_data: Any) -> None:
        """Push newly built peak data into the Phase 1 waveform preview if matching."""
        if self.wf_peak_path != source_key:
            return
        wf = self.page.wf_panel.waveform
        wf.set_peak_data(peak_data)
        if getattr(wf, "_loading", False):
            wf.set_preview_mode(
                peak_data.channels, peak_data.total_samples, peak_data.samplerate, peak_data
            )

    def _load_input_from_items(self, file_items: list[Any], channel_items: list[Any] | None = None) -> None:
        if not file_items and not channel_items:
            return
        if channel_items and not file_items:
            self.load_multi_input(None, channel_items)
            return
        if len(file_items) == 1 and not channel_items:
            self.load_input_waveform(file_items[0].data(0, Qt.ItemDataRole.UserRole)[1])
        else:
            self.load_multi_input(file_items, channel_items)

    def load_input_waveform(self, filename: str) -> None:
        """Load waveform preview for a single input file."""
        t0 = time.perf_counter()
        self.cancel_preview_workers()
        self.stop_preview()

        track = self.track_map().get(filename)
        if not track:
            return

        request_id = self.begin_waveform_request(
            "input-file",
            peak_path=track.filepath,
            file_count=1,
            channel_count=track.channels,
            description=filename,
        )
        if self.cached_audio and self.cached_audio[0] == track.filepath:
            self.show_waveform(self.cached_audio[1], self.cached_audio[3], request_id=request_id)
            return

        if self._wf_expanded:
            self.page.wf_panel.setVisible(True)
        key = self.wf_peak_path
        if key in self._state.peak_cache:
            self.page.wf_panel.waveform.set_preview_mode(
                track.channels, track.total_samples, track.samplerate, self._state.peak_cache[key]
            )
        else:
            self.page.wf_panel.waveform.set_loading(True)
            if key:
                self.peak_prioritize_requested.emit(key)

        self.page.wf_panel.play_btn.setEnabled(False)
        log.debug("[Trace] load_input_waveform setup for '%s': %.2f ms", filename, (time.perf_counter() - t0) * 1000)

        from ..analysis.worker import AudioLoadWorker

        worker = AudioLoadWorker(track, parent=self)
        self.audio_worker = worker
        worker.completed.connect(lambda t, fp=track.filepath, rid=request_id: self._on_audio_loaded(t, fp, rid))
        worker.error.connect(lambda msg, rid=request_id: self._on_audio_error(msg, rid))
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _on_audio_loaded(self, track: Any, filepath: str, request_id: int | None = None) -> None:
        if not self.is_current_waveform_request(request_id):
            log.debug("Discarding stale phase 1 audio load for request #%s", request_id)
            return
        self.audio_worker = None
        if track.audio_data is None:
            return
        self.cached_audio = (filepath, track.audio_data, track.audio_data, track.samplerate)
        self.show_waveform(track.audio_data, track.samplerate, request_id=request_id)

    def _on_audio_error(self, message: str, request_id: int) -> None:
        if not self.is_current_waveform_request(request_id):
            return
        self.audio_worker = None
        self.page.wf_panel.waveform.set_loading(False)
        self._shell.show_status(f"Audio load error: {message}")

    def load_multi_input(self, file_items: list[Any] | None, channel_items: list[Any] | None = None) -> None:
        """Load and stack waveforms for multiple input files or channels."""
        self.cancel_preview_workers()
        self.stop_preview()

        tmap = self.track_map()
        items = _collect_multi_input_specs(tmap, file_items, channel_items)
        if not items:
            return
        self._start_multi_worker("input", items)

    def _load_output_from_items(self, file_items: list[Any], channel_items: list[Any] | None = None) -> None:
        if not file_items and not channel_items:
            return
        if channel_items and not file_items:
            self.load_multi_output(None, channel_items)
            return
        if len(file_items) == 1 and not channel_items:
            self.load_output_waveform(file_items[0].data(0, Qt.ItemDataRole.UserRole)[1])
        else:
            self.load_multi_output(file_items, channel_items)

    def load_output_waveform(self, output_filename: str) -> None:
        """Resolve and display waveform for an output entry."""
        self.cancel_preview_workers()
        self.stop_preview()

        topo = self._state.topo_topology
        if not topo or not self._state.source_dir:
            return
        entry = next((e for e in topo.entries if e.output_filename == output_filename), None)
        if entry is None:
            return

        request_id = self.begin_waveform_request("output-file", file_count=1, description=output_filename)
        if self._wf_expanded:
            self.page.wf_panel.setVisible(True)
        self.page.wf_panel.waveform.set_loading(True)
        self.page.wf_panel.play_btn.setEnabled(False)

        from ..analysis.worker import TopoAudioResolveWorker

        worker = TopoAudioResolveWorker(entry, self._state.source_dir, parent=self)
        self.resolve_worker = worker
        worker.completed.connect(lambda audio, sr, rid=request_id: self._on_resolve_done(audio, sr, rid))
        worker.error.connect(lambda msg, rid=request_id: self._on_resolve_error(msg, rid))
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _on_resolve_done(self, audio_data: Any, samplerate: int, request_id: int | None = None) -> None:
        if not self.is_current_waveform_request(request_id):
            log.debug("Discarding stale phase 1 output resolve for request #%s", request_id)
            return
        self.resolve_worker = None
        self.cached_audio = ("__output__", audio_data, audio_data, samplerate)
        self.show_waveform(audio_data, samplerate, request_id=request_id)

    def _on_resolve_error(self, message: str, request_id: int) -> None:
        if not self.is_current_waveform_request(request_id):
            return
        self.resolve_worker = None
        self.page.wf_panel.waveform.set_loading(False)
        self._shell.show_status(f"Preview error: {message}")

    def load_multi_output(self, file_items: list[Any] | None, channel_items: list[Any] | None = None) -> None:
        """Load and stack waveforms for multiple output entries."""
        self.cancel_preview_workers()
        self.stop_preview()

        topo = self._state.topo_topology
        if not topo or not self._state.source_dir:
            return
        items = _collect_multi_output_specs(topo, file_items, channel_items)
        if not items:
            return
        self._start_multi_worker("output", items)

    def _start_multi_worker(self, side: str, items: list[tuple[Any, str, list[int] | None]]) -> None:
        request_id = self.begin_waveform_request(
            f"{side}-multi",
            file_count=len(items),
            channel_count=sum(len(item[2]) if item[2] is not None else 0 for item in items),
            description=", ".join(item[1] for item in items[:4]),
        )
        if self._wf_expanded:
            self.page.wf_panel.setVisible(True)
        self.page.wf_panel.waveform.set_loading(True)
        self.page.wf_panel.play_btn.setEnabled(False)

        from ..analysis.worker import TopoMultiAudioWorker

        worker = TopoMultiAudioWorker(items, side, self._state.source_dir or "", parent=self)
        self.multi_worker = worker
        worker.completed.connect(
            lambda display, playback, sr, labels, rid=request_id: self.on_multi_done(
                display, playback, sr, labels, rid
            )
        )
        worker.error.connect(lambda msg, rid=request_id: self._on_multi_error(msg, rid))
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def on_multi_done(
        self,
        display_audio: Any,
        playback_audio: Any,
        samplerate: int,
        labels: list[str],
        request_id: int | None = None,
    ) -> None:
        if not self.is_current_waveform_request(request_id):
            log.debug("Discarding stale phase 1 multi-load for request #%s", request_id)
            return
        self.multi_worker = None
        self.cached_audio = ("__multi__", display_audio, playback_audio, samplerate)
        self.cached_labels = labels
        self.show_waveform(display_audio, samplerate, labels=labels, request_id=request_id)

    def _on_multi_error(self, message: str, request_id: int) -> None:
        if not self.is_current_waveform_request(request_id):
            return
        self.multi_worker = None
        self.page.wf_panel.waveform.set_loading(False)
        self._shell.show_status(f"Multi-track load error: {message}")

    def show_waveform(
        self,
        audio_data: Any,
        samplerate: int,
        labels: list[str] | None = None,
        request_id: int | None = None,
    ) -> None:
        """Run WaveformLoadWorker and display result."""
        if not self.is_current_waveform_request(request_id):
            log.debug("Skipping stale phase 1 waveform setup for request #%s", request_id)
            return
        if (
            audio_data is None
            or (isinstance(audio_data, np.ndarray) and audio_data.size == 0)
            or (isinstance(audio_data, list) and not audio_data)
        ):
            self.page.wf_panel.waveform.set_loading(False)
            return

        if self._wf_expanded:
            self.page.wf_panel.setVisible(True)
        self.page.wf_panel.waveform.set_loading(True)
        self.pending_labels = labels
        self._cancel_wf_worker()

        rid = self.wf_request_id if request_id is None else request_id
        wf = self.page.wf_panel.waveform
        worker = WaveformLoadWorker(
            audio_data,
            samplerate,
            0,
            spec_n_fft=wf.spec_n_fft,
            spec_window=wf.spec_window,
            parent=self,
        )
        self.wf_worker = worker
        worker.completed.connect(lambda result, r=rid: self._on_wf_loaded(result, r))
        worker.failed.connect(lambda msg, r=rid: self._on_wf_error(msg, r))
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _cancel_wf_worker(self) -> None:
        worker = self.wf_worker
        self.wf_worker = None
        if worker is None or (isinstance(worker, QObject) and not isValid(worker)):
            return
        cancel_fn = getattr(worker, "cancel", None)
        if callable(cancel_fn):
            cancel_fn()
        for sig_name in ("completed", "failed"):
            sig = getattr(worker, sig_name, None)
            if sig is not None:
                try:
                    sig.disconnect()
                except RuntimeError:
                    pass

    def _on_wf_error(self, message: str, request_id: int | None = None) -> None:
        if not self.is_current_waveform_request(request_id):
            return
        self.wf_worker = None
        self.page.wf_panel.waveform.set_loading(False)
        self._shell.show_status(f"Waveform error: {message}")

    def _on_wf_loaded(self, result: dict[str, Any], request_id: int | None = None) -> None:
        if not self.is_current_waveform_request(request_id):
            log.debug("Discarding stale phase 1 waveform result for request #%s", request_id)
            return
        self.wf_worker = None
        wf = self.page.wf_panel.waveform
        wf.set_precomputed(result)
        key = self.wf_peak_path
        if key in self._state.peak_cache:
            wf.set_peak_data(self._state.peak_cache[key])
        elif key:
            self.peak_prioritize_requested.emit(key)
        self.page.wf_panel.update_play_mode_channels(len(result["channels"]), labels=self.pending_labels)
        self.page.wf_panel.play_btn.setEnabled(True)
        self.update_time_label(0)

    # ── Spectrogram & playback slots ─────────────────────────────────────

    @Slot(str)
    def _on_display_mode_changed(self, mode: str) -> None:
        if mode == "spectrogram" and self.cached_audio:
            if getattr(self.page.wf_panel.waveform._spec_renderer, "_spec_data", None) is None:
                self.show_waveform(self.cached_audio[1], self.cached_audio[3])

    @Slot(QAction)
    def _on_spec_fft_changed(self, action: QAction) -> None:
        self.page.wf_panel.waveform.set_spec_fft(int(action.data()))

    @Slot(QAction)
    def _on_spec_window_changed(self, action: QAction) -> None:
        self.page.wf_panel.waveform.set_spec_window(action.data())

    @Slot(QAction)
    def _on_spec_cmap_changed(self, action: QAction) -> None:
        self.page.wf_panel.waveform.set_colormap(action.data())

    @Slot(QAction)
    def _on_spec_floor_changed(self, action: QAction) -> None:
        self.page.wf_panel.waveform.set_spec_db_floor(float(action.data()))

    @Slot(QAction)
    def _on_spec_ceil_changed(self, action: QAction) -> None:
        self.page.wf_panel.waveform.set_spec_db_ceil(float(action.data()))

    @Slot()
    def play_preview(self) -> None:
        if not self.cached_audio:
            return
        _, _display, playback_audio, samplerate = self.cached_audio
        self.stop_preview()
        start = self.page.wf_panel.waveform._cursor_sample
        mode, channel = self.page.wf_panel.play_mode
        self._playback.play(playback_audio, samplerate, start, mode=mode, channel=channel)
        if self._playback.is_playing:
            self.page.wf_panel.play_btn.setEnabled(False)
            self.page.wf_panel.stop_btn.setEnabled(True)

    @Slot()
    def stop_preview(self, *, return_to_start: bool = True) -> None:
        was_playing = self._playback.is_playing
        start_sample = self._playback.play_start_sample
        self._playback.stop()
        self.page.wf_panel.stop_btn.setEnabled(False)
        self.page.wf_panel.play_btn.setEnabled(self.cached_audio is not None)
        if was_playing and return_to_start:
            self.page.wf_panel.waveform.set_cursor(start_sample)
            self.update_time_label(start_sample)

    @Slot(int)
    def _on_wf_seek(self, sample: int) -> None:
        if self._playback.is_playing:
            self.stop_preview(return_to_start=False)
        self.update_time_label(sample)

    def update_time_label(self, current_sample: int) -> None:
        if not self.cached_audio:
            return
        _, display_audio, _playback, sr = self.cached_audio
        if display_audio is None:
            total = 0
        elif isinstance(display_audio, list):
            total = display_audio[0].shape[0] if display_audio else 0
        else:
            total = display_audio.shape[0]
        self.page.wf_panel.time_label.setText(
            f"{format_duration(current_sample, sr)} / {format_duration(total, sr)}"
        )


def _split_tree_items(items: list[QTreeWidgetItem]) -> tuple[list[QTreeWidgetItem], list[QTreeWidgetItem]]:
    file_items: list[QTreeWidgetItem] = []
    channel_items: list[QTreeWidgetItem] = []
    for it in items:
        data = it.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            continue
        if data[0] == "file":
            file_items.append(it)
        elif data[0] == "channel":
            channel_items.append(it)
    return file_items, channel_items


def _collect_multi_input_specs(
    track_map: dict[str, Any],
    file_items: list[Any] | None,
    channel_items: list[Any] | None,
) -> list[tuple[Any, str, list[int] | None]]:
    items: list[tuple[Any, str, list[int] | None]] = []
    for fi in file_items or []:
        data = fi.data(0, Qt.ItemDataRole.UserRole)
        if not data or data[0] != "file":
            continue
        track = track_map.get(data[1])
        if track:
            items.append((track.filepath, os.path.splitext(data[1])[0], None))
    ch_map: dict[str, list[int]] = {}
    for ci in channel_items or []:
        data = ci.data(0, Qt.ItemDataRole.UserRole)
        if data and data[0] == "channel":
            ch_map.setdefault(data[1], []).append(data[2])
    for filename, channels in ch_map.items():
        track = track_map.get(filename)
        if track:
            items.append((track.filepath, os.path.splitext(filename)[0], sorted(channels)))
    return items


def _collect_multi_output_specs(
    topo: Any,
    file_items: list[Any] | None,
    channel_items: list[Any] | None,
) -> list[tuple[Any, str, list[int] | None]]:
    items: list[tuple[Any, str, list[int] | None]] = []
    for fi in file_items or []:
        data = fi.data(0, Qt.ItemDataRole.UserRole)
        if not data or data[0] != "file":
            continue
        entry = next((e for e in topo.entries if e.output_filename == data[1]), None)
        if entry:
            items.append((entry, os.path.splitext(data[1])[0], None))
    ch_map: dict[str, list[int]] = {}
    for ci in channel_items or []:
        data = ci.data(0, Qt.ItemDataRole.UserRole)
        if data and data[0] == "channel":
            ch_map.setdefault(data[1], []).append(data[2])
    for out_fn, channels in ch_map.items():
        entry = next((e for e in topo.entries if e.output_filename == out_fn), None)
        if entry:
            items.append((entry, os.path.splitext(out_fn)[0], sorted(channels)))
    return items


def _format_unverified_prompt(
    output_dir: str, unverified: dict[str, str]
) -> tuple[str, str, str, str, str]:
    """Build unified status-bar prompt parts, tooltip, and accessible description."""
    folder = os.path.basename(os.path.normpath(output_dir)) or output_dir
    names = sorted(unverified.keys())
    first = names[0]
    count = len(names)
    if count == 1:
        prefix = f"1 existing file in {folder}/ could not be verified ("
        suffix = "). Overwrite it?"
        accept_desc = f"Overwrite 1 unverified file ({first}) in {folder}/"
    else:
        prefix = f"{count} existing files in {folder}/ could not be verified (first: "
        suffix = "). Overwrite them?"
        accept_desc = f"Overwrite {count} unverified files in {folder}/"

    shown = names[:20]
    lines = [
        f"Folder: {output_dir}",
        (
            "These target files were not recorded in .sessionprep-layout.json or changed on disk. "
            "Only the listed target files will be overwritten; unrelated files in that folder "
            "will not be touched."
        ),
        *(f"\u2022 {name}" for name in shown),
    ]
    if count > len(shown):
        lines.append(f"\u2026 and {count - len(shown)} more")
    return prefix, first, suffix, "\n".join(lines), accept_desc
