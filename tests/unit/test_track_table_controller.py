"""Isolated unit tests for TrackTableController (Phase 2 & Phase 3 track tables)."""

from __future__ import annotations

from types import SimpleNamespace

from PySide6.QtWidgets import QDoubleSpinBox, QTableWidget, QTableWidgetItem

from sessionpreplib.models import ProcessorResult, SessionContext, TrackContext, TransferEntry
from sessionpreplib.processors.bimodal_normalize import BimodalNormalizeProcessor
from sessionprepgui.state import WorkspaceState
from sessionprepgui.tracks.table_controller import TrackTableController
from sessionprepgui.tracks.table_widgets import _DraggableTrackTable, _SetupDragTable


def _make_track(filename: str, *, status: str = "OK", gain_db: float = -3.0, cls: str = "Sustained"):
    track = TrackContext(filename, f"/tmp/{filename}", None, 48000, 1, 48000, "24-bit", "PCM_24", 1.0, status=status)
    if status == "OK":
        track.processor_results["bimodal_normalize"] = ProcessorResult(
            processor_id="bimodal_normalize",
            gain_db=gain_db,
            classification=cls,
            method="RMS → -18 dB",
            data={"gain_db_individual": gain_db, "fader_offset": 1.5},
        )
    return track


def test_track_table_controller_populates_ok_silent_and_error_rows(_font_app):
    t_ok = _make_track("vox.wav", gain_db=-2.5, cls="Sustained")
    t_silent = _make_track("sil.wav", gain_db=0.0, cls="Silent")
    t_err = _make_track("bad.wav", status="Error: corrupt")

    proc = BimodalNormalizeProcessor()
    proc.configure({})
    session = SessionContext([t_ok, t_silent, t_err], {}, processors=[proc])
    state = WorkspaceState()
    state.session = session

    table = _DraggableTrackTable()
    table.setColumnCount(8)
    table.setRowCount(3)
    for row, tr in enumerate(session.tracks):
        table.setItem(row, 0, QTableWidgetItem(tr.filename))

    shell = SimpleNamespace(progress_start=lambda *a, **k: None, progress_value=lambda *a, **k: None,
                            progress_finish=lambda *a, **k: None, progress_fail=lambda *a, **k: None)
    ctrl = TrackTableController(state, shell, track_table=table)
    try:
        ctrl.populate_table(session)
        assert ctrl.find_table_row("vox.wav") >= 0
        ok_row = ctrl.find_table_row("vox.wav")
        sil_row = ctrl.find_table_row("sil.wav")
        err_row = ctrl.find_table_row("bad.wav")

        spin = table.cellWidget(ok_row, 4)
        assert isinstance(spin, QDoubleSpinBox)
        assert spin.value() == -2.5
        assert table.item(sil_row, 3).text() == "Silent"
        assert table.item(err_row, 3).text() == "Error"
    finally:
        table.deleteLater()
        _font_app.processEvents()


def test_track_table_controller_populates_setup_table_from_manifest(_font_app):
    t_ok = _make_track("kick.wav", gain_db=-4.0, cls="Transient")
    session = SessionContext(
        tracks=[t_ok],
        config={},
        output_tracks=[t_ok],
        transfer_manifest=[TransferEntry("kick.wav", "kick.wav", "Kick", group="Drums")],
        daw_state={"pt": {"assignments": {"kick.wav": "f1"}}},
    )
    state = WorkspaceState()
    state.session = session
    state.active_daw_processor = SimpleNamespace(id="pt")

    track_table = QTableWidget(0, 8)
    setup_table = _SetupDragTable()
    setup_table.setColumnCount(7)
    shell = SimpleNamespace()
    ctrl = TrackTableController(state, shell, track_table=track_table, setup_table=setup_table)
    try:
        ctrl.populate_setup_table()
        assert setup_table.rowCount() == 1
        assert setup_table.item(0, 0).text() == "Kick"
        assert setup_table.item(0, 1).text() == "✓"
        assert setup_table.item(0, 2).text() == "kick.wav"
        assert setup_table.item(0, 4).text() == "-4.0 dB"
        assert setup_table.item(0, 5).text() == "+1.5 dB"
        assert setup_table.item(0, 6).text() == "Drums"
    finally:
        track_table.deleteLater()
        setup_table.deleteLater()
        _font_app.processEvents()
