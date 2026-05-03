from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QMessageBox

from sessionprepgui.daw.mixin import DawMixin
from sessionpreplib.models import SessionContext, TransferEntry


class _StatusBar:
    def __init__(self):
        self.messages: list[str] = []

    def showMessage(self, message: str):
        self.messages.append(message)


class _Harness:
    def __init__(self):
        self._active_daw_processor = SimpleNamespace(id="protools_0")
        self._session = SessionContext(
            tracks=[],
            config={},
            transfer_manifest=[
                TransferEntry("track-1", "track-1.wav", "Track 1"),
            ],
            daw_state={
                "protools_0": {
                    "assignments": {
                        "track-1": "folder-1",
                        "duplicate-1": "folder-1",
                    },
                    "track_order": {
                        "folder-1": ["track-1", "duplicate-1"],
                        "folder-2": ["duplicate-2"],
                    },
                }
            },
        )
        self._status_bar = _StatusBar()
        self.folder_tree_populated = 0
        self.setup_table_populated = 0
        self.lifecycle_updated = 0

    def _populate_folder_tree(self):
        self.folder_tree_populated += 1

    def _populate_setup_table(self):
        self.setup_table_populated += 1

    def _update_daw_lifecycle_buttons(self):
        self.lifecycle_updated += 1


def test_prune_assignments_to_manifest_removes_stale_entries():
    harness = _Harness()

    DawMixin._prune_assignments_to_manifest(harness)

    state = harness._session.daw_state["protools_0"]
    assert state["assignments"] == {"track-1": "folder-1"}
    assert state["track_order"] == {"folder-1": ["track-1"]}


def test_unassign_all_clears_assignment_state(monkeypatch):
    harness = _Harness()
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.Yes,
    )

    DawMixin._on_unassign_all(harness)

    state = harness._session.daw_state["protools_0"]
    assert state["assignments"] == {}
    assert state["track_order"] == {}
    assert harness.folder_tree_populated == 1
    assert harness.setup_table_populated == 1
    assert harness.lifecycle_updated == 1
    assert harness._status_bar.messages == [
        "All DAW folder assignments cleared."
    ]
