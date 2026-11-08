"""Navigation-tree | page splitters start without an unpainted gap."""
import copy

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QSplitter, QStackedWidget, QTreeWidget

from sessionprepgui.controls import fit_navigation_pane
from sessionprepgui.prefs.dialog import PreferencesDialog
from sessionprepgui.settings import build_defaults


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    return app


def _assert_no_gap(splitter):
    tree, page = splitter.widget(0), splitter.widget(1)
    # The splitter's own size for the tree equals the painted tree width, and
    # the page starts right after the handle (no surplus beside the tree).
    assert splitter.sizes()[0] == tree.width()
    assert page.x() - (tree.x() + tree.width()) <= splitter.handleWidth() + 1


def test_fit_navigation_pane_uses_the_tree_maximum(qapp):
    splitter = QSplitter(Qt.Horizontal)
    tree = QTreeWidget()
    tree.setMinimumWidth(140)
    tree.setMaximumWidth(200)
    splitter.addWidget(tree)
    splitter.addWidget(QStackedWidget())
    splitter.setStretchFactor(0, 0)
    splitter.setStretchFactor(1, 1)
    fit_navigation_pane(splitter)
    splitter.resize(1100, 600)
    splitter.show()
    qapp.processEvents()
    assert tree.width() == 200
    _assert_no_gap(splitter)
    splitter.resize(1400, 600)  # widening keeps the tree, gives the page the rest
    qapp.processEvents()
    assert tree.width() == 200
    _assert_no_gap(splitter)
    splitter.close()


def test_fit_navigation_pane_ignores_incomplete_splitters(qapp):
    splitter = QSplitter(Qt.Horizontal)
    fit_navigation_pane(splitter)  # no panes: nothing to do
    splitter.addWidget(QTreeWidget())
    fit_navigation_pane(splitter)  # one pane: nothing to do


def test_preferences_trees_have_no_gap_beside_the_splitter(qapp):
    dialog = PreferencesDialog(copy.deepcopy(build_defaults()))
    dialog.resize(1150, 720)
    dialog.show()
    qapp.processEvents()
    for index in range(dialog._tabs.count()):
        dialog._tabs.setCurrentIndex(index)
        qapp.processEvents()
        splitter = dialog._tabs.currentWidget().findChild(QSplitter)
        _assert_no_gap(splitter)
        assert splitter.widget(0).width() == splitter.widget(0).maximumWidth()
    dialog.close()


def test_fit_navigation_pane_leaves_uncapped_trees_to_qt(qapp):
    def build(fit):
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(QTreeWidget())
        splitter.addWidget(QStackedWidget())
        if fit:
            fit_navigation_pane(splitter)
        splitter.resize(1100, 600)
        splitter.show()
        qapp.processEvents()
        sizes = splitter.sizes()
        splitter.close()
        return sizes

    assert build(True) == build(False)
