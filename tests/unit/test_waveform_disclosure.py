"""Pointer and keyboard behavior of the shared waveform disclosure."""
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget, QToolButton

from sessionprepgui.icons import icon
from sessionprepgui.theme import apply_dark_theme
from sessionprepgui.waveform.panel import WaveformPanel


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    return app


@pytest.fixture(params=[False, True], ids=["track-layout", "file-preview"])
def preview(qapp, request):
    window = QWidget()
    layout = QVBoxLayout(window)
    before = QPushButton("Before")
    layout.addWidget(before)
    panel = WaveformPanel(analysis_mode=request.param)
    layout.addWidget(panel)
    layout.addWidget(QPushButton("After"))
    apply_dark_theme(window)
    window.resize(800, 400)
    panel.set_expanded(False)
    window.show()
    window.activateWindow()
    qapp.processEvents()
    yield window, before, panel
    window.close()


def blue_border_pixels(button):
    image = button.grab().toImage()
    points = {(x, y) for x in range(image.width())
              for y in (0, 1, image.height() - 2, image.height() - 1)}
    points |= {(x, y) for y in range(image.height())
               for x in (0, 1, image.width() - 2, image.width() - 1)}
    return sum(1 for x, y in points
               if image.pixelColor(x, y).red() < 120
               and 125 < image.pixelColor(x, y).green() < 190
               and image.pixelColor(x, y).blue() > 225)


def tab_to_disclosure(qapp, before, button):
    before.setFocus(Qt.TabFocusReason)
    for _ in range(20):
        QTest.keyClick(QApplication.focusWidget(), Qt.Key_Tab)
        qapp.processEvents()
        if button.hasFocus():
            return
    pytest.fail("Waveform disclosure could not be reached with Tab")


def test_collapsed_points_up_and_expanded_points_down(preview):
    _, _, panel = preview
    button = panel.disclosure_button
    assert button.icon().pixmap(16, 16).toImage() == icon("chevron-up").pixmap(16, 16).toImage()
    panel.set_expanded(True)
    assert button.icon().pixmap(16, 16).toImage() == icon("chevron-down").pixmap(16, 16).toImage()


def test_pointer_focus_does_not_look_selected(qapp, preview):
    _, before, panel = preview
    button = panel.disclosure_button
    before.setFocus()
    QTest.mouseClick(button, Qt.LeftButton)
    qapp.processEvents()
    assert panel.is_expanded()
    assert button.hasFocus()
    assert blue_border_pixels(button) == 0
    QTest.mouseClick(button, Qt.LeftButton)
    qapp.processEvents()
    assert not panel.is_expanded()
    assert blue_border_pixels(button) == 0


def test_tab_space_and_backtab_keep_keyboard_focus_visible(qapp, preview):
    _, before, panel = preview
    button = panel.disclosure_button
    tab_to_disclosure(qapp, before, button)
    assert blue_border_pixels(button) > 0
    QTest.keyClick(button, Qt.Key_Space)
    qapp.processEvents()
    assert panel.is_expanded() and button.hasFocus()
    assert blue_border_pixels(button) > 0
    QTest.keyClick(button, Qt.Key_Tab)
    qapp.processEvents()
    assert not button.hasFocus()
    QTest.keyClick(QApplication.focusWidget(), Qt.Key_Backtab)
    qapp.processEvents()
    assert button.hasFocus()
    assert blue_border_pixels(button) > 0


def test_mouse_click_after_keyboard_focus_clears_only_pointer_outline(qapp, preview):
    _, before, panel = preview
    button = panel.disclosure_button
    tab_to_disclosure(qapp, before, button)
    assert blue_border_pixels(button) > 0
    QTest.mouseClick(button, Qt.LeftButton)
    qapp.processEvents()
    assert button.hasFocus()
    assert blue_border_pixels(button) == 0
    fit = next(w for w in panel.preview_header.findChildren(QToolButton)
               if w.accessibleName() == "Fit waveform")
    fit.setFocus(Qt.TabFocusReason)
    qapp.processEvents()
    assert blue_border_pixels(fit) > 0


def test_keyboard_activation_after_pointer_focus_shows_outline(qapp, preview):
    _, before, panel = preview
    button = panel.disclosure_button
    before.setFocus()
    QTest.mouseClick(button, Qt.LeftButton)
    qapp.processEvents()
    assert blue_border_pixels(button) == 0
    QTest.keyClick(button, Qt.Key_Space)
    qapp.processEvents()
    assert not panel.is_expanded() and button.hasFocus()
    assert blue_border_pixels(button) > 0
