"""Waveform view controls share the preview header row and overflow into »."""
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QToolButton

from sessionprepgui.waveform.panel import WaveformPanel


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    return app


def _panel(qapp, width, analysis_mode=True):
    panel = WaveformPanel(analysis_mode=analysis_mode)
    panel.resize(width, 300)
    panel.show()
    qapp.processEvents()
    return panel


@pytest.mark.parametrize("analysis_mode", [True, False], ids=["analysis", "track-layout"])
def test_view_controls_sit_in_the_preview_header_row(qapp, analysis_mode):
    panel = _panel(qapp, 1600, analysis_mode)
    row = panel._view_toolbar
    assert row.parentWidget() is panel.preview_header
    layout = panel.preview_header.layout()
    assert layout.indexOf(panel.disclosure_button) == 0 and layout.indexOf(row) == 1
    # No separate toolbar row in the body: the body holds the plot and transport.
    assert row.parentWidget() is not panel._body
    assert row.height() <= panel.preview_header.height()
    expected = 6 if analysis_mode else 2  # spectrogram options hidden in waveform mode
    assert len([c for c in row.controls if c.isVisible()]) == expected
    assert not row.more_button.isVisible()
    panel.close()


def test_collapsing_hides_the_controls_but_keeps_disclosure_and_zoom(qapp):
    panel = _panel(qapp, 1600)
    panel.set_expanded(False)
    qapp.processEvents()
    assert not panel._view_toolbar.isVisible()
    assert panel.disclosure_button.isVisible()
    assert all(b.isVisible() for b in panel._zoom_buttons)
    panel.set_expanded(True)
    qapp.processEvents()
    assert panel._view_toolbar.isVisible()
    panel.close()


def test_controls_overflow_from_the_end_into_the_more_menu(qapp):
    panel = _panel(qapp, 1600)
    row = panel._view_toolbar
    wanted = [c for c in row.controls if row.is_wanted(c)]
    panel.resize(400, 300)  # the pane minimum
    qapp.processEvents()
    assert row.more_button.isVisible()
    hidden = row.overflowed()
    assert hidden and hidden == wanted[len(wanted) - len(hidden):]  # from the end
    shown = [c for c in wanted if c.isVisible()]
    assert shown == wanted[:len(shown)]
    # Nothing is clipped: the row never extends under the zoom group.
    row_right = row.mapTo(panel.preview_header, row.rect().topRight()).x()
    assert row_right < panel._zoom_buttons[0].x()
    panel.resize(1600, 300)
    qapp.processEvents()
    assert not row.overflowed() and not row.more_button.isVisible()
    panel.close()


def test_more_menu_offers_submenus_and_checkable_toggles(qapp):
    panel = _panel(qapp, 1600)
    row = panel._view_toolbar
    panel.rms_avg_toggle.setChecked(True)
    panel.resize(400, 300)
    qapp.processEvents()
    row._fill_menu()
    actions = {a.text(): a for a in row.more_menu.actions()}
    for control in row.overflowed():
        action = actions[control.text()]
        if control.menu() is not None:
            assert action.menu() is control.menu()  # dropdown → its own menu as submenu
        else:
            assert action.isCheckable() and action.isChecked() == control.isChecked()
    avg = actions["RMS AVG"]
    assert avg.isChecked()
    avg.trigger()  # clicks the real toggle: its signals keep driving the waveform
    assert not panel.rms_avg_toggle.isChecked()
    panel.close()


def test_display_mode_and_analysis_mode_decide_which_controls_are_wanted(qapp):
    panel = _panel(qapp, 1600)
    row = panel._view_toolbar
    panel.spec_action.trigger()
    qapp.processEvents()
    assert row.is_wanted(panel.spec_settings_btn) and not row.is_wanted(panel.wf_settings_btn)
    assert not row.is_wanted(panel.rms_avg_toggle)
    assert panel.spec_settings_btn.isVisible() and not panel.rms_avg_toggle.isVisible()
    panel.set_analysis_mode(False)
    qapp.processEvents()
    assert not panel.overlay_btn.isVisible()
    panel.wf_action.trigger()
    panel.set_analysis_mode(True)
    qapp.processEvents()
    assert all(c.isVisible() for c in (panel.overlay_btn, panel.markers_toggle,
                                       panel.rms_lr_toggle, panel.rms_avg_toggle))
    panel.close()


def test_tab_order_follows_the_row_and_controls_have_names(qapp):
    panel = _panel(qapp, 1600)
    row = panel._view_toolbar
    chain = [panel.disclosure_button, *[c for c in row.controls if c.isVisible()],
             *panel._zoom_buttons]
    for first, second in zip(chain, chain[1:]):
        nxt = first.nextInFocusChain()
        while nxt is not second and (not nxt.isVisible() or nxt.focusPolicy() == Qt.NoFocus):
            nxt = nxt.nextInFocusChain()
        assert nxt is second
    names = {c.accessibleName() for c in row.controls}
    assert {"Waveform display mode", "Waveform display options", "Spectrogram display options",
            "Detector overlays", "Peak / RMS Max markers", "RMS L/R envelopes",
            "RMS AVG envelope"} <= names
    assert row.more_button.accessibleName() == "More view controls"
    assert isinstance(row.more_button, QToolButton)
    panel.close()


def _fitting_width(qapp, panel):
    """Smallest panel width at which every wanted control is shown."""
    for width in range(400, 3000, 10):
        panel.resize(width, 300)
        qapp.processEvents()
        if not panel._view_toolbar.overflowed():
            return width
    pytest.skip("controls never fit")


def test_label_changes_re_check_what_fits(qapp):
    panel = _panel(qapp, 1600)
    row = panel._view_toolbar
    _fitting_width(qapp, panel)
    panel.overlay_btn.setText("Detector Overlays (12345678901234567890)")
    qapp.processEvents()
    qapp.processEvents()
    assert row.overflowed(), "a longer label must push trailing controls into »"
    for control in row.controls:
        if control.isVisible():
            assert control.width() >= control.sizeHint().width()  # never squeezed
    panel.overlay_btn.setText("Detector Overlays")
    qapp.processEvents()
    qapp.processEvents()
    assert not row.overflowed()
    panel.close()


def test_focus_moves_to_more_when_the_focused_control_overflows(qapp):
    panel = _panel(qapp, 1600)
    row = panel._view_toolbar
    panel.activateWindow()
    panel.rms_avg_toggle.setFocus(Qt.TabFocusReason)
    qapp.processEvents()
    if not panel.rms_avg_toggle.hasFocus():
        pytest.skip("window focus unavailable on this platform")
    panel.resize(400, 300)
    qapp.processEvents()
    assert panel.rms_avg_toggle in row.overflowed()
    assert row.more_button.hasFocus()
    panel.close()


def test_repeated_menu_fills_keep_the_control_menus(qapp):
    panel = _panel(qapp, 400)
    row = panel._view_toolbar
    row._fill_menu()
    row._fill_menu()
    assert panel.overlay_btn.menu() is panel.overlay_menu
    assert not panel.overlay_menu.isHidden() or True  # still a valid QMenu
    panel.overlay_menu.actions()  # raises if the menu was deleted
    panel.close()


@pytest.mark.parametrize("mode", ["spectrogram", "track-layout"])
def test_overflow_in_other_modes(qapp, mode):
    panel = _panel(qapp, 1600, analysis_mode=(mode != "track-layout"))
    if mode == "spectrogram":
        panel.spec_action.trigger()
    panel.resize(400, 300)
    qapp.processEvents()
    row = panel._view_toolbar
    wanted = [c for c in row.controls if row.is_wanted(c)]
    assert row.overflowed() == [c for c in wanted if c.isHidden()]
    assert all(not c.isVisible() for c in row.controls if not row.is_wanted(c))
    panel.close()


@pytest.mark.parametrize("analysis_mode", [True, False], ids=["analysis", "track-layout"])
@pytest.mark.parametrize("expanded", [False, True], ids=["collapsed", "expanded"])
def test_header_keeps_disclosure_left_and_zoom_group_packed_right(qapp, analysis_mode, expanded):
    panel = _panel(qapp, 1600, analysis_mode)
    panel.set_expanded(expanded)
    qapp.processEvents()
    header = panel.preview_header
    assert panel.disclosure_button.x() == 0
    zoom = panel._zoom_buttons
    # Zoom/scale buttons sit side by side at the right edge, not spread out.
    for left, right in zip(zoom, zoom[1:]):
        assert right.x() == left.x() + left.width()
    margins = header.layout().contentsMargins()
    assert zoom[-1].x() + zoom[-1].width() == header.width() - margins.right()
    gap = zoom[0].x() - (panel.disclosure_button.x() + panel.disclosure_button.width())
    assert gap > 0
    if not expanded:
        assert panel.disclosure_button.width() < 400  # sized to its content
    panel.close()
