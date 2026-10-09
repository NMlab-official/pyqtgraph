import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from tests.perf_helpers import count_calls, paints_per_update, show_and_wait

app = pg.mkQApp()


class _Dummy:
    def method(self, value):
        return value * 2


def test_count_calls_counts_and_restores():
    original = _Dummy.method
    obj = _Dummy()
    with count_calls(_Dummy, 'method') as counter:
        assert obj.method(2) == 4
        assert obj.method(3) == 6
        assert counter.count == 2
        counter.reset()
        assert counter.count == 0
    assert _Dummy.method is original
    assert 'method' not in vars(obj)


def test_count_calls_inherited_method_is_restored():
    class Child(_Dummy):
        pass

    with count_calls(Child, 'method') as counter:
        Child().method(1)
        _Dummy().method(1)
    assert counter.count == 1
    assert 'method' not in vars(Child)


def test_count_calls_on_instance():
    obj = _Dummy()
    with count_calls(obj, 'method') as counter:
        obj.method(1)
        _Dummy().method(1)
    assert counter.count == 1
    assert 'method' not in vars(obj)


def test_paints_per_update_autorange_streaming():
    pw = pg.PlotWidget()
    pw.resize(400, 300)
    curve = pw.plot(np.zeros(100))
    rng = np.random.default_rng(0)

    def update(i):
        # growing amplitude: the auto-range changes the y range on every update
        curve.setData(rng.normal(size=100) * (1 + i))

    try:
        assert paints_per_update(pw, pg.PlotCurveItem, update, n=10) <= 1.05
    finally:
        pw.close()


def test_show_and_wait_keeps_the_window_out_of_activation_and_input():
    # Activation changes and mouse moves repaint a window: the windows of tests running
    # in parallel, and the mouse cursor, used to add paints to the counted ones.
    pw = pg.PlotWidget()
    pw.resize(300, 200)
    try:
        show_and_wait(pw)
        flags = pw.windowFlags()
        assert flags & QtCore.Qt.WindowType.WindowDoesNotAcceptFocus
        assert flags & QtCore.Qt.WindowType.WindowTransparentForInput
        assert pw.testAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating)
        if QtWidgets.QApplication.platformName() == 'windows':
            # the offscreen platform activates any window shown
            assert not pw.isActiveWindow()
        # shown again after being hidden: the window is not recreated (macOS may report
        # it exposed only later)
        pw.hide()
        show_and_wait(pw)
        assert pw.windowFlags() == flags
    finally:
        pw.close()
