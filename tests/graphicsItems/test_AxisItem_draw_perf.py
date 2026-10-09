"""
Performance regression tests of the AxisItem drawing during a pan or a zoom.

- ``tickValues`` drops the ticks of a level close to a tick of a previous level by
  bisection, without numpy: same values as the numpy reference, no ``np.isclose``.
- ``generateDrawSpecs`` creates the tick end points without ``Point.__init__``.
- The axis draws its specifications directly when painted, instead of recording them
  into a ``QPicture`` then replaying it, when that draws the pixels of the replayed
  picture: Qt 6, raster engine at the resolution of the primary screen, cosmetic pens
  at most one pixel wide, no tick with equal ends, ``drawPicture`` not overridden.
  Otherwise the picture is recorded and replayed as before. The tests compare both
  drawings pixel for pixel.
"""
from __future__ import annotations

from collections.abc import Callable
from math import ceil

import numpy as np
import pytest

import pyqtgraph as pg
from pyqtgraph.graphicsItems import AxisItem as axisModule
from pyqtgraph.graphicsItems.AxisItem import _DIRECT_DRAWING, AxisItem, _tickLevelValues
from pyqtgraph.Point import Point
from pyqtgraph.Qt import QtGui
from tests.perf_helpers import count_calls, process_events, show_and_wait

app = pg.mkQApp()

direct_drawing = pytest.mark.skipif(
    not _DIRECT_DRAWING, reason="Qt 5 lays replayed text out again: axes replay pictures")


def _reference_tick_values(axis: AxisItem, minVal: float, maxVal: float,
                           size: float) -> list:
    """``AxisItem.tickValues`` before the bisection, comparing all pairs with numpy."""
    minVal, maxVal = sorted((minVal, maxVal))
    minVal *= axis.scale
    maxVal *= axis.scale
    ticks = []
    tickLevels = axis.tickSpacing(minVal, maxVal, size)
    allValues = np.array([])
    for i in range(len(tickLevels)):
        spacing, offset = tickLevels[i]
        start = (ceil((minVal-offset) / spacing) * spacing) + offset
        num = int((maxVal-start) / spacing) + 1
        values = (np.arange(num) * spacing + start) / axis.scale
        close = np.any(
            np.isclose(allValues, values[:, np.newaxis], rtol=0,
                       atol=spacing/axis.scale*0.01),
            axis=-1
        )
        values = values[~close]
        allValues = np.concatenate([allValues, values])
        ticks.append((spacing/axis.scale, values.tolist()))
    if axis.logMode:
        return axis.logTickValues(minVal, maxVal, size, ticks)
    return ticks


def _assert_same_ticks(actual: list, expected: list) -> None:
    assert len(actual) == len(expected)
    for (spacing, values), (refSpacing, refValues) in zip(actual, expected):
        assert spacing == refSpacing
        assert values == refValues  # exact float equality, value for value
        assert all(type(v) is type(r) for v, r in zip(values, refValues))


def _ranges(rng: np.random.Generator) -> list[tuple[float, float, float]]:
    ranges = [(0, 1, 500), (-3, 3, 300), (1e-6, 3e-6, 800), (100.5, 1100.5, 1200),
              (-1e9, 2e9, 700), (1e5, 1e5 + 0.01, 400), (5, -5, 600), (0.1, 0.1 + 1e-12, 90),
              (1.7e9, 1.7e9 + 3600, 1000), (-0.5, 0.25, 3)]
    for _ in range(300):
        center = rng.choice([0.0, 1.0, -7.3, 1e3, 1.7e9, 1e-4]) + rng.normal() * 10.0
        span = 10.0 ** rng.uniform(-6, 6)
        low = center - span * rng.uniform(0, 1)
        ranges.append((low, low + span, float(rng.uniform(20, 2500))))
    return ranges


@pytest.mark.parametrize('setup', ['default', 'scale', 'int_scale', 'negative_scale',
                                   'density', 'spacing', 'int_spacing', 'log', 'levels'])
def test_tick_values_equal_the_numpy_reference(setup):
    axis = AxisItem('bottom')
    if setup == 'scale':
        axis.setScale(1e-3)
    elif setup == 'int_scale':
        axis.setScale(3)
    elif setup == 'negative_scale':
        axis.setScale(-2.5)
    elif setup == 'density':
        axis.setTickDensity(2.7)
    elif setup == 'spacing':
        axis.setTickSpacing(major=10.0, minor=2.5)
    elif setup == 'int_spacing':
        axis.setTickSpacing(levels=[(10, 0), (3, 1)])
    elif setup == 'log':
        axis.setLogMode(True)
    elif setup == 'levels':
        axis.setStyle(maxTickLevel=0)
    rng = np.random.default_rng(7)
    for low, high, size in _ranges(rng):
        if setup in ('spacing', 'int_spacing') and abs(high - low) > 1e4:
            continue  # explicit spacings: too many ticks
        if setup == 'log' and max(abs(low), abs(high)) > 50:
            continue
        _assert_same_ticks(axis.tickValues(low, high, size),
                           _reference_tick_values(axis, low, high, size))


def test_tick_level_values_falls_back_to_numpy_for_other_inputs():
    levels = [(np.float64(10.0), 0), (np.float64(2.5), 0)]
    with count_calls(np, 'isclose') as isclose:
        ticks = _tickLevelValues(levels, 0.0, 100.0, 1.0)
    assert isclose.count == 2
    assert ticks == [(10.0, [float(v) for v in range(0, 101, 10)]),
                     (2.5, [v * 2.5 for v in range(41) if v % 4])]


def test_tick_values_do_not_compare_all_pairs():
    axis = AxisItem('bottom')
    with count_calls(np, 'isclose') as isclose:
        for low in np.linspace(0, 50, 20):
            axis.tickValues(low, low + 1000, 1200)
    assert isclose.count == 0


def _generate(axis: AxisItem) -> tuple:
    picture = QtGui.QPicture()
    painter = QtGui.QPainter(picture)
    try:
        return axis.generateDrawSpecs(painter)
    finally:
        painter.end()


def test_tick_points_are_created_without_point_init():
    pw = pg.PlotWidget()
    pw.resize(800, 500)
    show_and_wait(pw)
    axis = pw.getAxis('bottom')
    axis.unlinkFromView()
    axis.setRange(0, 1000)
    try:
        counts = []
        for maxTickLevel in (0, 2):
            axis.setStyle(maxTickLevel=maxTickLevel)
            with count_calls(Point, '__init__') as inits:
                _, tickSpecs, _ = _generate(axis)
            assert all(type(p1) is Point and type(p2) is Point for _, p1, p2 in tickSpecs)
            counts.append((len(tickSpecs), inits.count))
        # a constant number of points, whatever the number of ticks
        assert counts[0][0] != counts[1][0]
        assert counts[0][1] == counts[1][1]
    finally:
        pw.close()


# --------------------------------------------------------------------------------------
# direct drawing
# --------------------------------------------------------------------------------------

def _grab(widget) -> np.ndarray:
    process_events()
    image = widget.grab().toImage().convertToFormat(QtGui.QImage.Format.Format_ARGB32)
    return pg.functions.ndarray_from_qimage(image).copy()


def _axes(widget) -> list[AxisItem]:
    return [ax for plot in _plots(widget) for ax in (plot.getAxis(side) for side in
            ('left', 'bottom', 'right', 'top')) if ax.isVisible()]


def _plots(widget) -> list[pg.PlotItem]:
    if isinstance(widget, pg.PlotWidget):
        return [widget.getPlotItem()]
    return [item for item in widget.ci.items if isinstance(item, pg.PlotItem)]


def _replayed(widget, monkeypatch) -> np.ndarray:
    """Pixels of ``widget`` with axis pictures recorded and replayed, as before."""
    with monkeypatch.context() as patch:
        patch.setattr(axisModule, '_drawsAtPictureResolution', lambda p: False)
        for axis in _axes(widget):
            axis.picture = None
        image = _grab(widget)
        assert all(axis.picture._picture is not None for axis in _axes(widget))
    for axis in _axes(widget):
        axis.picture = None
    return image


def _plot(**kwargs) -> pg.PlotWidget:
    pw = pg.PlotWidget(**kwargs)
    pw.getPlotItem().hideButtons()
    pw.plot(np.cumsum(np.random.default_rng(0).standard_normal(500)), pen='y')
    pw.resize(640, 400)
    return pw


def _setup_grid(pw):
    pw.showGrid(x=True, y=True)


def _setup_translucent_grid(pw):
    pw.showGrid(x=True, y=True, alpha=0.6)
    for side in ('left', 'bottom'):
        pw.getAxis(side).setStyle(opaqueGrid=False)


def _setup_fonts(pw):
    pw.setFont(QtGui.QFont('Times New Roman', 15))
    pw.getAxis('left').setTickFont(QtGui.QFont('Courier New', 11))


def _setup_labels(pw):
    pw.setLabel('left', 'Amplitude', units='V')
    pw.setLabel('bottom', 'Time', units='s')
    pw.showAxis('top')
    pw.showAxis('right')
    pw.getAxis('left').setStyle(tickAlpha=0.5, tickTextOffset=9)


def _setup_log(pw):
    pw.getPlotItem().setLogMode(True, True)
    pw.plot(np.arange(1.0, 500.0), np.exp(np.arange(1.0, 500.0) / 50))


def _setup_date(pw):
    pw.setAxisItems({'bottom': pg.DateAxisItem()})
    pw.plot(1.7e9 + np.arange(500) * 60.0, np.random.default_rng(1).normal(size=500))


SETUPS: dict[str, Callable] = {
    'plain': lambda pw: None, 'grid': _setup_grid, 'translucent grid': _setup_translucent_grid,
    'fonts': _setup_fonts, 'labels': _setup_labels, 'log': _setup_log, 'date': _setup_date,
}


@direct_drawing
@pytest.mark.parametrize('setup', list(SETUPS))
def test_direct_drawing_equals_the_replayed_picture(setup, monkeypatch):
    pw = _plot()
    SETUPS[setup](pw)
    show_and_wait(pw)
    try:
        vb = pw.getViewBox()
        for step in range(6):
            if step % 2:
                vb.scaleBy((0.83, 0.91))
            else:
                vb.translateBy(x=vb.viewRange()[0][1] * 0.0131, y=0.37)
            process_events()
            direct = _grab(pw)
            assert all(ax.picture._direct and ax.picture._picture is None for ax in _axes(pw))
            np.testing.assert_array_equal(direct, _replayed(pw, monkeypatch))
    finally:
        pw.close()


@direct_drawing
def test_axes_draw_on_the_painted_device(monkeypatch):
    pw = _plot()
    show_and_wait(pw)
    devices = []
    original = AxisItem.drawPicture

    def drawPicture(self, p, *specs):
        devices.append(type(p.device()))
        return original(self, p, *specs)

    # replaced on the class, as a subclass would not: still drawn directly
    monkeypatch.setattr(AxisItem, 'drawPicture', drawPicture)
    try:
        with count_calls(AxisItem, 'generateDrawSpecs') as specs:
            vb = pw.getViewBox()
            for _ in range(5):
                devices.clear()
                specs.reset()
                vb.scaleBy((0.9, 0.9))
                pw.viewport().repaint()
                process_events()
                assert specs.count >= 2  # both axes changed
                # drawn on the viewport, not recorded into a QPicture
                assert len(devices) >= 2 and QtGui.QPicture not in devices
                assert all(ax.picture._picture is None for ax in _axes(pw))
    finally:
        pw.close()


@pytest.mark.parametrize('setup', ['wide axis pen', 'wide tick pen', 'non-cosmetic pen',
                                   'equal tick ends', 'overridden drawPicture'])
def test_other_drawings_are_recorded_and_replayed(setup):
    calls = []

    class RecordingAxis(AxisItem):
        def drawPicture(self, p, axisSpec, tickSpecs, textSpecs):
            calls.append(type(p.device()))
            super().drawPicture(p, axisSpec, tickSpecs, textSpecs)

    if setup == 'overridden drawPicture':
        pw = _plot(axisItems={'left': RecordingAxis('left')})
    else:
        pw = _plot()
    axis = pw.getAxis('left')
    if setup == 'wide axis pen':
        axis.setPen(pg.mkPen('c', width=3))
    elif setup == 'wide tick pen':
        axis.setTickPen(pg.mkPen('c', width=2))
    elif setup == 'non-cosmetic pen':
        pen = QtGui.QPen(QtGui.QColor('red'), 1)
        pen.setCosmetic(False)
        axis.setTickPen(pen)
    elif setup == 'equal tick ends':
        axis.setStyle(tickLength=0)
    show_and_wait(pw)
    try:
        assert axis.picture is not None
        # recorded when built, with the painter that measured the labels
        assert axis.picture._direct is False
        assert axis.picture._picture is not None
        picture = axis.picture._picture
        pw.viewport().repaint()
        process_events()
        assert axis.picture._picture is picture
        if calls:
            assert set(calls) == {QtGui.QPicture}
        assert pw.getAxis('bottom').picture._direct is _DIRECT_DRAWING
    finally:
        pw.close()


def test_picture_is_replayed_on_a_device_of_another_resolution(monkeypatch):
    pw = _plot()
    _setup_grid(pw)
    show_and_wait(pw)
    try:
        def render() -> np.ndarray:
            image = QtGui.QImage(400, 250, QtGui.QImage.Format.Format_ARGB32)
            image.fill(0xff000000)
            image.setDotsPerMeterX(11811)  # 300 dpi
            image.setDotsPerMeterY(11811)
            painter = QtGui.QPainter(image)
            pw.scene().render(painter)
            painter.end()
            return pg.functions.ndarray_from_qimage(image).copy()

        image = render()
        # a replayed picture scales the axis by the ratio of the resolutions
        assert all(ax.picture._direct is _DIRECT_DRAWING and ax.picture._picture is not None
                   for ax in _axes(pw))
        with monkeypatch.context() as patch:
            patch.setattr(axisModule, '_drawnAsReplayed', lambda specs: False)
            for ax in _axes(pw):
                ax.picture = None
            np.testing.assert_array_equal(render(), image)
    finally:
        pw.close()


def test_labels_are_drawn_with_the_font_they_were_measured_with(monkeypatch):
    pw = _plot()
    pw.setFont(QtGui.QFont('Times New Roman', 18))  # not the tick label font
    show_and_wait(pw)
    try:
        axis = pw.getAxis('bottom')
        fonts = []
        original = AxisItem.drawPicture

        def drawPicture(self, p, *specs):
            fonts.append(QtGui.QFont(p.font()))
            return original(self, p, *specs)

        monkeypatch.setattr(AxisItem, 'drawPicture', drawPicture)
        axis.picture = None
        _grab(pw)
        assert fonts
        assert all(f.family() == QtGui.QFont().family() for f in fonts)
        assert all(f.pointSizeF() == QtGui.QFont().pointSizeF() for f in fonts)
    finally:
        pw.close()
