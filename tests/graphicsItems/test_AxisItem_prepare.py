"""
Tests of the AxisItem picture built when the scene prepares (T2.1 follow-up).

An axis measures its tick labels while building its picture, and resizes itself to fit
them. The picture is built before Qt computes the regions to repaint, so that a new size
is laid out before the paint and each update is painted once. These tests count paint
events (with an event filter, so that every binding is measured) and picture builds, and
compare what was painted with a full repaint.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from tests.perf_helpers import (
    count_calls,
    paints_per_update,
    process_events,
    show_and_wait,
)

app = pg.mkQApp()


class _PaintWatcher(QtCore.QObject):
    """
    Event filter calling a function on each paint event of the watched objects.

    Parameters
    ----------
    callback : callable
        Called without arguments when a paint event is received, before the paint.
    """

    def __init__(self, callback: Callable[[], object]) -> None:
        super().__init__()
        self._callback = callback

    def eventFilter(self, obj: QtCore.QObject, ev: QtCore.QEvent) -> bool:
        """
        Call the callback on paint events; never filter the event out.

        Parameters
        ----------
        obj : QtCore.QObject
            The object receiving the event.
        ev : QtCore.QEvent
            The event.

        Returns
        -------
        bool
            Always False.
        """
        if ev.type() == QtCore.QEvent.Type.Paint:
            self._callback()
        return False


def _visible_axes(plots: list[pg.PlotItem]) -> list[pg.AxisItem]:
    return [plot.getAxis(name) for plot in plots
            for name in ('left', 'bottom', 'right', 'top') if plot.getAxis(name).isVisible()]


def _image(qimage: QtGui.QImage) -> np.ndarray:
    qimage = qimage.convertToFormat(QtGui.QImage.Format.Format_ARGB32)
    return pg.functions.ndarray_from_qimage(qimage).copy()


def _streaming_plot() -> tuple[pg.PlotWidget, Callable[[int], None]]:
    """A plot whose left axis gets wider and narrower as the amplitude changes."""
    pw = pg.PlotWidget(size=(400, 300))
    curve = pw.plot(np.zeros(100))
    rng = np.random.default_rng(0)

    def update(i: int) -> None:
        # the amplitude cycles over 6 decades: tick labels of different widths
        curve.setData(rng.normal(size=100) * 10.0 ** ((i % 7) - 3))

    return pw, update


def _linked_plots(ncols: int) -> tuple[pg.GraphicsLayoutWidget, list[pg.PlotItem],
                                       Callable[[int], None]]:
    """Six x-linked plots on ``ncols`` columns, with amplitudes changing differently."""
    win = pg.GraphicsLayoutWidget(size=(700, 600))
    plots = []
    for k in range(6):
        axisItems = {'bottom': pg.DateAxisItem()} if k == 5 else None
        plot = win.addPlot(row=k // ncols, col=k % ncols, axisItems=axisItems)
        if plots:
            plot.setXLink(plots[0])
        plots.append(plot)
    plots[0].setLabel('left', 'Voltage', units='V')
    plots[1].showAxis('right')
    plots[2].showGrid(x=True, y=True, alpha=0.5)
    rng = np.random.default_rng(1)
    curves = [plot.plot(pen=k) for k, plot in enumerate(plots)]

    def update(i: int) -> None:
        x = 1.7e9 + 60.0 * (np.arange(200) + 10 * i)
        for k, curve in enumerate(curves):
            amplitude = (1 + i) ** (1 + k % 3) * 10.0 ** (k % 4 - 2) * (-1) ** (i // 7)
            curve.setData(x, rng.normal(size=200).cumsum() * amplitude)

    return win, plots, update


def test_changing_axis_width_paints_once():
    pw, update = _streaming_plot()
    left = pw.getPlotItem().getAxis('left')
    widths = set()

    def update_and_record(i: int) -> None:
        update(i)
        widths.add(left.width())

    try:
        assert paints_per_update(pw, None, update_and_record) <= 1.05
        assert len(widths) >= 3  # the left axis did change its width
    finally:
        pw.close()


@pytest.mark.parametrize('ncols', [1, 2])
def test_linked_plots_with_changing_axis_widths_paint_once(ncols):
    win, _, update = _linked_plots(ncols)
    try:
        assert paints_per_update(win, None, update, n=30) <= 1.05
    finally:
        win.close()


def test_axis_pictures_are_built_before_the_paint():
    win, plots, update = _linked_plots(2)
    axes = _visible_axes(plots)
    missing = []
    watcher = _PaintWatcher(lambda: missing.extend(ax for ax in axes if ax.picture is None))
    show_and_wait(win)
    win.viewport().installEventFilter(watcher)
    try:
        for i in range(20):
            update(i)
            process_events()
        assert missing == []
        # nothing stays connected to the scene once the pictures are built
        assert all(ax._preparingScene is None for ax in axes)
    finally:
        win.viewport().removeEventFilter(watcher)
        win.close()


def test_axis_picture_is_built_at_most_once_per_update():
    pw, update = _streaming_plot()
    plot = pw.getPlotItem()
    left, bottom = plot.getAxis('left'), plot.getAxis('bottom')
    show_and_wait(pw)
    changes = 0
    try:
        with count_calls(left, 'drawPicture') as draws, \
                count_calls(left, 'generateDrawSpecs') as specs, \
                count_calls(bottom, 'generateDrawSpecs') as bottomSpecs:
            for i in range(30):
                draws.reset()
                specs.reset()
                bottomSpecs.reset()
                width = left.width()
                update(i)
                process_events()
                assert draws.count == 1
                if left.width() == width:
                    # the range of the bottom axis is fixed: nothing to rebuild
                    assert (specs.count, bottomSpecs.count) == (1, 0)
                else:
                    # the labels measured for the previous width are not drawn; the
                    # specifications of the bottom axis follow its new length
                    changes += 1
                    assert specs.count <= 2
                    assert bottomSpecs.count <= 2
        assert changes >= 3
    finally:
        pw.close()


@pytest.mark.skipif(
    QtWidgets.QApplication.platformName() != 'offscreen',
    reason="reads back the backing store with QScreen.grabWindow",
)
@pytest.mark.parametrize('layout', ['plot', 'grid'])
def test_rendering_equals_a_full_repaint(layout):
    # What was painted once per update equals a repaint whose axis pictures are built
    # from scratch while painting, as they were before.
    if layout == 'plot':
        win, update = _streaming_plot()
        plots = [win.getPlotItem()]
    else:
        win, plots, update = _linked_plots(2)
    axes = _visible_axes(plots)
    show_and_wait(win)
    try:
        for i in range(10):
            update(i)
            process_events()
            painted = _image(win.screen().grabWindow(win.winId()).toImage())
            sizes = [ax.size() for ax in axes]
            for ax in axes:
                ax.picture = None
            full = _image(win.grab().toImage())
            assert all(ax.picture is not None for ax in axes)
            np.testing.assert_array_equal(painted, full)
            process_events()
            assert [ax.size() for ax in axes] == sizes  # the layout was complete
    finally:
        win.close()


def _render(scene: QtWidgets.QGraphicsScene, size: tuple[int, int]) -> np.ndarray:
    image = QtGui.QImage(*size, QtGui.QImage.Format.Format_ARGB32)
    image.fill(0)
    painter = QtGui.QPainter(image)
    try:
        scene.render(painter, QtCore.QRectF(image.rect()), QtCore.QRectF(image.rect()))
    finally:
        painter.end()
    return _image(image)


def _standalone_axis(scene: QtWidgets.QGraphicsScene) -> pg.AxisItem:
    axis = pg.AxisItem('left')
    axis.setWidth(60)  # whatever the size of the labels in the platform font
    scene.addItem(axis)
    axis.setGeometry(QtCore.QRectF(0, 0, 60, 200))
    axis.setRange(-1234.5, 6789.0)
    return axis


def test_axis_outside_a_graphics_scene_builds_its_picture_in_paint():
    # The axis measures itself in device pixels: it needs a view, never shown here.
    plain = QtWidgets.QGraphicsScene()
    plainView = QtWidgets.QGraphicsView(plain)
    axis = _standalone_axis(plain)
    assert axis._preparingScene is None
    assert axis.picture is None
    plainImage = _render(plain, (80, 220))
    assert axis.picture is not None
    assert plainImage[..., 3].any()  # ticks and labels were drawn

    # A GraphicsScene without a visible view builds the picture when it prepares,
    # e.g. in render(); the picture is the same.
    scene = pg.GraphicsScene()
    view = QtWidgets.QGraphicsView(scene)
    prepared = _standalone_axis(scene)
    assert prepared._preparingScene() is scene
    process_events()
    assert prepared.picture is None  # hidden view: the prepare is deferred
    with count_calls(prepared, '_buildPicture') as builds:
        scene.prepareForPaint()
        assert prepared.picture is not None
        assert prepared._preparingScene is None
        np.testing.assert_array_equal(_render(scene, (80, 220)), plainImage)
    assert builds.count == 1
    del plainView, view


def test_picture_drawn_for_a_size_about_to_change_is_built_again():
    scene = pg.GraphicsScene()
    view = QtWidgets.QGraphicsView(scene)
    axis = pg.AxisItem('left')
    scene.addItem(axis)
    axis.setGeometry(QtCore.QRectF(0, 0, 40, 200))
    # labels wider than the space reserved by default, in any font
    axis.setTicks([[(v, f'tick label number {v}') for v in range(5)]])
    axis.setRange(0, 4)
    width = axis.maximumWidth()
    with count_calls(axis, 'generateDrawSpecs') as specs, \
            count_calls(axis, 'drawPicture') as draws:
        scene.prepareForPaint()
        # measured, not drawn: the axis needs a new width, then a new picture
        assert axis.maximumWidth() > width
        assert (specs.count, draws.count) == (1, 0)
        assert axis.picture is None
        assert axis._preparingScene() is scene
        assert scene._prepareRequested
        scene.prepareForPaint()
        assert (specs.count, draws.count) == (2, 1)
        assert axis.picture is not None
        assert axis._preparingScene is None
    assert axis.width() == axis.maximumWidth()  # drawn for its new width
    del view


def test_hidden_axes_are_not_prepared():
    pw = pg.PlotWidget(size=(300, 200))
    curve = pw.plot(np.arange(10.0))
    plot = pw.getPlotItem()
    show_and_wait(pw)
    top, right = plot.getAxis('top'), plot.getAxis('right')
    try:
        with count_calls(top, 'generateDrawSpecs') as topSpecs, \
                count_calls(right, 'generateDrawSpecs') as rightSpecs:
            curve.setData(np.arange(10.0) * 100)
            # the range of the hidden axes changed, nothing is pending for them
            assert top._preparingScene is None and right._preparingScene is None
            process_events()
        assert (topSpecs.count, rightSpecs.count) == (0, 0)
    finally:
        pw.close()


def test_axis_moved_to_another_scene_is_not_prepared_by_the_first():
    first, second = pg.GraphicsScene(), pg.GraphicsScene()
    axis = _standalone_axis(first)
    assert axis._preparingScene() is first
    first.removeItem(axis)
    second.addItem(axis)
    first.prepareForPaint()
    assert axis.picture is None
    assert axis._preparingScene is None  # disconnected from the first scene
    axis.setRange(0, 1)
    assert axis._preparingScene() is second
    second.prepareForPaint()
    assert axis.picture is not None


def test_subclass_overriding_paint_is_not_prepared():
    class PaintedAxis(pg.AxisItem):
        def paint(self, p, opt, widget):
            p.drawLine(0, 0, 10, 10)

    scene = pg.GraphicsScene()
    axis = PaintedAxis('left')
    scene.addItem(axis)
    axis.setRange(0, 10)
    with count_calls(axis, 'generateDrawSpecs') as specs:
        scene.prepareForPaint()
    assert axis._preparingScene is None
    assert specs.count == 0
