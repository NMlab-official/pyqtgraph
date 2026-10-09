"""
Tests of the grid drawing and range updates of :class:`~pyqtgraph.AxisItem`.

- The ticks of a level are drawn with one ``drawLines`` call: the pixels are those of
  one ``drawLine`` call per tick.
- With the ``opaqueGrid`` style option (default), translucent grid lines are drawn with
  an opaque pen blended with the known opaque background, which the raster engine draws
  much faster: identical pixels over the background, translucent pens kept otherwise.
- ``setRange`` updates the label only when the automatic SI prefix changes.
"""
from __future__ import annotations

import types
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import pytest

import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter
from pyqtgraph.graphicsItems.AxisItem import AxisItem, _opaqueGridPen
from pyqtgraph.Qt import QtGui
from tests.perf_helpers import count_calls, process_events, show_and_wait

app = pg.mkQApp()


def _reference_draw_picture(self, p, axisSpec, tickSpecs, textSpecs):
    """``AxisItem.drawPicture`` drawing each tick with its own ``drawLine`` call."""
    p.setRenderHint(p.RenderHint.Antialiasing, False)
    p.setRenderHint(p.RenderHint.TextAntialiasing, True)
    pen, p1, p2 = axisSpec
    p.setPen(pen)
    p.drawLine(p1, p2)
    for pen, p1, p2 in tickSpecs:
        p.setPen(pen)
        p.drawLine(p1, p2)
    if self.style['tickFont'] is not None:
        p.setFont(self.style['tickFont'])
    p.setPen(self.textPen())
    p.setClipRect(self.boundingRect().toAlignedRect())
    for rect, flags, text in textSpecs:
        p.drawText(rect, int(flags), text)


@contextmanager
def _painter() -> Iterator[QtGui.QPainter]:
    """Painter on a QPicture, as used to build the picture of an axis."""
    picture = QtGui.QPicture()
    painter = QtGui.QPainter(picture)
    try:
        yield painter
    finally:
        painter.end()


def _repaint(widget: pg.PlotWidget) -> None:
    """Repaint the viewport of ``widget`` now."""
    widget.viewport().repaint()
    process_events()


def _axes(widget: pg.PlotWidget) -> list[AxisItem]:
    return [widget.getPlotItem().getAxis(side) for side in ('bottom', 'left')]


def _set_opaque_grid(widget: pg.PlotWidget, opaque: bool) -> None:
    for axis in _axes(widget):
        axis.setStyle(opaqueGrid=opaque)


def _grab(widget: pg.PlotWidget) -> np.ndarray:
    """
    Repaint the axes of ``widget`` and return its pixels.

    Parameters
    ----------
    widget : PlotWidget
        The widget, shown.

    Returns
    -------
    np.ndarray
        The RGB32 pixels of the widget.
    """
    for axis in _axes(widget):
        axis._invalidatePicture()
    process_events()
    image = widget.grab().toImage().convertToFormat(QtGui.QImage.Format.Format_RGB32)
    return pg.functions.ndarray_from_qimage(image).copy()


@pytest.fixture
def grid_widget():
    """PlotWidget with a fixed range and no data, shown; grid enabled by each test."""
    widget = pg.PlotWidget()
    widget.resize(600, 400)
    widget.setRange(xRange=(0, 2000), yRange=(-60, 40))
    show_and_wait(widget)
    yield widget
    widget.close()


@pytest.mark.parametrize('grid', [(True, True), (True, False), (False, False)])
def test_ticks_drawn_by_level_are_pixel_identical(grid_widget, grid):
    grid_widget.plot(np.cumsum(np.random.default_rng(0).standard_normal(500)) * 5 + 1000,
                     pen=pg.mkPen('y', width=2))
    grid_widget.showGrid(x=grid[0], y=grid[1])
    # the reference draws the pens as given: translucent grid pens
    _set_opaque_grid(grid_widget, False)
    image = _grab(grid_widget)
    for axis in _axes(grid_widget):
        axis.drawPicture = types.MethodType(_reference_draw_picture, axis)
    expected = _grab(grid_widget)
    assert np.array_equal(image, expected)


@pytest.mark.parametrize('background', [None, (0, 0, 0)])
def test_opaque_grid_is_pixel_identical_over_the_background(grid_widget, background):
    # Lines of a single grid, over the background only (they would hide the ticks of
    # another axis or the data they cross): identical pixels.
    if background is not None:
        # a ViewBox background of the color of the view background
        grid_widget.getViewBox().setBackgroundColor(background)
    grid_widget.showGrid(x=True, y=False)
    _set_opaque_grid(grid_widget, False)
    expected = _grab(grid_widget)
    axis = grid_widget.getPlotItem().getAxis('bottom')
    axis.setStyle(opaqueGrid=True)
    image = _grab(grid_widget)
    assert axis._pictureGridBackground == QtGui.QColor('black').rgba()
    assert np.array_equal(image, expected)


def test_opaque_grid_pens_are_opaque(grid_widget):
    grid_widget.showGrid(x=True, y=True)
    axis = grid_widget.getPlotItem().getAxis('bottom')
    axis.setStyle(opaqueGrid=True)
    process_events()
    background = axis._gridBackgroundRgb()
    assert background is not None
    with _painter() as painter:
        _, tickSpecs, _ = axis.generateDrawSpecs(painter)
    translucent = {id(pen): pen for pen, _, _ in tickSpecs if pen.color().alpha() < 255}
    assert translucent, 'the grid pens should be translucent'
    for pen in translucent.values():
        opaque = _opaqueGridPen(pen, background)
        assert opaque.color().alpha() == 255
        assert opaque.widthF() == pen.widthF() and opaque.isCosmetic() == pen.isCosmetic()
    opaque_pen = QtGui.QPen(QtGui.QColor(10, 20, 30))
    assert _opaqueGridPen(opaque_pen, background) is opaque_pen


def test_opaque_grid_is_on_by_default(grid_widget):
    grid_widget.showGrid(x=True, y=True)
    _repaint(grid_widget)
    for axis in _axes(grid_widget):
        assert axis.style['opaqueGrid'] is True
        assert axis._pictureGridBackground == QtGui.QColor('black').rgba()
    _set_opaque_grid(grid_widget, False)
    _repaint(grid_widget)
    for axis in _axes(grid_widget):
        assert axis._pictureGridBackground is None


@pytest.mark.parametrize('setup', ['translucent view', 'viewbox color', 'no grid'])
def test_opaque_grid_needs_a_known_opaque_background(grid_widget, setup):
    axis = grid_widget.getPlotItem().getAxis('bottom')
    axis.setStyle(opaqueGrid=True)
    if setup == 'translucent view':
        grid_widget.setBackground(None)
    elif setup == 'viewbox color':
        # differs from the view background, which the ends of the lines can reach
        grid_widget.getViewBox().setBackgroundColor((30, 40, 80))
    if setup != 'no grid':
        grid_widget.showGrid(x=True, y=True)
    process_events()
    axis._buildPicture()
    assert axis._pictureGridBackground is None


def test_opaque_grid_follows_background_changes(grid_widget):
    grid_widget.showGrid(x=True, y=True)
    axis = grid_widget.getPlotItem().getAxis('bottom')
    axis.setStyle(opaqueGrid=True)
    _repaint(grid_widget)
    assert axis._pictureGridBackground == QtGui.QColor('black').rgba()
    # the background change does not invalidate the picture; the next paint rebuilds it
    grid_widget.setBackground('w')
    _repaint(grid_widget)
    assert axis._pictureGridBackground == QtGui.QColor('white').rgba()


def test_export_keeps_translucent_grid(grid_widget):
    grid_widget.showGrid(x=True, y=True)
    plot_item = grid_widget.getPlotItem()

    def export() -> np.ndarray:
        exporter = ImageExporter(plot_item)
        exporter.parameters()['width'] = 300
        image = exporter.export(toBytes=True)
        image = image.convertToFormat(QtGui.QImage.Format.Format_ARGB32)
        return pg.functions.ndarray_from_qimage(image).copy()

    _set_opaque_grid(grid_widget, False)
    expected = export()
    _set_opaque_grid(grid_widget, True)
    _repaint(grid_widget)
    assert all(axis._pictureGridBackground is not None for axis in _axes(grid_widget))
    assert np.array_equal(export(), expected)
    # opaque pens again after the export
    _repaint(grid_widget)
    assert all(axis._pictureGridBackground is not None for axis in _axes(grid_widget))


def test_set_range_updates_label_only_when_si_prefix_changes():
    axis = AxisItem('bottom')
    axis.setLabel('Voltage', units='V')
    axis.setRange(0.0, 0.9)
    assert axis.labelUnitPrefix == 'm'
    with count_calls(axis.label, 'setHtml') as setHtml:
        axis.setRange(0.1, 0.8)
        assert setHtml.count == 0
        assert axis.labelUnitPrefix == 'm'
        axis.setRange(0.0, 2000.0)
        assert setHtml.count == 1
    assert axis.labelUnitPrefix == 'k' and axis.autoSIPrefixScale == pytest.approx(1e-3)
    assert '(kV)' in axis.labelString()


def test_set_range_without_label_never_updates_it():
    axis = AxisItem('left')
    with count_calls(axis.label, 'setHtml') as setHtml:
        for mn, mx in [(0, 1), (1e-9, 2e-9), (0, 1e9)]:
            axis.setRange(mn, mx)
    assert setHtml.count == 0
    assert axis.autoSIPrefixScale == 1.0 and axis.labelUnitPrefix == ''


def test_explicit_si_prefix_update_still_updates_label():
    axis = AxisItem('bottom')
    axis.setLabel('Voltage', units='V')
    axis.setRange(0.0, 0.9)
    with count_calls(axis.label, 'setHtml') as setHtml:
        axis.updateAutoSIPrefix()
        axis.enableAutoSIPrefix(False)
    assert setHtml.count == 2
