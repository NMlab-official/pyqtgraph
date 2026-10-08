"""
Performance regression tests for :class:`~pyqtgraph.LegendItem` (T1.9).

Adding an entry used to call ``updateSize`` synchronously, which walks every entry and
relays out the whole grid, so that adding ``n`` entries was quadratic. The size update is
now coalesced per batch. These tests check the number of ``updateSize`` calls and that
the final size and layout are those of the eager update.
"""
from __future__ import annotations

import pytest

import pyqtgraph as pg
from pyqtgraph.exporters import SVGExporter
from pyqtgraph.graphicsItems.LegendItem import LegendItem
from pyqtgraph.Qt import QtGui
from tests.perf_helpers import count_calls, process_events

app = pg.mkQApp()

N_ENTRIES = 50


def _names(n: int) -> list[str]:
    """
    Entry names of varying length, so that column widths differ.

    Parameters
    ----------
    n : int
        Number of names.

    Returns
    -------
    list of str
        The names.
    """
    return [f'curve {k} ' + 'x' * ((7 * k) % 13) for k in range(n)]


def _legend_state(legend: LegendItem) -> tuple:
    """
    Size of the legend and geometry of all its entries.

    Parameters
    ----------
    legend : LegendItem
        The legend to inspect.

    Returns
    -------
    tuple
        ``(width, height, entry_geometries)``.
    """
    geometries = tuple((sample.geometry().getRect(), label.geometry().getRect())
                       for sample, label in legend.items)
    return legend.width(), legend.height(), geometries


def _build(col_count: int, eager: bool) -> tuple[pg.PlotWidget, LegendItem]:
    """
    Plot ``N_ENTRIES`` named curves with a legend.

    Parameters
    ----------
    col_count : int
        Number of legend columns.
    eager : bool
        Call ``updateSize`` after each entry, as the legend did before T1.9.

    Returns
    -------
    widget : PlotWidget
        The plot widget, shown.
    legend : LegendItem
        Its legend.
    """
    widget = pg.PlotWidget()
    widget.resize(600, 400)
    widget.show()
    process_events()
    legend = widget.addLegend(colCount=col_count)
    for name in _names(N_ENTRIES):
        widget.plot([0, 1, 2], name=name)
        if eager:
            legend.updateSize()
    process_events()
    return widget, legend


def test_adding_entries_coalesces_size_updates():
    widget = pg.PlotWidget()
    widget.show()
    process_events()
    legend = widget.addLegend()
    with count_calls(LegendItem, 'updateSize') as calls:
        for name in _names(N_ENTRIES):
            widget.plot([0, 1, 2], name=name)
        assert calls.count == 0
        process_events()
        assert calls.count == 1
        process_events()
        assert calls.count == 1
    assert len(legend.items) == N_ENTRIES
    widget.close()


def test_removing_entries_coalesces_size_updates():
    widget, legend = _build(1, eager=False)
    curves = [sample.item for sample, _ in legend.items]
    with count_calls(LegendItem, 'updateSize') as calls:
        for curve in curves[:20]:
            widget.removeItem(curve)
        assert calls.count == 0
        process_events()
        assert calls.count == 1
    state = _legend_state(legend)
    legend.updateSize()
    process_events()
    assert _legend_state(legend) == state
    assert len(legend.items) == N_ENTRIES - 20
    widget.close()


@pytest.mark.parametrize('col_count', [1, 3])
def test_coalesced_size_matches_eager_size(col_count: int):
    eager_widget, eager_legend = _build(col_count, eager=True)
    widget, legend = _build(col_count, eager=False)
    assert legend.height() > 0
    assert _legend_state(legend) == _legend_state(eager_legend)
    eager_widget.close()
    widget.close()


def test_size_updated_before_render_without_event_loop():
    widget = pg.PlotWidget()
    widget.resize(600, 400)
    legend = widget.addLegend()
    for name in _names(10):
        widget.plot([0, 1, 2], name=name)
    image = QtGui.QImage(600, 400, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
    with count_calls(LegendItem, 'updateSize') as calls:
        painter = QtGui.QPainter(image)
        try:
            widget.scene().render(painter)  # emits sigPrepareForPaint, as exporters do
        finally:
            painter.end()
        assert calls.count == 1
        state = _legend_state(legend)
        assert state[1] > 0
        process_events()
        assert calls.count == 1  # the pending timer was cancelled
    legend.updateSize()
    assert _legend_state(legend) == state
    widget.close()


def test_size_updated_before_svg_export_without_event_loop():
    widget = pg.PlotWidget()
    widget.resize(600, 400)
    widget.setRange(xRange=(0, 2), yRange=(0, 2), padding=0)
    legend = widget.addLegend()
    for name in _names(10):
        widget.plot([0, 1, 2], name=name)
    with count_calls(LegendItem, 'updateSize') as calls:
        SVGExporter(widget.plotItem).export(toBytes=True)
        assert calls.count == 1
        state = _legend_state(legend)
        assert state[1] > 0
    legend.updateSize()
    assert _legend_state(legend) == state
    widget.close()


def test_explicit_update_size_is_immediate():
    legend = pg.LegendItem()
    legend.addItem(pg.PlotDataItem(), name='first entry')
    assert legend.height() == 0
    with count_calls(LegendItem, 'updateSize') as calls:
        legend.updateSize()
        assert legend.height() > 0
        process_events()
        assert calls.count == 1

