"""
Performance regression tests of ScatterPlotItem.

These tests check deterministic invariants (calls made, objects created, cache
entries) instead of durations, see ``tests/perf_helpers.py``.
"""
import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

app = pg.mkQApp()


class _CountingBrush(QtGui.QBrush):
    """QBrush counting the calls of its ``__eq__`` method."""

    eqCalls = 0

    def __eq__(self, other):
        type(self).eqCalls += 1
        return super().__eq__(other)

    __hash__ = None


class _CountingPen(QtGui.QPen):
    """QPen counting the calls of its ``__eq__`` method."""

    eqCalls = 0

    def __eq__(self, other):
        type(self).eqCalls += 1
        return super().__eq__(other)

    __hash__ = None


# --------------------------------------------------------------------------------------
# T1.4: identity-based None test in _style
# --------------------------------------------------------------------------------------

def test_style_default_substitution_does_not_compare_qt_objects():
    n = 300
    brushes = [_CountingBrush(QtGui.QColor(c)) for c in ('red', 'green', 'blue')]
    pens = [_CountingPen(QtGui.QColor(c)) for c in ('cyan', 'magenta')]
    brushCol = [None if i % 4 == 0 else brushes[i % 3] for i in range(n)]
    penCol = [None if i % 5 == 0 else pens[i % 2] for i in range(n)]
    symbolCol = [None if i % 3 == 0 else 's' for i in range(n)]

    scatter = pg.ScatterPlotItem(x=np.arange(n), y=np.arange(n))
    # None marks an unset entry, which takes the item default
    scatter.data['brush'] = np.array(brushCol, dtype=object)
    scatter.data['pen'] = np.array(penCol, dtype=object)
    scatter.data['symbol'] = np.array(symbolCol, dtype=object)
    _CountingBrush.eqCalls = 0
    _CountingPen.eqCalls = 0
    symbol, pen, brush = scatter._style(['symbol', 'pen', 'brush'])
    assert _CountingBrush.eqCalls == 0
    assert _CountingPen.eqCalls == 0

    defaultBrush = scatter.opts['brush']
    defaultPen = scatter.opts['pen']
    for i in range(n):
        assert brush[i] is (defaultBrush if brushCol[i] is None else brushCol[i])
        assert pen[i] is (defaultPen if penCol[i] is None else penCol[i])
        assert symbol[i] == ('o' if symbolCol[i] is None else 's')


def test_style_size_default_and_hover():
    scatter = pg.ScatterPlotItem(x=np.arange(4.), y=np.arange(4.), size=[3, 4, 5, 6],
                                 hoverable=True, hoverSize=20, hoverBrush='r')
    scatter.data['size'][1] = -1
    scatter.data['hovered'][2] = True
    size, brush = scatter._style(['size', 'brush'])
    np.testing.assert_array_equal(size, [3, 7, 20, 6])
    assert brush[2] is scatter.opts['hoverBrush']
    assert brush[0] is scatter.opts['brush']
    # the stored data is not modified by _style
    np.testing.assert_array_equal(scatter.data['size'], [3, -1, 5, 6])


# --------------------------------------------------------------------------------------
# T1.5: SpotItems created only for the points hit
# --------------------------------------------------------------------------------------

class _HoverEvent:
    """Minimal stand-in of a HoverEvent for ScatterPlotItem.hoverEvent."""

    def __init__(self, pos, exit=False):
        self._pos = pos
        self.exit = exit

    def pos(self):
        return self._pos


def _numberOfSpotItems(scatter):
    return sum(item is not None for item in scatter.data['item'])


def _lineScatter(n=100_000, **kwargs):
    # spots of size 2.5 in data units on a line: a point query hits 3 spots
    return pg.ScatterPlotItem(x=np.arange(n, dtype=float), y=np.zeros(n), size=2.5,
                              pxMode=False, **kwargs)


def test_points_at_creates_spot_items_for_hits_only():
    scatter = _lineScatter()
    assert _numberOfSpotItems(scatter) == 0
    pts = scatter.pointsAt(QtCore.QPointF(50, 0))
    assert [pt.index() for pt in pts] == [51, 50, 49]  # reversed order kept
    assert _numberOfSpotItems(scatter) == 3
    assert all(isinstance(pt, pg.SpotItem) for pt in pts)
    assert pts[0].pos() == pg.Point(51, 0)

    # the SpotItems are reused, and points() still creates all of them
    again = scatter.pointsAt(QtCore.QPointF(50, 0))
    assert all(a is b for a, b in zip(pts, again))
    allPoints = scatter.points()
    assert len(allPoints) == len(scatter.data)
    assert _numberOfSpotItems(scatter) == len(scatter.data)
    assert allPoints[50] is pts[1]

    # setData drops the SpotItems
    scatter.setData(x=np.arange(10.), y=np.zeros(10), size=2.5, pxMode=False)
    assert _numberOfSpotItems(scatter) == 0
    assert len(scatter.pointsAt(QtCore.QPointF(100, 0))) == 0
    assert _numberOfSpotItems(scatter) == 0


def test_hover_creates_spot_items_for_hits_only():
    scatter = _lineScatter(hoverable=True, hoverBrush='r', tip=None)
    hovered = []
    scatter.sigHovered.connect(lambda item, points, ev: hovered.append(points))
    scatter.hoverEvent(_HoverEvent(QtCore.QPointF(20, 0)))
    assert [pt.index() for pt in hovered[-1]] == [21, 20, 19]
    assert _numberOfSpotItems(scatter) == 3
    assert scatter.data['hovered'].sum() == 3

    scatter.hoverEvent(_HoverEvent(QtCore.QPointF(20, 0), exit=True))
    assert len(hovered[-1]) == 0
    assert scatter.data['hovered'].sum() == 0
    assert _numberOfSpotItems(scatter) == 3
