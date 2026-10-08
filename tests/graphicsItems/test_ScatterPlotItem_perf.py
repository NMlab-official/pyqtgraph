"""
Performance regression tests of ScatterPlotItem.

These tests check deterministic invariants (calls made, objects created, cache
entries) instead of durations, see ``tests/perf_helpers.py``.
"""
import numpy as np

import pyqtgraph as pg
from pyqtgraph.graphicsItems.ScatterPlotItem import (
    SymbolAtlas,
    _brushValueKey,
    _mkMany,
    _penValueKey,
    _quantizeSize,
    renderSymbol,
)
from pyqtgraph.Qt import QtCore, QtGui
from tests.perf_helpers import process_events

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
# T1.3 follow-up: the view is told when spot sizes change the data bounds
# --------------------------------------------------------------------------------------

def test_size_change_updates_auto_range():
    pw = pg.PlotWidget()
    pw.resize(300, 200)
    pw.show()
    scatter = pg.ScatterPlotItem(x=[0., 1.], y=[0., 1.], size=0.1, pxMode=False)
    pw.addItem(scatter)
    for _ in range(5):
        app.processEvents()
    (x0, x1), _ = pw.getViewBox().viewRange()
    scatter.setSize(50)  # pads the data bounds by 0.7 * 50 in data units
    for _ in range(5):
        app.processEvents()
    (x0b, x1b), _ = pw.getViewBox().viewRange()
    assert x0b < x0 - 30 and x1b > x1 + 30
    pw.close()


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


# --------------------------------------------------------------------------------------
# T2.7: symbol atlas keyed by value, quantized sizes, memoized mkBrush/mkPen
# --------------------------------------------------------------------------------------

def _renderInView(items, xRange=(0, 10), yRange=(0, 10), size=(160, 120)):
    """Render items in a bare ViewBox and return the widget image."""
    view = pg.GraphicsView()
    vb = pg.ViewBox(enableMouse=False)
    view.setCentralItem(vb)
    view.resize(*size)
    for item in items:
        vb.addItem(item)
    vb.setRange(xRange=xRange, yRange=yRange, padding=0)
    view.show()
    process_events()
    img = view.grab().toImage()
    view.close()
    return img


def _scatterWithDpr1(**kwargs):
    scatter = pg.ScatterPlotItem()
    scatter.fragmentAtlas.setDevicePixelRatio(1.0)
    scatter.setData(**kwargs)
    return scatter


def test_tuple_colors_share_brushes_and_atlas_entries():
    rng = np.random.default_rng(0)
    n = 10_000
    colors = [(255, 0, 0, 200), (0, 255, 0, 200), (0, 0, 255, 200)]
    scatter = _scatterWithDpr1(x=rng.random(n), y=rng.random(n), size=7, pen=None,
                               brush=[colors[i] for i in rng.integers(0, 3, n)])
    assert len(scatter.fragmentAtlas) <= 3
    assert len({id(b) for b in scatter.data['brush']}) == 3
    # equal brushes given as distinct objects also share one entry
    scatter.setData(x=np.arange(3.), y=np.arange(3.), size=7, pen=None,
                    brush=[pg.mkBrush('r'), pg.mkBrush('r'), pg.mkBrush('r')])
    sr = scatter.data['sourceRect']
    assert len(set(zip(sr['x'].tolist(), sr['y'].tolist()))) == 1


def test_mk_many_memoizes_by_type_and_value():
    brushes = _mkMany([(1, 2, 3), (1, 2, 3), 'r', 1, 1.0, {'color': 'b'}, None, None],
                      QtGui.QBrush, pg.mkBrush)
    assert brushes[0] is brushes[1]
    assert brushes[6] is brushes[7]
    assert brushes[3] is not brushes[4]  # intColor(1) and grey level 1.0
    assert brushes[3].color() == pg.intColor(1)
    assert brushes[4].color() == QtGui.QColor(255, 255, 255)
    assert brushes[5].color() == pg.mkColor('b')
    assert brushes[6].style() == QtCore.Qt.BrushStyle.NoBrush
    given = QtGui.QBrush(QtGui.QColor('red'))
    assert _mkMany([given], QtGui.QBrush, pg.mkBrush)[0] is given


def test_repeated_set_data_with_new_pens_keeps_one_atlas_entry():
    scatter = _scatterWithDpr1(x=[1, 2], y=[1, 2])
    scatter.fragmentAtlas.clear()
    for _ in range(5):
        # pen=None makes a new QPen object at every call
        scatter.setData(x=[1, 2], y=[1, 2], pen=None, brush=(10, 20, 30))
    assert len(scatter.fragmentAtlas) == 1


def test_float_sizes_are_quantized():
    rng = np.random.default_rng(1)
    n = 10_000
    scatter = _scatterWithDpr1(x=rng.random(n), y=rng.random(n), size=rng.uniform(5, 15, n),
                               pen=None)
    assert len(scatter.fragmentAtlas) <= 41
    sizes = np.array([key[1] for key in scatter.fragmentAtlas._coords])
    np.testing.assert_array_equal(sizes * 4, np.round(sizes * 4))
    # sizes are rounded to the nearest quarter pixel
    assert _quantizeSize(7.1, 4) == 7.0
    assert _quantizeSize(7.2, 4) == 7.25
    assert _quantizeSize(7.0, 4 * 1.25) == 7.0  # integral sizes are kept as they are


def test_integral_sizes_render_exactly():
    pen = pg.mkPen('w', width=1.5)
    brush = pg.mkBrush(200, 100, 50)
    scatter = _scatterWithDpr1(x=[0, 1, 2], y=[0, 0, 0], size=[3, 8, 13], pen=pen,
                               brush=brush, symbol='t')
    atlas = scatter.fragmentAtlas
    for size in (3, 8, 13):
        key = ('t', float(size), _penValueKey(pen), _brushValueKey(brush))
        y, x, h, w = atlas._coords[key]
        img = renderSymbol('t', size, pen, brush)  # keep a reference: the array is a view
        expected = pg.functions.ndarray_from_qimage(img)
        np.testing.assert_array_equal(atlas._data[x:x + w, y:y + h], expected)


def test_atlas_keys_distinguish_rendering_differences():
    atlas = SymbolAtlas()
    gradient = QtGui.QLinearGradient(0, 0, 1, 1)
    gradBrush1 = QtGui.QBrush(gradient)
    gradBrush2 = QtGui.QBrush(gradient)
    miter = pg.mkPen('w', width=3)
    miter.setJoinStyle(QtCore.Qt.PenJoinStyle.MiterJoin)
    rounded = pg.mkPen('w', width=3)
    rounded.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
    nonCosmetic = pg.mkPen('w', width=3, cosmetic=False)
    brush = pg.mkBrush('r')
    path = QtGui.QPainterPath()
    path.addRect(QtCore.QRectF(-0.5, -0.5, 1, 1))
    styles = [
        ('s', 10, miter, gradBrush1),
        ('s', 10, miter, gradBrush2),  # gradient brushes are keyed by identity
        ('s', 10, rounded, brush),
        ('s', 10, miter, brush),
        ('s', 10, nonCosmetic, brush),
        (path, 10, miter, brush),
        (0, 10, miter, brush),
        ('s', 10, QtGui.QPen(miter), pg.mkBrush('r')),  # same as styles[3]
    ]
    coords = atlas[styles]
    assert len(atlas) == 7
    assert coords[7] == coords[3]


def test_atlas_size_is_capped():
    scatter = _scatterWithDpr1(x=[0], y=[0])
    scatter._atlasMaxEntries = 50
    n = 2000
    sizes = []
    for i in range(10):
        # 20 new colors at every update, as with colors following a live value
        colors = [(i, j, 0) for j in range(20)]
        scatter.setData(x=np.arange(n), y=np.arange(n), pen=None,
                        brush=[colors[k % 20] for k in range(n)])
        sizes.append(len(scatter.fragmentAtlas))
    assert max(sizes) <= 61
    assert scatter._atlasEntriesInUse() == 20


def test_tuple_and_qbrush_rendering_identical():
    colors = [(255, 0, 0), (0, 200, 0), (0, 0, 255, 128)]
    x = np.array([1., 3., 5., 7., 9.])
    y = np.array([2., 8., 5., 3., 7.])
    idx = [0, 1, 2, 1, 0]
    a = pg.ScatterPlotItem(x=x, y=y, size=[6, 9, 12, 9, 6], pen='w',
                           brush=[colors[i] for i in idx])
    b = pg.ScatterPlotItem(x=x, y=y, size=[6, 9, 12, 9, 6], pen=QtGui.QPen(pg.mkPen('w')),
                           brush=[QtGui.QBrush(pg.mkColor(colors[i])) for i in idx])
    imgA = _renderInView([a])
    imgB = _renderInView([b])
    assert imgA == imgB
    assert imgA != _renderInView([])  # something was drawn
