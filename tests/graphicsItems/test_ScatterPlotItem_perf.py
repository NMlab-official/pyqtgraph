"""
Performance regression tests of ScatterPlotItem.

These tests check deterministic invariants (calls made, objects created, cache
entries) instead of durations, see ``tests/perf_helpers.py``.
"""
import numpy as np
import pytest

import pyqtgraph as pg
from pyqtgraph import functions as fn
from pyqtgraph.graphicsItems import ScatterPlotItem as ScatterPlotItemModule
from pyqtgraph.graphicsItems.ScatterPlotItem import (
    SymbolAtlas,
    _brushValueKey,
    _mkMany,
    _penValueKey,
    _quantizeSize,
    _quantizeSizes,
    _SpotArrays,
    drawSymbol,
    renderSymbol,
)
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from tests.perf_helpers import count_calls, process_events, show_and_wait

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
    show_and_wait(view)
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
    brushes, _, _ = _mkMany([(1, 2, 3), (1, 2, 3), 'r', 1, 1.0, {'color': 'b'}, None, None],
                            QtGui.QBrush, pg.mkBrush)
    assert brushes[0] is brushes[1]
    assert brushes[6] is brushes[7]
    assert brushes[3] is not brushes[4]  # intColor(1) and grey level 1.0
    assert brushes[3].color() == pg.intColor(1)
    assert brushes[4].color() == QtGui.QColor(255, 255, 255)
    assert brushes[5].color() == pg.mkColor('b')
    assert brushes[6].style() == QtCore.Qt.BrushStyle.NoBrush
    given = QtGui.QBrush(QtGui.QColor('red'))
    assert _mkMany([given], QtGui.QBrush, pg.mkBrush)[0][0] is given


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


# --------------------------------------------------------------------------------------
# T2.8: styles grouped per unique combination, numeric fast path
# --------------------------------------------------------------------------------------

_RGBA = np.array([[255, 0, 0, 255], [0, 200, 0, 255], [0, 0, 255, 128], [250, 250, 0, 255]],
                 dtype=np.uint8)


def _expectedStyles(scatter):
    """Per-spot styles computed the reference way, spot by spot."""
    step = 4 * scatter.fragmentAtlas.devicePixelRatio()
    symbol, size, pen, brush = scatter._style(['symbol', 'size', 'pen', 'brush'])
    return [(sy, _quantizeSize(sz, step), p, b) for sy, sz, p, b in zip(symbol, size, pen, brush)]


def test_unique_styles_match_per_spot_styles():
    rng = np.random.default_rng(3)
    n = 500
    brushes = [pg.mkBrush(c) for c in 'rgb'] + [None]
    pens = [pg.mkPen('w'), pg.mkPen('y', width=2), None]
    scatter = pg.ScatterPlotItem(x=rng.random(n), y=rng.random(n), hoverable=True,
                                 hoverBrush='m', hoverSize=12)
    scatter.data['brush'] = np.array(brushes, dtype=object)[rng.integers(0, 4, n)]
    scatter.data['pen'] = np.array(pens, dtype=object)[rng.integers(0, 3, n)]
    # equal symbol strings that are distinct objects, as from a numpy string array
    scatter.data['symbol'] = np.array(['o', 's', 't'])[rng.integers(0, 3, n)].astype(object)
    scatter.data['symbol'][::7] = None
    scatter.data['size'] = rng.choice([-1, 5, 6.1, 6.2, 9], n)
    scatter.data['hovered'] = rng.random(n) < 0.1

    styles, inverse = scatter._uniqueStyles(scatter.data)
    expected = _expectedStyles(scatter)
    for i in range(n):
        got = styles[inverse[i]]
        assert got[0] == expected[i][0]
        assert got[1] == expected[i][1]
        assert got[2] is expected[i][2]
        assert got[3] is expected[i][3]
    # symbols are merged by value: 3 symbols x 4 sizes x 3 pens x 4 brushes at most
    assert len(styles) == len(set(map(lambda s: (s[0], s[1], id(s[2]), id(s[3])), styles)))

    # a selection of spots gives the same result
    mask = rng.random(n) < 0.3
    styles, inverse = scatter._uniqueStyles(scatter.data, mask)
    for got, i in zip((styles[k] for k in inverse), np.flatnonzero(mask)):
        assert got[0] == expected[i][0] and got[1] == expected[i][1]
        assert got[2] is expected[i][2] and got[3] is expected[i][3]


def test_atlas_queried_once_per_unique_style():
    rng = np.random.default_rng(0)
    n = 100_000
    palette = [pg.mkBrush(c) for c in ('r', 'g', 'b', 'y', 'c')]
    lookups = []
    original = SymbolAtlas.__getitem__

    def recording(self, styles):
        lookups.append(len(styles))
        return original(self, styles)

    sizes = rng.uniform(5, 15, n)
    colors = rng.integers(0, 4, n)
    SymbolAtlas.__getitem__ = recording
    try:
        scatter = pg.ScatterPlotItem(x=rng.random(n), y=rng.random(n), size=7, pen=None,
                                     brush=[palette[i] for i in rng.integers(0, 5, n)])
        scatter.setData(x=rng.random(n), y=rng.random(n), size=sizes, pen=None,
                        brush=_RGBA[colors])
    finally:
        SymbolAtlas.__getitem__ = original
    step = 4 * scatter.fragmentAtlas.devicePixelRatio()
    combinations = set(zip(_quantizeSizes(sizes, step).tolist(), colors.tolist()))
    assert lookups == [5, len(combinations)]
    # sizes in [5, 15] quantized with `step` levels per pixel, times 4 colors
    assert len(combinations) <= 4 * (int(np.ceil((15 - 5) * step)) + 1)


def test_rgba_array_fast_path(monkeypatch):
    import pyqtgraph.graphicsItems.ScatterPlotItem as module
    rng = np.random.default_rng(0)
    n = 10_000
    idx = rng.integers(0, 4, n)
    scans = []
    original = module._groupObjects
    monkeypatch.setattr(module, '_groupObjects',
                        lambda objs: scans.append(len(objs)) or original(objs))
    scatter = pg.ScatterPlotItem(x=rng.random(n), y=rng.random(n), brush=_RGBA[idx], pen=None)
    assert scans == []  # the grouping made while converting the colors is reused
    assert len({id(b) for b in scatter.data['brush']}) == 4
    points = scatter.points()
    for i in (0, 1, 2, n - 1):
        assert points[i].brush().color() == QtGui.QColor(*_RGBA[idx[i]].tolist())
    # RGB float arrays follow the mkColor conversion rules (int(), alpha 255)
    scatter.setData(x=[0, 1], y=[0, 1], brush=np.array([[10.7, 20., 30.], [np.nan, 1, 2]]))
    assert scatter.data['brush'][0].color() == QtGui.QColor(10, 20, 30, 255)
    assert scatter.data['brush'][1].color() == QtGui.QColor(0, 1, 2, 255)
    # pens accept the same arrays
    scatter.setData(x=[0, 1], y=[0, 1], pen=_RGBA[:2])
    assert scatter.data['pen'][1].color() == QtGui.QColor(0, 200, 0, 255)
    assert scatter.data['pen'][1].widthF() == 1.0
    # two-column arrays keep their mkColor meaning, (index, hues)
    scatter.setData(x=[0, 1], y=[0, 1], brush=np.array([[1, 9], [2, 9]]))
    assert scatter.data['brush'][0].color() == pg.intColor(1, 9)


def test_rgba_rendering_identical_to_qbrush_list():
    x = np.array([1., 3., 5., 7., 9., 2.])
    y = np.array([2., 8., 5., 3., 7., 6.])
    idx = [0, 1, 2, 3, 1, 0]
    a = pg.ScatterPlotItem(x=x, y=y, size=10, pen='w', brush=_RGBA[idx])
    b = pg.ScatterPlotItem(x=x, y=y, size=10, pen='w',
                           brush=[pg.mkBrush(*_RGBA[i].tolist()) for i in idx])
    imgA = _renderInView([a])
    assert imgA == _renderInView([b])
    assert imgA != _renderInView([])


def test_new_spot_array_initial_state():
    scatter = pg.ScatterPlotItem(x=[0.], y=[0.])
    fast = scatter._newSpotArray(5)
    reference = np.empty(5, dtype=scatter.data.dtype)
    reference['size'] = -1
    reference['visible'] = True
    assert fast.dtype == reference.dtype
    for name in reference.dtype.names:
        assert fast[name].tolist() == reference[name].tolist(), name


def test_set_data_semantics_kept():
    scatter = pg.ScatterPlotItem(x=[0, 1, 2], y=[0, 1, 2], data=['a', 'b', 'c'],
                                 brush=_RGBA[:3])
    first = scatter.data
    spot = scatter.points()[1]
    scatter.setData(x=[5, 6, 7], y=[5, 6, 7])
    assert scatter.data is not first  # a new array: old SpotItems keep their data
    assert spot.pos() == pg.Point(1, 1)
    assert spot.data() == 'b'
    assert scatter.data['data'].tolist() == [None, None, None]
    assert scatter.data['brush'].tolist() == [None, None, None]
    assert scatter.points()[1].brush() == scatter.opts['brush']


def test_same_length_set_data_reuses_the_spot_array():
    n = 1000
    scatter = pg.ScatterPlotItem(x=np.arange(n), y=np.arange(n), data=np.arange(n),
                                 brush=_RGBA[np.arange(n) % 4], size=np.full(n, 5.),
                                 hoverable=True)
    scatter.data['hovered'][3] = True
    first = scatter.data
    scatter.setData(x=np.arange(n) + 1., y=np.zeros(n), hoverable=True)
    assert scatter.data is first  # sliding window: reset in place
    np.testing.assert_array_equal(scatter.data['x'], np.arange(n) + 1.)
    reference = scatter._newSpotArray(n)
    for name in ('symbol', 'size', 'pen', 'brush', 'visible', 'data', 'hovered', 'item',
                 'sourceRect'):
        if name != 'sourceRect':
            assert scatter.data[name].tolist() == reference[name].tolist(), name
    assert (scatter.data['sourceRect']['w'] > 0).all()  # symbols looked up again

    # a different length allocates a new array
    scatter.setData(x=np.arange(10.), y=np.arange(10.))
    assert scatter.data is not first


def test_spot_array_not_reused_when_shared():
    n = 50
    scatter = pg.ScatterPlotItem(x=np.arange(n), y=np.arange(n) * 2.)
    x, y = scatter.getData()
    first = scatter.data
    scatter.setData(x=np.zeros(n), y=np.zeros(n))
    assert scatter.data is not first
    np.testing.assert_array_equal(y, np.arange(n) * 2.)  # views from getData stay valid

    # an input sharing memory with the spot array
    current = scatter.data
    scatter.setData(x=current['y'], y=current['x'] + 1)
    np.testing.assert_array_equal(scatter.data['x'], np.zeros(n))
    np.testing.assert_array_equal(scatter.data['y'], np.ones(n))
    assert scatter.data is not current

    # SpotItems handed out by a hit test keep their record
    scatter = pg.ScatterPlotItem(x=np.arange(n, dtype=float), y=np.zeros(n), size=0.5,
                                 pxMode=False)
    spot, = scatter.pointsAt(QtCore.QPointF(7, 0))
    scatter.setData(x=np.arange(n) + 100., y=np.zeros(n), size=0.5, pxMode=False)
    assert spot.pos() == pg.Point(7, 0)


# --------------------------------------------------------------------------------------
# T2.9: vectorized paint preparation
# --------------------------------------------------------------------------------------

def _referenceMask(scatter, rect):
    """Spots kept by the per-spot culling of the previous implementation."""
    data = scatter.data
    l, r, t, b = rect.left(), rect.right(), rect.top(), rect.bottom()
    if scatter.opts['pxMode'] and scatter.opts['useCache']:
        w = data['sourceRect']['w'] / 2
        h = data['sourceRect']['h'] / 2
    else:
        s, = scatter._style(['size'])
        w = s / 2
        h = s / 2
    if scatter.opts['pxMode']:
        px, py = scatter._pixelLengths()
        w = w * px
        h = h * py
    return (data['visible'] & (data['x'] + w > l) & (data['x'] - w < r)
            & (data['y'] + h > t) & (data['y'] - h < b))


def _referencePoints(transform, scatter, mask):
    """Device positions of the spots as computed by the previous implementation."""
    pts = np.vstack([scatter.data['x'], scatter.data['y']])
    pts = fn.transformCoordinates(transform, pts)
    return np.clip(pts, -2 ** 30, 2 ** 30)[:, mask].T


def _referenceFragments(scatter, transform, rect, dpr):
    mask = _referenceMask(scatter, rect)
    sr = scatter.data['sourceRect'][mask]
    frags = np.empty((len(sr), 10))
    frags[:, 0:2] = _referencePoints(transform, scatter, mask)
    frags[:, 2:6] = np.frombuffer(sr.copy(), dtype=int).reshape((-1, 4))
    frags[:, 6:10] = [1 / dpr, 1 / dpr, 0.0, 1.0]
    return frags


class _ReferenceScatter(pg.ScatterPlotItem):
    """ScatterPlotItem painting spots in pixel mode as the previous implementation."""

    def paint(self, p, option, widget):
        if not self.opts['pxMode']:
            return super().paint(p, option, widget)
        if self.opts['useCache']:
            # as the real paint: re-render the atlas for the pixel ratio of the widget
            dpr = self.fragmentAtlas.devicePixelRatio()
            if widget is not None and (dprNew := widget.devicePixelRatioF()) != dpr:
                self.fragmentAtlas.setDevicePixelRatio(dprNew)
                self.fragmentAtlas.clear()
                self.data['sourceRect'] = 0
                self.updateSpots()
        mask = _referenceMask(self, self.viewRect())
        pts = _referencePoints(p.transform(), self, mask)
        p.resetTransform()
        if self.opts['useCache']:
            dpr = self.fragmentAtlas.devicePixelRatio()
            frags = _referenceFragments(self, QtGui.QTransform(), self.viewRect(), dpr)
            frags[:, 0:2] = pts
            array = pg.Qt.internals.PrimitiveArray(QtGui.QPainter.PixmapFragment, 10)
            array.resize(len(frags))
            array.ndarray()[:] = frags
            p.drawPixmapFragments(*array.drawargs(), self.fragmentAtlas.pixmap)
        else:
            p.setRenderHint(p.RenderHint.Antialiasing, self.opts['antialias'])
            styles = zip(*self._style(['symbol', 'size', 'pen', 'brush'], idx=mask))
            for pt, style in zip(pts, styles):
                p.resetTransform()
                p.translate(*pt)
                drawSymbol(p, *style)


def _paintTestKwargs(rng, n=600):
    x = rng.normal(size=n) * 3
    y = rng.normal(size=n) * 3
    x[::50] = np.nan
    y[7::61] = np.inf
    palette = [pg.mkBrush(c) for c in ('r', 'g', 'b', 'y')]
    return {
        'uniform': dict(x=x, y=y, size=9),
        'varied': dict(x=x, y=y, size=rng.integers(2, 40, n).astype(float),
                       brush=[palette[i] for i in rng.integers(0, 4, n)],
                       symbol=list(np.array(['o', 's', 't', 'star'], dtype=object)[
                           rng.integers(0, 4, n)])),
        'colors': dict(x=x, y=y, size=12, brush=_RGBA[rng.integers(0, 4, n)]),
    }


def test_fragments_identical_to_reference():
    rng = np.random.default_rng(5)
    view = pg.GraphicsView()
    vb = pg.ViewBox(enableMouse=False)
    view.setCentralItem(vb)
    view.resize(300, 200)
    view.show()
    transforms = [QtGui.QTransform.fromScale(40, -30).translate(4, -5),
                  QtGui.QTransform().rotate(30).scale(25, 25),
                  QtGui.QTransform(30, 4, 0, -3, -35, 0, 120, 90, 1)]
    rects = [QtCore.QRectF(-8, -8, 16, 16), QtCore.QRectF(-20, -20, 40, 40),
             QtCore.QRectF(-1, -0.5, 2.5, 1.5), QtCore.QRectF(0.3, 2, 1, 4)]
    for name, kwargs in _paintTestKwargs(rng).items():
        scatter = pg.ScatterPlotItem(**kwargs)
        vb.addItem(scatter)
        vb.setRange(xRange=(-5, 5), yRange=(-5, 5), padding=0)
        process_events()
        hidden = np.zeros(len(scatter.data), bool)
        for step in range(2):
            if step == 1:  # hide some spots
                hidden[::9] = True
                scatter.setPointsVisible(~hidden)
            # repeated preparations reuse the columns already in the fragment buffer
            for transform in transforms:
                for rect in rects:
                    scatter._prepareFragments(transform, rect, 1.0)
                    expected = _referenceFragments(scatter, transform, rect, 1.0)
                    got = scatter._pixmapFragments.ndarray()
                    assert got.shape == expected.shape, (name, step)
                    assert np.array_equal(got, expected), (name, step)
                    np.testing.assert_array_equal(scatter._maskAt(rect),
                                                  _referenceMask(scatter, rect))
        vb.removeItem(scatter)
    view.close()


def test_paint_identical_to_reference():
    rng = np.random.default_rng(6)
    for name, kwargs in _paintTestKwargs(rng).items():
        for options in ({}, {'useCache': False}, {'pxMode': False}):
            for xRange, yRange in (((-4, 4), (-4, 4)), ((-12, 12), (-12, 12)),
                                   ((0.5, 2), (-1, 0.2))):
                kw = dict(kwargs, **options)
                if not kw.get('pxMode', True):
                    kw['size'] = 0.2
                images = []
                for cls in (pg.ScatterPlotItem, _ReferenceScatter):
                    scatter = cls(**kw)
                    hidden = np.zeros(len(scatter.data), bool)
                    hidden[3::11] = True
                    scatter.setPointsVisible(~hidden)
                    images.append(_renderInView([scatter], xRange, yRange, size=(200, 160)))
                assert images[0] == images[1], (name, options, xRange)
                assert images[0] != _renderInView([], xRange, yRange, size=(200, 160))


def test_paint_arrays_cached_until_spots_change():
    rng = np.random.default_rng(0)
    n = 1000
    view = pg.GraphicsView()
    vb = pg.ViewBox(enableMouse=False)
    view.setCentralItem(vb)
    view.resize(200, 150)
    scatter = pg.ScatterPlotItem(x=rng.random(n), y=rng.random(n), size=5)
    vb.addItem(scatter)
    show_and_wait(view)
    with count_calls(_SpotArrays, '__init__') as builds:
        for i in range(5):  # pans: the arrays are reused
            vb.setRange(xRange=(i * 0.1, 1 + i * 0.1), yRange=(0, 1), padding=0)
            process_events()
            scatter.pointsAt(QtCore.QPointF(0.5, 0.5))
        assert builds.count == 0
        # every change of the spots drops them
        scatter.setData(x=rng.random(n), y=rng.random(n), size=5)
        process_events()
        assert builds.count == 1
        scatter.setPointsVisible(np.arange(n) % 2 == 0, update=False)
        assert not scatter._maskAt(QtCore.QRectF(-1, -1, 3, 3))[1::2].any()
        assert builds.count == 2
        scatter.points()[0].setSize(30)
        assert scatter._spotArrays().halfWidth[0] > scatter._spotArrays().halfWidth[2]
        assert builds.count == 3
    view.close()


def test_paint_all_inside_skips_culling():
    # all spots inside the view: fragments are prepared without the culling pass
    scatter = pg.ScatterPlotItem(x=[1., 2., 3.], y=[1., 2., 3.], size=5)
    arrays = scatter._spotArrays()
    assert arrays.bounds == (1., 3., 1., 3.)
    with count_calls(scatter, '_pixelLengths') as lengths:
        scatter._prepareFragments(QtGui.QTransform(), QtCore.QRectF(0, 0, 4, 4), 1.0)
        assert lengths.count == 0
        scatter._prepareFragments(QtGui.QTransform(), QtCore.QRectF(1.5, 0, 4, 4), 1.0)
        assert lengths.count == 1
    assert len(scatter._pixmapFragments) == 2


# --------------------------------------------------------------------------------------
# T4.1: opt-in DeviceCoordinateCache
# --------------------------------------------------------------------------------------

def _scatterUnderCrosshair(**kwargs):
    rng = np.random.default_rng(0)
    pw = pg.PlotWidget()
    pw.resize(400, 300)
    scatter = pg.ScatterPlotItem(x=rng.normal(size=2000), y=rng.normal(size=2000), size=7,
                                 pen=None, brush=(255, 0, 0, 120), **kwargs)
    pw.addItem(scatter)
    line = pg.InfiniteLine(angle=90)
    pw.addItem(line, ignoreBounds=True)
    return pw, scatter, line


@pytest.fixture
def integral_screens(monkeypatch):
    # the device cache is only used when every screen has an integral pixel ratio
    monkeypatch.setattr(ScatterPlotItemModule, '_screensHaveIntegralPixelRatio', lambda: True)


def test_device_cache_not_used_with_fractional_pixel_ratio(monkeypatch):
    monkeypatch.setattr(ScatterPlotItemModule, '_screensHaveIntegralPixelRatio', lambda: False)
    scatter = pg.ScatterPlotItem(x=[0], y=[0], useDeviceCache=True)
    assert scatter.opts['useDeviceCache'] is True
    assert scatter.cacheMode() == QtWidgets.QGraphicsItem.CacheMode.NoCache


def test_device_cache_off_by_default(integral_screens):
    scatter = pg.ScatterPlotItem(x=[0], y=[0])
    assert scatter.opts['useDeviceCache'] is False
    assert scatter.cacheMode() == QtWidgets.QGraphicsItem.CacheMode.NoCache
    scatter.setUseDeviceCache(True)
    assert scatter.cacheMode() == QtWidgets.QGraphicsItem.CacheMode.DeviceCoordinateCache
    scatter.setData(x=[1], y=[1], useDeviceCache=False)
    assert scatter.cacheMode() == QtWidgets.QGraphicsItem.CacheMode.NoCache


def test_device_cache_not_used_with_other_composition_modes(integral_screens):
    plus = QtGui.QPainter.CompositionMode.CompositionMode_Plus
    sourceOver = QtGui.QPainter.CompositionMode.CompositionMode_SourceOver
    scatter = pg.ScatterPlotItem(x=[0], y=[0], useDeviceCache=True, compositionMode=plus)
    assert scatter.cacheMode() == QtWidgets.QGraphicsItem.CacheMode.NoCache
    scatter.setData(x=[0], y=[0], compositionMode=sourceOver)
    assert scatter.cacheMode() == QtWidgets.QGraphicsItem.CacheMode.DeviceCoordinateCache


def _scatterRepaintsPerUpdate(pw, update, n=10, warmup=3):
    # Count the Python-level fragment preparation done by each scatter paint rather
    # than wrapping the paint virtual, which is unreliable with PySide6.
    show_and_wait(pw)
    for i in range(warmup):
        update(i)
        process_events()
    with count_calls(pg.ScatterPlotItem, '_prepareFragments') as repaints:
        for i in range(warmup, warmup + n):
            update(i)
            process_events()
    return repaints.count / n


def test_device_cache_spares_repaints_under_crosshair():
    if not ScatterPlotItemModule._screensHaveIntegralPixelRatio():
        pytest.skip("fractional device pixel ratio: the device cache is not used")
    rates = {}
    images = {}
    for cached in (False, True):
        pw, scatter, line = _scatterUnderCrosshair(useDeviceCache=cached)
        rates[cached] = _scatterRepaintsPerUpdate(pw, lambda i: line.setValue(-1 + 0.1 * i))
        line.setValue(0.0)
        process_events()
        images[cached] = pw.grab().toImage()
        pw.close()
    assert rates[False] >= 1
    assert rates[True] == 0
    assert images[True] == images[False]


def test_device_cache_repaints_on_data_change():
    pw, scatter, line = _scatterUnderCrosshair(useDeviceCache=True)
    rng = np.random.default_rng(1)

    def update(i):
        scatter.setData(x=rng.normal(size=100), y=rng.normal(size=100), size=7)

    assert _scatterRepaintsPerUpdate(pw, update, n=5) >= 1
    pw.close()

