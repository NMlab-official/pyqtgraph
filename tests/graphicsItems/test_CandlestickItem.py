"""
Tests of CandlestickItem: bounds, view culling, OHLC aggregation, appending and
rendering against a straightforward per-candle painter.

Counts of drawn primitives and pixel comparisons are asserted, never timings.
"""
import math
from collections.abc import Callable

import numpy as np
import pytest

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from tests.perf_helpers import show_and_wait

app = pg.mkQApp()

# fields are (time, open, close, min, max), as in examples/customGraphicsItem.py
SIX = np.array([
    (1., 10, 13, 5, 15),
    (2., 13, 17, 9, 20),
    (3., 17, 14, 11, 23),
    (4., 14, 15, 5, 19),
    (5., 15, 9, 8, 22),
    (6., 9, 15, 8, 16),
])


def sixCandles() -> dict:
    """
    The six candles of the custom graphics item example.

    Returns
    -------
    dict
        ``x``, ``open``, ``high``, ``low`` and ``close`` arrays.
    """
    t, o, c, lo, hi = SIX.T
    return dict(x=t, open=o, high=hi, low=lo, close=c)


def randomCandles(n: int, seed: int = 0) -> dict:
    """
    Random walk candles at integer x positions.

    Parameters
    ----------
    n : int
        Number of candles.
    seed : int, default 0
        Random seed.

    Returns
    -------
    dict
        ``x``, ``open``, ``high``, ``low`` and ``close`` arrays.
    """
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.standard_normal(n))
    open_ = np.r_[close[0], close[:-1]] + rng.standard_normal(n) * 0.2
    high = np.maximum(open_, close) + rng.random(n)
    low = np.minimum(open_, close) - rng.random(n)
    return dict(x=np.arange(n, dtype=float), open=open_, high=high, low=low, close=close)


class RecordingPainter(QtGui.QPainter):
    """
    QPainter counting the rectangles and lines drawn by ``drawRects``/``drawLines``.

    Parameters
    ----------
    device : QtGui.QPaintDevice
        Device to paint on.
    """

    def __init__(self, device: QtGui.QPaintDevice) -> None:
        super().__init__(device)
        self.rects = 0
        self.lines = 0

    @staticmethod
    def _count(args: tuple) -> int:
        if len(args) == 2 and isinstance(args[1], int):
            return args[1]  # PySide: pointer to the first primitive, count
        return len(args[0])

    def drawRects(self, *args) -> None:
        self.rects += self._count(args)
        super().drawRects(*args)

    def drawLines(self, *args) -> None:
        self.lines += self._count(args)
        super().drawLines(*args)


def paint(item: QtWidgets.QGraphicsItem | None, rect: QtCore.QRectF,
          size: tuple[int, int] = (400, 300), antialias: bool = False,
          painter: Callable[[QtGui.QPainter], None] | None = None
          ) -> tuple[np.ndarray, RecordingPainter]:
    """
    Paint ``item`` (or call ``painter(p)``) into an image showing ``rect``.

    Parameters
    ----------
    item : QGraphicsItem or None
        Item whose ``paint`` method is called, unless ``painter`` is given.
    rect : QtCore.QRectF
        Item-coordinate rectangle mapped onto the whole image.
    size : tuple of int, default (400, 300)
        Image size.
    antialias : bool, default False
        Antialiasing render hint.
    painter : callable, optional
        Function drawing with the painter instead of ``item``.

    Returns
    -------
    image : numpy.ndarray
        Rendered pixels, shape (height, width, 4).
    recorder : RecordingPainter
        The painter used, with its counts.
    """
    img = QtGui.QImage(size[0], size[1], QtGui.QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(0)
    p = RecordingPainter(img)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, antialias)
    p.scale(size[0] / rect.width(), size[1] / rect.height())
    p.translate(-rect.left(), -rect.top())
    try:
        if painter is None:
            item.paint(p, QtWidgets.QStyleOptionGraphicsItem(), None)
        else:
            painter(p)
    finally:
        p.end()
    return pg.functions.ndarray_from_qimage(img).copy(), p


def referencePainter(item: pg.CandlestickItem) -> Callable[[QtGui.QPainter], None]:
    """
    Straightforward painter drawing each candle in turn: wick, then body.

    Parameters
    ----------
    item : CandlestickItem
        Item whose candles and styles are drawn.

    Returns
    -------
    callable
        Function drawing the candles with a given painter.
    """
    x, o, h, l, c = item.ohlc
    w = item.width

    def draw(p: QtGui.QPainter) -> None:
        for i in range(len(x)):
            up = c[i] >= o[i]
            pen = item.upPen if up else item.downPen
            p.setPen(pen if item.wickPen is None else item.wickPen)
            p.drawLine(QtCore.QLineF(x[i], l[i], x[i], h[i]))
            p.setPen(pen)
            p.setBrush(item.upBrush if up else item.downBrush)
            bottom, top = min(o[i], c[i]), max(o[i], c[i])
            p.drawRect(QtCore.QRectF(x[i] - w / 2, bottom, w, top - bottom))
    return draw


def levelArrays(level) -> dict:
    """
    Aggregated candles of a level, recovered from its geometry.

    Parameters
    ----------
    level : _Level
        Level of detail of a CandlestickItem.

    Returns
    -------
    dict
        For ``'up'`` and ``'down'``: array of rows ``(center, half width, bottom,
        top, low, high)``.
    """
    result = {}
    for name, side in (('up', level.up), ('down', level.down)):
        bodies = side.bodies.ndarray()
        wicks = side.wicks.ndarray()
        result[name] = np.column_stack([
            side.x.view, bodies[:, 2] / 2, bodies[:, 1], bodies[:, 1] + bodies[:, 3],
            wicks[:, 1], wicks[:, 3]])
    return result


def expectedBlocks(data: dict, k: int, width: float) -> dict:
    """
    Straightforward per-block aggregation, in the format of :func:`levelArrays`.

    Parameters
    ----------
    data : dict
        Candles, all valid.
    k : int
        Candles per block, blocks aligned to multiples of ``k``.
    width : float
        Candle width.

    Returns
    -------
    dict
        Expected rows for ``'up'`` and ``'down'``.
    """
    rows = {'up': [], 'down': []}
    n = len(data['x'])
    for start in range(0, n, k):
        stop = min(start + k, n)
        o, c = data['open'][start], data['close'][stop - 1]
        high, low = data['high'][start:stop].max(), data['low'][start:stop].min()
        center = (data['x'][start] + data['x'][stop - 1]) / 2
        row = (center, width * (stop - start) / 2, min(o, c), max(o, c), low, high)
        rows['up' if c >= o else 'down'].append(row)
    return {key: np.array(value).reshape(-1, 6) for key, value in rows.items()}


def test_default_width_and_bounds():
    item = pg.CandlestickItem(**sixCandles())
    assert item.width == pytest.approx(0.8)
    assert item.dataBounds(0) == pytest.approx((0.6, 6.4))
    assert item.dataBounds(1) == pytest.approx((5, 23))
    # only candles 2 and 3 intersect x in [1.5, 3.2]
    assert item.dataBounds(1, orthoRange=(1.5, 3.2)) == pytest.approx((9, 23))
    assert item.dataBounds(1, orthoRange=(10, 12)) == (None, None)
    rect = item.boundingRect()
    assert rect.left() <= 0.6 and rect.right() >= 6.4
    assert rect.top() <= 5 and rect.bottom() >= 23

    empty = pg.CandlestickItem()
    assert empty.dataBounds(0) == (None, None)
    assert empty.boundingRect().isEmpty()


def test_width_options():
    data = randomCandles(50)
    data['x'] = data['x'] * 2.0  # spacing 2
    item = pg.CandlestickItem(**data)
    assert item.width == pytest.approx(1.6)
    item.width = 0.5
    item.setData(**data)  # an explicit width is kept
    assert item.width == 0.5
    item.setData(width=1.0, **data)
    assert item.width == 1.0
    item.width = None  # back to the default
    assert item.width == pytest.approx(1.6)
    with pytest.raises(ValueError):
        item.width = -1


def test_options():
    item = pg.CandlestickItem(upBrush='b', downPen=None, wickPen='w', lod=False,
                              name='ohlc')
    assert item.upBrush.color() == pg.mkColor('b')
    assert item.downPen.style() == QtCore.Qt.PenStyle.NoPen
    assert item.wickPen.color() == pg.mkColor('w')
    assert item.lod is False
    assert item.name() == 'ohlc'
    assert item.implements('plotData')
    with pytest.raises(TypeError):
        item.setOpts(color='r')


@pytest.mark.parametrize('antialias', [False, True], ids=['aliased', 'antialiased'])
@pytest.mark.parametrize('wickPen', [None, 'w'])
def test_reference_render(antialias, wickPen):
    item = pg.CandlestickItem(wickPen=wickPen, **sixCandles())
    rect = QtCore.QRectF(0.3, 3.5, 6.4, 21)
    image, _ = paint(item, rect, antialias=antialias)
    expected, _ = paint(None, rect, antialias=antialias, painter=referencePainter(item))
    assert image.any()
    np.testing.assert_array_equal(image, expected)


def test_reference_render_zoomed():
    item = pg.CandlestickItem(upPen='w', downPen='y', **randomCandles(5000))
    rect = QtCore.QRectF(2000.3, 80, 60, 50)  # culled to about 60 candles, 6 px each
    image, recorder = paint(item, rect, size=(400, 300))
    expected, _ = paint(None, rect, size=(400, 300), painter=referencePainter(item))
    np.testing.assert_array_equal(image, expected)
    assert 60 <= recorder.rects <= 64
    assert recorder.lines == recorder.rects


@pytest.mark.parametrize('n', [1_000, 200_000])
def test_only_visible_candles_are_drawn(n):
    item = pg.CandlestickItem(width=0.4, **randomCandles(n))
    # candles 501 to 700 (200) are visible, 8 px each; bodies of candles 500 and
    # 701 (0.4 wide) stay out of [500.5, 700.5], even with the pen margin
    _, recorder = paint(item, QtCore.QRectF(500.5, 0, 200, 200), size=(1600, 100))
    assert recorder.rects == 200
    assert recorder.lines == 200


def test_lod_aggregates_blocks():
    n = 100_000
    item = pg.CandlestickItem(**randomCandles(n))
    # 0.008 px per candle: blocks of 512 candles, about 4 px each
    _, recorder = paint(item, QtCore.QRectF(-0.5, 0, n, 200), size=(1000, 100))
    assert recorder.rects == math.ceil(n / 512)
    assert recorder.lines == recorder.rects

    item.lod = False
    _, recorder = paint(item, QtCore.QRectF(-0.5, 0, n, 200), size=(1000, 100))
    assert recorder.rects == n


@pytest.mark.parametrize('k', [4, 32])
def test_exact_ohlc_aggregation(k):
    data = randomCandles(1000 + k // 2, seed=4)  # the last block is incomplete
    item = pg.CandlestickItem(**data)
    level = item._level(k, -math.inf, math.inf)
    actual = levelArrays(level)
    expected = expectedBlocks(data, k, item.width)
    for side in ('up', 'down'):
        assert len(expected[side]) > 0
        np.testing.assert_allclose(actual[side], expected[side], rtol=0, atol=1e-9)


def test_aggregation_known_case():
    x = np.arange(8, dtype=float)
    o = np.array([10, 11, 12, 13, 20, 19, 18, 17], dtype=float)
    c = np.array([11, 12, 13, 14, 19, 18, 17, 16], dtype=float)
    h = np.array([12, 15, 13, 14, 21, 25, 19, 18], dtype=float)
    lo = np.array([9, 10, 8, 12, 18, 17, 16, 15], dtype=float)
    item = pg.CandlestickItem(x=x, open=o, high=h, low=lo, close=c, width=0.5)
    arrays = levelArrays(item._level(4, -math.inf, math.inf))
    # block 0: open 10, close 14, high 15, low 8 (rising); block 1: 20, 16, 25, 15
    np.testing.assert_array_equal(arrays['up'], [[1.5, 1.0, 10, 14, 8, 15]])
    np.testing.assert_array_equal(arrays['down'], [[5.5, 1.0, 16, 20, 15, 25]])


@pytest.mark.parametrize('k', [1, 4, 16, 64])
def test_appendData_matches_setData(k):
    data = randomCandles(3000, seed=2)
    reference = pg.CandlestickItem(**data)
    item = pg.CandlestickItem(width=reference.width,
                              **{key: v[:1000] for key, v in data.items()})
    level = item._level(k, -math.inf, math.inf)  # cached for k >= 16: updated in place
    for start, stop in [(1000, 1001), (1001, 1017), (1017, 2500), (2500, 3000)]:
        item.appendData(**{key: v[start:stop] for key, v in data.items()})
    for key, values in zip(('x', 'open', 'high', 'low', 'close'), item.ohlc):
        np.testing.assert_array_equal(values, data[key])
    if k >= 16:
        assert item._level(k, -math.inf, math.inf) is level
    actual = levelArrays(item._level(k, -math.inf, math.inf))
    expected = levelArrays(reference._level(k, -math.inf, math.inf))
    for side in ('up', 'down'):
        np.testing.assert_array_equal(actual[side], expected[side])
    assert item.dataBounds(0) == reference.dataBounds(0)
    assert item.dataBounds(1) == reference.dataBounds(1)


def test_appendData_keeps_complete_blocks():
    data = randomCandles(1000, seed=3)
    item = pg.CandlestickItem(**{key: v[:998] for key, v in data.items()})
    before = levelArrays(item._level(64, -math.inf, math.inf))
    item.appendData(**{key: v[998:] for key, v in data.items()})
    after = levelArrays(item._level(64, -math.inf, math.inf))
    complete = 998 // 64  # blocks 0 to 14 are complete, block 15 changes
    for side in ('up', 'down'):
        unchanged = before[side][:, 0] < complete * 64
        kept = before[side][unchanged]
        np.testing.assert_array_equal(after[side][:len(kept)], kept)


def test_unsorted_data_is_sorted():
    data = sixCandles()
    order = np.array([3, 0, 5, 1, 4, 2])
    item = pg.CandlestickItem(**{key: v[order] for key, v in data.items()})
    for key, values in zip(('x', 'open', 'high', 'low', 'close'), item.ohlc):
        np.testing.assert_array_equal(values, data[key])
    # appending before the last candle sorts again
    item.setData(**{key: v[3:] for key, v in data.items()})
    item.appendData(**{key: v[:3] for key, v in data.items()})
    np.testing.assert_array_equal(item.ohlc[0], data['x'])


def test_non_finite_values():
    data = randomCandles(10)
    data['x'][2] = np.nan  # dropped
    data['close'][5] = np.nan  # kept but not drawn
    item = pg.CandlestickItem(**data)
    assert len(item.ohlc[0]) == 9
    _, recorder = paint(item, QtCore.QRectF(-0.5, 0, 10, 200))
    assert recorder.rects == 8
    low, high = item.dataBounds(1)
    assert np.isfinite(low) and np.isfinite(high)
    arrays = levelArrays(item._level(4, -math.inf, math.inf))
    assert len(arrays['up']) + len(arrays['down']) == 3


def test_getData_is_read_only():
    item = pg.CandlestickItem(**sixCandles())
    x, close = item.getData()
    np.testing.assert_array_equal(close, SIX[:, 2])
    with pytest.raises(ValueError):
        x[0] = 0
    np.testing.assert_array_equal(item.getOriginalDataset()[1], close)


def test_in_plot_with_legend_and_autorange():
    pw = pg.PlotWidget()
    pw.resize(400, 300)
    legend = pw.addLegend()
    item = pg.CandlestickItem(name='OHLC', **sixCandles())
    pw.addItem(item)
    show_and_wait(pw)
    assert item in pw.getPlotItem().listDataItems()
    assert len(legend.items) == 1
    (xmin, xmax), (ymin, ymax) = pw.viewRange()
    assert xmin <= 0.6 and xmax >= 6.4 and ymin <= 5 and ymax >= 23
    pw.grab()  # paints the legend sample and the candles
    pw.close()
