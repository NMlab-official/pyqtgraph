"""
Streaming tests of CandlestickItem: ``appendData(..., replaceLast=True)`` updates the
current candle in place, as each tick of a live chart does.

The state after a stream of appends and replacements (candles, bounds, cached levels
of detail, rendering) is compared with ``setData`` of the final candles. Values,
call ranges and pixels are asserted, never timings.
"""
import math

import numpy as np
import pytest

import pyqtgraph as pg
from pyqtgraph.graphicsItems import CandlestickItem as candlestickModule
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

app = pg.mkQApp()

STEP = 60.0
FIELDS = ('x', 'open', 'high', 'low', 'close')


def randomCandles(n: int, seed: int = 0) -> dict:
    """
    Random walk one-minute candles.

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
    return dict(x=STEP * np.arange(n), open=open_, high=high, low=low, close=close)


def candle(x: float, open: float, high: float, low: float, close: float) -> dict:
    """
    One candle, as keyword arguments of ``appendData``.

    Parameters
    ----------
    x, open, high, low, close : float
        The candle.

    Returns
    -------
    dict
        Arrays of length 1.
    """
    return {key: np.array([value], dtype=float)
            for key, value in zip(FIELDS, (x, open, high, low, close))}


class LiveFeed:
    """
    Candles fed by ticks: each tick updates the last candle, or starts a new one.

    The candles are also kept as plain arrays, the expected content of the item.

    Parameters
    ----------
    data : dict
        Initial candles.
    seed : int
        Random seed.
    """

    def __init__(self, data: dict, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.candles = [list(map(float, row)) for row in zip(*(data[f] for f in FIELDS))]

    def tick(self, nanProbability: float = 0.0) -> tuple[dict, bool]:
        """
        Next update of the feed.

        Parameters
        ----------
        nanProbability : float, default 0.0
            Probability that the updated candle gets a NaN close (an invalid candle).

        Returns
        -------
        candle : dict
            The new or updated candle, as keyword arguments of ``appendData``.
        replace : bool
            Whether it replaces the last candle.
        """
        x, o, h, lo, c = self.candles[-1]
        if self.rng.random() < 0.25:  # a new candle starts at the last close
            start = c if math.isfinite(c) else o
            row = [x + STEP, start, start, start, start]
            self.candles.append(row)
            replace = False
        else:
            price = (c if math.isfinite(c) else o) + self.rng.standard_normal()
            row = [x, o, max(h, price), min(lo, price), price]
            self.candles[-1] = row
            replace = True
        if self.rng.random() < nanProbability:
            row = row[:4] + [math.nan]
            self.candles[-1] = row
        return candle(*row), replace

    def data(self) -> dict:
        """
        The expected candles.

        Returns
        -------
        dict
            ``x``, ``open``, ``high``, ``low`` and ``close`` arrays.
        """
        columns = np.array(self.candles).T
        return dict(zip(FIELDS, columns))


def paint(item: QtWidgets.QGraphicsItem, rect: QtCore.QRectF,
          size: tuple[int, int] = (400, 300), antialias: bool = False) -> np.ndarray:
    """
    Paint ``item`` into an image showing ``rect``.

    Parameters
    ----------
    item : QGraphicsItem
        Item whose ``paint`` method is called.
    rect : QtCore.QRectF
        Item-coordinate rectangle mapped onto the whole image.
    size : tuple of int, default (400, 300)
        Image size.
    antialias : bool, default False
        Antialiasing render hint.

    Returns
    -------
    numpy.ndarray
        Rendered pixels, shape (height, width, 4).
    """
    img = QtGui.QImage(size[0], size[1], QtGui.QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(0)
    p = QtGui.QPainter(img)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, antialias)
    p.scale(size[0] / rect.width(), size[1] / rect.height())
    p.translate(-rect.left(), -rect.top())
    try:
        item.paint(p, QtWidgets.QStyleOptionGraphicsItem(), None)
    finally:
        p.end()
    return pg.functions.ndarray_from_qimage(img).copy()


def levelArrays(level) -> dict:
    """
    Aggregated candles of a level, as their body and wick coordinates.

    Parameters
    ----------
    level : _Level
        Level of detail of a CandlestickItem.

    Returns
    -------
    dict
        Centers, bodies and wicks of the rising (``'up'``) and falling (``'down'``)
        candles.
    """
    return {name: np.column_stack([side.x.view, side.bodies.ndarray(),
                                   side.wicks.ndarray()])
            for name, side in (('up', level.up), ('down', level.down))}


def assertSameAsSetData(item: pg.CandlestickItem, data: dict) -> pg.CandlestickItem:
    """
    Assert that ``item`` is in the state ``setData`` with ``data`` gives.

    Parameters
    ----------
    item : CandlestickItem
        Item updated by ``appendData``.
    data : dict
        Expected candles.

    Returns
    -------
    CandlestickItem
        The reference item, built by ``setData`` with the width of ``item``.
    """
    reference = pg.CandlestickItem(width=item.width, **data)
    for values, expected in zip(item.ohlc, reference.ohlc):
        np.testing.assert_array_equal(values, expected)
    assert item.dataBounds(0) == reference.dataBounds(0)
    assert item.dataBounds(1) == reference.dataBounds(1)
    x = data['x']
    for span in [(x[0], x[-1]), (x[-1] - 10 * STEP, x[-1]), (x[len(x) // 2], x[-1] + STEP)]:
        expected = reference.dataBounds(1, orthoRange=span)
        assert item.dataBounds(1, orthoRange=span) == expected
    for k, level in item._levels.items():
        actual = levelArrays(level)
        expected = levelArrays(reference._level(k, -math.inf, math.inf))
        for side in ('up', 'down'):
            np.testing.assert_array_equal(actual[side], expected[side])
    return reference


def lastCandlesRect(data: dict, count: int) -> QtCore.QRectF:
    """
    View of the last candles of a series.

    Parameters
    ----------
    data : dict
        Candles.
    count : int
        Number of candles shown.

    Returns
    -------
    QtCore.QRectF
        Rectangle in item coordinates, a little larger than the candles.
    """
    x = data['x']
    left, right = x[-count] - STEP / 2, x[-1] + STEP / 2
    low, high = np.nanmin(data['low'][-count:]), np.nanmax(data['high'][-count:])
    return QtCore.QRectF(left, low - 1, right - left, high - low + 2)


@pytest.mark.parametrize('nanProbability', [0.0, 0.05], ids=['valid', 'some NaN'])
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_stream_matches_setData(seed, nanProbability):
    data = randomCandles(1000, seed=seed)
    item = pg.CandlestickItem(**data)
    item.dataBounds(1)  # cached: updated incrementally from now on
    for k in (16, 64, 1024):
        item._level(k, -math.inf, math.inf)  # cached levels, updated incrementally
    levels = dict(item._levels)
    feed = LiveFeed(data, seed)
    replaced = 0
    for _ in range(400):
        update, replace = feed.tick(nanProbability)
        item.appendData(replaceLast=replace, **update)
        replaced += replace
    assert replaced > 200
    assert item._levels == levels  # the cached levels were updated, not rebuilt
    assertSameAsSetData(item, feed.data())


@pytest.mark.parametrize('antialias', [False, True], ids=['aliased', 'antialiased'])
@pytest.mark.parametrize('view', ['zoomed', 'lod window', 'lod cached'])
def test_stream_pixel_identical_to_setData(view, antialias):
    data = randomCandles(3000, seed=5)
    item = pg.CandlestickItem(upPen='w', downPen='y', **data)
    feed = LiveFeed(data, seed=5)
    counts = {'zoomed': 60, 'lod window': 500, 'lod cached': 3000}
    for _ in range(120):
        # paint between updates, as a live chart does: the caches must follow
        paint(item, lastCandlesRect(feed.data(), counts[view]), antialias=antialias)
        update, replace = feed.tick(0.02)
        item.appendData(replaceLast=replace, **update)
    final = feed.data()
    rect = lastCandlesRect(final, counts[view])
    reference = assertSameAsSetData(item, final)
    reference.setOpts(upPen='w', downPen='y')
    image = paint(item, rect, antialias=antialias)
    assert image.any()
    np.testing.assert_array_equal(image, paint(reference, rect, antialias=antialias))
    k = item._lodFactor(400 / rect.width())
    assert {'zoomed': k == 1, 'lod window': 1 < k < 16, 'lod cached': k >= 16}[view]


def test_bounds_after_replacing_the_extreme_candle(monkeypatch):
    data = randomCandles(500, seed=3)
    data['high'][-1] = 1000.0  # the last candle holds both extremes
    data['low'][-1] = -1000.0
    item = pg.CandlestickItem(**data)
    assert item.dataBounds(1) == (-1000.0, 1000.0)
    calls = []
    original = item._yRange

    def recordingYRange(start: int, stop: int):
        calls.append((start, stop))
        return original(start, stop)
    monkeypatch.setattr(item, '_yRange', recordingYRange)

    last = {key: values[-1] for key, values in data.items()}
    item.appendData(replaceLast=True, **candle(last['x'], last['open'], last['open'] + 1,
                                              last['open'] - 1, last['open']))
    data['high'][-1], data['low'][-1] = last['open'] + 1, last['open'] - 1
    data['close'][-1] = last['open']
    expected = (min(data['low'].min(), data['open'].min(), data['close'].min()),
                max(data['high'].max(), data['open'].max(), data['close'].max()))
    assert item.dataBounds(1) == pytest.approx(expected, abs=0)
    # no scan of the existing candles: only the replacing candle is read
    assert calls and all(stop - start <= 1 for start, stop in calls)
    assertSameAsSetData(item, data)


def test_replaceLast_recomputes_only_the_last_block(monkeypatch):
    data = randomCandles(1000, seed=4)  # 1000 = 15 * 64 + 40
    item = pg.CandlestickItem(**data)
    level = item._level(64, -math.inf, math.inf)
    lengths = []
    original = candlestickModule._Level._aggregate

    def recordingAggregate(self, x, *args):
        if self is level:
            lengths.append(len(x))
        return original(self, x, *args)
    monkeypatch.setattr(candlestickModule._Level, '_aggregate', recordingAggregate)

    last = {key: values[-1] for key, values in data.items()}
    item.appendData(replaceLast=True,
                    **candle(last['x'], last['open'], 500.0, 1.0, 400.0))
    assert lengths == [40]  # the last, incomplete block only
    for key, value in zip(FIELDS[2:], (500.0, 1.0, 400.0)):
        data[key][-1] = value
    assertSameAsSetData(item, data)
    assert item._level(64, -math.inf, math.inf) is level

    # a block holding only the replaced candle
    item.appendData(**{key: values[-24:] + (STEP * 24 if key == 'x' else 0)
                       for key, values in data.items()})
    assert len(item.ohlc[0]) == 1024 and lengths == [40, 64]
    lengths.clear()
    item.appendData(**candle(1024 * STEP, 100, 101, 99, 100.5))
    item.appendData(replaceLast=True, **candle(1024 * STEP, 100, 102, 98, 99))
    assert lengths == [1, 1]
    level = levelArrays(item._level(64, -math.inf, math.inf))
    assert level['down'][-1, 0] == 1024 * STEP  # the last block, falling now
    assert level['up'][-1, 0] < 1024 * STEP  # no rising copy of it is left


@pytest.mark.parametrize('position', ['same', 'later', 'between', 'before previous'])
def test_replacing_candle_position(position, monkeypatch):
    data = randomCandles(100, seed=6)
    item = pg.CandlestickItem(**data)
    setDataCalls = []
    original = item.setData

    def recordingSetData(**kwargs):
        setDataCalls.append(len(kwargs['x']))
        return original(**kwargs)
    monkeypatch.setattr(item, 'setData', recordingSetData)

    x = {'same': data['x'][-1], 'later': data['x'][-1] + 30,
         'between': data['x'][-2] + 10, 'before previous': data['x'][-5] + 10}[position]
    item.appendData(replaceLast=True, **candle(x, 90, 95, 85, 92))
    expected = {key: values[:-1] for key, values in data.items()}
    for key, value in zip(FIELDS, (x, 90, 95, 85, 92)):
        expected[key] = np.append(expected[key], value)
    order = np.argsort(expected['x'], kind='stable')
    expected = {key: values[order] for key, values in expected.items()}
    assertSameAsSetData(item, expected)
    # only a candle moved before its predecessor needs sorting all candles again
    assert setDataCalls == ([100] if position == 'before previous' else [])


def test_replaceLast_with_several_candles():
    data = randomCandles(300, seed=7)
    head = {key: values[:200] for key, values in data.items()}
    item = pg.CandlestickItem(**head)
    item.dataBounds(1)
    item._level(32, -math.inf, math.inf)
    # the first new candle replaces candle 199, the next ones follow it
    item.appendData(replaceLast=True, **{key: values[199:] for key, values in data.items()})
    assertSameAsSetData(item, data)


def test_replaceLast_with_few_candles_and_automatic_width():
    item = pg.CandlestickItem(**candle(0.0, 10, 10, 10, 10))
    item.appendData(replaceLast=True, **candle(0.0, 10, 12, 9, 11))
    item.appendData(**candle(STEP, 11, 11, 11, 11))
    item.appendData(replaceLast=True, **candle(STEP, 11, 13, 10, 12))
    final = dict(x=[0.0, STEP], open=[10.0, 11], high=[12.0, 13], low=[9.0, 10],
                 close=[11.0, 12])
    assert item.width == pg.CandlestickItem(**final).width == pytest.approx(0.8 * STEP)
    assertSameAsSetData(item, {key: np.array(v) for key, v in final.items()})


@pytest.mark.parametrize('kwargs, message', [
    (dict(x=[], open=[], high=[], low=[], close=[]), 'needs a new candle'),
    (dict(x=[np.nan], open=[1], high=[2], low=[0], close=[1]), 'finite x'),
    (dict(x=[60.0, 120.0], open=[1], high=[2], low=[0], close=[1]), 'same length'),
])
def test_replaceLast_errors_leave_the_item_unchanged(kwargs, message):
    data = randomCandles(10)
    item = pg.CandlestickItem(**data)
    with pytest.raises(ValueError, match=message):
        item.appendData(replaceLast=True, **kwargs)
    assertSameAsSetData(item, data)

    empty = pg.CandlestickItem()
    with pytest.raises(ValueError, match='existing candle'):
        empty.appendData(replaceLast=True, **candle(0.0, 1, 2, 0, 1))
    assert len(empty.ohlc[0]) == 0
