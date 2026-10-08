"""
Follow-up tests of CandlestickItem: y range of the candles within an x range, as
used by ``ViewBox.setAutoVisible(y=True)``, alone and above a linked volume plot.

Values and call counts are asserted, never timings.
"""
import numpy as np
import pytest

import pyqtgraph as pg

app = pg.mkQApp()


def randomCandles(n: int, seed: int = 0) -> dict:
    """
    Random walk candles at random, increasing x positions.

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
    x = np.cumsum(rng.uniform(0.5, 1.5, n))
    return dict(x=x, open=open_, high=high, low=low, close=close)


def bruteForceYRange(data: dict, width: float,
                     orthoRange: tuple[float, float]) -> tuple[float | None, float | None]:
    """
    Lowest and highest finite price of the candles whose body intersects a range.

    Parameters
    ----------
    data : dict
        ``x``, ``open``, ``high``, ``low`` and ``close`` arrays.
    width : float
        Width of the candle bodies.
    orthoRange : tuple of float
        Range of x.

    Returns
    -------
    tuple of float or None
        ``(low, high)``, or ``(None, None)`` without such candle.
    """
    lo, hi = sorted(orthoRange)
    low, high = None, None
    rows = zip(*(data[key].tolist() for key in ('x', 'open', 'high', 'low', 'close')))
    for x, o, h, l, c in rows:
        if not (x + 0.5 * width >= lo and x - 0.5 * width <= hi):
            continue
        bottom = [v for v in (l, o, c) if np.isfinite(v)]
        top = [v for v in (h, o, c) if np.isfinite(v)]
        if bottom:
            low = min(bottom) if low is None else min(low, *bottom)
        if top:
            high = max(top) if high is None else max(high, *top)
    return low, high


@pytest.mark.parametrize('layout', ['sorted', 'shuffled', 'nan'])
def test_orthoRange_matches_brute_force(layout):
    rng = np.random.default_rng(1)
    data = randomCandles(500, seed=2)
    if layout == 'shuffled':
        order = rng.permutation(500)
        data = {key: value[order] for key, value in data.items()}
    elif layout == 'nan':
        for key in ('open', 'high', 'low', 'close'):
            data[key][rng.integers(0, 500, 40)] = np.nan
    item = pg.CandlestickItem(width=0.6, **data)
    span = data['x'].max()
    for lo, hi in rng.uniform(-0.1 * span, 1.1 * span, (300, 2)):
        expected = bruteForceYRange(data, 0.6, (lo, hi))
        assert item.dataBounds(1, orthoRange=(lo, hi)) == pytest.approx(expected)
    # bounds are included: body of the first candle ends at x[0] + 0.3
    first = np.sort(data['x'])[0]
    assert item.dataBounds(1, orthoRange=(first + 0.3, first + 0.3))[0] is not None
    assert item.dataBounds(1, orthoRange=(span + 1, span + 2)) == (None, None)


def test_orthoRange_pen_padding():
    data = randomCandles(50)
    item = pg.CandlestickItem(width=0.6, upPen=pg.mkPen('g', width=0.5, cosmetic=False),
                              **data)
    low, high = bruteForceYRange(data, 0.6, (10, 20))
    assert item.dataBounds(1, orthoRange=(10, 20)) == pytest.approx((low - 0.25,
                                                                     high + 0.25))


def test_orthoRange_covering_all_candles_uses_cached_range(monkeypatch):
    data = randomCandles(1000)
    item = pg.CandlestickItem(**data)
    full = item.dataBounds(1)
    calls = []
    original = pg.CandlestickItem._yRange

    def countingYRange(self, start: int, stop: int):
        calls.append((start, stop))
        return original(self, start, stop)

    monkeypatch.setattr(pg.CandlestickItem, '_yRange', countingYRange)
    assert item.dataBounds(1, orthoRange=(-1e9, 1e9)) == full
    assert item.dataBounds(1) == full
    assert calls == []
    item.dataBounds(1, orthoRange=(10, 20))
    assert len(calls) == 1 and calls[0][1] - calls[0][0] < 20


def test_orthoRange_after_appendData():
    data = randomCandles(300)
    head = {key: value[:200] for key, value in data.items()}
    tail = {key: value[200:] for key, value in data.items()}
    item = pg.CandlestickItem(width=0.6, **head)
    item.dataBounds(1)  # fills the cached full range
    item.appendData(**tail)
    lastX = data['x'][-1]
    for orthoRange in ((data['x'][250], lastX), (-1e9, 1e9), (data['x'][100], lastX)):
        expected = bruteForceYRange(data, 0.6, orthoRange)
        assert item.dataBounds(1, orthoRange=orthoRange) == pytest.approx(expected)


def visibleYRange(data: dict, width: float, view: pg.ViewBox) -> tuple[float, float]:
    """
    Expected y range of the candles visible in a view, from their prices.

    Parameters
    ----------
    data : dict
        Candles, as returned by :func:`randomCandles`.
    width : float
        Width of the candle bodies.
    view : ViewBox
        View whose x range is used.

    Returns
    -------
    tuple of float
        Lowest low and highest high.
    """
    low, high = bruteForceYRange(data, width, tuple(view.viewRange()[0]))
    assert low is not None
    return low, high


def assertFits(view: pg.ViewBox, low: float, high: float) -> None:
    """
    Assert that the y range of ``view`` is ``[low, high]`` plus its auto-range padding.

    Parameters
    ----------
    view : ViewBox
        View to check.
    low, high : float
        Range that the view must fit.
    """
    ymin, ymax = view.viewRange()[1]
    span = high - low
    assert ymin <= low and ymax >= high
    assert ymin > low - 0.1 * span and ymax < high + 0.1 * span


def test_autoVisible_y_fits_visible_candles_when_panning():
    data = randomCandles(2000)
    pw = pg.PlotWidget()
    pw.resize(400, 300)
    item = pg.CandlestickItem(**data)
    pw.addItem(item)
    pw.show()
    for _ in range(3):
        app.processEvents()
    pw.setAutoVisible(y=True)
    vb = pw.getViewBox()
    pw.setXRange(data['x'][100], data['x'][160], padding=0)
    for _ in range(3):
        app.processEvents()
    assertFits(vb, *visibleYRange(data, item.width, vb))
    for step in (300, -150, 900):
        vb.translateBy(x=step)
        for _ in range(3):
            app.processEvents()
        assertFits(vb, *visibleYRange(data, item.width, vb))
    pw.close()


def test_autoVisible_y_with_linked_volume_plot():
    data = randomCandles(2000)
    volume = np.random.default_rng(3).uniform(1, 100, 2000)
    win = pg.GraphicsLayoutWidget(size=(500, 500))
    price = win.addPlot(row=0, col=0)
    bars = win.addPlot(row=1, col=0)
    bars.setXLink(price)
    candles = pg.CandlestickItem(**data)
    price.addItem(candles)
    bars.addItem(pg.BarGraphItem(x=data['x'], height=volume, width=candles.width,
                                 pen=None))
    for plot in (price, bars):
        plot.setAutoVisible(y=True)
    win.show()
    for _ in range(3):
        app.processEvents()

    def check() -> None:
        assertFits(price.vb, *visibleYRange(data, candles.width, price.vb))
        xmin, xmax = bars.vb.viewRange()[0]
        half = 0.5 * candles.width
        visible = (data['x'] + half >= xmin) & (data['x'] - half <= xmax)
        assertFits(bars.vb, 0.0, volume[visible].max())

    price.setXRange(data['x'][500], data['x'][560], padding=0)
    for _ in range(3):
        app.processEvents()
    check()
    for step in (200, -400):
        price.vb.translateBy(x=step)
        for _ in range(3):
            app.processEvents()
        check()
    win.close()
