"""
Follow-up tests of BarGraphItem: data bounds restricted to a range of the other axis
(``ViewBox.setAutoVisible``), percentile bounds and CSV export.

Values are asserted, never timings.
"""
import csv

import numpy as np
import pytest

import pyqtgraph as pg
from pyqtgraph.exporters import CSVExporter

app = pg.mkQApp()

HEIGHTS = np.array([5., 1, 7, 3, 9, 2, 8, 4, 6, 10])


def bruteForceBounds(item: pg.BarGraphItem, ax: int,
                     orthoRange: tuple[float, float]) -> tuple[float | None, float | None]:
    """
    Bounds of the bars intersecting ``orthoRange``, bar by bar, without pen.

    Parameters
    ----------
    item : BarGraphItem
        Item whose bars are tested.
    ax : int
        Axis of the bounds, 0 for x and 1 for y.
    orthoRange : tuple of float
        Range along the other axis.

    Returns
    -------
    tuple of float or None
        ``(min, max)`` over the intersecting bars with finite coordinates, or
        ``(None, None)``.
    """
    lo, hi = sorted(orthoRange)
    low, high = None, None
    for x0, y0, width, height in item._rectarray.ndarray().tolist():
        rect = ((x0, x0 + width), (y0, y0 + height))
        a, b = rect[1 - ax]
        if not (b >= lo and a <= hi):
            continue
        c, d = rect[ax]
        if np.isnan(c) or np.isnan(d):
            continue
        low = c if low is None else min(low, c)
        high = d if high is None else max(high, d)
    return low, high


def test_orthoRange_selects_intersecting_bars():
    item = pg.BarGraphItem(x=np.arange(10.), height=HEIGHTS, width=0.6, pen=None)
    # full range: unchanged
    assert item.dataBounds(0) == pytest.approx((-0.3, 9.3))
    assert item.dataBounds(1) == pytest.approx((0, 10))
    # bars 3, 4 and 5 (x0 = 4.7 <= 5.2) intersect [2.5, 5.2]
    assert item.dataBounds(1, orthoRange=(2.5, 5.2)) == pytest.approx((0, 9))
    assert item.dataBounds(1, orthoRange=(5.2, 2.5)) == pytest.approx((0, 9))
    # bounds are included: bar 5 ends at 5.3, bar 6 starts at 5.7
    assert item.dataBounds(1, orthoRange=(5.3, 5.3)) == pytest.approx((0, 2))
    assert item.dataBounds(1, orthoRange=(5.5, 5.7)) == pytest.approx((0, 8))
    # all bars
    assert item.dataBounds(1, orthoRange=(-100, 100)) == pytest.approx((0, 10))
    # between two bars, and outside of the data
    assert item.dataBounds(1, orthoRange=(5.4, 5.6)) == (None, None)
    assert item.dataBounds(1, orthoRange=(20, 30)) == (None, None)
    # x range of the bars reaching y in [8.5, 9.5]: bars 4 and 9
    assert item.dataBounds(0, orthoRange=(8.5, 9.5)) == pytest.approx((3.7, 9.3))
    assert item.dataBounds(0, orthoRange=(11, 12)) == (None, None)


def test_orthoRange_with_negative_heights_and_offsets():
    heights = HEIGHTS * np.where(np.arange(10) % 2, -1, 1)  # 5, -1, 7, -3, ...
    item = pg.BarGraphItem(x=np.arange(10.), y0=1.0, height=heights, width=0.6)
    # bars 1, 2, 3: 1 - 1 = 0, 1 + 7 = 8, 1 - 3 = -2
    assert item.dataBounds(1, orthoRange=(0.8, 3.0)) == pytest.approx((-2, 8))


@pytest.mark.parametrize('layout', ['sorted', 'shuffled', 'overlapping', 'nan'])
def test_orthoRange_matches_brute_force(layout):
    rng = np.random.default_rng(1)
    n = 300
    x = np.cumsum(rng.uniform(0.5, 2.0, n))
    width = rng.uniform(0.1, 0.5, n)  # right edges sorted as well
    height = rng.normal(0, 5, n)
    y0 = rng.normal(0, 1, n)
    if layout == 'overlapping':
        # sorted left edges, right edges not sorted: some bars span many others
        width[::17] = rng.uniform(5, 40, len(width[::17]))
    elif layout == 'shuffled':
        order = rng.permutation(n)
        x, width, height, y0 = x[order], width[order], height[order], y0[order]
    elif layout == 'nan':
        height[::13] = np.nan
    item = pg.BarGraphItem(x0=x, width=width, y0=y0, height=height, pen=None)
    assert item._xSorted == (layout != 'shuffled')
    assert item._x1Sorted == (layout in ('sorted', 'nan'))
    span = x.max()
    for lo, hi in rng.uniform(-0.1 * span, 1.1 * span, (200, 2)):
        expected = bruteForceBounds(item, 1, (lo, hi))
        assert item.dataBounds(1, orthoRange=(lo, hi)) == pytest.approx(expected)
    for lo, hi in rng.uniform(-15, 15, (100, 2)):
        expected = bruteForceBounds(item, 0, (lo, hi))
        assert item.dataBounds(0, orthoRange=(lo, hi)) == pytest.approx(expected)


def test_orthoRange_ignores_bars_with_nan():
    heights = HEIGHTS.copy()
    heights[[4, 9]] = np.nan
    item = pg.BarGraphItem(x=np.arange(10.), height=heights, width=0.6)
    assert item.dataBounds(1) == pytest.approx((0, 8))  # full range: NaN bars ignored
    assert item.dataBounds(1, orthoRange=(-100, 100)) == pytest.approx((0, 8))
    assert item.dataBounds(1, orthoRange=(2.5, 5.2)) == pytest.approx((0, 3))
    assert item.dataBounds(1, orthoRange=(3.8, 4.2)) == (None, None)
    assert item.dataBounds(1, frac=0.5) == pytest.approx(
        (0, np.percentile(heights[np.isfinite(heights)], 75)))


def test_orthoRange_pen_padding():
    x, heights = np.arange(10.), HEIGHTS
    thick = pg.BarGraphItem(x=x, height=heights, width=0.6,
                            pen=pg.mkPen('w', width=0.4, cosmetic=False))
    assert thick.dataBounds(1, orthoRange=(2.5, 5.2)) == pytest.approx((-0.2, 9.2))
    assert thick.dataBounds(1) == pytest.approx((-0.2, 10.2))
    assert thick.dataBounds(1, orthoRange=(20, 30)) == (None, None)
    cosmetic = pg.BarGraphItem(x=x, height=heights, width=0.6, pen=pg.mkPen('w', width=3))
    assert cosmetic.dataBounds(1, orthoRange=(2.5, 5.2)) == pytest.approx((0, 9))


def test_frac_uses_percentiles():
    rng = np.random.default_rng(2)
    heights = rng.exponential(1.0, 1000)
    heights[[10, 500]] = 1000.0  # spikes
    y0 = rng.normal(0, 0.1, 1000)
    item = pg.BarGraphItem(x=np.arange(1000.), y0=y0, height=heights, width=0.8, pen=None)
    expected = (np.percentile(y0, 5), np.percentile(y0 + heights, 95))
    assert item.dataBounds(1, frac=0.9) == pytest.approx(expected)
    assert item.dataBounds(1, frac=0.9)[1] < 10  # the spikes are left out
    # combined with a range of x: bars 0 to 199
    expected = (np.percentile(y0[:200], 5), np.percentile((y0 + heights)[:200], 95))
    assert item.dataBounds(1, frac=0.9, orthoRange=(-1, 199.2)) == pytest.approx(expected)
    # x percentiles, of the left and right edges
    x0 = np.arange(1000.) - 0.4
    expected = (np.percentile(x0, 25), np.percentile(x0 + 0.8, 75))
    assert item.dataBounds(0, frac=0.5) == pytest.approx(expected)
    assert item.dataBounds(1, frac=1.0) == item.dataBounds(1)
    with pytest.raises(ValueError):
        item.dataBounds(1, frac=0.0)


def test_empty_item_bounds():
    item = pg.BarGraphItem(x=[], height=[], width=1)
    assert item.dataBounds(1) == (None, None)
    assert item.dataBounds(1, orthoRange=(0, 1)) == (None, None)
    assert item.dataBounds(1, frac=0.5) == (None, None)


def test_autoVisible_y_fits_visible_bars_when_panning():
    n = 1000
    x = np.arange(n, dtype=float)
    heights = 1.0 + x  # the highest visible bar is the rightmost one
    pw = pg.PlotWidget()
    pw.resize(400, 300)
    item = pg.BarGraphItem(x=x, height=heights, width=0.6)
    pw.addItem(item)
    pw.show()
    for _ in range(3):
        app.processEvents()
    assert pw.viewRange()[1][1] >= n
    pw.setAutoVisible(y=True)
    vb = pw.getViewBox()

    def checkFitsVisibleBars() -> None:
        (xmin, xmax), (ymin, ymax) = vb.viewRange()
        visible = (x + 0.3 >= xmin) & (x - 0.3 <= xmax)
        top = heights[visible].max()
        assert ymin <= 0 < ymin + 0.1 * top
        assert top <= ymax < 1.1 * top

    pw.setXRange(100, 200, padding=0)
    for _ in range(3):
        app.processEvents()
    checkFitsVisibleBars()
    for step in (300, -150, 400):
        vb.translateBy(x=step)
        for _ in range(3):
            app.processEvents()
        checkFitsVisibleBars()
    pw.close()


def test_getOriginalDataset():
    x = np.arange(4.)
    heights = np.array([3., -1, 2, 5])
    item = pg.BarGraphItem(x=x, height=heights, width=0.5)
    ox, oh = item.getOriginalDataset()
    np.testing.assert_array_equal(ox, x)
    np.testing.assert_array_equal(oh, heights)
    assert ox.dtype == oh.dtype == np.float64
    # scalar options are repeated, integer sequences converted
    item = pg.BarGraphItem(x=range(3), height=2, width=0.5)
    np.testing.assert_array_equal(item.getOriginalDataset()[0], [0, 1, 2])
    np.testing.assert_array_equal(item.getOriginalDataset()[1], [2, 2, 2])
    # bars given by their edges: centers and signed heights y1 - y0
    item = pg.BarGraphItem(x0=[0., 2], x1=[1., 4], y0=[1., 1], y1=[4., -1])
    np.testing.assert_array_equal(item.getOriginalDataset()[0], [0.5, 3])
    np.testing.assert_array_equal(item.getOriginalDataset()[1], [3, -2])
    item = pg.BarGraphItem(x1=[1., 4], width=1, y1=[4., 5])
    np.testing.assert_array_equal(item.getOriginalDataset()[0], [0.5, 3.5])
    np.testing.assert_array_equal(item.getOriginalDataset()[1], [4, 5])
    # the result is a copy
    item = pg.BarGraphItem(x=x, height=heights, width=0.5)
    item.getOriginalDataset()[1][0] = 100
    assert heights[0] == 3
    assert pg.BarGraphItem(x=[], height=[], width=1).getOriginalDataset() == (None, None)


def test_csv_export_of_bars_and_candles(tmp_path):
    plot = pg.PlotItem()
    x = np.arange(5.)
    volume = np.array([10., 30, 20, 50, 40])
    plot.addItem(pg.BarGraphItem(x=x, height=volume, width=0.6, name='volume'))
    close = np.array([101., 102, 100.5, 103, 104])
    candles = pg.CandlestickItem(x=x[::-1], open=close[::-1] - 1, high=close[::-1] + 2,
                                 low=close[::-1] - 2, close=close[::-1], name='price')
    plot.addItem(candles)
    plot.addItem(pg.BarGraphItem(x0=[0., 2, 4], x1=[1., 3, 5], height=[1., 2, 3]))
    plot.addItem(pg.BarGraphItem(x=[], height=[], width=1))  # no data: skipped

    fileName = tmp_path / 'bars.csv'
    CSVExporter(plot).export(fileName=str(fileName))
    with open(fileName, newline='') as file:
        rows = list(csv.reader(file))
    assert rows[0] == ['volume_x', 'volume_y', 'price_x', 'price_y', 'x0002', 'y0002']
    values = [[float(v) if v else None for v in row] for row in rows[1:]]
    assert len(values) == 5
    columns = list(zip(*values))
    assert columns[0] == tuple(x) and columns[1] == tuple(volume)
    assert columns[2] == tuple(x) and columns[3] == tuple(close)  # sorted by x
    assert columns[4] == (0.5, 2.5, 4.5, None, None)
    assert columns[5] == (1, 2, 3, None, None)
