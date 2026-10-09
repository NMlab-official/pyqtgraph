"""
Tests of the min/max pyramid of the 'peak' downsampling and of the automatic data
reduction (``autoReduce``) of :class:`~pyqtgraph.PlotDataItem`.

The tests compare results with direct computations and check cache states; they never
assert durations (see ``benchmarks/scenarios.py``, scenario S14, for the timings).
"""
import math

import numpy as np
import pytest

import pyqtgraph as pg
from pyqtgraph.graphicsItems import PlotDataItem as pdi_module
from pyqtgraph.graphicsItems._MinMaxPyramid import MinMaxPyramid
from tests.perf_helpers import process_events, show_and_wait

app = pg.mkQApp()

try:
    import numba  # noqa: F401
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False


@pytest.fixture(params=[False, pytest.param(True, marks=pytest.mark.skipif(
    not HAVE_NUMBA, reason='numba is not installed'))], ids=['numpy', 'numba'])
def use_numba(request):
    old = pg.getConfigOption('useNumba')
    pg.setConfigOptions(useNumba=request.param)
    yield request.param
    pg.setConfigOptions(useNumba=old)


@pytest.fixture
def plot_widget():
    pw = pg.PlotWidget()
    pw.resize(400, 300)
    show_and_wait(pw)
    yield pw
    pw.close()


@pytest.fixture
def low_thresholds(monkeypatch):
    """Use the pyramid for blocks of 64 values or more and any range of blocks."""
    monkeypatch.setattr(MinMaxPyramid, 'efficientBlockSize', staticmethod(lambda: 64))
    monkeypatch.setattr(pdi_module._PeakBlockCache, 'PYRAMID_MIN_VALUES', 0)


def _direct(y, first, end, ds):
    blocks = y[first * ds:end * ds].reshape(end - first, ds)
    return blocks.max(axis=1), blocks.min(axis=1)


def _random_data(n, dtype, rng):
    if np.dtype(dtype).kind == 'f':
        y = rng.standard_normal(n).astype(dtype)
        for value, count in ((np.nan, 3), (np.inf, 2), (-np.inf, 2)):
            y[rng.integers(0, n, count)] = value
        return y
    info = np.iinfo(dtype)
    return rng.integers(info.min, info.max, n, dtype=dtype, endpoint=True)


def _check_blocks(pyramid, y, first, end, ds):
    out_max = np.empty(end - first, dtype=y.dtype)
    out_min = np.empty(end - first, dtype=y.dtype)
    pyramid.blocks(first, end, ds, out_max, out_min)
    ref_max, ref_min = _direct(y, first, end, ds)
    np.testing.assert_array_equal(out_max, ref_max)
    np.testing.assert_array_equal(out_min, ref_min)


# --------------------------------------------------------------------------------------
# MinMaxPyramid
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize('dtype', [np.float64, np.float32, np.int16, np.uint8, np.int64])
@pytest.mark.parametrize('n', [64, 100, 1000, 4097, 65_536, 100_003])
def test_pyramid_blocks_match_direct_computation(use_numba, dtype, n):
    rng = np.random.default_rng(n)
    y = _random_data(n, dtype, rng)
    pyramid = MinMaxPyramid(y)
    min_ds = MinMaxPyramid.minBlockSize()
    sizes = [min_ds, min_ds + 1, 127, 128, 1000, 1024, n // 2, n]
    sizes += [int(v) for v in rng.integers(min_ds, max(min_ds + 1, n // 2), 30)]
    for ds in sizes:
        if not min_ds <= ds <= n:
            continue
        num_blocks = n // ds
        # all blocks, single blocks at both ends, and random ranges
        ranges = [(0, num_blocks), (0, 1), (num_blocks - 1, num_blocks)]
        for _ in range(3):
            first = int(rng.integers(0, num_blocks))
            ranges.append((first, int(rng.integers(first + 1, num_blocks + 1))))
        for first, end in ranges:
            _check_blocks(pyramid, y, first, end, ds)


def test_pyramid_propagates_nan_like_numpy(use_numba):
    n = 1 << 14
    y = np.arange(n, dtype=float)
    ds = 100
    pyramid_values = {
        3: np.nan,         # first value of a block
        ds - 1: np.nan,    # last value of a block
        5 * ds + 40: np.nan,  # within a cell of the block
        9 * ds + 7: np.inf,
        11 * ds + 50: -np.inf,
    }
    for index, value in pyramid_values.items():
        y[index] = value
    pyramid = MinMaxPyramid(y)
    _check_blocks(pyramid, y, 0, n // ds, ds)
    out_max = np.empty(1)
    out_min = np.empty(1)
    pyramid.blocks(5, 6, ds, out_max, out_min)
    assert np.isnan(out_max[0]) and np.isnan(out_min[0])


def test_pyramid_extend_matches_rebuild(use_numba):
    rng = np.random.default_rng(0)
    y = rng.standard_normal(300_000)
    pyramid = MinMaxPyramid(y[:1000])
    for size in (1001, 1023, 1024, 1056, 5000, 70_000, 300_000):
        pyramid.extend(y[:size])
        rebuilt = MinMaxPyramid(y[:size])
        assert len(pyramid._len) == len(rebuilt._len)
        for level in range(MinMaxPyramid.BASE, MinMaxPyramid.BASE + len(rebuilt._len)):
            for got, expected in zip(pyramid.levelExtremes(level),
                                     rebuilt.levelExtremes(level)):
                np.testing.assert_array_equal(got, expected)
        ds = 1000
        _check_blocks(pyramid, y[:size], 0, size // ds, ds)


def test_pyramid_size_and_errors():
    n = 1 << 20
    pyramid = MinMaxPyramid(np.zeros(n))
    # about n / 16 values per extreme over all levels
    assert sum(pyramid._len) == (n >> MinMaxPyramid.BASE) * 2 - 1
    out = np.empty(1)
    with pytest.raises(ValueError):
        pyramid.blocks(0, 1, MinMaxPyramid.minBlockSize() - 1, out, out)
    with pytest.raises(ValueError):
        pyramid.blocks(0, 2, n, out, out)
    with pytest.raises(ValueError):
        pyramid.levelExtremes(MinMaxPyramid.BASE - 1)
    mx, _ = pyramid.levelExtremes(MinMaxPyramid.BASE)
    assert not mx.flags.writeable


# --------------------------------------------------------------------------------------
# 'peak' downsampling of PlotDataItem with the pyramid
# --------------------------------------------------------------------------------------

def _view_sequence(n):
    """x ranges of zoom steps in and out, and of pan steps."""
    ranges = []
    lo, hi = 0.0, float(n)
    for scale in [0.8] * 6 + [1.25] * 4:
        center = 0.5 * (lo + hi) + 0.013 * (hi - lo)
        half = 0.5 * (hi - lo) * scale
        lo, hi = max(center - half, 0.0), min(center + half, float(n))
        ranges.append((lo, hi))
    for _ in range(4):
        shift = 0.07 * (hi - lo)
        lo, hi = lo + shift, hi + shift
        ranges.append((lo, hi))
    return ranges


def _displayed_sequence(plot_widget, x, y, **opts):
    item = plot_widget.plot(x, y, autoDownsample=True, downsampleMethod='peak', **opts)
    plot_widget.getPlotItem().enableAutoRange(False)
    shown = []
    for lo, hi in _view_sequence(len(y)):
        plot_widget.setXRange(lo, hi, padding=0)
        process_events()
        xd, yd = item.getData()
        shown.append((xd.copy(), yd.copy(), item._displayKey))
    plot_widget.removeItem(item)
    return shown, item


@pytest.mark.parametrize('clip', [False, True])
def test_peak_display_identical_with_pyramid(plot_widget, monkeypatch, clip):
    rng = np.random.default_rng(1)
    n = 300_000
    x = np.arange(n, dtype=float)
    y = np.cumsum(rng.standard_normal(n))
    y[rng.integers(0, n, 5)] = np.nan
    # reference: the blocks are computed from the data only
    monkeypatch.setattr(MinMaxPyramid, 'efficientBlockSize', staticmethod(lambda: 1 << 62))
    expected, item = _displayed_sequence(plot_widget, x, y, clipToView=clip)
    assert item._peakCache.pyramid is None
    monkeypatch.setattr(MinMaxPyramid, 'efficientBlockSize', staticmethod(lambda: 64))
    monkeypatch.setattr(pdi_module._PeakBlockCache, 'PYRAMID_MIN_VALUES', 0)
    shown, item = _displayed_sequence(plot_widget, x, y, clipToView=clip)
    assert item._peakCache.pyramid is not None
    assert item._peakCache.pyramidBlocks > 0
    assert len(shown) == len(expected)
    for (xs, ys, key), (xe, ye, key_e) in zip(shown, expected):
        assert key == key_e
        np.testing.assert_array_equal(xs, xe)
        np.testing.assert_array_equal(ys, ye)


def test_peak_display_matches_numpy_reduction(plot_widget, low_thresholds):
    rng = np.random.default_rng(2)
    n = 200_000
    y = np.cumsum(rng.standard_normal(n))
    shown, item = _displayed_sequence(plot_widget, np.arange(n, dtype=float), y,
                                      clipToView=True)
    for _, yd, (ds, first, end) in shown:
        complete = min(end, n // ds)
        ref_max, ref_min = _direct(y, first, complete, ds)
        np.testing.assert_array_equal(yd[0:2 * (complete - first):2], ref_max)
        np.testing.assert_array_equal(yd[1:2 * (complete - first):2], ref_min)


def _zoom(plot_widget, item, *ranges):
    for lo, hi in ranges:
        plot_widget.setXRange(lo, hi, padding=0)
        process_events()
        item.getData()


def test_pyramid_built_when_the_factor_changes(plot_widget, low_thresholds):
    n = 200_000
    item = plot_widget.plot(np.random.default_rng(3).normal(size=n), autoDownsample=True)
    plot_widget.getPlotItem().enableAutoRange(False)
    _zoom(plot_widget, item, (0, n))
    item.setData(item.getOriginalDataset()[1].copy())  # forget the initial auto-range
    _zoom(plot_widget, item, (0, n))
    # the factor did not change for the data: no pyramid
    assert item._peakCache.pyramid is None
    _zoom(plot_widget, item, (0, 0.8 * n))
    # the blocks computed for the previous factor read the data once: no pyramid yet
    assert item._peakCache.pyramid is None
    _zoom(plot_widget, item, (0, 0.6 * n))
    pyramid = item._peakCache.pyramid
    assert pyramid is not None and pyramid.y is item._datasetMapped.y
    _zoom(plot_widget, item, (0, 0.6 * n), (0, 0.9 * n))
    # shared by the caches of all factors
    assert item._peakCache.pyramid is pyramid


def test_pyramid_dropped_by_set_data(plot_widget, low_thresholds):
    rng = np.random.default_rng(4)
    n = 200_000
    item = plot_widget.plot(rng.normal(size=n), autoDownsample=True)
    plot_widget.getPlotItem().enableAutoRange(False)
    _zoom(plot_widget, item, (0, n), (0, 0.8 * n))
    old = item._peakCache.pyramid
    assert old is not None
    y = rng.normal(size=n)
    item.setData(y)
    assert item._peakCache is None
    _zoom(plot_widget, item, (0, 0.7 * n), (0, 0.9 * n), (0, 0.8 * n))
    pyramid = item._peakCache.pyramid
    assert pyramid is not None and pyramid is not old
    assert pyramid.y is item._datasetMapped.y
    np.testing.assert_array_equal(pyramid.y, y)
    # the same array modified in place and set again: the pyramid is rebuilt
    y[:] = rng.normal(size=n)
    item.setData(y)
    _zoom(plot_widget, item, (0, 0.6 * n), (0, 0.8 * n), (0, 0.7 * n))
    assert item._peakCache.pyramid is not pyramid
    ds, first, end = item._displayKey
    ref_max, _ = _direct(y, first, min(end, n // ds), ds)
    np.testing.assert_array_equal(item.getData()[1][0:2 * len(ref_max):2], ref_max)


def test_pyramid_follows_the_mapping(plot_widget, low_thresholds):
    n = 200_000
    y = np.random.default_rng(5).uniform(1.0, 100.0, n)
    item = plot_widget.plot(y, autoDownsample=True)
    plot_widget.getPlotItem().enableAutoRange(False)
    _zoom(plot_widget, item, (0, n), (0, 0.8 * n))
    assert item._peakCache.pyramid.y is item._dataset.y
    item.setLogMode(False, True)
    _zoom(plot_widget, item, (0, 0.7 * n), (0, 0.9 * n), (0, 0.8 * n))
    mapped = item._datasetMapped.y
    assert not np.shares_memory(mapped, y)
    assert item._peakCache.pyramid.y is mapped
    ds, first, end = item._displayKey
    ref_max, ref_min = _direct(mapped, first, min(end, n // ds), ds)
    yd = item.getData()[1]
    np.testing.assert_array_equal(yd[0:2 * len(ref_max):2], ref_max)
    np.testing.assert_array_equal(yd[1:2 * len(ref_min):2], ref_min)


def test_pyramid_extended_by_append_data(plot_widget, low_thresholds):
    rng = np.random.default_rng(6)
    n = 200_000
    y = rng.normal(size=n + 5000)
    item = plot_widget.plot(y[:n], autoDownsample=True)
    plot_widget.getPlotItem().enableAutoRange(False)
    _zoom(plot_widget, item, (0, n), (0, 0.8 * n))
    pyramid = item._peakCache.pyramid
    assert pyramid is not None
    item.appendData(y[n:])
    _zoom(plot_widget, item, (0, 0.7 * (n + 5000)), (0, n + 5000))
    assert item._peakCache.pyramid is pyramid
    assert pyramid.y is item._datasetMapped.y
    rebuilt = MinMaxPyramid(y)
    for level in range(MinMaxPyramid.BASE, MinMaxPyramid.BASE + len(rebuilt._len)):
        np.testing.assert_array_equal(pyramid.levelExtremes(level)[0],
                                      rebuilt.levelExtremes(level)[0])


def test_pyramid_dropped_by_clear(plot_widget, low_thresholds):
    n = 200_000
    item = plot_widget.plot(np.random.default_rng(7).normal(size=n), autoDownsample=True)
    plot_widget.getPlotItem().enableAutoRange(False)
    _zoom(plot_widget, item, (0, n), (0, 0.8 * n))
    assert item._peakCache.pyramid is not None
    item.clear()
    assert item._peakCache is None


def test_pyramid_not_built_for_small_data(plot_widget):
    n = 100_000  # below PYRAMID_MIN_VALUES
    item = plot_widget.plot(np.random.default_rng(8).normal(size=n), autoDownsample=True)
    plot_widget.getPlotItem().enableAutoRange(False)
    _zoom(plot_widget, item, (0, n), (0, 0.8 * n), (0, 0.6 * n))
    assert item._peakCache.pyramid is None


def test_pyramid_used_with_default_thresholds(plot_widget):
    n = 3_000_000
    y = np.cumsum(np.random.default_rng(9).standard_normal(n))
    item = plot_widget.plot(y, autoDownsample=True)
    plot_widget.getPlotItem().enableAutoRange(False)
    _zoom(plot_widget, item, (0, n), (0, 0.9 * n))
    cache = item._peakCache
    assert cache.ds >= MinMaxPyramid.efficientBlockSize()
    assert cache.pyramid is not None and cache.pyramidBlocks > 0
    ds, first, end = item._displayKey
    ref_max, ref_min = _direct(y, first, min(end, n // ds), ds)
    yd = item.getData()[1]
    np.testing.assert_array_equal(yd[0:2 * len(ref_max):2], ref_max)
    np.testing.assert_array_equal(yd[1:2 * len(ref_min):2], ref_min)


# --------------------------------------------------------------------------------------
# autoReduce
# --------------------------------------------------------------------------------------

def _unreduced(xd, yd, x, y):
    # the displayed arrays are the data itself (views of the arrays given)
    return (len(xd) == len(x) and np.shares_memory(xd, x)
            and len(yd) == len(y) and np.shares_memory(yd, y))


def _dense_line(n=1_000_000, seed=10):
    rng = np.random.default_rng(seed)
    return np.arange(n, dtype=float), np.cumsum(rng.standard_normal(n))


def test_auto_reduce_is_disabled_by_default(plot_widget):
    x, y = _dense_line()
    item = plot_widget.plot(x, y)
    assert item.opts['autoReduce'] is None
    plot_widget.setXRange(2e5, 4e5, padding=0)
    process_events()
    # nothing is clipped nor downsampled
    xd, yd = item.getData()
    assert _unreduced(xd, yd, x, y)


def test_auto_reduce_configuration_option():
    assert pg.getConfigOption('autoReduce') is None
    try:
        pg.setConfigOptions(autoReduce=10)
        assert pg.PlotDataItem().opts['autoReduce'] == 10
    finally:
        pg.setConfigOptions(autoReduce=None)
    assert pg.PlotDataItem().opts['autoReduce'] is None
    for value in (0, -1, math.inf, math.nan):
        with pytest.raises(ValueError):
            pg.setConfigOption('autoReduce', value)
        with pytest.raises(ValueError):
            pg.PlotDataItem().setAutoReduce(value)
        with pytest.raises(ValueError):
            pg.PlotDataItem([1, 2, 3], autoReduce=value)
    assert pg.getConfigOption('autoReduce') is None


@pytest.mark.parametrize('use_setter', [False, True])
def test_auto_reduce_matches_clip_and_peak(plot_widget, use_setter):
    x, y = _dense_line()
    if use_setter:
        item = plot_widget.plot(x, y)
        item.setAutoReduce(10)
    else:
        item = plot_widget.plot(x, y, autoReduce=10)
    reference = plot_widget.plot(x, y, clipToView=True, autoDownsample=True)
    plot_widget.getPlotItem().enableAutoRange(False)
    for lo, hi in ((2e5, 4e5), (2.1e5, 4.1e5), (0, 1e6), (5e5, 5.5e5)):
        plot_widget.setXRange(lo, hi, padding=0)
        process_events()
        xd, yd = item.getData()
        xr, yr = reference.getData()
        assert len(xd) < 0.1 * len(x)
        np.testing.assert_array_equal(xd, xr)
        np.testing.assert_array_equal(yd, yr)


def test_auto_reduce_below_the_threshold_keeps_the_data(plot_widget):
    x, y = _dense_line(n=2000)  # fewer than 10 points per pixel
    item = plot_widget.plot(x, y, autoReduce=10)
    plot_widget.setXRange(200, 400, padding=0)
    process_events()
    xd, yd = item.getData()
    assert _unreduced(xd, yd, x, y)


def test_auto_reduce_clips_without_downsampling_when_zoomed_in(plot_widget):
    x, y = _dense_line()
    item = plot_widget.plot(x, y, autoReduce=10)
    plot_widget.setXRange(5e5, 5e5 + 1000, padding=0)
    process_events()
    xd, yd = item.getData()
    ds, start, end = item._displayKey
    assert ds == 1
    np.testing.assert_array_equal(xd, x[start:end])
    np.testing.assert_array_equal(yd, y[start:end])
    assert xd[0] <= 5e5 and xd[-1] >= 5e5 + 1000


def test_auto_reduce_downsamples_with_x_auto_range(plot_widget):
    x, y = _dense_line()
    item = plot_widget.plot(x, y, autoReduce=10)
    process_events()
    assert plot_widget.getViewBox().autoRangeEnabled()[0]
    xd, yd = item.getData()
    # not clipped (the x range follows the data), but downsampled with 'peak'
    assert xd[0] == x[item._displayKey[0] // 2] and xd[-1] == x[-1]
    assert len(xd) < 0.1 * len(x)
    assert np.nanmax(yd) == y.max() and np.nanmin(yd) == y.min()


@pytest.mark.parametrize('case', ['decreasing', 'nan'])
def test_auto_reduce_requires_increasing_x(plot_widget, case):
    x, y = _dense_line()
    if case == 'decreasing':
        x = x.copy()
        x[500_000] = -1.0
    else:
        x = x.copy()
        x[10] = np.nan
    item = plot_widget.plot(x, y, autoReduce=10)
    plot_widget.setXRange(2e5, 4e5, padding=0)
    process_events()
    xd, yd = item.getData()
    assert _unreduced(xd, yd, x, y)


def test_auto_reduce_keeps_the_auto_range_bounds(plot_widget):
    x, y = _dense_line()
    item = plot_widget.plot(x, y, autoReduce=10)
    plain = pg.PlotDataItem(x, y)
    plot_widget.setXRange(2e5, 2.1e5, padding=0)
    process_events()
    assert item._autoReduceClipped
    assert len(item.getData()[0]) < len(x)
    for ax in (0, 1):
        assert item.dataBounds(ax) == pytest.approx(plain.dataBounds(ax))
    # the bounds of the visible part are still available to the auto-visible range
    lo, hi = item.dataBounds(1, orthoRange=(2e5, 2.1e5))
    visible = y[200_000:210_001]
    assert lo <= visible.min() and hi >= visible.max()
    assert hi - lo < y.max() - y.min()


def test_auto_reduce_follows_the_view(plot_widget):
    x, y = _dense_line()
    item = plot_widget.plot(x, y, autoReduce=10)
    plot_widget.getPlotItem().enableAutoRange(False)
    vb = plot_widget.getViewBox()
    plot_widget.setXRange(2e5, 4e5, padding=0)
    process_events()
    for _ in range(5):
        vb.translateBy(x=5e4)
        process_events()
        (lo, hi), _ = vb.viewRange()
        xd, _ = item.getData()
        assert xd[0] <= lo + 1000 and xd[-1] >= hi - 1000


def test_auto_reduce_checks_appended_points_only(plot_widget, monkeypatch):
    x, y = _dense_line()
    item = plot_widget.plot(x[:900_000], y[:900_000], autoReduce=10)
    plot_widget.setXRange(2e5, 4e5, padding=0)
    process_events()
    assert item._autoReduceClipped
    checked = []
    original = pdi_module._isNonDecreasing

    def counting(arr, *args, **kwargs):
        checked.append(len(arr))
        return original(arr, *args, **kwargs)

    monkeypatch.setattr(pdi_module, '_isNonDecreasing', counting)
    item.appendData(x[900_000:900_100], y[900_000:900_100])
    process_events()
    assert item._autoReduceClipped
    assert checked and max(checked) <= 101
    # a decreasing point disables the reduction
    item.appendData([0.0], [0.0])
    process_events()
    assert not item._autoReduceClipped
    assert len(item.getData()[0]) == 900_101


def test_auto_reduce_with_implicit_x_skips_the_order_check(plot_widget, monkeypatch):
    calls = []
    monkeypatch.setattr(pdi_module, '_isNonDecreasing',
                        lambda arr, *args, **kwargs: calls.append(len(arr)) or True)
    _, y = _dense_line()
    item = plot_widget.plot(y, autoReduce=10)
    plot_widget.setXRange(2e5, 4e5, padding=0)
    process_events()
    assert item._autoReduceClipped
    assert calls == []


def test_is_non_decreasing():
    check = pdi_module._isNonDecreasing
    assert check(np.array([]))
    assert check(np.array([1.0]))
    assert check(np.array([1, 1, 2, 2, 3]), chunk=2)
    assert not check(np.array([1, 2, 3, 2, 4]), chunk=2)
    assert not check(np.array([1, 2, 3, 4, 3]), chunk=2)
    assert not check(np.array([0.0, np.nan, 1.0]))
    values = np.arange(10_000.0)
    for chunk in (1, 7, 64, 1 << 20):
        assert check(values, chunk=chunk)
        values[-1] = 0.0
        assert not check(values, chunk=chunk)
        values[-1] = 9999.0
