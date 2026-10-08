"""
Performance regression tests for :class:`~pyqtgraph.PlotDataItem`.

The tests count calls and check cache states (see ``tests/perf_helpers.py``); they never
assert durations.
"""
import numpy as np
import pytest

import pyqtgraph as pg
from tests.perf_helpers import count_calls, process_events

app = pg.mkQApp()


@pytest.fixture
def plot_widget():
    pw = pg.PlotWidget()
    pw.resize(400, 300)
    pw.show()
    process_events()
    yield pw
    pw.close()


# --------------------------------------------------------------------------------------
# T1.2: no curve/scatter update on y pan/zoom when the dynamic range limit is not engaged
# --------------------------------------------------------------------------------------

def test_yrange_change_without_clipping_does_not_resend_data(plot_widget):
    y = np.random.default_rng(0).normal(size=10_000)
    item = plot_widget.plot(y, symbol='o', symbolSize=3)
    process_events()
    with count_calls(pg.PlotCurveItem, 'setData') as curve_calls, \
            count_calls(pg.ScatterPlotItem, 'setData') as scatter_calls:
        for k in range(10):
            plot_widget.setYRange(-5 - k, 5 + k, padding=0)
            process_events()
    assert curve_calls.count == 0
    assert scatter_calls.count == 0
    assert item.curve.isVisible() and item.scatter.isVisible()


def test_dynamic_range_limit_clips_extreme_values(plot_widget):
    y = np.zeros(101)
    y[50] = 1e12
    item = plot_widget.plot(y)
    plot_widget.setYRange(0, 1, padding=0)
    process_events()
    limit = item.opts['dynamicRangeLimit']
    _, y_disp = item.getData()
    # the 1e12 spike is limited to about `limit` view heights above the view
    assert y_disp.max() == pytest.approx(limit, rel=1e-6)
    assert item.curve.yData.max() == pytest.approx(limit, rel=1e-6)
    # the original data is untouched
    assert item.getOriginalDataset()[1].max() == 1e12


def test_dynamic_range_limit_engages_and_releases_with_zoom(plot_widget):
    y = np.zeros(101)
    y[50] = 1e12
    item = plot_widget.plot(y)
    plot_widget.setYRange(-1e11, 1.1e12, padding=0)
    process_events()
    # no clipping needed at this zoom level: the full data is displayed
    assert item.curve.yData.max() == 1e12

    # zoom in: clipping becomes necessary and must be applied
    plot_widget.setYRange(0, 1, padding=0)
    process_events()
    assert item.curve.yData.max() < 1e7

    # zoom further within the clipping regime: the clip level follows the view
    plot_widget.setYRange(0, 1e-3, padding=0)
    process_events()
    assert item.curve.yData.max() < 1e4

    # zoom out again: clipping is released, the original values are displayed
    plot_widget.setYRange(-1e11, 1.1e12, padding=0)
    process_events()
    assert item.curve.yData.max() == 1e12


def test_unchanged_display_data_is_not_resent(plot_widget):
    item = plot_widget.plot(np.arange(100.0), np.arange(100.0) ** 2)
    process_events()
    with count_calls(pg.PlotCurveItem, 'setData') as curve_calls:
        # no clipping, no downsampling: the displayed arrays are unchanged
        item.viewRangeChanged()
        item.updateItems(styleUpdate=False)
    assert curve_calls.count == 0
    with count_calls(pg.PlotCurveItem, 'setData') as curve_calls:
        # a style update is always forwarded
        item.setPen('r')
    assert curve_calls.count == 1
    with count_calls(pg.PlotCurveItem, 'setData') as curve_calls:
        # new data is always forwarded, even without style change
        item.setData(np.arange(100.0), np.arange(100.0))
    assert curve_calls.count == 1


# --------------------------------------------------------------------------------------
# T1.6: automatic downsampling without full-size copies, ds from the visible points
# --------------------------------------------------------------------------------------

@pytest.fixture
def isfinite_sizes(monkeypatch):
    """Record the size of every array passed to ``np.isfinite``."""
    sizes = []
    original = np.isfinite

    def recording_isfinite(arr, *args, **kwargs):
        sizes.append(np.size(arr))
        return original(arr, *args, **kwargs)

    monkeypatch.setattr(np, 'isfinite', recording_isfinite)
    return sizes


def _old_auto_ds(view_width: float, dx: float, px_width: float, factor: float) -> int:
    # reference: the downsampling factor computed before T1.6
    return int(max(1.0, abs(view_width / dx / (px_width * factor))))


def test_auto_downsample_does_not_copy_full_x(plot_widget, isfinite_sizes):
    n = 100_000
    x = np.linspace(0.0, 1.0, n)
    y = np.sin(x * 100)
    item = plot_widget.plot(dynamicRangeLimit=None)
    item.setDownsampling(auto=True, method='peak')
    plot_widget.setXRange(0.0, 1.0)
    process_events()
    isfinite_sizes.clear()
    item.setData(x, y)
    item.getData()  # the display update may be deferred to the next paint
    assert item._adsLastValue > 1
    assert all(size < n for size in isfinite_sizes)


def test_auto_downsample_factor_unchanged_on_uniform_x(plot_widget):
    n = 200_000
    x = np.arange(n, dtype=float)
    y = np.random.default_rng(0).normal(size=n)
    item = plot_widget.plot(x, y)
    item.setDownsampling(auto=True, method='peak')
    vb = plot_widget.getViewBox()
    factor = item.opts['autoDownsampleFactor']
    for clip in (False, True):
        item.setClipToView(clip)
        for k in (3, 7, 20):
            # place the exact factor in the middle between two integers
            view_width = (k + 0.5) * vb.width() * factor
            plot_widget.setXRange(5000.0, 5000.0 + view_width, padding=0)
            process_events()
            item.getData()
            assert item._adsLastValue == _old_auto_ds(view_width, 1.0, vb.width(), factor)


def test_auto_downsample_with_gap_bounds_displayed_points(plot_widget):
    # two dense blocks separated by a gap as wide as the data, as with market data
    # without nights or week-ends: the global mean spacing overestimates the spacing
    x_gap = np.concatenate([np.arange(0.0, 50_000.0), np.arange(100_000.0, 150_000.0)])
    x_uniform = np.arange(50_000.0, 150_000.0)
    y = np.random.default_rng(0).normal(size=len(x_gap))
    displayed = []
    for x in (x_uniform, x_gap):
        item = plot_widget.plot(x, y)
        item.setDownsampling(auto=True, method='peak')
        item.setClipToView(True)
        plot_widget.setXRange(100_000.0, 150_000.0, padding=0)
        process_events()
        displayed.append(len(item.getData()[0]))
        factor = item.opts['autoDownsampleFactor']
        plot_widget.removeItem(item)
    width = plot_widget.getViewBox().width()
    # about `factor` peaks per pixel (2 points per peak), as without the gap
    assert displayed[1] <= displayed[0] + 4
    assert displayed[1] <= 2 * 1.1 * factor * width


def test_finite_index_range():
    from pyqtgraph.graphicsItems.PlotDataItem import _finiteIndexRange

    a = np.arange(1000.0)
    assert _finiteIndexRange(a) == (0, 999)
    a[:300] = np.nan
    a[-500:] = np.inf
    assert _finiteIndexRange(a) == (300, 499)
    a[:] = np.nan
    assert _finiteIndexRange(a) is None
    a[10] = 1.0
    assert _finiteIndexRange(a) is None
    a[20] = 2.0
    assert _finiteIndexRange(a) == (10, 20)
    assert _finiteIndexRange(np.array([1.0])) is None
    assert _finiteIndexRange(np.arange(5)) == (0, 4)


# --------------------------------------------------------------------------------------
# T1.11: API bugs that disabled optimizations
# --------------------------------------------------------------------------------------

def test_clip_to_view_while_parenting_does_not_raise(plot_widget):
    # the view is the PlotWidget while the item is being parented; this raised
    # "AttributeError: autoRangeEnabled" inside a Qt virtual method
    item = plot_widget.plot(np.arange(100.0), clipToView=True, autoDownsample=True)
    process_events()
    assert len(item.getData()[0]) == 100


def test_clip_to_view_in_graphics_view_without_viewbox():
    view = pg.GraphicsView()
    item = pg.PlotDataItem(np.arange(100.0), clipToView=True)
    view.addItem(item)
    try:
        # no ViewBox to clip to: the full data is displayed
        assert len(item.getData()[0]) == 100
    finally:
        view.close()


def test_data_bounds_upper_limit_with_step_mode_and_symbols():
    x = np.arange(11.0)
    y = np.arange(10.0)
    item = pg.PlotDataItem(x, y, stepMode='center', symbol='o')
    # the curve spans the step boundaries, the symbols sit at the step centers
    assert item.dataBounds(0) == (0.0, 10.0)
    assert item.dataBounds(1) == (0.0, 9.0)


# --------------------------------------------------------------------------------------
# T2.2: a single O(N) pass per setData
# --------------------------------------------------------------------------------------

@pytest.fixture
def full_scans(monkeypatch):
    """Count numpy reductions and finiteness tests applied to arrays of a given size."""
    calls: dict[str, int] = {}
    size = {'n': None}

    def wrap(name):
        original = getattr(np, name)

        def recording(arr, *args, **kwargs):
            if size['n'] is not None and np.size(arr) == size['n']:
                calls[name] = calls.get(name, 0) + 1
            return original(arr, *args, **kwargs)

        monkeypatch.setattr(np, name, recording)

    for name in ('isfinite', 'min', 'max', 'nanmin', 'nanmax'):
        wrap(name)

    def start(n):
        calls.clear()
        size['n'] = n
        return calls

    return start


def test_set_data_scans_finite_data_once(plot_widget, full_scans):
    n = 200_003  # a size no other array of the test has
    rng = np.random.default_rng(0)
    item = plot_widget.plot()
    process_events()
    calls = full_scans(n)
    item.setData(np.arange(n, dtype=float), rng.normal(size=n))
    process_events()  # auto-range, bounding rect and paint
    # one min and one max per axis, no finiteness test of the full arrays
    assert calls == {'min': 2, 'max': 2}
    assert item.curve.opts['connect'] == 'all'
    assert item.curve.opts['skipFiniteCheck'] is True


def test_finite_data_connects_all_from_first_set_data():
    item = pg.PlotDataItem(np.arange(10.0), np.arange(10.0))
    assert item.curve.opts['connect'] == 'all'
    assert item.curve.opts['skipFiniteCheck'] is True


def test_non_finite_data_still_interrupts_curve():
    y = np.arange(10.0)
    y[5] = np.nan
    item = pg.PlotDataItem(np.arange(10.0), y)
    assert item.curve.opts['connect'] == 'finite'
    assert item.curve.opts['skipFiniteCheck'] is False
    y = np.arange(10.0)
    y[5] = np.inf
    item.setData(np.arange(10.0), y)
    assert item.curve.opts['connect'] == 'finite'
    # bounds ignore the non-finite value
    assert item.dataBounds(1) == (0.0, 9.0)
    path = item.curve.getPath()
    # the curve is interrupted: two move-to elements
    moves = [i for i in range(path.elementCount()) if path.elementAt(i).isMoveTo()]
    assert len(moves) == 2


def test_curve_bounds_reuse_precomputed_values():
    x = np.linspace(-3.0, 7.0, 1000)
    y = np.sin(x) * 5
    item = pg.PlotDataItem(x, y)
    with count_calls(pg.PlotCurveItem, '_computeDataBounds') as scans:
        assert item.curve.dataBounds(0) == (-3.0, 7.0)
        assert item.curve.dataBounds(1) == (float(y.min()), float(y.max()))
    assert scans.count == 0
    # a partial range is still computed from the data
    with count_calls(pg.PlotCurveItem, '_computeDataBounds') as scans:
        item.curve.dataBounds(1, frac=0.5)
        item.curve.dataBounds(1, orthoRange=(0.0, 1.0))
    assert scans.count == 2
    # the same values as computed from the data
    reference = pg.PlotCurveItem(x, y)
    for ax in (0, 1):
        assert item.curve.dataBounds(ax) == reference.dataBounds(ax)


def test_curve_bounds_hint_includes_fill_level_and_pen():
    x = np.arange(10.0)
    pen = pg.mkPen(width=2, cosmetic=False)
    item = pg.PlotDataItem(x, x, fillLevel=-5.0, brush='r', pen=pen)
    reference = pg.PlotCurveItem(x, x, fillLevel=-5.0, brush='r', pen=pen)
    for ax in (0, 1):
        assert item.curve.dataBounds(ax) == reference.dataBounds(ax)


def test_dynamic_range_limit_uses_displayed_data(plot_widget):
    # the extreme value lies outside of the clipped x range: no clipping is needed
    x = np.arange(1000.0)
    y = np.zeros(1000)
    y[10] = 1e12
    item = plot_widget.plot(x, y, clipToView=True)
    plot_widget.setRange(xRange=(500, 600), yRange=(0, 1), padding=0)
    process_events()
    assert not item._drlClipActive
    x_disp, y_disp = item.getData()
    assert x_disp[0] >= 499 and y_disp.max() == 0
    # back to the extreme value: clipping is applied
    plot_widget.setXRange(0, 100, padding=0)
    process_events()
    assert item._drlClipActive
    assert item.getData()[1].max() < 1e7


# --------------------------------------------------------------------------------------
# T2.3: incremental streaming, aligned and cached peak blocks
# --------------------------------------------------------------------------------------

def _stream_pair(plot_widget, **opts):
    streamed = plot_widget.plot(**opts)
    reference = plot_widget.plot(**opts)
    return streamed, reference


@pytest.mark.parametrize('opts', [
    {},
    {'autoDownsample': True},
    {'downsample': 7},
    {'downsample': 7, 'downsampleMethod': 'mean'},
    {'downsample': 7, 'downsampleMethod': 'subsample'},
    {'autoDownsample': True, 'clipToView': True},
    {'symbol': 'o', 'downsample': 3},
])
def test_append_data_matches_set_data(plot_widget, opts):
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.uniform(0.5, 1.5, 5000))
    y = rng.normal(size=5000)
    y[1234] = np.nan
    streamed, reference = _stream_pair(plot_widget, **opts)
    if opts.get('clipToView'):
        plot_widget.setXRange(x[1000], x[3000], padding=0)
    streamed.setData(x[:100], y[:100])
    end = 100
    for size in (1, 1, 5, 300, 1, 2000, 7, 2585):
        streamed.appendData(x[end:end + size], y[end:end + size])
        end += size
        reference.setData(x[:end], y[:end])
        reference._adsLastValue = streamed._adsLastValue  # same hysteresis state
        process_events()
        for got, expected in zip(streamed.getOriginalDataset(), reference.getOriginalDataset()):
            np.testing.assert_array_equal(got, expected)
        for got, expected in zip(streamed.getData(), reference.getData()):
            np.testing.assert_array_equal(got, expected)
        for ax in (0, 1):
            assert streamed.dataBounds(ax) == reference.dataBounds(ax)
    assert end == 5000


def test_append_data_scalars_and_implicit_x():
    item = pg.PlotDataItem()
    item.appendData([1.0, 2.0])
    item.appendData(3.0)
    item.appendData(y=[4.0, 5.0])
    x, y = item.getOriginalDataset()
    np.testing.assert_array_equal(x, np.arange(5))
    np.testing.assert_array_equal(y, [1.0, 2.0, 3.0, 4.0, 5.0])
    item.setData([0.0, 1.0], [2.0, 3.0])
    with pytest.raises(TypeError):
        item.appendData(4.0)
    with pytest.raises(ValueError):
        item.appendData([2.0, 3.0], [4.0])
    item.appendData(2.0, 4.0)
    np.testing.assert_array_equal(item.getOriginalDataset()[1], [2.0, 3.0, 4.0])


def test_append_data_with_mapping_falls_back_to_set_data():
    item = pg.PlotDataItem(np.arange(1.0, 11.0), np.arange(1.0, 11.0))
    item.setLogMode(False, True)
    item.appendData([11.0, 12.0], [11.0, 12.0])
    x, y = item.getData()
    np.testing.assert_allclose(y, np.log10(np.arange(1.0, 13.0)))


def test_append_data_is_amortized_constant():
    item = pg.PlotDataItem(np.arange(10.0), np.zeros(10))
    buffers = set()
    for k in range(10, 5000):
        item.appendData(float(k), float(k))
        buffers.add(id(item.getOriginalDataset()[1].base))
    # the buffer capacity grows geometrically: few reallocations
    assert len(buffers) <= 4
    np.testing.assert_array_equal(item.yData[10:], np.arange(10.0, 5000.0))
    # bounds are updated from the appended points only
    with count_calls(pg.graphicsItems.PlotDataItem.PlotDataset, '_getArrayBounds') as scans:
        item.appendData(5000.0, -1.0)
    assert scans.count == 2
    assert item.dataRect() == pg.QtCore.QRectF(
        pg.QtCore.QPointF(0.0, -1.0), pg.QtCore.QPointF(5000.0, 4999.0)
    )


def test_peak_blocks_are_not_recomputed_while_streaming(plot_widget):
    item = plot_widget.plot(downsample=10, downsampleMethod='peak')
    item.setData(np.random.default_rng(0).normal(size=100_000))
    process_events()
    cache = item._peakCache
    assert cache.computedBlocks == 10_000
    for k in range(100):
        item.appendData(float(k))
        process_events()
    # only the blocks completed by the appended points were computed
    assert item._peakCache is cache
    assert cache.computedBlocks == 10_010
    _, y = item.getData()
    # complete blocks only
    assert len(y) == 2 * 10_010


def test_peak_blocks_do_not_move_with_the_view(plot_widget):
    rng = np.random.default_rng(0)
    x = np.arange(100_000.0)
    y = rng.normal(size=100_000)
    item = plot_widget.plot(x, y, downsample=10, downsampleMethod='peak', clipToView=True)
    shown = []
    for offset in (0.0, 3.3, 6.6):  # shifts by a fraction of a block
        plot_widget.setXRange(50_000.0 + offset, 60_000.0 + offset, padding=0)
        process_events()
        xd, yd = item.getData()
        keep = (xd >= 50_100) & (xd <= 59_900)
        shown.append((xd[keep], yd[keep]))
    for xd, yd in shown[1:]:
        np.testing.assert_array_equal(xd, shown[0][0])
        np.testing.assert_array_equal(yd, shown[0][1])
    # panning computes only the blocks entering the view
    computed = item._peakCache.computedBlocks
    plot_widget.setXRange(50_100.0, 60_100.0, padding=0)
    process_events()
    item.getData()
    assert item._peakCache.computedBlocks - computed <= 11


# --------------------------------------------------------------------------------------
# T2.4: the displayed data is computed once per frame
# --------------------------------------------------------------------------------------

def _streaming_data(n: int = 100_000):
    rng = np.random.default_rng(0)
    return np.arange(n + 100, dtype=float), np.cumsum(rng.standard_normal(n + 100))


def test_streaming_with_clip_to_view_updates_once_per_frame(plot_widget):
    x, y = _streaming_data()
    item = plot_widget.plot(clipToView=True, autoDownsample=True)
    plot_widget.getPlotItem().enableAutoRange(x=False, y=True)

    def frame(k):
        item.setData(x[:k], y[:k])
        plot_widget.setXRange(x[k - 5000], x[k - 1], padding=0)
        process_events()

    for k in range(100_000, 100_003):
        frame(k)
    with count_calls(pg.PlotDataItem, 'updateItems') as updates:
        for k in range(100_003, 100_023):
            frame(k)
    assert updates.count == 20
    # the displayed data follows the view
    assert item.getData()[0][-1] == x[100_022 - 1]


def test_streaming_with_auto_range_updates_once_per_frame(plot_widget):
    x, y = _streaming_data()
    item = plot_widget.plot(autoDownsample=True)
    for k in range(100_000, 100_003):
        item.setData(x[:k], y[:k])
        process_events()
    with count_calls(pg.PlotDataItem, 'updateItems') as updates, \
            count_calls(pg.PlotCurveItem, 'paint') as paints:
        for k in range(100_003, 100_023):
            item.setData(x[:k], y[:k])
            process_events()
    assert updates.count == 20
    # the deferred update is applied before the paint, not during it
    assert paints.count <= 21


def test_get_data_after_deferred_update_is_synchronous(plot_widget):
    x, y = _streaming_data(10_000)
    item = plot_widget.plot(x[:5000], y[:5000], autoDownsample=True)
    process_events()
    item.setData(x, y * 2)
    # nothing was computed yet, but all accessors are consistent with the new data
    assert item._displayDirty
    np.testing.assert_array_equal(item.getOriginalDataset()[1], y * 2)
    x_disp, y_disp = item.getData()
    assert x_disp[-1] > 5000
    np.testing.assert_array_equal(item.curve.yData, y_disp)
    assert not item._displayDirty


def test_curve_property_applies_deferred_update(plot_widget):
    x, y = _streaming_data(1000)
    item = plot_widget.plot(x[:500], y[:500], clipToView=True)
    process_events()
    item.setData(x, y)
    assert len(item.curve.xData) == len(item.getData()[0])
    lower = plot_widget.plot(x[:500], y[:500] - 1, clipToView=True)
    fill = pg.FillBetweenItem(item, lower)
    plot_widget.addItem(fill)
    lower.setData(x, y - 1)  # FillBetweenItem reads the curve paths synchronously
    assert fill.path().boundingRect().right() == pytest.approx(item.curve.xData[-1])


def test_auto_range_converges_in_one_frame(plot_widget):
    x, y = _streaming_data(10_000)
    item = plot_widget.plot(x[:100], y[:100], autoDownsample=True)
    process_events()
    item.setData(x, y * 10)
    process_events()
    (xmin, xmax), (ymin, ymax) = plot_widget.getViewBox().viewRange()
    assert xmin <= x[0] and xmax >= x[-1]
    assert ymin <= np.min(y * 10) and ymax >= np.max(y * 10)


@pytest.mark.parametrize('process', [True, False])
def test_deferred_update_renders_identically(process):
    x, y = _streaming_data(20_000)

    def make(data_x, data_y):
        pw = pg.PlotWidget()
        pw.resize(300, 200)
        item = pw.plot(data_x, data_y, autoDownsample=True, clipToView=True, pen='y')
        pw.setRange(xRange=(5000, 15000), yRange=(y.min(), y.max()), padding=0)
        pw.show()
        process_events()
        return pw, item

    pw, item = make(x[:10_000], y[:10_000])
    reference, reference_item = make(x, y)
    # the downsampling factor depends on the final widget size
    reference_item.setData(x, y)
    process_events()
    try:
        item.setData(x, y)
        if process:
            process_events()
        # without event processing, the render itself applies the deferred update
        image = pw.grab().toImage()
        expected = reference.grab().toImage()
        assert image == expected
    finally:
        pw.close()
        reference.close()


def test_update_is_immediate_without_view_dependency():
    # outside of a scene, or without clipToView/autoDownsample, nothing is deferred
    item = pg.PlotDataItem(autoDownsample=True)
    with count_calls(pg.PlotCurveItem, 'setData') as calls:
        item.setData(np.arange(10.0))
    assert calls.count == 1 and not item._displayDirty


# --------------------------------------------------------------------------------------
# T2.5: styles are only forwarded when they changed (regression tests for #1653)
# --------------------------------------------------------------------------------------

def _scatter_colors(item: pg.PlotDataItem) -> list[str]:
    return [p.brush().color().name() for p in item.scatter.points()]


def _scatter_symbols(item: pg.PlotDataItem) -> list[str]:
    return [p.symbol() for p in item.scatter.points()]


def test_per_point_scatter_styles_kept_with_same_length():
    brushes = [pg.mkBrush(c) for c in ('r', 'g', 'b')]
    item = pg.PlotDataItem([0, 1, 2], [3, 4, 5], symbol=['o', 's', 't'],
                           symbolBrush=brushes, symbolSize=[5, 6, 7], data=['a', 'b', 'c'])
    expected = ['#ff0000', '#00ff00', '#0000ff']
    assert _scatter_colors(item) == expected
    item.setData([0, 1, 2], [5, 4, 3])
    assert _scatter_colors(item) == expected
    assert _scatter_symbols(item) == ['o', 's', 't']
    assert [p.size() for p in item.scatter.points()] == [5, 6, 7]
    assert [p.data() for p in item.scatter.points()] == ['a', 'b', 'c']
    np.testing.assert_array_equal(item.scatter.getData()[1], [5, 4, 3])


def test_per_point_scatter_styles_with_different_length():
    brushes = [pg.mkBrush(c) for c in ('r', 'g', 'b')]
    item = pg.PlotDataItem([0, 1, 2], [3, 4, 5], symbolBrush=brushes)
    # new per-point styles given with data of another length are applied
    item.setData([0, 1, 2, 3], [5, 4, 3, 2], symbolBrush=brushes + [pg.mkBrush('y')])
    assert _scatter_colors(item) == ['#ff0000', '#00ff00', '#0000ff', '#ffff00']
    # and kept by a following update of the same length
    item.setData([0, 1, 2, 3], [1, 2, 3, 4])
    assert _scatter_colors(item) == ['#ff0000', '#00ff00', '#0000ff', '#ffff00']
    # per-point styles that do not match the new length are rejected, as before
    with pytest.raises(Exception):
        item.setData([0, 1], [1, 2])


def test_uniform_scatter_style_kept_with_new_data():
    item = pg.PlotDataItem([0, 1, 2], [3, 4, 5], symbol='s', symbolBrush='r', symbolSize=4)
    item.setData([0, 1, 2, 3, 4], [5, 4, 3, 2, 1])
    assert _scatter_colors(item) == ['#ff0000'] * 5
    assert _scatter_symbols(item) == ['s'] * 5
    item.setSymbolBrush('g')
    item.setData([0, 1], [5, 4])
    assert _scatter_colors(item) == ['#00ff00'] * 2


def test_streaming_does_not_forward_unchanged_styles():
    item = pg.PlotDataItem(np.arange(10.0), pen='r', fillLevel=0.0, brush='b', symbol='o')
    with count_calls(pg.PlotCurveItem, 'setPen') as pens, \
            count_calls(pg.PlotCurveItem, 'setBrush') as brushes, \
            count_calls(pg.PlotCurveItem, 'setFillLevel') as levels, \
            count_calls(pg.PlotCurveItem, 'setShadowPen') as shadows, \
            count_calls(pg.ScatterPlotItem, 'setBrush') as symbol_brushes:
        for k in range(10):
            item.setData(np.arange(10.0) * k)
        item.appendData(10.0)
    assert (pens.count, brushes.count, levels.count, shadows.count) == (0, 0, 0, 0)
    assert symbol_brushes.count == 0
    # the curve keeps its style
    assert item.curve.opts['pen'].color().name() == '#ff0000'
    assert item.curve.opts['fillLevel'] == 0.0
    # a style change is forwarded
    with count_calls(pg.PlotCurveItem, 'setPen') as pens:
        item.setPen('g')
    assert pens.count == 1
    assert item.curve.opts['pen'].color().name() == '#00ff00'


def test_style_set_before_data_is_applied_with_data():
    item = pg.PlotDataItem()
    item.setPen('r')
    item.setSymbol('t')
    item.setData([1.0, 2.0, 3.0])
    assert item.curve.opts['pen'].color().name() == '#ff0000'
    assert _scatter_symbols(item) == ['t'] * 3


def test_connect_follows_the_data_without_style_update():
    item = pg.PlotDataItem(np.arange(10.0))
    assert item.curve.opts['connect'] == 'all'
    y = np.arange(10.0)
    y[3] = np.nan
    item.setData(y)
    assert item.curve.opts['connect'] == 'finite'
    assert item.curve.opts['skipFiniteCheck'] is False
    item.setData(np.arange(10.0))
    assert item.curve.opts['connect'] == 'all'
    item.setData(np.arange(10.0), connect='pairs')
    item.setData(np.arange(10.0) + 1)
    assert item.curve.opts['connect'] == 'pairs'
