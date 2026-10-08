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
