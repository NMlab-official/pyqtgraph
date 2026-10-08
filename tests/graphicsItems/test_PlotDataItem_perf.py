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
