"""
Performance regression tests for :class:`~pyqtgraph.PlotItem`.

The tests count calls and check cache states (see ``tests/perf_helpers.py``); they never
assert durations.
"""
import numpy as np
import pytest

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from tests.perf_helpers import count_calls

app = pg.mkQApp()


def _param_list_state(plot: pg.PlotItem) -> list[tuple[str, bool]]:
    widget = plot.ctrl.avgParamList
    return [
        (widget.item(i).text(), widget.item(i).checkState() == QtCore.Qt.CheckState.Checked)
        for i in range(widget.count())
    ]


# --------------------------------------------------------------------------------------
# T1.9: the averaging parameter list is not rebuilt for every added curve
# --------------------------------------------------------------------------------------

def test_add_item_does_not_rebuild_param_list():
    plot = pg.PlotItem()
    data = np.arange(10.0)
    with count_calls(pg.PlotItem, 'updateParamList') as rebuilds:
        for k in range(50):
            plot.plot(data + k, params={'k': k, ('group', 'id'): k % 3})
    assert rebuilds.count == 0
    assert _param_list_state(plot) == [('k', False), ('group.id', False)]


def test_incremental_param_list_matches_rebuild():
    plot = pg.PlotItem()
    plot.paramList['b'] = True  # check state restored from a saved state
    data = np.arange(10.0)
    plot.plot(data)
    plot.plot(data, params={'a': 1})
    plot.addItem(pg.PlotCurveItem(data), params={'c': 0, 'a': 2})
    plot.plot(data, params={'b': 1, ('x', 'y'): 2})
    plot.plot(data, params={'a': 3, 'b': 4})
    incremental = _param_list_state(plot)
    plot.updateParamList()
    assert incremental == _param_list_state(plot)
    assert incremental == [('a', False), ('c', False), ('b', True), ('x.y', False)]
    assert plot.paramList == {'a': False, 'c': False, 'b': True, 'x.y': False}


def test_remove_item_param_list():
    plot = pg.PlotItem()
    data = np.arange(10.0)
    plain = [plot.plot(data) for _ in range(5)]
    tagged = plot.plot(data, params={'only': 1})
    with count_calls(pg.PlotItem, 'updateParamList') as rebuilds:
        for item in plain:
            plot.removeItem(item)
    # curves without parameters leave the list unchanged
    assert rebuilds.count == 0
    assert _param_list_state(plot) == [('only', False)]
    plot.removeItem(tagged)
    assert _param_list_state(plot) == []


# --------------------------------------------------------------------------------------
# T1.11: options set on a data item are not overwritten by PlotItem.addItem
# --------------------------------------------------------------------------------------

def _display_opts(item: pg.PlotDataItem) -> tuple:
    return tuple(item.opts[k] for k in
                 ('downsample', 'autoDownsample', 'downsampleMethod', 'clipToView'))


def test_plot_keeps_explicit_clip_and_downsampling_options():
    plot = pg.PlotItem()
    y = np.arange(100.0)
    item = plot.plot(y, clipToView=True, autoDownsample=True)
    assert _display_opts(item) == (1, True, 'peak', True)
    item = plot.plot(y, downsample=4, downsampleMethod='mean')
    assert _display_opts(item) == (4, False, 'mean', False)


def test_plot_defaults_apply_to_unset_options():
    plot = pg.PlotItem()
    plot.setDownsampling(ds=3, auto=True, mode='subsample')
    plot.setClipToView(True)
    y = np.arange(100.0)
    assert _display_opts(plot.plot(y)) == (3, True, 'subsample', True)
    # explicit values win, including values equal to the item defaults
    item = plot.plot(y, autoDownsample=False, clipToView=False)
    assert _display_opts(item) == (3, False, 'subsample', False)
    # options set through the setters before adding the item are kept as well
    item = pg.PlotDataItem(y)
    item.setClipToView(False)
    item.setDownsampling(ds=2)
    plot.addItem(item)
    assert _display_opts(item) == (2, True, 'peak', False)


def test_plot_menu_still_applies_to_all_curves():
    plot = pg.PlotItem()
    y = np.arange(100.0)
    items = [plot.plot(y, clipToView=True), plot.plot(y)]
    plot.ctrl.clipToViewCheck.setChecked(True)
    plot.ctrl.clipToViewCheck.setChecked(False)
    assert [item.opts['clipToView'] for item in items] == [False, False]


# --------------------------------------------------------------------------------------
# Context-menu actions with data items other than PlotDataItem
# --------------------------------------------------------------------------------------

class _Candles(pg.GraphicsObject):
    """A data item with none of the optional methods (no name, getData, clear...)."""

    def implements(self, interface=None):
        return interface == 'plotData' if interface is not None else ['plotData']

    def boundingRect(self):
        return QtCore.QRectF(0, 0, 1, 1)

    def paint(self, p, *args):
        pass

    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        return (0.0, 1.0)


def _mixed_plot():
    plot = pg.PlotItem()
    x = np.arange(1.0, 11.0)
    data_item = plot.plot(x, x)
    others = [
        pg.PlotCurveItem(x, x),
        pg.ScatterPlotItem(x, x),
        pg.BarGraphItem(x=x, height=x, width=0.5),
        _Candles(),
    ]
    for item in others:
        plot.addItem(item)
    return plot, data_item, others


@pytest.mark.parametrize('action', [
    'downsample', 'clipToView', 'fft', 'alpha', 'average', 'log', 'subtractMean',
    'derivative', 'phasemap', 'restoreState',
])
def test_menu_actions_with_other_data_items(action):
    plot, data_item, others = _mixed_plot()
    ctrl = plot.ctrl
    if action == 'downsample':
        plot.setDownsampling(ds=3, auto=True, mode='mean')
        assert (data_item.opts['downsample'], data_item.opts['autoDownsample'],
                data_item.opts['downsampleMethod']) == (3, True, 'mean')
    elif action == 'clipToView':
        ctrl.clipToViewCheck.setChecked(True)
        assert data_item.opts['clipToView']
    elif action == 'fft':
        ctrl.fftCheck.setChecked(True)
        assert data_item.opts['fftMode']
    elif action == 'alpha':
        ctrl.alphaGroup.setChecked(True)
        ctrl.alphaSlider.setValue(ctrl.alphaSlider.maximum() // 2)
        assert data_item.opacity() < 1.0
        assert all(item.opacity() == 1.0 for item in others)
    elif action == 'average':
        ctrl.averageGroup.setChecked(True)
        # averages of the items providing x and y data (bars included)
        assert len(plot.avgCurves) == 1
        n, average = next(iter(plot.avgCurves.values()))
        assert n == 4  # PlotDataItem, PlotCurveItem, ScatterPlotItem, BarGraphItem
        np.testing.assert_array_equal(average.yData, np.arange(1.0, 11.0))
    elif action == 'log':
        ctrl.logYCheck.setChecked(True)
        assert data_item.opts['logMode'] == [False, True]
    elif action == 'subtractMean':
        ctrl.subtractMeanCheck.setChecked(True)
        assert data_item.opts['subtractMeanMode']
    elif action == 'derivative':
        ctrl.derivativeCheck.setChecked(True)
        assert data_item.opts['derivativeMode']
    elif action == 'phasemap':
        ctrl.phasemapCheck.setChecked(True)
        assert data_item.opts['phasemapMode']
    elif action == 'restoreState':
        state = plot.saveState()
        state['fftCheck'] = True
        state['clipToViewCheck'] = True
        plot.restoreState(state)
        assert data_item.opts['fftMode'] and data_item.opts['clipToView']
    # the other items are still plotted
    assert all(item in plot.curves for item in others)


def test_max_traces_with_other_data_items():
    plot, data_item, others = _mixed_plot()
    ctrl = plot.ctrl
    ctrl.maxTracesSpin.setValue(2)
    ctrl.maxTracesCheck.setChecked(True)
    # the two most recent data items are visible
    assert [item.isVisible() for item in [data_item] + others] == [
        False, False, False, True, True]
    ctrl.forgetTracesCheck.setChecked(True)
    # the hidden items are removed, whether they can be cleared or not
    assert plot.curves == others[-2:]


def test_write_csv_with_other_data_items(tmp_path):
    plot, data_item, others = _mixed_plot()
    path = tmp_path / 'data.csv'
    plot.writeCsv(str(path))
    first = path.read_text().splitlines()[0].rstrip(',').split(',')
    # x and y of the PlotDataItem, PlotCurveItem, ScatterPlotItem and BarGraphItem
    assert [float(v) for v in first] == [1.0, 1.0] * 4


def test_average_of_curves_excludes_the_average_itself():
    plot = pg.PlotItem()
    x = np.arange(10.0)
    plot.plot(x, x)
    plot.plot(x, 3 * x)
    plot.ctrl.averageGroup.setChecked(True)
    (n, average), = plot.avgCurves.values()
    assert n == 2
    np.testing.assert_allclose(average.yData, 2 * x)
    # a curve added afterwards gets the right weight
    plot.plot(x, 6 * x)
    (n, average), = plot.avgCurves.values()
    assert n == 3
    np.testing.assert_allclose(average.yData, 10 / 3 * x)
