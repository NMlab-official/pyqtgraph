"""
Performance regression tests for :class:`~pyqtgraph.PlotItem`.

The tests count calls and check cache states (see ``tests/perf_helpers.py``); they never
assert durations.
"""
import numpy as np

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
