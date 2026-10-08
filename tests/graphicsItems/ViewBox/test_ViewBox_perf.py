"""
Deterministic performance tests of ViewBox: auto-range requests and the per-item
bounds cache of ``childrenBounds``.
"""
import numpy as np
import pytest

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from tests.perf_helpers import count_calls, process_events, show_and_wait

app = pg.mkQApp()


@pytest.fixture
def plot():
    pw = pg.PlotWidget()
    pw.resize(400, 300)
    show_and_wait(pw)
    yield pw
    pw.close()


def test_ignoreBounds_item_moves_do_not_autorange(plot):
    plot.plot(np.arange(100.0))
    line = pg.InfiniteLine(pos=10, angle=90)
    plot.addItem(line, ignoreBounds=True)
    vb = plot.getViewBox()
    process_events()
    assert vb.autoRangeEnabled() == [True, True]
    with count_calls(pg.ViewBox, 'updateAutoRange') as autoRanges, \
            count_calls(vb, 'queueUpdateAutoRange') as queued:
        for i in range(100):
            line.setPos(i)
            process_events(1)
    assert queued.count == 0
    assert autoRanges.count == 0


def test_bounded_child_change_triggers_autorange(plot):
    # The PlotCurveItem reports its bounds change; the ViewBox walks up to the
    # PlotDataItem that was added to it.
    item = plot.plot(np.arange(10.0))
    vb = plot.getViewBox()
    process_events()
    with count_calls(vb, 'queueUpdateAutoRange') as queued:
        item.setData(np.arange(10.0) * 100)
    assert queued.count >= 1
    process_events()
    assert vb.viewRange()[1][1] >= 900


def test_childrenBounds_second_call_uses_cache(plot):
    vb = plot.getViewBox()
    lines = [pg.InfiniteLine(pos=k, angle=0) for k in range(20)]
    for line in lines:
        plot.addItem(line)
    plot.plot(np.arange(50.0))
    first = vb.childrenBounds()
    with count_calls(pg.InfiniteLine, 'dataBounds') as lineBounds, \
            count_calls(pg.PlotDataItem, 'dataBounds') as curveBounds:
        assert vb.childrenBounds() == first
    assert lineBounds.count == 0
    assert curveBounds.count == 0

    lines[3].setPos(100)
    with count_calls(pg.InfiniteLine, 'dataBounds') as lineBounds:
        bounds = vb.childrenBounds()
    assert lineBounds.count == 2  # x and y bounds of the moved line only
    assert bounds[1][1] == 100

    vb._itemBoundsCache.clear()
    assert vb.childrenBounds() == bounds


def test_childrenBounds_cache_key(plot):
    vb = plot.getViewBox()
    plot.plot(np.arange(100.0))
    full = vb.childrenBounds()
    with count_calls(pg.PlotDataItem, 'dataBounds') as calls:
        half = vb.childrenBounds(frac=(1.0, 0.5))
        ortho = vb.childrenBounds(orthoRange=(None, [0, 10]))
    assert calls.count == 4
    assert half[1][1] < full[1][1]
    assert ortho[1][1] == pytest.approx(10, abs=0.1)  # with the pen pixel padding


def test_childrenBounds_sees_unnotified_rotation(plot):
    # setRotation does not notify the ViewBox; the transform in the cache key does.
    vb = plot.getViewBox()
    line = pg.InfiniteLine(pos=5, angle=0)
    plot.addItem(line)
    assert vb.childrenBounds() == [None, [5, 5]]
    line.setAngle(90)
    assert vb.childrenBounds() == [[0, 0], None]


def test_childrenBounds_sees_child_visibility(plot):
    # PlotDataItem.dataBounds depends on the visibility of its curve.
    vb = plot.getViewBox()
    item = plot.plot(np.arange(10.0))
    assert vb.childrenBounds()[1] == pytest.approx([0, 9], abs=0.1)
    item.curve.hide()  # no bounds change is reported
    assert vb.childrenBounds() == [None, None]
    item.curve.show()
    assert vb.childrenBounds()[1] == pytest.approx([0, 9], abs=0.1)


def test_removeItem_forgets_item(plot):
    vb = plot.getViewBox()
    item = plot.plot(np.arange(10.0))
    vb.childrenBounds()
    assert item in vb._boundedItems
    assert item in vb._itemBoundsCache
    plot.removeItem(item)
    assert item not in vb._boundedItems
    assert item not in vb._itemBoundsCache
    assert vb.childrenBounds() == [None, None]


def test_queueUpdateAutoRange_repaints_only_with_autorange(plot):
    vb = plot.getViewBox()
    plot.plot(np.arange(10.0))
    process_events()
    with count_calls(vb, 'update') as updates:
        vb.queueUpdateAutoRange()
    assert updates.count == 1

    vb.disableAutoRange()
    process_events()
    with count_calls(vb, 'update') as updates:
        vb.queueUpdateAutoRange()
        vb.addItem(pg.InfiniteLine(pos=3, angle=0))
    assert updates.count == 0

    # enabling the auto-range again still applies the pending request
    vb.enableAutoRange()
    process_events()
    assert vb.viewRange()[1][0] <= 0 and vb.viewRange()[1][1] >= 9


def test_autorange_still_follows_bounded_items(plot):
    vb = plot.getViewBox()
    line = pg.InfiniteLine(pos=0, angle=0)
    plot.addItem(line)
    plot.plot(np.arange(10.0))
    process_events()
    line.setPos(QtCore.QPointF(0, 50))
    process_events()
    assert vb.viewRange()[1][1] >= 50


def test_view_transform_change_is_not_reported_as_bounds_change(plot):
    # ChildGroup is exempt from informing the ViewBox of its transform changes,
    # which happen on every pan and zoom.
    vb = plot.getViewBox()
    plot.plot(np.arange(100.0))
    vb.enableAutoRange(x=False, y=True)
    process_events()
    with count_calls(vb, 'itemBoundsChanged') as notified, \
            count_calls(vb, 'queueUpdateAutoRange') as queued:
        for _ in range(5):
            vb.translateBy(x=1)
            process_events()
    assert notified.count == 0
    assert queued.count == 0
