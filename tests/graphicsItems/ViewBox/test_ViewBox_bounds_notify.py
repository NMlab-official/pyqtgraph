import numpy as np

import pyqtgraph as pg
from tests.perf_helpers import process_events

app = pg.mkQApp()


def _plot(curve):
    pw = pg.PlotWidget()
    pw.resize(300, 200)
    pw.addItem(curve)
    pw.show()
    process_events()
    return pw


def test_fill_level_change_updates_autorange():
    # T1.3 caches the bounds of each item in the view: style setters that change
    # dataBounds must notify the view, or the auto-range keeps stale bounds.
    curve = pg.PlotCurveItem(np.arange(10.), np.arange(10.) + 100)
    pw = _plot(curve)
    assert pw.getViewBox().viewRange()[1][0] > 50
    curve.setFillLevel(0)
    process_events()
    assert pw.getViewBox().viewRange()[1][0] <= 0
    pw.close()


def test_non_cosmetic_pen_change_updates_autorange():
    curve = pg.PlotCurveItem(np.arange(10.), np.zeros(10))
    pw = _plot(curve)
    before = pw.getViewBox().viewRange()[1]
    curve.setPen(pg.mkPen('w', width=40, cosmetic=False))
    process_events()
    after = pw.getViewBox().viewRange()[1]
    assert after[1] - after[0] > before[1] - before[0]
    pw.close()


def test_text_anchor_change_notifies_view():
    text = pg.TextItem('label')
    pw = pg.PlotWidget()
    pw.addItem(text)
    vb = pw.getViewBox()
    calls = []
    original = vb.itemBoundsChanged
    vb.itemBoundsChanged = lambda item: (calls.append(item), original(item))
    try:
        text.setAnchor((1, 1))
    finally:
        del vb.itemBoundsChanged
    assert text in calls
    pw.close()
