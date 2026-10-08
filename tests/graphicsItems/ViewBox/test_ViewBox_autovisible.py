import numpy as np
import pytest

import pyqtgraph as pg
from tests.perf_helpers import process_events

app = pg.mkQApp()


@pytest.fixture
def plot_widget():
    pw = pg.PlotWidget()
    pw.resize(400, 300)
    pw.show()
    yield pw
    pw.close()


def test_autovisible_y_follows_manual_x_pan(plot_widget):
    # A manual x pan disables the x auto-range; the y auto-range restricted to the
    # visible data must still be applied (it used to depend on the x auto-range,
    # which only went unnoticed while every view update re-queued an auto-range).
    plot_widget.plot(np.arange(1000.), np.arange(1000.))
    vb = plot_widget.getViewBox()
    vb.setAutoVisible(y=True)
    vb.enableAutoRange(y=True)
    plot_widget.setXRange(0, 10, padding=0)
    process_events()
    ymin, ymax = vb.viewRange()[1]
    assert ymin > -5 and ymax < 15
    vb.translateBy(x=500)
    process_events()
    ymin, ymax = vb.viewRange()[1]
    assert ymin > 490 and ymax < 515


def test_autovisible_x_follows_manual_y_pan(plot_widget):
    plot_widget.plot(np.arange(1000.), np.arange(1000.))
    vb = plot_widget.getViewBox()
    vb.setAutoVisible(x=True)
    vb.enableAutoRange(x=True)
    plot_widget.setYRange(0, 10, padding=0)
    process_events()
    xmin, xmax = vb.viewRange()[0]
    assert xmin > -5 and xmax < 15
