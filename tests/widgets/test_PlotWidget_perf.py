"""
The useOpenGL argument of PlotWidget and GraphicsLayoutWidget reaches GraphicsView.

No OpenGL context is needed: GraphicsView.useOpenGL is replaced by a recorder.
"""
import pytest

import pyqtgraph as pg

app = pg.mkQApp()


@pytest.fixture
def useOpenGL_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(pg.GraphicsView, 'useOpenGL', lambda self, b=True: calls.append(b))
    return calls


@pytest.mark.parametrize('useOpenGL', [True, False])
def test_PlotWidget_forwards_useOpenGL(useOpenGL_calls, useOpenGL):
    pw = pg.PlotWidget(useOpenGL=useOpenGL)
    try:
        assert useOpenGL_calls == [useOpenGL]
        # the argument no longer reaches PlotItem, which plotted a spurious curve
        assert pw.getPlotItem().listDataItems() == []
    finally:
        pw.close()


def test_PlotWidget_useOpenGL_default_from_config(useOpenGL_calls):
    pw = pg.PlotWidget(title='kwargs still reach the PlotItem')
    try:
        assert useOpenGL_calls == [pg.getConfigOption('useOpenGL')]
        assert pw.getPlotItem().titleLabel.text == 'kwargs still reach the PlotItem'
    finally:
        pw.close()


@pytest.mark.parametrize('useOpenGL', [True, False])
def test_GraphicsLayoutWidget_forwards_useOpenGL(useOpenGL_calls, useOpenGL):
    win = pg.GraphicsLayoutWidget(useOpenGL=useOpenGL, border=True)
    try:
        assert useOpenGL_calls == [useOpenGL]
    finally:
        win.close()
