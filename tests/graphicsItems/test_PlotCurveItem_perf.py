"""
Performance-oriented regression tests of ``PlotCurveItem`` rendering.

They check deterministic properties (which drawing primitive is used, which caches are
built) and that the faster drawing code renders pixel-identical images.
"""
import numpy as np
import pytest

import pyqtgraph as pg
from pyqtgraph import functions as fn
from pyqtgraph.graphicsItems.PlotCurveItem import PlotCurveItem
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from tests.perf_helpers import count_calls

app = pg.mkQApp()


def render_item(item, rect, size=(400, 300), brush=None):
    """
    Paint ``item`` into an image, mapping ``rect`` onto the whole image.

    Parameters
    ----------
    item : QtWidgets.QGraphicsItem
        Item to paint.
    rect : QtCore.QRectF
        Rectangle, in item coordinates, mapped onto the image.
    size : tuple of int, default (400, 300)
        Image size.
    brush : QtGui.QBrush or None
        Brush set on the painter before painting, if any.

    Returns
    -------
    np.ndarray
        The ARGB32 pixels, of shape (height, width).
    """
    img = QtGui.QImage(size[0], size[1],
                       QtGui.QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(0)
    painter = QtGui.QPainter(img)
    painter.scale(size[0] / rect.width(), size[1] / rect.height())
    painter.translate(-rect.left(), -rect.top())
    if brush is not None:
        painter.setBrush(brush)
    item.paint(painter, QtWidgets.QStyleOptionGraphicsItem(), None)
    painter.end()
    return pg.functions.ndarray_from_qimage(img).view(np.uint32)[..., 0].copy()


def render_with_path(item, rect, size=(400, 300)):
    """
    Paint ``item`` like :func:`render_item`, forcing the drawing of its QPainterPath.

    Parameters
    ----------
    item : PlotCurveItem
        Curve to paint.
    rect : QtCore.QRectF
        Rectangle, in item coordinates, mapped onto the image.
    size : tuple of int, default (400, 300)
        Image size.

    Returns
    -------
    np.ndarray
        The ARGB32 pixels, of shape (height, width).
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(PlotCurveItem, '_shouldUseDrawPolyline', lambda *args: False)
        return render_item(item, rect, size)


def data_rects(x, y):
    """
    Return the full data rectangle and a narrow one around the end of the curve.

    Parameters
    ----------
    x, y : np.ndarray
        Data of the curve.

    Returns
    -------
    list of QtCore.QRectF
        The rectangles.
    """
    x0, x1 = float(np.nanmin(x)), float(np.nanmax(x))
    y0, y1 = float(np.nanmin(y)), float(np.nanmax(y))
    full = QtCore.QRectF(x0, y0, x1 - x0, y1 - y0)
    end = QtCore.QRectF(x1 - 0.01 * (x1 - x0), y0, 0.011 * (x1 - x0), y1 - y0)
    return [full, end]


def random_walk(n, seed=0):
    """
    Return the data of a random walk.

    Parameters
    ----------
    n : int
        Number of points.
    seed : int, default 0
        Seed of the random generator.

    Returns
    -------
    x, y : np.ndarray
        Coordinates of the points.
    """
    rng = np.random.default_rng(seed)
    return np.arange(n, dtype=np.float64), np.cumsum(rng.standard_normal(n))


# --------------------------------------------------------------------------------------
# T2.6: drawPolyline instead of drawPath for single polylines
# --------------------------------------------------------------------------------------

POLYLINE_CASES = [
    # pen, antialias, connect, skipFiniteCheck, number of points, non-finite values
    (dict(color='w'), True, 'all', False, 5000, False),
    (dict(color=(255, 200, 0, 120)), True, 'finite', False, 30001, False),
    (dict(color='w', width=0), True, 'all', True, 20001, False),
    (dict(color=(255, 0, 0, 150), width=3), False, 'all', True, 5000, False),
    (dict(color='c', width=2, style=QtCore.Qt.PenStyle.DashLine), False, 'finite',
     True, 5000, False),
    (dict(color='w', width=3), True, 'all', False, 40001, True),
]


@pytest.mark.parametrize('pen, antialias, connect, skipFiniteCheck, n, nonfinite',
                         POLYLINE_CASES)
def test_polyline_pixel_identical(pen, antialias, connect, skipFiniteCheck, n,
                                  nonfinite):
    x, y = random_walk(n)
    if nonfinite:
        y[::97] = np.nan
    curves = [pg.PlotCurveItem(x=x, y=y, pen=pg.mkPen(**pen), antialias=antialias,
                               connect=connect, skipFiniteCheck=skipFiniteCheck)
              for _ in range(2)]
    for rect in data_rects(x, y):
        with count_calls(fn, 'arrayToQPath') as path_builds:
            fast = render_item(curves[0], rect)
        assert path_builds.count == 0
        assert curves[0].path is None
        assert (fast == render_with_path(curves[1], rect)).all()


def test_polyline_segmented_mode_off():
    # 'off' keeps drawing continuous lines: the polyline qualifies, the segments not
    x, y = random_walk(3000)
    curves = [pg.PlotCurveItem(x=x, y=y, pen=pg.mkPen('w', width=3), antialias=False)
              for _ in range(2)]
    for curve in curves:
        curve.setSegmentedLineMode('off')
    rect = data_rects(x, y)[0]
    with count_calls(fn, 'arrayToQPath') as path_builds:
        fast = render_item(curves[0], rect)
    assert path_builds.count == 0
    assert (fast == render_with_path(curves[1], rect)).all()


def test_polyline_getPath_unchanged():
    x, y = random_walk(25000)
    y[100] = np.nan
    curve = pg.PlotCurveItem(x=x, y=y, antialias=True)
    render_item(curve, data_rects(x, y)[0])
    assert curve.path is None
    # the path is built on request only, and is the same as before
    assert curve.getPath() == fn.arrayToQPath(x, y, connect='all', finiteCheck=True)


def test_polyline_cached_until_data_changes():
    x, y = random_walk(2000)
    curve = pg.PlotCurveItem(x=x, y=y, antialias=True)
    rect = data_rects(x, y)[0]
    with count_calls(fn, '_arrayToQPath_all_vertices') as vertices:
        for _ in range(3):
            render_item(curve, rect)
        assert vertices.count == 1
        polyline = curve._polyline
        # streaming: the polyline buffer is refilled, not reallocated
        curve.setData(np.append(x, 2000.0), np.append(y, 0.0))
        render_item(curve, rect)
        assert vertices.count == 2
        assert curve._polyline is polyline
        assert len(curve._polyline) == 2001
        # options changing the vertices invalidate the cache
        curve.setSkipFiniteCheck(True)
        render_item(curve, rect)
        assert vertices.count == 3


def test_polyline_memory_released_after_shrink():
    x, y = random_walk(100_000)
    curve = pg.PlotCurveItem(x=x, y=y, antialias=True)
    render_item(curve, data_rects(x, y)[0])
    polyline = curve._polyline
    curve.setData(x[:1000], y[:1000])
    render_item(curve, data_rects(x[:1000], y[:1000])[0])
    assert curve._polyline is not polyline
    assert len(curve._polyline) == 1000


@pytest.mark.parametrize('setup', ['aliased thin pen', 'shadow pen', 'fill level',
                                   'step mode', 'connect pairs', 'finite with NaN',
                                   'export', 'painter brush'])
def test_path_still_drawn(setup):
    # cases where the polyline is not used: the path is built and drawn as before
    x, y = random_walk(1000)
    kwargs = dict(antialias=True)
    brush = None
    if setup == 'aliased thin pen':
        # Qt draws the polyline and the path differently for this pen
        kwargs['antialias'] = False
    elif setup == 'shadow pen':
        kwargs['shadowPen'] = pg.mkPen('r', width=4)
    elif setup == 'fill level':
        kwargs['fillLevel'] = 0.0
    elif setup == 'step mode':
        kwargs['stepMode'] = 'left'
    elif setup == 'connect pairs':
        kwargs['connect'] = 'pairs'
    elif setup == 'finite with NaN':
        y[::10] = np.nan
        kwargs['connect'] = 'finite'
    elif setup == 'painter brush':
        brush = pg.mkBrush('b')
    curve = pg.PlotCurveItem(x=x, y=y, **kwargs)
    if setup == 'export':
        curve.setExportMode(True, {'antialias': True})
    with count_calls(fn, 'arrayToQPath') as path_builds:
        render_item(curve, data_rects(x, y)[0], brush=brush)
    assert path_builds.count == 1
    assert curve._polyline is None
