"""
Performance-oriented regression tests of ``PlotCurveItem`` rendering.

They check deterministic properties (which drawing primitive is used, which caches are
built) and that the faster drawing code renders pixel-identical images.
"""
import contextlib

import numpy as np
import pytest

import pyqtgraph as pg
from pyqtgraph import functions as fn
from pyqtgraph.graphicsItems.PlotCurveItem import PlotCurveItem
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from tests.perf_helpers import count_calls, process_events

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


# --------------------------------------------------------------------------------------
# T4.2: partial repaints draw only the vertices around the exposed rectangle
# --------------------------------------------------------------------------------------


def render_exposed(item, view, device_clip, exposed=True, brush=None, size=(400, 300)):
    """
    Paint ``item`` like ``QGraphicsView`` repainting a part of the viewport.

    The painter is clipped to ``device_clip`` and, if ``exposed`` is True, the style
    option exposes that rectangle only, as with ``ItemUsesExtendedStyleOption``.

    Parameters
    ----------
    item : QtWidgets.QGraphicsItem
        Item to paint.
    view : QtCore.QRectF
        Rectangle, in item coordinates, mapped onto the whole image.
    device_clip : QtCore.QRect
        Repainted rectangle, in pixels.
    exposed : bool, default True
        Set ``exposedRect`` to the repainted rectangle; otherwise it stays empty, as
        for an item without ``ItemUsesExtendedStyleOption``.
    brush : QtGui.QBrush or None
        Brush set on the painter before painting, if any.
    size : tuple of int, default (400, 300)
        Image size.

    Returns
    -------
    np.ndarray
        The ARGB32 pixels, of shape (height, width).
    """
    img = QtGui.QImage(size[0], size[1],
                       QtGui.QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(0)
    painter = QtGui.QPainter(img)
    painter.setClipRect(device_clip)
    painter.scale(size[0] / view.width(), size[1] / view.height())
    painter.translate(-view.left(), -view.top())
    if brush is not None:
        painter.setBrush(brush)
    option = QtWidgets.QStyleOptionGraphicsItem()
    if exposed:
        option.exposedRect = painter.transform().inverted()[0].mapRect(
            QtCore.QRectF(device_clip))
    item.paint(painter, option, None)
    painter.end()
    return pg.functions.ndarray_from_qimage(img).view(np.uint32)[..., 0].copy()


def zoomed_view(x, y, zoom):
    """
    Return a view rectangle showing a fraction ``1 / zoom`` of the x range.

    Parameters
    ----------
    x, y : np.ndarray
        Data of the curve.
    zoom : float
        Zoom factor along x.

    Returns
    -------
    QtCore.QRectF
        The view rectangle.
    """
    x0, x1 = float(np.nanmin(x)), float(np.nanmax(x))
    y0, y1 = float(np.nanmin(y)), float(np.nanmax(y))
    return QtCore.QRectF(x0 + 0.37 * (x1 - x0), y0 - 1, (x1 - x0) / zoom, y1 - y0 + 2)


@contextlib.contextmanager
def recorded_slices(method='_getVertexSlicePolyline'):
    """
    Record the sizes of the vertex slices drawn by ``PlotCurveItem.paint``.

    Parameters
    ----------
    method : str, default '_getVertexSlicePolyline'
        Private method building the slices, called with ``(start, stop)``.

    Yields
    ------
    list of int
        Number of vertices of each slice drawn.
    """
    original = getattr(PlotCurveItem, method)
    sizes = []

    def wrapper(self, start, stop):
        sizes.append(stop - start)
        return original(self, start, stop)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(PlotCurveItem, method, wrapper)
        yield sizes


EXPOSED_CASES = [
    # pen, antialias, connect, segmentedLineMode, number of points, zoom
    (dict(color='w'), False, 'all', 'auto', 20000, 1),
    (dict(color='w'), True, 'all', 'auto', 20000, 20),
    (dict(color=(255, 255, 0, 100)), True, 'finite', 'auto', 3000, 150),
    (dict(color=(255, 255, 0, 100), width=0), False, 'all', 'on', 20000, 5),
    (dict(color='w'), True, 'all', 'on', 3000, 1000),
]


@pytest.mark.parametrize('pen, antialias, connect, segmented, n, zoom', EXPOSED_CASES)
def test_exposed_slice_pixel_identical(pen, antialias, connect, segmented, n, zoom):
    x, y = random_walk(n)
    curve = pg.PlotCurveItem(x=x, y=y, pen=pg.mkPen(**pen), antialias=antialias,
                             connect=connect)
    curve.setSegmentedLineMode(segmented)
    view = zoomed_view(x, y, zoom)
    method = ('_getVertexSliceSegments' if segmented == 'on'
              else '_getVertexSlicePolyline')
    for left, width in [(0, 3), (150, 5), (233, 40), (397, 3)]:
        clip = QtCore.QRect(left, 0, width, 300)
        with recorded_slices(method) as slices:
            partial = render_exposed(curve, view, clip)
        assert len(slices) == 1 and slices[0] < n // 2
        assert (partial == render_exposed(curve, view, clip, exposed=False)).all()


def test_exposed_slice_is_small():
    x, y = random_walk(100_000)
    curve = pg.PlotCurveItem(x=x, y=y)
    view = zoomed_view(x, y, 1)
    with recorded_slices() as slices, count_calls(fn, 'arrayToQPath') as path_builds:
        render_exposed(curve, view, QtCore.QRect(200, 0, 5, 300))
    # 5 pixels of 400 at 250 points per pixel, plus margins
    assert len(slices) == 1 and slices[0] < 5000
    # the path of the whole curve was not built
    assert path_builds.count == 0
    assert curve.path is None


@pytest.mark.parametrize('setup', ['whole item exposed', 'wide pen', 'dashed pen',
                                   'decreasing x', 'fill', 'large part exposed',
                                   'no exposed rectangle', 'painter brush'])
def test_exposed_whole_curve_drawn(setup):
    x, y = random_walk(20000)
    kwargs = {}
    clip = QtCore.QRect(150, 0, 5, 300)
    exposed = True
    brush = None
    if setup == 'wide pen':
        kwargs['pen'] = pg.mkPen('w', width=2)
    elif setup == 'dashed pen':
        kwargs['pen'] = pg.mkPen('w', style=QtCore.Qt.PenStyle.DashLine)
    elif setup == 'decreasing x':
        x = x[::-1].copy()
    elif setup == 'fill':
        kwargs.update(fillLevel=0.0, brush='b')
    elif setup == 'large part exposed':
        clip = QtCore.QRect(50, 0, 250, 300)
    elif setup == 'whole item exposed':
        clip = QtCore.QRect(0, 0, 400, 300)
    elif setup == 'no exposed rectangle':
        exposed = False
    elif setup == 'painter brush':
        brush = pg.mkBrush('b')
    curve = pg.PlotCurveItem(x=x, y=y, **kwargs)
    view = zoomed_view(x, y, 1)
    with recorded_slices() as slices:
        render_exposed(curve, view, clip, exposed=exposed, brush=brush)
    assert slices == []


def test_view_partial_repaint():
    # a cursor line moving over a long curve repaints a few vertices of it, and the
    # screen shows the same pixels as after a full repaint
    pw = pg.PlotWidget()
    pw.resize(400, 300)
    pw.show()
    x, y = random_walk(100_000)
    curve = pw.plot(x, y, pen='w').curve
    line = pg.InfiniteLine(pos=30_000, angle=90, pen='r')
    pw.addItem(line, ignoreBounds=True)
    process_events(5)
    pw.getViewBox().disableAutoRange()
    process_events(5)
    flag = QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemUsesExtendedStyleOption
    assert curve.flags() & flag

    def screen():
        image = pw.screen().grabWindow(pw.winId()).toImage()
        if image.isNull():
            pytest.skip('the platform cannot grab the window')
        image = image.convertToFormat(QtGui.QImage.Format.Format_ARGB32)
        return pg.functions.ndarray_from_qimage(image).copy()

    with recorded_slices() as slices:
        for pos in (30_500, 31_000, 31_500):
            line.setPos(pos)
            process_events()
    # each move repaints the old and new positions of the line, 500 points apart
    assert len(slices) == 3
    assert max(slices) < len(x) // 10
    partial = screen()
    pw.viewport().repaint()
    process_events()
    assert (partial == screen()).all()
    pw.close()
