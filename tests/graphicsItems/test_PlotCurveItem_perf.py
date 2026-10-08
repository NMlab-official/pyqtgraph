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
                                   'step mode', 'connect array', 'finite with NaN',
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
    elif setup == 'connect array':
        kwargs['connect'] = np.arange(len(x)) % 3 != 0
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
                                   'decreasing x', 'fill outline', 'large part exposed',
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
    elif setup == 'fill outline':
        kwargs.update(fillLevel=0.0, brush='b', fillOutline=True)
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


# --------------------------------------------------------------------------------------
# T4.3: fill paths built faster, and skipped outside the exposed rectangle
# --------------------------------------------------------------------------------------

def reference_fill_paths(x, y, baseline, chunksize=150):
    """
    Build the fill paths of a run of finite points as before T4.3.

    Parameters
    ----------
    x, y : np.ndarray
        Finite coordinates of the run.
    baseline : float
        Fill level.
    chunksize : int, default 150
        Number of curve points per chunk.

    Returns
    -------
    list of QtGui.QPainterPath
        One path per chunk.
    """
    paths = []
    offset = 0
    xybuf = np.empty((chunksize + 3, 2))
    while offset < len(x) - 1:
        subx = x[offset:offset + chunksize]
        suby = y[offset:offset + chunksize]
        size = len(subx)
        xyview = xybuf[:size + 3]
        xyview[:-3, 0] = subx
        xyview[:-3, 1] = suby
        xyview[-3:, 0] = subx[[-1, 0, 0]]
        xyview[-3:, 1] = [baseline, baseline, suby[0]]
        offset += size - 1
        paths.append(
            fn._arrayToQPath_all(xyview[:, 0], xyview[:, 1], finiteCheck=False))
    return paths


def path_elements(path):
    """
    Return the elements of a path as an array of (type, x, y) rows.

    Parameters
    ----------
    path : QtGui.QPainterPath
        The path.

    Returns
    -------
    np.ndarray
        The elements, of shape (elementCount, 3).
    """
    elements = np.empty((path.elementCount(), 3))
    for i in range(path.elementCount()):
        element = path.elementAt(i)
        elements[i] = (0.0 if element.isMoveTo() else 1.0, element.x, element.y)
    return elements


@pytest.mark.parametrize('n', [2, 3, 149, 150, 151, 299, 1000, 1789])
@pytest.mark.parametrize('dtype', [np.float64, np.int32])
def test_fill_paths_identical_to_reference(n, dtype):
    rng = np.random.default_rng(n)
    x = np.arange(n).astype(dtype)
    y = (rng.standard_normal(n) * 10).astype(dtype)
    paths, bounds = [], []
    PlotCurveItem._construct_finite_segment_FillPaths(x, y, 1.5, 150, paths, bounds)
    reference = reference_fill_paths(x, y, 1.5)
    assert len(paths) == len(reference)
    for path, expected, box in zip(paths, reference, np.concatenate(bounds)):
        np.testing.assert_array_equal(path_elements(path), path_elements(expected))
        rect = expected.controlPointRect()
        expected_box = (rect.left(), rect.right(), rect.top(), rect.bottom())
        np.testing.assert_allclose(box, expected_box, rtol=1e-12)


def test_fill_paths_cached():
    x, y = random_walk(5000)
    curve = pg.PlotCurveItem(x=x, y=y, fillLevel=0.0, brush='b')
    rect = data_rects(x, y)[0]
    render_item(curve, rect)
    fill = curve._fillPathList
    assert len(fill.paths) == len(fill.bounds) == 34
    render_item(curve, rect)
    assert curve._fillPathList is fill
    curve.setData(x, y + 1)
    render_item(curve, rect)
    assert curve._fillPathList is not fill


@contextlib.contextmanager
def recorded_fill_paths():
    """
    Record how many fill paths ``PlotCurveItem.paint`` fills.

    Yields
    ------
    list of int
        Number of paths filled by each paint.
    """
    original = PlotCurveItem._getVisibleFillPaths
    counts = []

    def wrapper(self, *args):
        paths = original(self, *args)
        counts.append(len(paths))
        return paths

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(PlotCurveItem, '_getVisibleFillPaths', wrapper)
        yield counts


@pytest.mark.parametrize('antialias', [False, True])
@pytest.mark.parametrize('pen', [None, 'w'])
@pytest.mark.parametrize('zoom', [1, 25])
def test_fill_exposed_pixel_identical(antialias, pen, zoom):
    x, y = random_walk(20000)
    y[rng_indices(20000, 0.01)] = np.nan
    curve = pg.PlotCurveItem(x=x, y=y, pen=pen, fillLevel=0.0,
                             brush=(80, 120, 250, 100), antialias=antialias,
                             connect='finite')
    view = zoomed_view(x, y, zoom)
    total = len(curve._getFillPathList(None).paths)
    for left, width in [(0, 400), (0, 3), (150, 5), (397, 3)]:
        clip = QtCore.QRect(left, 0, width, 300)
        with recorded_fill_paths() as filled:
            partial = render_exposed(curve, view, clip)
        if width < 10:
            assert filled[0] < total // 10
        assert (partial == render_exposed(curve, view, clip, exposed=False)).all()


def rng_indices(n, fraction, seed=1):
    """
    Return random indices of an array.

    Parameters
    ----------
    n : int
        Length of the array.
    fraction : float
        Fraction of the indices to return.
    seed : int, default 1
        Seed of the random generator.

    Returns
    -------
    np.ndarray
        The indices.
    """
    return np.flatnonzero(np.random.default_rng(seed).random(n) < fraction)


def test_view_partial_repaint_filled():
    # a cursor line over a filled curve fills a few chunks and draws a few vertices,
    # and the screen shows the same pixels as after a full repaint
    pw = pg.PlotWidget()
    pw.resize(400, 300)
    pw.show()
    x, y = random_walk(100_000)
    curve = pw.plot(x, y, pen='w', fillLevel=0.0, brush=(80, 120, 250, 100)).curve
    line = pg.InfiniteLine(pos=30_000, angle=90, pen='r')
    pw.addItem(line, ignoreBounds=True)
    process_events(5)
    pw.getViewBox().disableAutoRange()
    process_events(5)
    total = len(curve._getFillPathList(None).paths)

    def screen():
        image = pw.screen().grabWindow(pw.winId()).toImage()
        if image.isNull():
            pytest.skip('the platform cannot grab the window')
        image = image.convertToFormat(QtGui.QImage.Format.Format_ARGB32)
        return pg.functions.ndarray_from_qimage(image).copy()

    with recorded_fill_paths() as filled, recorded_slices() as slices:
        for pos in (30_500, 31_000, 31_500):
            line.setPos(pos)
            process_events()
    assert len(filled) == 3 and max(filled) < total // 10
    assert len(slices) == 3 and max(slices) < len(x) // 10
    partial = screen()
    pw.viewport().repaint()
    process_events()
    assert (partial == screen()).all()
    pw.close()


# --------------------------------------------------------------------------------------
# T4.4: connect='pairs' drawn with drawLines instead of a QDataStream-built path
# --------------------------------------------------------------------------------------

def pairs_data(kind, n=4001):
    """
    Return the data of a ``connect='pairs'`` curve.

    Parameters
    ----------
    kind : str
        ``'walk'``, ``'nan'`` (with non-finite values), ``'bars'`` (vertical
        segments, as error bars) or ``'zero length'`` (some segments of length 0).
    n : int, default 4001
        Number of points; an odd number leaves the last point unpaired.

    Returns
    -------
    x, y : np.ndarray
        Coordinates of the points.
    """
    x, y = random_walk(n)
    if kind == 'nan':
        y[rng_indices(n, 0.05)] = np.nan
    elif kind == 'bars':
        x = np.repeat(np.arange((n + 1) // 2, dtype=np.float64), 2)[:n]
    elif kind == 'zero length':
        x[1::4] = x[0::4][:len(x[1::4])]
        y[1::4] = y[0::4][:len(y[1::4])]
    return x, y


def render_with_pairs_path(item, rect, size=(400, 300)):
    """
    Paint ``item`` like :func:`render_item`, forcing the drawing of its path.

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
        mp.setattr(PlotCurveItem, '_shouldDrawPairsAsLines', lambda *args: False)
        return render_item(item, rect, size)


@pytest.mark.parametrize('pen', [dict(color='w'), dict(color=(255, 255, 0, 100)),
                                 dict(color='w', width=0),
                                 dict(color='c', style=QtCore.Qt.PenStyle.DashLine)])
@pytest.mark.parametrize('antialias', [False, True])
@pytest.mark.parametrize('kind', ['walk', 'nan', 'bars'])
def test_pairs_drawn_as_lines_pixel_identical(pen, antialias, kind):
    x, y = pairs_data(kind)
    curves = [pg.PlotCurveItem(x=x, y=y, pen=pg.mkPen(**pen), antialias=antialias,
                               connect='pairs') for _ in range(2)]
    for rect in data_rects(x, y):
        with count_calls(fn, 'arrayToQPath') as path_builds:
            lines = render_item(curves[0], rect)
        assert path_builds.count == 0
        assert curves[0].path is None
        assert (lines == render_with_pairs_path(curves[1], rect)).all()


@pytest.mark.parametrize('setup', ['zero length', 'flat cap', 'wide antialiased pen',
                                   'non-cosmetic pen', 'segmented mode off',
                                   'unchecked non-finite', 'export', 'tiny on screen'])
def test_pairs_path_still_drawn(setup):
    x, y = pairs_data('zero length' if setup == 'zero length' else 'walk')
    pen = pg.mkPen('w')
    kwargs = dict(connect='pairs')
    rect = data_rects(x, y)[0]
    if setup == 'flat cap':
        pen.setCapStyle(QtCore.Qt.PenCapStyle.FlatCap)
    elif setup == 'wide antialiased pen':
        pen = pg.mkPen('w', width=2)
        kwargs['antialias'] = True
    elif setup == 'non-cosmetic pen':
        pen.setCosmetic(False)
    elif setup == 'unchecked non-finite':
        y[::7] = np.nan
        kwargs['skipFiniteCheck'] = True
    elif setup == 'tiny on screen':
        # segments far below a pixel: Qt may see their ends as equal
        x[1::2] = x[0::2][:len(x[1::2])] + 1e-9
        y[1::2] = y[0::2][:len(y[1::2])]
        rect = QtCore.QRectF(-1e6, rect.top(), 2e6, rect.height())
    curve = pg.PlotCurveItem(x=x, y=y, pen=pen, **kwargs)
    if setup == 'segmented mode off':
        curve.setSegmentedLineMode('off')
    elif setup == 'export':
        curve.setExportMode(True, {'antialias': False})
    with count_calls(fn, 'arrayToQPath') as path_builds:
        render_item(curve, rect)
    assert path_builds.count == 1


def test_pairs_lengths_cached():
    x, y = pairs_data('walk')
    curve = pg.PlotCurveItem(x=x, y=y, connect='pairs')
    rect = data_rects(x, y)[0]
    with count_calls(np, 'isfinite') as finite_checks:
        render_item(curve, rect)
        first = finite_checks.count
        render_item(curve, rect)
        render_item(curve, rect)
        assert finite_checks.count == first
    assert curve._vertexCache.pairLengths is not None
