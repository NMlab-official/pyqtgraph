"""
Rendering and performance regression tests for NonUniformImage.

The cells are drawn as one image on pixel-based devices. The tests compare it with the
rectangle-per-cell rendering (still used for other paint devices) and with the expected
color of every cell, and check deterministic cache states and call counts, never
durations.
"""
import numpy as np
import pytest

import pyqtgraph as pg
import pyqtgraph.functions as fn
from pyqtgraph.graphicsItems.NonUniformImage import NonUniformImage
from pyqtgraph.Qt import QtCore, QtGui
from tests.perf_helpers import count_calls, show_and_wait

app = pg.mkQApp()

Format = QtGui.QImage.Format


def _item(seed=0, nx=23, ny=17, nans=True, lut=None, levels=None, border=None):
    """
    Make a NonUniformImage with random non-uniform coordinates.

    Parameters
    ----------
    seed : int, default 0
        Seed of the random generator.
    nx, ny : int, default 23, 17
        Number of samples along x and y.
    nans : bool, default True
        Put NaN and infinite values in the data.
    lut : array_like or None, default None
        Lookup table, viridis with 256 entries if None.
    levels : tuple of float or None, default None
        Levels, automatic if None.
    border : QPen or None, default None
        Border pen.

    Returns
    -------
    NonUniformImage
        The item.
    """
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.uniform(0.2, 3.0, nx))
    y = np.cumsum(rng.uniform(0.2, 3.0, ny)) - 10
    z = rng.normal(size=(nx, ny))
    if nans:
        z[rng.random(z.shape) < 0.1] = np.nan
        z[0, 0] = np.inf
        z[-1, -1] = -np.inf
    item = NonUniformImage(x, y, z, border=border)
    item.setLookupTable(
        pg.colormap.get('viridis').getLookupTable(nPts=256) if lut is None else lut)
    if levels is not None:
        item.setLevels(levels)
    return item


def _paint(draw, size=(320, 240), dpr=1.0) -> np.ndarray:
    """
    Paint onto an image with an opaque background and return its pixels.

    Parameters
    ----------
    draw : callable
        Function receiving the active QPainter.
    size : tuple of int, default (320, 240)
        Logical size of the image.
    dpr : float, default 1.0
        Device pixel ratio of the image.

    Returns
    -------
    np.ndarray
        Pixels of shape (height, width, 4), as int.
    """
    img = QtGui.QImage(int(size[0] * dpr), int(size[1] * dpr), Format.Format_ARGB32_Premultiplied)
    img.setDevicePixelRatio(dpr)
    img.fill(QtGui.QColor(20, 40, 60))
    p = QtGui.QPainter(img)
    draw(p)
    p.end()
    return fn.ndarray_from_qimage(img).astype(int)


def _transform(item, flip_x=False, flip_y=True, zoom=1.0, shift=(0.0, 0.0)) -> QtGui.QTransform:
    """
    Return a transform showing the item in a 320x240 image.

    Parameters
    ----------
    item : NonUniformImage
        Item to show.
    flip_x, flip_y : bool, default False, True
        Invert the axes (the y axis of a ViewBox is inverted).
    zoom : float, default 1.0
        Zoom factor around the center of the item.
    shift : tuple of float, default (0.0, 0.0)
        Translation in device pixels.

    Returns
    -------
    QtGui.QTransform
        Item to device transform.
    """
    rect = item.boundingRect()
    sx = 300 / rect.width() * zoom * (-1 if flip_x else 1)
    sy = 220 / rect.height() * zoom * (-1 if flip_y else 1)
    center = rect.center()
    return QtGui.QTransform(sx, 0, 0, sy, 160 - sx * center.x() + shift[0],
                            120 - sy * center.y() + shift[1])


def _raster(item, tr, **kwargs) -> np.ndarray:
    """
    Paint the item with a transform through its regular paint method.

    Parameters
    ----------
    item : NonUniformImage
        Item to paint.
    tr : QtGui.QTransform
        Item to device transform.
    **kwargs
        Arguments of :func:`_paint`.

    Returns
    -------
    np.ndarray
        Pixels.
    """
    def draw(p):
        p.setTransform(tr)
        item.paint(p, None, None)
    return _paint(draw, **kwargs)


def _rectangles(item, tr, **kwargs) -> np.ndarray:
    """
    Paint the item as one rectangle per cell (the previous rendering).

    Parameters
    ----------
    item : NonUniformImage
        Item to paint.
    tr : QtGui.QTransform
        Item to device transform.
    **kwargs
        Arguments of :func:`_paint`.

    Returns
    -------
    np.ndarray
        Pixels.
    """
    picture = QtGui.QPicture()
    p = QtGui.QPainter(picture)
    item.paint(p, None, None)  # not pixel-based: rectangles
    p.end()

    def draw(p):
        p.setTransform(tr)
        p.drawPicture(0, 0, picture)
    return _paint(draw, **kwargs)


@pytest.mark.parametrize('flip_x', [False, True])
@pytest.mark.parametrize('flip_y', [False, True])
@pytest.mark.parametrize('zoom,shift', [(1.0, (0, 0)), (0.73, (0.31, -0.47)), (3.3, (17.6, 9.2))])
def test_image_matches_rectangles(flip_x, flip_y, zoom, shift):
    item = _item(border=fn.mkPen('r'))
    tr = _transform(item, flip_x, flip_y, zoom, shift)
    np.testing.assert_array_equal(_raster(item, tr), _rectangles(item, tr))


@pytest.mark.parametrize('lut', [
    pytest.param(pg.colormap.get('plasma').getLookupTable(nPts=512), id='lut512'),
    pytest.param(np.random.default_rng(5).integers(256, size=(77, 4), dtype=np.uint8),
                 id='rgba77'),
    pytest.param(np.linspace(0, 1, 50), id='float1d'),
])
def test_image_matches_rectangles_other_luts(lut):
    item = _item(seed=1, lut=lut, levels=(-1.5, 2.0))
    tr = _transform(item, zoom=1.4, shift=(3.3, 1.1))
    np.testing.assert_array_equal(_raster(item, tr), _rectangles(item, tr))


def test_image_matches_rectangles_with_clip_and_hidpi():
    item = _item(seed=2)
    tr = _transform(item, zoom=1.2)
    clip = tr.inverted()[0].mapRect(QtCore.QRectF(40.3, 30.6, 150.2, 120.9))

    def raster(p):
        p.setTransform(tr)
        p.setClipRect(clip)
        item.paint(p, None, None)

    def rectangles(p):
        p.setTransform(tr)
        p.setClipRect(clip)
        p.drawPicture(0, 0, item._generateVectorPicture())

    for dpr in (1.0, 2.0):
        np.testing.assert_array_equal(_paint(raster, dpr=dpr), _paint(rectangles, dpr=dpr))


def test_cell_centers_have_lut_colors():
    lut = pg.colormap.get('viridis').getLookupTable(nPts=256)
    item = _item(seed=3, lut=lut, levels=(-1.0, 1.0))
    tr = _transform(item, flip_y=False)
    pixels = _raster(item, tr)
    x, y, z = item.data
    xm = np.pad(x, 1, mode='edge')
    xm = (xm[:-1] + xm[1:]) / 2
    ym = np.pad(y, 1, mode='edge')
    ym = (ym[:-1] + ym[1:]) / 2
    index = np.clip(np.floor((z + 1.0) * 256 / 2.0), 0, 255)
    for i in range(len(x)):
        for j in range(len(y)):
            center = tr.map(QtCore.QPointF((xm[i] + xm[i + 1]) / 2, (ym[j] + ym[j + 1]) / 2))
            b, g, r, a = pixels[int(center.y()), int(center.x())]
            if np.isnan(z[i, j]):
                assert (r, g, b) == (20, 40, 60)  # transparent: background
            else:
                assert (r, g, b, a) == tuple(lut[int(index[i, j])]) + (255,)


def test_bounds():
    x = np.array([1.0, 3.0, 10.0])
    y = np.array([-1.0, 2.0, 4.0, 4.5])
    item = NonUniformImage(x, y, np.zeros((3, 4)))
    assert item.boundingRect() == QtCore.QRectF(1.0, -1.0, 9.0, 5.5)


def test_one_image_per_view():
    item = _item(seed=4, nx=200, ny=100, nans=False)
    tr = _transform(item)
    with count_calls(NonUniformImage, '_renderVisibleCells') as renders, \
            count_calls(NonUniformImage, '_generateVectorPicture') as rectangles, \
            count_calls(NonUniformImage, 'generatePicture') as colors:
        _raster(item, tr)
        _raster(item, tr)
        assert (renders.count, colors.count) == (1, 1)
        # a new view renders a new image from the same cell colors
        _raster(item, _transform(item, shift=(2.5, 0)))
        assert (renders.count, colors.count) == (2, 1)
        # new levels: new colors and image
        item.setLevels((0, 1))
        _raster(item, tr)
        assert (renders.count, colors.count) == (3, 2)
        # new lookup table
        item.setLookupTable(pg.colormap.get('plasma').getLookupTable(nPts=256))
        _raster(item, tr)
        assert (renders.count, colors.count) == (4, 3)
        assert rectangles.count == 0
    assert item.picture is None


def test_rectangles_for_other_devices_and_rotation():
    item = _item(seed=5)
    tr = _transform(item)
    with count_calls(NonUniformImage, '_generateVectorPicture') as rectangles:
        _rectangles(item, tr)
        assert rectangles.count == 1
        assert item.picture is not None
        rotated = QtGui.QTransform(tr).rotate(15)
        reference = _paint(lambda p: (p.setTransform(rotated), p.drawPicture(0, 0, item.picture)))
        np.testing.assert_array_equal(_raster(item, rotated), reference)
        assert rectangles.count == 1  # the picture is reused


def test_setData():
    item = _item(seed=6, nans=False)
    lut = item.lut
    first_levels = item.getLevels()
    tr = _transform(item)
    _raster(item, tr)

    rng = np.random.default_rng(7)
    x = np.cumsum(rng.uniform(0.2, 3.0, 30))
    y = np.cumsum(rng.uniform(0.2, 3.0, 12))
    z = rng.normal(size=(30, 12)) * 10
    item.setData(x, y, z)
    assert item.lut is lut
    assert item.boundingRect() == QtCore.QRectF(x[0], y[0], x[-1] - x[0], y[-1] - y[0])
    # levels computed from the data follow the data
    assert item.getLevels() != first_levels
    assert item.getLevels() == (z.min(), z.max())
    reference = _item(seed=6, nans=False)
    reference.setData(x, y, z)
    tr = _transform(item)
    np.testing.assert_array_equal(_raster(item, tr), _raster(reference, tr))

    # levels set explicitly are kept
    item.setLevels((-1, 1))
    item.setData(x, y, z / 10)
    assert item.getLevels() == (-1, 1)

    with pytest.raises(ValueError, match="monotonically increasing"):
        item.setData(x[::-1], y, z)
    with pytest.raises(ValueError, match="match the shape of z"):
        item.setData(x, y, z.T)


def test_picture_none_discards_renderings():
    item = _item(seed=8, nans=False)
    tr = _transform(item)
    before = _raster(item, tr)
    item.data[2][:] = item.data[2][::-1].copy()  # modified in place
    item.picture = None
    after = _raster(item, tr)
    assert (before != after).any()
    reference = _item(seed=8, nans=False)
    reference.data[2][:] = reference.data[2][::-1].copy()
    reference.picture = None
    np.testing.assert_array_equal(after, _rectangles(reference, tr))


def test_callable_and_missing_luts():
    item = _item(seed=9)
    tr = _transform(item)
    item.setLookupTable(lambda z: pg.colormap.get('CET-L1').getLookupTable(nPts=300))
    np.testing.assert_array_equal(_raster(item, tr), _rectangles(item, tr))
    item.setLookupTable(None)  # grayscale
    np.testing.assert_array_equal(_raster(item, tr), _rectangles(item, tr))


def test_in_view_box():
    view = pg.GraphicsView()
    viewbox = pg.ViewBox()
    view.setCentralWidget(viewbox)
    view.resize(200, 160)
    item = _item(seed=10, border=fn.mkPen('g'))
    viewbox.addItem(item)
    show_and_wait(view)
    rendered = view.grab().toImage()
    assert item._rendered is not None
    assert item.picture is None
    # the viewport only shows the view box area: the image is clipped to it
    width = rendered.width() * rendered.devicePixelRatio()
    assert item._rendered[1].width() <= width
    view.close()
