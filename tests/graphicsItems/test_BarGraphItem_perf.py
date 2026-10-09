"""
Rendering and level-of-detail tests of BarGraphItem: view culling, outline LOD,
per-pixel-column aggregation and multi-colour drawing.

Counts of drawn rectangles and pixel comparisons are asserted, never timings.
"""
from collections.abc import Callable

import numpy as np
import pytest

import pyqtgraph as pg
from pyqtgraph.Qt import QT_LIB, QtCore, QtGui, QtWidgets, internals
from tests.perf_helpers import show_and_wait

app = pg.mkQApp()


class RecordingPainter(QtGui.QPainter):
    """
    QPainter recording the rectangle count, pen style and brush of each ``drawRects``.

    Parameters
    ----------
    device : QtGui.QPaintDevice
        Device to paint on.
    """

    def __init__(self, device: QtGui.QPaintDevice) -> None:
        super().__init__(device)
        self.calls = []

    def drawRects(self, *args) -> None:
        if len(args) == 2 and isinstance(args[1], int):
            count = args[1]  # PySide: pointer to the first rectangle, count
        else:
            count = len(args[0])
        self.calls.append((count, self.pen().style(), self.brush().color().rgba()))
        super().drawRects(*args)

    @property
    def rectCount(self) -> int:
        """int: Total number of rectangles drawn."""
        return sum(call[0] for call in self.calls)


def paint(item: QtWidgets.QGraphicsItem | None, rect: QtCore.QRectF,
          size: tuple[int, int] = (400, 300), antialias: bool = False,
          painter: Callable[[QtGui.QPainter], None] | None = None
          ) -> tuple[np.ndarray, RecordingPainter]:
    """
    Paint ``item`` (or call ``painter(p)``) into an image showing ``rect``.

    Parameters
    ----------
    item : QGraphicsItem or None
        Item whose ``paint`` method is called, unless ``painter`` is given.
    rect : QtCore.QRectF
        Item-coordinate rectangle mapped onto the whole image.
    size : tuple of int, default (400, 300)
        Image size.
    antialias : bool, default False
        Antialiasing render hint.
    painter : callable, optional
        Function drawing with the painter instead of ``item``.

    Returns
    -------
    image : numpy.ndarray
        Rendered pixels, shape (height, width, 4).
    recorder : RecordingPainter
        The painter used, with its recorded calls.
    """
    img = QtGui.QImage(size[0], size[1], QtGui.QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(0)
    p = RecordingPainter(img)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, antialias)
    p.scale(size[0] / rect.width(), size[1] / rect.height())
    p.translate(-rect.left(), -rect.top())
    if painter is None:
        item.paint(p, QtWidgets.QStyleOptionGraphicsItem(), None)
    else:
        painter(p)
    p.end()
    return pg.functions.ndarray_from_qimage(img).copy(), p


def referencePainter(x0: np.ndarray, y0: np.ndarray, width: np.ndarray,
                     height: np.ndarray, pens: list, brushes: list
                     ) -> Callable[[QtGui.QPainter], None]:
    """
    Straightforward painter drawing each bar, in order, with its own pen and brush.

    Parameters
    ----------
    x0, y0, width, height : numpy.ndarray
        Bar geometry.
    pens, brushes : list
        One pen and one brush specification per bar.

    Returns
    -------
    callable
        Function drawing the bars with a given painter.
    """
    y0 = np.minimum(y0, y0 + height)
    height = np.abs(height)

    def draw(p: QtGui.QPainter) -> None:
        for i in range(len(x0)):
            p.setPen(pg.mkPen(pens[i]))
            p.setBrush(pg.mkBrush(brushes[i]))
            p.drawRect(QtCore.QRectF(x0[i], y0[i], width[i], height[i]))
    return draw


def barData(n: int, width: float, seed: int = 3) -> tuple[np.ndarray, ...]:
    """
    Random bars at integer x positions, with positive and negative heights.

    Parameters
    ----------
    n : int
        Number of bars.
    width : float
        Width of every bar.
    seed : int, default 3
        Random seed.

    Returns
    -------
    tuple of numpy.ndarray
        x0, y0, width, height.
    """
    rng = np.random.default_rng(seed)
    x0 = np.arange(n, dtype=float) - width / 2
    y0 = rng.random(n) * 0.5 - 0.5
    height = rng.random(n) * 2 - 0.5
    return x0, y0, np.full(n, width), height


COLORS = [(255, 0, 0), (0, 200, 0), (0, 0, 255, 150)]
STYLES = {
    'mono pen': lambda c: dict(pen='w', brush='g'),
    'mono no pen': lambda c: dict(pen=None, brush='g'),
    'mono wide pen': lambda c: dict(pen=pg.mkPen('w', width=3), brush='g'),
    'mono data pen': lambda c: dict(pen=pg.mkPen('w', width=0.2, cosmetic=False), brush='g'),
    'brushes': lambda c: dict(pen='w', brushes=[COLORS[i] for i in c]),
    'brushes no pen': lambda c: dict(pen=None, brushes=[COLORS[i] for i in c]),
    'pens': lambda c: dict(pens=[COLORS[i] for i in c], brush='y'),
    'pens and brushes': lambda c: dict(pens=[COLORS[i] for i in c],
                                       brushes=[COLORS[(i + 1) % 3] for i in c]),
}


def styleLists(opts: dict, n: int) -> tuple[list, list]:
    """
    Per-bar pens and brushes of BarGraphItem options.

    Parameters
    ----------
    opts : dict
        Options with ``pen``/``pens`` and ``brush``/``brushes``.
    n : int
        Number of bars.

    Returns
    -------
    tuple of list
        Pens and brushes, one per bar.
    """
    pens = opts.get('pens') or [opts.get('pen')] * n
    brushes = opts.get('brushes') or [opts.get('brush')] * n
    return pens, brushes


@pytest.mark.parametrize('style', list(STYLES))
@pytest.mark.parametrize('width', [0.6, 1.0, 1.7], ids=['gap', 'touching', 'overlapping'])
@pytest.mark.parametrize('view', [(-2.0, 304.0), (100.3, 40.0), (57.71, 7.0), (269.5, 40.0)],
                         ids=['all', 'zoom40', 'zoom7', 'edge'])
@pytest.mark.parametrize('antialias', [False, True], ids=['aliased', 'antialiased'])
def test_pixel_identical_without_lod(style, width, view, antialias):
    n = 300
    x0, y0, w, height = barData(n, width)
    colorIndex = np.random.default_rng(5).integers(0, 3, n)
    opts = STYLES[style](colorIndex)
    hasOutline = 'pens' in opts or opts['pen'] is not None
    if hasOutline and 400 / view[1] * width < 2:
        pytest.skip('outline level of detail active at this zoom')
    item = pg.BarGraphItem(x0=x0, y0=y0, width=w, height=height, **opts)
    rect = QtCore.QRectF(view[0], -1.0, view[1], 3.0)
    image, _ = paint(item, rect, antialias=antialias)
    expected, _ = paint(None, rect, antialias=antialias,
                        painter=referencePainter(x0, y0, w, height, *styleLists(opts, n)))
    np.testing.assert_array_equal(image, expected)


def test_pixel_identical_in_plot_widget():
    n = 2000
    x0, y0, w, height = barData(n, 0.6)
    colorIndex = np.random.default_rng(1).integers(0, 3, n)
    opts = STYLES['pens and brushes'](colorIndex)
    reference = referencePainter(x0, y0, w, height, *styleLists(opts, n))

    class ReferenceBars(pg.BarGraphItem):
        def paint(self, p, *args):
            reference(p)

    def grab(item):
        pw = pg.PlotWidget()
        pw.resize(400, 300)
        pw.addItem(item)
        pw.setRange(xRange=(1000.3, 1040.3), yRange=(-1, 2), padding=0)
        show_and_wait(pw)
        image = pw.grab().toImage()
        pw.close()
        return pg.functions.ndarray_from_qimage(image).copy()

    bars = pg.BarGraphItem(x0=x0, y0=y0, width=w, height=height, **opts)
    with pytest.MonkeyPatch.context() as mp:
        recorder = []
        original = pg.BarGraphItem._drawGroups

        def drawGroups(self, painter, rects, groups, withPen):
            recorder.append(sum(end - begin for _, begin, end in groups))
            return original(self, painter, rects, groups, withPen)
        mp.setattr(pg.BarGraphItem, '_drawGroups', drawGroups)
        image = grab(bars)
    expected = grab(ReferenceBars(x0=x0, y0=y0, width=w, height=height, **opts))
    np.testing.assert_array_equal(image, expected)
    # culled to the view range of the ViewBox
    assert recorder and max(recorder) <= 42


@pytest.mark.parametrize('n', [1_000, 100_000])
def test_only_visible_bars_are_drawn(n):
    # bars span [i - 0.2, i + 0.2]; the view shows bars 101 to 300 with 8 px per bar
    item = pg.BarGraphItem(x=np.arange(n, dtype=float), height=np.ones(n), width=0.4,
                           pen='w', brush='g')
    _, recorder = paint(item, QtCore.QRectF(100.5, 0, 200, 1), size=(1600, 100))
    assert recorder.rectCount == 200


def test_unsorted_bars_are_all_drawn():
    n = 500
    x = np.random.default_rng(0).permutation(n).astype(float)
    item = pg.BarGraphItem(x=x, height=np.ones(n), width=0.4, pen='w', brush='g')
    _, recorder = paint(item, QtCore.QRectF(100.5, 0, 200, 1), size=(1600, 100))
    assert recorder.rectCount == n


def test_bars_outside_view_with_outline_margin():
    # a wide cosmetic pen extends beyond the bar: the bar just left of the view stays
    item = pg.BarGraphItem(x=np.arange(1000.0), height=np.ones(1000), width=0.4,
                           pen=pg.mkPen('w', width=9), brush='g')
    _, recorder = paint(item, QtCore.QRectF(100.5, 0, 200, 1), size=(1600, 100))
    assert recorder.rectCount == 202


def test_lodPen():
    n = 2000
    x = np.arange(n, dtype=float)
    view = QtCore.QRectF(0, 0, 500, 1)  # 3.2 px per unit: bars of 1.92 px

    item = pg.BarGraphItem(x=x, height=np.ones(n), width=0.6, pen='w', brush='g')
    _, recorder = paint(item, view, size=(1600, 100))
    assert recorder.calls
    assert all(style == QtCore.Qt.PenStyle.NoPen for _, style, _ in recorder.calls)

    item.setOpts(lodPen=False)
    _, recorder = paint(item, view, size=(1600, 100))
    assert all(style == QtCore.Qt.PenStyle.SolidLine for _, style, _ in recorder.calls)

    # bars of 2.4 px keep their outline
    item.setOpts(lodPen=True)
    _, recorder = paint(item, QtCore.QRectF(0, 0, 400, 1), size=(1600, 100))
    assert all(style == QtCore.Qt.PenStyle.SolidLine for _, style, _ in recorder.calls)


def test_lodPen_keeps_thin_bars_visible():
    # bars of 0.3 px, one every 3 px: without outline they must stay one pixel wide
    n = 400
    item = pg.BarGraphItem(x0=np.arange(n) + 0.35, height=np.ones(n), width=0.1,
                           pen='w', brush='g')
    image, recorder = paint(item, QtCore.QRectF(0, 0, n, 1), size=(1200, 10))
    assert all(style == QtCore.Qt.PenStyle.NoPen for _, style, _ in recorder.calls)
    columns = np.flatnonzero(image.any(axis=-1).any(axis=0))
    assert len(columns) == n


def test_aggregation_envelope():
    # 10 bars per pixel column, no bar straddles two columns
    n, perColumn = 10_000, 10
    rng = np.random.default_rng(2)
    y0 = rng.integers(0, 50, n).astype(float)
    height = rng.integers(1, 50, n).astype(float)
    item = pg.BarGraphItem(x0=np.arange(n, dtype=float), width=0.5, y0=y0, height=height,
                           pen=None, brush='g')
    image, recorder = paint(item, QtCore.QRectF(0, 0, n, 100), size=(n // perColumn, 100))
    assert recorder.rectCount == n // perColumn

    ylo = y0.reshape(-1, perColumn).min(axis=1).astype(int)
    yhi = (y0 + height).reshape(-1, perColumn).max(axis=1).astype(int)
    rows = np.arange(100)[:, None]
    expected = (rows >= ylo[None, :]) & (rows < yhi[None, :])
    np.testing.assert_array_equal(image.any(axis=-1), expected)

    item.setOpts(lod=False)
    _, recorder = paint(item, QtCore.QRectF(0, 0, n, 100), size=(n // perColumn, 100))
    assert recorder.rectCount == n


def test_multicolor_draws_runs_in_order():
    n = 1000
    brushes = ['r'] * 3 + ['g'] * 5 + ['b'] * 2  # 3 runs per block of 10 bars
    item = pg.BarGraphItem(x=np.arange(n, dtype=float), height=np.ones(n), width=0.6,
                           pen='w', brushes=brushes * (n // 10))
    # bars 100 to 199 visible, 16 px each: no level of detail
    _, recorder = paint(item, QtCore.QRectF(99.5, 0, 100, 1), size=(1600, 10))
    assert recorder.rectCount == 100
    assert len(recorder.calls) == 30


def test_multicolor_lod_draws_one_call_per_style():
    n = 20_000
    rng = np.random.default_rng(0)
    # distinct tuple objects: styles are merged by value
    brushes = [(255, 0, 0) if up else (0, 255, 0) for up in rng.random(n) > 0.5]
    item = pg.BarGraphItem(x=np.arange(n, dtype=float), height=rng.random(n), width=0.6,
                           pen='w', brushes=brushes)
    image, recorder = paint(item, QtCore.QRectF(-0.5, 0, n, 1), size=(1000, 50))
    assert len(recorder.calls) == 2
    assert recorder.rectCount == 1000
    # every pixel column is drawn
    assert image.any(axis=-1).any(axis=0).all()


def test_dataBounds_include_widest_pen():
    pens = [pg.mkPen('w', width=4, cosmetic=False), pg.mkPen('w', width=1, cosmetic=False)]
    item = pg.BarGraphItem(x=[0.0, 1.0], height=[1.0, 1.0], width=0.5, pens=pens)
    assert item.dataBounds(0) == (-0.25 - 2, 1.25 + 2)


def test_setOpts_updates_rendering():
    item = pg.BarGraphItem(x=np.arange(100.0), height=np.ones(100), width=0.5, brush='r',
                           pen=None)
    view = QtCore.QRectF(-0.5, 0, 100, 2)
    paint(item, view)
    item.setOpts(height=np.full(100, 2.0), brush='b')
    image, _ = paint(item, view)
    assert (image[:, :, 0] > 0).any()  # blue channel of ARGB32 (little endian BGRA)
    assert not (image[:, :, 2] > 0).any()


@pytest.mark.parametrize('use_array', [None, False])
def test_primitivearray_sliced_drawargs(use_array):
    arr = internals.PrimitiveArray(QtCore.QRectF, 4, use_array=use_array)
    arr.resize(10)
    memory = arr.ndarray()
    memory[:, 0] = np.arange(10) * 10
    memory[:, 1] = 0
    memory[:, 2] = 5
    memory[:, 3] = 10

    def draw(args):
        img = QtGui.QImage(100, 10, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(0)
        p = QtGui.QPainter(img)
        p.setPen(QtGui.QPen(QtCore.Qt.PenStyle.NoPen))
        p.setBrush(QtGui.QBrush(QtGui.QColor('white')))
        p.drawRects(*args)
        p.end()
        return np.flatnonzero(pg.functions.ndarray_from_qimage(img)[5, :, 0])

    np.testing.assert_array_equal(draw(arr.drawargs()), draw(arr.drawargs(0, 10)))
    expected = np.concatenate([np.arange(5) + 10 * i for i in range(3, 7)])
    np.testing.assert_array_equal(draw(arr.drawargs(3, 7)), expected)
    np.testing.assert_array_equal(draw(arr.drawargs(-2, None)),
                                  np.concatenate([np.arange(80, 85), np.arange(90, 95)]))
    assert len(draw(arr.drawargs(5, 5))) == 0
    assert len(draw(arr.drawargs(7, 3))) == 0


@pytest.mark.skipif(not QT_LIB.startswith('PyQt'), reason='sip specific')
def test_primitivearray_sliced_drawargs_old_sip(monkeypatch):
    from pyqtgraph.Qt import sip

    class OldSip:
        SIP_VERSION = 0x60700
        array = staticmethod(sip.array)
        voidptr = staticmethod(sip.voidptr)

    arr = internals.PrimitiveArray(QtCore.QRectF, 4)
    if not arr.use_sip_array:
        pytest.skip('sip.array not used')
    arr.resize(6)
    arr.ndarray()[:] = np.arange(24, dtype=float).reshape(6, 4)
    monkeypatch.setattr(internals, 'sip', OldSip)
    (sliced,) = arr.drawargs(2, 5)
    assert [r.getRect() for r in sliced] == [tuple(range(4 * i, 4 * i + 4)) for i in (2, 3, 4)]
