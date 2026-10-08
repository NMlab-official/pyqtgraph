"""
Reproducible end-to-end performance scenarios (S01 to S12).

The scenario identifiers match ``PERFORMANCE_PLAN.md`` at the root of the repository.
Every scenario prints one line per variant with the median duration and, where
relevant, call counters (paints per update, ``setData`` calls, ...).

Usage::

    QT_QPA_PLATFORM=offscreen python benchmarks/scenarios.py            # all scenarios
    QT_QPA_PLATFORM=offscreen python benchmarks/scenarios.py S05 S06    # a selection
    QT_QPA_PLATFORM=offscreen python benchmarks/scenarios.py --full     # include 1e7 sizes

The numbers are wall-clock medians and depend on the machine, the Qt binding and the
Qt platform plugin. Compare runs made on the same machine only.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import statistics
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import pyqtgraph as pg
from pyqtgraph.graphicsItems.NonUniformImage import NonUniformImage
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

__all__ = ['SCENARIOS', 'Result', 'run', 'main']

_T0 = 1.7e9  # an epoch timestamp, so that DateAxisItem shows realistic dates


@dataclass
class Result:
    """
    Outcome of one scenario variant.

    Parameters
    ----------
    name : str
        Scenario identifier and variant, e.g. ``"S05[300pts]"``.
    value : float
        Median measured value.
    unit : str
        Unit of ``value``.
    counters : dict[str, float]
        Optional per-iteration call counters.
    """

    name: str
    value: float
    unit: str
    counters: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        extra = ' '.join(f'{k}={v:.2f}' for k, v in self.counters.items())
        return f'{self.name:40s} {self.value:10.2f} {self.unit:12s} {extra}'


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def _app() -> QtWidgets.QApplication:
    """
    Return the running QApplication, creating it if needed.

    Returns
    -------
    QtWidgets.QApplication
        The application instance.
    """
    return pg.mkQApp()


def _process(n: int = 3) -> None:
    """
    Process pending Qt events, including queued paint events.

    Parameters
    ----------
    n : int, default 3
        Number of ``processEvents`` passes.
    """
    app = _app()
    for _ in range(n):
        app.processEvents()


def _timed(fn: Callable[[], object], repeat: int = 5, warmup: int = 2,
           budget_s: float = 20.0) -> tuple[float, int]:
    """
    Time ``fn`` and return the median duration and the total number of calls.

    The number of repetitions is reduced automatically when a single call is slow, so
    that a scenario never runs much longer than ``budget_s``.

    Parameters
    ----------
    fn : callable
        Function to time; called without arguments.
    repeat : int, default 5
        Maximum number of timed calls.
    warmup : int, default 2
        Untimed calls made first.
    budget_s : float, default 20.0
        Approximate time budget in seconds.

    Returns
    -------
    median_ms : float
        Median duration of the timed calls, in milliseconds.
    ncalls : int
        Number of calls made, warmup included (used to normalise call counters).
    """
    times: list[float] = []
    ncalls = 0
    t_start = time.perf_counter()
    for i in range(warmup + repeat):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        ncalls += 1
        if i >= warmup or dt > budget_s / 4:
            times.append(dt)
        if times and time.perf_counter() - t_start > budget_s:
            break
    return statistics.median(times) * 1e3, ncalls


def _median_ms(fn: Callable[[], object], repeat: int = 5, warmup: int = 2,
               budget_s: float = 20.0) -> float:
    """
    Time ``fn`` and return the median duration.

    Parameters
    ----------
    fn : callable
        Function to time; called without arguments.
    repeat : int, default 5
        Maximum number of timed calls.
    warmup : int, default 2
        Untimed calls made first.
    budget_s : float, default 20.0
        Approximate time budget in seconds.

    Returns
    -------
    float
        Median duration in milliseconds.
    """
    return _timed(fn, repeat, warmup, budget_s)[0]


class CallCounter:
    """
    Count calls of a method on a class while active.

    Parameters
    ----------
    cls : type
        Class whose method is wrapped.
    name : str
        Method name.
    """

    def __init__(self, cls: type, name: str) -> None:
        self._cls = cls
        self._name = name
        self._orig = cls.__dict__.get(name)
        self._count = 0

    @property
    def count(self) -> int:
        """int: Number of calls observed so far."""
        return self._count

    def reset(self) -> None:
        """Reset the call count to zero."""
        self._count = 0

    def __enter__(self) -> CallCounter:
        orig = getattr(self._cls, self._name)
        counter = self

        def wrapper(*args, **kwargs):
            counter._count += 1
            return orig(*args, **kwargs)

        setattr(self._cls, self._name, wrapper)
        return self

    def __exit__(self, *exc) -> None:
        if self._orig is None:
            delattr(self._cls, self._name)
        else:
            setattr(self._cls, self._name, self._orig)


@contextlib.contextmanager
def _counters(*specs: tuple[type, str]) -> Iterator[list[CallCounter]]:
    """
    Activate several :class:`CallCounter` at once.

    Parameters
    ----------
    *specs : tuple of (type, str)
        Class and method name pairs.

    Yields
    ------
    list of CallCounter
        The active counters, in the order given.
    """
    with contextlib.ExitStack() as stack:
        yield [stack.enter_context(CallCounter(c, n)) for c, n in specs]


def _plot_widget(width: int = 1000, height: int = 600) -> pg.PlotWidget:
    """
    Create, show and lay out a PlotWidget.

    Parameters
    ----------
    width, height : int
        Widget size in pixels.

    Returns
    -------
    PlotWidget
        The visible widget.
    """
    pw = pg.PlotWidget()
    pw.resize(width, height)
    pw.show()
    _process(5)
    return pw


def _render_item(item: QtWidgets.QGraphicsItem, rect: QtCore.QRectF,
                 size: tuple[int, int] = (1600, 900)) -> Callable[[], None]:
    """
    Build a function painting ``item`` into an image, mapping ``rect`` to the image.

    Parameters
    ----------
    item : QGraphicsItem
        Item to paint (its ``paint`` method is called directly).
    rect : QRectF
        Item-coordinate rectangle mapped onto the full image.
    size : tuple of int, default (1600, 900)
        Image size.

    Returns
    -------
    callable
        Function performing one paint.
    """
    img = QtGui.QImage(size[0], size[1], QtGui.QImage.Format.Format_ARGB32_Premultiplied)

    def paint() -> None:
        img.fill(0)
        p = QtGui.QPainter(img)
        p.scale(size[0] / rect.width(), size[1] / rect.height())
        p.translate(-rect.left(), -rect.top())
        item.paint(p, QtWidgets.QStyleOptionGraphicsItem(), None)
        p.end()

    return paint


# --------------------------------------------------------------------------------------
# scenarios
# --------------------------------------------------------------------------------------

def s01_streaming_line(full: bool) -> list[Result]:
    """
    S01: append one point per frame to a long line, then setData and render.

    Parameters
    ----------
    full : bool
        Include 1e7 points.

    Returns
    -------
    list of Result
        One result per (size, mode).
    """
    results = []
    sizes = [1_000_000, 10_000_000] if full else [1_000_000]
    rng = np.random.default_rng(0)
    for n in sizes:
        x = np.arange(n + 1000, dtype=np.float64)
        y = np.cumsum(rng.standard_normal(n + 1000))
        for mode in ('default', 'autoDownsample', 'clip+autoDownsample'):
            pw = _plot_widget()
            item = pw.plot(pen='y')
            if mode != 'default':
                pw.getPlotItem().setDownsampling(auto=True, mode='peak')
            if mode == 'clip+autoDownsample':
                pw.getPlotItem().setClipToView(True)
                pw.getPlotItem().enableAutoRange(x=False, y=True)
            state = {'k': n}

            def frame() -> None:
                k = state['k'] = state['k'] + 1
                item.setData(x[:k], y[:k])
                if mode == 'clip+autoDownsample':
                    pw.setXRange(x[k - 5000], x[k - 1], padding=0)
                _process(2)

            frame()
            with _counters((pg.PlotCurveItem, 'paint')) as (paints,):
                ms, nframes = _timed(frame, repeat=7)
            results.append(Result(f'S01[{n:.0e},{mode}]', ms, 'ms/frame',
                                  {'paints/frame': paints.count / nframes}))
            pw.close()
    return results


def s02_pan_y(full: bool) -> list[Result]:
    """
    S02: step-wise vertical pan over a long line with default options.

    Parameters
    ----------
    full : bool
        Include 1e7 points.

    Returns
    -------
    list of Result
        One result per size.
    """
    results = []
    sizes = [1_000_000, 10_000_000] if full else [1_000_000]
    rng = np.random.default_rng(0)
    for n in sizes:
        pw = _plot_widget()
        item = pw.plot(np.arange(n, dtype=np.float64), np.cumsum(rng.standard_normal(n)))
        _process(5)
        pw.getPlotItem().enableAutoRange(False)
        vb = pw.getViewBox()

        def step() -> None:
            vb.translateBy(y=0.01 * (vb.viewRange()[1][1] - vb.viewRange()[1][0]))
            _process(2)

        with _counters((pg.PlotCurveItem, 'setData'), (pg.PlotCurveItem, 'paint')) as (sd, paints):
            ms, steps = _timed(step, repeat=7)
        results.append(Result(f'S02[{n:.0e}]', ms, 'ms/step',
                              {'curve.setData/step': sd.count / steps,
                               'paints/step': paints.count / steps}))
        pw.close()
        del item
    return results


def _scatter_kwargs(variant: str, n: int, rng: np.random.Generator) -> dict:
    """
    Build ScatterPlotItem.setData keyword arguments for a variant of S03.

    Parameters
    ----------
    variant : str
        One of ``uniform``, ``5 QBrush``, ``tuples``, ``float sizes``.
    n : int
        Number of points.
    rng : numpy.random.Generator
        Random generator.

    Returns
    -------
    dict
        Keyword arguments for ``setData``.
    """
    kw: dict = {'x': rng.random(n), 'y': rng.random(n), 'size': 7, 'pen': None}
    if variant == '5 QBrush':
        palette = [pg.mkBrush(c) for c in ('r', 'g', 'b', 'y', 'c')]
        kw['brush'] = [palette[i] for i in rng.integers(0, 5, n)]
    elif variant == 'tuples':
        colors = [(255, 0, 0, 200), (0, 255, 0, 200), (0, 0, 255, 200)]
        kw['brush'] = [colors[i] for i in rng.integers(0, 3, n)]
    elif variant == 'float sizes':
        kw['size'] = rng.uniform(5, 15, n)
    return kw


def s03_scatter_setdata(full: bool) -> list[Result]:
    """
    S03: ScatterPlotItem.setData cost for several styling variants.

    Parameters
    ----------
    full : bool
        Include 1e6 points.

    Returns
    -------
    list of Result
        One result per (size, variant).
    """
    results = []
    sizes = [100_000, 1_000_000] if full else [100_000]
    rng = np.random.default_rng(0)
    for n in sizes:
        for variant in ('uniform', '5 QBrush', 'tuples', 'float sizes'):
            if n > 100_000 and variant in ('tuples', 'float sizes'):
                continue
            item = pg.ScatterPlotItem()
            kw = _scatter_kwargs(variant, n, rng)
            ms = _median_ms(lambda: item.setData(**kw), repeat=3, warmup=1, budget_s=30)
            results.append(Result(f'S03[{n:.0e},{variant}]', ms, 'ms/setData',
                                  {'atlas entries': float(len(item.fragmentAtlas))}))
    return results


def s04_scatter_interaction(full: bool) -> list[Result]:
    """
    S04: first hover after setData, pointsAt and vertical pan of a scatter plot.

    Parameters
    ----------
    full : bool
        Include 1e6 points.

    Returns
    -------
    list of Result
        Results for hover, pointsAt and pan.
    """
    results = []
    sizes = [100_000, 1_000_000] if full else [100_000]
    rng = np.random.default_rng(0)
    for n in sizes:
        x, y = rng.random(n), rng.random(n)
        item = pg.ScatterPlotItem(x=x, y=y, size=7, hoverable=True)

        def first_points_at() -> None:
            item.setData(x=x, y=y, size=7, hoverable=True)
            item.pointsAt(QtCore.QPointF(0.5, 0.5))

        def setdata_only() -> None:
            item.setData(x=x, y=y, size=7, hoverable=True)

        base = _median_ms(setdata_only, repeat=3, warmup=1)
        both = _median_ms(first_points_at, repeat=3, warmup=1)
        results.append(Result(f'S04[{n:.0e},first pointsAt]', max(both - base, 0.0), 'ms'))

        pw = _plot_widget()
        palette = [pg.mkBrush('r'), pg.mkBrush('g')]
        pdi = pw.plot(x, y, pen=None, symbol='o', symbolSize=5,
                      symbolBrush=[palette[i] for i in rng.integers(0, 2, n)])
        _process(5)
        pw.getPlotItem().enableAutoRange(False)
        vb = pw.getViewBox()

        def pan() -> None:
            vb.translateBy(y=0.01)
            _process(2)

        with _counters((pg.ScatterPlotItem, 'setData')) as (sd,):
            ms, steps = _timed(pan, repeat=5)
        results.append(Result(f'S04[{n:.0e},PlotDataItem pan y]', ms, 'ms/step',
                              {'scatter.setData/step': sd.count / steps}))
        pw.close()
        del pdi
    return results


def _linked_layout(npts: int, nlines: int = 0, ntext: int = 0,
                   ignore_bounds: bool = False) -> tuple:
    """
    Build the 3 linked date-axis plots used by S05 and S06.

    Parameters
    ----------
    npts : int
        Points per curve.
    nlines, ntext : int
        Number of horizontal InfiniteLines and TextItems added to the price plot.
    ignore_bounds : bool
        Add the lines and texts with ``ignoreBounds=True``.

    Returns
    -------
    tuple
        ``(widget, plots, curves, volume, indicator, x)``.
    """
    w = pg.GraphicsLayoutWidget(size=(1400, 900))
    plots = []
    for i in range(3):
        p = w.addPlot(row=i, col=0, axisItems={'bottom': pg.DateAxisItem()})
        if i > 0:
            p.setXLink(plots[0])
        plots.append(p)
    rng = np.random.default_rng(0)
    x = _T0 + 60.0 * np.arange(npts)
    curves = [plots[0].plot(x, 100 + np.cumsum(rng.normal(size=npts)) + k,
                            pen=pg.intColor(k, 20)) for k in range(20)]
    vol = plots[1].plot(x, rng.random(npts) * 1000, pen='w', fillLevel=0,
                        brush=(100, 100, 255, 100))
    ind = plots[2].plot(x, rng.normal(size=npts), pen='y')
    for j in range(nlines):
        plots[0].addItem(pg.InfiniteLine(pos=100 + (j - nlines / 2) * 0.2, angle=0),
                         ignoreBounds=ignore_bounds)
    for j in range(ntext):
        t = pg.TextItem(f'order {j}', anchor=(0, 0.5))
        t.setPos(x[-1], 100 + (j - ntext / 2) * 0.2)
        plots[0].addItem(t, ignoreBounds=ignore_bounds)
    w.show()
    _process(5)
    return w, plots, curves, vol, ind, x


def s05_linked_streaming(full: bool) -> list[Result]:
    """
    S05: 3 linked plots with DateAxisItem and 22 streaming curves.

    Parameters
    ----------
    full : bool
        Unused, kept for a uniform signature.

    Returns
    -------
    list of Result
        One result per variant, with paints per update.
    """
    results = []
    variants = [('300pts', 300, 0, 0, 0.0), ('2000pts trend', 2000, 0, 0, 0.5),
                ('300pts+400 items', 300, 200, 200, 0.0)]
    for label, npts, nlines, ntext, trend in variants:
        w, plots, curves, vol, ind, x = _linked_layout(npts, nlines, ntext)
        rng = np.random.default_rng(1)
        ys = [c.yData.copy() for c in curves]
        vy, iy = vol.yData.copy(), ind.yData.copy()
        state = {'i': 0}

        def update() -> None:
            i = state['i'] = state['i'] + 1
            xx = x + 60 * i
            for k, c in enumerate(curves):
                yk = ys[k]
                yk[:-1] = yk[1:]
                yk[-1] = yk[-2] + rng.normal() + trend
                c.setData(xx, yk)
            vy[:-1] = vy[1:]
            vy[-1] = rng.random() * 1000
            vol.setData(xx, vy)
            iy[:-1] = iy[1:]
            iy[-1] = rng.normal()
            ind.setData(xx, iy)
            _process(3)

        with _counters((pg.GraphicsView, 'paintEvent')) as (paints,):
            ms, nupd = _timed(update, repeat=20, warmup=3)
        results.append(Result(f'S05[{label}]', ms, 'ms/update',
                              {'paints/update': paints.count / nupd}))
        w.close()
    return results


def s06_crosshair(full: bool) -> list[Result]:
    """
    S06: crosshair following real mouse-move events over the S05 layout.

    One vertical InfiniteLine per plot, one horizontal line and a TextItem are added
    with ``ignoreBounds=True`` and moved from ``sigMouseMoved``, as in a trading UI.

    Parameters
    ----------
    full : bool
        Unused, kept for a uniform signature.

    Returns
    -------
    list of Result
        Duration per mouse move.
    """
    old_limit = pg.getConfigOption('mouseRateLimit')
    pg.setConfigOptions(mouseRateLimit=-1)
    w, plots, curves, vol, ind, x = _linked_layout(2000)
    vlines = []
    for p in plots:
        v = pg.InfiniteLine(angle=90, movable=False, pen='c')
        p.addItem(v, ignoreBounds=True)
        vlines.append(v)
    hline = pg.InfiniteLine(angle=0, movable=False, pen='c')
    plots[0].addItem(hline, ignoreBounds=True)
    label = pg.TextItem('', anchor=(0, 1), fill=(0, 0, 0, 150))
    plots[0].addItem(label, ignoreBounds=True)
    vb0 = plots[0].vb

    def mouse_moved(pos: QtCore.QPointF) -> None:
        if plots[0].sceneBoundingRect().contains(pos):
            mp = vb0.mapSceneToView(pos)
            for v in vlines:
                v.setPos(mp.x())
            hline.setPos(mp.y())
            label.setText(f'{mp.x():.0f}  {mp.y():.2f}')
            label.setPos(mp.x(), mp.y())

    w.scene().sigMouseMoved.connect(mouse_moved)
    viewport = w.viewport()
    _process(5)
    geo = vb0.sceneBoundingRect()
    state = {'i': 0}

    def move() -> None:
        i = state['i'] = state['i'] + 1
        px = geo.left() + 20 + (i * 7) % int(geo.width() - 40)
        py = geo.top() + 20 + (i * 3) % int(geo.height() - 40)
        gp = viewport.mapToGlobal(QtCore.QPoint(int(px), int(py)))
        ev = QtGui.QMouseEvent(QtCore.QEvent.Type.MouseMove, QtCore.QPointF(px, py),
                               QtCore.QPointF(gp), QtCore.Qt.MouseButton.NoButton,
                               QtCore.Qt.MouseButton.NoButton,
                               QtCore.Qt.KeyboardModifier.NoModifier)
        QtWidgets.QApplication.sendEvent(viewport, ev)
        _process(1)

    with _counters((pg.PlotCurveItem, 'paint')) as (paints,):
        ms, nmoves = _timed(move, repeat=60, warmup=10)
    w.close()
    pg.setConfigOptions(mouseRateLimit=old_limit)
    return [Result('S06[crosshair 2000pts]', ms, 'ms/move',
                   {'curve paints/move': paints.count / nmoves})]


def s07_many_curves_zoom(full: bool) -> list[Result]:
    """
    S07: vertical zoom steps with 500 curves.

    Parameters
    ----------
    full : bool
        Unused, kept for a uniform signature.

    Returns
    -------
    list of Result
        Duration per zoom step.
    """
    pw = _plot_widget()
    rng = np.random.default_rng(0)
    for k in range(500):
        pw.plot(np.cumsum(rng.normal(size=200)) + k * 0.1, pen=pg.intColor(k, 500))
    _process(5)
    pw.getPlotItem().enableAutoRange(False)
    vb = pw.getViewBox()
    state = {'s': 0.98}

    def zoom() -> None:
        state['s'] = 1 / state['s']
        vb.scaleBy(y=state['s'])
        _process(2)

    with _counters((pg.PlotCurveItem, 'setData')) as (sd,):
        ms, steps = _timed(zoom, repeat=7)
    pw.close()
    return [Result('S07[500 curves zoom y]', ms, 'ms/step',
                   {'curve.setData/step': sd.count / steps})]


def s08_bars(full: bool) -> list[Result]:
    """
    S08: BarGraphItem paint cost, full view and zoomed view.

    Parameters
    ----------
    full : bool
        Unused, kept for a uniform signature.

    Returns
    -------
    list of Result
        One result per variant.
    """
    results = []
    rng = np.random.default_rng(0)
    for n in (20_000, 500_000):
        x = np.arange(n, dtype=np.float64)
        h = rng.random(n)
        for pen in ('d', None):
            bars = pg.BarGraphItem(x=x, height=h, width=0.6, brush='g', pen=pen)
            if n == 20_000:
                full_view = _render_item(bars, QtCore.QRectF(-1, 0, n + 1, 1.0))
                results.append(Result(f'S08[{n:.0e},pen={pen},full view]',
                                      _median_ms(full_view, repeat=3), 'ms/paint'))
            zoomed = _render_item(bars, QtCore.QRectF(n - 200, 0, 200, 1.0))
            results.append(Result(f'S08[{n:.0e},pen={pen},200 visible]',
                                  _median_ms(zoomed, repeat=5), 'ms/paint'))
    return results


def s09_fill_between(full: bool) -> list[Result]:
    """
    S09: FillBetweenItem update when both bounding curves change.

    Parameters
    ----------
    full : bool
        Unused, kept for a uniform signature.

    Returns
    -------
    list of Result
        Duration per update with the number of path rebuilds.
    """
    n = 200_000
    x = np.arange(n, dtype=np.float64)
    y = np.cumsum(np.random.default_rng(0).standard_normal(n))
    pw = _plot_widget()
    upper = pw.plot(x, y + 2)
    lower = pw.plot(x, y - 2)
    fill = pg.FillBetweenItem(upper, lower, brush=(50, 50, 200, 80))
    pw.addItem(fill)
    _process(5)
    state = {'d': 2.0}

    def update() -> None:
        state['d'] = 4.0 - state['d']
        upper.setData(x, y + state['d'])
        lower.setData(x, y - state['d'])
        _process(2)

    with _counters((pg.FillBetweenItem, 'updatePath')) as (up,):
        ms, nupd = _timed(update, repeat=5)
    pw.close()
    return [Result('S09[FillBetween 2e5]', ms, 'ms/update', {'updatePath/update': up.count / nupd})]


def s10_heatmap(full: bool) -> list[Result]:
    """
    S10: ImageItem setImage and render of a 2000x1000 float32 heatmap.

    Parameters
    ----------
    full : bool
        Unused, kept for a uniform signature.

    Returns
    -------
    list of Result
        One result per variant.
    """
    results = []
    rng = np.random.default_rng(0)
    base = rng.random((2000, 1000), dtype=np.float32)
    cmap = pg.colormap.get('viridis')
    variants = [('no NaN, lut256', 0.0, 256), ('0.04% NaN, lut256', 0.0004, 256),
                ('30% NaN, lut256', 0.3, 256), ('no NaN, lut512', 0.0, 512)]
    for label, nan_frac, npts in variants:
        data = base.copy()
        if nan_frac:
            data[rng.random(data.shape) < nan_frac] = np.nan
        item = pg.ImageItem(axisOrder='row-major')
        lut = cmap.getLookupTable(nPts=npts)

        def update() -> None:
            item.setImage(data, autoLevels=False, levels=(0, 1), lut=lut)
            item.render()

        results.append(Result(f'S10[{label}]', _median_ms(update, repeat=7), 'ms/update'))
    return results


def s11_legend(full: bool) -> list[Result]:
    """
    S11: add 500 named curves to a plot with a legend.

    Parameters
    ----------
    full : bool
        Unused, kept for a uniform signature.

    Returns
    -------
    list of Result
        Total duration with and without legend.
    """
    results = []
    data = np.random.default_rng(0).normal(size=100)
    for legend in (True, False):
        pw = _plot_widget()
        if legend:
            pw.addLegend()
        t0 = time.perf_counter()
        for k in range(500):
            pw.plot(data + k, name=f'curve {k}')
        _process(2)
        results.append(Result(f'S11[500 curves, legend={legend}]',
                              (time.perf_counter() - t0) * 1e3, 'ms'))
        pw.close()
    return results


def s12_non_uniform(full: bool) -> list[Result]:
    """
    S12: NonUniformImage and PColorMeshItem update and paint.

    Parameters
    ----------
    full : bool
        Use the 2000x1000 NonUniformImage size (slow before optimisation).

    Returns
    -------
    list of Result
        Update and paint durations.
    """
    results = []
    rng = np.random.default_rng(0)
    nx, ny = (2000, 1000) if full else (500, 250)
    x = np.cumsum(rng.uniform(0.5, 1.5, nx))
    y = np.cumsum(rng.uniform(0.5, 1.5, ny))
    z = rng.random((nx, ny))
    lut = pg.colormap.get('viridis').getLookupTable(nPts=256)
    holder: dict = {}

    def update() -> None:
        item = NonUniformImage(x, y, z)
        item.setLookupTable(lut)
        item.generatePicture()
        holder['item'] = item

    ms_update = _median_ms(update, repeat=3, warmup=0, budget_s=30)
    item = holder['item']
    paint = _render_item(item, item.boundingRect())
    results.append(Result(f'S12[NonUniformImage {nx}x{ny} update]', ms_update, 'ms'))
    results.append(Result(f'S12[NonUniformImage {nx}x{ny} paint]',
                          _median_ms(paint, repeat=3, warmup=1, budget_s=30), 'ms'))
    xm, ym = np.meshgrid(np.arange(401.0), np.arange(201.0), indexing='ij')
    zm = rng.random((400, 200))
    mesh = pg.PColorMeshItem()
    results.append(Result('S12[PColorMeshItem 400x200 setData]',
                          _median_ms(lambda: mesh.setData(xm, ym, zm), repeat=3, warmup=1),
                          'ms'))
    return results


SCENARIOS: dict[str, Callable[[bool], list[Result]]] = {
    'S01': s01_streaming_line,
    'S02': s02_pan_y,
    'S03': s03_scatter_setdata,
    'S04': s04_scatter_interaction,
    'S05': s05_linked_streaming,
    'S06': s06_crosshair,
    'S07': s07_many_curves_zoom,
    'S08': s08_bars,
    'S09': s09_fill_between,
    'S10': s10_heatmap,
    'S11': s11_legend,
    'S12': s12_non_uniform,
}


def run(names: list[str], full: bool = False) -> list[Result]:
    """
    Run scenarios and print their results as they complete.

    Parameters
    ----------
    names : list of str
        Scenario identifiers; all scenarios when empty.
    full : bool, default False
        Include the largest (slow) sizes.

    Returns
    -------
    list of Result
        All results, in execution order.
    """
    _app()
    pg.setConfigOptions(antialias=False)
    selected = names or list(SCENARIOS)
    results: list[Result] = []
    for name in selected:
        for res in SCENARIOS[name.upper()](full):
            print(res, flush=True)
            results.append(res)
    return results


def main(argv: list[str] | None = None) -> int:
    """
    Command line entry point.

    Parameters
    ----------
    argv : list of str, optional
        Command line arguments, ``sys.argv[1:]`` by default.

    Returns
    -------
    int
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('scenarios', nargs='*', help='scenario ids (S01 ... S12)')
    parser.add_argument('--full', action='store_true', help='include the largest sizes')
    args = parser.parse_args(argv)
    unknown = [s for s in args.scenarios if s.upper() not in SCENARIOS]
    if unknown:
        parser.error(f'unknown scenario(s): {", ".join(unknown)}')
    _app()
    print(f'pyqtgraph {pg.__version__}, {pg.Qt.QT_LIB} {QtCore.qVersion()}, '
          f'numpy {np.__version__}, platform {QtGui.QGuiApplication.platformName() or "?"}')
    run(args.scenarios, full=args.full)
    return 0


if __name__ == '__main__':
    sys.exit(main())
