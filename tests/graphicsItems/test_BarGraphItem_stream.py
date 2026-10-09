"""
Tests of BarGraphItem.appendData: bars appended while streaming, or replacing the last
bar, compared with setting all bars at once (state, options, bounds and pixels), and
the incremental work it does (style de-duplication, level of detail, growth buffers).

Values, call counts and pixels are asserted, never timings.
"""
import csv

import numpy as np
import pytest

import pyqtgraph as pg
from pyqtgraph.exporters import CSVExporter
from pyqtgraph.graphicsItems import BarGraphItem as barModule
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from tests.perf_helpers import show_and_wait

app = pg.mkQApp()

COLORS = [(38, 166, 154), (239, 83, 80), (0, 0, 255, 150)]


def render(item: pg.BarGraphItem, rect: QtCore.QRectF, size: tuple[int, int] = (400, 120),
           antialias: bool = False) -> np.ndarray:
    """
    Paint an item into an image showing a rectangle of item coordinates.

    Parameters
    ----------
    item : BarGraphItem
        Item whose ``paint`` method is called.
    rect : QtCore.QRectF
        Item-coordinate rectangle mapped onto the whole image.
    size : tuple of int, default (400, 120)
        Image size.
    antialias : bool, default False
        Antialiasing render hint.

    Returns
    -------
    numpy.ndarray
        Rendered pixels, shape (height, width, 4).
    """
    img = QtGui.QImage(size[0], size[1], QtGui.QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(0)
    p = QtGui.QPainter(img)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, antialias)
    p.scale(size[0] / rect.width(), size[1] / rect.height())
    p.translate(-rect.left(), -rect.top())
    item.paint(p, QtWidgets.QStyleOptionGraphicsItem(), None)
    p.end()
    return pg.functions.ndarray_from_qimage(img).copy()


def bars(n: int, seed: int = 0) -> dict[str, np.ndarray]:
    """
    Random bars at integer x positions, with positive and negative heights.

    Parameters
    ----------
    n : int
        Number of bars.
    seed : int, default 0
        Random seed.

    Returns
    -------
    dict of numpy.ndarray
        ``x0``, ``width``, ``y0``, ``height`` and ``color`` (an index into COLORS).
    """
    rng = np.random.default_rng(seed)
    return {
        'x0': np.arange(n, dtype=float) - 0.4,
        'width': np.full(n, 0.8),
        'y0': rng.random(n) * 0.5 - 0.5,
        'height': rng.random(n) * 2 - 0.5,
        'color': rng.integers(0, 2, n),
    }


def assertSameState(item: pg.BarGraphItem, reference: pg.BarGraphItem) -> None:
    """
    Check that an item holds the same bars and derived state as a reference item.

    Parameters
    ----------
    item : BarGraphItem
        Item built with appendData.
    reference : BarGraphItem
        Item built with setOpts from all bars at once.
    """
    np.testing.assert_array_equal(item._rectarray.ndarray(), reference._rectarray.ndarray())
    np.testing.assert_array_equal(item._dataBounds, reference._dataBounds)
    assert item._xSorted == reference._xSorted
    assert item._x1Sorted == reference._x1Sorted
    if reference._xSorted:
        np.testing.assert_array_equal(item._x0, reference._x0)
        np.testing.assert_array_equal(item._x1, reference._x1)
    else:
        assert item._x0 is None and item._x1 is None
    np.testing.assert_allclose(item._meanWidth, reference._meanWidth, rtol=1e-12)
    for withPen in (True, False):
        ids, expected = item._styleIds(withPen), reference._styleIds(withPen)
        assert (ids is None) == (expected is None)
        if ids is not None:
            np.testing.assert_array_equal(ids, expected)
    for table, expected in ((item._pens, reference._pens),
                            (item._brushes, reference._brushes)):
        assert table.uniques == expected.uniques
    assert item._penWidth == reference._penWidth
    assert item._hasPen == reference._hasPen
    for a, b in zip(item.getOriginalDataset(), reference.getOriginalDataset()):
        np.testing.assert_array_equal(a, b)
    for ax in (0, 1):
        assert item.dataBounds(ax) == pytest.approx(reference.dataBounds(ax), nan_ok=True)
        assert item.dataBounds(ax, frac=0.8) == pytest.approx(
            reference.dataBounds(ax, frac=0.8), nan_ok=True)
        assert item.dataBounds(ax, orthoRange=(10.2, 20.7)) == pytest.approx(
            reference.dataBounds(ax, orthoRange=(10.2, 20.7)), nan_ok=True)


LAYOUTS = {
    # name: (options of all bars, keys given per bar)
    'x, width': lambda d: dict(x=d['x0'] + 0.4, width=0.8, height=d['height']),
    'x0, x1, y0, y1': lambda d: dict(x0=d['x0'], x1=d['x0'] + d['width'], y0=d['y0'],
                                     y1=d['y0'] + d['height']),
    'x1, width, y, height': lambda d: dict(x1=d['x0'] + d['width'], width=d['width'],
                                           y=d['y0'], height=d['height']),
    'unsorted': lambda d: dict(x=(d['x0'] * 7919) % len(d['x0']), width=0.8,
                               height=d['height']),
    'overlapping': lambda d: dict(x0=d['x0'], width=np.where(np.arange(len(d['x0'])) % 7,
                                                             0.8, 9.5),
                                  height=d['height']),
    'nan': lambda d: dict(x=d['x0'] + 0.4, width=0.8,
                          height=np.where(np.arange(len(d['x0'])) % 11, d['height'], np.nan)),
}


@pytest.mark.parametrize('layout', list(LAYOUTS))
@pytest.mark.parametrize('chunk', [1, 5])
def test_appendData_matches_setOpts(layout, chunk):
    n, first = 120, 50
    opts = LAYOUTS[layout](bars(n))
    perBar = [key for key, value in opts.items() if np.ndim(value)]

    def head(stop):
        return {key: (value[:stop] if key in perBar else value) for key, value in opts.items()}

    item = pg.BarGraphItem(pen=None, **head(first))
    for start in range(first, n, chunk):
        stop = min(start + chunk, n)
        item.appendData(**{key: opts[key][start:stop] for key in perBar})
        reference = pg.BarGraphItem(pen=None, **head(stop))
        assertSameState(item, reference)
        for key in opts:
            np.testing.assert_array_equal(item.opts[key], reference.opts[key])
    assert all(len(values) == n for values in item.getOriginalDataset())
    assert all(not item.opts[key].flags.writeable for key in perBar)


STYLES = {
    'mono pen': lambda c: dict(pen='w', brush='g'),
    'brushes no pen': lambda c: dict(pen=None, brushes=[COLORS[i] for i in c]),
    'brushes': lambda c: dict(pen='w', brushes=[COLORS[i] for i in c]),
    'pens and brushes': lambda c: dict(pens=[COLORS[i] for i in c],
                                       brushes=[COLORS[(i + 1) % 3] for i in c]),
}
# views of the last bars of 3000: in order, without outline (1.07 px per bar), and
# aggregated per pixel column (all bars)
VIEWS = {
    'zoomed': QtCore.QRectF(2959.5, -1.0, 40.0, 3.0),
    'thin': QtCore.QRectF(2699.5, -1.0, 300.0, 3.0),
    'aggregated': QtCore.QRectF(-1.0, -1.0, 3001.5, 3.0),
}


def styledItem(data: dict, style: str, stop: int, **extra) -> pg.BarGraphItem:
    """
    Item holding the first bars of ``data``, given all at once.

    Parameters
    ----------
    data : dict
        As returned by :func:`bars`.
    style : str
        Key of ``STYLES``.
    stop : int
        Number of bars.
    **extra
        More options.

    Returns
    -------
    BarGraphItem
        The item.
    """
    opts = STYLES[style](data['color'][:stop])
    return pg.BarGraphItem(x0=data['x0'][:stop], width=0.8, y0=data['y0'][:stop],
                           height=data['height'][:stop], **opts, **extra)


def appendArgs(data: dict, style: str, start: int, stop: int) -> dict:
    """
    Options appending bars ``start`` to ``stop - 1`` of ``data``.

    Parameters
    ----------
    data : dict
        As returned by :func:`bars`.
    style : str
        Key of ``STYLES``.
    start, stop : int
        Range of the new bars.

    Returns
    -------
    dict
        Keyword arguments of appendData.
    """
    opts = STYLES[style](data['color'][start:stop])
    args = {key: opts[key] for key in ('pens', 'brushes') if key in opts}
    return dict(x0=data['x0'][start:stop], y0=data['y0'][start:stop],
                height=data['height'][start:stop], **args)


@pytest.mark.parametrize('style', list(STYLES))
@pytest.mark.parametrize('view', list(VIEWS))
@pytest.mark.parametrize('antialias', [False, True], ids=['aliased', 'antialiased'])
def test_appendData_renders_like_setOpts(style, view, antialias):
    data = bars(3000, seed=4)
    rect = VIEWS[view]
    item = styledItem(data, style, 2900)
    render(item, rect, antialias=antialias)  # fills the level of detail caches
    start = 2900
    for count in (1, 1, 3, 1, 40, 1, 53):
        item.appendData(**appendArgs(data, style, start, start + count))
        start += count
        image = render(item, rect, antialias=antialias)
        expected = render(styledItem(data, style, start), rect, antialias=antialias)
        np.testing.assert_array_equal(image, expected)
    assert start == 3000
    assert image.any()


@pytest.mark.parametrize('style', ['brushes no pen', 'pens and brushes'])
def test_replaceLast_matches_setOpts(style):
    rng = np.random.default_rng(7)
    n = 2500
    data = bars(n, seed=1)
    item = styledItem(data, style, n)
    expected = {key: list(data[key]) for key in ('x0', 'y0', 'height', 'color')}
    views = [VIEWS['aggregated'], QtCore.QRectF(2480.5, -1.0, 100.0, 3.0)]
    for rect in views:
        render(item, rect)  # fills the level of detail caches
    for step in range(60):
        replace = bool(rng.random() < 0.6)
        count = int(rng.integers(1, 3))
        position = len(expected['x0']) - replace
        new = {
            'x0': position + np.arange(count) - 0.4 + rng.uniform(-0.1, 0.1, count),
            'y0': rng.random(count) * 0.5 - 0.5,
            'height': rng.random(count) * 2 - 0.5,
            # the third colour only appears in new bars, often replaced
            'color': rng.integers(0, 3, count),
        }
        styles = STYLES[style](new['color'])
        item.appendData(replaceLast=replace, x0=new['x0'], y0=new['y0'],
                        height=new['height'],
                        **{key: styles[key] for key in ('pens', 'brushes') if key in styles})
        for key, values in expected.items():
            if replace:
                values.pop()
            values.extend(new[key])
        arrays = {key: np.array(values) for key, values in expected.items()}
        reference = pg.BarGraphItem(
            x0=arrays['x0'], width=0.8, y0=arrays['y0'], height=arrays['height'],
            **STYLES[style](arrays['color']))
        assertSameState(item, reference)
        if step % 5 == 0:
            for rect in views:
                np.testing.assert_array_equal(render(item, rect), render(reference, rect))


@pytest.mark.parametrize('view', ['thin', 'aggregated'])
def test_replaceLast_redraws_cached_level_of_detail(view):
    data = bars(3000, seed=5)
    item = styledItem(data, 'brushes', 3000)
    rect = VIEWS[view]
    render(item, rect)
    # same bars visible, same view: only the replaced bar tells the cache is stale
    item.appendData(x0=data['x0'][-1:], y0=[-0.9], height=[2.8], brushes=[COLORS[1]],
                    replaceLast=True)
    data['y0'][-1], data['height'][-1], data['color'][-1] = -0.9, 2.8, 1
    np.testing.assert_array_equal(render(item, rect),
                                  render(styledItem(data, 'brushes', 3000), rect))


@pytest.mark.parametrize('inside', [False, True], ids=['outside view', 'inside view'])
def test_new_brush_redraws_combined_styles(inside):
    # with per-bar pens and brushes, a new brush changes the combined style indices
    data = bars(3000, seed=6)
    item = styledItem(data, 'pens and brushes', 3000, lodPen=False)
    rect = VIEWS['aggregated']
    render(item, rect)
    x0 = 2999.6 if inside else 3100.0
    item.appendData(x0=[x0], y0=[0.], height=[1.], pens=[COLORS[2]], brushes=[COLORS[0]])
    opts = STYLES['pens and brushes'](data['color'])
    reference = pg.BarGraphItem(
        x0=np.r_[data['x0'], x0], width=0.8, y0=np.r_[data['y0'], 0.],
        height=np.r_[data['height'], 1.], pens=opts['pens'] + [COLORS[2]],
        brushes=opts['brushes'] + [COLORS[0]], lodPen=False)
    assertSameState(item, reference)
    np.testing.assert_array_equal(render(item, rect), render(reference, rect))


def test_replaceLast_updates_bounds_and_styles():
    item = pg.BarGraphItem(x=[0., 1.], height=[1., 2.], width=0.8,
                           pens=['w', 'w'], brushes=['g', 'r'])
    item.appendData(x=[2.], height=[10.], pens=[pg.mkPen('w', width=4, cosmetic=False)],
                    brushes=['b'])
    assert item.dataBounds(1) == pytest.approx((-2, 12))
    assert len(item._pens.uniques) == 2 and len(item._brushes.uniques) == 3
    # the volume of the last bar decreases and it gets an existing style back
    item.appendData(x=[2.], height=[1.5], pens=['w'], brushes=['g'], replaceLast=True)
    assert item.dataBounds(1) == pytest.approx((0, 2))
    assert len(item._pens.uniques) == 1 and len(item._brushes.uniques) == 2
    assert item._penWidth == [0, 1]
    np.testing.assert_array_equal(item.getData()[1], [1., 2., 1.5])
    assert len(item.opts['brushes']) == 3
    assert item.opts['brushes'][-1] == 'g'
    # in place: the last bar moves too
    item.appendData(x=[2.5], height=[3.], pens=['w'], brushes=['r'], replaceLast=True)
    assert item.dataBounds(0) == pytest.approx((-0.4, 2.9))
    np.testing.assert_array_equal(item.getData()[0], [0., 1., 2.5])


def test_replaceLast_restores_sorted_index():
    item = pg.BarGraphItem(x=np.arange(10.), height=np.ones(10), width=0.8)
    item.appendData(x=[3.], height=[1.])  # out of order
    assert not item._xSorted and item._x0 is None
    item.appendData(x=[10.], height=[2.], replaceLast=True)
    reference = pg.BarGraphItem(x=np.arange(11.), height=np.r_[np.ones(10), 2.], width=0.8)
    assertSameState(item, reference)


def test_lod_recomputes_only_the_last_column(monkeypatch):
    n = 30_000
    data = bars(n + 50, seed=2)
    item = styledItem(data, 'brushes', n)
    rect = QtCore.QRectF(-1.0, -1.0, n + 100.0, 3.0)  # about 75 bars per pixel column
    calls = []
    original = pg.BarGraphItem._columnBlock

    def columnBlock(self, begin, stop, *args):
        calls.append((begin, stop))
        return original(self, begin, stop, *args)

    monkeypatch.setattr(pg.BarGraphItem, '_columnBlock', columnBlock)
    render(item, rect)
    assert calls == [(0, n)]
    lastColumn = int(item._columns.first[-1])
    assert lastColumn > n - 200
    for k in range(n, n + 50):
        calls.clear()
        item.appendData(**appendArgs(data, 'brushes', k, k + 1))
        image = render(item, rect)
        # the columns before the one of the previous last bar are kept
        assert len(calls) == 1
        begin, stop = calls[0]
        assert stop == k + 1 and begin > k - 200
    np.testing.assert_array_equal(image, render(styledItem(data, 'brushes', n + 50), rect))

    # bars appended outside of the view leave the cached rectangles valid
    calls.clear()
    item.appendData(x0=[n + 500.], y0=[0.], height=[1.], brushes=['r'])
    render(item, rect)
    assert calls == []
    # a replaced bar moving into the view: the column of the last visible bar again
    item.appendData(x0=[n + 49 - 0.4], y0=[0.], height=[1.], brushes=['r'], replaceLast=True)
    render(item, rect)
    assert len(calls) == 1 and calls[0][0] > n - 200


def test_appendData_is_incremental(monkeypatch):
    n = 20_000
    rng = np.random.default_rng(3)
    palette = [pg.mkBrush(c) for c in COLORS[:2]]
    brushes = [palette[i] for i in rng.integers(0, 2, n + 2000)]
    item = pg.BarGraphItem(x=np.arange(n, dtype=float), height=rng.random(n), width=0.8,
                           brushes=brushes[:n], pen=None)
    sizes = []
    original = barModule._uniqueStyles

    def uniqueStyles(specs, *args):
        specs = list(specs)
        sizes.append(len(specs))
        return original(specs, *args)

    monkeypatch.setattr(barModule, '_uniqueStyles', uniqueStyles)
    monkeypatch.setattr(pg.BarGraphItem, '_prepareData',
                        lambda self: pytest.fail('all bars processed again'))
    capacities = set()
    for k in range(n, n + 2000):
        item.appendData(x=[float(k)], height=[rng.random()], brushes=brushes[k:k + 1])
        capacities.add(len(item._rectarray._array))
    assert len(item._rectarray) == n + 2000
    assert set(sizes) == {1}
    # geometric growth: few reallocations, little unused capacity
    assert len(capacities) <= 3
    assert max(capacities) < 1.5 * (n + 2000) + 1
    assert len(item.opts['x']) == n + 2000 and len(item.opts['brushes']) == n + 2000


def test_append_to_empty_item():
    item = pg.BarGraphItem(x=[], height=[], width=0.6, brushes=[], pen=None)
    assert item.dataBounds(0) == (None, None)
    item.appendData(x=[0.], height=[2.], brushes=['g'])
    item.appendData(x=[1., 2.], height=[1., 3.], brushes=['r', 'g'])
    reference = pg.BarGraphItem(x=[0., 1., 2.], height=[2., 1., 3.], width=0.6,
                                brushes=['g', 'r', 'g'], pen=None)
    assertSameState(item, reference)
    rect = QtCore.QRectF(-1, -0.5, 4, 4)
    np.testing.assert_array_equal(render(item, rect), render(reference, rect))


def test_more_styles_than_bars():
    # the extra brush is not used by the next bar, which gets its own brush
    item = pg.BarGraphItem(x=[0., 1.], height=[1., 1.], width=0.8, pen=None,
                           brushes=['r', 'g', 'b'])
    item.appendData(x=[2.], height=[1.], brushes=['y'])
    reference = pg.BarGraphItem(x=[0., 1., 2.], height=[1., 1., 1.], width=0.8, pen=None,
                                brushes=['r', 'g', 'y'])
    assertSameState(item, reference)
    assert item.opts['brushes'] == ['r', 'g', 'y']
    item = pg.BarGraphItem(x=[0., 1.], height=[1., 1.], width=0.8, brushes=['r'])
    with pytest.raises(IndexError):
        item.appendData(x=[2.], height=[1.], brushes=['y'])


def test_scalar_options():
    item = pg.BarGraphItem(x=[0., 1.], height=[1., 2.], width=0.5, y0=0.0)
    item.appendData(x=[2.], height=[3.])
    assert item.opts['width'] == 0.5 and item.opts['y0'] == 0.0
    item.appendData(x=[3.], height=[4.], width=0.5)  # same value: still one value
    assert item.opts['width'] == 0.5
    item.appendData(x=[4.], height=[5.], width=[0.9])
    np.testing.assert_array_equal(item.opts['width'], [0.5] * 4 + [0.9])
    with pytest.raises(ValueError, match="'width' holds one value per bar"):
        item.appendData(x=[5.], height=[6.])
    # a float applies to all new bars
    item.appendData(x=[5., 6.], height=2.0, width=0.9)
    np.testing.assert_array_equal(item.opts['height'], [1., 2., 3., 4., 5., 2., 2.])
    reference = pg.BarGraphItem(x=np.arange(7.), height=[1., 2., 3., 4., 5., 2., 2.],
                                width=[0.5] * 4 + [0.9] * 3, y0=0.0)
    assertSameState(item, reference)


def test_invalid_options_leave_item_unchanged():
    item = pg.BarGraphItem(x=[0., 1.], height=[1., 2.], width=0.5, brushes=['r', 'g'])
    memory = item._rectarray.ndarray().copy()
    opts = {key: (value.copy() if isinstance(value, np.ndarray) else value)
            for key, value in item.opts.items()}
    cases = [
        (TypeError, dict(x=[2.], height=[1.], brushes=['r'], pen='w')),
        (TypeError, dict(x=[2.], height=[1.], brushes=['r'], name='volume')),
        (ValueError, dict(x0=[2.], height=[1.], brushes=['r'])),
        (ValueError, dict(height=[1.], brushes=['r'])),
        (ValueError, dict(x=[2.], brushes=['r'])),
        (ValueError, dict(x=[2.], height=[1.])),
        (ValueError, dict(x=[2.], height=[1.], brushes=['r', 'g'])),
        (ValueError, dict(x=[2., 3.], height=[1., 2., 3.], brushes=['r', 'g'])),
        (ValueError, dict(x=[[2.]], height=[1.], brushes=['r'])),
        (ValueError, dict(brushes=['r'])),
        (ValueError, dict(x=[], height=[], brushes=[], replaceLast=True)),
    ]
    for error, kwargs in cases:
        with pytest.raises(error):
            item.appendData(**kwargs)
    np.testing.assert_array_equal(item._rectarray.ndarray(), memory)
    assert item.opts.keys() == opts.keys()
    for key, value in opts.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(item.opts[key], value)
        else:
            assert item.opts[key] == value
    assert len(item._brushes.ids) == 2
    # shared brush: no per-bar brushes
    shared = pg.BarGraphItem(x=[0.], height=[1.], width=0.5, brush='r')
    with pytest.raises(ValueError, match='share one brush'):
        shared.appendData(x=[1.], height=[1.], brushes=['g'])
    empty = pg.BarGraphItem(x=[], height=[], width=0.5)
    with pytest.raises(ValueError, match='replaceLast'):
        empty.appendData(x=[1.], height=[1.], replaceLast=True)
    # no new bar: nothing happens
    item.appendData(x=[], height=[], brushes=[])
    assert len(item._rectarray) == 2


def test_setOpts_after_appendData():
    item = pg.BarGraphItem(x=[0., 1.], height=[1., 2.], width=0.5, brushes=['r', 'g'])
    item.appendData(x=[2.], height=[3.], brushes=['b'])
    x = item.opts['x']
    item.setOpts(pen=None)  # data options kept
    item.appendData(x=[3.], height=[4.], brushes=['r'])
    np.testing.assert_array_equal(item.getData()[0], [0., 1., 2., 3.])
    np.testing.assert_array_equal(x, [0., 1., 2.])  # an earlier view is not extended
    item.setOpts(x=item.opts['x'] + 10, brush='y', brushes=None)
    item.appendData(x=[14.], height=[5.])
    reference = pg.BarGraphItem(x=np.arange(10., 15.), height=[1., 2., 3., 4., 5.],
                                width=0.5, pen=None, brush='y')
    assertSameState(item, reference)


def test_shape_includes_appended_bars():
    item = pg.BarGraphItem(x=[0., 1.], height=[1., 2.], width=0.5)
    path = item.shape()
    item.appendData(x=[2., 3.], height=[3., 1.])
    reference = pg.BarGraphItem(x=[0., 1., 2., 3.], height=[1., 2., 3., 1.], width=0.5)
    assert item.shape() == reference.shape()
    assert path != item.shape()  # the returned path is not extended
    item.appendData(x=[3.], height=[5.], replaceLast=True)
    assert item.shape().boundingRect() == QtCore.QRectF(-0.25, 0, 3.5, 5)


def test_csv_export_includes_appended_bars(tmp_path):
    plot = pg.PlotItem()
    item = pg.BarGraphItem(x=[0., 1.], height=[10., 30.], width=0.6, name='volume')
    plot.addItem(item)
    item.appendData(x=[2., 3.], height=[20., 50.])
    item.appendData(x=[3.], height=[55.], replaceLast=True)
    fileName = tmp_path / 'bars.csv'
    CSVExporter(plot).export(fileName=str(fileName))
    with open(fileName, newline='') as file:
        rows = list(csv.reader(file))
    assert rows[0] == ['volume_x', 'volume_y']
    assert [[float(v) for v in row] for row in rows[1:]] == [
        [0, 10], [1, 30], [2, 20], [3, 55]]


def test_autorange_follows_appended_bars():
    pw = pg.PlotWidget()
    pw.resize(400, 300)
    item = pg.BarGraphItem(x=np.arange(100.), height=np.ones(100), width=0.8)
    pw.addItem(item)
    show_and_wait(pw)
    (_, xmax), (_, ymax) = pw.viewRange()
    assert xmax < 110 and ymax < 1.5
    item.appendData(x=[150.], height=[8.])
    for _ in range(3):
        app.processEvents()
    (_, xmax), (_, ymax) = pw.viewRange()
    assert xmax >= 150.4 and ymax >= 8
    item.appendData(x=[150.], height=[3.], replaceLast=True)
    for _ in range(3):
        app.processEvents()
    _, (_, ymax) = pw.viewRange()
    assert 3 <= ymax < 4
    pw.close()


# bars with non-finite coordinates are not drawn and do not count in the bounds

HEIGHTS = np.array([5., 1, 7, 3, 9, 2, 8, 4, 6, 10])
NON_FINITE = {
    'nan height': ('height', np.nan),
    'nan x': ('x', np.nan),
    'inf height': ('height', np.inf),
    '-inf height': ('height', -np.inf),
    'nan width': ('width', np.nan),
    '-inf y0': ('y0', -np.inf),
}


def finiteBounds(item: pg.BarGraphItem, ax: int) -> tuple[float | None, float | None]:
    """
    Range of the bars with finite coordinates along an axis, bar by bar, without pen.

    Parameters
    ----------
    item : BarGraphItem
        Item whose bars are measured.
    ax : int
        0 for x, 1 for y.

    Returns
    -------
    tuple of float or None
        Lowest lower edge and highest upper edge of the bars whose rectangle is
        finite, or ``(None, None)`` when there is none.
    """
    lows, highs = [], []
    for row in item._rectarray.ndarray().tolist():
        if all(np.isfinite(row)):
            lows.append(row[ax])
            highs.append(row[ax] + row[ax + 2])
    if not lows:
        return None, None
    return min(lows), max(highs)


@pytest.mark.parametrize('case', list(NON_FINITE))
@pytest.mark.parametrize('position', [0, 4, 9], ids=['first', 'middle', 'last'])
def test_bounds_ignore_non_finite_bars(case, position):
    opts = {'x': np.arange(10.), 'height': HEIGHTS.copy(), 'width': np.full(10, 0.6),
            'y0': np.full(10, -1.0)}
    key, value = NON_FINITE[case]
    opts[key][position] = value
    with np.errstate(invalid='ignore'):  # inf - inf in the rectangles
        item = pg.BarGraphItem(pen=pg.mkPen('w', width=0.2, cosmetic=False), **opts)
    for ax in (0, 1):
        low, high = finiteBounds(item, ax)
        assert item.dataBounds(ax) == pytest.approx((low - 0.1, high + 0.1))
        # as the restricted and percentile ranges
        assert item.dataBounds(ax, orthoRange=(-100, 100)) == pytest.approx(
            item.dataBounds(ax))
        assert item.dataBounds(ax, frac=0.99)[1] <= item.dataBounds(ax)[1]
    (xmin, xmax), (ymin, ymax) = item.dataBounds(0), item.dataBounds(1)
    rect = item.boundingRect()
    assert rect.isValid()
    assert (rect.left(), rect.right()) == pytest.approx((xmin, xmax))
    assert (rect.top(), rect.bottom()) == pytest.approx((ymin, ymax))


def test_bounds_of_sorted_bars_match_orthoRange():
    # with sorted bars the range of all bars within an x range reuses the full range
    heights = HEIGHTS.copy()
    heights[[2, 9]] = np.nan
    item = pg.BarGraphItem(x=np.arange(10.), height=heights, width=0.6)
    assert item._x1Sorted
    assert item.dataBounds(1) == pytest.approx((0, 9))
    assert item.dataBounds(1, orthoRange=(-100, 100)) == pytest.approx((0, 9))
    assert item.dataBounds(1, orthoRange=(1.5, 8.5)) == pytest.approx((0, 9))
    # along both axes
    assert item.dataBounds(0) == pytest.approx((-0.3, 8.3))
    assert item.dataBounds(0, orthoRange=(-100, 100)) == pytest.approx((-0.3, 8.3))
    assert item.dataBounds(0, frac=0.999) == pytest.approx(
        (np.percentile(np.r_[0:2, 3:9] - 0.3, 0.05), np.percentile(np.r_[0:2, 3:9] + 0.3,
                                                                 99.95)))
    infinite = pg.BarGraphItem(x=np.arange(10.), height=np.r_[HEIGHTS[:9], np.inf],
                               width=0.6)
    assert infinite.dataBounds(1) == pytest.approx((0, 9))
    assert infinite.dataBounds(1, orthoRange=(-100, 100)) == pytest.approx((0, 9))
    assert infinite.dataBounds(1, orthoRange=(7.5, 9.5)) == pytest.approx((0, 6))


def test_bounds_without_finite_bars():
    item = pg.BarGraphItem(x=np.arange(3.), height=np.full(3, np.nan), width=0.5)
    for ax in (0, 1):
        assert item.dataBounds(ax) == (None, None)
        assert item.dataBounds(ax, orthoRange=(-10, 10)) == (None, None)
        assert item.dataBounds(ax, frac=0.5) == (None, None)
    assert item.boundingRect() == QtCore.QRectF()
    # appended bars bring finite bounds, replaced ones take them away again
    item.appendData(x=[3.], height=[2.])
    assert item.dataBounds(0) == pytest.approx((2.75, 3.25))
    assert item.dataBounds(1) == pytest.approx((0, 2))
    assert item.boundingRect() == QtCore.QRectF(2.75, 0, 0.5, 2)
    item.appendData(x=[3.], height=[np.nan], replaceLast=True)
    assert item.dataBounds(0) == item.dataBounds(1) == (None, None)
    item.appendData(x=[np.nan, 4.], height=[5., 4.])
    assert item.dataBounds(0) == pytest.approx((3.75, 4.25))
    assert item.dataBounds(1) == pytest.approx((0, 4))


def test_autorange_includes_bars_with_nan():
    pw = pg.PlotWidget()
    pw.resize(400, 300)
    heights = 1.0 + np.arange(100.)
    heights[50] = np.nan
    item = pg.BarGraphItem(x=np.arange(100.) + 1000, height=heights, width=0.8)
    pw.addItem(item)
    show_and_wait(pw)
    (xmin, xmax), (ymin, ymax) = pw.viewRange()
    assert 990 < xmin <= 999.6 and 1099.4 <= xmax < 1110
    assert -10 < ymin <= 0 and 100 <= ymax < 110
    with np.errstate(invalid='ignore'):
        item.appendData(x=[1100., 1101.], height=[np.inf, 150.])
    for _ in range(3):
        app.processEvents()
    (_, xmax), (_, ymax) = pw.viewRange()
    assert 1101.4 <= xmax < 1115 and 150 <= ymax < 165
    pw.close()
