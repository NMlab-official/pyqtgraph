"""
Performance regression tests for :meth:`~pyqtgraph.AxisItem.generateDrawSpecs` (T2.10).

Tick strings (default ``AxisItem.tickStrings`` and ``DateAxisItem.tickStrings``) and the
size of each tick label (per font) are cached across calls. These tests check the cache
states with call counts, that the caches are invalidated or bypassed when needed, and
that the generated specifications are those computed without caches.
"""
from __future__ import annotations

import locale
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import pytest

import pyqtgraph as pg
from pyqtgraph.graphicsItems.AxisItem import _TICK_TEXT_FLAGS, AxisItem, _LRUCache
from pyqtgraph.graphicsItems.DateAxisItem import HOUR_SPACING
from pyqtgraph.Point import Point
from pyqtgraph.Qt import QtCore, QtGui
from tests.perf_helpers import count_calls, process_events

app = pg.mkQApp()

T0 = 1.7e9  # an epoch timestamp, so that DateAxisItem shows realistic dates

NUMERIC_RANGES = [(0, 1), (0.25, 1.25), (-3, 3), (1e-6, 3e-6), (100, 1100), (100.5, 1100.5),
                  (-1e9, 2e9), (1e5, 1e5 + 0.01), (0, 1)]
LOG_RANGES = [(-2, 3), (-1.5, 3.5), (0, 0.5), (-8, 12), (-2, 3)]
DATE_RANGES = [(T0 + k * 0.37 * span, T0 + k * 0.37 * span + span)
               for span in (30, 3600, 86400, 30 * 86400, 3 * 365 * 86400) for k in range(3)]


def _serialize(specs: tuple | None) -> tuple | None:
    """
    Convert draw specifications to plain comparable values.

    Parameters
    ----------
    specs : tuple or None
        The value returned by ``generateDrawSpecs``.

    Returns
    -------
    tuple or None
        Nested tuples of numbers and strings.
    """
    if specs is None:
        return None
    axisSpec, tickSpecs, textSpecs = specs

    def pen(p: QtGui.QPen) -> tuple:
        return p.color().getRgb(), p.widthF(), p.isCosmetic()

    def point(pt: QtCore.QPointF) -> tuple:
        assert type(pt) is Point
        return pt.x(), pt.y()

    return (
        (pen(axisSpec[0]), point(axisSpec[1]), point(axisSpec[2])),
        tuple((pen(p), point(p1), point(p2)) for p, p1, p2 in tickSpecs),
        tuple((r.getRect(), int(flags), text) for r, flags, text in textSpecs),
    )


@contextmanager
def _painter() -> Iterator[QtGui.QPainter]:
    """
    Painter on a QPicture, as used by ``AxisItem.paint``.

    Yields
    ------
    QtGui.QPainter
        The active painter.
    """
    picture = QtGui.QPicture()
    painter = QtGui.QPainter(picture)
    try:
        yield painter
    finally:
        painter.end()


def _make_plot(kind: str) -> pg.PlotWidget:
    """
    Plot widget whose bottom and left axes are of the given kind.

    Parameters
    ----------
    kind : {'numeric', 'log', 'date'}
        Kind of axes.

    Returns
    -------
    PlotWidget
        The widget, shown.
    """
    if kind == 'date':
        widget = pg.PlotWidget(axisItems={'bottom': pg.DateAxisItem(),
                                          'left': pg.DateAxisItem('left')})
    else:
        widget = pg.PlotWidget()
    widget.resize(700, 400)
    if kind == 'log':
        widget.plotItem.setLogMode(True, True)
    widget.show()
    app.processEvents()
    return widget


def _generate(axis: AxisItem) -> tuple | None:
    """
    Call ``generateDrawSpecs`` the way ``AxisItem.paint`` does.

    Parameters
    ----------
    axis : AxisItem
        The axis.

    Returns
    -------
    tuple or None
        The serialized specifications.
    """
    with _painter() as painter:
        if axis.style['tickFont'] is not None:
            painter.setFont(axis.style['tickFont'])
        return _serialize(axis.generateDrawSpecs(painter))


def _disable_caches(axis: AxisItem) -> None:
    """
    Make every cache lookup of ``axis`` miss, to compute reference specifications.

    Parameters
    ----------
    axis : AxisItem
        The axis.
    """
    axis._tickStringsCache = _LRUCache(0)
    axis._tickTextSizeCache = _LRUCache(0)  # shadows the shared cache


@pytest.mark.parametrize('kind,ranges', [('numeric', NUMERIC_RANGES), ('log', LOG_RANGES),
                                         ('date', DATE_RANGES)])
@pytest.mark.parametrize('orientation', ['bottom', 'left'])
def test_specs_identical_with_and_without_caches(kind: str, ranges: list, orientation: str):
    cached_widget = _make_plot(kind)
    reference_widget = _make_plot(kind)
    cached = cached_widget.getAxis(orientation)
    reference = reference_widget.getAxis(orientation)
    _disable_caches(reference)
    for axis in (cached, reference):
        axis.unlinkFromView()
    process_events(5)
    # Both widgets go through the same steps, so that their geometry stays equal.
    for low, high in ranges:
        for axis in (cached, reference):
            axis.setRange(low, high)
        process_events()
        # twice: the second call of the cached axis is served from the caches
        for _ in range(2):
            specs = _generate(cached)
            assert specs == _generate(reference)
            assert specs is not None and specs[1]
    cached_widget.close()
    reference_widget.close()


def test_text_flags_are_unchanged():
    flags = QtCore.Qt.AlignmentFlag
    dont_clip = QtCore.Qt.TextFlag.TextDontClip
    assert _TICK_TEXT_FLAGS == {
        'left': flags.AlignRight | flags.AlignVCenter | dont_clip,
        'right': flags.AlignLeft | flags.AlignVCenter | dont_clip,
        'top': flags.AlignHCenter | flags.AlignBottom | dont_clip,
        'bottom': flags.AlignHCenter | flags.AlignTop | dont_clip,
    }


def test_tick_strings_are_cached():
    widget = _make_plot('numeric')
    axis = widget.getAxis('bottom')
    axis.unlinkFromView()
    axis.setRange(0, 10)
    with count_calls(AxisItem, 'tickStrings') as calls:
        first = _generate(axis)
        assert calls.count > 0
        calls.reset()
        assert _generate(axis) == first
        assert calls.count == 0
        axis.setRange(-20, 30)  # new tick values
        _generate(axis)
        assert calls.count > 0
    widget.close()


def test_tick_text_measured_once_per_string_and_font():
    widget = _make_plot('numeric')
    axis = widget.getAxis('bottom')
    axis.unlinkFromView()
    axis.setRange(0, 10)
    AxisItem._tickTextSizeCache.clear()
    with _painter() as painter, count_calls(painter, 'boundingRect') as measures:
        axis.generateDrawSpecs(painter)
        first = measures.count
        assert first > 0
        axis.generateDrawSpecs(painter)
        assert measures.count == first
    # another axis showing the same labels with the same font shares the measurements
    other = pg.AxisItem('top')
    other.setParentItem(widget.plotItem)
    other.setGeometry(axis.geometry())
    other.setRange(0, 10)
    with _painter() as painter, count_calls(painter, 'boundingRect') as measures:
        other.generateDrawSpecs(painter)
        assert measures.count == 0
    # a new font is measured again
    font = QtGui.QFont()
    font.setPointSize(23)
    axis.setTickFont(font)
    with _painter() as painter, count_calls(painter, 'boundingRect') as measures:
        painter.setFont(font)
        axis.generateDrawSpecs(painter)
        assert measures.count == first
    widget.close()


def test_text_size_cache_is_bounded_lru():
    cache = _LRUCache(3)
    for key in 'abc':
        cache.put(key, key.upper())
    assert cache.get('a') == 'A'  # 'a' becomes the most recently used entry
    cache.put('d', 'D')
    assert len(cache) == 3
    assert 'b' not in cache
    assert [key in cache for key in 'acd'] == [True, True, True]
    assert cache.get('b') is None


@pytest.mark.parametrize('change', [
    lambda axis: axis.setStyle(tickLength=7),
    lambda axis: axis.setTickFont(QtGui.QFont()),
    lambda axis: axis.setLabel('Voltage', units='V'),
    lambda axis: axis.setLogMode(True),
], ids=['style', 'font', 'label', 'logMode'])
def test_tick_strings_cache_invalidated(change: Callable[[AxisItem], object]):
    widget = _make_plot('numeric')
    axis = widget.getAxis('bottom')
    axis.unlinkFromView()
    axis.setRange(1, 3)
    _generate(axis)
    assert len(axis._tickStringsCache) > 0
    change(axis)
    assert len(axis._tickStringsCache) == 0
    widget.close()


def test_log_mode_strings_follow_log_mode():
    widget = _make_plot('numeric')
    axis = widget.getAxis('bottom')
    axis.unlinkFromView()
    axis.setRange(1, 3)
    linear = [text for _, _, text in _generate(axis)[2]]
    axis.logMode = True  # set directly: the log mode is also part of the cache key
    log = [text for _, _, text in _generate(axis)[2]]
    assert linear != log
    assert '10²' in log
    widget.close()


def test_overridden_tick_strings_are_not_cached():
    class PrefixAxis(pg.AxisItem):
        prefix = 'a'

        def tickStrings(self, values, scale, spacing):
            return [self.prefix + s for s in super().tickStrings(values, scale, spacing)]

    axis = PrefixAxis('bottom')
    widget = pg.PlotWidget(axisItems={'bottom': axis})
    widget.resize(700, 400)
    widget.show()
    app.processEvents()
    axis.unlinkFromView()
    axis.setRange(0, 10)
    assert all(text.startswith('a') for _, _, text in _generate(axis)[2])
    PrefixAxis.prefix = 'b'
    assert all(text.startswith('b') for _, _, text in _generate(axis)[2])
    widget.close()


def _date_axis() -> tuple[pg.PlotWidget, pg.DateAxisItem]:
    """
    Plot widget with an unlinked bottom DateAxisItem showing one day.

    Returns
    -------
    widget : PlotWidget
        The widget, shown.
    axis : DateAxisItem
        Its bottom axis.
    """
    axis = pg.DateAxisItem()
    widget = pg.PlotWidget(axisItems={'bottom': axis})
    widget.resize(700, 400)
    widget.show()
    app.processEvents()
    axis.unlinkFromView()
    axis.setRange(T0, T0 + 86400)
    return widget, axis


def _texts(axis: AxisItem) -> list[str]:
    """
    Tick label strings drawn by ``axis``.

    Parameters
    ----------
    axis : AxisItem
        The axis.

    Returns
    -------
    list of str
        The label strings.
    """
    return [text for _, _, text in _generate(axis)[2]]


def test_date_axis_tick_strings_are_cached():
    widget, axis = _date_axis()
    with count_calls(pg.DateAxisItem, 'tickStrings') as calls:
        first = _texts(axis)
        assert calls.count > 0
        calls.reset()
        assert _texts(axis) == first
        assert calls.count == 0
    # the UTC offset is part of the cache key
    values = [T0 + 3600 * k for k in range(4)]
    hours = axis._tickStringsCached(values, 1.0, HOUR_SPACING)
    assert axis._tickStringsCached(values, 1.0, HOUR_SPACING) == hours
    axis.utcOffset = 3 * 3600
    assert axis._tickStringsCached(values, 1.0, HOUR_SPACING) != hours
    widget.close()


def test_date_axis_tick_strings_follow_locale(monkeypatch: pytest.MonkeyPatch):
    widget, axis = _date_axis()
    _generate(axis)
    with count_calls(pg.DateAxisItem, 'tickStrings') as calls:
        _generate(axis)
        assert calls.count == 0
        current = locale.setlocale(locale.LC_TIME)
        # strftime output depends on LC_TIME: another locale must not reuse the strings
        monkeypatch.setattr(locale, 'setlocale',
                            lambda category, value=None: current + '-other')
        _generate(axis)
        assert calls.count > 0
    widget.close()


@pytest.mark.skipif(not hasattr(time, 'tzset'), reason='time.tzset is not available')
def test_date_axis_tick_strings_follow_local_time_zone(monkeypatch: pytest.MonkeyPatch):
    widget, axis = _date_axis()
    values = [T0 + 3600 * k for k in range(4)]
    try:
        monkeypatch.setenv('TZ', 'UTC')
        time.tzset()
        _generate(axis)  # selects the zoom level
        utc = axis._tickStringsCached(values, 1.0, HOUR_SPACING)
        monkeypatch.setenv('TZ', 'Asia/Tokyo')
        time.tzset()
        if QtCore.QTimeZone.systemTimeZoneId().data() != b'Asia/Tokyo':
            pytest.skip('Qt does not follow the TZ environment variable here')
        tokyo = axis._tickStringsCached(values, 1.0, HOUR_SPACING)
        assert tokyo == axis.tickStrings(values, 1.0, HOUR_SPACING)
        assert tokyo != utc
    finally:
        monkeypatch.undo()
        time.tzset()
        widget.close()


def test_overridden_date_tick_strings_are_not_cached():
    class UpperDateAxis(pg.DateAxisItem):
        suffix = '!'

        def tickStrings(self, values, scale, spacing):
            return [s + self.suffix for s in super().tickStrings(values, scale, spacing)]

    axis = UpperDateAxis()
    widget = pg.PlotWidget(axisItems={'bottom': axis})
    widget.resize(700, 400)
    widget.show()
    app.processEvents()
    axis.unlinkFromView()
    axis.setRange(T0, T0 + 86400)
    assert all(text.endswith('!') for text in _texts(axis))
    UpperDateAxis.suffix = '?'
    assert all(text.endswith('?') for text in _texts(axis))
    widget.close()
