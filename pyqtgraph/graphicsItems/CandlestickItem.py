"""
Vectorized OHLC candlesticks with view culling and level of detail.
"""
import math

import numpy as np

from .. import functions as fn
from ..Qt import QtCore, QtGui
from .BarGraphItem import _GrowableArray, _PrimitiveBuffer, _visibleRect
from .GraphicsObject import GraphicsObject

__all__ = ['CandlestickItem']

# Below this candle width (in device pixels) candles are aggregated, when ``lod`` is on.
_LOD_MIN_WIDTH_PX = 3.0
# Default candle width, relative to the median spacing of x.
_DEFAULT_WIDTH_RATIO = 0.8
# Levels aggregating at least this many candles per block are computed for all candles
# and cached; smaller blocks are computed for the visible candles only, which are few.
_CACHED_MIN_K = 16


def _union(a: tuple[float | None, float | None],
           b: tuple[float | None, float | None]) -> tuple[float | None, float | None]:
    """
    Smallest range containing two ranges.

    Parameters
    ----------
    a, b : tuple of float or None
        ``(low, high)`` ranges; ``(None, None)`` is empty.

    Returns
    -------
    tuple of float or None
        The union, ``(None, None)`` when both are empty.
    """
    if a[0] is None:
        return b
    if b[0] is None:
        return a
    return min(a[0], b[0]), max(a[1], b[1])


def _geometry(x: np.ndarray, halfWidth: np.ndarray, open: np.ndarray, high: np.ndarray,
              low: np.ndarray, close: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Body rectangles and wick lines of candles.

    Parameters
    ----------
    x, halfWidth : numpy.ndarray
        Center and half width of the bodies.
    open, high, low, close : numpy.ndarray
        Prices, all finite.

    Returns
    -------
    bodies : numpy.ndarray
        ``(x, y, width, height)`` of each body, from open to close.
    wicks : numpy.ndarray
        ``(x1, y1, x2, y2)`` of each wick, from low to high.
    """
    top = np.maximum(open, close)
    bottom = np.minimum(open, close)
    bodies = np.empty((len(x), 4))
    bodies[:, 0] = x - halfWidth
    bodies[:, 1] = bottom
    bodies[:, 2] = 2.0 * halfWidth
    bodies[:, 3] = top - bottom
    wicks = np.empty((len(x), 4))
    wicks[:, 0] = x
    wicks[:, 1] = np.minimum(low, bottom)
    wicks[:, 2] = x
    wicks[:, 3] = np.maximum(high, top)
    return bodies, wicks


class _Side:
    """Candles of one direction (rising or falling): centers, bodies and wicks."""

    def __init__(self) -> None:
        self.x = _GrowableArray()
        self.bodies = _PrimitiveBuffer(QtCore.QRectF)
        self.wicks = _PrimitiveBuffer(QtCore.QLineF)

    def __len__(self) -> int:
        return len(self.x)

    def clear(self) -> None:
        """Remove all candles."""
        self.x.clear()
        self.bodies.clear()
        self.wicks.clear()

    def pop(self) -> None:
        """Remove the last candle."""
        self.x.pop()
        self.bodies.pop()
        self.wicks.pop()

    def extend(self, x: np.ndarray, bodies: np.ndarray, wicks: np.ndarray,
               rows: np.ndarray | None = None) -> None:
        """
        Append candles, sorted by x and after the existing ones.

        Parameters
        ----------
        x : numpy.ndarray
            Centers of the candles.
        bodies, wicks : numpy.ndarray
            Body rectangles and wick lines, shape ``(len(x), 4)``.
        rows : numpy.ndarray or None, default None
            Indices of the candles to append; all when ``None``.
        """
        if rows is None:
            self.x.extend(x)
            self.bodies.reserve(len(x))[:] = bodies
            self.wicks.reserve(len(x))[:] = wicks
        elif len(rows):
            self.x.extend(x[rows])
            np.take(bodies, rows, axis=0, out=self.bodies.reserve(len(rows)))
            np.take(wicks, rows, axis=0, out=self.wicks.reserve(len(rows)))

    def visible(self, xmin: float, xmax: float) -> tuple[int, int]:
        """
        Range of the candles whose center lies within ``[xmin, xmax]``.

        Parameters
        ----------
        xmin, xmax : float
            Range of x.

        Returns
        -------
        tuple of int
            ``start, stop`` indices.
        """
        centers = self.x.view
        start = int(np.searchsorted(centers, xmin, side='left'))
        stop = int(np.searchsorted(centers, xmax, side='right'))
        return start, stop


class _Level:
    """
    Candles aggregated by blocks of ``k`` consecutive candles.

    Block ``j`` holds candles ``j*k`` to ``(j+1)*k - 1``: blocks are aligned to
    absolute multiples of ``k``, so that appending candles, or replacing the last
    one, only changes the last block. Level 1 holds the candles themselves.

    Parameters
    ----------
    k : int
        Number of candles per block.
    """

    def __init__(self, k: int) -> None:
        self.k = k
        self.up = _Side()
        self.down = _Side()
        # index of the last block stored in a side, and that side: it is the only
        # block computed again when candles are appended or the last one replaced
        self._lastBlock = -1
        self._lastSide = None

    def update(self, x: np.ndarray, open: np.ndarray, high: np.ndarray, low: np.ndarray,
               close: np.ndarray, width: float, fromCandle: int) -> None:
        """
        Compute the blocks holding candles ``fromCandle`` and later.

        Parameters
        ----------
        x, open, high, low, close : numpy.ndarray
            All candles, sorted by x.
        width : float
            Width of one candle.
        fromCandle : int
            Candles before this index are unchanged since the previous update; 0 for
            a full computation. Otherwise, at least the number of candles of the
            previous update minus one: candles were appended, possibly replacing
            the last one, so that only the last stored block can hold changed
            candles.
        """
        k = self.k
        count = len(x)
        firstBlock = fromCandle // k
        if fromCandle == 0:
            self.up.clear()
            self.down.clear()
        elif self._lastBlock >= firstBlock:
            self._lastSide.pop()  # the last block holds changed candles
        # as fromCandle never decreases, the blocks stored before cannot change
        self._lastBlock, self._lastSide = -1, None
        first = firstBlock * k
        if first >= count:
            return
        x, open, high, low, close = (a[first:] for a in (x, open, high, low, close))
        valid = (np.isfinite(open) & np.isfinite(high)
                 & np.isfinite(low) & np.isfinite(close))
        allValid = bool(valid.all())
        if k == 1:
            if not allValid:
                x, open, high, low, close = (a[valid] for a in (x, open, high, low, close))
            halfWidth = np.full(len(x), 0.5 * width)
        else:
            x, halfWidth, open, high, low, close = self._aggregate(
                x, open, high, low, close, valid, allValid, width)
        if len(x) == 0:
            return
        bodies, wicks = _geometry(x, halfWidth, open, high, low, close)
        up = close >= open
        if up.all():
            self.up.extend(x, bodies, wicks)
        elif not up.any():
            self.down.extend(x, bodies, wicks)
        else:
            self.up.extend(x, bodies, wicks, np.flatnonzero(up))
            self.down.extend(x, bodies, wicks, np.flatnonzero(~up))
        # the last stored block holds the last valid candle
        lastValid = len(valid) - 1 - int(np.argmax(valid[::-1]))
        self._lastBlock = firstBlock + lastValid // k
        self._lastSide = self.up if up[-1] else self.down

    def _aggregate(self, x: np.ndarray, open: np.ndarray, high: np.ndarray,
                   low: np.ndarray, close: np.ndarray, valid: np.ndarray, allValid: bool,
                   width: float) -> tuple[np.ndarray, ...]:
        """
        Aggregate candles by blocks of ``self.k``, starting at a block boundary.

        Open is the open of the first candle of the block, close the close of the last
        one, high the maximum high and low the minimum low. Invalid candles (with a
        non-finite price) are ignored; blocks without valid candle are dropped.

        Parameters
        ----------
        x, open, high, low, close : numpy.ndarray
            Candles, the first one starting a block.
        valid : numpy.ndarray
            Whether all prices of each candle are finite.
        allValid : bool
            ``valid.all()``.
        width : float
            Width of one candle.

        Returns
        -------
        tuple of numpy.ndarray
            Center, half width, open, high, low and close of the blocks.
        """
        k = self.k
        count = len(x)
        starts = np.arange(0, count, k)
        sizes = np.minimum(k, count - starts)
        ends = starts + sizes - 1
        center = 0.5 * (x[starts] + x[ends])
        halfWidth = 0.5 * width * sizes
        if allValid:
            return (center, halfWidth, open[starts], np.maximum.reduceat(high, starts),
                    np.minimum.reduceat(low, starts), close[ends])
        index = np.arange(count)
        firstValid = np.minimum.reduceat(np.where(valid, index, count), starts)
        lastValid = np.maximum.reduceat(np.where(valid, index, -1), starts)
        keep = firstValid < count
        high = np.fmax.reduceat(np.where(valid, high, np.nan), starts)
        low = np.fmin.reduceat(np.where(valid, low, np.nan), starts)
        return (center[keep], halfWidth[keep], open[firstValid[keep]], high[keep],
                low[keep], close[lastValid[keep]])


class CandlestickItem(GraphicsObject):
    """
    OHLC candlesticks, drawn with a handful of vectorized calls.

    Each candle has a body from its open to its close price, filled with
    ``upBrush`` and outlined with ``upPen`` when the close is at or above the open,
    with ``downBrush`` and ``downPen`` otherwise, and a wick from its low to its high
    price, drawn below the body with ``wickPen`` (or with the pen of the body when
    ``wickPen`` is ``None``).

    Candles are stored sorted by x; only the candles within the visible x range are
    drawn. When ``lod`` is enabled and candles are narrower than 3 device pixels,
    consecutive candles are aggregated by blocks of ``k`` (a power of two): open of
    the first candle, close of the last one, highest high and lowest low. Blocks
    are aligned to multiples of ``k``, so that they do not change while panning.
    Levels of 16 candles per block or more are computed once for all candles and
    cached per ``k`` (appending candles only updates their last block); smaller
    blocks are computed for the visible candles only, which are few.

    Candles with a non-finite x are ignored; candles with a non-finite price are
    not drawn.

    The item implements the ``plotData`` interface, so that a :class:`PlotItem`
    auto-ranges on it and lists it in its legend when it has a name.
    """

    _optionNames = ('upBrush', 'downBrush', 'upPen', 'downPen', 'wickPen', 'width',
                    'lod', 'name')
    _dataNames = ('x', 'open', 'high', 'low', 'close')

    def __init__(self, **opts) -> None:
        """
        Create the item, optionally with data and options.

        Parameters
        ----------
        **opts
            x, open, high, low, close : array_like, optional
                Candles, passed to :meth:`setData` when given (all together).
            width : float or None, default None
                Width of the candle bodies, in x units. ``None`` uses 0.8 times the
                median spacing of x (0.8 with fewer than two distinct x).
            upBrush, downBrush : QBrush or color, optional
                Fill of rising and falling candles; green and red by default.
            upPen, downPen : QPen or color, optional
                Outline of rising and falling candles, and their wicks unless
                ``wickPen`` is set; green and red by default. ``None`` disables it.
            wickPen : QPen or color or None, default None
                Pen of all wicks; ``None`` uses ``upPen`` and ``downPen``.
            lod : bool, default True
                Aggregate candles narrower than 3 device pixels.
            name : str or None, default None
                Name of the item, e.g. shown by a legend.
        """
        GraphicsObject.__init__(self)
        self._x = _GrowableArray()
        self._open = _GrowableArray()
        self._high = _GrowableArray()
        self._low = _GrowableArray()
        self._close = _GrowableArray()
        self._width = 1.0
        self._autoWidth = True
        # cached levels of detail (k >= _CACHED_MIN_K), for all candles
        self._levels = {}
        # level of detail computed for the visible candles only, and its key
        self._window = None
        self._windowKey = None
        # incremented by every data change, invalidates the window
        self._version = 0
        # y range of all candles, and of all candles but the last one (the head),
        # which replacing the last candle starts from; _yCount candles are covered
        self._yBounds = None
        self._yHeadBounds = None
        self._yCount = 0
        self._upBrush = fn.mkBrush('g')
        self._downBrush = fn.mkBrush('r')
        self._upPen = fn.mkPen('g')
        self._downPen = fn.mkPen('r')
        self._wickPen = None
        self._lod = True
        self._name = None
        self._updatePenWidth()

        data = {key: opts.pop(key) for key in self._dataNames if key in opts}
        width = opts.pop('width', None)
        self.setOpts(**opts)
        if data:
            self.setData(width=width, **data)
        elif width is not None:
            self.width = width

    # ------------------------------------------------------------------ data

    def setData(self, *, x: np.ndarray, open: np.ndarray, high: np.ndarray,
                low: np.ndarray, close: np.ndarray, width: float | None = None) -> None:
        """
        Replace all candles.

        Parameters
        ----------
        x : array_like
            Position of the candles, e.g. timestamps. Need not be sorted.
        open, high, low, close : array_like
            Prices, same length as ``x``.
        width : float or None, default None
            Width of the candle bodies. ``None`` keeps a width set explicitly before
            (by ``width`` or :meth:`setOpts`), and otherwise uses 0.8 times the median
            spacing of x.
        """
        x, open, high, low, close = self._normalize(x, open, high, low, close)
        if len(x) > 1 and not (x[1:] >= x[:-1]).all():
            order = np.argsort(x, kind='stable')
            x, open, high, low, close = (a[order] for a in (x, open, high, low, close))
        for store, values in zip(self._stores(), (x, open, high, low, close)):
            store.clear()
            store.extend(values)
        if width is not None:
            self._width = self._checkWidth(width)
            self._autoWidth = False
        elif self._autoWidth:
            self._width = self._defaultWidth()
        self._dataChanged(0)

    def appendData(self, *, x: np.ndarray, open: np.ndarray, high: np.ndarray,
                   low: np.ndarray, close: np.ndarray, replaceLast: bool = False) -> None:
        """
        Append candles, e.g. while streaming, or update the last candle in place.

        In a live chart, each tick updates the current candle (its high, low and
        close; its open is fixed) with ``replaceLast=True``, and a new candle is
        appended when its period starts.

        Candles that follow the existing ones are handled incrementally: only the new
        candles, the bounds and the last aggregated block of each cached level of
        detail are computed, so that the cost does not depend on the number of
        candles. Candles follow the existing ones when their x are sorted and the
        first one is at or after the x of the last candle kept: the last candle, or
        with ``replaceLast=True`` the candle before the replaced one (so the
        replacing candle may keep the x of the replaced one, the usual case, or move
        it, provided it stays at or after its predecessor). Otherwise all candles are
        sorted again, as by :meth:`setData`. Either way the result is the same as
        :meth:`setData` with all candles and the current width; the automatic width
        is not recomputed (unless fewer than two candles were kept).

        Parameters
        ----------
        x : array_like
            Position of the new candles. Candles with a non-finite x are ignored,
            except that the candle replacing the last one must have a finite x.
        open, high, low, close : array_like
            Prices of the new candles, same length as ``x``.
        replaceLast : bool, default False
            If True, the first new candle replaces the last candle instead of
            following it. The item must have at least one candle.

        Raises
        ------
        ValueError
            If the arrays have different lengths, or with ``replaceLast=True``, if
            there is no candle to replace, no new candle or the replacing candle has a
            non-finite x. The item is left unchanged.

        Examples
        --------
        One-minute candles fed by trades:

        >>> item = pg.CandlestickItem(x=[0.0], open=[10], high=[10], low=[10], close=[10])
        >>> item.appendData(x=[0.0], open=[10], high=[12], low=[10], close=[11],
        ...                 replaceLast=True)  # a trade at 12, then one at 11
        >>> item.appendData(x=[60.0], open=[11], high=[11], low=[11], close=[11])
        """
        if replaceLast:
            count = len(self._x)
            if count == 0:
                raise ValueError('replaceLast=True needs an existing candle to replace')
            first = np.asarray(x, dtype=np.float64).reshape(-1)[:1]
            if len(first) == 0:
                raise ValueError('replaceLast=True needs a new candle')
            if not math.isfinite(first[0]):
                raise ValueError('the candle replacing the last one needs a finite x')
        x, open, high, low, close = self._normalize(x, open, high, low, close)
        if len(x) == 0:
            return
        kept = len(self._x) - 1 if replaceLast else len(self._x)  # unchanged candles
        follows = ((kept == 0 or x[0] >= self._x.view[kept - 1])
                   and bool((x[1:] >= x[:-1]).all()))
        if not follows:
            old = (store.view[:kept] for store in self._stores())
            new = (x, open, high, low, close)
            merged = [np.concatenate((a, b)) for a, b in zip(old, new)]
            self.setData(**dict(zip(self._dataNames, merged)))
            return
        for store, values in zip(self._stores(), (x, open, high, low, close)):
            if replaceLast:
                store.pop()
            store.extend(values)
        if self._autoWidth and kept < 2:
            self._width = self._defaultWidth()
            self._dataChanged(0)
        else:
            self._dataChanged(kept)

    def getData(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Positions and close prices of the candles, sorted by x.

        Returns
        -------
        tuple of numpy.ndarray
            ``x`` and ``close`` (read-only views).
        """
        return self._readOnly(self._x), self._readOnly(self._close)

    def getOriginalDataset(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Positions and close prices of the candles, as exported by :class:`CSVExporter`.

        Returns
        -------
        tuple of numpy.ndarray
            Same as :meth:`getData`.
        """
        return self.getData()

    def _stores(self) -> tuple[_GrowableArray, ...]:
        """
        Storage of the candles.

        Returns
        -------
        tuple of _GrowableArray
            x, open, high, low and close.
        """
        return self._x, self._open, self._high, self._low, self._close

    @staticmethod
    def _normalize(*arrays) -> list[np.ndarray]:
        """
        Convert candle arrays to float64 and drop candles with a non-finite x.

        Parameters
        ----------
        *arrays : array_like
            x, open, high, low and close.

        Returns
        -------
        list of numpy.ndarray
            One-dimensional arrays of equal length.
        """
        arrays = [np.asarray(a, dtype=np.float64).reshape(-1) for a in arrays]
        if len({len(a) for a in arrays}) != 1:
            raise ValueError('x, open, high, low and close must have the same length')
        finite = np.isfinite(arrays[0])
        if not finite.all():
            arrays = [a[finite] for a in arrays]
        return arrays

    def _defaultWidth(self) -> float:
        """
        Default candle width: 0.8 times the median spacing of x.

        Returns
        -------
        float
            Width in x units; 0.8 when it cannot be determined.
        """
        x = self._x.view
        if len(x) > 1:
            steps = np.diff(x)
            steps = steps[steps > 0]
            if len(steps):
                return _DEFAULT_WIDTH_RATIO * float(np.median(steps))
        return _DEFAULT_WIDTH_RATIO

    @staticmethod
    def _checkWidth(width: float) -> float:
        """
        Validate a candle width.

        Parameters
        ----------
        width : float
            Width in x units.

        Returns
        -------
        float
            The width.
        """
        width = float(width)
        if not (math.isfinite(width) and width >= 0):
            raise ValueError(f'width must be a finite non-negative number, got {width}')
        return width

    def _dataChanged(self, fromCandle: int) -> None:
        """
        Update the cached geometry and bounds after a data or width change.

        Parameters
        ----------
        fromCandle : int
            Index of the first new or replaced candle after :meth:`appendData`;
            candles before it are unchanged. 0 when all candles or the width changed.
        """
        self._version += 1
        if fromCandle == 0:
            self._levels = {}
            self._yBounds = self._yHeadBounds = None
        else:
            arrays = [store.view for store in self._stores()]
            for level in self._levels.values():
                level.update(*arrays, self._width, fromCandle)
            if self._yBounds is not None:
                # the head bounds cover [0, fromCandle) before the change: the
                # previous bounds, or the previous head bounds after replacing the
                # last candle, which may have held the highest or lowest price
                count = len(self._x)
                base = self._yHeadBounds if fromCandle < self._yCount else self._yBounds
                self._setYBounds(_union(base, self._yRange(fromCandle, count - 1)))
        self.prepareGeometryChange()
        self.update()
        self.informViewBoundsChanged()

    def _setYBounds(self, head: tuple[float | None, float | None]) -> None:
        """
        Cache the y range of all candles, from that of all candles but the last.

        Parameters
        ----------
        head : tuple of float or None
            Lowest and highest price of all candles but the last one.
        """
        count = len(self._x)
        self._yHeadBounds = head
        self._yBounds = _union(head, self._yRange(count - 1, count))
        self._yCount = count

    def _level(self, k: int, xmin: float, xmax: float) -> _Level:
        """
        Candles aggregated by blocks of ``k``, at least those within an x range.

        Levels with ``k >= 16`` cover all candles and are cached until the data
        changes (appending only updates them). Smaller blocks are computed for the
        candles within the range only, plus one block on each side.

        Parameters
        ----------
        k : int
            Number of candles per block; 1 for the candles themselves.
        xmin, xmax : float
            Range of x that must be covered.

        Returns
        -------
        _Level
            The level.
        """
        arrays = [store.view for store in self._stores()]
        if k >= _CACHED_MIN_K:
            level = self._levels.get(k)
            if level is None:
                level = _Level(k)
                level.update(*arrays, self._width, 0)
                self._levels[k] = level
            return level
        x = arrays[0]
        start = int(np.searchsorted(x, xmin, side='left')) // k - 1
        stop = -(-int(np.searchsorted(x, xmax, side='right')) // k) + 1
        start, stop = max(start, 0) * k, min(stop * k, len(x))
        key = (k, start, stop, self._version)
        if self._windowKey != key:
            if self._window is None:
                self._window = _Level(k)
            self._window.k = k
            self._window.update(*(a[start:stop] for a in arrays), self._width, 0)
            self._windowKey = key
        return self._window

    # --------------------------------------------------------------- options

    def setOpts(self, **opts) -> None:
        """
        Set several options at once.

        Parameters
        ----------
        **opts
            Any of ``upBrush``, ``downBrush``, ``upPen``, ``downPen``, ``wickPen``,
            ``width``, ``lod`` and ``name``; see the properties of the same name.
        """
        unknown = set(opts) - set(self._optionNames)
        if unknown:
            names = ', '.join(sorted(unknown))
            raise TypeError(f'unknown CandlestickItem option(s): {names}')
        for key, value in opts.items():
            if key == 'name':
                self._name = value
            else:
                setattr(self, key, value)

    @property
    def upBrush(self) -> QtGui.QBrush:
        """QtGui.QBrush: Fill of the rising candles (close at or above open)."""
        return self._upBrush

    @upBrush.setter
    def upBrush(self, brush) -> None:
        self._upBrush = fn.mkBrush(brush)
        self.update()

    @property
    def downBrush(self) -> QtGui.QBrush:
        """QtGui.QBrush: Fill of the falling candles (close below open)."""
        return self._downBrush

    @downBrush.setter
    def downBrush(self, brush) -> None:
        self._downBrush = fn.mkBrush(brush)
        self.update()

    @property
    def upPen(self) -> QtGui.QPen:
        """QtGui.QPen: Outline of the rising candles, and their wicks by default."""
        return self._upPen

    @upPen.setter
    def upPen(self, pen) -> None:
        self._upPen = fn.mkPen(pen)
        self._penChanged()

    @property
    def downPen(self) -> QtGui.QPen:
        """QtGui.QPen: Outline of the falling candles, and their wicks by default."""
        return self._downPen

    @downPen.setter
    def downPen(self, pen) -> None:
        self._downPen = fn.mkPen(pen)
        self._penChanged()

    @property
    def wickPen(self) -> QtGui.QPen | None:
        """QtGui.QPen or None: Pen of all wicks; ``None`` uses ``upPen``/``downPen``."""
        return self._wickPen

    @wickPen.setter
    def wickPen(self, pen) -> None:
        self._wickPen = None if pen is None else fn.mkPen(pen)
        self._penChanged()

    @property
    def width(self) -> float:
        """
        float: Width of the candle bodies, in x units.

        Setting ``None`` restores the default, 0.8 times the median spacing of x.
        """
        return self._width

    @width.setter
    def width(self, width: float | None) -> None:
        if width is None:
            self._autoWidth = True
            width = self._defaultWidth()
        else:
            self._autoWidth = False
            width = self._checkWidth(width)
        if width != self._width:
            self._width = width
            self._dataChanged(0)

    @property
    def lod(self) -> bool:
        """bool: Whether candles narrower than 3 device pixels are aggregated."""
        return self._lod

    @lod.setter
    def lod(self, lod: bool) -> None:
        self._lod = bool(lod)
        self.update()

    @property
    def ohlc(self) -> tuple[np.ndarray, ...]:
        """
        tuple of numpy.ndarray: The candles, sorted by x.

        ``(x, open, high, low, close)``, as read-only views.
        """
        return tuple(self._readOnly(store) for store in self._stores())

    @property
    def opts(self) -> dict:
        """
        dict: Description of the legend sample, as read by :class:`LegendItem`.

        The sample is a rising candle: no line, a symbol filled with ``upBrush``.
        """
        return {
            'name': self._name,
            'pen': None,
            'brush': self._upBrush,
            'symbol': _legendSymbol(),
            'size': 16,
        }

    @staticmethod
    def _readOnly(store: _GrowableArray) -> np.ndarray:
        """
        Read-only view of a candle array.

        Parameters
        ----------
        store : _GrowableArray
            Candle storage.

        Returns
        -------
        numpy.ndarray
            View that cannot be written to.
        """
        view = store.view
        view.flags.writeable = False
        return view

    def _penChanged(self) -> None:
        """Update the pen widths used by the bounds after a pen change."""
        self._updatePenWidth()
        self.prepareGeometryChange()
        self.update()
        self.informViewBoundsChanged()

    def _updatePenWidth(self) -> None:
        """Store the widest pen widths, ``[non-cosmetic in data units, cosmetic in px]``."""
        widths = [0.0, 0.0]
        for pen in (self._upPen, self._downPen, self._wickPen):
            if pen is not None and pen.style() != QtCore.Qt.PenStyle.NoPen:
                cosmetic = int(pen.isCosmetic())
                widths[cosmetic] = max(widths[cosmetic], pen.widthF())
        self._penWidth = widths

    # ----------------------------------------------------- plotData interface

    def implements(self, interface: str | None = None) -> bool | list[str]:
        """
        Interfaces implemented by the item.

        Parameters
        ----------
        interface : str or None, default None
            Interface name to test.

        Returns
        -------
        bool or list of str
            Whether ``interface`` is implemented, or the list of interfaces.
        """
        interfaces = ['plotData']
        if interface is None:
            return interfaces
        return interface in interfaces

    def name(self) -> str | None:
        """
        Name of the item, e.g. shown by a legend.

        Returns
        -------
        str or None
            The ``name`` option.
        """
        return self._name

    # -------------------------------------------------------------- geometry

    def _yRange(self, start: int, stop: int) -> tuple[float | None, float | None]:
        """
        Lowest and highest price of a range of candles.

        Parameters
        ----------
        start, stop : int
            Range of candles.

        Returns
        -------
        tuple of float or None
            ``(None, None)`` when the range has no finite price.
        """
        if stop <= start:
            return None, None
        span = slice(start, stop)
        low = min(np.fmin.reduce(store.view[span])
                  for store in (self._low, self._open, self._close))
        high = max(np.fmax.reduce(store.view[span])
                   for store in (self._high, self._open, self._close))
        if math.isnan(low) or math.isnan(high):
            return None, None
        return float(low), float(high)

    def dataBounds(self, ax: int, frac: float = 1.0,
                   orthoRange: tuple[float, float] | None = None
                   ) -> tuple[float | None, float | None]:
        """
        Range of the data along an axis.

        Parameters
        ----------
        ax : int
            0 for x, 1 for y.
        frac : float, default 1.0
            Ignored; the full range is always returned.
        orthoRange : tuple of float or None, default None
            For ``ax=1``, only the candles whose body intersects this x range
            (bounds included) are considered, so that the y range fits the visible
            candles, from their lowest low to their highest high: this is what
            ``ViewBox.setAutoVisible(y=True)`` uses. The candles are found by binary
            search, and the range of all candles is cached. Ignored for ``ax=0``.

        Returns
        -------
        tuple of float or None
            ``(min, max)``, or ``(None, None)`` without data or, with
            ``orthoRange``, without candle within it.
        """
        count = len(self._x)
        if count == 0:
            return None, None
        penPad = 0.5 * self._penWidth[0]
        if ax == 0:
            x = self._x.view
            pad = 0.5 * self._width + penPad
            return float(x[0]) - pad, float(x[-1]) + pad
        start, stop = 0, count
        if orthoRange is not None:
            x = self._x.view
            halfWidth = 0.5 * self._width
            xmin, xmax = min(orthoRange), max(orthoRange)
            start = int(np.searchsorted(x, xmin - halfWidth, side='left'))
            stop = int(np.searchsorted(x, xmax + halfWidth, side='right'))
        if start == 0 and stop == count:
            # all candles, e.g. a zoomed out view
            if self._yBounds is None:
                self._setYBounds(self._yRange(0, count - 1))
            low, high = self._yBounds
        else:
            low, high = self._yRange(start, stop)
        if low is None:
            return None, None
        return low - penPad, high + penPad

    def pixelPadding(self) -> float:
        """
        Padding needed around the data bounds for cosmetic pens.

        Returns
        -------
        float
            Half the widest cosmetic pen width, in device pixels.
        """
        return (self._penWidth[1] or 1) * 0.5

    def boundingRect(self) -> QtCore.QRectF:
        """
        Bounds of all candles, including pens.

        Returns
        -------
        QtCore.QRectF
            Rectangle in item coordinates; empty without data.
        """
        xmn, xmx = self.dataBounds(ax=0)
        ymn, ymx = self.dataBounds(ax=1)
        if xmn is None or ymn is None:
            return QtCore.QRectF()
        px = py = 0.0
        pxPad = self.pixelPadding()
        if pxPad > 0:
            px, py = self.pixelVectors()
            px = 0.0 if px is None else px.length() * pxPad
            py = 0.0 if py is None else py.length() * pxPad
        return QtCore.QRectF(xmn - px, ymn - py, (2 * px) + xmx - xmn, (2 * py) + ymx - ymn)

    # -------------------------------------------------------------- painting

    def _lodFactor(self, pxPerUnit: float) -> int:
        """
        Number of candles aggregated per drawn candle.

        Parameters
        ----------
        pxPerUnit : float
            Device pixels per unit of x.

        Returns
        -------
        int
            The smallest power of two making aggregated candles at least 3 device
            pixels wide; 1 when candles are wide enough or ``lod`` is off.
        """
        count = len(self._x)
        widthPx = self._width * pxPerUnit
        if not self._lod or count < 2 or widthPx <= 0 or widthPx >= _LOD_MIN_WIDTH_PX:
            return 1
        k = 2 ** math.ceil(math.log2(_LOD_MIN_WIDTH_PX / widthPx))
        return min(k, 2 ** math.ceil(math.log2(count)))

    def paint(self, p: QtGui.QPainter, *args) -> None:
        """
        Draw the visible candles, aggregated when zoomed out.

        Parameters
        ----------
        p : QtGui.QPainter
            Destination painter, mapping item coordinates to the device.
        *args
            ``QStyleOptionGraphicsItem`` and widget, unused.
        """
        if len(self._x) == 0:
            return
        tr = p.combinedTransform()
        pxPerUnit = math.hypot(tr.m11(), tr.m12())  # device pixels per unit of x
        k = self._lodFactor(pxPerUnit)
        rect = _visibleRect(self, p) if pxPerUnit > 0 else None
        if rect is None:
            xmin, xmax = -math.inf, math.inf
        else:
            # bodies extend beyond their center by half their width, outlines by half
            # the pen width, plus antialiasing
            pad = (0.5 * self._width * k + 0.5 * self._penWidth[0]
                   + (0.5 * (self._penWidth[1] or 1) + 1.0) / pxPerUnit)
            xmin, xmax = rect.left() - pad, rect.right() + pad
        level = self._level(k, xmin, xmax)
        sides = (
            (level.up, self._upPen, self._upBrush),
            (level.down, self._downPen, self._downBrush),
        )
        ranges = [side.visible(xmin, xmax) for side, _, _ in sides]
        for (side, pen, _), (start, stop) in zip(sides, ranges):
            if stop > start:
                p.setPen(pen if self._wickPen is None else self._wickPen)
                p.drawLines(*side.wicks.drawargs(start, stop))
        for (side, pen, brush), (start, stop) in zip(sides, ranges):
            if stop > start:
                p.setPen(pen)
                p.setBrush(brush)
                p.drawRects(*side.bodies.drawargs(start, stop))


def _legendSymbol() -> QtGui.QPainterPath:
    """
    Candle-shaped legend symbol, a filled body and wick in a unit square.

    Returns
    -------
    QtGui.QPainterPath
        Path centered on the origin, spanning -0.5 to 0.5.
    """
    path = QtGui.QPainterPath()
    path.setFillRule(QtCore.Qt.FillRule.WindingFill)
    path.addRect(QtCore.QRectF(-0.25, -0.3, 0.5, 0.6))
    path.addRect(QtCore.QRectF(-0.04, -0.5, 0.08, 1.0))
    return path
