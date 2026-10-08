import itertools
import math
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

from .. import functions as fn
from .. import getConfigOption
from .. import Qt
from ..Qt import QtCore, QtGui, QtWidgets
from .GraphicsObject import GraphicsObject

__all__ = ['BarGraphItem']

# With ``lodPen``, outlines are not drawn when the mean bar width is below this many
# device pixels: they would cover most of each bar and their cost is pure overdraw.
_LOD_PEN_MIN_WIDTH_PX = 2.0
# With ``lod``, bars are aggregated per device pixel column when there are more than
# this many visible bars per pixel column.
_LOD_MAX_BARS_PER_PIXEL = 1.0
# Options giving the geometry of the bars, each with one value per bar or for all bars.
_COORDINATE_KEYS = ('x', 'x0', 'x1', 'width', 'y', 'y0', 'y1', 'height')
# Options giving one style per bar, and the option giving one style to all bars.
_STYLE_KEYS = {'pens': 'pen', 'brushes': 'brush'}


def _penKey(pen: QtGui.QPen) -> tuple:
    """
    Cheap hashable summary of a pen, used to bucket pens before comparing them.

    Parameters
    ----------
    pen : QtGui.QPen
        Pen to summarize.

    Returns
    -------
    tuple
        Equal pens have equal keys; different pens may share a key.
    """
    return pen.style(), pen.widthF(), pen.isCosmetic(), pen.color().rgba()


def _brushKey(brush: QtGui.QBrush) -> tuple:
    """
    Cheap hashable summary of a brush, used to bucket brushes before comparing them.

    Parameters
    ----------
    brush : QtGui.QBrush
        Brush to summarize.

    Returns
    -------
    tuple
        Equal brushes have equal keys; different brushes may share a key.
    """
    return brush.style(), brush.color().rgba()


def _uniqueStyles(specs: Iterable, klass: type, make: Callable,
                  key: Callable) -> tuple[list, np.ndarray]:
    """
    Deduplicate a per-bar list of pens or brushes.

    Each distinct object is converted once with ``make`` (unless it already is a
    ``klass`` instance), then the results are merged by value, so that a list built
    from color tuples yields as many styles as there are distinct colors.

    Parameters
    ----------
    specs : iterable
        One pen or brush specification per bar, anything accepted by ``make``.
    klass : type
        ``QtGui.QPen`` or ``QtGui.QBrush``.
    make : callable
        ``fn.mkPen`` or ``fn.mkBrush``.
    key : callable
        :func:`_penKey` or :func:`_brushKey`.

    Returns
    -------
    uniques : list
        Distinct ``klass`` instances, in order of first appearance.
    ids : numpy.ndarray
        Index into ``uniques`` of the style of each entry of ``specs``.
    """
    objs = list(specs)  # keeps every object alive, so that their ids are unique
    if not objs:
        return [], np.zeros(0, dtype=np.intp)
    oids = np.fromiter(map(id, objs), dtype=np.uintp, count=len(objs))
    # distinct objects in order of first appearance; few of them is the common case,
    # which a handful of vectorized comparisons handle faster than a sort
    distinct = []
    rest = oids
    while len(rest) and len(distinct) < 8:
        distinct.append(rest[0])
        rest = rest[rest != rest[0]]
    if len(rest) == 0:
        first = np.empty(len(distinct), dtype=np.intp)
        inverse = np.empty(len(oids), dtype=np.intp)
        for j, oid in enumerate(distinct):
            mask = oids == oid
            inverse[mask] = j
            first[j] = mask.argmax()
    else:
        _, first, inverse = np.unique(oids, return_index=True, return_inverse=True)
        appearance = np.argsort(first, kind='stable')
        first = first[appearance]
        rank = np.empty_like(appearance)
        rank[appearance] = np.arange(len(appearance))
        inverse = rank[inverse.reshape(-1)]
    uniques = []
    buckets = {}
    remap = np.empty(len(first), dtype=np.intp)
    for j, index in enumerate(first.tolist()):
        obj = objs[index]
        if not isinstance(obj, klass):
            obj = make(obj)
        bucket = buckets.setdefault(key(obj), [])
        for k in bucket:
            if uniques[k] == obj:
                break
        else:
            k = len(uniques)
            uniques.append(obj)
            bucket.append(k)
        remap[j] = k
    return uniques, remap[inverse]


def _runs(values: np.ndarray) -> list[tuple[int, int, int]]:
    """
    Runs of equal consecutive values.

    Parameters
    ----------
    values : numpy.ndarray
        One-dimensional integer array.

    Returns
    -------
    list of tuple of int
        ``(value, begin, end)`` for each run: ``values[begin:end]`` all equal ``value``.
    """
    if len(values) == 0:
        return []
    cuts = (np.flatnonzero(values[1:] != values[:-1]) + 1).tolist()
    edges = [0, *cuts, len(values)]
    return [(int(values[b]), b, e) for b, e in zip(edges[:-1], edges[1:])]


def _visibleRect(item: GraphicsObject, painter: QtGui.QPainter) -> QtCore.QRectF | None:
    """
    Part of ``item`` that can be visible while ``painter`` paints it.

    This is the view rectangle of the item's ViewBox (or GraphicsView) when it has one.
    An item painted outside of any view (e.g. directly into a ``QImage``) uses the area
    of the paint device, intersected with the painter clip.

    Parameters
    ----------
    item : GraphicsObject
        Item being painted.
    painter : QtGui.QPainter
        Active painter, whose transform maps item coordinates to the device.

    Returns
    -------
    QtCore.QRectF or None
        Rectangle in item coordinates, or ``None`` when it cannot be determined, in
        which case nothing should be culled.
    """
    rect = item.viewRect()
    if rect is not None:
        return rect
    device = painter.device()
    # only raster-like devices have a meaningful drawable area
    if not isinstance(device, (QtGui.QImage, QtGui.QPixmap, QtWidgets.QWidget)):
        return None
    width, height = device.width(), device.height()
    if width <= 0 or height <= 0:
        return None
    inverse, invertible = painter.combinedTransform().inverted()
    if not invertible:
        return None
    rect = inverse.mapRect(QtCore.QRectF(0, 0, width, height))
    if painter.hasClipping():
        rect = rect.intersected(painter.clipBoundingRect())
    return rect


class _GrowableArray:
    """
    One-dimensional array with a logical length and a geometric capacity.

    Appending is amortised O(1) per element. The interface is a superset of the
    class of the same name in :mod:`pyqtgraph.graphicsItems.CandlestickItem`.

    Parameters
    ----------
    dtype : numpy.dtype, default numpy.float64
        Element type.
    """

    def __init__(self, dtype: np.dtype = np.float64) -> None:
        self._data = np.empty(0, dtype=dtype)
        self._size = 0

    @classmethod
    def wrap(cls, values: np.ndarray) -> '_GrowableArray':
        """
        Array holding ``values`` without copying them.

        Parameters
        ----------
        values : numpy.ndarray
            One-dimensional contiguous array, handed over: nothing else may write
            to it afterwards.

        Returns
        -------
        _GrowableArray
            Array of ``len(values)`` elements, whose capacity is ``values`` itself.
        """
        array = cls(values.dtype)
        array._data = values
        array._size = len(values)
        return array

    def __len__(self) -> int:
        return self._size

    @property
    def view(self) -> np.ndarray:
        """numpy.ndarray: The ``len(self)`` valid elements (a view, not a copy)."""
        return self._data[:self._size]

    def clear(self) -> None:
        """Remove all elements, keeping the capacity."""
        self._size = 0

    def extend(self, values: np.ndarray) -> None:
        """
        Append elements, growing the capacity geometrically when needed.

        Parameters
        ----------
        values : numpy.ndarray
            Elements to append.
        """
        need = self._size + len(values)
        if need > len(self._data):
            data = np.empty(max(need, len(self._data) * 3 // 2), dtype=self._data.dtype)
            data[:self._size] = self._data[:self._size]
            self._data = data
        self._data[self._size:need] = values
        self._size = need

    def pop(self) -> None:
        """Remove the last element."""
        self._size -= 1

    def truncate(self, size: int) -> None:
        """
        Keep the first elements only.

        Parameters
        ----------
        size : int
            Number of elements kept, if there are more.
        """
        self._size = min(self._size, size)


class _PrimitiveBuffer:
    """
    ``PrimitiveArray`` with a logical length and a geometric capacity.

    The underlying array is only ever resized to its capacity, and drawn through
    ``drawargs(start, stop)``, so that appending keeps the existing primitives with
    every binding, including ``sip.array`` older than 6.7.8 which reallocates on any
    resize. The interface is a superset of the class of the same name in
    :mod:`pyqtgraph.graphicsItems.CandlestickItem`.

    Parameters
    ----------
    klass : type
        ``QtCore.QRectF`` or ``QtCore.QLineF``.
    """

    def __init__(self, klass: type) -> None:
        self._array = Qt.internals.PrimitiveArray(klass, 4)
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def ndarray(self) -> np.ndarray:
        """
        Coordinates of the primitives.

        Returns
        -------
        numpy.ndarray
            View of shape ``(len(self), 4)``.
        """
        return self._array.ndarray()[:self._size]

    def clear(self) -> None:
        """Remove all primitives, keeping the capacity."""
        self._size = 0

    def pop(self) -> None:
        """Remove the last primitive."""
        self._size -= 1

    def resize(self, count: int) -> None:
        """
        Set the number of primitives, without keeping their coordinates.

        Parameters
        ----------
        count : int
            Number of primitives, whose coordinates are then to be filled.
        """
        if count > len(self._array):
            self._array.resize(count)
        self._size = count

    def reserve(self, count: int) -> np.ndarray:
        """
        Append ``count`` primitives and return their coordinates, to be filled.

        Parameters
        ----------
        count : int
            Number of primitives to append.

        Returns
        -------
        numpy.ndarray
            Writable view of shape ``(count, 4)``.
        """
        need = self._size + count
        capacity = len(self._array)
        if need > capacity:
            kept = self._array.ndarray()[:self._size].copy()
            self._array.resize(max(need, capacity * 3 // 2))
            self._array.ndarray()[:self._size] = kept
        view = self._array.ndarray()[self._size:need]
        self._size = need
        return view

    def drawargs(self, start: int | None = None, stop: int | None = None) -> tuple:
        """
        Arguments to draw a range of the primitives.

        Parameters
        ----------
        start, stop : int or None, default None
            Range of primitives, following Python slicing rules within
            ``len(self)``; all primitives by default.

        Returns
        -------
        tuple
            Arguments for ``QPainter.drawRects`` or ``QPainter.drawLines``.
        """
        start, stop, _ = slice(start, stop).indices(self._size)
        return self._array.drawargs(start, max(start, stop))

    def instances(self) -> Iterator:
        """
        The primitives, as Qt objects sharing the coordinates.

        Returns
        -------
        iterator
            The first ``len(self)`` primitives.
        """
        return itertools.islice(self._array.instances(), self._size)


def _normalizedCoords(opts: Mapping) -> tuple[np.ndarray, ...]:
    """
    Edges of the bars described by the coordinate options.

    Parameters
    ----------
    opts : mapping
        ``x``, ``x0``, ``x1``, ``width``, ``y``, ``y0``, ``y1`` and ``height``, each
        missing, ``None``, a float or one value per bar; see :class:`BarGraphItem`.

    Returns
    -------
    x0, y0, x1, y1 : numpy.ndarray
        Left, bottom, right and top edges, ``x0 <= x1`` and ``y0 <= y1``; float64
        arrays or 0-d values broadcastable to the number of bars.
    """
    def asarray(x):
        if x is None or np.isscalar(x) or isinstance(x, np.ndarray):
            return x
        return np.array(x)

    x = asarray(opts.get('x'))
    x0 = asarray(opts.get('x0'))
    x1 = asarray(opts.get('x1'))
    width = asarray(opts.get('width'))

    if x0 is None:
        if width is None:
            raise Exception('must specify either x0 or width')
        if x1 is not None:
            x0 = x1 - width
        elif x is not None:
            x0 = x - width/2.
        else:
            raise Exception('must specify at least one of x, x0, or x1')
    if width is None:
        if x1 is None:
            raise Exception('must specify either x1 or width')
        width = x1 - x0

    y = asarray(opts.get('y'))
    y0 = asarray(opts.get('y0'))
    y1 = asarray(opts.get('y1'))
    height = asarray(opts.get('height'))

    if y0 is None:
        if height is None:
            y0 = 0
        elif y1 is not None:
            y0 = y1 - height
        elif y is not None:
            y0 = y - height/2.
        else:
            y0 = 0
    if height is None:
        if y1 is None:
            raise Exception('must specify either y1 or height')
        height = y1 - y0

    # ensure x0 < x1 and y0 < y1
    t0, t1 = x0, x0 + width
    x0 = np.minimum(t0, t1, dtype=np.float64)
    x1 = np.maximum(t0, t1, dtype=np.float64)
    t0, t1 = y0, y0 + height
    y0 = np.minimum(t0, t1, dtype=np.float64)
    y1 = np.maximum(t0, t1, dtype=np.float64)

    # here, all of x0, y0, x1, y1 are numpy objects,
    # BUT could possibly be numpy scalars
    return x0, y0, x1, y1


def _finiteBounds(x0: np.ndarray, y0: np.ndarray, x1: np.ndarray,
                  y1: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Bounds of the bars whose edges are all finite.

    Parameters
    ----------
    x0, y0, x1, y1 : numpy.ndarray
        Edges of the bars, ``x0 <= x1`` and ``y0 <= y1`` (or NaN), one value per bar.

    Returns
    -------
    tuple of tuple of float
        ``((xmin, xmax), (ymin, ymax))``; NaN when no bar has finite edges.
    """
    bounds = (float(np.min(x0)), float(np.max(x1)), float(np.min(y0)), float(np.max(y1)))
    if not all(map(math.isfinite, bounds)):
        # a NaN propagates to the bounds, and as x0 <= x1 and y0 <= y1, so does an
        # infinite edge: only then are bars left out
        finite = np.isfinite(x0) & np.isfinite(x1) & np.isfinite(y0) & np.isfinite(y1)
        if finite.any():
            bounds = (float(np.min(x0[finite])), float(np.max(x1[finite])),
                      float(np.min(y0[finite])), float(np.max(y1[finite])))
        else:
            bounds = (math.nan,) * 4
    return (bounds[0], bounds[1]), (bounds[2], bounds[3])


def _lower(a: float, b: float) -> float:
    """
    Lower of two bounds, NaN standing for no bound (as :func:`numpy.fmin`).

    Parameters
    ----------
    a, b : float
        Bounds.

    Returns
    -------
    float
        The smaller one; the other one when one is NaN.
    """
    return b if math.isnan(a) else a if math.isnan(b) else min(a, b)


def _upper(a: float, b: float) -> float:
    """
    Upper of two bounds, NaN standing for no bound (as :func:`numpy.fmax`).

    Parameters
    ----------
    a, b : float
        Bounds.

    Returns
    -------
    float
        The larger one; the other one when one is NaN.
    """
    return b if math.isnan(a) else a if math.isnan(b) else max(a, b)


class _BarStats:
    """
    Summary of consecutive bars: bounds, order along x and widths.

    The summary of bars followed by other bars is :meth:`then`, so that appending
    bars only summarizes the new ones. An empty summary (``count == 0``) is the
    neutral element.

    Attributes
    ----------
    count : int
        Number of bars.
    bounds : tuple of tuple of float
        ``((xmin, xmax), (ymin, ymax))`` of the bars whose edges are all finite,
        NaN when there is none.
    finite : bool
        Whether the left and right edges of all rectangles are finite.
    x0Sorted, x1Sorted : bool
        Whether the left, respectively right, edges increase.
    firstX0, firstX1, lastX0, lastX1 : float
        Left and right edges of the first and of the last bar.
    widthSum : float
        Sum of the finite widths.
    widthCount : int
        Number of finite widths.
    """

    __slots__ = ('bounds', 'count', 'finite', 'firstX0', 'firstX1', 'lastX0', 'lastX1',
                 'widthCount', 'widthSum', 'x0Sorted', 'x1Sorted')

    def __init__(self) -> None:
        self.count = 0
        self.bounds = ((math.nan, math.nan), (math.nan, math.nan))
        self.finite = True
        self.x0Sorted = self.x1Sorted = True
        self.firstX0 = self.firstX1 = self.lastX0 = self.lastX1 = math.nan
        self.widthSum = 0.0
        self.widthCount = 0

    @classmethod
    def of(cls, x0: np.ndarray, y0: np.ndarray, x1: np.ndarray, y1: np.ndarray,
           left: np.ndarray, widths: np.ndarray, right: np.ndarray) -> '_BarStats':
        """
        Summary of bars.

        Parameters
        ----------
        x0, y0, x1, y1 : numpy.ndarray
            Edges of the bars as normalized from the options, one value per bar.
        left, widths, right : numpy.ndarray
            Left edge, width and right edge (``left + widths``) of their rectangles.

        Returns
        -------
        _BarStats
            The summary.
        """
        stats = cls()
        count = len(left)
        if count == 0:
            return stats
        stats.count = count
        stats.bounds = _finiteBounds(x0, y0, x1, y1)
        stats.finite = bool(np.isfinite(right).all())
        if stats.finite:
            stats.widthSum, stats.widthCount = float(widths.sum()), count
        else:
            finite = np.isfinite(widths)
            stats.widthSum, stats.widthCount = float(widths[finite].sum()), int(finite.sum())
        stats.x0Sorted = bool((left[1:] >= left[:-1]).all())
        stats.x1Sorted = bool((right[1:] >= right[:-1]).all())
        stats.firstX0, stats.lastX0 = float(left[0]), float(left[-1])
        stats.firstX1, stats.lastX1 = float(right[0]), float(right[-1])
        return stats

    def then(self, other: '_BarStats') -> '_BarStats':
        """
        Summary of these bars followed by other bars.

        Parameters
        ----------
        other : _BarStats
            Summary of the bars that follow.

        Returns
        -------
        _BarStats
            Summary of all bars.
        """
        if other.count == 0:
            return self
        if self.count == 0:
            return other
        stats = _BarStats()
        stats.count = self.count + other.count
        (a, b), (c, d) = self.bounds
        (e, f), (g, h) = other.bounds
        stats.bounds = ((_lower(a, e), _upper(b, f)), (_lower(c, g), _upper(d, h)))
        stats.finite = self.finite and other.finite
        stats.x0Sorted = self.x0Sorted and other.x0Sorted and other.firstX0 >= self.lastX0
        stats.x1Sorted = self.x1Sorted and other.x1Sorted and other.firstX1 >= self.lastX1
        stats.firstX0, stats.firstX1 = self.firstX0, self.firstX1
        stats.lastX0, stats.lastX1 = other.lastX0, other.lastX1
        stats.widthSum = self.widthSum + other.widthSum
        stats.widthCount = self.widthCount + other.widthCount
        return stats


class _StyleTable:
    """
    Pens or brushes of the bars: their distinct values and the style of each bar.

    Parameters
    ----------
    klass : type
        ``QtGui.QPen`` or ``QtGui.QBrush``.
    make : callable
        ``fn.mkPen`` or ``fn.mkBrush``.
    key : callable
        :func:`_penKey` or :func:`_brushKey`.
    """

    def __init__(self, klass: type, make: Callable, key: Callable) -> None:
        self._klass = klass
        self._make = make
        self._key = key
        # distinct styles, in order of first appearance
        self.uniques: list = []
        # index into uniques of the style of each bar; None when all bars share uniques[0]
        self.ids: _GrowableArray | None = None
        # indices of the uniques by key, and number of bars of each unique; on demand
        self._buckets: dict | None = None
        self._counts: np.ndarray | None = None

    @property
    def perBar(self) -> bool:
        """bool: Whether each bar has its own style index."""
        return self.ids is not None

    def setShared(self, spec) -> None:
        """
        Give one style to all bars.

        Parameters
        ----------
        spec : QPen or QBrush or color
            Style, anything accepted by ``make``.
        """
        self.uniques = [self._make(spec)]
        self.ids = None
        self._buckets = self._counts = None

    def setPerBar(self, specs: Iterable) -> None:
        """
        Give one style to each bar.

        Parameters
        ----------
        specs : iterable
            One style per bar, anything accepted by ``make``.
        """
        self.uniques, ids = _uniqueStyles(specs, self._klass, self._make, self._key)
        self.ids = _GrowableArray.wrap(ids)
        self._buckets = self._counts = None

    def convert(self, specs: Sequence) -> tuple[list, np.ndarray]:
        """
        Distinct styles of new bars, without changing the table.

        Parameters
        ----------
        specs : sequence
            One style per new bar, anything accepted by ``make``.

        Returns
        -------
        styles : list
            Distinct styles, ``klass`` instances.
        ids : numpy.ndarray
            Index into ``styles`` of the style of each new bar.
        """
        return _uniqueStyles(specs, self._klass, self._make, self._key)

    def append(self, styles: list, ids: np.ndarray, nbars: int, replaceLast: bool) -> bool:
        """
        Append the styles of new bars.

        The distinct styles stay those used by the bars, in order of first
        appearance: the style indices are those of setting all styles at once.

        Parameters
        ----------
        styles, ids : list, numpy.ndarray
            Styles of the new bars, as returned by :meth:`convert`.
        nbars : int
            Number of bars before the call. Style indices beyond it, from more styles
            than bars, are dropped first.
        replaceLast : bool
            Whether the first new bar replaces the last bar.

        Returns
        -------
        bool
            Whether :attr:`uniques` changed (by value).
        """
        truncated = len(self.ids) > nbars
        if truncated:
            self.ids.truncate(nbars)
            self._counts = None
        removed = []
        if replaceLast or truncated:
            # styles may have lost their last bar: count the bars of each style
            self._usage()
            if replaceLast:
                self._counts[self.ids.view[-1]] -= 1
                self.ids.pop()
            # a style used by no bar can only be among the last ones
            removed = self._dropUnused()
        kept = len(self.uniques)
        newIds = np.fromiter(map(self._index, styles), dtype=np.intp, count=len(styles))[ids]
        self.ids.extend(newIds)
        if self._counts is not None:
            np.add.at(self._counts, newIds, 1)
        added = self.uniques[kept:]
        return len(added) != len(removed) or any(
            new != old for new, old in zip(added, removed))

    def _usage(self) -> np.ndarray:
        """
        Number of bars of each distinct style, computed on first use.

        Returns
        -------
        numpy.ndarray
            One count per entry of :attr:`uniques`, kept up to date afterwards.
        """
        if self._counts is None:
            self._counts = np.bincount(self.ids.view, minlength=len(self.uniques))
        return self._counts

    def _index(self, style) -> int:
        """
        Index of a style in :attr:`uniques`, which it is added to if new.

        Parameters
        ----------
        style : QPen or QBrush
            Style, compared by value.

        Returns
        -------
        int
            Index into :attr:`uniques`.
        """
        if self._buckets is None:
            self._buckets = {}
            for index, unique in enumerate(self.uniques):
                self._buckets.setdefault(self._key(unique), []).append(index)
        bucket = self._buckets.setdefault(self._key(style), [])
        for index in bucket:
            if self.uniques[index] == style:
                return index
        index = len(self.uniques)
        self.uniques.append(style)
        bucket.append(index)
        if self._counts is not None:
            self._counts = np.append(self._counts, 0)
        return index

    def _dropUnused(self) -> list:
        """
        Remove the last distinct styles while no bar uses them.

        Returns
        -------
        list
            The removed styles, in index order.
        """
        removed = []
        while self.uniques and self._counts[len(self.uniques) - 1] == 0:
            style = self.uniques.pop()
            if self._buckets is not None:
                self._buckets[self._key(style)].remove(len(self.uniques))
            removed.append(style)
        self._counts = self._counts[:len(self.uniques)]
        return removed[::-1]


class _Columns:
    """
    Bars aggregated per device pixel column, kept between paints.

    Parameters
    ----------
    first : numpy.ndarray
        Index of the first bar of each column.
    xlo, ylo, xhi, yhi : numpy.ndarray
        Rectangle drawn for each column.
    style : numpy.ndarray or None
        Style index of each column, -1 for none; ``None`` with a single style.

    Attributes
    ----------
    key : tuple or None
        ``(start, scale, offset, withPen)`` of the aggregation.
    stop : int
        Index after the last aggregated bar.
    unchanged : int
        Bars before this index are unchanged since the aggregation.
    """

    __slots__ = ('first', 'key', 'stop', 'style', 'unchanged', 'xhi', 'xlo', 'yhi', 'ylo')

    def __init__(self, first: np.ndarray, xlo: np.ndarray, ylo: np.ndarray,
                 xhi: np.ndarray, yhi: np.ndarray, style: np.ndarray | None) -> None:
        self.first = first
        self.xlo = xlo
        self.ylo = ylo
        self.xhi = xhi
        self.yhi = yhi
        self.style = style
        self.key = None
        self.stop = 0
        self.unchanged = 0

    def _arrays(self) -> tuple[np.ndarray, ...]:
        """
        Per-column arrays.

        Returns
        -------
        tuple of numpy.ndarray
            ``first``, ``xlo``, ``ylo``, ``xhi``, ``yhi`` and ``style`` if any.
        """
        arrays = (self.first, self.xlo, self.ylo, self.xhi, self.yhi)
        return arrays if self.style is None else (*arrays, self.style)

    def head(self, count: int) -> '_Columns':
        """
        The first columns.

        Parameters
        ----------
        count : int
            Number of columns kept.

        Returns
        -------
        _Columns
            Columns ``0`` to ``count - 1``.
        """
        arrays = [array[:count] for array in self._arrays()]
        if self.style is None:
            arrays.append(None)
        return _Columns(*arrays)

    def then(self, other: '_Columns') -> '_Columns':
        """
        These columns followed by other columns.

        Parameters
        ----------
        other : _Columns
            Columns of the following bars, with a style if these have one.

        Returns
        -------
        _Columns
            All columns.
        """
        arrays = [np.concatenate(pair) for pair in zip(self._arrays(), other._arrays())]
        if self.style is None:
            arrays.append(None)
        return _Columns(*arrays)


class BarGraphItem(GraphicsObject):
    def __init__(self, **opts) -> None:
        """
        Bar graph, typically volume bars or a histogram.

        Valid keyword options are:
        x, x0, x1, y, y0, y1, width, height, pen, brush, pens, brushes, name,
        lodPen, lod

        x specifies the x-position of the center of the bar.
        x0, x1 specify left and right edges of the bar, respectively.
        width specifies distance from x0 to x1.
        You may specify any combination:

            x, width
            x0, width
            x1, width
            x0, x1

        Likewise y, y0, y1, and height.
        If only height is specified, then y0 will be set to 0

        Example uses:

            BarGraphItem(x=range(5), height=[1,5,2,4,3], width=0.5)

        Only the bars within the visible x range are drawn when the bars are sorted
        by x. When the view is not zoomed out, the rendering is identical to drawing
        every bar with its own pen and brush. While streaming, :meth:`appendData`
        adds bars (or replaces the last one) at a cost independent of the number of
        bars.

        Parameters
        ----------
        **opts
            x, x0, x1, width : array_like or float
                Horizontal position and extent of the bars, see above.
            y, y0, y1, height : array_like or float
                Vertical position and extent of the bars, see above.
            pen : QPen or color, optional
                Outline of all bars. Defaults to the ``foreground`` config option.
            brush : QBrush or color, optional
                Fill of all bars. Defaults to ``(128, 128, 128)``.
            pens, brushes : sequence, optional
                One pen or brush (or anything accepted by :func:`~pyqtgraph.mkPen`
                or :func:`~pyqtgraph.mkBrush`) per bar; they take precedence over
                ``pen`` and ``brush``.
            name : str, optional
                Name of the item, e.g. shown by a legend.
            lodPen : bool, default True
                Level of detail of the outlines: while the mean bar width is below
                2 device pixels, bars are drawn without outline (the outline would
                cover most of each bar), and at least one device pixel wide.
            lod : bool, default True
                Level of detail of the bars: when the bars are sorted by x and there
                is more than one visible bar per device pixel column, one rectangle
                is drawn per pixel column, spanning the minimum y0 to the maximum y1
                of the bars of the column. With several pens or brushes, each column
                takes the style of its tallest bar.
        """
        GraphicsObject.__init__(self)
        self.opts = dict(
            x=None,
            y=None,
            x0=None,
            y0=None,
            x1=None,
            y1=None,
            name=None,
            height=None,
            width=None,
            pen=None,
            brush=None,
            pens=None,
            brushes=None,
            lodPen=True,
            lod=True,
        )

        if 'pen' not in opts:
            opts['pen'] = getConfigOption('foreground')
        if 'brush' not in opts:
            opts['brush'] = (128, 128, 128)
        # the first call to _updateColors() will thus always be an update

        self._rectarray = _PrimitiveBuffer(QtCore.QRectF)
        # rectangles drawn instead of the bars at reduced level of detail
        self._lodRects = Qt.internals.PrimitiveArray(QtCore.QRectF, 4)
        # level of detail cache: key of the last computation and its result, valid
        # while the bars before _lodUnchanged are unchanged
        self._lodKey = None
        self._lodValue = None
        self._lodUnchanged = 0
        # bars aggregated per pixel column, updated incrementally (_Columns or None)
        self._columns = None
        self._noPen = QtGui.QPen(QtCore.Qt.PenStyle.NoPen)
        self._pens = _StyleTable(QtGui.QPen, fn.mkPen, _penKey)
        self._brushes = _StyleTable(QtGui.QBrush, fn.mkBrush, _brushKey)
        # style index of each bar with per-bar pens and brushes, built on demand, and
        # the number of distinct brushes it was built with
        self._comboIds = None
        self._comboBrushes = 0
        # growth buffers of the options extended by appendData, by option name
        self._optArrays = {}
        self._optLists = {}
        # summaries of all bars, and of all bars but the last one
        self._stats = _BarStats()
        self._headStats = _BarStats()
        # left edges and running maximum of the right edges, when sorted by x
        self._x0Store = None
        self._x1Store = None
        self._shape = None
        self.picture = None
        self.setOpts(**opts)

    def setOpts(self, **opts) -> None:
        """
        Update the options of the item; see :meth:`__init__` for the valid keys.

        Parameters
        ----------
        **opts
            Options to change. Options not given keep their current value.
        """
        self.opts.update(opts)
        for key in opts:
            # the option holds the given value again, which appendData copies first
            self._optArrays.pop(key, None)
            self._optLists.pop(key, None)
        self.picture = None
        self._shape = None
        self._lodKey = None
        self._columns = None
        self._comboIds = None
        self._prepareData()
        self._updateColors(opts)
        self.prepareGeometryChange()
        self.update()
        self.informViewBoundsChanged()

    def appendData(self, *, replaceLast: bool = False, **opts) -> None:
        """
        Append bars, e.g. while streaming, or replace the last bar.

        Only the new bars are processed: their rectangles, the bounds, whether the
        bars stay sorted by x, their styles, and the last pixel column of the level
        of detail cached for the current view. The cost is amortised O(1) per new
        bar, and the rendering is identical to setting all bars at once with
        :meth:`setOpts`.

        The new bars are described by the options describing the existing bars.
        An option holding one value per bar (e.g. ``x`` or ``height``) needs the
        values of the new bars; an option holding a single value (e.g. a float
        ``width``) applies to the new bars unless given. Afterwards, :attr:`opts`
        describes all bars, and so do :meth:`getData`, :meth:`getOriginalDataset`
        and the CSV export: the coordinate options extended here become read-only
        float64 views of growth buffers, and ``pens``/``brushes`` lists owned by the
        item.

        Parameters
        ----------
        replaceLast : bool, default False
            If True, the first new bar replaces the last bar instead of following it,
            e.g. while the volume of the current candle grows. The item must have at
            least one bar.
        **opts
            x, x0, x1, width, y, y0, y1, height : array_like or float
                Geometry of the new bars, see :meth:`__init__`: one value per new
                bar, or a float for all new bars. Only the options describing the
                existing bars are accepted.
            pens, brushes : sequence
                One pen or brush per new bar. Required when the bars have per-bar
                ``pens`` or ``brushes``, not accepted when they share ``pen`` or
                ``brush`` (use :meth:`setOpts` to change the styles).

        Raises
        ------
        TypeError
            If an option is not per-bar data, e.g. ``pen`` or ``name``.
        ValueError
            If an option is missing, does not describe the existing bars, or has a
            length different from the other options. The item is left unchanged.
        IndexError
            If the item has fewer per-bar pens or brushes than bars.

        Examples
        --------
        Volume bars of streamed candles, the last one growing while its candle is open:

        >>> bars = pg.BarGraphItem(x=[], height=[], width=0.8, brushes=[])
        >>> bars.appendData(x=[0.0], height=[120.0], brushes=['g'])
        >>> bars.appendData(x=[0.0], height=[180.0], brushes=['r'], replaceLast=True)
        >>> bars.appendData(x=[1.0], height=[40.0], brushes=['g'])
        >>> bars.getData()[1]
        array([180.,  40.])
        """
        unknown = sorted(set(opts) - set(_COORDINATE_KEYS) - set(_STYLE_KEYS))
        if unknown:
            raise TypeError(f"appendData() got options that are not per-bar data: "
                            f"{', '.join(unknown)}; use setOpts() instead")
        nbefore = len(self._rectarray)
        if replaceLast and nbefore == 0:
            raise ValueError('replaceLast=True needs an existing bar to replace')
        count, edges, values = self._newBars(opts)
        if count == 0:
            if replaceLast:
                raise ValueError('replaceLast=True needs a new bar')
            return
        styles = self._newStyles(opts, count, nbefore)

        # the options are valid: update the state
        self._appendGeometry(edges, replaceLast)
        self._appendOptions(values, nbefore, replaceLast)
        self._appendStyles(styles, count, nbefore, replaceLast)
        first = nbefore - 1 if replaceLast else nbefore  # first changed bar
        self._lodUnchanged = min(self._lodUnchanged, first)
        if self._columns is not None:
            self._columns.unchanged = min(self._columns.unchanged, first)
        self.picture = None
        self.prepareGeometryChange()
        self.update()
        self.informViewBoundsChanged()

    def _newBars(self, opts: dict) -> tuple[int, tuple[np.ndarray, ...], dict]:
        """
        Validate and normalize the coordinate options given to :meth:`appendData`.

        Parameters
        ----------
        opts : dict
            Options given to :meth:`appendData`.

        Returns
        -------
        count : int
            Number of new bars.
        edges : tuple of numpy.ndarray
            ``x0``, ``y0``, ``x1``, ``y1`` of the new bars, ``count`` values each.
        values : dict
            Given coordinate options, as float64 arrays of ``count`` values.
        """
        given = {}
        for key in _COORDINATE_KEYS:
            if key not in opts:
                continue
            if self.opts[key] is None:
                raise ValueError(f"'{key}' does not describe the existing bars; use "
                                 f"setOpts() to change how the bars are described")
            value = np.asarray(opts[key], dtype=np.float64)
            if value.ndim > 1:
                raise ValueError(f"'{key}' must be a float or a one-dimensional array")
            given[key] = value
        if not given:
            raise ValueError('appendData() needs the coordinates of the new bars')
        layout = {}
        for key in _COORDINATE_KEYS:
            current = self.opts[key]
            if key in given:
                layout[key] = given[key]
            elif current is not None:
                if np.ndim(current) > 0:
                    raise ValueError(f"'{key}' holds one value per bar: appendData() "
                                     f"needs the values of the new bars")
                layout[key] = current
        try:
            shape = np.broadcast_shapes(*(value.shape for value in given.values()))
        except ValueError:
            raise ValueError('the options of the new bars have different lengths') from None
        count = shape[0] if shape else 1
        edges = tuple(np.broadcast_to(edge, (count,)) for edge in _normalizedCoords(layout))
        values = {key: np.broadcast_to(value, (count,)) for key, value in given.items()}
        return count, edges, values

    def _newStyles(self, opts: dict, count: int, nbefore: int) -> dict:
        """
        Validate and convert the per-bar styles given to :meth:`appendData`.

        Parameters
        ----------
        opts : dict
            Options given to :meth:`appendData`.
        count : int
            Number of new bars.
        nbefore : int
            Number of bars before the call.

        Returns
        -------
        dict
            By option name (``'pens'``, ``'brushes'``), for the per-bar styles: the
            distinct styles of the new bars, the index of the style of each new bar
            into them, and the given specifications.
        """
        styles = {}
        for key, single in _STYLE_KEYS.items():
            table = self._table(key)
            if not table.perBar:
                if key in opts:
                    raise ValueError(f"the bars share one {single}: '{key}' cannot be "
                                     f"appended, use setOpts({key}=...) first")
                continue
            if key not in opts:
                raise ValueError(f"the bars have one {single} each: appendData() needs "
                                 f"'{key}' for the new bars")
            specs = list(opts[key])
            if len(specs) != count:
                raise ValueError(f'{len(specs)} {key} given for {count} new bars')
            if len(table.ids) < nbefore:
                raise IndexError(
                    f"BarGraphItem: {len(table.ids)} {key} given for {nbefore} bars")
            styles[key] = (*table.convert(specs), specs)
        return styles

    def _table(self, key: str) -> _StyleTable:
        """
        Style table of an option.

        Parameters
        ----------
        key : str
            ``'pens'`` or ``'brushes'``.

        Returns
        -------
        _StyleTable
            The pens or the brushes of the bars.
        """
        return self._pens if key == 'pens' else self._brushes

    def _appendGeometry(self, edges: tuple[np.ndarray, ...], replaceLast: bool) -> None:
        """
        Append the rectangles of new bars, and update the summary and the x index.

        Parameters
        ----------
        edges : tuple of numpy.ndarray
            ``x0``, ``y0``, ``x1``, ``y1`` of the new bars.
        replaceLast : bool
            Whether the first new bar replaces the last bar.
        """
        x0, y0, x1, y1 = edges
        count = len(x0)
        if replaceLast:
            self._rectarray.pop()
            if self._x0Store is not None:
                self._x0Store.pop()
                self._x1Store.pop()
            before = self._headStats
            self._shape = None
        else:
            before = self._stats
        nbefore = len(self._rectarray)
        memory = self._rectarray.reserve(count)
        memory[:, 0] = x0
        memory[:, 1] = y0
        memory[:, 2] = x1 - x0
        memory[:, 3] = y1 - y0
        left = memory[:, 0].copy()
        widths = memory[:, 2]
        right = left + widths
        arrays = (x0, y0, x1, y1, left, widths, right)
        head = before.then(_BarStats.of(*(a[:-1] for a in arrays)))
        self._setStats(head, head.then(_BarStats.of(*(a[-1:] for a in arrays))))
        self._extendIndex(nbefore, left, right)
        if self._shape is not None:
            for rect in memory.tolist():
                self._shape.addRect(QtCore.QRectF(*rect))

    def _appendOptions(self, values: dict, nbefore: int, replaceLast: bool) -> None:
        """
        Extend the coordinate options given to :meth:`appendData`.

        Parameters
        ----------
        values : dict
            Given options, one float64 value per new bar.
        nbefore : int
            Number of bars before the call.
        replaceLast : bool
            Whether the first new bar replaces the last bar.
        """
        for key, new in values.items():
            store = self._optArrays.get(key)
            if store is None:
                old = self.opts[key]
                if np.ndim(old) == 0 and bool((new == old).all()):
                    continue  # still a single value for all bars
                store = _GrowableArray()
                store.extend(np.broadcast_to(np.asarray(old, dtype=np.float64), (nbefore,)))
                self._optArrays[key] = store
            if replaceLast:
                store.pop()
            store.extend(new)
            view = store.view
            view.flags.writeable = False
            self.opts[key] = view

    def _appendStyles(self, styles: dict, count: int, nbefore: int,
                      replaceLast: bool) -> None:
        """
        Append the per-bar styles given to :meth:`appendData`.

        Parameters
        ----------
        styles : dict
            As returned by :meth:`_newStyles`.
        count : int
            Number of new bars.
        nbefore : int
            Number of bars before the call.
        replaceLast : bool
            Whether the first new bar replaces the last bar.
        """
        nbrushes = len(self._brushes.uniques)
        for key, (uniques, ids, specs) in styles.items():
            if self._table(key).append(uniques, ids, nbefore, replaceLast) and key == 'pens':
                self._updatePenStats()
            specList = self._optLists.get(key)
            if specList is None:
                specList = list(self.opts[key])[:nbefore]
                self._optLists[key] = specList
                self.opts[key] = specList
            if replaceLast:
                specList.pop()
            specList.extend(specs)
        # the style indices of the existing bars are unchanged, except those combining
        # per-bar pens and brushes when the number of distinct brushes changed
        combo = self._comboIds
        if len(self._brushes.uniques) != nbrushes and self._pens.perBar \
                and self._brushes.perBar:
            self._comboIds = None
            self._lodKey = None
            self._columns = None
        elif combo is not None:
            if len(combo) != nbefore:
                self._comboIds = None
            else:
                if replaceLast:
                    combo.pop()
                pens, brushes = self._pens.ids.view, self._brushes.ids.view
                combo.extend(pens[-count:] * self._comboBrushes + brushes[-count:])

    def _updatePenWidth(self, pen):
        no_pen = pen is None or pen.style() == QtCore.Qt.PenStyle.NoPen
        if no_pen:
            return

        idx = pen.isCosmetic()
        self._penWidth[idx] = max(self._penWidth[idx], pen.widthF())

    def _updatePenStats(self) -> None:
        """Widest pens (``_penWidth``) and whether any bar has an outline (``_hasPen``)."""
        self._penWidth = [0, 0]
        for pen in self._pens.uniques:
            self._updatePenWidth(pen)
        self._hasPen = any(
            pen.style() != QtCore.Qt.PenStyle.NoPen for pen in self._pens.uniques)

    def _updateColors(self, opts: dict) -> None:
        """
        Rebuild the pens and brushes from the options, if ``opts`` changed them.

        Per-bar ``pens`` and ``brushes`` are reduced to their distinct values, plus one
        style index per bar.

        Parameters
        ----------
        opts : dict
            Options passed to the current :meth:`setOpts` call.
        """
        # the logic here is to permit the user to update only data
        # without updating pens/brushes

        # update only if fresh pen/pens supplied
        if 'pen' in opts or 'pens' in opts:
            if self.opts['pens'] is None:
                # pens not configured, use single pen
                self._pens.setShared(self.opts['pen'])
                self._sharedPen = self._pens.uniques[0]
            else:
                # pens configured, ignore single pen (if any)
                self._pens.setPerBar(self.opts['pens'])
                self._sharedPen = None
            self._updatePenStats()

        # update only if fresh brush/brushes supplied
        if 'brush' in opts or 'brushes' in opts:
            if self.opts['brushes'] is None:
                # brushes not configured, use single brush
                self._brushes.setShared(self.opts['brush'])
                self._sharedBrush = self._brushes.uniques[0]
            else:
                # brushes configured, ignore single brush (if any)
                self._brushes.setPerBar(self.opts['brushes'])
                self._sharedBrush = None

        self._singleColor = (
            self._sharedPen is not None and
            self._sharedBrush is not None
        )

    def _getNormalizedCoords(self) -> tuple[np.ndarray, ...]:
        """
        Edges of the bars described by the options.

        Returns
        -------
        x0, y0, x1, y1 : numpy.ndarray
            See :func:`_normalizedCoords`.
        """
        return _normalizedCoords(self.opts)

    def _prepareData(self) -> None:
        """Compute the bar rectangles, their summary and the x index used for culling."""
        x0, y0, x1, y1 = self._getNormalizedCoords()
        if x0.size == 0 or y0.size == 0:
            self._rectarray.resize(0)
            self._setStats(_BarStats(), _BarStats())
            self._setIndex(None, None)
            return

        count = max(x0.size, y0.size)
        self._rectarray.resize(count)
        memory = self._rectarray.ndarray()
        memory[:, 0] = x0
        memory[:, 1] = y0
        memory[:, 2] = x1 - x0
        memory[:, 3] = y1 - y0
        # a copy, never a view of the rectangles (np.ascontiguousarray returns one
        # bar as a view), which appending may reallocate
        left = memory[:, 0].copy()
        widths = memory[:, 2]
        # x1 is finite only if both x0 and the width are
        right = left + widths
        arrays = [np.broadcast_to(edge, (count,)) for edge in (x0, y0, x1, y1)]
        arrays += [left, widths, right]
        head = _BarStats.of(*(a[:-1] for a in arrays))
        self._setStats(head, head.then(_BarStats.of(*(a[-1:] for a in arrays))))
        if not self._xSorted:
            self._setIndex(None, None)
        elif self._x1Sorted:
            self._setIndex(left, right)
        else:
            # right edges are searched for the first visible bar: make them sorted
            self._setIndex(left, np.maximum.accumulate(right))

    def _setStats(self, head: _BarStats, stats: _BarStats) -> None:
        """
        Store the summary of the bars and the attributes derived from it.

        Parameters
        ----------
        head : _BarStats
            Summary of all bars but the last one.
        stats : _BarStats
            Summary of all bars.
        """
        self._headStats = head
        self._stats = stats
        # None (NaN in the summary) when no bar has finite edges, e.g. without bars
        self._dataBounds = tuple(
            tuple(None if math.isnan(bound) else bound for bound in pair)
            for pair in stats.bounds)
        self._meanWidth = stats.widthSum / stats.widthCount if stats.widthCount else math.nan
        self._xSorted = bool(stats.count) and stats.finite and stats.x0Sorted
        # whether the right edges are sorted too, i.e. self._x1 holds the right edges
        # themselves rather than their running maximum
        self._x1Sorted = self._xSorted and stats.x1Sorted

    def _setIndex(self, left: np.ndarray | None, right: np.ndarray | None) -> None:
        """
        Keep the searchable edge arrays of bars sorted by x.

        Parameters
        ----------
        left : numpy.ndarray or None
            Left edges, handed over; ``None`` when the bars are not sorted by x.
        right : numpy.ndarray or None
            Running maximum of the right edges, handed over.
        """
        if left is None:
            self._x0Store = self._x1Store = None
            self._x0 = self._x1 = None
        else:
            self._x0Store = _GrowableArray.wrap(left)
            self._x1Store = _GrowableArray.wrap(right)
            self._x0, self._x1 = left, right

    def _extendIndex(self, nbefore: int, left: np.ndarray, right: np.ndarray) -> None:
        """
        Extend the searchable edge arrays with new bars.

        Parameters
        ----------
        nbefore : int
            Number of bars before the new ones.
        left, right : numpy.ndarray
            Left and right edges of the new bars.
        """
        if not self._xSorted:
            self._setIndex(None, None)
            return
        x0, x1 = self._x0Store, self._x1Store
        if x0 is None or len(x0) != nbefore:
            # sorted only now: index all bars
            memory = self._rectarray.ndarray()
            left = memory[:, 0].copy()
            self._setIndex(left, np.maximum.accumulate(left + memory[:, 2]))
            return
        right = np.maximum.accumulate(right)
        if len(x1):
            right = np.maximum(right, x1.view[-1])
        x0.extend(left)
        x1.extend(right)
        self._x0, self._x1 = x0.view, x1.view

    def _styleIds(self, withPen: bool) -> np.ndarray | None:
        """
        Style index of each bar, decoded by :meth:`_styleOf`.

        Parameters
        ----------
        withPen : bool
            Whether the pens are part of the style. When ``False``, the bars are
            drawn without outline and only their brushes are distinguished.

        Returns
        -------
        numpy.ndarray or None
            One index per bar, or ``None`` when all bars share the same style.
        """
        nbars = len(self._rectarray)
        pens = self._pens if withPen and self._pens.perBar else None
        brushes = self._brushes if self._brushes.perBar else None
        for name, table in (('pens', pens), ('brushes', brushes)):
            if table is not None and len(table.ids) < nbars:
                raise IndexError(
                    f"BarGraphItem: {len(table.ids)} {name} given for {nbars} bars")
        if brushes is None:
            # with a single brush, the style index is the pen index
            return None if pens is None else pens.ids.view[:nbars]
        if pens is None:
            return brushes.ids.view[:nbars]
        nbrushes = len(brushes.uniques)
        combo = self._comboIds
        if combo is None or len(combo) != nbars or self._comboBrushes != nbrushes:
            combo = _GrowableArray.wrap(
                pens.ids.view[:nbars] * nbrushes + brushes.ids.view[:nbars])
            self._comboIds, self._comboBrushes = combo, nbrushes
        return combo.view

    def _styleOf(self, sid: int, withPen: bool) -> tuple[QtGui.QPen, QtGui.QBrush]:
        """
        Pen and brush of a style index returned by :meth:`_styleIds`.

        Parameters
        ----------
        sid : int
            Style index; ``0`` when all bars share the same style.
        withPen : bool
            Same value as given to :meth:`_styleIds`; ``False`` yields no pen.

        Returns
        -------
        pen : QtGui.QPen
            Outline pen.
        brush : QtGui.QBrush
            Fill brush.
        """
        brushes = self._brushes.uniques
        nbrushes = len(brushes)
        pen = self._pens.uniques[sid // nbrushes] if withPen else self._noPen
        return pen, brushes[sid % nbrushes]

    def _drawGroups(self, painter: QtGui.QPainter, rects,
                    groups: list[tuple[int, int, int]], withPen: bool) -> None:
        """
        Draw ranges of rectangles, one ``drawRects`` call per range.

        Parameters
        ----------
        painter : QtGui.QPainter
            Destination painter.
        rects : PrimitiveArray or _PrimitiveBuffer
            Rectangles to draw from.
        groups : list of tuple of int
            ``(style index, begin, end)``: rectangles ``begin`` to ``end - 1`` are
            drawn with that style, see :meth:`_styleOf`.
        withPen : bool
            Whether outlines are drawn.
        """
        for sid, begin, end in groups:
            pen, brush = self._styleOf(sid, withPen)
            painter.setPen(pen)
            painter.setBrush(brush)
            painter.drawRects(*rects.drawargs(begin, end))

    def _drawInOrder(self, painter: QtGui.QPainter, start: int, stop: int) -> None:
        """
        Draw bars in order, each with its own pen and brush.

        Consecutive bars of the same style are drawn by a single ``drawRects`` call,
        which renders exactly like drawing them one by one.

        Parameters
        ----------
        painter : QtGui.QPainter
            Destination painter.
        start, stop : int
            Range of the bars to draw.
        """
        ids = self._styleIds(True)
        if ids is None:
            groups = [(0, start, stop)]
        else:
            groups = [(sid, start + b, start + e) for sid, b, e in _runs(ids[start:stop])]
        self._drawGroups(painter, self._rectarray, groups, True)

    def _render(self, painter: QtGui.QPainter) -> None:
        """
        Draw all bars, in order, each with its own pen and brush.

        Parameters
        ----------
        painter : QtGui.QPainter
            Destination painter.
        """
        self._drawInOrder(painter, 0, len(self._rectarray))

    def drawPicture(self) -> None:
        """
        Record all bars into ``self.picture``, a ``QPicture``.

        :meth:`paint` does not use it any more; kept for backward compatibility.
        """
        self.picture = QtGui.QPicture()
        painter = QtGui.QPainter(self.picture)
        self._render(painter)
        painter.end()

    def _visibleSlice(self, painter: QtGui.QPainter,
                      pxPerUnit: float) -> tuple[int, int, float, float]:
        """
        Range of the bars that can be visible, for bars sorted by x.

        Parameters
        ----------
        painter : QtGui.QPainter
            Active painter.
        pxPerUnit : float
            Device pixels per unit of x.

        Returns
        -------
        start, stop : int
            Bars ``start`` to ``stop - 1`` intersect the visible x range, extended by
            the pen width. All bars when they are not sorted by x.
        xmin, xmax : float
            Visible x range; infinite when unknown.
        """
        nbars = len(self._rectarray)
        rect = _visibleRect(self, painter) if self._xSorted and pxPerUnit > 0 else None
        if rect is None:
            return 0, nbars, -math.inf, math.inf
        xmin, xmax = rect.left(), rect.right()
        # outlines extend beyond the bars by half the pen width, plus antialiasing
        pad = (0.5 * self._penWidth[0]
               + (0.5 * (self._penWidth[1] or 1) + 1.0) / pxPerUnit)
        start = int(np.searchsorted(self._x1, xmin - pad, side='left'))
        stop = int(np.searchsorted(self._x0, xmax + pad, side='right'))
        return start, stop, xmin, xmax

    def paint(self, p: QtGui.QPainter, *args) -> None:
        """
        Draw the bars.

        Only bars within the visible x range are drawn when the bars are sorted by x.
        When zoomed out, the ``lodPen`` and ``lod`` options reduce the level of detail.

        Parameters
        ----------
        p : QtGui.QPainter
            Destination painter, mapping item coordinates to the device.
        *args
            ``QStyleOptionGraphicsItem`` and widget, unused.
        """
        nbars = len(self._rectarray)
        if nbars == 0:
            return
        tr = p.combinedTransform()
        pxPerUnit = math.hypot(tr.m11(), tr.m12())  # device pixels per unit of x
        start, stop, xmin, xmax = self._visibleSlice(p, pxPerUnit)
        if stop <= start:
            return

        withPen = not (
            self.opts['lodPen']
            and self._hasPen
            and pxPerUnit * self._meanWidth < _LOD_PEN_MIN_WIDTH_PX
        )
        if self.opts['lod'] and self._xSorted and pxPerUnit > 0:
            axisAligned = tr.isAffine() and tr.m12() == 0.0 and tr.m21() == 0.0
            span = min(self._x1[stop - 1], xmax) - max(self._x0[start], xmin)
            barsPerPixel = (stop - start) / max(span * pxPerUnit, 1.0)
            if axisAligned and barsPerPixel > _LOD_MAX_BARS_PER_PIXEL:
                key = ('aggregate', start, stop, tr.m11(), tr.dx(), withPen)
                if self._lodKey != key or stop > self._lodUnchanged:
                    groups = self._aggregate(start, stop, tr.m11(), tr.dx(), withPen)
                    self._setLodValue(key, (self._lodRects, groups))
                self._drawGroups(p, *self._lodValue, withPen)
                return

        if withPen:
            self._drawInOrder(p, start, stop)
            return
        key = ('withoutPen', start, stop, pxPerUnit)
        if self._lodKey != key or stop > self._lodUnchanged:
            self._setLodValue(key, self._withoutPen(start, stop, pxPerUnit))
        self._drawGroups(p, *self._lodValue, False)

    def _setLodValue(self, key: tuple, value: tuple) -> None:
        """
        Cache what is drawn at reduced level of detail.

        Parameters
        ----------
        key : tuple
            Mode, range of bars and transform the value was computed for.
        value : tuple
            Rectangles and groups to pass to :meth:`_drawGroups`.
        """
        self._lodKey = key
        self._lodValue = value
        self._lodUnchanged = len(self._rectarray)

    def _withoutPen(self, start: int, stop: int, pxPerUnit: float) -> tuple:
        """
        Bars of a range to draw without outline, grouped by brush.

        Bars narrower than a device pixel are widened to one pixel, as they could
        otherwise vanish without their outline.

        Parameters
        ----------
        start, stop : int
            Range of bars.
        pxPerUnit : float
            Device pixels per unit of x.

        Returns
        -------
        rects : PrimitiveArray or _PrimitiveBuffer
            The bar rectangles, or a reordered and widened copy of the range.
        groups : list of tuple of int
            ``(style index, begin, end)`` ranges of ``rects`` to draw.
        """
        memory = self._rectarray.ndarray()[start:stop]
        minWidth = 1.0 / pxPerUnit if pxPerUnit > 0 else 0.0
        thin = memory[:, 2] < minWidth
        ids = self._styleIds(False)
        if ids is None:
            if not thin.any():
                return self._rectarray, [(0, start, stop)]
            order = slice(None)
            groups = [(0, 0, stop - start)]
        else:
            ids = ids[start:stop]
            # a stable sort of small integers is a linear-time radix sort
            keys = ids.astype(np.uint16) if len(self._brushes.uniques) <= 65536 else ids
            order = np.argsort(keys, kind='stable')
            groups = _runs(ids[order])
            thin = thin[order]
        self._lodRects.resize(stop - start)
        out = self._lodRects.ndarray()
        out[:] = memory[order]
        out[thin, 0] += 0.5 * (out[thin, 2] - minWidth)
        out[thin, 2] = minWidth
        return self._lodRects, groups

    def _aggregate(self, start: int, stop: int, scale: float, offset: float,
                   withPen: bool) -> list[tuple[int, int, int]]:
        """
        Aggregate sorted bars per device pixel column into ``self._lodRects``.

        Each column rectangle spans the column (at least) and the bars whose left edge
        falls in it, from their minimum y0 to their maximum y1. The columns are kept
        in ``self._columns``: when only the bars after some index changed (e.g. bars
        were appended), the columns before the one holding that index are reused.

        Parameters
        ----------
        start, stop : int
            Range of the visible bars, sorted by x.
        scale, offset : float
            Device x coordinate is ``scale * x + offset``.
        withPen : bool
            Whether pens are part of the style of the bars.

        Returns
        -------
        list of tuple of int
            ``(style index, begin, end)``: rectangles ``begin`` to ``end - 1`` of
            ``self._lodRects`` use that style. With several styles, a column takes the
            style of its tallest bar.
        """
        ids = self._styleIds(withPen)
        key = (start, scale, offset, withPen)
        cached = self._columns
        kept = None
        begin = start
        if cached is not None and cached.key == key:
            unchanged = min(cached.stop, cached.unchanged, stop)
            if unchanged > start:
                # the column of the last unchanged bar may gain bars: compute it again
                index = int(np.searchsorted(cached.first, unchanged - 1, side='right')) - 1
                if index > 0:
                    kept = cached.head(index)
                    begin = int(cached.first[index])
        columns = self._columnBlock(begin, stop, scale, offset, ids)
        if kept is not None:
            columns = kept.then(columns)
        columns.key, columns.stop, columns.unchanged = key, stop, len(self._rectarray)
        self._columns = columns

        valid = np.isfinite(columns.ylo) & np.isfinite(columns.yhi)
        if columns.style is None:
            order = np.flatnonzero(valid)
            groups = [(0, 0, len(order))]
        else:
            selected = np.flatnonzero(valid & (columns.style >= 0))
            order = selected[np.argsort(columns.style[selected], kind='stable')]
            groups = _runs(columns.style[order])

        self._lodRects.resize(len(order))
        out = self._lodRects.ndarray()
        out[:, 0] = columns.xlo[order]
        out[:, 1] = columns.ylo[order]
        out[:, 2] = columns.xhi[order] - out[:, 0]
        out[:, 3] = columns.yhi[order] - out[:, 1]
        return groups

    def _columnBlock(self, begin: int, stop: int, scale: float, offset: float,
                     ids: np.ndarray | None) -> _Columns:
        """
        Aggregate a range of sorted bars per device pixel column.

        Parameters
        ----------
        begin, stop : int
            Range of bars, ``begin`` being the first bar of a column.
        scale, offset : float
            Device x coordinate is ``scale * x + offset``.
        ids : numpy.ndarray or None
            Style index of each bar, ``None`` with a single style.

        Returns
        -------
        _Columns
            One rectangle per column; with several styles, the style of the tallest
            bar of each column.
        """
        memory = self._rectarray.ndarray()[begin:stop]
        x0 = self._x0[begin:stop]
        column = np.floor(x0 * scale + offset)
        first = np.concatenate(([0], np.flatnonzero(column[1:] != column[:-1]) + 1))
        y0 = memory[:, 1]
        heights = memory[:, 3]
        ylo = np.fmin.reduceat(y0, first)
        yhi = np.fmax.reduceat(y0 + heights, first)
        xhi = np.fmax.reduceat(x0 + memory[:, 2], first)
        edgeA = (column[first] - offset) / scale
        edgeB = (column[first] + 1.0 - offset) / scale
        xlo = np.minimum(np.minimum(edgeA, edgeB), x0[first])
        xhi = np.maximum(np.maximum(edgeA, edgeB), xhi)

        columnStyle = None
        if ids is not None:
            # style of the tallest bar of each column
            counts = np.diff(np.append(first, len(x0)))
            segment = np.repeat(np.arange(len(first)), counts)
            tallest = np.flatnonzero(heights == np.fmax.reduceat(heights, first)[segment])
            keep = np.ones(len(tallest), dtype=bool)
            keep[1:] = segment[tallest[1:]] != segment[tallest[:-1]]
            tallest = tallest[keep]
            columnStyle = np.full(len(first), -1, dtype=np.intp)
            columnStyle[segment[tallest]] = ids[begin:stop][tallest]
        return _Columns(first + begin, xlo, ylo, xhi, yhi, columnStyle)

    def shape(self) -> QtGui.QPainterPath:
        """
        Outline of all bars, built on first use and extended by :meth:`appendData`.

        Returns
        -------
        QtGui.QPainterPath
            One rectangle per bar (a shallow copy of the cached path).
        """
        if self._shape is None:
            shape = QtGui.QPainterPath()
            for rect in self._rectarray.instances():
                shape.addRect(rect)
            self._shape = shape
        return QtGui.QPainterPath(self._shape)

    def implements(self, interface=None):
        ints = ['plotData']
        if interface is None:
            return ints
        return interface in ints

    def name(self):
        return self.opts.get('name', None)

    def getData(self) -> tuple:
        """
        Positions and heights of the bars, the ``x`` and ``height`` options.

        Returns
        -------
        x : array_like or float or None
            The ``x`` option. After :meth:`appendData`, a read-only view holding
            all bars, which the next ``appendData(replaceLast=True)`` may change.
        height : array_like or float or None
            The ``height`` option, likewise.
        """
        return self.opts.get('x'),  self.opts.get('height')

    def getOriginalDataset(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Positions and heights of the bars, one value per bar, e.g. for
        :class:`~pyqtgraph.exporters.CSVExporter`.

        These are the ``x`` and ``height`` options returned by :meth:`getData`, as
        float arrays with one value per bar (a scalar option is repeated). When the
        bars were not given by ``x`` (but by ``x0``, ``x1`` and ``width``) their
        centers are returned, and when they were not given by ``height`` (but by
        ``y1``) their signed heights ``y1 - y0``.

        Returns
        -------
        x : numpy.ndarray or None
            Center of each bar, or ``None`` without bars.
        height : numpy.ndarray or None
            Height of each bar, or ``None`` without bars.
        """
        nbars = len(self._rectarray)
        if nbars == 0:
            return None, None
        x, height = self.getData()
        if x is None:
            memory = self._rectarray.ndarray()
            x = memory[:, 0] + 0.5 * memory[:, 2]
        if height is None:
            # setOpts has checked that y1 is given; y0 defaults to 0 without height
            y0 = self.opts.get('y0')
            height = (np.asarray(self.opts['y1'], dtype=np.float64)
                      - (0.0 if y0 is None else np.asarray(y0, dtype=np.float64)))
        return tuple(
            np.broadcast_to(np.asarray(values, dtype=np.float64), (nbars,)).copy()
            for values in (x, height))

    def dataBounds(self, ax: int, frac: float = 1.0,
                   orthoRange: tuple[float, float] | None = None
                   ) -> tuple[float | None, float | None]:
        """
        Range of the bars along an axis, including half the width of non-cosmetic pens.

        Parameters
        ----------
        ax : int
            0 for x, 1 for y.
        frac : float, default 1.0
            Fraction of the bars to fit, in ``(0, 1]``. Below 1, the range spans from
            the ``50 * (1 - frac)`` percentile of the lower edges of the bars to the
            ``50 * (1 + frac)`` percentile of their upper edges, as the percentiles
            of the data of the other plot items, so that a few outliers (e.g. volume
            spikes) do not stretch an auto-range.
        orthoRange : tuple of float or None, default None
            Only the bars whose extent along the other axis intersects this range
            (bounds included) are considered. For ``ax=1``, these are the bars within
            the visible x range, which ``ViewBox.setAutoVisible(y=True)`` fits; when
            the bars are sorted by x they are found by binary search.

        Returns
        -------
        tuple of float or None
            ``(min, max)``, or ``(None, None)`` without data or when no bar with
            finite edges is considered.

        Notes
        -----
        All ranges ignore the bars with a non-finite coordinate (e.g. a NaN height),
        along both axes. The full range (``frac=1`` without ``orthoRange``) is
        computed once per data change, and updated incrementally by
        :meth:`appendData`.
        """
        if ax not in (0, 1):
            raise ValueError(f'ax must be 0 or 1, got {ax}')
        if frac >= 1.0 and orthoRange is None:
            # _dataBounds is available after _prepareData()
            bounds = self._dataBounds[ax]
        elif frac <= 0.0:
            raise ValueError(f'frac must be in (0, 1], got {frac}')
        else:
            bounds = self._partialBounds(ax, frac, orthoRange)
        if bounds[0] is None or bounds[1] is None:
            return None, None
        # _penWidth is available after _updateColors()
        pw = self._penWidth[0] * 0.5
        return (bounds[0] - pw, bounds[1] + pw)

    def _partialBounds(self, ax: int, frac: float,
                       orthoRange: tuple[float, float] | None
                       ) -> tuple[float | None, float | None]:
        """
        Range of some bars along an axis, without pen; see :meth:`dataBounds`.

        Parameters
        ----------
        ax : int
            0 for x, 1 for y.
        frac : float
            Fraction of the bars to fit, in ``(0, 1]``.
        orthoRange : tuple of float or None
            Range along the other axis that the bars must intersect; ``None`` for all
            bars.

        Returns
        -------
        tuple of float or None
            ``(min, max)``, or ``(None, None)`` when no bar with finite coordinates
            is considered.
        """
        memory = self._rectarray.ndarray()
        searched = orthoRange is not None and ax == 1 and self._xSorted
        if orthoRange is not None:
            lo, hi = sorted(float(value) for value in orthoRange)
        if searched:
            # first bar whose right edge reaches lo, first bar starting after hi
            start = int(np.searchsorted(self._x1, lo, side='left'))
            stop = int(np.searchsorted(self._x0, hi, side='right'))
            if frac >= 1.0 and self._x1Sorted and stop - start == len(memory):
                # all bars, e.g. a zoomed out view: reuse the full range
                ymin, ymax = self._dataBounds[1]
                if ymin is None or ymax is None:
                    return None, None
                return ymin, ymax
            memory = memory[start:stop]
        # bars with a non-finite coordinate are ignored, as by the full range
        finite = np.isfinite(memory).all(axis=1)
        if not finite.all():
            memory = memory[finite]
        if searched:
            if not self._x1Sorted:
                # self._x1 is the running maximum of the right edges: a bar within
                # the slice can still end before lo
                memory = memory[memory[:, 0] + memory[:, 2] >= lo]
        elif orthoRange is not None:
            other = 1 - ax
            low = memory[:, other]
            memory = memory[(low <= hi) & (low + memory[:, other + 2] >= lo)]
        if len(memory) == 0:
            return None, None
        low = memory[:, ax]
        high = low + memory[:, ax + 2]
        if frac >= 1.0:
            bmin, bmax = np.min(low), np.max(high)
        else:
            bmin = np.percentile(low, 50 * (1 - frac))
            bmax = np.percentile(high, 50 * (1 + frac))
        return float(bmin), float(bmax)

    def pixelPadding(self):
        # _penWidth is available after _updateColors()
        pw = (self._penWidth[1] or 1) * 0.5
        return pw

    def boundingRect(self) -> QtCore.QRectF:
        """
        Bounds of the bars, including their pens; bars with non-finite edges are
        ignored, see :meth:`dataBounds`.

        Returns
        -------
        QtCore.QRectF
            Rectangle in item coordinates; empty without bars with finite edges.
        """
        xmn, xmx = self.dataBounds(ax=0)
        if xmn is None or xmx is None:
            return QtCore.QRectF()
        ymn, ymx = self.dataBounds(ax=1)
        if ymn is None or ymx is None:
            return QtCore.QRectF()

        px = py = 0
        pxPad = self.pixelPadding()
        if pxPad > 0:
            # determine length of pixel in local x, y directions
            px, py = self.pixelVectors()
            px = 0 if px is None else px.length()
            py = 0 if py is None else py.length()
            # return bounds expanded by pixel size
            px *= pxPad
            py *= pxPad

        return QtCore.QRectF(xmn-px, ymn-py, (2*px)+xmx-xmn, (2*py)+ymx-ymn)
