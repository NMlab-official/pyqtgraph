import math
from collections.abc import Callable, Iterable

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
        every bar with its own pen and brush.

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

        self._rectarray = Qt.internals.PrimitiveArray(QtCore.QRectF, 4)
        # rectangles drawn instead of the bars at reduced level of detail
        self._lodRects = Qt.internals.PrimitiveArray(QtCore.QRectF, 4)
        # level of detail cache: key of the last computation and its result
        self._lodKey = None
        self._lodValue = None
        self._noPen = QtGui.QPen(QtCore.Qt.PenStyle.NoPen)
        self._styleIdCache = {}
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
        self.picture = None
        self._shape = None
        self._lodKey = None
        self._styleIdCache = {}
        self._prepareData()
        self._updateColors(opts)
        self.prepareGeometryChange()
        self.update()
        self.informViewBoundsChanged()

    def _updatePenWidth(self, pen):
        no_pen = pen is None or pen.style() == QtCore.Qt.PenStyle.NoPen
        if no_pen:
            return

        idx = pen.isCosmetic()
        self._penWidth[idx] = max(self._penWidth[idx], pen.widthF())

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
            self._penWidth = [0, 0]

            if self.opts['pens'] is None:
                # pens not configured, use single pen
                pen = fn.mkPen(self.opts['pen'])
                self._sharedPen = pen
                self._uniquePens = [pen]
                self._penIds = None
            else:
                # pens configured, ignore single pen (if any)
                self._uniquePens, self._penIds = _uniqueStyles(
                    self.opts['pens'], QtGui.QPen, fn.mkPen, _penKey)
                self._sharedPen = None
            for pen in self._uniquePens:
                self._updatePenWidth(pen)
            self._hasPen = any(
                pen.style() != QtCore.Qt.PenStyle.NoPen for pen in self._uniquePens)

        # update only if fresh brush/brushes supplied
        if 'brush' in opts or 'brushes' in opts:
            if self.opts['brushes'] is None:
                # brushes not configured, use single brush
                brush = fn.mkBrush(self.opts['brush'])
                self._sharedBrush = brush
                self._uniqueBrushes = [brush]
                self._brushIds = None
            else:
                # brushes configured, ignore single brush (if any)
                self._uniqueBrushes, self._brushIds = _uniqueStyles(
                    self.opts['brushes'], QtGui.QBrush, fn.mkBrush, _brushKey)
                self._sharedBrush = None

        self._singleColor = (
            self._sharedPen is not None and
            self._sharedBrush is not None
        )

    def _getNormalizedCoords(self):
        def asarray(x):
            if x is None or np.isscalar(x) or isinstance(x, np.ndarray):
                return x
            return np.array(x)

        x = asarray(self.opts.get('x'))
        x0 = asarray(self.opts.get('x0'))
        x1 = asarray(self.opts.get('x1'))
        width = asarray(self.opts.get('width'))
        
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
            
        y = asarray(self.opts.get('y'))
        y0 = asarray(self.opts.get('y0'))
        y1 = asarray(self.opts.get('y1'))
        height = asarray(self.opts.get('height'))

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

    def _prepareData(self) -> None:
        """Compute the bar rectangles, the data bounds and the x index used for culling."""
        x0, y0, x1, y1 = self._getNormalizedCoords()
        if x0.size == 0 or y0.size == 0:
            self._dataBounds = (None, None), (None, None)
            self._rectarray.resize(0)
            self._prepareIndex(self._rectarray.ndarray())
            return

        xmn, xmx = np.min(x0), np.max(x1)
        ymn, ymx = np.min(y0), np.max(y1)
        self._dataBounds = (xmn, xmx), (ymn, ymx)

        self._rectarray.resize(max(x0.size, y0.size))
        memory = self._rectarray.ndarray()
        memory[:, 0] = x0
        memory[:, 1] = y0
        memory[:, 2] = x1 - x0
        memory[:, 3] = y1 - y0
        self._prepareIndex(memory)

    def _prepareIndex(self, memory: np.ndarray) -> None:
        """
        Detect whether the bars are sorted by x and keep searchable edge arrays.

        Parameters
        ----------
        memory : numpy.ndarray
            Rectangles of the bars, shape ``(n, 4)``: x0, y0, width, height.
        """
        widths = memory[:, 2]
        x0 = np.ascontiguousarray(memory[:, 0])
        x1 = x0 + widths
        # x1 is finite only if both x0 and the width are
        allFinite = bool(np.isfinite(x1).all())
        if allFinite and len(widths):
            self._meanWidth = float(widths.mean())
        else:
            finite = np.isfinite(widths)
            self._meanWidth = float(widths[finite].mean()) if finite.any() else math.nan
        self._xSorted = bool(len(x0) > 0 and allFinite and (x0[1:] >= x0[:-1]).all())
        # whether the right edges are sorted too, i.e. self._x1 holds the right edges
        # themselves rather than their running maximum
        self._x1Sorted = self._xSorted and bool((x1[1:] >= x1[:-1]).all())
        if self._xSorted:
            if not self._x1Sorted:
                # right edges are searched for the first visible bar: make them sorted
                x1 = np.maximum.accumulate(x1)
            self._x0, self._x1 = x0, x1
        else:
            self._x0 = self._x1 = None

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
        if withPen in self._styleIdCache:
            return self._styleIdCache[withPen]
        nbars = len(self._rectarray)
        penIds = self._penIds if withPen else None
        brushIds = self._brushIds
        if penIds is None and brushIds is None:
            ids = None
        else:
            for name, perBar in (('pens', penIds), ('brushes', brushIds)):
                if perBar is not None and len(perBar) < nbars:
                    raise IndexError(
                        f"BarGraphItem: {len(perBar)} {name} given for {nbars} bars")
            ids = np.zeros(nbars, dtype=np.intp)
            if penIds is not None:
                ids += penIds[:nbars] * len(self._uniqueBrushes)
            if brushIds is not None:
                ids += brushIds[:nbars]
        self._styleIdCache[withPen] = ids
        return ids

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
        nbrushes = len(self._uniqueBrushes)
        pen = self._uniquePens[sid // nbrushes] if withPen else self._noPen
        return pen, self._uniqueBrushes[sid % nbrushes]

    def _drawGroups(self, painter: QtGui.QPainter, rects,
                    groups: list[tuple[int, int, int]], withPen: bool) -> None:
        """
        Draw ranges of rectangles, one ``drawRects`` call per range.

        Parameters
        ----------
        painter : QtGui.QPainter
            Destination painter.
        rects : PrimitiveArray
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
                if self._lodKey != key:
                    groups = self._aggregate(start, stop, tr.m11(), tr.dx(), withPen)
                    self._lodValue = self._lodRects, groups
                    self._lodKey = key
                self._drawGroups(p, *self._lodValue, withPen)
                return

        if withPen:
            self._drawInOrder(p, start, stop)
            return
        key = ('withoutPen', start, stop, pxPerUnit)
        if self._lodKey != key:
            self._lodValue = self._withoutPen(start, stop, pxPerUnit)
            self._lodKey = key
        self._drawGroups(p, *self._lodValue, False)

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
        rects : PrimitiveArray
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
            keys = ids.astype(np.uint16) if len(self._uniqueBrushes) <= 65536 else ids
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
        falls in it, from their minimum y0 to their maximum y1.

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
        memory = self._rectarray.ndarray()[start:stop]
        x0 = self._x0[start:stop]
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
        valid = np.isfinite(ylo) & np.isfinite(yhi)

        ids = self._styleIds(withPen)
        if ids is None:
            order = np.flatnonzero(valid)
            groups = [(0, 0, len(order))]
        else:
            # style of the tallest bar of each column
            counts = np.diff(np.append(first, len(x0)))
            segment = np.repeat(np.arange(len(first)), counts)
            tallest = np.flatnonzero(heights == np.fmax.reduceat(heights, first)[segment])
            keep = np.ones(len(tallest), dtype=bool)
            keep[1:] = segment[tallest[1:]] != segment[tallest[:-1]]
            tallest = tallest[keep]
            columnStyle = np.full(len(first), -1, dtype=np.intp)
            columnStyle[segment[tallest]] = ids[start:stop][tallest]
            selected = np.flatnonzero(valid & (columnStyle >= 0))
            order = selected[np.argsort(columnStyle[selected], kind='stable')]
            groups = _runs(columnStyle[order])

        self._lodRects.resize(len(order))
        out = self._lodRects.ndarray()
        out[:, 0] = xlo[order]
        out[:, 1] = ylo[order]
        out[:, 2] = xhi[order] - out[:, 0]
        out[:, 3] = yhi[order] - out[:, 1]
        return groups
            
    def shape(self):
        if self._shape is None:
            shape = QtGui.QPainterPath()
            rects = self._rectarray.instances()
            for rect in rects:
                shape.addRect(rect)
            self._shape = shape
        return self._shape

    def implements(self, interface=None):
        ints = ['plotData']
        if interface is None:
            return ints
        return interface in ints

    def name(self):
        return self.opts.get('name', None)

    def getData(self):
        return self.opts.get('x'),  self.opts.get('height')

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
            ``(min, max)``, or ``(None, None)`` without data or when no bar is
            considered.

        Notes
        -----
        The full range (``frac=1`` without ``orthoRange``) is computed once per data
        change; it is NaN when a coordinate of a bar is NaN. The restricted and
        percentile ranges ignore the bars with a NaN coordinate.
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
        if orthoRange is not None:
            lo, hi = sorted(float(value) for value in orthoRange)
            if ax == 1 and self._xSorted:
                # first bar whose right edge reaches lo, first bar starting after hi
                start = int(np.searchsorted(self._x1, lo, side='left'))
                stop = int(np.searchsorted(self._x0, hi, side='right'))
                if frac >= 1.0 and self._x1Sorted and stop - start == len(memory):
                    # all bars, e.g. a zoomed out view: reuse the full range if finite
                    ymin, ymax = self._dataBounds[1]
                    if math.isfinite(ymin) and math.isfinite(ymax):
                        return float(ymin), float(ymax)
                memory = memory[start:stop]
                if not self._x1Sorted:
                    # self._x1 is the running maximum of the right edges: a bar
                    # within the slice can still end before lo
                    memory = memory[memory[:, 0] + memory[:, 2] >= lo]
            else:
                other = 1 - ax
                low = memory[:, other]
                memory = memory[(low <= hi) & (low + memory[:, other + 2] >= lo)]
        if len(memory) == 0:
            return None, None
        low = memory[:, ax]
        high = low + memory[:, ax + 2]
        if frac >= 1.0:
            bmin, bmax = np.fmin.reduce(low), np.fmax.reduce(high)
        else:
            low = low[np.isfinite(low)]
            high = high[np.isfinite(high)]
            if len(low) == 0 or len(high) == 0:
                return None, None
            bmin = np.percentile(low, 50 * (1 - frac))
            bmax = np.percentile(high, 50 * (1 + frac))
        if math.isnan(bmin) or math.isnan(bmax):
            return None, None
        return float(bmin), float(bmax)

    def pixelPadding(self):
        # _penWidth is available after _updateColors()
        pw = (self._penWidth[1] or 1) * 0.5
        return pw

    def boundingRect(self):
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
