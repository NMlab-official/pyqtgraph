import enum
import itertools
import math
import operator
import weakref
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured

from .. import Qt, debug
from .. import functions as fn
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from .GraphicsObject import GraphicsObject

if TYPE_CHECKING:
    from ..GraphicsScene.mouseEvents import HoverEvent

__all__ = ['ScatterPlotItem', 'SpotItem']


## Build all symbol paths
name_list = ['o', 's', 't', 't1', 't2', 't3', 'd', '+', 'x', 'p', 'h', 'star', '|', '_',
             'arrow_up', 'arrow_right', 'arrow_down', 'arrow_left', 'crosshair']
Symbols = OrderedDict([(name, QtGui.QPainterPath()) for name in name_list])
Symbols['o'].addEllipse(QtCore.QRectF(-0.5, -0.5, 1, 1))
Symbols['s'].addRect(QtCore.QRectF(-0.5, -0.5, 1, 1))

def makeCrosshair(r=0.5, w=1, h=1):
    path = QtGui.QPainterPath()
    rect = QtCore.QRectF(-r, -r, r * 2, r * 2)
    path.addEllipse(rect)
    path.moveTo(-w, 0)
    path.lineTo(w, 0)
    path.moveTo(0, -h)
    path.lineTo(0, h)
    return path
Symbols['crosshair'] = makeCrosshair()

coords = {
    't': [(-0.5, -0.5), (0, 0.5), (0.5, -0.5)],
    't1': [(-0.5, 0.5), (0, -0.5), (0.5, 0.5)],
    't2': [(-0.5, -0.5), (-0.5, 0.5), (0.5, 0)],
    't3': [(0.5, 0.5), (0.5, -0.5), (-0.5, 0)],
    'd': [(0., -0.5), (-0.4, 0.), (0, 0.5), (0.4, 0)],
    '+': [
        (-0.5, -0.1), (-0.5, 0.1), (-0.1, 0.1), (-0.1, 0.5),
        (0.1, 0.5), (0.1, 0.1), (0.5, 0.1), (0.5, -0.1),
        (0.1, -0.1), (0.1, -0.5), (-0.1, -0.5), (-0.1, -0.1)
    ],
    'p': [(0, -0.5), (-0.4755, -0.1545), (-0.2939, 0.4045),
          (0.2939, 0.4045), (0.4755, -0.1545)],
    'h': [(0.433, 0.25), (0., 0.5), (-0.433, 0.25), (-0.433, -0.25),
          (0, -0.5), (0.433, -0.25)],
    'star': [(0, -0.5), (-0.1123, -0.1545), (-0.4755, -0.1545),
             (-0.1816, 0.059), (-0.2939, 0.4045), (0, 0.1910),
             (0.2939, 0.4045), (0.1816, 0.059), (0.4755, -0.1545),
             (0.1123, -0.1545)],
    '|': [(-0.1, 0.5),(0.1, 0.5), (0.1, -0.5), (-0.1, -0.5)],
    'arrow_up': [
        (-0.125, 0.125), (0, 0), (0.125, 0.125),
        (0.05, 0.125), (0.05, 0.5), (-0.05, 0.5), (-0.05, 0.125)
    ]
}
for k, c in coords.items():
    Symbols[k].moveTo(*c[0])
    for x,y in c[1:]:
        Symbols[k].lineTo(x, y)
    Symbols[k].closeSubpath()
tr = QtGui.QTransform()
tr.rotate(45)
Symbols['x'] = tr.map(Symbols['+'])
tr.rotate(45)
Symbols['arrow_right'] = tr.map(Symbols['arrow_up'])
Symbols['arrow_down'] = tr.map(Symbols['arrow_right'])
Symbols['arrow_left'] = tr.map(Symbols['arrow_down'])

# already rotated 90 degrees from earlier commands
Symbols['_'] = tr.map(Symbols['|'])
_DEFAULT_STYLE = {'symbol': None, 'size': -1, 'pen': None, 'brush': None, 'visible': True}


def drawSymbol(painter, symbol, size, pen, brush):
    if symbol is None:
        return
    painter.scale(size, size)
    painter.setPen(pen)
    painter.setBrush(brush)
    if isinstance(symbol, str):
        symbol = Symbols[symbol]
    if np.isscalar(symbol):
        symbol = list(Symbols.values())[symbol % len(Symbols)]
    painter.drawPath(symbol)


def renderSymbol(symbol, size, pen, brush, device=None, dpr=1.0):
    """
    Render a symbol specification to QImage.
    Symbol may be either a QPainterPath or one of the keys in the Symbols dict.
    If *device* is None, a new QPixmap will be returned. Otherwise,
    the symbol will be rendered into the device specified (See QPainter documentation
    for more information).
    """
    ## Render a spot with the given parameters to a pixmap
    penPxWidth = max(math.ceil(pen.widthF()), 1)
    if device is None:
        side = int(math.ceil(dpr*(size+penPxWidth)))
        device = QtGui.QImage(side, side, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
        device.setDevicePixelRatio(dpr)
        device.fill(QtCore.Qt.GlobalColor.transparent)
    p = QtGui.QPainter(device)
    try:
        p.setRenderHint(p.RenderHint.Antialiasing)
        p.translate(device.width()/dpr*0.5, device.height()/dpr*0.5)
        drawSymbol(p, symbol, size, pen, brush)
    finally:
        p.end()
    return device


def _mkPen(*args, **kwargs):
    """
    Wrapper for fn.mkPen which avoids creating a new QPen object if passed one as its
    sole argument, so that the pens given by the user are stored as they are.
    """
    if len(args) == 1 and isinstance(args[0], QtGui.QPen):
        return args[0]
    else:
        return fn.mkPen(*args, **kwargs)


def _mkBrush(*args, **kwargs):
    """
    Wrapper for fn.mkBrush which avoids creating a new QBrush object if passed one as its
    sole argument, so that the brushes given by the user are stored as they are.
    """
    if len(args) == 1 and isinstance(args[0], QtGui.QBrush):
        return args[0]
    else:
        return fn.mkBrush(*args, **kwargs)


def _isNoneMask(col: np.ndarray) -> np.ndarray:
    """
    Return a mask of the entries of an object array that are ``None``.

    The entries are tested by identity. ``np.equal(col, None)`` would instead call the
    ``__eq__`` method of every entry, which is slow for Qt objects such as ``QPen`` and
    ``QBrush``.

    Parameters
    ----------
    col : numpy.ndarray
        One-dimensional array, usually of ``object`` dtype.

    Returns
    -------
    numpy.ndarray
        Boolean array of the same length as ``col``, ``True`` where the entry is ``None``.
    """
    # iterating over a list is faster than iterating over an object array
    return np.fromiter(map(operator.is_, col.tolist(), itertools.repeat(None)),
                       dtype=bool, count=len(col))


def _mkMany(values: Iterable, qtype: type, make: Callable[[object], object]
            ) -> tuple[np.ndarray, list | None, np.ndarray | None]:
    """
    Convert each value with ``make``, building one object per unique hashable value.

    Values that are already instances of ``qtype`` are kept as they are. The other
    values are converted once per distinct ``(type, value)`` pair (the type keeps, for
    example, ``1`` and ``1.0`` apart, which ``mkColor`` interprets differently); equal
    values then share the same object, which keeps the symbol atlas small. Unhashable
    values (dicts, lists, arrays) are converted one by one.

    Parameters
    ----------
    values : iterable
        Values accepted by ``make``, such as colors or ``QBrush`` objects.
    qtype : type
        Type of the objects produced, ``QtGui.QPen`` or ``QtGui.QBrush``.
    make : callable
        Conversion function, such as :func:`~pyqtgraph.mkBrush`.

    Returns
    -------
    objects : numpy.ndarray
        Object array holding one ``qtype`` object per value, in the same order.
    reps : list or None
        The distinct objects made, when every value was converted through the
        memo; ``None`` otherwise.
    codes : numpy.ndarray or None
        Index in ``reps`` of each value, or ``None`` when ``reps`` is ``None``.
    """
    if isinstance(values, np.ndarray):
        values = values.tolist()  # iterating over a list is faster
    else:
        values = list(values)
    n = len(values)
    if all(map(isinstance, values, itertools.repeat(qtype))):
        return np.fromiter(values, dtype=object, count=n), None, None
    cache = {}
    reps = []
    codes = []
    out = []
    append = out.append
    for value in values:
        if isinstance(value, qtype):
            append(value)
            codes = None
            continue
        key = (type(value), value)
        try:
            code = cache.get(key)
        except TypeError:  # unhashable value
            append(make(value))
            codes = None
            continue
        if code is None:
            code = cache[key] = len(reps)
            reps.append(make(value))
        append(reps[code])
        if codes is not None:
            codes.append(code)
    objects = np.fromiter(out, dtype=object, count=n)
    if codes is None:
        return objects, None, None
    return objects, reps, np.array(codes, dtype=np.intp)


def _colorArrayToObjects(colors: object, make: Callable[[object], object]
                         ) -> tuple[np.ndarray, list, np.ndarray] | None:
    """
    Convert an array of RGB(A) colors into pens or brushes, one per distinct color.

    This is the numeric fast path of :meth:`ScatterPlotItem.setBrush` and
    :meth:`ScatterPlotItem.setPen`. The colors give the same objects as converting
    each row with ``make``: channels are 0-255 values, converted with ``int()``
    (non-finite values become 0), and the alpha channel defaults to 255.

    Parameters
    ----------
    colors : object
        Candidate ``(N, 3)`` or ``(N, 4)`` numeric array.
    make : callable
        Conversion of an ``(r, g, b, a)`` tuple, :func:`~pyqtgraph.mkBrush` or
        :func:`~pyqtgraph.mkPen`.

    Returns
    -------
    tuple or None
        ``(objects, reps, codes)``: the object array of ``N`` pens or brushes, equal
        colors sharing the same object, the distinct objects, and the index in
        ``reps`` of each color. ``None`` if ``colors`` is not such an array or has
        channels outside 0-255, which are left to the generic path.
    """
    if not (isinstance(colors, np.ndarray) and colors.ndim == 2
            and colors.shape[1] in (3, 4) and colors.dtype.kind in 'uif'):
        return None
    values = colors
    if values.dtype != np.uint8:
        if values.dtype.kind == 'f':
            values = np.trunc(np.where(np.isfinite(values), values, 0))
        if values.size and (values.min() < 0 or values.max() > 255):
            return None
    rgba = np.full((len(values), 4), 255, dtype=np.uint8)
    rgba[:, :values.shape[1]] = values
    packed = rgba.view(np.uint32).ravel()
    unique, codes = np.unique(packed, return_inverse=True)
    first = _representatives(codes, len(unique))
    reps = [make(tuple(row)) for row in rgba[first].tolist()]
    objects = np.empty(len(reps), dtype=object)
    objects[:] = reps
    return objects[codes], reps, codes


def _representatives(codes: np.ndarray, n: int) -> np.ndarray:
    """
    Return the position of one element of each group.

    Parameters
    ----------
    codes : numpy.ndarray
        Group index of each element, with every group in ``range(n)`` present.
    n : int
        Number of groups.

    Returns
    -------
    numpy.ndarray
        For each group, the index of one of its elements (the last one).
    """
    first = np.empty(n, dtype=np.intp)
    first[codes] = np.arange(len(codes))
    return first


def _groupObjects(objs: list) -> tuple[list, np.ndarray | None]:
    """
    Group objects by identity.

    Parameters
    ----------
    objs : list
        The objects.

    Returns
    -------
    reps : list
        One object per group.
    codes : numpy.ndarray or None
        Group index of each object, or ``None`` when all objects are the same one.
    """
    ids = np.fromiter(map(id, objs), dtype=np.intp, count=len(objs))
    if len(ids) == 0 or (ids == ids[0]).all():
        return objs[:1], None
    unique, codes = np.unique(ids, return_inverse=True)
    return [objs[i] for i in _representatives(codes, len(unique)).tolist()], codes


def _mergeEqual(reps: list, codes: np.ndarray | None) -> tuple[list, np.ndarray | None]:
    """
    Merge the groups of equal hashable representatives.

    Used for symbols, where equal strings are often distinct objects (for instance
    when taken from a numpy string array).

    Parameters
    ----------
    reps : list
        One object per group.
    codes : numpy.ndarray or None
        Group index of each element, or ``None`` for a single group.

    Returns
    -------
    reps : list
        The representatives of the merged groups.
    codes : numpy.ndarray or None
        The merged group index of each element.
    """
    if codes is None:
        return reps, codes
    index = {}
    remap = []
    merged = []
    for rep in reps:
        key = (type(rep), rep)
        try:
            j = index.get(key)
        except TypeError:  # unhashable, such as QPainterPath: keep the identity group
            j = None
            key = None
        if j is None:
            j = len(merged)
            merged.append(rep)
            if key is not None:
                index[key] = j
        remap.append(j)
    if len(merged) == len(reps):
        return reps, codes
    return merged, np.array(remap, dtype=np.intp)[codes]


def _groupValues(values: np.ndarray) -> tuple[list, np.ndarray | None]:
    """
    Group equal numeric values.

    Parameters
    ----------
    values : numpy.ndarray
        One-dimensional numeric array.

    Returns
    -------
    reps : list
        The distinct values, as Python scalars.
    codes : numpy.ndarray or None
        Index of each value in ``reps``, or ``None`` when all values are equal.
    """
    if len(values) == 0 or (values == values[0]).all():
        return values[:1].tolist(), None
    reps, codes = np.unique(values, return_inverse=True)
    return reps.tolist(), codes


def _quantizeSize(size: float, step: float) -> float:
    """
    Round a symbol size to a multiple of ``1 / step``.

    Integral sizes are returned unchanged, so that they render exactly as requested
    whatever the device pixel ratio. Non-finite sizes are returned unchanged.

    Parameters
    ----------
    size : float
        Symbol size in pixels.
    step : float
        Number of quantization steps per pixel.

    Returns
    -------
    float
        The quantized size.
    """
    size = float(size)
    if size.is_integer() or not math.isfinite(size):
        return size
    return round(size * step) / step


def _quantizeSizes(sizes: np.ndarray, step: float) -> np.ndarray:
    """
    Vectorized :func:`_quantizeSize`, giving the same values.

    Parameters
    ----------
    sizes : numpy.ndarray
        Symbol sizes in pixels.
    step : float
        Number of quantization steps per pixel.

    Returns
    -------
    numpy.ndarray
        The quantized sizes, as a new float64 array.
    """
    sizes = np.array(sizes, dtype=np.float64)
    rounded = np.rint(sizes * step) / step
    change = np.isfinite(sizes) & (sizes != np.trunc(sizes))
    sizes[change] = rounded[change]
    return sizes


def _enumValue(value: enum.Enum | int) -> int:
    """
    Return the integer value of a Qt enum member, whatever the Qt binding.

    Integers hash much faster than ``enum.Enum`` members, whose ``__hash__`` is
    implemented in Python.

    Parameters
    ----------
    value : enum.Enum or int
        Enum member (PyQt6, PySide6) or integer-like enum value (PyQt5).

    Returns
    -------
    int
        The value.
    """
    return value.value if isinstance(value, enum.Enum) else int(value)


# brush styles whose rendering is not fully described by the color and the style
_SHAPED_BRUSH_STYLES = frozenset((
    QtCore.Qt.BrushStyle.LinearGradientPattern,
    QtCore.Qt.BrushStyle.RadialGradientPattern,
    QtCore.Qt.BrushStyle.ConicalGradientPattern,
    QtCore.Qt.BrushStyle.TexturePattern,
))
# brush styles whose rendering does not depend on the brush transform
_PLAIN_BRUSH_STYLES = frozenset((
    QtCore.Qt.BrushStyle.NoBrush,
    QtCore.Qt.BrushStyle.SolidPattern,
))


def _brushValueKey(brush: QtGui.QBrush) -> tuple | None:
    """
    Return a hashable key describing how a brush renders, if it can be built.

    Parameters
    ----------
    brush : QtGui.QBrush
        The brush.

    Returns
    -------
    tuple or None
        ``(rgba, style)`` as integers, or ``None`` for gradient and texture brushes
        and for transformed hatch patterns, which must be keyed by identity.
    """
    style = brush.style()
    if style in _SHAPED_BRUSH_STYLES:
        return None
    if style not in _PLAIN_BRUSH_STYLES and not brush.transform().isIdentity():
        return None
    return (brush.color().rgba(), _enumValue(style))


def _penValueKey(pen: QtGui.QPen) -> tuple | None:
    """
    Return a hashable key describing how a pen renders, if it can be built.

    Besides the color, width, style and cosmetic flag, the key holds the cap and join
    styles, the miter limit and the dash pattern, which all change the rendered
    symbol.

    Parameters
    ----------
    pen : QtGui.QPen
        The pen.

    Returns
    -------
    tuple or None
        The key, or ``None`` when the pen brush cannot be keyed by value.
    """
    brushKey = _brushValueKey(pen.brush())
    if brushKey is None:
        return None
    style = pen.style()
    dash = tuple(pen.dashPattern()) if style == QtCore.Qt.PenStyle.CustomDashLine else ()
    return (brushKey, pen.widthF(), _enumValue(style), pen.isCosmetic(),
            _enumValue(pen.capStyle()), _enumValue(pen.joinStyle()), pen.miterLimit(),
            dash, pen.dashOffset())


class SymbolAtlas(object):
    """
    Used to efficiently construct a single QPixmap containing all rendered symbols
    for a ScatterPlotItem. This is required for fragment rendering.

    Use example:
        atlas = SymbolAtlas()
        sc1 = atlas[[('o', 5, QPen(..), QBrush(..))]]
        sc2 = atlas[[('t', 10, QPen(..), QBrush(..))]]
        pm = atlas.pixmap

    Symbols are keyed by value: pens and brushes that render the same share one
    entry, whatever the Python objects. Gradient and texture brushes, and custom
    ``QPainterPath`` symbols, are keyed by identity. Non-integral sizes are rounded
    to a quarter of a device pixel (``1 / (4 * devicePixelRatio)`` logical pixel),
    and the symbol is rendered at the rounded size.
    """
    _idGenerator = itertools.count()

    def __init__(self):
        self._dpr = 1.0
        self.clear()

    def __getitem__(self, styles):
        """
        Given a list of tuples, (symbol, size, pen, brush), return a list of coordinates of
        corresponding symbols within the atlas. Note that these coordinates may change if the atlas is rebuilt.
        """
        keys, inverse, renderStyles = self._uniqueKeys(styles)
        new = {key: style for key, style in zip(keys, renderStyles) if key not in self._coords}

        if new:
            self._extend(new)

        coords = [self._coords[key] for key in keys]
        return [coords[i] for i in inverse]

    def __len__(self):
        return len(self._coords)

    def devicePixelRatio(self):
        return self._dpr

    def setDevicePixelRatio(self, dpr):
        self._dpr = dpr

    @property
    def pixmap(self):
        if self._pixmap is None:
            self._pixmap = self._createPixmap()
        return self._pixmap

    @property
    def maxWidth(self):
        # return the max logical width
        return self._maxWidth / self._dpr

    def rebuild(self, styles=None):
        profiler = debug.Profiler()  # noqa: profiler prints on GC
        if styles is None:
            data = []
        else:
            keys = self._uniqueKeys(styles)[0]
            data = list(self._itemData(keys))

        self.clear()
        if data:
            self._extendFromData(data)

    def clear(self):
        self._data = np.zeros((0, 0, 4), dtype=np.ubyte)  # numpy array of atlas image
        self._coords = {}
        self._pixmap = None
        self._maxWidth = 0
        self._totalWidth = 0
        self._totalArea = 0
        self._pos = (0, 0)
        self._rowShape = (0, 0)

    def diagnostics(self):
        n = len(self)
        w, h, _ = self._data.shape
        a = self._totalArea
        return dict(count=n,
                    width=w,
                    height=h,
                    area=w * h,
                    area_used=1.0 if n == 0 else a / (w * h),
                    squareness=1.0 if n == 0 else 2 * w * h / (w**2 + h**2))

    @staticmethod
    def _identityKey(obj: object) -> tuple[str, int]:
        """
        Return a key identifying an object for the lifetime of the program.

        A counter value is stored on the object: unlike ``id()``, it is never reused by
        another object.

        Parameters
        ----------
        obj : object
            Object accepting new attributes, such as a Qt object.

        Returns
        -------
        tuple
            ``('id', n)``, which never equals a value key.
        """
        try:
            return ('id', obj._id)
        except AttributeError:
            obj._id = next(SymbolAtlas._idGenerator)
            return ('id', obj._id)

    def _uniqueKeys(self, styles: Sequence[tuple]) -> tuple[list[tuple], list[int], list[tuple]]:
        """
        Compute the distinct atlas keys of a sequence of styles.

        The key of a style is ``(symbolKey, quantizedSize, penKey, brushKey)``: pens and
        brushes are described by value (see :func:`_penValueKey` and
        :func:`_brushValueKey`) when possible, by identity otherwise, and the size is
        quantized (see :func:`_quantizeSize`).

        Parameters
        ----------
        styles : sequence of tuple
            ``(symbol, size, pen, brush)`` tuples.

        Returns
        -------
        keys : list of tuple
            The distinct keys, in order of first appearance.
        inverse : list of int
            For each style, the index of its key in ``keys``.
        renderStyles : list of tuple
            For each key, the ``(symbol, size, pen, brush)`` style to render, with the
            quantized size.
        """
        step = 4 * self._dpr
        identityKey = self._identityKey
        keys = []
        inverse = []
        renderStyles = []
        append = inverse.append
        # Styles are first grouped by object identity, which is cheap; the value keys
        # are then computed once per group. The identities are only valid during this
        # call, while ``styles`` holds the objects: a pen or brush modified between two
        # calls is keyed by its new value.
        byIdentity = {}
        byValue = {}
        penKeys = {}
        brushKeys = {}
        for symbol, size, pen, brush in styles:
            groupKey = (id(symbol), size, id(pen), id(brush))
            index = byIdentity.get(groupKey)
            if index is None:
                penKey = penKeys.get(id(pen))
                if penKey is None:
                    penKey = penKeys[id(pen)] = _penValueKey(pen) or identityKey(pen)
                brushKey = brushKeys.get(id(brush))
                if brushKey is None:
                    brushKey = brushKeys[id(brush)] = (_brushValueKey(brush)
                                                       or identityKey(brush))
                symbolKey = symbol if isinstance(symbol, (str, int)) else identityKey(symbol)
                key = (symbolKey, _quantizeSize(size, step), penKey, brushKey)
                index = byValue.get(key)
                if index is None:
                    index = byValue[key] = len(keys)
                    keys.append(key)
                    renderStyles.append((symbol, key[1], pen, brush))
                byIdentity[groupKey] = index
            append(index)
        return keys, inverse, renderStyles

    def _itemData(self, keys):
        for key in keys:
            y, x, h, w = self._coords[key]
            yield key, self._data[x:x + w, y:y + h]

    def _extend(self, styles):
        profiler = debug.Profiler()

        images = []
        data = []
        for key, style in styles.items():
            img = renderSymbol(*style, dpr=self._dpr)
            arr = fn.ndarray_from_qimage(img)
            images.append(img)  # keep these to delay garbage collection
            data.append((key, arr))

        profiler('render')
        self._extendFromData(data)
        profiler('insert')

    def _extendFromData(self, data):
        self._pack(data)

        # expand array if necessary
        wNew, hNew = self._minDataShape()
        wOld, hOld, _ = self._data.shape
        if (wNew > wOld) or (hNew > hOld):
            arr = np.zeros((wNew, hNew, 4), dtype=np.ubyte)
            arr[:wOld, :hOld] = self._data
            self._data = arr

        # insert data into array
        for key, arr in data:
            y, x, h, w = self._coords[key]
            self._data[x:x+w, y:y+h] = arr

        self._pixmap = None

    def _pack(self, data):
        # pack each item rectangle as efficiently as possible into a larger, expanding, approximate square
        n = len(self)
        wMax = self._maxWidth
        wSum = self._totalWidth
        aSum = self._totalArea
        x, y = self._pos
        wRow, hRow = self._rowShape

        # update packing statistics
        for _, arr in data:
            w, h, _ = arr.shape
            wMax = max(w, wMax)
            wSum += w
            aSum += w * h
        n += len(data)

        # maybe expand row width for squareness and to accommodate largest width
        wRowEst = int(wSum / (n ** 0.5))
        if wRowEst > 2 * wRow:
            wRow = wRowEst
        wRow = max(wMax, wRow)

        # set coordinates by packing along rows
        # sort by rectangle height first to improve packing density
        for key, arr in sorted(data, key=lambda data: data[1].shape[1]):
            w, h, _ = arr.shape
            if x + w > wRow:
                # move up a row
                x = 0
                y += hRow
                hRow = h
            hRow = max(h, hRow)
            self._coords[key] = (y, x, h, w)
            x += w

        self._maxWidth = wMax
        self._totalWidth = wSum
        self._totalArea = aSum
        self._pos = (x, y)
        self._rowShape = (wRow, hRow)

    def _minDataShape(self):
        x, y = self._pos
        w, h = self._rowShape
        return int(w), int(y + h)

    def _createPixmap(self):
        profiler = debug.Profiler()  # noqa: profiler prints on GC
        if self._data.size == 0:
            pm = QtGui.QPixmap(0, 0)
        else:
            img = fn.ndarray_to_qimage(self._data,
                QtGui.QImage.Format.Format_ARGB32_Premultiplied)
            pm = QtGui.QPixmap.fromImage(img)
        return pm


class _SpotArrays:
    """
    Contiguous copies of the spot fields used to paint and to hit-test.

    The fields of the structured spot array are strided (one record per spot), which
    makes every vectorized pass over them slow. These copies are built on demand from
    :attr:`ScatterPlotItem.data` and dropped by :meth:`ScatterPlotItem.invalidate`
    whenever the spots change.

    Parameters
    ----------
    data : numpy.ndarray
        Structured spot array.
    """

    __slots__ = ('serial', 'x', 'y', 'sourceRect', 'uniformRect', 'halfWidth', 'halfHeight',
                 'visible', 'allVisible', 'bounds')

    _serials = itertools.count()

    def __init__(self, data: np.ndarray) -> None:
        n = len(data)
        self.serial = next(_SpotArrays._serials)  # identifies these arrays
        self.x = np.array(data['x'], dtype=np.float64)
        self.y = np.array(data['y'], dtype=np.float64)
        # one pass over the records for the four fields (sx, sy, sw, sh)
        self.sourceRect = structured_to_unstructured(data['sourceRect'], dtype=np.float64)
        # the source rect shared by all the spots (uniform style), if any
        self.uniformRect = None
        if n and (self.sourceRect == self.sourceRect[0]).all():
            self.uniformRect = tuple(self.sourceRect[0].tolist())
        # half symbol sizes in device pixels: a scalar when uniform, else one per spot
        self.halfWidth = self._halfSizes(self.sourceRect[:, 2])
        self.halfHeight = self._halfSizes(self.sourceRect[:, 3])
        self.visible = np.array(data['visible'], dtype=bool)
        self.allVisible = bool(self.visible.all())
        # NaN coordinates give NaN bounds, which then never compare as inside a view
        self.bounds = (None if n == 0 else
                       (self.x.min(), self.x.max(), self.y.min(), self.y.max()))

    @staticmethod
    def _halfSizes(sizes: np.ndarray) -> float | np.ndarray:
        """
        Halve symbol sizes, returning a scalar when they are all equal.

        Parameters
        ----------
        sizes : numpy.ndarray
            Symbol widths or heights.

        Returns
        -------
        float or numpy.ndarray
            ``sizes / 2``, or its single value; both give the same results in
            arithmetic with the spot coordinates.
        """
        if len(sizes) and (sizes == sizes[0]).all():
            return float(sizes[0]) / 2
        return sizes / 2


def _mapPoints(transform: QtGui.QTransform, x: np.ndarray, y: np.ndarray,
               out: np.ndarray) -> None:
    """
    Map points to device coordinates, as :func:`~pyqtgraph.functions.transformCoordinates`.

    The operations are done in the same order as ``transformCoordinates`` followed by
    a clip to +-2**30 (larger coordinates crash Qt), so that the results are
    bit-identical; perspective is ignored likewise.

    Parameters
    ----------
    transform : QtGui.QTransform
        Item to device transform.
    x, y : numpy.ndarray
        Coordinates of the points.
    out : numpy.ndarray
        ``(N, 2)`` float64 array, or view, receiving the mapped coordinates.
    """
    for col, (mx, my, d) in enumerate(((transform.m11(), transform.m21(), transform.dx()),
                                       (transform.m12(), transform.m22(), transform.dy()))):
        # computed in a contiguous array: passes over a strided column are slow
        with np.errstate(invalid='ignore', over='ignore'):
            mapped = x * mx
            mapped += y * my
            mapped += d
        np.clip(mapped, -2 ** 30, 2 ** 30, out=mapped)
        out[:, col] = mapped


def _screensHaveIntegralPixelRatio() -> bool:
    """
    Tell whether every screen has an integral device pixel ratio.

    ``QGraphicsItem.DeviceCoordinateCache`` renders the item into a pixmap that is not
    aware of fractional device pixel ratios (e.g. a 125 % display scale): the cached
    rendering then differs visibly from a direct paint.

    Returns
    -------
    bool
        True if no screen has a fractional device pixel ratio, or if there is no GUI
        application yet.
    """
    if QtGui.QGuiApplication.instance() is None:
        return True
    return all(float(screen.devicePixelRatio()).is_integer()
               for screen in QtGui.QGuiApplication.screens())


class ScatterPlotItem(GraphicsObject):
    """
    Displays a set of x/y points. Instances of this class are created
    automatically as part of PlotDataItem; these rarely need to be instantiated
    directly.

    The size, shape, pen, and fill brush may be set for each point individually
    or for all points.


    ============================  ===============================================
    **Signals:**
    sigPlotChanged(self)          Emitted when the data being plotted has changed
    sigClicked(self, points, ev)  Emitted when points are clicked. Sends a list
                                  of all the points under the mouse pointer.
    sigHovered(self, points, ev)  Emitted when the item is hovered. Sends a list
                                  of all the points under the mouse pointer.
    ============================  ===============================================

    """
    #sigPointClicked = QtCore.Signal(object, object)
    sigClicked = QtCore.Signal(object, object, object)
    sigHovered = QtCore.Signal(object, object, object)
    sigPlotChanged = QtCore.Signal(object)

    def __init__(self, *args, **kwargs):
        """
        Accepts the same arguments as setData()
        """
        profiler = debug.Profiler()
        GraphicsObject.__init__(self)

        self.picture = None   # QPicture used for rendering when pxmode==False
        self.fragmentAtlas = SymbolAtlas()
        if screen := QtGui.QGuiApplication.primaryScreen():
            self.fragmentAtlas.setDevicePixelRatio(screen.devicePixelRatio())
        # The atlas is rebuilt with the styles in use when it holds more entries than
        # this, unless at least half of its entries are in use.
        self._atlasMaxEntries = 4096
        # During addPoints: grouping of the per-spot pens and brushes, when known from
        # their conversion, keyed by column name (see _setStyleColumn)
        self._styleCodes = None
        # True once views of self.data were handed out (SpotItems, getData), which
        # forbids reusing the array in place for the next setData
        self._dataShared = False
        # During setData: previous spot array that addPoints may reuse in place
        self._reusableData = None
        # Contiguous copies of the spot fields for painting and hit tests, built on
        # demand and dropped by invalidate() (see _spotArrays)
        self._paintCache = None
        # Columns already written in the pixmap fragment buffer, as (key, rows) where
        # the key holds the buffer address (see _prepareFragments)
        self._fragmentRects = None
        self._fragmentConstants = None

        dtype = [
            ('x', float),
            ('y', float),
            ('size', float),
            ('symbol', object),
            ('pen', object),
            ('brush', object),
            ('visible', bool),
            ('data', object),
            ('hovered', bool),
            ('item', object),
            ('sourceRect', [
                ('x', int),
                ('y', int),
                ('w', int),
                ('h', int)
            ])
        ]

        self.data = np.empty(0, dtype=dtype)
        self._initialSpot = self._makeInitialSpot(self.data.dtype)
        self.bounds = [None, None]  ## caches data bounds
        self._maxSpotWidth = 0      ## maximum size of the scale-variant portion of all spots
        self._maxSpotPxWidth = 0    ## maximum size of the scale-invariant portion of all spots
        self._pixmapFragments = Qt.internals.PrimitiveArray(QtGui.QPainter.PixmapFragment, 10)
        self.opts = {
            'pxMode': True,
            'useCache': True,  ## If useCache is False, symbols are re-drawn on every paint.
            'useDeviceCache': False,  ## If True, the rendered item is cached in device coordinates.
            'antialias': getConfigOption('antialias'),
            'compositionMode': None,
            'name': None,
            'symbol': 'o',
            'size': 7,
            'pen': fn.mkPen(getConfigOption('foreground')),
            'brush': fn.mkBrush(100, 100, 150),
            'hoverable': False,
            'tip': 'x: {x:.3g}\ny: {y:.3g}\ndata={data}'.format,
        }
        self.opts.update(
            {'hover' + opt.title(): _DEFAULT_STYLE[opt] for opt in ['symbol', 'size', 'pen', 'brush']}
        )
        profiler()
        self.setData(*args, **kwargs)
        profiler('setData')
        # track when the tooltip is cleared so we only clear it once
        # this allows another item in the VB to set the tooltip
        self._toolTipCleared = True

    def setData(self, *args, **kwargs):
        """
        **Ordered Arguments:**

        * If there is only one unnamed argument, it will be interpreted like the 'spots' argument.
        * If there are two unnamed arguments, they will be interpreted as sequences of x and y values.

        ====================== ===============================================================================================
        **Keyword Arguments:**
        *spots*                Optional list of dicts. Each dict specifies parameters for a single spot:
                               {'pos': (x,y), 'size', 'pen', 'brush', 'symbol'}. This is just an alternate method
                               of passing in data for the corresponding arguments.
        *x*,*y*                1D arrays of x,y values.
        *pos*                  2D structure of x,y pairs (such as Nx2 array or list of tuples)
        *pxMode*               If True, spots are always the same size regardless of scaling, and size is given in px.
                               Otherwise, size is in scene coordinates and the spots scale with the view. To ensure
                               effective caching, QPen and QBrush objects should be reused as much as possible.
                               Default is True
        *symbol*               can be one (or a list) of symbols. For a list of supported symbols, see 
                               :func:`~ScatterPlotItem.setSymbol`. QPainterPath is also supported to specify custom symbol
                               shapes. To properly obey the position and size, custom symbols should be centered at (0,0) and
                               width and height of 1.0. Note that it is also possible to 'install' custom shapes by setting 
                               ScatterPlotItem.Symbols[key] = shape.
        *pen*                  The pen (or list of pens) to use for drawing spot outlines.
        *brush*                The brush (or list of brushes) to use for filling spots. For one color per spot, an
                               (N, 4) ``uint8`` array of RGBA values is the fastest (see :func:`~ScatterPlotItem.setBrush`).
        *size*                 The size (or list of sizes) of spots. If *pxMode* is True, this value is in pixels. Otherwise,
                               it is in the item's local coordinate system. In pixel mode, non-integral sizes are rounded
                               to a quarter of a device pixel.
        *data*                 a list of python objects used to uniquely identify each spot.
        *hoverable*            If True, sigHovered is emitted with a list of hovered points, a tool tip is shown containing
                               information about them, and an optional separate style for them is used. Default is False.
        *tip*                  A string-valued function of a spot's (x, y, data) values. Set to None to prevent a tool tip
                               from being shown.
        *hoverSymbol*          A single symbol to use for hovered spots. Set to None to keep symbol unchanged. Default is None.
        *hoverSize*            A single size to use for hovered spots. Set to -1 to keep size unchanged. Default is -1.
        *hoverPen*             A single pen to use for hovered spots. Set to None to keep pen unchanged. Default is None.
        *hoverBrush*           A single brush to use for hovered spots. Set to None to keep brush unchanged. Default is None.
        *useCache*             (bool) By default, generated point graphics items are cached to
                               improve performance. Setting this to False can improve image quality
                               in certain situations.
        *useDeviceCache*       (bool) Keep the rendered scatter plot in a pixmap of the view, so that items moving
                               over it (crosshair, cursors) do not repaint it. Default is False; see
                               :func:`~ScatterPlotItem.setUseDeviceCache` for the trade-offs.
        *antialias*            Whether to draw symbols with antialiasing. Note that if pxMode is True, symbols are
                               always rendered with antialiasing (since the rendered symbols can be cached, this
                               incurs very little performance cost)
        *compositionMode*      If specified, this sets the composition mode used when drawing the
                               scatter plot (see QPainter::CompositionMode in the Qt documentation).
        *name*                 The name of this item. Names are used for automatically
                               generating LegendItem entries and by some exporters.
        ====================== ===============================================================================================

        When the number of spots does not change (sliding window), the structured array
        ``self.data`` is reset and reused in place, unless SpotItems (:meth:`points`,
        :meth:`pointsAt`, hover and click signals) or :meth:`getData` views of it were
        handed out, or an argument shares its memory: those keep referring to the
        previous spots. References to ``self.data`` taken by other means see the new
        spots.
        """
        oldData = self.data  ## this causes cached pixmaps to be preserved while new data is registered.
        # The spot array is reused in place when the number of spots does not change,
        # unless views of it were handed out (see addPoints).
        self._reusableData = None if self._dataShared else oldData
        self.clear()  ## clear out all old data
        try:
            self.addPoints(*args, **kwargs)
        finally:
            self._reusableData = None

    def addPoints(self, *args, **kwargs):
        """
        Add new points to the scatter plot.
        Arguments are the same as setData()
        """

        ## deal with non-keyword arguments
        if len(args) == 1:
            kwargs['spots'] = args[0]
        elif len(args) == 2:
            kwargs['x'] = args[0]
            kwargs['y'] = args[1]
        elif len(args) > 2:
            raise Exception('Only accepts up to two non-keyword arguments.')

        ## convert 'pos' argument to 'x' and 'y'
        if 'pos' in kwargs:
            pos = kwargs['pos']
            if isinstance(pos, np.ndarray):
                kwargs['x'] = pos[:,0]
                kwargs['y'] = pos[:,1]
            else:
                x = []
                y = []
                for p in pos:
                    if isinstance(p, QtCore.QPointF):
                        x.append(p.x())
                        y.append(p.y())
                    else:
                        x.append(p[0])
                        y.append(p[1])
                kwargs['x'] = x
                kwargs['y'] = y

        ## determine how many spots we have
        if 'spots' in kwargs:
            numPts = len(kwargs['spots'])
        elif 'y' in kwargs and kwargs['y'] is not None:
            numPts = len(kwargs['y'])
        else:
            kwargs['x'] = []
            kwargs['y'] = []
            numPts = 0

        ## Clear current SpotItems since the data references they contain will no longer be current
        self.data['item'][...] = None

        reusable = self._reusableData
        self._reusableData = None
        if (reusable is not None and len(self.data) == 0 and len(reusable) == numPts
                and 'spots' not in kwargs
                and not any(isinstance(value, np.ndarray) and np.may_share_memory(value, reusable)
                            for value in kwargs.values())):
            # setData with as many spots as before (sliding window): reset the previous
            # array in place. Nothing outside the item refers to it (no SpotItem, no
            # getData view, no input sharing its memory).
            self.data = reusable
            self._resetSpotArray(self.data)
        else:
            ## Extend record array
            oldData = self.data
            self.data = self._newSpotArray(len(oldData) + numPts)
            self._dataShared = False

            self.data[:len(oldData)] = oldData
            #for i in range(len(oldData)):
                #oldData[i]['item']._data = self.data[i]  ## Make sure items have proper reference to new array

        newData = self.data[len(self.data) - numPts:]

        # style columns of the new spots that keep their initial unset value
        if 'spots' in kwargs:
            unset = frozenset()
        else:
            unset = frozenset(k for k in ('symbol', 'size', 'pen', 'brush')
                              if not isinstance(kwargs.get(k), (list, np.ndarray)))

        if 'spots' in kwargs:
            spots = kwargs['spots']
            for i in range(len(spots)):
                spot = spots[i]
                for k in spot:
                    if k == 'pos':
                        pos = spot[k]
                        if isinstance(pos, QtCore.QPointF):
                            x,y = pos.x(), pos.y()
                        else:
                            x,y = pos[0], pos[1]
                        newData[i]['x'] = x
                        newData[i]['y'] = y
                    elif k == 'pen':
                        newData[i][k] = _mkPen(spot[k])
                    elif k == 'brush':
                        newData[i][k] = _mkBrush(spot[k])
                    elif k in ['x', 'y', 'size', 'symbol', 'data']:
                        newData[i][k] = spot[k]
                    else:
                        raise Exception("Unknown spot parameter: %s" % k)
        elif 'y' in kwargs:
            newData['x'] = kwargs['x']
            newData['y'] = kwargs['y']

        if 'name' in kwargs:
            self.opts['name'] = kwargs['name']
        if 'pxMode' in kwargs:
            self.setPxMode(kwargs['pxMode'])
        if 'antialias' in kwargs:
            self.opts['antialias'] = kwargs['antialias']
        if 'compositionMode' in kwargs:
            self.opts['compositionMode'] = kwargs['compositionMode']
            if self.opts['useDeviceCache']:
                self._applyDeviceCache()
        if 'hoverable' in kwargs:
            self.opts['hoverable'] = bool(kwargs['hoverable'])
        if 'tip' in kwargs:
            self.opts['tip'] = kwargs['tip']
        if 'useCache' in kwargs:
            self.opts['useCache'] = kwargs['useCache']
        if 'useDeviceCache' in kwargs:
            self.setUseDeviceCache(kwargs['useDeviceCache'])

        ## Set any extra parameters provided in keyword arguments
        self._styleCodes = {}
        try:
            for k in ['pen', 'brush', 'symbol', 'size']:
                if k in kwargs:
                    setMethod = getattr(self, 'set' + k[0].upper() + k[1:])
                    setMethod(kwargs[k], update=False, dataSet=newData, mask=kwargs.get('mask', None))
                kh = 'hover' + k.title()
                if kh in kwargs:
                    vh = kwargs[kh]
                    if k == 'pen':
                        vh = _mkPen(vh)
                    elif k == 'brush':
                        vh = _mkBrush(vh)
                    self.opts[kh] = vh
        finally:
            known = self._styleCodes
            self._styleCodes = None
        if 'data' in kwargs:
            self.setPointData(kwargs['data'], dataSet=newData)

        self.prepareGeometryChange()
        self.informViewBoundsChanged()
        self.bounds = [None, None]
        self.invalidate()
        self._updateSpots(newData, unset=unset, known=known, new=True)
        self.sigPlotChanged.emit(self)

    def _newSpotArray(self, n: int) -> np.ndarray:
        """
        Allocate the structured array of ``n`` spots in their initial state.

        Object fields are ``None``, ``size`` is -1 (default size), ``visible`` is True,
        and the other fields are zero. ``np.zeros`` followed by a copy of the initial
        record is several times faster than ``np.empty``, which initializes object
        fields record by record.

        Parameters
        ----------
        n : int
            Number of spots.

        Returns
        -------
        numpy.ndarray
            New structured array with the dtype of ``self.data``.
        """
        data = np.zeros(n, dtype=self.data.dtype)
        data[...] = self._initialSpot
        return data

    def _resetSpotArray(self, data: np.ndarray) -> None:
        """
        Put a spot array back in the initial state of :meth:`_newSpotArray`.

        All the fields, positions included, are written in a single pass over the
        records, which is about 2.5 times faster than one pass per field at 1e6 spots.

        Parameters
        ----------
        data : numpy.ndarray
            Structured spot array, modified in place.
        """
        data[...] = self._initialSpot

    @staticmethod
    def _makeInitialSpot(dtype: np.dtype) -> np.ndarray:
        """
        Build the record of a spot in its initial state.

        Parameters
        ----------
        dtype : numpy.dtype
            Structured dtype of the spot array.

        Returns
        -------
        numpy.ndarray
            One-element array: object fields ``None``, ``size`` -1 (default size),
            ``visible`` True, other fields zero.
        """
        spot = np.zeros(1, dtype=dtype)
        for name, (fieldType, *_) in dtype.fields.items():
            if fieldType.hasobject:
                spot[name] = None
        spot['size'] = -1
        spot['visible'] = True
        return spot

    def invalidate(self) -> None:
        """
        Clear any cached drawing state and schedule a repaint.

        Must be called (directly, or through :meth:`updateSpots`) after modifying the
        spot array ``self.data`` in place.
        """
        self.picture = None
        self._paintCache = None
        self.update()

    def _spotArrays(self) -> _SpotArrays:
        """
        Return the contiguous copies of the spot fields, building them if needed.

        Returns
        -------
        _SpotArrays
            Arrays valid until the next :meth:`invalidate`.
        """
        if self._paintCache is None:
            self._paintCache = _SpotArrays(self.data)
        return self._paintCache

    def getData(self):
        self._dataShared = True  # views of self.data are handed out
        return self.data['x'], self.data['y']

    def implements(self, interface=None):
        ints = ['plotData']
        if interface is None:
            return ints
        return interface in ints

    def name(self):
        return self.opts.get('name', None)

    def _resetSymbols(self, dataSet: np.ndarray) -> None:
        """
        Mark spots for a new symbol atlas lookup after a style change.

        Nothing is written while :meth:`addPoints` applies the style arguments: its
        spots are new and not looked up yet, and a pass over the records costs about
        20 ms at 1e6 spots.

        Parameters
        ----------
        dataSet : numpy.ndarray
            Structured array (or view) of the spots whose style changed.
        """
        if self._styleCodes is None:
            dataSet['sourceRect'] = 0
            self._paintCache = None

    def _setStyleColumn(self, name: str, dataSet: np.ndarray, objects: np.ndarray,
                        reps: list | None, codes: np.ndarray | None) -> None:
        """
        Store per-spot pens or brushes, and record their grouping when it is known.

        During :meth:`addPoints`, a grouping known from the conversion of the input
        (one object per distinct color) is handed over to :meth:`_updateSpots`, which
        then does not need to group the column again.

        Parameters
        ----------
        name : str
            Column name, ``'pen'`` or ``'brush'``.
        dataSet : numpy.ndarray
            Structured array (or view) of the spots.
        objects : numpy.ndarray
            Object array of one pen or brush per spot.
        reps : list or None
            The distinct objects of ``objects``, if known.
        codes : numpy.ndarray or None
            Index in ``reps`` of each spot, if known.
        """
        dataSet[name] = objects
        if self._styleCodes is not None:
            if codes is None:
                self._styleCodes.pop(name, None)
            else:
                self._styleCodes[name] = (reps, codes)

    def setPen(self, *args, **kwargs) -> None:
        """
        Set the pen(s) used to draw the outline around each spot.

        If a list or array is provided, then the pen for each spot will be set
        separately; equal hashable specifications (such as color tuples) share one
        ``QPen``. Otherwise, the arguments are passed to :func:`~pyqtgraph.mkPen` and
        used as the default pen for all spots which do not have a pen explicitly set.

        The fastest way to give one color per spot is an ``(N, 4)`` (or ``(N, 3)``)
        numeric array of RGBA (RGB) values in the range 0-255, preferably of
        ``uint8`` dtype: one cosmetic pen of width 1 is made per distinct color.

        Parameters
        ----------
        *args
            A list or array holding one pen specification per spot, an ``(N, 4)`` or
            ``(N, 3)`` array of colors, or the arguments of :func:`~pyqtgraph.mkPen`.
        **kwargs
            Keyword arguments of :func:`~pyqtgraph.mkPen`, and the internal options
            ``update`` (bool, default True: update the spots now), ``dataSet``
            (structured array of the spots to change, default all spots) and ``mask``
            (selection applied to a per-spot list or array).
        """
        update = kwargs.pop('update', True)
        dataSet = kwargs.pop('dataSet', self.data)

        if len(args) == 1 and (isinstance(args[0], np.ndarray) or isinstance(args[0], list)):
            pens = args[0]
            if 'mask' in kwargs and kwargs['mask'] is not None:
                pens = pens[kwargs['mask']]
            if len(pens) != len(dataSet):
                raise Exception("Number of pens does not match number of points (%d != %d)" % (len(pens), len(dataSet)))
            converted = (_colorArrayToObjects(pens, fn.mkPen)
                         or _mkMany(pens, QtGui.QPen, fn.mkPen))
            self._setStyleColumn('pen', dataSet, *converted)
        else:
            self.opts['pen'] = _mkPen(*args, **kwargs)

        self._resetSymbols(dataSet)
        if update:
            self.updateSpots(dataSet)

    def setBrush(self, *args, **kwargs) -> None:
        """
        Set the brush(es) used to fill the interior of each spot.

        If a list or array is provided, then the brush for each spot will be set
        separately; equal hashable specifications (such as color tuples) share one
        ``QBrush``. Otherwise, the arguments are passed to :func:`~pyqtgraph.mkBrush`
        and used as the default brush for all spots which do not have a brush
        explicitly set.

        The fastest way to give one color per spot, for instance buy and sell
        colors of trades, is an ``(N, 4)`` (or ``(N, 3)``) numeric array of RGBA
        (RGB) values in the range 0-255, preferably of ``uint8`` dtype: one brush is
        made per distinct color. An object array built from a palette, such as
        ``np.array(palette, dtype=object)[indices]`` with ``palette`` a list of
        ``QBrush``, is nearly as fast, since spots are grouped by brush object.

        Parameters
        ----------
        *args
            A list or array holding one brush specification per spot, an ``(N, 4)``
            or ``(N, 3)`` array of colors, or the arguments of
            :func:`~pyqtgraph.mkBrush`.
        **kwargs
            Keyword arguments of :func:`~pyqtgraph.mkBrush`, and the internal options
            ``update`` (bool, default True: update the spots now), ``dataSet``
            (structured array of the spots to change, default all spots) and ``mask``
            (selection applied to a per-spot list or array).
        """
        update = kwargs.pop('update', True)
        dataSet = kwargs.pop('dataSet', self.data)

        if len(args) == 1 and (isinstance(args[0], np.ndarray) or isinstance(args[0], list)):
            brushes = args[0]
            if 'mask' in kwargs and kwargs['mask'] is not None:
                brushes = brushes[kwargs['mask']]
            if len(brushes) != len(dataSet):
                raise Exception("Number of brushes does not match number of points (%d != %d)" % (len(brushes), len(dataSet)))
            converted = (_colorArrayToObjects(brushes, fn.mkBrush)
                         or _mkMany(brushes, QtGui.QBrush, fn.mkBrush))
            self._setStyleColumn('brush', dataSet, *converted)
        else:
            self.opts['brush'] = _mkBrush(*args, **kwargs)

        self._resetSymbols(dataSet)
        if update:
            self.updateSpots(dataSet)

    def setSymbol(self, symbol: object, update: bool = True, dataSet: np.ndarray | None = None,
                  mask: np.ndarray | None = None) -> None:
        """Set the symbol(s) used to draw each spot.
        If a list or array is provided, then the symbol for each spot will be set separately.
        Otherwise, the argument will be used as the default symbol for
        all spots which do not have a symbol explicitly set.

        Parameters
        ----------
        symbol : str, int, QtGui.QPainterPath, list or numpy.ndarray
            A symbol, or one symbol per spot.
        update : bool, default True
            Update the spots now.
        dataSet : numpy.ndarray, optional
            Structured array of the spots to change; defaults to all spots.
        mask : numpy.ndarray, optional
            Selection applied to a per-spot list or array.

        **Supported symbols:**

        * 'o'  circle (default)
        * 's'  square
        * 't'  triangle
        * 'd'  diamond
        * '+'  plus
        * 't1' triangle pointing upwards
        * 't2'  triangle pointing right side
        * 't3'  triangle pointing left side
        * 'p'  pentagon
        * 'h'  hexagon
        * 'star'
        * '|' vertical line
        * '_' horizontal line
        * 'x'  cross
        * 'arrow_up'
        * 'arrow_right'
        * 'arrow_down'
        * 'arrow_left'
        * 'crosshair'
        * any QPainterPath to specify custom symbol shapes.

        """
        if dataSet is None:
            dataSet = self.data

        if isinstance(symbol, np.ndarray) or isinstance(symbol, list):
            symbols = symbol
            if mask is not None:
                symbols = symbols[mask]
            if len(symbols) != len(dataSet):
                raise Exception("Number of symbols does not match number of points (%d != %d)" % (len(symbols), len(dataSet)))
            dataSet['symbol'] = symbols
        else:
            self.opts['symbol'] = symbol
            self._spotPixmap = None

        self._resetSymbols(dataSet)
        if update:
            self.updateSpots(dataSet)

    def setSize(self, size: float | list | np.ndarray, update: bool = True,
                dataSet: np.ndarray | None = None, mask: np.ndarray | None = None) -> None:
        """
        Set the size(s) used to draw each spot.

        If a list or array is provided, then the size for each spot will be set
        separately. Otherwise, the argument will be used as the default size for all
        spots which do not have a size explicitly set. A float array is the fastest
        way to give one size per spot (for instance proportional to trade volumes);
        in pixel mode, non-integral sizes are rounded to a quarter of a device pixel.

        Parameters
        ----------
        size : float, list or numpy.ndarray
            A size, or one size per spot (-1 for the default size).
        update : bool, default True
            Update the spots now.
        dataSet : numpy.ndarray, optional
            Structured array of the spots to change; defaults to all spots.
        mask : numpy.ndarray, optional
            Selection applied to a per-spot list or array.
        """
        if dataSet is None:
            dataSet = self.data

        if isinstance(size, np.ndarray) or isinstance(size, list):
            sizes = size
            if mask is not None:
                sizes = sizes[mask]
            if len(sizes) != len(dataSet):
                raise Exception("Number of sizes does not match number of points (%d != %d)" % (len(sizes), len(dataSet)))
            dataSet['size'] = sizes
        else:
            self.opts['size'] = size
            self._spotPixmap = None

        self._resetSymbols(dataSet)
        if update:
            self.updateSpots(dataSet)


    def setPointsVisible(self, visible: bool | list | np.ndarray, update: bool = True,
                         dataSet: np.ndarray | None = None,
                         mask: np.ndarray | None = None) -> None:
        """
        Set whether or not each spot is visible.

        If a list or array is provided, then the visibility for each spot will be set
        separately. Otherwise, the argument will be used for all spots.

        Parameters
        ----------
        visible : bool, list or numpy.ndarray
            Visibility of all spots, or of each spot.
        update : bool, default True
            Update the spots now.
        dataSet : numpy.ndarray, optional
            Structured array of the spots to change; defaults to all spots.
        mask : numpy.ndarray, optional
            Selection applied to a per-spot list or array.
        """
        self._paintCache = None
        if dataSet is None:
            dataSet = self.data

        if isinstance(visible, np.ndarray) or isinstance(visible, list):
            visibilities = visible
            if mask is not None:
                visibilities = visibilities[mask]
            if len(visibilities) != len(dataSet):
                raise Exception("Number of visibilities does not match number of points (%d != %d)" % (len(visibilities), len(dataSet)))
            dataSet['visible'] = visibilities
        else:
            dataSet['visible'] = visible

        dataSet['sourceRect'] = 0
        if update:
            self.updateSpots(dataSet)
        
    def setPointData(self, data, dataSet=None, mask=None):
        if dataSet is None:
            dataSet = self.data

        if isinstance(data, np.ndarray) or isinstance(data, list):
            if mask is not None:
                data = data[mask]
            if len(data) != len(dataSet):
                raise Exception("Length of meta data does not match number of points (%d != %d)" % (len(data), len(dataSet)))

        ## Bug: If data is a numpy record array, then items from that array must be copied to dataSet one at a time.
        ## (otherwise they are converted to tuples and thus lose their field names.
        if isinstance(data, np.ndarray) and (data.dtype.fields is not None)and len(data.dtype.fields) > 1:
            for i, rec in enumerate(data):
                dataSet['data'][i] = rec
        else:
            dataSet['data'] = data

    def setPxMode(self, mode):
        if self.opts['pxMode'] == mode:
            return

        self.opts['pxMode'] = mode
        self.invalidate()

    def setUseDeviceCache(self, enabled: bool) -> None:
        """
        Keep the rendered scatter plot in a pixmap of the view, or stop doing so.

        This sets the ``DeviceCoordinateCache`` cache mode of the item (see
        ``QGraphicsItem.setCacheMode``). Qt then paints the item once into a
        viewport-sized pixmap and copies the pixmap when other items change: a
        crosshair or cursor line moving over a large scatter plot no longer repaints
        it (about 23 ms to 1.5 ms per mouse move with 1e5 spots), and pans repaint only
        the newly exposed band.

        It is off by default because it does not pay off when the scatter plot itself
        changes at every frame or the view is zoomed continuously: the item is then
        painted into the pixmap and the pixmap copied, which costs a few percent more
        (about +8 % at 1e5 spots), and each view holds one more pixmap of its size.
        Composition modes other than ``CompositionMode_SourceOver`` would compose the
        spots with a transparent pixmap instead of the scene, so the cache is not used
        with them. Qt's device cache is not aware of fractional device pixel ratios
        (e.g. a 125 % display scale), where the cached rendering visibly differs from
        a direct paint: the cache is not used either when a screen has a fractional
        device pixel ratio at the time the option is set. (On curves, such a cache was
        measured to slow a crosshair down.)

        Parameters
        ----------
        enabled : bool
            Whether to cache the rendered item.
        """
        self.opts['useDeviceCache'] = bool(enabled)
        self._applyDeviceCache()

    def _applyDeviceCache(self) -> None:
        """Set the cache mode of the item from the ``useDeviceCache`` and composition mode options."""
        cmode = self.opts['compositionMode']
        cached = (
            self.opts['useDeviceCache']
            and cmode in (None, QtGui.QPainter.CompositionMode.CompositionMode_SourceOver)
            and _screensHaveIntegralPixelRatio()
        )
        self.setCacheMode(QtWidgets.QGraphicsItem.CacheMode.DeviceCoordinateCache if cached
                          else QtWidgets.QGraphicsItem.CacheMode.NoCache)

    def updateSpots(self, dataSet: np.ndarray | None = None) -> None:
        """
        Look up the symbols of spots whose style changed, and schedule a repaint.

        Parameters
        ----------
        dataSet : numpy.ndarray, optional
            Structured array (or view) of the spots to update; defaults to all spots.
        """
        self._updateSpots(dataSet)

    def _updateSpots(self, dataSet: np.ndarray | None = None,
                     unset: frozenset[str] = frozenset(),
                     known: dict[str, tuple[list, np.ndarray]] | None = None,
                     new: bool = False) -> None:
        """
        Implementation of :meth:`updateSpots`.

        The spots are grouped by style (see :meth:`_uniqueStyles`): the symbol atlas
        is queried once per distinct style, and the atlas positions are then
        broadcast to the spots with a single vectorized assignment.

        Parameters
        ----------
        dataSet : numpy.ndarray, optional
            Structured array (or view) of the spots to update; defaults to all spots.
        unset : frozenset of str, default frozenset()
            Style columns known to hold only their initial unset value in
            ``dataSet``, which are then not scanned.
        known : dict, optional
            Known grouping ``(reps, codes)`` of style columns of ``dataSet``, keyed by
            column name (see :meth:`_setStyleColumn`).
        new : bool, default False
            True when all the spots of ``dataSet`` are new (see :meth:`addPoints`): they
            are all looked up, without scanning their atlas positions first.
        """
        profiler = debug.Profiler()  # noqa: profiler prints on GC
        if dataSet is None:
            dataSet = self.data

        invalidate = False
        if self.opts['pxMode'] and self.opts['useCache']:
            if new:
                lookup, idx = len(dataSet) > 0, None
            else:
                mask = dataSet['sourceRect']['w'] == 0
                lookup = bool(np.any(mask))
                idx = None if lookup and mask.all() else mask
            if lookup:
                invalidate = True
                styles, inverse = self._uniqueStyles(dataSet, idx, unset,
                                                     known if idx is None else None)
                sourceRect = dataSet['sourceRect']
                coords = np.array(self.fragmentAtlas[styles], dtype=sourceRect.dtype)
                coords = coords[0] if inverse is None else coords[inverse]
                if idx is None:
                    sourceRect[...] = coords
                else:
                    sourceRect[idx] = coords

            self._maybeRebuildAtlas()
        else:
            invalidate = True

        self._updateMaxSpotSizes(data=dataSet)

        if invalidate:
            self.invalidate()

    def _uniqueStyles(self, data: np.ndarray, idx: np.ndarray | None = None,
                      unset: frozenset[str] = frozenset(),
                      known: dict[str, tuple[list, np.ndarray]] | None = None
                      ) -> tuple[list[tuple], np.ndarray | None]:
        """
        Group spots by effective style.

        The effective style is the one :meth:`_style` gives (item defaults for unset
        entries, hover style for hovered spots), with sizes quantized as in the
        symbol atlas (see :func:`_quantizeSize`). Each style column is coded with
        numpy, by object identity for symbols, pens and brushes and by value for
        sizes; the codes are then combined. Only the distinct combinations are turned
        into Python tuples.

        Parameters
        ----------
        data : numpy.ndarray
            Structured spot array.
        idx : numpy.ndarray, optional
            Boolean mask selecting the spots; defaults to all spots.
        unset : frozenset of str, default frozenset()
            Style columns known to hold only their initial unset value.
        known : dict, optional
            Known grouping ``(reps, codes)`` of style columns of all the spots of
            ``data``, keyed by column name; only used when ``idx`` is ``None``.

        Returns
        -------
        styles : list of tuple
            The distinct ``(symbol, size, pen, brush)`` styles. Objects that are
            distinct but equal give distinct styles, which the atlas merges.
        inverse : numpy.ndarray or None
            Index in ``styles`` of each selected spot, or ``None`` when all the
            spots have the style ``styles[0]``.
        """
        n = len(data) if idx is None else int(np.count_nonzero(idx))
        if n == 0:
            return [], np.zeros(0, dtype=np.intp)
        hovered = None
        if self.opts['hoverable']:
            hovered = data['hovered'] if idx is None else data['hovered'][idx]
            if not hovered.any():
                hovered = None
        step = 4 * self.fragmentAtlas.devicePixelRatio()
        if known is None or idx is not None:
            known = {}

        reps = []   # per style column: the distinct values
        codes = []  # per style column: index in reps of each spot, or None if uniform
        for opt in ('symbol', 'size', 'pen', 'brush'):
            default = self.opts[opt]
            if opt in unset:
                colReps, colCodes = [default], None
            elif opt in known:
                colReps, colCodes = known[opt]
                if len(colReps) == 1:
                    colCodes = None
            elif opt == 'size':
                col = data['size'] if idx is None else data['size'][idx]
                col = np.where(col == _DEFAULT_STYLE['size'], default, col)
                colReps, colCodes = _groupValues(_quantizeSizes(col, step))
            else:
                col = data[opt] if idx is None else data[opt][idx]
                colReps, colCodes = _groupObjects(col.tolist())
                colReps = [default if rep is None else rep for rep in colReps]
                if opt == 'symbol':
                    colReps, colCodes = _mergeEqual(colReps, colCodes)

            hoverValue = self.opts['hover' + opt.title()]
            if hovered is not None and hoverValue != _DEFAULT_STYLE[opt]:
                if opt == 'size':
                    hoverValue = _quantizeSize(hoverValue, step)
                if colCodes is None:
                    colCodes = np.zeros(n, dtype=np.intp)
                else:
                    colCodes = colCodes.copy()  # known codes may be shared
                colCodes[hovered] = len(colReps)
                colReps = colReps + [hoverValue]
            reps.append(colReps)
            codes.append(colCodes)

        active = [i for i, colCodes in enumerate(codes) if colCodes is not None]
        if not active:
            return [tuple(colReps[0] for colReps in reps)], None
        if len(active) == 1:
            # a single varying column: its codes index the styles directly
            col = active[0]
            styles = [tuple(rep if i == col else reps[i][0] for i in range(4))
                      for rep in reps[col]]
            return styles, codes[col]

        combined = codes[active[0]].astype(np.int64)
        radix = len(reps[active[0]])
        for col in active[1:]:
            size = len(reps[col])
            if radix * size >= 2 ** 62:
                # renumber the combinations seen so far to avoid an overflow
                _, combined = np.unique(combined, return_inverse=True)
                radix = int(combined.max()) + 1
            combined = combined * size + codes[col]
            radix *= size
        unique, inverse = np.unique(combined, return_inverse=True)
        styles = []
        for i in _representatives(inverse, len(unique)).tolist():
            styles.append(tuple(colReps[0] if colCodes is None else colReps[colCodes[i]]
                                for colReps, colCodes in zip(reps, codes)))
        return styles, inverse

    def _atlasEntriesInUse(self) -> int:
        """
        Count the symbol atlas entries used by the spots.

        Returns
        -------
        int
            Number of distinct atlas positions referenced by the spots.
        """
        sr = self.data['sourceRect']
        sr = sr[sr['w'] != 0]
        return len(np.unique((sr['x'].astype(np.int64) << 32) | sr['y'].astype(np.int64)))

    def _maybeRebuildAtlas(self, threshold: int = 4, minlen: int = 1000) -> None:
        """
        Rebuild the symbol atlas with the styles in use when it holds too many entries.

        The atlas is rebuilt when it holds more than ``minlen`` entries and more than
        ``threshold`` entries per spot, or when it holds more than
        ``self._atlasMaxEntries`` entries of which less than half are in use. The
        second rule bounds the atlas when the styles keep changing, for instance with
        colors that follow a live value; requiring half of the entries to be unused
        avoids rebuilding at every update when the styles in use alone exceed the cap.

        Parameters
        ----------
        threshold : int, default 4
            Maximum number of atlas entries per spot.
        minlen : int, default 1000
            Number of atlas entries below which the per-spot rule does not apply.
        """
        n = len(self.fragmentAtlas)
        rebuild = n > minlen and n > threshold * len(self.data)
        if not rebuild and n > self._atlasMaxEntries:
            rebuild = n > 2 * self._atlasEntriesInUse()
        if rebuild:
            self.fragmentAtlas.rebuild(self._uniqueStyles(self.data)[0])
            self.data['sourceRect'] = 0
            self.updateSpots()

    def _style(self, opts: list[str], data: np.ndarray | None = None,
               idx: np.ndarray | slice | None = None,
               scale: float | None = None) -> Iterator[np.ndarray]:
        """
        Generate the effective style columns of a set of spots.

        Unset entries (``None`` for ``symbol``, ``pen`` and ``brush``, ``-1`` for
        ``size``) are replaced by the item default, and hovered spots take the hover
        style when one is set.

        Parameters
        ----------
        opts : list of str
            Names of the style columns to generate, among ``'symbol'``, ``'size'``,
            ``'pen'`` and ``'brush'``.
        data : numpy.ndarray, optional
            Structured spot array; defaults to ``self.data``.
        idx : numpy.ndarray or slice, optional
            Boolean mask or index selecting the spots; defaults to all spots.
        scale : float, optional
            Factor applied to the ``size`` column.

        Yields
        ------
        numpy.ndarray
            One new array per name of ``opts``, in the same order.
        """
        if data is None:
            data = self.data

        if idx is None:
            idx = np.s_[:]

        for opt in opts:
            col = data[opt][idx]
            if col.base is not None:
                col = col.copy()

            if self.opts['hoverable']:
                val = self.opts['hover' + opt.title()]
                if val != _DEFAULT_STYLE[opt]:
                    col[data['hovered'][idx]] = val

            default = _DEFAULT_STYLE[opt]
            if default is None:
                # identity test: np.equal would call QPen/QBrush.__eq__ per element
                col[_isNoneMask(col)] = self.opts[opt]
            else:
                col[np.equal(col, default)] = self.opts[opt]

            if opt == 'size' and scale is not None:
                col *= scale

            yield col

    def _updateMaxSpotSizes(self, **kwargs) -> None:
        """
        Update the maximum spot sizes, which pad the data bounds and bounding rect.

        When they change, the geometry change is announced to the scene and the data
        bounds change to the view (which caches the data bounds of its items).

        Parameters
        ----------
        **kwargs
            Arguments of :meth:`_style` selecting the spots measured (``data``,
            ``idx``).
        """
        if self.opts['pxMode'] and self.opts['useCache']:
            w, pw = 0, self.fragmentAtlas.maxWidth
        else:
            w, pw = max(itertools.chain([(self._maxSpotWidth, self._maxSpotPxWidth)],
                              self._measureSpotSizes(**kwargs)))
        changed = (w, pw) != (self._maxSpotWidth, self._maxSpotPxWidth)
        boundsChanged = w != self._maxSpotWidth
        if changed:
            self.prepareGeometryChange()  # before the bounding rect changes
        self._maxSpotWidth = w
        self._maxSpotPxWidth = pw
        self.bounds = [None, None]
        if boundsChanged:
            self.informViewBoundsChanged()

    def _measureSpotSizes(self, **kwargs):
        """Generate pairs (width, pxWidth) for spots in data"""
        styles = zip(*self._style(['size', 'pen'], **kwargs))

        if self.opts['pxMode']:
            for size, pen in styles:
                yield 0, size + pen.widthF()
        else:
            for size, pen in styles:
                if pen.isCosmetic():
                    yield size, pen.widthF()
                else:
                    yield size + pen.widthF(), 0

    def clear(self):
        """Remove all spots from the scatter plot"""
        #self.clearItems()
        self._maxSpotWidth = 0
        self._maxSpotPxWidth = 0
        self.data = np.empty(0, dtype=self.data.dtype)
        self.bounds = [None, None]
        self.invalidate()

    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        if frac >= 1.0 and orthoRange is None and self.bounds[ax] is not None:
            return self.bounds[ax]

        #self.prepareGeometryChange()
        if self.data is None or len(self.data) == 0:
            return (None, None)

        if ax == 0:
            d = self.data['x']
            d2 = self.data['y']
        elif ax == 1:
            d = self.data['y']
            d2 = self.data['x']
        else:
            raise ValueError("Invalid axis value")

        if orthoRange is not None:
            mask = (d2 >= orthoRange[0]) * (d2 <= orthoRange[1])
            d = d[mask]

            if d.size == 0:
                return (None, None)

        if frac >= 1.0:
            bounds = (np.nanmin(d) - self._maxSpotWidth*0.7072, np.nanmax(d) + self._maxSpotWidth*0.7072)
            if orthoRange is None:
                # only the full-range bounds are cached
                self.bounds[ax] = bounds
            return bounds
        elif frac <= 0.0:
            raise Exception("Value for parameter 'frac' must be > 0. (got %s)" % str(frac))
        else:
            mask = np.isfinite(d)
            d = d[mask]
            return np.percentile(d, [50 * (1 - frac), 50 * (1 + frac)])

    def pixelPadding(self):
        return self._maxSpotPxWidth*0.7072

    def boundingRect(self):
        (xmn, xmx) = self.dataBounds(ax=0)
        (ymn, ymx) = self.dataBounds(ax=1)
        if xmn is None or xmx is None:
            xmn = 0
            xmx = 0
        if ymn is None or ymx is None:
            ymn = 0
            ymx = 0

        px = py = 0.0
        pxPad = self.pixelPadding()
        if pxPad > 0:
            # determine length of pixel in local x, y directions
            px, py = self.pixelVectors()
            try:
                px = 0 if px is None else px.length()
            except OverflowError:
                px = 0
            try:
                py = 0 if py is None else py.length()
            except OverflowError:
                py = 0

            # return bounds expanded by pixel size
            px *= pxPad
            py *= pxPad
        return QtCore.QRectF(xmn-px, ymn-py, (2*px)+xmx-xmn, (2*py)+ymx-ymn)

    def viewTransformChanged(self):
        # The cached data bounds do not depend on the view (pixel padding is applied
        # in boundingRect), so they are kept: recomputing them is O(N).
        self.prepareGeometryChange()
        GraphicsObject.viewTransformChanged(self)

    def setExportMode(self, *args, **kwargs):
        GraphicsObject.setExportMode(self, *args, **kwargs)
        self.invalidate()

    @debug.warnOnException  ## raising an exception here causes crash
    def paint(self, p, option, widget):
        profiler = debug.Profiler()
        cmode = self.opts.get('compositionMode', None)
        if cmode is not None:
            p.setCompositionMode(cmode)
        #p.setPen(fn.mkPen('r'))
        #p.drawRect(self.boundingRect())

        if self._exportOpts is not False:
            aa = self._exportOpts.get('antialias', True)
            scale = self._exportOpts.get('resolutionScale', 1.0)  ## exporting to image; pixel resolution may have changed
        else:
            aa = self.opts['antialias']
            scale = 1.0

        if self.opts['pxMode'] is True:
            if self.opts['useCache'] and self._exportOpts is False:
                # Draw symbols from pre-rendered atlas

                dpr = self.fragmentAtlas.devicePixelRatio()
                if widget is not None and (dpr_new := widget.devicePixelRatioF()) != dpr:
                    # force a re-render if dpr changed
                    dpr = dpr_new
                    self.fragmentAtlas.setDevicePixelRatio(dpr)
                    self.fragmentAtlas.clear()
                    self.data['sourceRect'] = 0
                    self.updateSpots()

                # Cull points that are outside view, and map the others using the
                # painter's world transform so they are drawn with pixel-valued sizes
                self._prepareFragments(p.transform(), self.viewRect(), dpr)
                p.resetTransform()

                profiler('prep')
                drawargs = self._pixmapFragments.drawargs()
                p.drawPixmapFragments(*drawargs, self.fragmentAtlas.pixmap)
                profiler('draw')
            else:
                # Cull points that are outside view
                viewMask = self._maskAt(self.viewRect())

                # Map points using painter's world transform so they are drawn with pixel-valued sizes
                arrays = self._spotArrays()
                pts = np.empty((np.count_nonzero(viewMask), 2))
                _mapPoints(p.transform(), arrays.x[viewMask], arrays.y[viewMask], pts)
                p.resetTransform()

                # render each symbol individually
                p.setRenderHint(p.RenderHint.Antialiasing, aa)

                for pt, style in zip(
                        pts,
                        zip(*(self._style(['symbol', 'size', 'pen', 'brush'], idx=viewMask, scale=scale)))
                ):
                    p.resetTransform()
                    p.translate(*pt)
                    drawSymbol(p, *style)
        else:
            if self.picture is None:
                self.picture = QtGui.QPicture()
                p2 = QtGui.QPainter(self.picture)

                for x, y, style in zip(
                        self.data['x'],
                        self.data['y'],
                        zip(*self._style(['symbol', 'size', 'pen', 'brush'], scale=scale))
                ):
                    p2.resetTransform()
                    p2.translate(x, y)
                    drawSymbol(p2, *style)
                p2.end()

            p.setRenderHint(p.RenderHint.Antialiasing, aa)
            self.picture.play(p)

    def points(self):
        self._dataShared = True  # SpotItems and the 'item' column view are handed out
        m = np.equal(self.data['item'], None)
        for i in np.argwhere(m)[:, 0]:
            rec = self.data[i]
            if rec['item'] is None:
                rec['item'] = SpotItem(rec, self, i)
        return self.data['item']

    def _pointsForIndices(self, idx: np.ndarray) -> np.ndarray:
        """
        Return the SpotItems of the given spots, creating only the missing ones.

        Unlike :meth:`points`, which creates a SpotItem for every spot, this method only
        creates those of ``idx``. Hit tests use it so that their cost does not depend on
        the total number of spots.

        Parameters
        ----------
        idx : numpy.ndarray
            One-dimensional integer array of spot indices.

        Returns
        -------
        numpy.ndarray
            Object array of :class:`SpotItem`, in the order of ``idx``.
        """
        items = self.data['item']
        if len(idx):
            self._dataShared = True  # the SpotItems are views of self.data
        for i in idx.tolist():
            if items[i] is None:
                items[i] = SpotItem(self.data[i], self, i)
        return items[idx]

    def pointsAt(self, pos: QtCore.QPointF | QtCore.QRectF) -> np.ndarray:
        """
        Return the visible spots overlapping a position or a rectangle.

        Only the SpotItems of the spots found are created.

        Parameters
        ----------
        pos : QtCore.QPointF or QtCore.QRectF
            Position or rectangle in item coordinates.

        Returns
        -------
        numpy.ndarray
            Object array of :class:`SpotItem`, in reverse data order (the spot drawn on
            top first).
        """
        return self._pointsForIndices(np.flatnonzero(self._maskAt(pos))[::-1])

    def _prepareFragments(self, transform: QtGui.QTransform, viewRect: QtCore.QRectF,
                          dpr: float) -> None:
        """
        Fill the pixmap fragment array with the spots visible in a view rectangle.

        Spots are culled with the test of :meth:`_maskAt`, on contiguous arrays and
        with a scalar margin when all the symbols have the same size; culling is
        skipped altogether when all the spots lie inside the view. (A scalar margin
        for varied sizes would keep more spots, which are not always invisible: the
        test under-estimates the extent of symbols in rotated items.) Only the kept
        spots are mapped to device coordinates.

        Each pass over the fragment array (80 bytes per row) touches all of its
        memory, so the columns are computed in a contiguous block and copied in one
        pass, and the columns already in place are not written again: the source
        rects when they are uniform or belong to the same spots as in the previous
        paint, and the constant scale, rotation and opacity columns.

        Parameters
        ----------
        transform : QtGui.QTransform
            Item to device transform of the painter.
        viewRect : QtCore.QRectF
            Visible area, in item coordinates.
        dpr : float
            Device pixel ratio of the symbol atlas.
        """
        left, right, top, bottom = self._rectEdges(viewRect)
        arrays = self._spotArrays()
        x, y = arrays.x, arrays.y
        bounds = arrays.bounds
        if (arrays.allVisible and bounds is not None and bounds[0] > left
                and bounds[1] < right and bounds[2] > top and bounds[3] < bottom):
            mask = None  # all the spots are inside the view
        else:
            # the operations of _maskAt, which give the same values
            px, py = self._pixelLengths()
            mx = arrays.halfWidth * px
            my = arrays.halfHeight * py
            mask = x + mx > left
            mask &= x - mx < right
            mask &= y + my > top
            mask &= y - my < bottom
            if not arrays.allVisible:
                mask &= arrays.visible
            x, y = x[mask], y[mask]

        n = len(x)
        self._pixmapFragments.resize(n)
        if n == 0:
            self._fragmentRects = self._fragmentConstants = None
            return
        frags = self._pixmapFragments.ndarray()
        # The address identifies the buffer: a reallocated buffer is created while the
        # previous one, whose state is recorded, still exists, so it gets another
        # address (and the states are dropped when the buffer is emptied above).
        address = frags.__array_interface__['data'][0]

        if arrays.uniformRect is not None:
            rectKey = (address, 'uniform', arrays.uniformRect)
        elif mask is None:
            rectKey = (address, 'all', arrays.serial)
        else:
            rectKey = None  # rects of a subset of the spots: not reusable
        if self._fragmentRows(self._fragmentRects, rectKey, n):
            positions = np.empty((n, 2))
            _mapPoints(transform, x, y, positions)
            frags[:, 0:2] = positions  # target center x, y
        else:
            block = np.empty((n, 6))
            _mapPoints(transform, x, y, block[:, 0:2])
            if arrays.uniformRect is not None:
                block[:, 2:6] = arrays.uniformRect  # sx, sy, sw, sh
            elif mask is None:
                block[:, 2:6] = arrays.sourceRect
            else:
                block[:, 2:6] = arrays.sourceRect[mask]
            frags[:, 0:6] = block
            self._fragmentRects = self._writtenRows(self._fragmentRects, rectKey, n)

        constKey = (address, dpr)
        if not self._fragmentRows(self._fragmentConstants, constKey, n):
            frags[:, 6:10] = [1/dpr, 1/dpr, 0.0, 1.0]   # scaleX, scaleY, rotation, opacity
            self._fragmentConstants = self._writtenRows(self._fragmentConstants, constKey, n)

    @staticmethod
    def _fragmentRows(state: tuple | None, key: tuple | None, n: int) -> bool:
        """
        Tell whether fragment columns are already in place for ``n`` rows.

        Parameters
        ----------
        state : tuple or None
            ``(key, rows)`` recorded when the columns were last written, or ``None``.
        key : tuple or None
            Key of the columns needed now (fragment buffer address and contents);
            ``None`` when they cannot be reused.
        n : int
            Number of fragments.

        Returns
        -------
        bool
            True if the first ``n`` rows already hold the columns.
        """
        return key is not None and state is not None and state[0] == key and state[1] >= n

    @staticmethod
    def _writtenRows(state: tuple | None, key: tuple | None, n: int) -> tuple | None:
        """
        Return the state to record after writing fragment columns for ``n`` rows.

        Parameters
        ----------
        state : tuple or None
            Previous ``(key, rows)`` state.
        key : tuple or None
            Key of the columns written; ``None`` when they cannot be reused.
        n : int
            Number of rows written.

        Returns
        -------
        tuple or None
            The new ``(key, rows)`` state; rows written earlier with the same key stay
            valid.
        """
        if key is None:
            return None
        if state is not None and state[0] == key:
            return key, max(n, state[1])
        return key, n

    @staticmethod
    def _rectEdges(obj: QtCore.QPointF | QtCore.QRectF) -> tuple[float, float, float, float]:
        """
        Return the edges of a point or a rectangle.

        Parameters
        ----------
        obj : QtCore.QPointF or QtCore.QRectF
            Point or rectangle.

        Returns
        -------
        tuple of float
            ``(left, right, top, bottom)``.

        Raises
        ------
        TypeError
            If ``obj`` is neither a ``QPointF`` nor a ``QRectF``.
        """
        if isinstance(obj, QtCore.QPointF):
            return obj.x(), obj.x(), obj.y(), obj.y()
        elif isinstance(obj, QtCore.QRectF):
            return obj.left(), obj.right(), obj.top(), obj.bottom()
        raise TypeError

    def _pixelLengths(self) -> tuple[float, float]:
        """
        Return the length of a device pixel in the local x and y directions.

        Returns
        -------
        tuple of float
            ``(px, py)``, 0 where unknown.
        """
        px, py = self.pixelVectors()
        try:
            px = 0 if px is None else px.length()
        except OverflowError:
            px = 0
        try:
            py = 0 if py is None else py.length()
        except OverflowError:
            py = 0
        return px, py

    def _maskAt(self, obj: QtCore.QPointF | QtCore.QRectF) -> np.ndarray:
        """
        Return a boolean mask indicating all points that overlap obj, a QPointF or QRectF.

        Parameters
        ----------
        obj : QtCore.QPointF or QtCore.QRectF
            Point or rectangle, in item coordinates.

        Returns
        -------
        numpy.ndarray
            Boolean mask over the spots, ``True`` for the visible spots overlapping
            ``obj``.
        """
        l, r, t, b = self._rectEdges(obj)
        arrays = self._spotArrays()

        if self.opts['pxMode'] and self.opts['useCache']:
            w = arrays.halfWidth
            h = arrays.halfHeight
        else:
            s, = self._style(['size'])
            w = s / 2
            h = s / 2

        if self.opts['pxMode']:
            # determine length of pixel in local x, y directions
            px, py = self._pixelLengths()
            w = w * px  # not in place: w may be a cached array
            h = h * py

        return (arrays.visible
                & (arrays.x + w > l)
                & (arrays.x - w < r)
                & (arrays.y + h > t)
                & (arrays.y - h < b))

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            pts = self.pointsAt(ev.pos())
            if len(pts) > 0:
                self.ptsClicked = pts
                ev.accept()
                self.sigClicked.emit(self, self.ptsClicked, ev)
            else:
                #print "no spots"
                ev.ignore()
        else:
            ev.ignore()

    def hoverEvent(self, ev: 'HoverEvent') -> None:
        """
        Update the hovered spots, their tool tip, and emit ``sigHovered``.

        Nothing is done unless the item was created with ``hoverable=True``. Only the
        SpotItems of the hovered spots are created.

        Parameters
        ----------
        ev : HoverEvent
            The hover event delivered by the scene.
        """
        if self.opts['hoverable']:
            old = self.data['hovered']

            if ev.exit:
                new = np.zeros_like(self.data['hovered'])
            else:
                new = self._maskAt(ev.pos())

            if self._hasHoverStyle():
                self.data['sourceRect'][old ^ new] = 0
                self.data['hovered'] = new
                self.updateSpots()

            points = self._pointsForIndices(np.flatnonzero(new)[::-1])

            # Show information about hovered points in a tool tip
            vb = self.getViewBox()
            if vb is not None and self.opts['tip'] is not None:
                if len(points) > 0:
                    cutoff = 3
                    tip = [self.opts['tip'](x=pt.pos().x(), y=pt.pos().y(), data=pt.data())
                           for pt in points[:cutoff]]
                    if len(points) > cutoff:
                        tip.append('({} others...)'.format(len(points) - cutoff))
                    vb.setToolTip('\n\n'.join(tip))
                    self._toolTipCleared = False
                elif not self._toolTipCleared:
                    vb.setToolTip("")
                    self._toolTipCleared = True

            self.sigHovered.emit(self, points, ev)

    def _hasHoverStyle(self):
        return any(self.opts['hover' + opt.title()] != _DEFAULT_STYLE[opt]
                   for opt in ['symbol', 'size', 'pen', 'brush'])


class SpotItem(object):
    """
    Class referring to individual spots in a scatter plot.
    These can be retrieved by calling ScatterPlotItem.points() or
    by connecting to the ScatterPlotItem's click signals.
    """

    def __init__(self, data, plot, index):
        self._data = data
        self._index = index
        # SpotItems are kept in plot.data["items"] numpy object array which
        # does not support cyclic garbage collection (numpy issue 6581).
        # Keeping a strong ref to plot here would leak the cycle
        self.__plot_ref = weakref.ref(plot)

    @property
    def _plot(self):
        return self.__plot_ref()

    def data(self):
        """Return the user data associated with this spot."""
        return self._data['data']

    def index(self):
        """Return the index of this point as given in the scatter plot data."""
        return self._index

    def size(self):
        """Return the size of this spot.
        If the spot has no explicit size set, then return the ScatterPlotItem's default size instead."""
        if self._data['size'] == -1:
            return self._plot.opts['size']
        else:
            return self._data['size']

    def pos(self):
        return Point(self._data['x'], self._data['y'])

    def viewPos(self):
        return self._plot.mapToView(self.pos())

    def setSize(self, size):
        """Set the size of this spot.
        If the size is set to -1, then the ScatterPlotItem's default size
        will be used instead."""
        self._data['size'] = size
        self.updateItem()

    def symbol(self):
        """Return the symbol of this spot.
        If the spot has no explicit symbol set, then return the ScatterPlotItem's default symbol instead.
        """
        symbol = self._data['symbol']
        if symbol is None:
            symbol = self._plot.opts['symbol']
        try:
            n = int(symbol)
            symbol = list(Symbols.keys())[n % len(Symbols)]
        except:
            pass
        return symbol

    def setSymbol(self, symbol):
        """Set the symbol for this spot.
        If the symbol is set to '', then the ScatterPlotItem's default symbol will be used instead."""
        self._data['symbol'] = symbol
        self.updateItem()

    def pen(self):
        pen = self._data['pen']
        if pen is None:
            pen = self._plot.opts['pen']
        return fn.mkPen(pen)

    def setPen(self, *args, **kwargs):
        """Set the outline pen for this spot"""
        self._data['pen'] = _mkPen(*args, **kwargs)
        self.updateItem()

    def resetPen(self):
        """Remove the pen set for this spot; the scatter plot's default pen will be used instead."""
        self._data['pen'] = None  ## Note this is NOT the same as calling setPen(None)
        self.updateItem()

    def brush(self):
        brush = self._data['brush']
        if brush is None:
            brush = self._plot.opts['brush']
        return fn.mkBrush(brush)

    def setBrush(self, *args, **kwargs):
        """Set the fill brush for this spot"""
        self._data['brush'] = _mkBrush(*args, **kwargs)
        self.updateItem()

    def resetBrush(self):
        """Remove the brush set for this spot; the scatter plot's default brush will be used instead."""
        self._data['brush'] = None  ## Note this is NOT the same as calling setBrush(None)
        self.updateItem()


    def isVisible(self):
        return self._data['visible']

    def setVisible(self, visible):
        """Set whether or not this spot is visible."""
        self._data['visible'] = visible
        self.updateItem()
    
    def setData(self, data):
        """Set the user-data associated with this spot"""
        self._data['data'] = data

    def updateItem(self):
        self._data['sourceRect'] = (0, 0, 0, 0)  # numpy <=1.13.1 won't let us set this with a single zero
        self._plot.updateSpots(self._data.reshape(1))
