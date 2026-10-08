import enum
import itertools
import math
import operator
import weakref
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import TYPE_CHECKING

import numpy as np

from .. import Qt, debug
from .. import functions as fn
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui
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


def _mkMany(values: Iterable, qtype: type, make: Callable[[object], object]) -> list:
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
    list
        One ``qtype`` object per value, in the same order.
    """
    cache = {}
    out = []
    append = out.append
    for value in values:
        if isinstance(value, qtype):
            append(value)
            continue
        key = (type(value), value)
        try:
            obj = cache.get(key)
        except TypeError:  # unhashable value
            append(make(value))
            continue
        if obj is None:
            obj = cache[key] = make(value)
        append(obj)
    return out


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
        self.bounds = [None, None]  ## caches data bounds
        self._maxSpotWidth = 0      ## maximum size of the scale-variant portion of all spots
        self._maxSpotPxWidth = 0    ## maximum size of the scale-invariant portion of all spots
        self._pixmapFragments = Qt.internals.PrimitiveArray(QtGui.QPainter.PixmapFragment, 10)
        self.opts = {
            'pxMode': True,
            'useCache': True,  ## If useCache is False, symbols are re-drawn on every paint.
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
        *brush*                The brush (or list of brushes) to use for filling spots.
        *size*                 The size (or list of sizes) of spots. If *pxMode* is True, this value is in pixels. Otherwise,
                               it is in the item's local coordinate system.
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
        *antialias*            Whether to draw symbols with antialiasing. Note that if pxMode is True, symbols are
                               always rendered with antialiasing (since the rendered symbols can be cached, this
                               incurs very little performance cost)
        *compositionMode*      If specified, this sets the composition mode used when drawing the
                               scatter plot (see QPainter::CompositionMode in the Qt documentation).
        *name*                 The name of this item. Names are used for automatically
                               generating LegendItem entries and by some exporters.
        ====================== ===============================================================================================
        """
        oldData = self.data  ## this causes cached pixmaps to be preserved while new data is registered.
        self.clear()  ## clear out all old data
        self.addPoints(*args, **kwargs)

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

        ## Extend record array
        oldData = self.data
        self.data = np.empty(len(oldData)+numPts, dtype=self.data.dtype)
        ## note that np.empty initializes object fields to None and string fields to ''

        self.data[:len(oldData)] = oldData
        #for i in range(len(oldData)):
            #oldData[i]['item']._data = self.data[i]  ## Make sure items have proper reference to new array

        newData = self.data[len(oldData):]
        newData['size'] = -1  ## indicates to use default size
        newData['visible'] = True

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
        if 'hoverable' in kwargs:
            self.opts['hoverable'] = bool(kwargs['hoverable'])
        if 'tip' in kwargs:
            self.opts['tip'] = kwargs['tip']
        if 'useCache' in kwargs:
            self.opts['useCache'] = kwargs['useCache']

        ## Set any extra parameters provided in keyword arguments
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
        if 'data' in kwargs:
            self.setPointData(kwargs['data'], dataSet=newData)

        self.prepareGeometryChange()
        self.informViewBoundsChanged()
        self.bounds = [None, None]
        self.invalidate()
        self.updateSpots(newData)
        self.sigPlotChanged.emit(self)

    def invalidate(self):
        ## clear any cached drawing state
        self.picture = None
        self.update()

    def getData(self):
        return self.data['x'], self.data['y']

    def implements(self, interface=None):
        ints = ['plotData']
        if interface is None:
            return ints
        return interface in ints

    def name(self):
        return self.opts.get('name', None)

    def setPen(self, *args, **kwargs) -> None:
        """
        Set the pen(s) used to draw the outline around each spot.

        If a list or array is provided, then the pen for each spot will be set
        separately; equal hashable specifications (such as color tuples) share one
        ``QPen``. Otherwise, the arguments are passed to :func:`~pyqtgraph.mkPen` and
        used as the default pen for all spots which do not have a pen explicitly set.

        Parameters
        ----------
        *args
            A list or array holding one pen specification per spot, or the arguments
            of :func:`~pyqtgraph.mkPen`.
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
            dataSet['pen'] = _mkMany(pens, QtGui.QPen, fn.mkPen)
        else:
            self.opts['pen'] = _mkPen(*args, **kwargs)

        dataSet['sourceRect'] = 0
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

        Parameters
        ----------
        *args
            A list or array holding one brush specification per spot, or the
            arguments of :func:`~pyqtgraph.mkBrush`.
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
            dataSet['brush'] = _mkMany(brushes, QtGui.QBrush, fn.mkBrush)
        else:
            self.opts['brush'] = _mkBrush(*args, **kwargs)

        dataSet['sourceRect'] = 0
        if update:
            self.updateSpots(dataSet)

    def setSymbol(self, symbol, update=True, dataSet=None, mask=None):
        """Set the symbol(s) used to draw each spot.
        If a list or array is provided, then the symbol for each spot will be set separately.
        Otherwise, the argument will be used as the default symbol for
        all spots which do not have a symbol explicitly set.

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

        dataSet['sourceRect'] = 0
        if update:
            self.updateSpots(dataSet)

    def setSize(self, size, update=True, dataSet=None, mask=None):
        """Set the size(s) used to draw each spot.
        If a list or array is provided, then the size for each spot will be set separately.
        Otherwise, the argument will be used as the default size for
        all spots which do not have a size explicitly set."""
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

        dataSet['sourceRect'] = 0
        if update:
            self.updateSpots(dataSet)


    def setPointsVisible(self, visible, update=True, dataSet=None, mask=None):
        """Set whether or not each spot is visible.
        If a list or array is provided, then the visibility for each spot will be set separately.
        Otherwise, the argument will be used for all spots."""
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

    def updateSpots(self, dataSet=None):
        profiler = debug.Profiler()  # noqa: profiler prints on GC
        if dataSet is None:
            dataSet = self.data

        invalidate = False
        if self.opts['pxMode'] and self.opts['useCache']:
            mask = dataSet['sourceRect']['w'] == 0
            if np.any(mask):
                invalidate = True
                coords = self.fragmentAtlas[self._atlasStyles(data=dataSet, idx=mask)]
                dataSet['sourceRect'][mask] = coords

            self._maybeRebuildAtlas()
        else:
            invalidate = True

        self._updateMaxSpotSizes(data=dataSet)

        if invalidate:
            self.invalidate()

    def _atlasStyles(self, data: np.ndarray | None = None,
                     idx: np.ndarray | slice | None = None) -> list[tuple]:
        """
        Return the symbol atlas styles of a set of spots.

        The sizes are quantized as the atlas does (see :func:`_quantizeSize`), so that
        spots whose sizes round to the same value form a single group in the atlas
        lookup.

        Parameters
        ----------
        data : numpy.ndarray, optional
            Structured spot array; defaults to ``self.data``.
        idx : numpy.ndarray or slice, optional
            Boolean mask or index selecting the spots; defaults to all spots.

        Returns
        -------
        list of tuple
            One ``(symbol, size, pen, brush)`` tuple per selected spot.
        """
        symbol, size, pen, brush = self._style(['symbol', 'size', 'pen', 'brush'],
                                               data=data, idx=idx)
        size = _quantizeSizes(size, 4 * self.fragmentAtlas.devicePixelRatio())
        return list(zip(symbol.tolist(), size.tolist(), pen.tolist(), brush.tolist()))

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
            self.fragmentAtlas.rebuild(self._atlasStyles())
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
            # Cull points that are outside view
            viewMask = self._maskAt(self.viewRect())

            # Map points using painter's world transform so they are drawn with pixel-valued sizes
            pts = np.vstack([self.data['x'], self.data['y']])
            pts = fn.transformCoordinates(p.transform(), pts)
            pts = fn.clip_array(pts, -2 ** 30, 2 ** 30)  # prevent Qt segmentation fault.
            p.resetTransform()

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

                # x, y is the center of the target rect
                xy = pts[:, viewMask].T
                sr = self.data['sourceRect'][viewMask]

                self._pixmapFragments.resize(sr.size)
                frags = self._pixmapFragments.ndarray()
                frags[:, 0:2] = xy
                frags[:, 2:6] = np.frombuffer(sr, dtype=int).reshape((-1, 4)) # sx, sy, sw, sh
                frags[:, 6:10] = [1/dpr, 1/dpr, 0.0, 1.0]   # scaleX, scaleY, rotation, opacity

                profiler('prep')
                drawargs = self._pixmapFragments.drawargs()
                p.drawPixmapFragments(*drawargs, self.fragmentAtlas.pixmap)
                profiler('draw')
            else:
                # render each symbol individually
                p.setRenderHint(p.RenderHint.Antialiasing, aa)

                for pt, style in zip(
                        pts[:, viewMask].T,
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

    def _maskAt(self, obj):
        """
        Return a boolean mask indicating all points that overlap obj, a QPointF or QRectF.
        """
        if isinstance(obj, QtCore.QPointF):
            l = r = obj.x()
            t = b = obj.y()
        elif isinstance(obj, QtCore.QRectF):
            l = obj.left()
            r = obj.right()
            t = obj.top()
            b = obj.bottom()
        else:
            raise TypeError

        if self.opts['pxMode'] and self.opts['useCache']:
            w = self.data['sourceRect']['w']
            h = self.data['sourceRect']['h']
        else:
            s, = self._style(['size'])
            w = h = s

        w = w / 2
        h = h / 2

        if self.opts['pxMode']:
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
            w *= px
            h *= py

        return (self.data['visible']
                & (self.data['x'] + w > l)
                & (self.data['x'] - w < r)
                & (self.data['y'] + h > t)
                & (self.data['y'] - h < b))

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
