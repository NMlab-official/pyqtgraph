import sys
import weakref
from bisect import bisect_left
from collections import OrderedDict
from collections.abc import Hashable
from math import ceil, copysign, floor, frexp, isfinite, log10, sqrt

import numpy as np

from .. import debug as debug
from .. import functions as fn
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from .GraphicsWidget import GraphicsWidget

__all__ = ['AxisItem']

_AlignmentFlag = QtCore.Qt.AlignmentFlag

# Flags used to draw the tick labels, per orientation; computed once instead of once
# per label (combining enum flags is comparatively slow with PyQt6 / PySide6).
_TICK_TEXT_FLAGS = {
    'left': (_AlignmentFlag.AlignRight | _AlignmentFlag.AlignVCenter)
            | QtCore.Qt.TextFlag.TextDontClip,
    'right': (_AlignmentFlag.AlignLeft | _AlignmentFlag.AlignVCenter)
             | QtCore.Qt.TextFlag.TextDontClip,
    'top': (_AlignmentFlag.AlignHCenter | _AlignmentFlag.AlignBottom)
           | QtCore.Qt.TextFlag.TextDontClip,
    'bottom': (_AlignmentFlag.AlignHCenter | _AlignmentFlag.AlignTop)
              | QtCore.Qt.TextFlag.TextDontClip,
}

# Rectangle in which tick labels are measured, see AxisItem._measureTickText.
_TEXT_MEASURE_RECT = QtCore.QRectF(0, 0, 100, 100)

# Whether the axes may draw directly instead of replaying a QPicture: Qt 5 lays the
# text of a picture out again when replaying it, which can move a label by a pixel.
_DIRECT_DRAWING = int(QtCore.qVersion().split('.')[0]) >= 6

# Render hints, besides Antialiasing, that a replayed QPicture set to their recorded
# value, off for the pictures the axes used to record; see _AxisPicture.play.
_REPLAYED_RENDER_HINTS = QtGui.QPainter.RenderHint.SmoothPixmapTransform
if hasattr(QtGui.QPainter.RenderHint, 'NonCosmeticBrushPatterns'):  # Qt >= 6.4
    _REPLAYED_RENDER_HINTS |= QtGui.QPainter.RenderHint.NonCosmeticBrushPatterns

_newPoint = Point.__new__
_initPointF = QtCore.QPointF.__init__


def _point(x: float, y: float) -> Point:
    """
    Return ``Point(x, y)``, skipping the argument parsing of ``Point.__init__``.

    :meth:`AxisItem.generateDrawSpecs` creates two points per tick.

    Parameters
    ----------
    x, y : float
        Coordinates of the point.

    Returns
    -------
    Point
        The new point.
    """
    point = _newPoint(Point)
    _initPointF(point, x, y)
    return point


def _tickLevelValuesNumpy(
    levels: list[tuple[float, float]],
    minVal: float,
    maxVal: float,
    scale: float
) -> list[tuple[float, list[float]]]:
    """
    Compute the tick values of each level by comparing all value pairs with numpy.

    Reference implementation of :func:`_tickLevelValues`, used for the inputs its
    faster path does not handle.

    Parameters
    ----------
    levels : list of tuple of float, float
        ``(spacing, offset)`` of each tick level, as returned by
        :meth:`AxisItem.tickSpacing`.
    minVal, maxVal : float
        The range of values, scaled, with ``minVal <= maxVal``.
    scale : float
        The scale of the axis, see :meth:`AxisItem.setScale`.

    Returns
    -------
    list of tuple of float, list of float
        ``(spacing, values)`` of each level, as returned by :meth:`AxisItem.tickValues`.
    """
    ticks = []
    allValues = np.array([])
    for spacing, offset in levels:
        ## determine starting tick
        start = (ceil((minVal-offset) / spacing) * spacing) + offset

        ## determine number of ticks
        num = int((maxVal-start) / spacing) + 1
        values = (np.arange(num) * spacing + start) / scale
        ## remove any ticks that were present in higher levels
        ## we assume here that if the difference between a tick value and a previously seen tick value
        ## is less than spacing/100, then they are 'equal' and we can ignore the new tick.
        close = np.any(
            np.isclose(
                allValues,
                values[:, np.newaxis],
                rtol=0,
                atol=spacing/scale*0.01
            ),
            axis=-1
        )
        values = values[~close]
        allValues = np.concatenate([allValues, values])
        ticks.append((spacing/scale, values.tolist()))
    return ticks


def _isPlainPositive(value: object, types: tuple[type, ...]) -> bool:
    """
    Tell whether ``value`` is a finite positive number of exactly one of ``types``.

    Parameters
    ----------
    value : object
        The value to test.
    types : tuple of type
        The accepted types; subclasses, e.g. numpy scalars, are not accepted.

    Returns
    -------
    bool
        True if ``value`` is such a number.
    """
    return type(value) in types and isfinite(value) and value > 0


def _tickLevelValues(
    levels: list[tuple[float, float]],
    minVal: float,
    maxVal: float,
    scale: float
) -> list[tuple[float, list[float]]]:
    """
    Compute the tick values of each level, dropping those already in a previous level.

    A value is dropped when it is within 1 % of the spacing of its level of a value of
    a previous level. The result equals that of :func:`_tickLevelValuesNumpy`, value
    for value, which compares all the value pairs with numpy: for the few dozen ticks
    of an axis, that costs more than the comparisons themselves. When the spacings and
    the scale are positive and all numbers are finite Python floats (or ints for the
    offsets and the scale), the values of a level are in non-decreasing order, and the
    values close to a previous value are found by bisection instead. Other inputs use
    :func:`_tickLevelValuesNumpy`.

    Parameters
    ----------
    levels : list of tuple of float, float
        ``(spacing, offset)`` of each tick level, as returned by
        :meth:`AxisItem.tickSpacing`.
    minVal, maxVal : float
        The range of values, scaled, with ``minVal <= maxVal``.
    scale : float
        The scale of the axis, see :meth:`AxisItem.setScale`.

    Returns
    -------
    list of tuple of float, list of float
        ``(spacing, values)`` of each level, as returned by :meth:`AxisItem.tickValues`.
    """
    if not _isPlainPositive(scale, (float, int)) or not all(
        _isPlainPositive(spacing, (float,))
        and type(offset) in (float, int) and isfinite(offset)
        # numpy warns about a non-finite tolerance
        and isfinite(spacing/scale*0.01)
        for spacing, offset in levels
    ):
        return _tickLevelValuesNumpy(levels, minVal, maxVal, scale)
    ticks = []
    previous = []
    for spacing, offset in levels:
        start = (ceil((minVal-offset) / spacing) * spacing) + offset
        num = int((maxVal-start) / spacing) + 1
        # the operations of the numpy version, in the same order: the same values
        values = [(k * spacing + start) / scale for k in range(num)]
        if values and not (isfinite(values[0]) and isfinite(values[-1])):
            return _tickLevelValuesNumpy(levels, minVal, maxVal, scale)
        if previous and values:
            atol = spacing/scale*0.01
            count = len(values)
            close = set()
            for prev in previous:
                # values are sorted, and so are their distances to prev on each side
                # of its insertion point: scan both sides while they are close
                k = j = bisect_left(values, prev)
                while k < count and abs(prev - values[k]) <= atol:
                    close.add(k)
                    k += 1
                k = j - 1
                while k >= 0 and abs(prev - values[k]) <= atol:
                    close.add(k)
                    k -= 1
            if close:
                values = [v for k, v in enumerate(values) if k not in close]
        previous.extend(values)
        ticks.append((spacing/scale, values))
    return ticks


def _pinnedFont(font: QtGui.QFont) -> QtGui.QFont:
    """
    Return a copy of ``font`` with all its attributes marked as set.

    A painter resolves the attributes of a font that are not set against the font of
    its device: a widget painter would draw the default font as the font of the widget.
    A font read from a data stream has all its attributes set, as had the fonts a
    replayed ``QtGui.QPicture`` drew text with.

    Parameters
    ----------
    font : QtGui.QFont
        The font.

    Returns
    -------
    QtGui.QFont
        The copy.
    """
    data = QtCore.QByteArray()
    stream = QtCore.QDataStream(data, QtCore.QIODevice.OpenModeFlag.WriteOnly)
    stream << font
    pinned = QtGui.QFont()
    stream = QtCore.QDataStream(data, QtCore.QIODevice.OpenModeFlag.ReadOnly)
    stream >> pinned
    return pinned


def _isThinCosmeticPen(pen: object) -> bool:
    """
    Tell whether ``pen`` is a cosmetic pen at most one pixel wide.

    Parameters
    ----------
    pen : object
        The pen of a drawing specification.

    Returns
    -------
    bool
        True for such a ``QtGui.QPen``.
    """
    return isinstance(pen, QtGui.QPen) and pen.isCosmetic() and pen.widthF() <= 1


def _drawnAsReplayed(specs: tuple) -> bool:
    """
    Tell whether :meth:`AxisItem.drawPicture` draws ``specs`` as a replayed picture.

    A ``QtGui.QPicture`` records each line as a two-point polyline, or as a filled
    rectangle if its ends are equal. The raster engine draws lines and two-point
    polylines alike with cosmetic pens at most one pixel wide, but strokes them
    differently with wider pens near pixel boundaries. With Qt 6, the text is drawn
    alike (see ``_DIRECT_DRAWING``).

    Parameters
    ----------
    specs : tuple
        The value returned by :meth:`AxisItem.generateDrawSpecs`.

    Returns
    -------
    bool
        True if drawing ``specs`` directly with the raster engine gives the pixels of
        the replayed picture.
    """
    axisSpec, tickSpecs, _ = specs
    lastPen = None
    for pen, p1, p2 in (axisSpec, *tickSpecs):
        if pen is not lastPen:
            if not _isThinCosmeticPen(pen):
                return False
            lastPen = pen
        if p1 == p2:  # fuzzy, as the comparison made when recording
            return False
    return True


# Resolution a QPicture is recorded for; see _AxisPicture.play.
_resolutionPicture = None


def _drawsAtPictureResolution(p: QtGui.QPainter) -> bool:
    """
    Tell whether ``p`` paints with the raster engine at the resolution of pictures.

    A replayed picture is scaled by the ratio of the resolution of the painted device to
    the resolution pictures are recorded for, that of the primary screen.

    Parameters
    ----------
    p : QtGui.QPainter
        The active painter.

    Returns
    -------
    bool
        True for the raster engine, on a device whose logical resolution is that of the
        primary screen.
    """
    global _resolutionPicture
    engine = p.paintEngine()
    device = p.device()
    if (
        engine is None or device is None
        or engine.type() != QtGui.QPaintEngine.Type.Raster
    ):
        return False
    if _resolutionPicture is None:
        _resolutionPicture = QtGui.QPicture()
    return (device.logicalDpiX() == _resolutionPicture.logicalDpiX()
            and device.logicalDpiY() == _resolutionPicture.logicalDpiY())


class _AxisPicture:
    """
    Drawing specifications of an axis, which :meth:`AxisItem.paint` draws.

    The axis used to record its specifications into a ``QtGui.QPicture`` when they
    changed and to replay the picture when painted. Recording costs about as much as
    drawing, and replaying draws again: drawing the specifications directly costs half
    as much when the axis is painted once per change, e.g. while panning or zooming,
    and about as much as a replay otherwise. The drawing is that of the replayed
    picture, see :meth:`play`; when it may not be, the picture is recorded when the
    specifications are generated and replayed, as before.

    Parameters
    ----------
    axis : AxisItem
        The axis, referenced weakly.
    specs : tuple or None
        The value returned by :meth:`AxisItem.generateDrawSpecs`; None draws nothing.
    font : QtGui.QFont
        The font of the painter the tick labels were measured with, with all its
        attributes set (see :func:`_pinnedFont`).
    picture : QtGui.QPicture or None
        The picture ``specs`` were recorded into, or None if they may be drawn
        directly (see :func:`_drawnAsReplayed`).
    """

    __slots__ = ('_axis', '_specs', '_font', '_direct', '_picture')

    def __init__(
        self,
        axis: 'AxisItem',
        specs: tuple | None,
        font: QtGui.QFont,
        picture: QtGui.QPicture | None
    ) -> None:
        self._axis = weakref.ref(axis)
        self._specs = specs
        self._font = font
        self._direct = picture is None
        self._picture = picture

    def play(self, p: QtGui.QPainter) -> None:
        """
        Draw the axis with ``p`` by calling :meth:`AxisItem.drawPicture`.

        The specifications are drawn directly with the raster engine at the resolution
        of the primary screen, unless they were recorded into a picture: as a replayed
        picture did, the labels are drawn with the font they were measured with, the
        render hints that a picture replays are set as recorded, and the painter state
        is left as the drawing leaves it. Otherwise the picture is replayed, recorded
        first if needed.

        Parameters
        ----------
        p : QtGui.QPainter
            The painter, in the local coordinates of the axis.
        """
        axis = self._axis()
        if self._specs is None or axis is None:
            return
        if self._direct and _drawsAtPictureResolution(p):
            p.setFont(self._font)
            p.setRenderHints(_REPLAYED_RENDER_HINTS, False)
            axis._drawingDirectly = True
            try:
                axis.drawPicture(p, *self._specs)
            finally:
                axis._drawingDirectly = False
            return
        if self._picture is None:
            picture = QtGui.QPicture()
            painter = QtGui.QPainter(picture)
            try:
                painter.setFont(self._font)
                axis.drawPicture(painter, *self._specs)
            finally:
                painter.end()
            self._picture = picture
        self._picture.play(p)


class _LRUCache:
    """
    Mapping bounded to a maximum number of entries, least recently used first out.

    Parameters
    ----------
    maxSize : int
        Maximum number of entries kept.
    """

    __slots__ = ('_data', '_maxSize')

    def __init__(self, maxSize: int) -> None:
        self._data = OrderedDict()
        self._maxSize = maxSize

    def __len__(self) -> int:
        """
        Return the number of entries.

        Returns
        -------
        int
            The number of cached entries.
        """
        return len(self._data)

    def __contains__(self, key: Hashable) -> bool:
        """
        Tell whether ``key`` is cached, without marking it as recently used.

        Parameters
        ----------
        key : Hashable
            The key to look up.

        Returns
        -------
        bool
            True if ``key`` is cached.
        """
        return key in self._data

    def get(self, key: Hashable) -> object | None:
        """
        Return the value stored for ``key`` and mark it as recently used.

        Parameters
        ----------
        key : Hashable
            The key to look up.

        Returns
        -------
        object or None
            The stored value, or None if ``key`` is not cached.
        """
        value = self._data.get(key)
        if value is not None:
            self._data.move_to_end(key)
        return value

    def put(self, key: Hashable, value: object) -> None:
        """
        Store ``value`` for ``key``, evicting the least recently used entry if full.

        Parameters
        ----------
        key : Hashable
            The key.
        value : object
            The value to store; must not be None.
        """
        self._data[key] = value
        self._data.move_to_end(key)
        if len(self._data) > self._maxSize:
            self._data.popitem(last=False)

    def clear(self) -> None:
        """Remove all entries."""
        self._data.clear()


# Blended colors, ``(color QRgba, background QRgb) -> QColor``; see _opaqueGridPen.
_blendedColorCache = _LRUCache(256)


def _opaqueGridPen(pen: QtGui.QPen, background: int) -> QtGui.QPen:
    """
    Return an opaque pen drawing like ``pen`` over an opaque background.

    The color is computed by the raster paint engine itself, which blends ``pen``'s
    color over the background, so that drawing the returned pen over that background
    gives the very pixels that ``pen`` gives.

    Parameters
    ----------
    pen : QtGui.QPen
        The translucent pen.
    background : int
        The opaque background color, as a QRgb value.

    Returns
    -------
    QtGui.QPen
        ``pen`` itself if it is opaque or not a solid color pen, else an opaque copy.
    """
    color = pen.color()
    if pen.brush().style() != QtCore.Qt.BrushStyle.SolidPattern or color.alpha() == 255:
        return pen
    key = (color.rgba(), background)
    blended = _blendedColorCache.get(key)
    if blended is None:
        image = QtGui.QImage(1, 1, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(QtGui.QColor.fromRgba(background))
        painter = QtGui.QPainter(image)
        painter.fillRect(QtCore.QRect(0, 0, 1, 1), color)  # blended: SourceOver
        painter.end()
        blended = image.pixelColor(0, 0)
        _blendedColorCache.put(key, blended)
    opaquePen = QtGui.QPen(pen)
    opaquePen.setColor(blended)
    return opaquePen


class AxisItem(GraphicsWidget):
    """
    GraphicsItem showing a single plot axis with ticks, values, and label.
    
    Can be configured to fit on any side of a plot, automatically synchronize its
    displayed scale with ViewBox items. Ticks can be extended to draw a grid.
    
    If maxTickLength is negative, ticks point into the plot.

    Parameters
    ----------
    orientation : {'left', 'right', 'top', 'bottom'}
        The side of the plot the axis is attached to.
    pen : QPen or None
        Pen used when drawing axis and (by default) ticks.
    textPen : QPen or None
        Pen used when drawing tick labels.
    tickPen : QPen or None
        Pen used when drawing ticks.
    linkView : ViewBox or None
        Causes the range of values displayed in the axis to be linked to the visible
        range of a ViewBox.
    parent : QtWidgets.QGraphicsItem or None
        Parent Qt object to set to. End users are not expected to set, pyqtgraph should
        set correctly on its own.
    maxTickLength : int
        Maximum length of ticks to draw in pixels. Negative values draw into the
        plot, positive values draw outward.  Default -5.
    showValues : bool
        Whether to display values adjacent to ticks. Default true.
    **args
        All additional keyword arguments are passed to :func:`setLabel`.
    """

    # Size of the tick label strings, ``(text, font and device key) -> (width,
    # height)``, shared by all axes: linked axes usually display the same labels.
    # Keyed by font, so it never needs to be invalidated.
    _tickTextSizeCache = _LRUCache(4096)

    def __init__(
            self,
            orientation: str,
            pen: object = None,
            textPen: object = None,
            tickPen: object = None,
            linkView: object = None,
            parent: QtWidgets.QGraphicsItem | None = None,
            maxTickLength: int = -5,
            showValues: bool = True,
            **args: object,
    ) -> None:
        super().__init__(parent)
        # Tick strings, ``(values, scale, spacing, extra state) -> strings``; see
        # _tickStringsCacheKey.
        self._tickStringsCache = _LRUCache(64)
        # While a picture is pending, weak reference to the scene whose
        # sigPrepareForPaint is connected to _prepareForPaint; see _invalidatePicture.
        self._preparingScene = None
        # True while _buildPicture runs: the picture it builds is assigned afterwards.
        self._buildingPicture = False
        # The font the tick labels were last measured with, and its copy given to
        # _AxisPicture (see _pinnedFont), or None.
        self._labelFonts = None
        # True while _AxisPicture.play draws directly: drawPicture keeps the font.
        self._drawingDirectly = False
        self.label = QtWidgets.QGraphicsTextItem(self)
        self.picture = None
        # Background color (QRgb) the grid pens of the picture were blended with, or
        # None if they were not; see the ``opaqueGrid`` style option.
        self._pictureGridBackground = None
        self.orientation = orientation

        if orientation in {'left', 'right'}:
            self.label.setRotation(-90)
            # allow labels on vertical axis to extend above and below the length of the axis
            hide_overlapping_labels = False
        elif orientation in {'top', 'bottom'}:
            # stop labels on horizontal axis from overlapping so vertical axis labels have room
            hide_overlapping_labels = True
        else:
            raise ValueError(
                "Orientation argument must be one of 'left', 'right', 'top', or 'bottom'."
            )
        self.style = {
            'tickTextOffset': [5, 2],  ## (horizontal, vertical) spacing between text and axis
            'tickTextWidth': 30,  ## space reserved for tick text
            'tickTextHeight': 18,
            'autoExpandTextSpace': True,  ## automatically expand text space if needed
            'autoReduceTextSpace': True,
            'hideOverlappingLabels': hide_overlapping_labels,
            'tickFont': None,
            'stopAxisAtTick': (False, False),  ## whether axis is drawn to edge of box or to last tick
            'textFillLimits': [  ## how much of the axis to fill up with tick text, maximally.
                (0, 0.8),    ## never fill more than 80% of the axis
                (2, 0.6),    ## If we already have 2 ticks with text, fill no more than 60% of the axis
                (4, 0.4),    ## If we already have 4 ticks with text, fill no more than 40% of the axis
                (6, 0.2),    ## If we already have 6 ticks with text, fill no more than 20% of the axis
            ],
            'showValues': showValues,
            'tickLength': maxTickLength,
            'maxTickLevel': 2,
            'maxTextLevel': 2,
            'tickAlpha': None,  ## If not none, use this alpha for all ticks.
            'opaqueGrid': True,  ## blend grid lines with a known opaque background
        }

        self.textWidth = 30  ## Keeps track of maximum width / height of tick text
        self.textHeight = 18

        # If the user specifies a width / height, remember that setting
        # indefinitely.
        self.fixedWidth = None
        self.fixedHeight = None

        self.logMode = False

        self._tickDensity = 1.0   # used to adjust scale the number of automatically generated ticks
        self._tickLevels  = None  # used to override the automatic ticking system with explicit ticks
        self._tickSpacing = None  # used to override default tickSpacing method
        self.scale = 1.0
        self.autoSIPrefix = True
        self.autoSIPrefixScale = 1.0

        self.labelText = ""
        self.labelUnits = ""
        self.labelUnitPrefix = ""
        self.unitPower = 1
        self.labelStyle = {}
        self._siPrefixEnableRanges = None
        self.setRange(0, 1)
        self.setLabel(**args)
        self.showLabel(False)

        if pen is None:
            self.setPen()
        else:
            self.setPen(pen)

        if textPen is None:
            self.setTextPen()
        else:
            self.setTextPen(textPen)

        if tickPen is None:
            self.setTickPen()
        else:
            self.setTickPen(tickPen)

        self._linkedView = None
        if linkView is not None:
            self._linkToView_internal(linkView)

        self.grid = False


    def setStyle(self, **kwargs: object) -> None:
        """
        Set various style options.

        Parameters
        ----------
        **kwargs
            Here are a list of supported arguments.

            ===================== ======================================================
            Property              Description
            ===================== ======================================================
            tickLength            ``int``
                                  The maximum length of ticks in pixels. Positive values
                                  point toward the text; negative values point away.

            tickTextOffset        ``int`` 
                                  Reserved spacing between text and axis in pixels.

            tickTextWidth         ``int``
                                  Horizontal space reserved for tick text in pixels.

            tickTextHeight        ``int``
                                  Vertical space reserved for tick text in pixels.

            autoExpandTextSpace   ``bool``
                                  Automatically expand text space if the tick strings
                                  become too long.

            autoReduceTextSpace   ``bool``
                                  Automatically shrink the axis if necessary.

            hideOverlappingLabels ``bool`` or ``int``

                                  - ``True``  (default for horizontal axis): Hide tick
                                    labels which extend beyond the AxisItem's geometry
                                    rectangle.
                                  - ``False`` (default for vertical axis): Labels may be
                                    drawn extending beyond the extent of the axis.
                                  - ``int`` sets the tolerance limit for how many pixels
                                    a label is allowed to extend beyond the axis.
                                    Defaults to 15 for
                                    ``hideOverlappingLabels = False``.

            tickFont              :class:`QFont` or ``None``
                                  Determines the font used for tick values. Use None for
                                  the default font.
            
            stopAxisAtTick        tuple of ``bool, bool`` 
                                  The first element represents the horizontal axis, the
                                  second element represents the vertical axis.

                                  - ``True`` - The axis line is drawn only as far as the
                                    last tick.
                                  - ``False`` - The line is drawn to the edge of the
                                    :class:`~pyqtgraph.AxisItem` boundary.

            textFillLimits        list of ``(int, float)``
                                  This structure determines how the AxisItem decides how
                                  many ticks should have text appear next to them.
                                  The first value corresponds to the tick number.  The
                                  second value corresponds to the fill percentage. Each
                                  tuple in the list specifies what fraction of the axis
                                  length may be occupied by text, given the number of
                                  ticks that already have text displayed.
                                  
                                  For example ::

                                    [
                                        # Never fill more than 80% of the axis
                                        (0, 0.8),
                                        # If we already have 2 ticks with text, fill no
                                        # more than 60% of the axis
                                        (2, 0.6), 
                                        # If we already have 4 ticks with text, fill no
                                        # more than 40% of the axis
                                        (4, 0.4), 
                                        # If we already have 6 ticks with text, fill no
                                        # more than 20% of the axis
                                        (6, 0.2)
                                    ]
                                                
            showValues            ``bool``
                                  indicates whether text is displayed adjacent to ticks.
            
            tickAlpha             ``float``, ``int`` or ``None`` 
                                  If ``None``, pyqtgraph will draw the ticks with the
                                  alpha it deems appropriate. Otherwise, the alpha will
                                  be fixed at the value passed. With ``int``, accepted
                                  values are [0..255]. With value of type ``float``,
                                  accepted values are from [0..1].

            maxTickLevel          ``int``
                                  default: 2

                                  Tick (and grid line) density level.

                                  - 0: Show major ticks only
                                  - 1: Show major ticks and one level of minor ticks
                                  - 2: Show major ticks and two levels of minor ticks
                                    (higher CPU usage)

            opaqueGrid            ``bool``
                                  default: True

                                  If ``True``, translucent grid lines are drawn with
                                  an opaque pen, whose color is the grid color blended
                                  with the background: that of the
                                  :class:`~pyqtgraph.GraphicsView`, which the
                                  background color of the linked view must match if
                                  it is set. Over that background, the result is
                                  identical, and much faster to draw with the raster
                                  paint engine, especially with a fractional device
                                  pixel ratio. But grid lines are drawn over the items
                                  of the view (data, ticks of the other axis), which
                                  they then hide where they cross them instead of
                                  tinting them. Translucent pens are kept if the
                                  background is not a known opaque color, and for
                                  exports. Set to ``False`` to draw translucent grid
                                  lines over the items.
            ===================== ======================================================

        Raises
        ------
        NameError
            Raised when the name of a keyword argument is not recognized.
        TypeError
            Raised when a value for a keyword argument is of the wrong type.
        """
        for kwd, value in kwargs.items():
            if kwd not in self.style:
                raise NameError(f"{kwd} is not a valid style argument.")

            if (
                kwd in (
                    'tickLength',
                    'tickTextOffset',
                    'tickTextWidth',
                    'tickTextHeight'
                ) and 
                not isinstance(value, int)
            ):
                raise TypeError(f"Argument '{kwd}' must be int")

            if kwd == 'tickTextOffset':
                if self.orientation in ('left', 'right'):
                    self.style['tickTextOffset'][0] = value
                else:
                    self.style['tickTextOffset'][1] = value
            elif kwd == 'stopAxisAtTick':
                if len(value) != 2 or not all(isinstance(val, bool) for val in value):
                    raise TypeError(
                        "Argument 'stopAxisAtTick' must have type (bool, bool)"
                    )
                self.style[kwd] = value
            else:
                self.style[kwd] = value

        self._invalidateTickCaches()
        self._invalidatePicture()
        self._adjustSize()
        self.update()

    def close(self):
        self.scene().removeItem(self.label)
        self.label = None
        self.scene().removeItem(self)

    def setGrid(self, grid: int | float | bool):
        """
        Set the alpha value for the grid, or ``False`` to disable.

        When grid lines are enabled, the axis tick lines are extended to cover the
        extent of the linked ViewBox, if any.

        Parameters
        ----------
        grid : bool or int or float
            Alpha value to apply to :class:`~pyqtgraph.GridItem`.
            
            - ``False`` - Disable the grid.
            - ``int`` - Values between [0, 255] to set the alpha of the grid to.
            - ``float`` - Values between [0..1] to set the alpha of the grid to.
        """
        if isinstance(grid, float):
            grid = int(grid * 255)
            grid = min(grid, 255)
            grid = max(grid, 0)
        self.grid = grid
        self._invalidatePicture()
        self.prepareGeometryChange()
        self.update()

    def setLogMode(
        self,
        *args: bool,
        **kwargs: bool
    ) -> None:
        """
        Set log scaling for x and / or y axes.

        If two positional arguments are provided, the first will set log scaling
        for the x axis and the second for the y axis. If a single positional
        argument is provided, it will set the log scaling along the direction of
        the AxisItem. Alternatively, x and y can be passed as keyword arguments.

        If an axis is set to log scale, ticks are displayed on a logarithmic scale and
        values are adjusted accordingly. The linked ViewBox will be informed of the
        change.

        Parameters
        ----------
        *args : bool
            If length 1, sets log mode regardless of orientation.  If length 2, the
            first element toggles log mode for x-axis, and the second element toggles
            log mode for the y-axis.
        **kwargs : bool
            Pass a dictionary with keys `x` and `y`, where the values are ``bool`` to
            set the log mode for the respective `x` or `y` axis.  Trying to set the `y`
            axis log mode while this axis item is horizontal (or vice versa) will be
            ignored.

        See Also
        --------
        :meth:`~pyqtgraph.PlotItem.setLogMode`
            The method called to shift the values of the data.
        """
        if len(args) == 1:
            self.logMode = args[0]
        else:
            if len(args) == 2:
                x, y = args
            else:
                x = kwargs.get('x')
                y = kwargs.get('y')

            if x is not None and self.orientation in ('top', 'bottom'):
                self.logMode = x
            if y is not None and self.orientation in ('left', 'right'):
                self.logMode = y

        # inform the linked views of the change
        if self._linkedView is not None:
            if self.orientation in ('top', 'bottom'):
                self._linkedView().setLogMode('x', self.logMode)
            elif self.orientation in ('left', 'right'):
                self._linkedView().setLogMode('y', self.logMode)

        self._invalidateTickCaches()
        self._invalidatePicture()
        self.update()

    def setTickFont(self, font: QtGui.QFont | None) -> None:
        """
        Set the font used for tick values.
        
        Parameters
        ----------
        font : QtGui.QFont or None
            The font to use for the tick values. Set to ``None`` for the default font.
        """
        self.style['tickFont'] = font
        self._invalidateTickCaches()
        self._invalidatePicture()
        self.prepareGeometryChange()
        # Need to re-allocate space depending on font size?
        self.update()

    def resizeEvent(self, ev=None):
        # Set the position of the label
        nudge = 5
        # self.label is set to None on close, but resize events can still occur.
        if self.label is None:
            self.picture = None
            return

        br = self.label.boundingRect()
        p = QtCore.QPointF(0, 0)
        if self.orientation == 'left':
            p.setY(int(self.size().height()/2 + br.width()/2))
            p.setX(-nudge)
        elif self.orientation == 'right':
            p.setY(int(self.size().height()/2 + br.width()/2))
            p.setX(int(self.size().width()-br.height()+nudge))
        elif self.orientation == 'top':
            p.setY(-nudge)
            p.setX(int(self.size().width()/2. - br.width()/2.))
        elif self.orientation == 'bottom':
            p.setX(int(self.size().width()/2. - br.width()/2.))
            p.setY(int(self.size().height()-br.height()+nudge))
        self.label.setPos(p)
        self._invalidatePicture()

    def showLabel(self, show: bool=True):
        """
        Show or hide the label text for this axis.

        Parameters
        ----------
        show : bool, optional
            Show the label text, by default True.
        """
        self.label.setVisible(show)
        if self.orientation in ['left', 'right']:
            self._updateWidth()
        else:
            self._updateHeight()
        if self.autoSIPrefix:
            self.updateAutoSIPrefix()

    def setLabel(
        self,
        text: str | None=None,
        units: str | None=None,
        unitPrefix: str | None=None,
        siPrefixEnableRanges: tuple[tuple[float, float], ...] | None=None,
        unitPower: int | float=1,
        **kwargs: object
    ) -> None:
        """
        Set the text displayed adjacent to the axis.

        Parameters
        ----------
        text : str
            The text (excluding units) to display on the label for this axis.
        units : str
            The units for this axis. Units should generally be given without any scaling
            prefix (eg, 'V' instead of 'mV'). The scaling prefix will be automatically
            prepended based on the range of data displayed.
        unitPrefix : str
            An extra prefix to prepend to the units.
        siPrefixEnableRanges : tuple of tuple of float, float, optional
            The ranges in which automatic SI prefix scaling is enabled. Defaults to
            everywhere, unless units is empty, in which case it defaults to
            ``((0., 1.), (1e9, inf))``.
        unitPower : int or float, optional
            The power to which the units are raised. For example, if units='m²', the
            unitPower should be 2. This ensures correct scaling when using SI prefixes.
            Supports positive, negative and non-integral powers.  Default is 1.
            Note: The power only affects the scaling, not the units themselves. For
            example, with units='m' and unitPower=2, the displayed units will still be 'm'.
        **kwargs
            All extra keyword arguments become CSS style options for the ``<span>`` tag
            which will surround the axis label and units. Note that CSS attributes are
            not always valid python arguments. Examples: ``color='#FFF'``,
            ``**{'font-size': '14pt'}``.

        Notes
        -----
        The final text generated for the label will usually take the form::

            <span style="...args...">{text} (prefix{units})</span>
        """
        self.labelText = text or ""
        self.labelUnits = units or ""
        self.labelUnitPrefix = unitPrefix or ""
        self.unitPower = unitPower
        if kwargs:
            self.labelStyle = kwargs
        self.setSIPrefixEnableRanges(siPrefixEnableRanges)
        # Account empty string and `None` for units and text
        visible = bool(text or units)
        self.showLabel(visible)
        self._invalidateTickCaches()
        self._updateLabel()

    def setSIPrefixEnableRanges(self, ranges=None):
        """
        Set the ranges in which automatic SI prefix scaling is enabled.

        This function allows you to define specific ranges where SI prefixes will be
        used. By default, SI prefix scaling is enabled everywhere, unless units are
        empty, in which case it defaults to ``((0., 1.), (1e9, inf))``.

        Parameters
        ----------
        ranges : tuple of tuple of float, float, optional
            A tuple of ranges where SI prefix scaling is enabled. Each range is a tuple
            containing two floats representing the start and end of the range.
        """
        self._siPrefixEnableRanges = ranges

    def getSIPrefixEnableRanges(self):
        """
        Get the ranges in which automatic SI prefix scaling is enabled.

        Returns
        -------
        tuple of tuple of float, float
            A tuple of ranges where SI prefix scaling is enabled. Each range is a tuple
            containing two floats representing the start and end of the range. If no
            custom ranges are set, then the default ranges are returned. The default
            ranges are ``((0., 1.), (1e9, inf))`` if units are empty, and 
            ``((0., inf))`` otherwise.
        """
        if self._siPrefixEnableRanges is not None:
            return self._siPrefixEnableRanges
        elif self.labelUnits == '':
            return (0., 1.), (1e9, float('inf'))
        else:
            return ((0., float('inf')),)

    def _updateLabel(self):
        self.label.setHtml(self.labelString())
        self._adjustSize()
        self._invalidatePicture()
        self.update()

    def labelString(self) -> str:
        """
        Generate the label string based on current label, units, and prefix.

        Returns
        -------
        str
            The complete label string, including units and any prefixes.
        """
        if self.labelUnits == '':
            if not self.autoSIPrefix or self.autoSIPrefixScale == 1.0:
                units = ''
            else:
                units = f'(x{1.0 / self.autoSIPrefixScale:g})'
        else:
            units = f'({self.labelUnitPrefix}{self.labelUnits})'

        s = f'{self.labelText} {units}'

        style = ';'.join([f'{k}: {self.labelStyle[k]}' for k in self.labelStyle])

        return f"<span style='{style}'>{s}</span>"

    def _updateMaxTextSize(self, x: int):
        ## Informs that the maximum tick size orthogonal to the axis has
        ## changed; we use this to decide whether the item needs to be resized
        ## to accommodate.
        if self.orientation in ['left', 'right']:
            if self.style["autoReduceTextSpace"]:
                if x > self.textWidth or x < self.textWidth - 10:
                    self.textWidth = x
            else:
                mx = max(self.textWidth, x)
                if mx > self.textWidth or mx < self.textWidth - 10:
                    self.textWidth = mx
            if self.style['autoExpandTextSpace']:
                self._updateWidth()

        else:
            if self.style['autoReduceTextSpace']:
                if x > self.textHeight or x < self.textHeight - 10:
                    self.textHeight = x
            else:
                mx = max(self.textHeight, x)
                if mx > self.textHeight or mx < self.textHeight - 10:
                    self.textHeight = mx
            if self.style['autoExpandTextSpace']:
                self._updateHeight()

    def _adjustSize(self):
        if self.orientation in ['left', 'right']:
            self._updateWidth()
        else:
            self._updateHeight()

    def setHeight(self, h: int | None=None):
        """
        Set the height of this axis reserved for ticks and tick labels.

        The height of the axis label is automatically added.

        Parameters
        ----------
        h : int or None, optional
            If ``None``, then the value will be determined automatically based on the
            size of the tick text, by default None.
        """
        self.fixedHeight = h
        self._updateHeight()

    def _updateHeight(self):
        if not self.isVisible():
            h = 0
        elif self.fixedHeight is None:
            if not self.style['showValues']:
                h = 0
            elif self.style['autoExpandTextSpace']:
                h = self.textHeight
            else:
                h = self.style['tickTextHeight']
            h += self.style['tickTextOffset'][1] if self.style['showValues'] else 0
            h += max(0, self.style['tickLength'])
            if self.label.isVisible():
                h += self.label.boundingRect().height() * 0.8
        else:
            h = self.fixedHeight

        self.setMaximumHeight(h)
        self.setMinimumHeight(h)
        self.picture = None

    def setWidth(self, w: int | None=None):
        """
        Set the width of this axis reserved for ticks and tick labels.

        The width of the axis label is automatically added.

        Parameters
        ----------
        w : int or None, optional
            If ``None``, then the value will be determined automatically based on the
            size of the tick text, by default None.
        """        
        self.fixedWidth = w
        self._updateWidth()

    def _updateWidth(self):
        if not self.isVisible():
            w = 0
        elif self.fixedWidth is None:
            if not self.style['showValues']:
                w = 0
            elif self.style['autoExpandTextSpace']:
                w = self.textWidth
            else:
                w = self.style['tickTextWidth']
            w += self.style['tickTextOffset'][0] if self.style['showValues'] else 0
            w += max(0, self.style['tickLength'])
            if self.label.isVisible():
                w += self.label.boundingRect().height() * 0.8  ## bounding rect is usually an overestimate
        else:
            w = self.fixedWidth

        self.setMaximumWidth(w)
        self.setMinimumWidth(w)
        self.picture = None

    def pen(self) -> QtGui.QPen:
        """
        Get the pen used for drawing text, axes, ticks, and grid lines.

        If no custom pen has been set, this method will return a pen with the
        default foreground color.

        Returns
        -------
        QPen
            The pen used to draw text, axes, ticks, and grid lines.
        """
        if self._pen is None:
            return fn.mkPen(getConfigOption('foreground'))
        return fn.mkPen(self._pen)

    def setPen(self, *args, **kwargs):
        """
        Set the pen used for drawing text, axes, ticks, and grid lines.
        
        If no arguments given, the default foreground color will be used.

        Parameters
        ----------
        *args
            Arguments relayed to :func:`~pyqtgraph.mkPen`.
        **kwargs
            Arguments relayed to `:func:`~pyqtgraph.mkPen`.

        See Also
        --------
        :func:`setConfigOption <pyqtgraph.setConfigOption>`
            Option to change the default foreground color.
        """        
        self.picture = None
        if args or kwargs:
            self._pen = fn.mkPen(*args, **kwargs)
        else:
            self._pen = fn.mkPen(getConfigOption('foreground'))
        self.labelStyle['color'] = self._pen.color().name() #   #RRGGBB
        self._updateLabel()

    def textPen(self) -> QtGui.QPen:
        """
        Get the pen used for drawing text.

        If no custom text pen has been set, this method will return a pen with the
        default foreground color.

        Returns
        -------
        QPen
            The pen used to draw text.
        """
        if self._textPen is None:
            return fn.mkPen(getConfigOption('foreground'))
        return fn.mkPen(self._textPen)

    def setTextPen(self, *args, **kwargs):
        """
        Set the pen used for drawing text.

        If no arguments given, the default foreground color will be used.
        
        Parameters
        ----------
        *args
            Arguments relayed to :func:`~pyqtgraph.mkPen`.
        **kwargs
            Arguments relayed to `:func:`~pyqtgraph.mkPen`.

        See Also
        --------
        :func:`setConfigOption <pyqtgraph.setConfigOption>`
            Option to change the default foreground color.
        """     
        self.picture = None
        if args or kwargs:
            self._textPen = fn.mkPen(*args, **kwargs)
        else:
            self._textPen = fn.mkPen(getConfigOption('foreground'))
        self.labelStyle['color'] = self._textPen.color().name() #   #RRGGBB
        self._updateLabel()

    def tickPen(self) -> QtGui.QPen:
        """
        Get the pen used for drawing ticks.

        If no custom tick pen has been set, this method will return the axis's
        main pen.

        Returns
        -------
        QPen
            The pen used to draw tick marks.
        """
        return self.pen() if self._tickPen is None else fn.mkPen(self._tickPen)

    def setTickPen(self, *args, **kwargs):
        """
        Set the pen used for drawing ticks.

        If no arguments given, the default foreground color will be used.
        
        Parameters
        ----------
        *args
            Arguments relayed to :func:`~pyqtgraph.mkPen`.
        **kwargs
            Arguments relayed to `:func:`~pyqtgraph.mkPen`.

        See Also
        --------
        :func:`setConfigOption <pyqtgraph.setConfigOption>`
            Option to change the default foreground color.
        """   
        self.picture = None
        self._tickPen = fn.mkPen(*args, **kwargs) if args or kwargs else None
        self._updateLabel()

    def setScale(self, scale=1.0):
        """
        Set the value scaling for this axis.

        Setting this value causes the axis to draw ticks and tick labels as if the view
        coordinate system were scaled.

        Parameters
        ----------
        scale : float, optional
            Value to scale the drawing of ticks and tick labels as if the view
            coordinate system was scaled, by default 1.0.
        """
        if scale != self.scale:
            self.scale = scale
            self._updateLabel()

    def enableAutoSIPrefix(self, enable=True):
        """
        Enable (or disable) automatic SI prefix scaling on this axis.

        When enabled, this feature automatically determines the best SI prefix
        to prepend to the label units, while ensuring that axis values are scaled
        accordingly.

        For example, if the axis spans values from -0.1 to 0.1 and has units set
        to 'V' then the axis would display values -100 to 100
        and the units would appear as 'mV'

        This feature is enabled by default, and is only available when a suffix
        (unit string) is provided to display on the label.

        Parameters
        ----------
        enable : bool, optional
            Enable Auto SI prefix, by default True.
        """

        self.autoSIPrefix = enable
        self.updateAutoSIPrefix()

    def updateAutoSIPrefix(self):
        self.autoSIPrefixScale, self.labelUnitPrefix = self._autoSIPrefix()
        self._updateLabel()

    def _autoSIPrefix(self) -> tuple[float, str]:
        """
        Compute the automatic SI prefix scaling for the current range.

        Returns
        -------
        tuple of (float, str)
            The scale applied to the tick values and the prefix of the label units;
            ``(1.0, '')`` if the label is hidden or the range is out of the SI prefix
            enable ranges.
        """
        scale = 1.0
        prefix = ''
        if self.label.isVisible():
            _range = 10**np.array(self.range) if self.logMode else self.range
            scaling_value = max(abs(_range[0]), abs(_range[1])) * self.scale
            if any(low <= scaling_value <= high for low, high in self.getSIPrefixEnableRanges()):
                (scale, prefix) = fn.siScale(scaling_value, power=self.unitPower)
        return scale, prefix

    def setRange(self, mn: float, mx: float):
        """
        Set the range of values displayed by the axis.

        Usually this is handled automatically by linking the axis to a ViewBox with
        :func:`linkToView <pyqtgraph.AxisItem.linkToView>`.

        Parameters
        ----------
        mn : float
            Bottom value to set the range to.
        mx : float
            Top value to set the range to.

        Raises
        ------
        ValueError
            When non-finite values are passed.
        """

        if not isfinite(mn) or not isfinite(mx):
            raise ValueError(f"Not setting range to [{mn}, {mx}]")
        self.range = [mn, mx]
        if self.autoSIPrefix:
            scale, prefix = self._autoSIPrefix()
            if scale != self.autoSIPrefixScale or prefix != self.labelUnitPrefix:
                # the label text changes; this also redraws the ticks
                self.autoSIPrefixScale = scale
                self.labelUnitPrefix = prefix
                self._updateLabel()
                return
            # same label text: only the ticks change, skip the costly label update
        self._invalidatePicture()
        self.update()

    def linkedView(self):
        """
        Return the ViewBox linked to this axis.

        Returns
        -------
        ViewBox
            The linked ViewBox, or ``None`` if there is no ViewBox linked.
        """
        return None if self._linkedView is None else self._linkedView()

    def _linkToView_internal(self, view):
        # We need this code to be available without override,
        # even though DateAxisItem overrides the user-side linkToView method
        self.unlinkFromView()

        self._linkedView = weakref.ref(view)
        if self.orientation in ['right', 'left']:
            view.sigYRangeChanged.connect(self.linkedViewChanged)
        else:
            view.sigXRangeChanged.connect(self.linkedViewChanged)
        view.sigResized.connect(self.linkedViewChanged)

    def linkToView(self, view):
        """
        Link to a ViewBox, causing its displayed range to match the view range.

        This is usually called automatically by the ViewBox.

        Parameters
        ----------
        view : ViewBox
            The view to link to.
        """
        self._linkToView_internal(view)

    def unlinkFromView(self):
        """
        Unlink this axis from its linked ViewBox.
        """
        oldView = self.linkedView()
        self._linkedView = None
        if oldView is not None:
            oldView.sigResized.disconnect(self.linkedViewChanged)
            if self.orientation in ['right', 'left']:
                oldView.sigYRangeChanged.disconnect(self.linkedViewChanged)
            else:
                oldView.sigXRangeChanged.disconnect(self.linkedViewChanged)

    @QtCore.Slot(object)
    @QtCore.Slot(object, object)
    def linkedViewChanged(self, view, newRange=None):
        """
        Call when the linked view range has changed.

        Parameters
        ----------
        view : ViewBox
            The view whose range has changed.
        newRange : tuple of float, float, optional
            The new range of the view, by default None.
        """
        if self.orientation in ['right', 'left']:
            if newRange is None:
                newRange = view.viewRange()[1]
            if view.yInverted():
                self.setRange(*newRange[::-1])
            else:
                self.setRange(*newRange)
        else:
            if newRange is None:
                newRange = view.viewRange()[0]
            if view.xInverted():
                self.setRange(*newRange[::-1])
            else:
                self.setRange(*newRange)

    def boundingRect(self):
        m = 0
        hide_overlapping_labels = self.style['hideOverlappingLabels']
        if hide_overlapping_labels is True:
            pass # skip further checks
        elif hide_overlapping_labels is False:
            m = 15
        else:
            try:
                m = int( self.style['hideOverlappingLabels'] )
            except ValueError: pass # ignore any non-numeric value

        linkedView = self.linkedView()
        if linkedView is not None and self.grid is not False:
            return (
                self.mapRectFromParent(self.geometry()) | 
                linkedView.mapRectToItem(self, linkedView.boundingRect())
            )
        rect = self.mapRectFromParent(self.geometry())
        ## extend rect if ticks go in negative direction
        ## also extend to account for text that flows past the edges
        tl = self.style['tickLength']
        if self.orientation == 'left':
            rect = rect.adjusted(0, -m, -min(0,tl), m)
        elif self.orientation == 'right':
            rect = rect.adjusted(min(0,tl), -m, 0, m)
        elif self.orientation == 'top':
            rect = rect.adjusted(-m, 0, m, -min(0,tl))
        elif self.orientation == 'bottom':
            rect = rect.adjusted(-m, min(0,tl), m, 0)
        return rect

    def shape(self):
        # override shape() to exclude grid lines from getting mouse events
        rect = self.mapRectFromParent(self.geometry())
        path = QtGui.QPainterPath()
        path.addRect(rect)
        return path

    def paint(
        self,
        p: QtGui.QPainter,
        opt: QtWidgets.QStyleOptionGraphicsItem | None,
        widget: QtWidgets.QWidget | None
    ) -> None:
        """
        Paint the axis, its ticks and tick labels.

        The picture, the drawing specifications of the axis, is normally built when
        the scene prepares, before Qt computes the regions to repaint (see
        :meth:`_invalidatePicture`). It is built here if it was not, e.g. outside a
        :class:`GraphicsScene <pyqtgraph.GraphicsScene>`. :meth:`drawPicture` then
        draws it.

        Parameters
        ----------
        p : QtGui.QPainter
            The painter, in local coordinates.
        opt : QtWidgets.QStyleOptionGraphicsItem or None
            Style options, unused.
        widget : QtWidgets.QWidget or None
            The widget painted on, unused.
        """
        if self.picture is None or (
            self.style['opaqueGrid'] and self.grid is not False
            # the background changed, or an export started or ended
            and self._gridBackgroundRgb() != self._pictureGridBackground
        ):
            self.picture = self._buildPicture()
        self.picture.play(p)

    def _buildPicture(self, keepSize: bool = False) -> _AxisPicture | None:
        """
        Generate the drawing specifications that :meth:`paint` draws.

        Generating the specifications measures the tick labels with a painter on a
        ``QtGui.QPicture``, which may change the size constraints of the axis (see
        :meth:`_updateMaxTextSize`); the layout applies them later. The specifications
        are drawn into the picture only if drawing them directly when painted may not
        give the pixels of the replayed picture, see :class:`_AxisPicture`.

        Parameters
        ----------
        keepSize : bool, default False
            If True and the size constraints of the axis changed while generating the
            specifications, return None: the specifications are for a geometry about
            to change.

        Returns
        -------
        _AxisPicture or None
            The specifications, or None if ``keepSize`` is True and the size
            constraints changed.
        """
        profiler = debug.Profiler()
        picture = QtGui.QPicture()
        painter = QtGui.QPainter(picture)
        self._buildingPicture = True
        try:
            if self.style["tickFont"]:
                painter.setFont(self.style["tickFont"])
            constraints = self._sizeConstraints() if keepSize else None
            specs = self.generateDrawSpecs(painter)
            profiler('generate specs')
            if constraints is not None and self._sizeConstraints() != constraints:
                return None
            # set even without specs: paint compares it to the current background
            self._pictureGridBackground = (
                self._gridBackgroundRgb()
                if self.style['opaqueGrid'] and self.grid is not False else None
            )
            font = painter.font()
            # a drawPicture override may draw anything: record its drawing, replayed
            direct = (
                _DIRECT_DRAWING
                and specs is not None
                and getattr(self.drawPicture, '__func__', None) is AxisItem.drawPicture
                and _drawnAsReplayed(specs)
            )
            if specs is not None and not direct:
                self.drawPicture(painter, *specs)
                profiler('draw picture')
        finally:
            self._buildingPicture = False
            painter.end()
        if self._labelFonts is None or self._labelFonts[0] != font:
            self._labelFonts = (font, _pinnedFont(font))
        return _AxisPicture(self, specs, self._labelFonts[1], None if direct else picture)

    def _sizeConstraints(self) -> tuple[float, float, float, float]:
        """
        Return the size constraints that the layout applies to the axis.

        Returns
        -------
        tuple of float
            The minimum and maximum widths, then the minimum and maximum heights.
        """
        return (self.minimumWidth(), self.maximumWidth(),
                self.minimumHeight(), self.maximumHeight())

    def _gridBackgroundRgb(self) -> int | None:
        """
        Return the opaque background color under the grid lines, if it is known.

        Grid lines are drawn over the linked view: over the background of the graphics
        views showing the scene, and over the background color of the linked view if
        it is set. The ends of the grid lines can lie a pixel outside the background
        of the linked view: both backgrounds must have the same color.

        Returns
        -------
        int or None
            The background color as a QRgb value, or None while exporting, without a
            linked view, or if the background is not a single opaque color.
        """
        if self._exportOpts is not False:
            return None
        view = self.linkedView()
        scene = self.scene()
        if view is None or scene is None:
            return None
        brushes = [v.backgroundBrush() for v in scene.views()]
        rectItem = getattr(view, 'background', None)
        if isinstance(rectItem, QtWidgets.QGraphicsRectItem) and rectItem.isVisible():
            brushes.append(rectItem.brush())
        colors = set()
        for brush in brushes:
            if brush.style() != QtCore.Qt.BrushStyle.SolidPattern:
                return None
            colors.add(brush.color().rgba())
        if len(colors) != 1:
            return None
        rgb = colors.pop()
        return rgb if rgb >> 24 == 0xFF else None

    def _invalidatePicture(self) -> None:
        """
        Drop the picture of the axis and have it built when the scene prepares.

        In a :class:`GraphicsScene <pyqtgraph.GraphicsScene>`, :meth:`_prepareForPaint`
        is connected to its ``sigPrepareForPaint`` signal until the picture is built,
        and a prepare is requested (see :meth:`GraphicsScene.requestPrepare
        <pyqtgraph.GraphicsScene.requestPrepare>`). The picture is thus built before
        Qt computes the regions to repaint, and a new size of the tick labels is laid
        out before the paint instead of after it, which cost a second paint.
        Otherwise :meth:`paint` builds the picture, as it does for hidden axes and for
        subclasses overriding :meth:`paint`, which may not use the picture. :meth:`paint`
        also rebuilds a picture whose grid pens were blended with another background
        than the current one (see the ``opaqueGrid`` style option).

        Changes made while the picture is built (see :meth:`_buildPicture`) only drop
        the picture: the built picture is assigned afterwards. The size constraints
        set by :meth:`_updateWidth` and :meth:`_updateHeight` do not change the
        picture by themselves; a new geometry reaches :meth:`resizeEvent`.
        """
        self.picture = None
        if self._buildingPicture or not self.isVisible():
            return
        if type(self).paint is not AxisItem.paint:
            return
        scene = self.scene()
        requestPrepare = getattr(scene, 'requestPrepare', None)
        if requestPrepare is None:
            return
        preparing = self._preparingScene
        if preparing is None or preparing() is not scene:
            self._disconnectPrepare()
            # Connected while a picture is pending only, not for as long as the item is
            # in the scene (from itemChange): the signal stays cheap to emit, and
            # itemChange also runs while a scene is torn down, when connecting can
            # crash PySide6.
            scene.sigPrepareForPaint.connect(self._prepareForPaint)
            self._preparingScene = weakref.ref(scene)
        requestPrepare()

    def _disconnectPrepare(self) -> QtWidgets.QGraphicsScene | None:
        """
        Disconnect :meth:`_prepareForPaint` from the scene it is connected to, if any.

        Returns
        -------
        QtWidgets.QGraphicsScene or None
            The scene that was connected, or None if there was none or it was
            deleted.
        """
        preparing, self._preparingScene = self._preparingScene, None
        scene = None if preparing is None else preparing()
        if scene is not None:
            try:
                scene.sigPrepareForPaint.disconnect(self._prepareForPaint)
            except (TypeError, RuntimeError):
                # TypeError and RuntimeError are from PyQt and PySide, respectively
                pass
        return scene

    @QtCore.Slot()
    def _prepareForPaint(self) -> None:
        """
        Build the picture invalidated by :meth:`_invalidatePicture`, then disconnect.

        Connected to ``sigPrepareForPaint`` of the scene only while a picture is
        pending. When the tick labels need a new size, the picture, which would be
        drawn for the previous geometry, is not kept: the scene lays out the new size
        and prepares again (see :meth:`GraphicsScene.event
        <pyqtgraph.GraphicsScene.event>`), and the picture is built for the new
        geometry then.
        """
        scene = self._disconnectPrepare()
        if (
            self.picture is not None
            or scene is None
            or scene is not self.scene()
            or not self.isVisible()
        ):
            return
        picture = self._buildPicture(keepSize=True)
        if picture is None:
            self._invalidatePicture()
        else:
            self.picture = picture


    def setTickDensity(self, density=1.0):
        """
        Set the density of ticks displayed on the axis.

        A higher density value means that more ticks will be displayed. The density
        value is used in conjunction with the tickSpacing method to determine the
        actual tick locations.

        Parameters
        ----------
        density : float, optional
            Density of ticks to display, by default 1.0.
        """
        self._tickDensity = density
        self._invalidatePicture()
        self.update()


    def setTicks(
        self,
        ticks: list[list[tuple[float, str]]] | None
    ):
        """
        Explicitly determine which ticks to display.

        This overrides the behavior specified by
        :meth:`~pyqtgraph.AxisItem.tickSpacing`, :meth:`~pyqtgraph.AxisItem.tickValues`,
        and :meth:`~pyqtgraph.AxisItem.tickStrings`.
        
        The format for *ticks* looks like::

            [
                [
                    (majorTickValue1, majorTickString1),
                    (majorTickValue2, majorTickString2),
                    ...
                ],
                [
                    (minorTickValue1, minorTickString1),
                    (minorTickValue2, minorTickString2),
                    ...
                ],
                ...
            ]
        
        The two levels of major and minor ticks are expected. A third tier of additional
        ticks is optional. If *ticks* is ``None``, then the default tick system will be
        used.

        Parameters
        ----------
        ticks : list of list of float, str or None
            Explicitly set tick display information.
        
        See Also
        --------
        :meth:`~pyqtgraph.AxisItem.tickSpacing`
            How tick spacing is configured.
        :meth:`~pyqtgraph.AxisItem.tickValues`
            How tick values are set.
        :meth:`~pyqtgraph.AxisItem.tickStrings`
            How tick strings are specified.
        """        

        self._tickLevels = ticks
        self._invalidatePicture()
        self.update()

    def setTickSpacing(
        self,
        major: float | None=None,
        minor: float | None=None,
        levels: list[tuple[float, float]] | None=None
    ):
        """
        Explicitly determine the spacing of major and minor ticks.

        This overrides the default behavior of the tickSpacing method, and disables
        the effect of setTicks(). Arguments may be either *major* and *minor*,
        or *levels* which is a list of ``(spacing, offset)`` tuples for each
        tick level desired. If no arguments are given, then the default
        behavior of tickSpacing is enabled.

        Parameters
        ----------
        major : float, optional
            Spacing for major ticks, by default None.
        minor : float, optional
            Spacing for minor ticks, by default None.
        levels : list of tuple of float, float, optional
            A list of (spacing, offset) tuples for each tick level, by default None.

        Examples
        --------
        .. code-block:: python

            # two levels, all offsets = 0
            axis.setTickSpacing(5., 1.)
            # three levels, all offsets = 0
            axis.setTickSpacing(levels=[(3., 0.), (1., 0.), (0.25, 0.)])
            # reset to default
            axis.setTickSpacing()
        """

        if levels is None:
            levels = None if major is None else [(major, 0.), (minor, 0.)]
        self._tickSpacing = levels
        self._invalidatePicture()
        self.update()

    def tickSpacing(self, minVal: float, maxVal: float, size: float):
        """
        Determine the spacing of ticks on the axis.

        This method is called whenever the axis needs to be redrawn and is a
        good method to override in subclasses that require control over tick locations.

        Parameters
        ----------
        minVal : float
            Minimum value being displayed on the axis.
        maxVal : float
            Maximum value being displayed on the axis.
        size : float
            Length of the axis in pixels.

        Returns
        -------
        list of tuple of float, float
            A list of tuples, one for each tick level.
            Each tuple contains two values: ``(spacing, offset)``.  The spacing value
            is the distance between ticks, and the offset is the first tick relative to
            *minVal*. For example, if ``result[0]`` is ``(10, 0)``, then major ticks 
            will be displayed every 10 units and the first major tick will correspond to
            ``minVal``. If instead ``result[0]`` is ``(10, 5)``, then major ticks will
            be displayed every 10 units, but the first major tick will correspond to
            ``minVal + 5``.

            .. code-block:: python

                [
                    (major_tick_spacing, offset),
                    (minor_tick_spacing, offset),
                    (sub_minor_tick_spacing, offset),
                    ...
                ]
        """
        # First check for explicit tick spacing
        if self._tickSpacing is not None:
            return self._tickSpacing

        dif = abs(maxVal - minVal)
        if dif == 0:
            return []

        ref_size = 300. # axes longer than this display more than the minimum number of major ticks
        minNumberOfIntervals = max(
            2.25,       # 2.0 ensures two tick marks. Fudged increase to 2.25 allows room for tick labels.
            2.25 * self._tickDensity * sqrt(size/ref_size) # sub-linear growth of tick spacing with size
        )

        majorMaxSpacing = dif / minNumberOfIntervals

        # We want to calculate the power of 10 just below the maximum spacing.
        # Then divide by ten so that the scale factors for subdivision all become intergers.
        # p10unit = 10**( floor( log10(majorMaxSpacing) ) ) / 10

        # And we want to do it without a log operation:
        mantissa, exp2 = frexp(majorMaxSpacing) # IEEE 754 float already knows its exponent, no need to calculate
        p10unit = 10. ** ( # approximate a power of ten base factor just smaller than the given number
            floor(            # int would truncate towards zero to give wrong results for negative exponents
                (exp2-1)      # IEEE 754 exponent is ceiling of true exponent --> estimate floor by subtracting 1
                / 3.32192809488736 # division by log2(10)=3.32 converts base 2 exponent to base 10 exponent
            ) - 1             # subtract one extra power of ten so that we can work with integer scale factors >= 5
        )
        # neglecting the mantissa can underestimate by one power of 10 when the true value is JUST above the threshold.
        if 100. * p10unit <= majorMaxSpacing: # Cheaper to check this than to use a more complicated approximation.
            majorScaleFactor = 10
            p10unit *= 10.
        else:
            for majorScaleFactor in (50, 20, 10):
                if majorScaleFactor * p10unit <= majorMaxSpacing:
                    break # find the first value that is smaller or equal
        majorInterval = majorScaleFactor * p10unit
        # manual sanity check: print(f"{majorMaxSpacing:.2e} > {majorInterval:.2e} = {majorScaleFactor:.2e} x {p10unit:.2e}")
        levels = [
            (majorInterval, 0),
        ]

        if self.style['maxTickLevel'] >= 1:
            minorMinSpacing = 2 * dif/size   # no more than one minor tick per two pixels
            trials = (5, 10) if majorScaleFactor == 10 else (10, 20, 50)
            for minorScaleFactor in trials:
                minorInterval = minorScaleFactor * p10unit
                if minorInterval >= minorMinSpacing:
                    break # find the first value that is larger or equal to allowed minimum of 1 per 2px
            levels.append((minorInterval, 0))

        # extra ticks at 10% of major interval are pretty, but eat up CPU
        if self.style['maxTickLevel'] >= 2: # consider only when enabled
            if majorScaleFactor == 10:
                trials = (1, 2, 5, 10) # start at 10% of major interval, increase if needed
            elif majorScaleFactor == 20:
                trials = (2, 5, 10, 20) # start at 10% of major interval, increase if needed
            elif majorScaleFactor == 50:
                trials = (5, 10, 50) # start at 10% of major interval, increase if needed
            else: # invalid value
                trials = () # skip extra interval
                extraInterval = minorInterval
            for extraScaleFactor in trials:
                extraInterval = extraScaleFactor * p10unit
                if extraInterval >= minorMinSpacing or extraInterval == minorInterval:
                    break # find the first value that is larger or equal to allowed minimum of 1 per 2px
            if extraInterval < minorInterval: # add extra interval only if it is visible
                levels.append((extraInterval, 0))
        return levels


    def tickValues(
        self,
        minVal: float,
        maxVal: float,
        size: float
    ) -> list[tuple[float | None, list[float]]]:
        """
        Return the values and spacing of ticks to draw.

        The values returned are essentially the same as those returned by
        :meth:`~pyqtgraph.AxisItem.tickSpacing`, but with the addition of
        explicit tick values for each tick level. This method is a good
        method to override in subclasses.

        Parameters
        ----------
        minVal : float
            Minimum value to generate tick values for.
        maxVal : float
            Maximum value to generate tick values for.
        size : float
            The length of the axis in pixels.

        Returns
        -------
        list of tuple of float, list of float
            A list of tuples, one for each tick level. Each tuple contains two
            values: ``(spacing, values)``, where *spacing* is the distance between
            ticks and *values* is a list of tick values.
        """
        minVal, maxVal = sorted((minVal, maxVal))

        minVal *= self.scale
        maxVal *= self.scale

        tickLevels = self.tickSpacing(minVal, maxVal, size)
        ## ticks closer than spacing/100 to a tick of a higher level are removed
        ticks = _tickLevelValues(tickLevels, minVal, maxVal, self.scale)
        if self.logMode:
            return self.logTickValues(minVal, maxVal, size, ticks)
        return ticks

    def logTickValues(self, minVal, maxVal, size, stdTicks):
        """
        Return tick values for log-scale axes.

        This method is called by :meth:`~pyqtgraph.AxisItem.tickValues` when the axis
        is in logarithmic mode. It is a good method to override in subclasses.

        Parameters
        ----------
        minVal : float
            Minimum value to generate tick values for.
        maxVal : float
            Maximum value to generate tick values for.
        size : float
            The length of the axis in pixels.
        stdTicks : list of tuple of float, float
            The tick values generated by the standard
            :meth:`~pyqtgraph.AxisItem.tickValues` method.

        Returns
        -------
        list of tuple of float, float or list of tuple of None, float
            A list of tuples, one for each tick level. Each tuple contains two
            values: ``(spacing, values)``, where *spacing* is the distance between
            ticks and *values* is a list of tick values.
        """

        ## start with the tick spacing given by tickValues().
        ## Any level whose spacing is < 1 needs to be converted to log scale
        ticks = [(spacing, t) for spacing, t in stdTicks if spacing >= 1.0]
        if len(ticks) < 3:
            v1 = int(floor(minVal))
            v2 = int(ceil(maxVal))

            # minor = [v + np.log10(np.arange(1, 10)) for v in range(v1, v2)]
            minor = []
            for v in range(v1, v2):
                minor.extend(v + np.log10(np.arange(1, 10)))
            minor = [x for x in minor if x > minVal and x < maxVal]
            ticks.append((None, minor))
        return ticks

    def tickStrings(self, values: list[float], scale: float, spacing: float):
        """
        Return the strings that should be displayed at each tick value.

        This method is used to generate tick strings, and is called automatically.

        Parameters
        ----------
        values : list of float
            List of tick values.
        scale : float
            The scaling factor for tick values.
        spacing : float
            The spacing between ticks.

        Returns
        -------
        list of str
            List of strings to display at each tick value.
        """

        if self.logMode:
            return self.logTickStrings(values, scale, spacing)

        places = max(0, ceil(-log10(spacing * scale)))
        strings = []
        for v in values:
            vs = v * scale
            if abs(vs) < .001 or abs(vs) >= 10000:
                vstr = "%g" % vs
            else:
                vstr = ("%%0.%df" % places) % vs
            strings.append(vstr)
        return strings

    def logTickStrings(self, values: list[float], scale: float, spacing: float):
        """
        Return the strings that should be displayed at each tick value in log mode.

        This method is called by :meth:`~pyqtgraph.AxisItem.tickStrings` when the axis
        is in logarithmic mode. It is a good method to override in subclasses.

        Parameters
        ----------
        values : list of float
            List of tick values.
        scale : float
            The scaling factor for tick values.
        spacing : float
            The spacing between ticks.

        Returns
        -------
        list of str
            List of strings to display at each tick value.
        """
        estrings = [
            "%0.1g"%x
            for x in 10 ** np.array(values).astype(float) * np.array(scale)
        ]
        convdict = {"0": "⁰",
                    "1": "¹",
                    "2": "²",
                    "3": "³",
                    "4": "⁴",
                    "5": "⁵",
                    "6": "⁶",
                    "7": "⁷",
                    "8": "⁸",
                    "9": "⁹",
                    }
        dstrings = []
        for e in estrings:
            if e.count("e"):
                v, p = e.split("e")
                sign = "⁻" if p[0] == "-" else ""
                pot = "".join([convdict[pp] for pp in p[1:].lstrip("0")])
                v = "" if v == "1" else f"{v}·"
                dstrings.append(f"{v}10{sign}{pot}")
            else:
                dstrings.append(e)
        return dstrings

    def _invalidateTickCaches(self) -> None:
        """
        Forget the cached tick strings of this axis.

        Called when the font, the style, the label or the log mode change (the cache
        keys also hold the state the strings depend on). The shared tick label size
        cache is keyed by font and needs no invalidation.
        """
        self._tickStringsCache.clear()

    def _tickStringsCacheKey(self, spacing: float | None) -> tuple | None:
        """
        Return what the tick strings depend on, besides their arguments.

        The default :meth:`tickStrings` and :meth:`logTickStrings` depend on the log
        mode only. Subclasses whose strings depend on other state must override this
        method; overriding :meth:`tickStrings` or :meth:`logTickStrings` alone
        disables the cache.

        Parameters
        ----------
        spacing : float or None
            The spacing between ticks.

        Returns
        -------
        tuple or None
            Extra key items of the tick strings cache, or None to call
            :meth:`tickStrings` without caching.
        """
        if (
            getattr(self.tickStrings, '__func__', None) is not AxisItem.tickStrings
            or getattr(self.logTickStrings, '__func__', None) is not AxisItem.logTickStrings
        ):
            return None
        return (self.logMode,)

    def _tickStringsCached(
        self,
        values: list[float],
        scale: float,
        spacing: float | None
    ) -> list[str | None]:
        """
        Return ``self.tickStrings(values, scale, spacing)``, from a cache if possible.

        See :meth:`_tickStringsCacheKey` for when the strings are cached.

        Parameters
        ----------
        values : list of float
            Tick values.
        scale : float
            The scaling factor for tick values.
        spacing : float or None
            The spacing between ticks.

        Returns
        -------
        list of str or None
            A new list, which the caller may modify.
        """
        extraKey = self._tickStringsCacheKey(spacing)
        if extraKey is None:
            return self.tickStrings(values, scale, spacing)
        key = (tuple(values), scale, spacing, extraKey)
        strings = self._tickStringsCache.get(key)
        if strings is None:
            strings = tuple(self.tickStrings(values, scale, spacing))
            self._tickStringsCache.put(key, strings)
        return list(strings)

    @staticmethod
    def _tickTextFontKey(p: QtGui.QPainter) -> tuple:
        """
        Identify what the size of a tick label measured with ``p`` depends on.

        Parameters
        ----------
        p : QtGui.QPainter
            The painter used to measure the labels.

        Returns
        -------
        tuple
            The font key, the paint engine type and the resolution of the paint
            device.
        """
        device = p.device()
        engine = p.paintEngine()
        return (
            p.font().key(),
            None if engine is None else engine.type(),
            None if device is None else device.logicalDpiX(),
            None if device is None else device.logicalDpiY(),
        )

    def _measureTickText(self, p: QtGui.QPainter, fontKey: tuple,
                         text: str) -> tuple[float, float]:
        """
        Measure a tick label and store its size in the shared cache.

        Parameters
        ----------
        p : QtGui.QPainter
            The painter used to measure the label, with the tick font set.
        fontKey : tuple
            The value of :meth:`_tickTextFontKey` for ``p``.
        text : str
            The label.

        Returns
        -------
        tuple of float, float
            Width and height of the label.
        """
        br = p.boundingRect(_TEXT_MEASURE_RECT, QtCore.Qt.AlignmentFlag.AlignCenter, text)
        ## boundingRect is usually just a bit too large
        ## (but this probably depends on per-font metrics?)
        size = (br.width(), br.height() * 0.8)
        self._tickTextSizeCache.put((text, fontKey), size)
        return size

    def generateDrawSpecs(self, p: QtGui.QPainter) -> tuple | None:
        """
        Generate the drawing specifications for the axis, ticks, and labels.

        This method determines all the coordinates and other information needed to draw
        the axis, including tick positions, tick labels, and axis label. It returns a
        tuple of values that are used to draw the axis. This is a good method to
        override in subclasses that need more control over the appearance of the axis.

        Tick strings (see :meth:`_tickStringsCacheKey`) and the measured size of each
        label (per font) are cached across calls, so that a range change only formats
        and measures the labels that were not shown before.

        Parameters
        ----------
        p : QPainter
            The painter used to draw the axis.

        Returns
        -------
        tuple or None
            A tuple containing the drawing specifications for the axis, ticks, and
            labels, or None if the axis has no length on the device. The tuple
            contains the following values:

            - ``axisSpec``: A tuple containing the pen, start point, and end point of
              the axis line.
            - ``tickSpecs``: A list of tuples, one for each tick. Each tuple contains
              the pen, start point, and end point of the tick line.
            - ``textSpecs``: A list of tuples, one for each tick label. Each tuple
              contains the bounding rectangle, alignment flags, and text of the label.
        
        :meta private:
        """
        profiler = debug.Profiler()
        if self.style['tickFont'] is not None:
            p.setFont(self.style['tickFont'])
        bounds = self.mapRectFromParent(self.geometry())

        linkedView = self.linkedView()
        if linkedView is None or self.grid is False:
            tickBounds = bounds
        else:
            tickBounds = linkedView.mapRectToItem(self, linkedView.boundingRect())

        left_offset = -1.0
        right_offset = 1.0
        top_offset = -1.0
        bottom_offset = 1.0
        if self.orientation == 'left':
            span = (bounds.topRight() + Point(left_offset, top_offset),
                    bounds.bottomRight() + Point(left_offset, bottom_offset))
            tickStart = tickBounds.right()
            tickStop = bounds.right()
            tickDir = -1
            axis = 0
        elif self.orientation == 'right':
            span = (bounds.topLeft() + Point(right_offset, top_offset),
                    bounds.bottomLeft() + Point(right_offset, bottom_offset))
            tickStart = tickBounds.left()
            tickStop = bounds.left()
            tickDir = 1
            axis = 0
        elif self.orientation == 'top':
            span = (bounds.bottomLeft() + Point(left_offset, top_offset),
                    bounds.bottomRight() + Point(right_offset, top_offset))
            tickStart = tickBounds.bottom()
            tickStop = bounds.bottom()
            tickDir = -1
            axis = 1
        elif self.orientation == 'bottom':
            span = (bounds.topLeft() + Point(left_offset, bottom_offset),
                    bounds.topRight() + Point(right_offset, bottom_offset))
            tickStart = tickBounds.top()
            tickStop = bounds.top()
            tickDir = 1
            axis = 1
        else:
            raise ValueError(
                "self.orientation must be in {'left', 'right', 'top', 'bottom'}"
            )
        ## determine size of this item in pixels
        points = list(map(self.mapToDevice, span))
        if None in points:
            return
        lengthInPixels = Point(points[1] - points[0]).length()
        if lengthInPixels == 0:
            return

        # Determine major / minor / subminor axis ticks
        if self._tickLevels is None:
            tickLevels = self.tickValues(self.range[0], self.range[1], lengthInPixels)
            tickStrings = None
        else:
            ## parse self.tickLevels into the formats returned by tickLevels() and tickStrings()
            tickLevels = []
            tickStrings = []
            for level in self._tickLevels:
                values = []
                strings = []
                tickLevels.append((None, values))
                tickStrings.append(strings)
                for val, strn in level:
                    values.append(val)
                    strings.append(strn)

        ## determine mapping between tick values and local coordinates
        dif = self.range[1] - self.range[0]
        if dif == 0:
            xScale = 1
            offset = 0
        else:
            if axis == 0:
                xScale = -bounds.height() / dif
            else:
                xScale = bounds.width() / dif
            if not isfinite(xScale):
                xScale = copysign(sys.float_info.max, xScale)
            if axis == 0:
                offset = self.range[0] * xScale - bounds.height()
            else:
                offset = self.range[0] * xScale

        xRange = [x * xScale - offset for x in self.range]
        xMin = min(xRange)
        xMax = max(xRange)

        profiler('init')

        tickPositions = [] # remembers positions of previously drawn ticks

        ## compute coordinates to draw ticks
        ## draw three different intervals, long ticks first
        tickSpecs = []
        for i in range(len(tickLevels)):
            ticks = tickLevels[i][1]

            ## length of tick
            tickLength = self.style['tickLength'] / ((i*0.5)+1.0)

            lineAlpha = self.style["tickAlpha"]
            if lineAlpha is None:
                lineAlpha = 255 / (i+1)
                if self.grid is not False:
                    lineAlpha *= self.grid/255. * fn.clip_scalar((0.05  * lengthInPixels / (len(ticks)+1)), 0., 1.)
            elif isinstance(lineAlpha, float):
                lineAlpha *= 255
                lineAlpha = max(0, int(round(lineAlpha)))
                lineAlpha = min(255, int(round(lineAlpha)))
            elif isinstance(lineAlpha, int):
                if (lineAlpha > 255) or (lineAlpha < 0):
                    raise ValueError("lineAlpha should be [0..255]")
            else:
                raise TypeError("Line Alpha should be of type None, float or int")
            tickPen = self.tickPen()
            if tickPen.brush().style() == QtCore.Qt.BrushStyle.SolidPattern: # only adjust simple color pens
                tickPen = QtGui.QPen(tickPen) # copy to a new QPen
                color = QtGui.QColor(tickPen.color()) # copy to a new QColor
                color.setAlpha(int(lineAlpha)) # adjust opacity
                tickPen.setColor(color)

            ## coordinates of both tick ends along the axis normal
            tickEnd = tickStop
            if self.grid is False:
                tickEnd += tickLength*tickDir
            tickStartF = float(tickStart)
            tickEndF = float(tickEnd)
            ## determine actual position to draw each tick; None for out-of-bounds
            ## ticks, which are not drawn
            positions = [(v * xScale) - offset for v in ticks]
            positions = [None if x < xMin or x > xMax else x for x in positions]
            tickPositions.append(positions)
            visible = [float(x) for x in positions if x is not None]
            if axis == 0:
                tickSpecs.extend([(tickPen, _point(tickStartF, xf), _point(tickEndF, xf))
                                  for xf in visible])
            else:
                tickSpecs.extend([(tickPen, _point(xf, tickStartF), _point(xf, tickEndF))
                                  for xf in visible])
        profiler('compute ticks')


        if self.style['stopAxisAtTick'][0] is True:
            minTickPosition = min(map(min, tickPositions))
            if axis == 0:
                stop = max(span[0].y(), minTickPosition)
                span[0].setY(stop)
            else:
                stop = max(span[0].x(), minTickPosition)
                span[0].setX(stop)
        if self.style['stopAxisAtTick'][1] is True:
            maxTickPosition = max(map(max, tickPositions))
            if axis == 0:
                stop = min(span[1].y(), maxTickPosition)
                span[1].setY(stop)
            else:
                stop = min(span[1].x(), maxTickPosition)
                span[1].setX(stop)
        axisSpec = (self.pen(), span[0], span[1])


        textOffset = self.style['tickTextOffset'][axis]  ## spacing between axis and text
        textSize2 = 0
        lastTextSize2 = 0
        textRects = []
        textSpecs = []  ## list of draw

        # If values are hidden, return early
        if not self.style['showValues']:
            return (axisSpec, tickSpecs, textSpecs)

        ## Constant for all labels: text measurement key, label flags, label offset
        ## from the axis and clipping rectangle
        fontKey = self._tickTextFontKey(p)
        textSizeCache = self._tickTextSizeCache
        textFlags = _TICK_TEXT_FLAGS[self.orientation]
        labelOffset = max(0,self.style['tickLength']) + textOffset
        br = self.boundingRect()

        for i in range(min(len(tickLevels), self.style['maxTextLevel']+1)):
            ## Get the list of strings to display for this level
            if tickStrings is None:
                spacing, values = tickLevels[i]
                strings = self._tickStringsCached(values, self.autoSIPrefixScale * self.scale, spacing)
            else:
                strings = tickStrings[i]

            if len(strings) == 0:
                continue

            ## ignore strings belonging to ticks that were previously ignored
            for j in range(len(strings)):
                if tickPositions[i][j] is None:
                    strings[j] = None

            ## Measure density of text; decide whether to draw this level
            ## (sizes are (width, height) tuples, measured once per string and font)
            rects = []
            for s in strings:
                if s is None:
                    rects.append(None)
                else:
                    size = textSizeCache.get((s, fontKey))
                    if size is None:
                        size = self._measureTickText(p, fontKey, s)
                    rects.append(size)
                    textRects.append(size)

            ## measure all text, make sure there's enough room: textSize along the
            ## axis, textSize2 across it (sizes are finite: max equals np.max)
            textSize2 = max([r[axis] for r in textRects]) if textRects else 0

            if i > 0:  ## always draw top level
                ## If the strings are too crowded, stop drawing text now.
                ## We use three different crowding limits based on the number
                ## of texts drawn so far.
                textSize = np.sum([r[1 - axis] for r in textRects]) if textRects else 0
                textFillRatio = float(textSize) / lengthInPixels
                finished = False
                for nTexts, limit in self.style['textFillLimits']:
                    if len(textSpecs) >= nTexts and textFillRatio >= limit:
                        finished = True
                        break
                if finished:
                    break

            lastTextSize2 = textSize2

            # Determine exactly where tick text should be drawn
            positions = tickPositions[i]
            for j in range(len(strings)):
                vstr = strings[j]
                if vstr is None: ## this tick was ignored because it is out of bounds
                    continue
                x = positions[j]
                width, height = rects[j]

                if self.orientation == 'left':
                    rect = QtCore.QRectF(tickStop-labelOffset-width, x-(height/2), width, height)
                elif self.orientation == 'right':
                    rect = QtCore.QRectF(tickStop+labelOffset, x-(height/2), width, height)
                elif self.orientation == 'top':
                    rect = QtCore.QRectF(x-width/2., tickStop-labelOffset-height, width, height)
                else:  # 'bottom'
                    rect = QtCore.QRectF(x-width/2., tickStop+labelOffset, width, height)

                # br.contains(rect) suffers from floating point rounding errors
                if br & rect != rect:
                    continue

                textSpecs.append((rect, textFlags, vstr))
        profiler('compute text')

        ## update max text size if needed.
        self._updateMaxTextSize(lastTextSize2)

        return axisSpec, tickSpecs, textSpecs

    def drawPicture(
        self,
        p: QtGui.QPainter,
        axisSpec: tuple,
        tickSpecs: list[tuple],
        textSpecs: list[tuple]
    ) -> None:
        """
        Draw the axis line, the ticks and the tick labels.

        :meth:`paint` calls this method with the specifications generated by
        :meth:`generateDrawSpecs`. The ticks of each run sharing a pen object (a tick
        level) are drawn with a single ``drawLines`` call, which takes the end points of
        the ticks as point pairs. The text antialiasing hint of ``p`` is used as it is:
        the axis used to draw into a ``QtGui.QPicture``, which does not replay that
        hint. When :meth:`paint` draws directly, the tick font is not set: ``p`` has
        the font the labels were measured with (see :class:`_AxisPicture`).

        Parameters
        ----------
        p : QtGui.QPainter
            The painter, in the local coordinates of the axis.
        axisSpec : tuple
            The pen, start point and end point of the axis line.
        tickSpecs : list of tuple
            The pen, start point and end point (``QtCore.QPointF``) of each tick.
        textSpecs : list of tuple
            The bounding rectangle, alignment flags and text of each tick label.
        """
        profiler = debug.Profiler()

        p.setRenderHint(p.RenderHint.Antialiasing, False)

        ## draw long line along axis
        pen, p1, p2 = axisSpec
        p.setPen(pen)
        p.drawLine(p1, p2)
        # p.translate(0.5,0)  ## resolves some damn pixel ambiguity

        ## draw ticks: one drawLines call per run of ticks sharing a pen (a tick level)
        background = self._pictureGridBackground
        points = []
        lastPen = None
        for pen, p1, p2 in tickSpecs:
            if pen is not lastPen:
                if points:
                    p.drawLines(points)
                    points = []
                lastPen = pen
                if background is not None:
                    pen = _opaqueGridPen(pen, background)
                p.setPen(pen)
            points.append(p1)
            points.append(p2)
        if points:
            p.drawLines(points)
        profiler('draw ticks')

        # Draw all text; when drawing directly, the painter has the font the labels
        # were measured with, the tick font as a picture recorded it
        if self.style['tickFont'] is not None and not self._drawingDirectly:
            p.setFont(self.style['tickFont'])
        p.setPen(self.textPen())
        bounding = self.boundingRect().toAlignedRect()
        p.setClipRect(bounding)
        for rect, flags, text in textSpecs:
            p.drawText(rect, int(flags), text)

        profiler('draw text')

    def show(self):
        super().show()
        if self.orientation in ['left', 'right']:
            self._updateWidth()
        else:
            self._updateHeight()

    def hide(self):
        super().hide()
        if self.orientation in ['left', 'right']:
            self._updateWidth()
        else:
            self._updateHeight()

    def wheelEvent(self, event):
        lv = self.linkedView()
        if lv is None:
            return
        # Did the event occur inside the linked ViewBox (and not over the axis itself)?
        if lv.sceneBoundingRect().contains(event.scenePos()):
            event.ignore()
            return
        else:
            # pass event to linked viewbox with appropriate single axis zoom parameter
            if self.orientation in ['left', 'right']:
                lv.wheelEvent(event, axis=1)
            else:
                lv.wheelEvent(event, axis=0)
        event.accept()

    def mouseDragEvent(self, event):
        lv = self.linkedView()
        if lv is None:
            return
        # Did the mouse down event occur inside the linked ViewBox (and not the axis)?
        if lv.sceneBoundingRect().contains(event.buttonDownScenePos()):
            event.ignore()
            return
        # otherwise pass event to linked viewbox with appropriate single axis parameter
        if self.orientation in ['left', 'right']:
            return lv.mouseDragEvent(event, axis=1)
        else:
            return lv.mouseDragEvent(event, axis=0)

    def mouseClickEvent(self, event):
        lv = self.linkedView()
        if lv is None:
            return
        return lv.mouseClickEvent(event)
