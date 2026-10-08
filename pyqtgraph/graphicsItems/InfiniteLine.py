from collections.abc import Sequence
from math import atan2, degrees, hypot
from typing import Any

import numpy as np

from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from .GraphicsItem import GraphicsItem
from .GraphicsObject import GraphicsObject
from .TextItem import TextItem
from .ViewBox import ViewBox

__all__ = ['InfiniteLine', 'InfLineLabel']


class InfiniteLine(GraphicsObject):
    """
    **Bases:** :class:`GraphicsObject <pyqtgraph.GraphicsObject>`

    Displays a line of infinite length.
    This line may be dragged to indicate a position in data coordinates.

    =============================== ===================================================
    **Signals:**
    sigDragged(self)
    sigPositionChangeFinished(self)
    sigPositionChanged(self)
    sigClicked(self, ev)
    =============================== ===================================================
    """

    sigDragged = QtCore.Signal(object)
    sigPositionChangeFinished = QtCore.Signal(object)
    sigPositionChanged = QtCore.Signal(object)
    sigClicked = QtCore.Signal(object, object)

    # Item changes after which the bounds, expressed in local coordinates, must be
    # recomputed: they move the line relative to its view.
    _geometryChanges = (
        QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged,
        QtWidgets.QGraphicsItem.GraphicsItemChange.ItemTransformHasChanged,
        QtWidgets.QGraphicsItem.GraphicsItemChange.ItemRotationHasChanged,
        QtWidgets.QGraphicsItem.GraphicsItemChange.ItemScaleHasChanged,
        QtWidgets.QGraphicsItem.GraphicsItemChange.ItemTransformOriginPointHasChanged,
    )

    def __init__(self, pos: float | Sequence[float] | QtCore.QPointF | None = None,
                 angle: float = 90, pen: Any = None, movable: bool = False,
                 bounds: Sequence[float | None] | None = None, hoverPen: Any = None,
                 label: str | None = None, labelOpts: dict[str, Any] | None = None,
                 span: tuple[float, float] = (0, 1),
                 markers: Sequence[tuple[str, float, float]] | None = None,
                 name: str | None = None) -> None:
        """
        Parameters
        ----------
        pos : float or sequence of float or QtCore.QPointF, optional
            Position of the line. This can be a QPointF or a single value for
            vertical/horizontal lines.
        angle : float, default 90
            Angle of line in degrees. 0 is horizontal, 90 is vertical.
        pen : Any, optional
            Pen to use when drawing line. Can be any arguments that are valid
            for :func:`mkPen <pyqtgraph.mkPen>`. Default pen is transparent
            yellow.
        movable : bool, default False
            If True, the line can be dragged to a new position by the user.
        bounds : sequence of float or None, optional
            Optional [min, max] bounding values. Bounds are only valid if the
            line is vertical or horizontal.
        hoverPen : Any, optional
            Pen to use when drawing line when the mouse cursor hovers over it (only
            used when movable=True). Can be any arguments that are valid for
            :func:`mkPen <pyqtgraph.mkPen>`. Default pen is red.
        label : str, optional
            Text to be displayed in a label attached to the line, or
            None to show no label (default is None). May optionally
            include formatting strings to display the line value.
        labelOpts : dict, optional
            A dict of keyword arguments to use when constructing the
            text label. See :class:`InfLineLabel <pyqtgraph.graphicsItems.InfiniteLine.InfLineLabel>`.
        span : tuple of float, default (0, 1)
            Optional tuple (min, max) giving the range over the view to draw
            the line. For example, with a vertical line, use span=(0.5, 1)
            to draw only on the top half of the view.
        markers : sequence of tuple, optional
            List of (marker, position, size) tuples, one per marker to display
            on the line. See the addMarker method.
        name : str, optional
            Name of the item
        """
        # Cache variables for managing bounds. They are kept up to date by
        # _updateBoundingRect, which is enabled at the end of __init__.
        self._boundingRect = QtCore.QRectF()
        self._endPoints = (0, 1)
        self._line = QtCore.QLineF(0.0, 0.0, 1.0, 0.0)
        self._lastViewSize = None
        self._trackGeometry = False

        self._name = name

        GraphicsObject.__init__(self)

        if bounds is None:              ## allowed value boundaries for orthogonal lines
            self.maxRange = [None, None]
        else:
            self.maxRange = bounds
        self.moving = False
        self.setMovable(movable)
        self.mouseHovering = False
        self.p = [0, 0]
        self.setAngle(angle)

        if pos is None:
            pos = Point(0,0)
        self.setPos(pos)

        if pen is None:
            pen = (200, 200, 100)
        self.setPen(pen)
        
        if hoverPen is None:
            self.setHoverPen(color=(255,0,0), width=self.pen.width())
        else:
            self.setHoverPen(hoverPen)
        
        self.span = span
        self.currentPen = self.pen

        self.markers = []
        self._maxMarkerSize = 0
        if markers is not None:
            for m in markers:
                self.addMarker(*m)

        self._trackGeometry = True
        self._updateBoundingRect()

        if label is not None:
            labelOpts = {} if labelOpts is None else labelOpts
            self.label = InfLineLabel(self, text=label, **labelOpts)

    def setMovable(self, m):
        """Set whether the line is movable by the user."""
        self.movable = m
        self.setAcceptHoverEvents(m)

    def setBounds(self, bounds):
        """Set the (minimum, maximum) allowable values when dragging."""
        self.maxRange = bounds
        self.setValue(self.value())
        
    def bounds(self):
        """Return the (minimum, maximum) values allowed when dragging.
        """
        return self.maxRange[:]
        
    def setPen(self, *args: Any, **kwargs: Any) -> None:
        """
        Set the pen for drawing the line.

        Parameters
        ----------
        *args, **kwargs : Any
            Any arguments that are valid for :func:`mkPen <pyqtgraph.mkPen>`.
        """
        self.pen = fn.mkPen(*args, **kwargs)
        self._updateBoundingRect()  # the bounds depend on the pen width
        if not self.mouseHovering:
            self.currentPen = self.pen
            self.update()

    def setHoverPen(self, *args: Any, **kwargs: Any) -> None:
        """
        Set the pen for drawing the line while the mouse hovers over it.

        If the line is not movable, then hovering is also disabled.

        Added in version 0.9.9.

        Parameters
        ----------
        *args, **kwargs : Any
            Any arguments that are valid for :func:`mkPen <pyqtgraph.mkPen>`. If no
            width is given, the width of the regular pen is used.
        """
        # If user did not supply a width, then copy it from pen
        widthSpecified = ((len(args) == 1 and 
                           (isinstance(args[0], QtGui.QPen) or
                           (isinstance(args[0], dict) and 'width' in args[0]))
                          ) or 'width' in kwargs)
        self.hoverPen = fn.mkPen(*args, **kwargs)
        if not widthSpecified:
            self.hoverPen.setWidth(self.pen.width())
        self._updateBoundingRect()  # the bounds depend on the hover pen width

        if self.mouseHovering:
            self.currentPen = self.hoverPen
            self.update()
        
    def addMarker(self, marker: str, position: float = 0.5, size: float = 10.0) -> None:
        """
        Add a marker to be displayed on the line.

        Parameters
        ----------
        marker : str
            String indicating the style of marker to add:
            ``'<|'``, ``'|>'``, ``'>|'``, ``'|<'``, ``'<|>'``,
            ``'>|<'``, ``'^'``, ``'v'``, ``'o'``
        position : float, default 0.5
            Position (0.0-1.0) along the visible extent of the line
            to place the marker.
        size : float, default 10.0
            Size of the marker in pixels.
        """
        path = QtGui.QPainterPath()
        if marker == 'o': 
            path.addEllipse(QtCore.QRectF(-0.5, -0.5, 1, 1))
        if '<|' in marker:
            p = QtGui.QPolygonF([Point(0.5, 0), Point(0, -0.5), Point(-0.5, 0)])
            path.addPolygon(p)
            path.closeSubpath()
        if '|>' in marker:
            p = QtGui.QPolygonF([Point(0.5, 0), Point(0, 0.5), Point(-0.5, 0)])
            path.addPolygon(p)
            path.closeSubpath()
        if '>|' in marker:
            p = QtGui.QPolygonF([Point(0.5, -0.5), Point(0, 0), Point(-0.5, -0.5)])
            path.addPolygon(p)
            path.closeSubpath()
        if '|<' in marker:
            p = QtGui.QPolygonF([Point(0.5, 0.5), Point(0, 0), Point(-0.5, 0.5)])
            path.addPolygon(p)
            path.closeSubpath()
        if '^' in marker:
            p = QtGui.QPolygonF([Point(0, -0.5), Point(0.5, 0), Point(0, 0.5)])
            path.addPolygon(p)
            path.closeSubpath()
        if 'v' in marker:
            p = QtGui.QPolygonF([Point(0, -0.5), Point(-0.5, 0), Point(0, 0.5)])
            path.addPolygon(p)
            path.closeSubpath()
        
        self.markers.append((path, position, size))
        self._maxMarkerSize = max([m[2] / 2. for m in self.markers])
        self._updateBoundingRect()
        self.update()

    def clearMarkers(self) -> None:
        """Remove all markers from this line."""
        self.markers = []
        self._maxMarkerSize = 0
        self._updateBoundingRect()
        self.update()
        
    def setAngle(self, angle):
        """
        Takes angle argument in degrees.
        0 is horizontal; 90 is vertical.

        Note that the use of value() and setValue() changes if the line is
        not vertical or horizontal.
        """
        self.angle = angle #((angle+45) % 180) - 45   ##  -45 <= angle < 135
        self.resetTransform()
        self.setRotation(self.angle)
        self.update()

    def setPos(self, pos: float | Sequence[float] | QtCore.QPointF) -> None:
        """
        Set the position of the line.

        Parameters
        ----------
        pos : float or sequence of float or QtCore.QPointF
            New position. A single value is accepted for horizontal and vertical
            lines; it is clipped to :meth:`bounds` for these lines.
        """
        if isinstance(pos, (list, tuple, np.ndarray)) and not np.ndim(pos) == 0:
            newPos = list(pos)
        elif isinstance(pos, QtCore.QPointF):
            newPos = [pos.x(), pos.y()]
        else:
            if self.angle == 90:
                newPos = [pos, 0]
            elif self.angle == 0:
                newPos = [0, pos]
            else:
                raise Exception("Must specify 2D coordinate for non-orthogonal lines.")

        ## check bounds (only works for orthogonal lines)
        if self.angle == 90:
            if self.maxRange[0] is not None:
                newPos[0] = max(newPos[0], self.maxRange[0])
            if self.maxRange[1] is not None:
                newPos[0] = min(newPos[0], self.maxRange[1])
        elif self.angle == 0:
            if self.maxRange[0] is not None:
                newPos[1] = max(newPos[1], self.maxRange[0])
            if self.maxRange[1] is not None:
                newPos[1] = min(newPos[1], self.maxRange[1])

        if self.p != newPos:
            self.p = newPos
            # itemChange updates the cached bounds before the signal is emitted
            GraphicsObject.setPos(self, Point(self.p))
            self.sigPositionChanged.emit(self)

    def getXPos(self):
        return self.p[0]

    def getYPos(self):
        return self.p[1]

    def getPos(self):
        return self.p

    def value(self):
        """Return the value of the line. Will be a single number for horizontal and
        vertical lines, and a list of [x,y] values for diagonal lines."""
        if self.angle%180 == 0:
            return self.getYPos()
        elif self.angle%180 == 90:
            return self.getXPos()
        else:
            return self.getPos()

    def setValue(self, v):
        """Set the position of the line. If line is horizontal or vertical, v can be
        a single value. Otherwise, a 2D coordinate must be specified (list, tuple and
        QPointF are all acceptable)."""
        self.setPos(v)
    
    def setSpan(self, mn: float, mx: float) -> None:
        """
        Set the fraction of the view, along the line, over which it is drawn.

        Parameters
        ----------
        mn, mx : float
            Start and end of the line, from 0 to 1. For example, use ``(0.5, 1)`` to
            draw a vertical line on the top half of the view only.
        """
        if self.span != (mn, mx):
            self.span = (mn, mx)
            self._updateBoundingRect()
            self.update()

    def _orthoPixelSize(self, axisAligned: bool) -> float:
        """
        Return the size of a device pixel orthogonal to the line, in local units.

        Parameters
        ----------
        axisAligned : bool
            True if the local axes of the line are parallel to the axes of its view.
            When the device transform is axis-aligned too (the view is neither
            rotated nor sheared on screen), the pixel size is read from it directly
            instead of the generic, much slower, :meth:`pixelVectors`.

        Returns
        -------
        float
            Length of one pixel orthogonal to the line, or 0 if the item is not
            displayed yet.
        """
        if axisAligned:
            dt = self.deviceTransform_()
            if dt is None:
                return 0.0
            # The line runs along the local x axis. When the device transform maps
            # the local axes onto the device axes, a pixel orthogonal to the line is
            # 1 / (device scale of the local y axis).
            if dt.m12() == 0.0 and dt.m21() == 0.0 and dt.m22() != 0.0:
                return 1.0 / abs(dt.m22())
            if dt.m11() == 0.0 and dt.m22() == 0.0 and dt.m21() != 0.0:
                return 1.0 / abs(dt.m21())
        _, ortho = self.pixelVectors(direction=QtCore.QPointF(1.0, 0.0))
        return 0.0 if ortho is None else abs(ortho.y())

    def _visibleExtent(self) -> tuple[float, float, bool] | None:
        """
        Return the extent of the view along the line, in local x coordinates.

        Horizontal and vertical lines placed in a :class:`ViewBox` (the usual case)
        read the view range directly. Other lines map the view rectangle to local
        coordinates with the generic, slower, :meth:`viewRect`.

        Returns
        -------
        start : float
            Local x coordinate where the view starts along the line.
        length : float
            Length of the view along the line, in local units.
        axisAligned : bool
            True if the local axes of the line are parallel to the view axes.
        None
            Returned when the line is not in a view.
        """
        vb = self.getViewBox()
        if vb is None:
            return None
        if isinstance(vb, ViewBox):
            # transform from local to view coordinates
            tr = self.itemTransform(vb.innerSceneItem())[0]
            if tr.isAffine():
                m11 = tr.m11()
                m12 = tr.m12()
                # The arithmetic below matches the generic mapping of the view
                # rectangle (QTransform.mapRect), so that the end points are the same.
                if m11 == 1.0 and m12 == 0.0 and tr.m21() == 0.0:
                    # local x is view x shifted: horizontal line (any y scale or flip)
                    (x0, x1), _ = vb.viewRange()
                    return x0 - tr.dx(), x1 - x0, True
                if m11 == 0.0 and m12 == 1.0 and tr.m22() == 0.0:
                    # local x is view y shifted: vertical line (rotated by 90 degrees)
                    _, (y0, y1) = vb.viewRange()
                    top = y0 - tr.dy()
                    bottom = (y0 + (y1 - y0)) - tr.dy()  # rounded as QRectF.bottom()
                    return top, bottom - top, True
        vr = self.viewRect()  # bounds of the containing view mapped to local coords
        if vr is None:
            return None
        return vr.left(), vr.width(), False

    def _computeBoundingRect(self) -> tuple[QtCore.QRectF, float, float] | None:
        """
        Compute the bounding rectangle and the visible end points of the line.

        This method has no side effect; :meth:`_updateBoundingRect` stores its
        result and notifies Qt of geometry changes.

        Returns
        -------
        rect : QtCore.QRectF
            Bounding rectangle in local coordinates, padded by the pen width, the
            marker size and one pixel orthogonally to the line.
        left, right : float
            Local x coordinates of the end points of the line, given the span.
        None
            Returned when the line is not in a view.
        """
        extent = self._visibleExtent()
        if extent is None:
            return None
        start, length, axisAligned = extent
        px = self._orthoPixelSize(axisAligned)
        pw = max(self.pen.width() / 2, self.hoverPen.width() / 2)
        w = (self._maxMarkerSize + pw + 1) * px
        left = start + length * self.span[0]
        right = start + length * self.span[1]
        rect = QtCore.QRectF(left, -w, right - left, 2 * w).normalized()
        return rect, left, right

    def _updateBoundingRect(self) -> None:
        """
        Recompute the cached bounding rectangle and visible end points of the line.

        Called whenever the bounds may have changed: view transform, position or
        transform of the line, pens, markers and span. ``prepareGeometryChange`` is
        called *before* the cached rectangle changes, as required by Qt, so that
        :meth:`boundingRect` only reads the cache.
        """
        if not self._trackGeometry:
            return  # still in __init__
        computed = self._computeBoundingRect()
        vb = self.getViewBox()
        viewSize = None if vb is None else vb.size()
        if computed is None:
            rect = QtCore.QRectF()
        else:
            rect, left, right = computed
            if (left, right) != self._endPoints:
                self._endPoints = (left, right)
                self._line = QtCore.QLineF(left, 0.0, right, 0.0)
        if rect != self._boundingRect or viewSize != self._lastViewSize:
            self.prepareGeometryChange()
            self._boundingRect = rect
            self._lastViewSize = viewSize

    def boundingRect(self) -> QtCore.QRectF:
        """
        Return the cached bounding rectangle of the line.

        Returns
        -------
        QtCore.QRectF
            Bounding rectangle in local coordinates; it spans the visible part of the
            view along the line. The cache is kept up to date by
            :meth:`_updateBoundingRect`.
        """
        return self._boundingRect

    def _getVisibleEndpoints(self) -> tuple[QtCore.QPointF, QtCore.QPointF]:
        """
        Return the end points of the line clipped to its view.

        Returns
        -------
        pt1, pt2 : QtCore.QPointF
            End points in local coordinates. Oblique lines are clipped to the
            bounding rectangle of their :class:`ViewBox`.
        """
        line = self._line
        pt1 = line.p1()
        pt2 = line.p2()

        if self.angle % 90 != 0:
            view = self.getViewBox()
            if not isinstance(view, ViewBox):
                return pt1, pt2

            path = QtGui.QPainterPath()
            path.moveTo(pt1)
            path.lineTo(pt2)
            path = self.itemTransform(view)[0].map(path)

            bounds = QtGui.QPainterPath()
            bounds.addRect(view.boundingRect())

            paths = bounds.intersected(path).toSubpathPolygons(QtGui.QTransform())
            if len(paths) > 0:
                pts = list(paths[0])
                pt1 = self.mapFromItem(view, pts[0])
                pt2 = self.mapFromItem(view, pts[1])

        return pt1, pt2

    def paint(self, p: QtGui.QPainter, *args: Any) -> None:
        """
        Draw the line and its markers.

        Parameters
        ----------
        p : QtGui.QPainter
            Painter, in local coordinates.
        *args : Any
            Style option and widget, unused.
        """
        if self.angle % 180 not in (0, 90):
            p.setRenderHint(p.RenderHint.Antialiasing)

        pen = self.currentPen
        pen.setJoinStyle(QtCore.Qt.PenJoinStyle.MiterJoin)
        p.setPen(pen)
        p.drawLine(self._line)

        if len(self.markers) == 0:
            return

        # paint markers in native coordinate system
        tr = p.transform()
        p.resetTransform()

        pt1, pt2 = self._getVisibleEndpoints()
        start = tr.map(pt1)
        end = tr.map(pt2)
        up = tr.map(pt1 + QtCore.QPointF(0.0, 1.0))
        dif = end - start
        length = hypot(dif.x(), dif.y())
        angle = degrees(atan2(dif.y(), dif.x()))

        p.translate(start)
        p.rotate(angle)

        up = up - start
        det = up.x() * dif.y() - dif.x() * up.y()
        p.scale(1, 1 if det > 0 else -1)

        p.setBrush(fn.mkBrush(self.currentPen.color()))
        #p.setPen(fn.mkPen(None))
        tr = p.transform()
        for path, pos, size in self.markers:
            p.setTransform(tr)
            x = length * pos
            p.translate(x, 0)
            p.scale(size, size)
            p.drawPath(path)

    def dataBounds(self, axis, frac=1.0, orthoRange=None):
        if axis == 0:
            return None   ## x axis should never be auto-scaled
        else:
            return (0,0)

    def mouseDragEvent(self, ev):
        if self.movable and ev.button() == QtCore.Qt.MouseButton.LeftButton:
            if ev.isStart():
                self.moving = True
                self.cursorOffset = self.pos() - self.mapToParent(ev.buttonDownPos())
                self.startPosition = self.pos()
            ev.accept()

            if not self.moving:
                return

            self.setPos(self.cursorOffset + self.mapToParent(ev.pos()))
            self.sigDragged.emit(self)
            if ev.isFinish():
                self.moving = False
                self.sigPositionChangeFinished.emit(self)

    def mouseClickEvent(self, ev):
        self.sigClicked.emit(self, ev)
        if self.moving and ev.button() == QtCore.Qt.MouseButton.RightButton:
            ev.accept()
            self.setPos(self.startPosition)
            self.moving = False
            self.sigDragged.emit(self)
            self.sigPositionChangeFinished.emit(self)

    def hoverEvent(self, ev):
        if (not ev.isExit()) and self.movable and ev.acceptDrags(QtCore.Qt.MouseButton.LeftButton):
            self.setMouseHover(True)
        else:
            self.setMouseHover(False)

    def setMouseHover(self, hover):
        ## Inform the item that the mouse is (not) hovering over it
        if self.mouseHovering == hover:
            return
        self.mouseHovering = hover
        if hover:
            self.currentPen = self.hoverPen
        else:
            self.currentPen = self.pen
        self.update()

    def viewTransformChanged(self) -> None:
        """
        Called whenever the transformation matrix of the view has changed.
        (eg, the view range has changed or the view was resized)

        The cached bounds are recomputed here, and Qt is notified if they changed,
        so that :meth:`boundingRect` never has to.
        """
        GraphicsItem.viewTransformChanged(self)
        self._updateBoundingRect()

    def itemChange(self, change: QtWidgets.QGraphicsItem.GraphicsItemChange,
                   value: Any) -> Any:
        """
        Keep the cached bounds up to date when the line moves within its view.

        The labels of the line (:class:`InfLineLabel`) are also updated when the
        transform or the parent of the line changes.

        Parameters
        ----------
        change : QtWidgets.QGraphicsItem.GraphicsItemChange
            Kind of change.
        value : Any
            Value associated with the change.

        Returns
        -------
        Any
            The value returned by the base class.
        """
        ret = super().itemChange(change, value)
        if change in self._geometryChanges:
            try:
                tracking = self._trackGeometry
            except AttributeError:
                # the item is being garbage collected
                return ret
            if tracking:
                GraphicsItem.viewTransformChanged(self)  # the local view rect moved
                self._updateBoundingRect()
                if change != QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
                    # the labels keep their scale and orientation relative to the
                    # line without checking the transform before each paint
                    for child in self.childItems():
                        if isinstance(child, InfLineLabel):
                            child.updateTransform()
        elif change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemParentHasChanged:
            # whether the labels must check their transform before each paint
            # depends on the parent of the line (see InfLineLabel._needsPaintSync)
            for child in self.childItems():
                if isinstance(child, InfLineLabel):
                    child._updatePaintSync()
        return ret

    def setName(self, name):
        self._name = name

    def name(self):
        return self._name


class InfLineLabel(TextItem):
    """
    A TextItem that attaches itself to an InfiniteLine.
    
    This class extends TextItem with the following features:
    
      * Automatically positions adjacent to the line at a fixed position along
        the line and within the view box.
      * Automatically reformats text when the line value has changed.
      * Can optionally be dragged to change its location along the line.
      * Optionally aligns to its parent line.

    =============== ==================================================================
    **Arguments:**
    line            The InfiniteLine to which this label will be attached.
    text            String to display in the label. May contain a {value} formatting
                    string to display the current value of the line.
    movable         Bool; if True, then the label can be dragged along the line.
    position        Relative position (0.0-1.0) within the view to position the label
                    along the line.
    anchors         List of (x,y) pairs giving the text anchor positions that should
                    be used when the line is moved to one side of the view or the
                    other. This allows text to switch to the opposite side of the line
                    as it approaches the edge of the view. These are automatically
                    selected for some common cases, but may be specified if the 
                    default values give unexpected results.
    =============== ==================================================================
    
    All extra keyword arguments are passed to TextItem. A particularly useful
    option here is to use `rotateAxis=(1, 0)`, which will cause the text to
    be automatically rotated parallel to the line.
    """
    def __init__(self, line, text="", movable=False, position=0.5, anchors=None, **kwargs):
        self.line = line
        self.movable = movable
        self.moving = False
        self.orthoPos = position  # text will always be placed on the line at a position relative to view bounds
        self.format = text
        self.line.sigPositionChanged.connect(self.valueChanged)
        self._endpoints = (None, None)
        if anchors is None:
            # automatically pick sensible anchors
            rax = kwargs.get('rotateAxis', None)
            if rax is not None:
                if tuple(rax) == (1,0):
                    anchors = [(0.5, 0), (0.5, 1)]
                else:
                    anchors = [(0, 0.5), (1, 0.5)]
            else:
                if line.angle % 180 == 0:
                    anchors = [(0.5, 0), (0.5, 1)]
                else:
                    anchors = [(0, 0.5), (1, 0.5)]
            
        self.anchors = anchors
        TextItem.__init__(self, **kwargs)
        self.setParentItem(line)
        self.valueChanged()

    def valueChanged(self):
        if not self.isVisible():
            return
        value = self.line.value()
        self.setText(self.format.format(value=value))
        self.updatePosition()

    def getEndpoints(self):
        # calculate points where line intersects view box
        # (in line coordinates)
        if self._endpoints[0] is None:
            lr = self.line.boundingRect()
            pt1 = Point(lr.left(), 0)
            pt2 = Point(lr.right(), 0)
            
            if self.line.angle % 90 != 0:
                # more expensive to find text position for oblique lines.
                view = self.getViewBox()
                if not self.isVisible() or not isinstance(view, ViewBox):
                    # not in a viewbox, skip update
                    return (None, None)
                p = QtGui.QPainterPath()
                p.moveTo(pt1)
                p.lineTo(pt2)
                p = self.line.itemTransform(view)[0].map(p)
                vr = QtGui.QPainterPath()
                vr.addRect(view.boundingRect())
                paths = vr.intersected(p).toSubpathPolygons(QtGui.QTransform())
                if len(paths) > 0:
                    l = list(paths[0])
                    pt1 = self.line.mapFromItem(view, l[0])
                    pt2 = self.line.mapFromItem(view, l[1])
            self._endpoints = (pt1, pt2)
        return self._endpoints
    
    def updatePosition(self):
        # update text position to relative view location along line
        self._endpoints = (None, None)
        pt1, pt2 = self.getEndpoints()
        if pt1 is None:
            return
        pt = pt2 * self.orthoPos + pt1 * (1-self.orthoPos)
        self.setPos(pt)
        
        # update anchor to keep text visible as it nears the view box edge
        vr = self.line.viewRect()
        if vr is not None:
            self.setAnchor(self.anchors[0 if vr.center().y() < 0 else 1])
        
    def setVisible(self, v):
        TextItem.setVisible(self, v)
        if v:
            self.valueChanged()
            
    def setMovable(self, m):
        """Set whether this label is movable by dragging along the line.
        """
        self.movable = m
        self.setAcceptHoverEvents(m)
        
    def setPosition(self, p):
        """Set the relative position (0.0-1.0) of this label within the view box
        and along the line. 
        
        For horizontal (angle=0) and vertical (angle=90) lines, a value of 0.0
        places the text at the bottom or left of the view, respectively. 
        """
        self.orthoPos = p
        self.updatePosition()
        
    def setFormat(self, text):
        """Set the text format string for this label.
        
        May optionally contain "{value}" to include the lines current value
        (the text will be reformatted whenever the line is moved).
        """
        self.format = text
        self.valueChanged()
        
    def mouseDragEvent(self, ev):
        if self.movable and ev.button() == QtCore.Qt.MouseButton.LeftButton:
            if ev.isStart():
                self._moving = True
                self._cursorOffset = self._posToRel(ev.buttonDownPos())
                self._startPosition = self.orthoPos
            ev.accept()

            if not self._moving:
                return

            rel = self._posToRel(ev.pos())
            self.orthoPos = fn.clip_scalar(self._startPosition + rel - self._cursorOffset, 0., 1.)
            self.updatePosition()
            if ev.isFinish():
                self._moving = False

    def mouseClickEvent(self, ev):
        if self.moving and ev.button() == QtCore.Qt.MouseButton.RightButton:
            ev.accept()
            self.orthoPos = self._startPosition
            self.moving = False

    def hoverEvent(self, ev):
        if not ev.isExit() and self.movable:
            ev.acceptDrags(QtCore.Qt.MouseButton.LeftButton)

    def viewTransformChanged(self):
        GraphicsItem.viewTransformChanged(self)
        self.updatePosition()
        TextItem.viewTransformChanged(self)

    def _needsPaintSync(self) -> bool:
        """
        Return whether the transform must be checked before every paint.

        The line updates the transform of its labels when its own transform changes
        (see :meth:`InfiniteLine.itemChange`), so a label only needs to check its
        transform before every paint when the line is not a direct child of a
        :class:`ViewBox`.

        Returns
        -------
        bool
            True if the label has to connect to ``sigPrepareForPaint``.
        """
        line = self.parentItem()
        if line is not self.line:
            return TextItem._needsPaintSync(self)
        return not self._isViewChild(line.parentItem())

    def _posToRel(self, pos):
        # convert local position to relative position along line between view bounds
        pt1, pt2 = self.getEndpoints()
        if pt1 is None:
            return 0
        pos = self.mapToParent(pos)
        return (pos.x() - pt1.x()) / (pt2.x()-pt1.x())
