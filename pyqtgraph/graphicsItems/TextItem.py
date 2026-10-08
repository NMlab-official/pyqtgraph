from collections.abc import Sequence
from math import atan2, degrees
from typing import Any

from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from .GraphicsObject import GraphicsObject

__all__ = ['TextItem']

class TextItem(GraphicsObject):
    """
    GraphicsItem displaying unscaled text (the text will always appear normal even inside a scaled ViewBox). 
    """
    def __init__(self, text: str = '', color: Any = (200, 200, 200), html: str | None = None,
                 anchor: Sequence[float] | QtCore.QPointF = (0, 0), border: Any = None,
                 fill: Any = None, angle: float = 0,
                 rotateAxis: Sequence[float] | QtCore.QPointF | None = None,
                 ensureInBounds: bool = False) -> None:
        """
        The effects of the `rotateAxis` and `angle` arguments are added independently. So for example:

          * rotateAxis=None, angle=0 -> normal horizontal text
          * rotateAxis=None, angle=90 -> normal vertical text
          * rotateAxis=(1, 0), angle=0 -> text aligned with x axis of its parent
          * rotateAxis=(0, 1), angle=0 -> text aligned with y axis of its parent
          * rotateAxis=(1, 0), angle=90 -> text orthogonal to x axis of its parent

        Parameters
        ----------
        text : str, default ''
            The text to display
        color : Any, default (200, 200, 200)
            The color of the text (any format accepted by pg.mkColor)
        html : str, optional
            If specified, this overrides both *text* and *color*
        anchor : sequence of float or QtCore.QPointF, default (0, 0)
            A QPointF or (x,y) sequence indicating what region of the text box will
            be anchored to the item's position. A value of (0,0) sets the upper-left corner
            of the text box to be at the position specified by setPos(), while a value of (1,1)
            sets the lower-right corner.
        border : Any, optional
            A pen to use when drawing the border
        fill : Any, optional
            A brush to use when filling within the border
        angle : float, default 0
            Angle in degrees to rotate text. Default is 0; text will be displayed upright.
        rotateAxis : sequence of float or QtCore.QPointF, optional
            If None, then a text angle of 0 always points along the +x axis of the scene.
            If a QPointF or (x,y) sequence is given, then it represents a vector direction
            in the parent's coordinate system that the 0-degree line will be aligned to. This
            Allows text to follow both the position and orientation of its parent while still
            discarding any scale and shear factors.
        ensureInBounds : bool, default False
            Ensures that the entire TextItem will be visible when using autorange, but may
            produce runaway scaling in certain circumstances (See issue #2642). Setting to
            "True" retains legacy behavior.
        """
        self.anchor = Point(anchor)
        self.rotateAxis = None if rotateAxis is None else Point(rotateAxis)
        self._lastTransform = None
        self._lastTransformKey = None  # part of _lastTransform the text transform depends on
        self._lastScene = None  # scene of the last paint
        self._syncedScene = None  # scene whose sigPrepareForPaint triggers updateTransform
        self._ensureInBounds = ensureInBounds
        self._quietTransformChange = False
        #self.angle = 0
        GraphicsObject.__init__(self)
        self.textItem = QtWidgets.QGraphicsTextItem()
        self.textItem.setParentItem(self)

        # Note: The following is pretty scuffed; ideally there would likely be 
        # some inheritance changes, But this is the least-intrusive thing that 
        # works for now
        if ensureInBounds:
            self.dataBounds = None
        
        self._bounds = QtCore.QRectF()
        if html is None:
            self.setColor(color)
            self.setText(text)
        else:
            self.setHtml(html)
        self.fill = fn.mkBrush(fill)
        self.border = fn.mkPen(border)
        self.setAngle(angle)

    def setText(self, text, color=None):
        """
        Set the text of this item. 
        
        This method sets the plain text of the item; see also setHtml().
        """
        if color is not None:
            self.setColor(color)
        self.setPlainText(text)

    def setPlainText(self, text):
        """
        Set the plain text to be rendered by this item. 
        
        See QtWidgets.QGraphicsTextItem.setPlainText().
        """
        if text != self.toPlainText():
            self.textItem.setPlainText(text)
            self.updateTextPos()

    def toPlainText(self):
        return self.textItem.toPlainText()
        
    def setHtml(self, html):
        """
        Set the HTML code to be rendered by this item. 
        
        See QtWidgets.QGraphicsTextItem.setHtml().
        """
        if self.toHtml() != html:
            self.textItem.setHtml(html)
            self.updateTextPos()
        
    def toHtml(self):
        return self.textItem.toHtml()
        
    def setTextWidth(self, *args):
        """
        Set the width of the text.
        
        If the text requires more space than the width limit, then it will be
        wrapped into multiple lines.
        
        See QtWidgets.QGraphicsTextItem.setTextWidth().
        """
        self.textItem.setTextWidth(*args)
        self.updateTextPos()
        
    def setFont(self, *args):
        """
        Set the font for this text. 
        
        See QtWidgets.QGraphicsTextItem.setFont().
        """
        self.textItem.setFont(*args)
        self.updateTextPos()
        
    def setAngle(self, angle):
        """
        Set the angle of the text in degrees.

        This sets the rotation angle of the text as a whole, measured
        counter-clockwise from the x axis of the parent. Note that this rotation
        angle does not depend on horizontal/vertical scaling of the parent.
        """
        self.angle = angle
        self.updateTransform(force=True)

    def setAnchor(self, anchor):
        self.anchor = Point(anchor)
        self.updateTextPos()
        # the anchor enters dataBounds, which the view may have cached
        self.informViewBoundsChanged()

    def setColor(self, color):
        """
        Set the color for this text.
        
        See QtWidgets.QGraphicsItem.setDefaultTextColor().
        """
        self.color = fn.mkColor(color)
        self.textItem.setDefaultTextColor(self.color)
        
    def updateTextPos(self):
        # update text position to obey anchor
        r = self.textItem.boundingRect()
        tl = self.textItem.mapToParent(r.topLeft())
        br = self.textItem.mapToParent(r.bottomRight())
        offset = (br - tl) * self.anchor
        self.textItem.setPos(-offset)

    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        """
        Returns only the anchor point for when calculating view ranges.
        
        Sacrifices some visual polish for fixing issue #2642.
        """
        if orthoRange:
            range_min, range_max = orthoRange[0], orthoRange[1]
            if not range_min <= self.anchor[ax] <= range_max:
                return [None, None]

        return [self.anchor[ax], self.anchor[ax]]
        
    def boundingRect(self):
        return self.textItem.mapRectToParent(self.textItem.boundingRect())

    def viewTransformChanged(self) -> None:
        """
        Called whenever the transform of the view has changed.

        The scale of the text is updated here. Inside a :class:`ViewBox`, this is the
        only update needed when the view is panned or zoomed.
        """
        self.updateTransform()

    def viewChanged(self, view: Any, oldView: Any) -> None:
        """
        Called when the item has been added to or removed from a view.

        Parameters
        ----------
        view : ViewBox or GraphicsView or None
            New view of the item.
        oldView : ViewBox or GraphicsView or None
            Previous view of the item.
        """
        self._updatePaintSync()

    def itemChange(self, change: QtWidgets.QGraphicsItem.GraphicsItemChange,
                   value: Any) -> Any:
        """
        React to item changes that affect the transform of the text.

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
        GraphicsItemChange = QtWidgets.QGraphicsItem.GraphicsItemChange
        # getattr: itemChange may be called while the item is garbage collected
        if getattr(self, '_quietTransformChange', False) and change in (
                GraphicsItemChange.ItemTransformChange,
                GraphicsItemChange.ItemTransformHasChanged):
            # Skip GraphicsObject.itemChange, which would make the ViewBox recompute
            # its auto-range: see updateTransform.
            return QtWidgets.QGraphicsObject.itemChange(self, change, value)
        ret = super().itemChange(change, value)
        if change in (GraphicsItemChange.ItemParentHasChanged,
                      GraphicsItemChange.ItemSceneHasChanged):
            self._updatePaintSync()
        elif change == GraphicsItemChange.ItemVisibleHasChanged:
            # also sent when a hidden parent is shown again
            self.updateTransform()
        return ret

    def _needsPaintSync(self) -> bool:
        """
        Return whether the transform must be checked before every paint.

        The scale of the text depends on the scene transform of the parent item. Qt
        does not signal changes of that transform, except when the parent is a
        :class:`ViewBox` (or the item group holding its children), whose transform
        changes call :meth:`viewTransformChanged`. Other items, whose parent may be
        rotated or scaled at any time, check their transform before every paint, from
        the ``sigPrepareForPaint`` signal of the scene.

        Returns
        -------
        bool
            True if the item has to connect to ``sigPrepareForPaint``.
        """
        return not self._isViewChild(self.parentItem())

    def _isViewChild(self, item: QtWidgets.QGraphicsItem | None) -> bool:
        """
        Return whether ``item`` is the ViewBox containing this item or its child group.

        Parameters
        ----------
        item : QtWidgets.QGraphicsItem or None
            Item to test, typically the parent of this item.

        Returns
        -------
        bool
            True if ``item`` is the :class:`ViewBox` of this item or the group in
            which the ViewBox places its children.
        """
        view = self.getViewBox()
        if item is None or view is None or not hasattr(view, 'implements') \
                or not view.implements('ViewBox'):
            return False
        return item is view or item is view.innerSceneItem()

    def _updatePaintSync(self) -> None:
        """Connect to, or disconnect from, ``sigPrepareForPaint`` of the scene as needed."""
        scene = self.scene()
        if scene is not None and (not hasattr(scene, 'sigPrepareForPaint')
                                  or not self._needsPaintSync()):
            scene = None
        synced = self._syncedScene
        if scene is synced:
            return
        if synced is not None:
            try:
                synced.sigPrepareForPaint.disconnect(self.updateTransform)
            except (TypeError, RuntimeError):
                # TypeError and RuntimeError are from PyQt and PySide, respectively
                pass
        if scene is not None:
            scene.sigPrepareForPaint.connect(self.updateTransform)
        self._syncedScene = scene

    def paint(self, p: QtGui.QPainter, *args: Any) -> None:
        """
        Draw the border and the background of the text.

        Parameters
        ----------
        p : QtGui.QPainter
            Painter, in local coordinates.
        *args : Any
            Style option and widget, unused.
        """
        s = self.scene()
        if s is not self._lastScene:
            # first paint in this scene: make sure the transform is up to date
            self._lastScene = s
            self._updatePaintSync()
            self.updateTransform()
            p.setTransform(self.sceneTransform())

        if self.border.style() != QtCore.Qt.PenStyle.NoPen or self.fill.style() != QtCore.Qt.BrushStyle.NoBrush:
            p.setPen(self.border)
            p.setBrush(self.fill)
            p.setRenderHint(p.RenderHint.Antialiasing, True)
            p.drawPolygon(self.textItem.mapToParent(self.textItem.boundingRect()))
        
    def setVisible(self, v):
        GraphicsObject.setVisible(self, v)
        if v:
            self.updateTransform()
    
    @QtCore.Slot()
    def updateTransform(self, force: bool = False) -> None:
        """
        Update the transform of the item so that the text is not scaled.

        The item gets the correct orientation and scaling relative to the scene, but
        inherits its position from its parent. This is similar to setting
        ``ItemIgnoresTransformations``, but does not break mouse interaction and
        collision detection.

        Parameters
        ----------
        force : bool, default False
            Recompute the transform even if the scale and orientation of the parent
            did not change. A translation of the parent (e.g. a pan of the view) never
            requires an update.
        """
        if not self.isVisible():
            return

        p = self.parentItem()
        if p is None:
            pt = QtGui.QTransform()
        else:
            pt = p.sceneTransform()

        # The translation of the parent is discarded below: it does not matter
        # unless the transform is projective.
        key = (pt.m11(), pt.m12(), pt.m13(), pt.m21(), pt.m22(), pt.m23(), pt.m33())
        if key[2] != 0.0 or key[5] != 0.0:
            key += (pt.m31(), pt.m32())
        if not force and key == self._lastTransformKey:
            self._lastTransform = pt
            return

        t = fn.invertQTransform(pt)
        # reset translation
        t.setMatrix(t.m11(), t.m12(), t.m13(), t.m21(), t.m22(), t.m23(), 0, 0, t.m33())

        # apply rotation
        angle = -self.angle
        if self.rotateAxis is not None:
            d = pt.map(self.rotateAxis) - pt.map(QtCore.QPointF(0.0, 0.0))
            a = degrees(atan2(d.y(), d.x()))
            angle += a
        t.rotate(angle)
        # The offset of the text (updateTextPos) does not depend on this transform.
        # Unless the whole text must stay in the auto-range, the ViewBox is not told
        # about this change: the bounds then only contain the anchor point, which
        # the transform barely moves, and re-running the auto-range on every zoom
        # can feed back into the range (issue #2642).
        self._quietTransformChange = not self._ensureInBounds
        try:
            self.setTransform(t)
        finally:
            self._quietTransformChange = False
        self._lastTransform = pt
        self._lastTransformKey = key
