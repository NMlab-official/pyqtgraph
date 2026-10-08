from typing import Union

import numpy as np

from .. import functions as fn
from ..Qt import QtGui, QtWidgets, QtCore
from .PlotCurveItem import PlotCurveItem
from .PlotDataItem import PlotDataItem

__all__ = ['FillBetweenItem']

# Subpaths of a curve path, as half-open index ranges ``starts[i]:stops[i]`` into the
# coordinate arrays ``x`` and ``y``: ``(x, y, starts, stops)``.
_Subpaths = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


class FillBetweenItem(QtWidgets.QGraphicsPathItem):
    """
    GraphicsItem filling the space between two PlotDataItems.

    The fill path is rebuilt lazily: a change of either curve only marks the path as
    out of date, and the path is rebuilt once, the next time it is needed to paint the
    item or to compute its geometry. A hidden item therefore costs nothing to keep up to
    date.
    """
    def __init__(
        self,
        curve1: Union[PlotDataItem, PlotCurveItem],
        curve2: Union[PlotDataItem, PlotCurveItem],
        brush=None,
        pen=None,
        fillRule: QtCore.Qt.FillRule=QtCore.Qt.FillRule.OddEvenFill
    ):
        """FillBetweenItem fills a region between two curves with a specified
        :class:`~QtGui.QBrush`. 

        Parameters
        ----------
        curve1 : :class:`~pyqtgraph.PlotDataItem` | :class:`~pyqtgraph.PlotCurveItem`
            Line to draw fill from
        curve2 : :class:`~pyqtgraph.PlotDataItem` | :class:`~pyqtgraph.PlotCurveItem`
            Line to draw fill to
        brush : color_like, optional
            Arguments accepted by :func:`~pyqtgraph.mkBrush`, used
            to create the :class:`~QtGui.QBrush` instance used to draw the item
            by default None
        pen : color_like, optional
            Arguments accepted by :func:`~pyqtgraph.mkColor`, used
            to create the :class:`~QtGui.QPen` instance used to draw the item
            by default ``None``
        fillRule : QtCore.Qt.FillRule, optional
            FillRule to be applied to the underlying :class:`~QtGui.QPainterPath`
            instance, by default ``QtCore.Qt.FillRule.OddEvenFill``

        Raises
        ------
        ValueError
            Raised when ``None`` is passed in as either ``curve1``
            or ``curve2``
        TypeError
            Raised when either ``curve1`` or ``curve2`` is not either
            :class:`~pyqtgraph.PlotDataItem` or :class:`~pyqtgraph.PlotCurveItem`
        """
        super().__init__()
        # The fill path is held by a QGraphicsPathItem kept outside any scene, which
        # computes boundingRect() and shape() exactly as QGraphicsPathItem does. The
        # path of this item stays empty: its setPath() announces a geometry change,
        # which must not happen inside boundingRect(), where the path is rebuilt.
        self._geometryItem = QtWidgets.QGraphicsPathItem()
        # True while a curve change has been announced (prepareGeometryChange) but the
        # path not rebuilt yet.
        self._pathDirty = True
        self._shape: QtGui.QPainterPath | None = None
        self.curves = None
        self._fillRule = fillRule
        if curve1 is not None and curve2 is not None:
            self.setCurves(curve1, curve2)
        elif curve1 is not None or curve2 is not None:
            raise ValueError("Must specify two curves to fill between.")

        if brush is not None:
            self.setBrush(brush)
        self.setPen(pen)

    def fillRule(self):
        return self._fillRule

    def setFillRule(
        self,
        fillRule: QtCore.Qt.FillRule=QtCore.Qt.FillRule.OddEvenFill
    ) -> None:
        """Set the underlying :class:`~QtGui.QPainterPath` to the specified 
        :class:`~QtCore.Qt.FillRule`

        This can be useful for allowing in the filling of voids.

        Parameters
        ----------
        fillRule : QtCore.Qt.FillRule
            A member of the :class:`~QtCore.Qt.FillRule` enum
        """
        self._fillRule = fillRule
        self._invalidatePath()
        
    def setBrush(self, *args, **kwargs):
        """Change the fill brush. Accepts the same arguments as :func:`~pyqtgraph.mkBrush`
        """
        QtWidgets.QGraphicsPathItem.setBrush(self, fn.mkBrush(*args, **kwargs))
        
    def setPen(self, *args, **kwargs) -> None:
        """
        Change the fill pen.

        Parameters
        ----------
        *args
            Positional arguments accepted by :func:`~pyqtgraph.mkPen`.
        **kwargs
            Keyword arguments accepted by :func:`~pyqtgraph.mkPen`.
        """
        pen = fn.mkPen(*args, **kwargs)
        QtWidgets.QGraphicsPathItem.setPen(self, pen)
        # the pen enters the bounding rectangle and the shape
        self._geometryItem.setPen(pen)
        self._shape = None

    def setCurves(
        self,
        curve1: Union[PlotDataItem, PlotCurveItem],
        curve2: Union[PlotDataItem, PlotCurveItem]
    ):
        """Method to set the Curves to draw the FillBetweenItem between

        Parameters
        ----------
        curve1 : :class:`~pyqtgraph.PlotDataItem` | :class:`~pyqtgraph.PlotCurveItem`
            Line to draw fill from
        curve2 : :class:`~pyqtgraph.PlotDataItem` | :class:`~pyqtgraph.PlotCurveItem`
            Line to draw fill to
    
        Raises
        ------
        TypeError
            Raised when input arguments are not either :class:`~pyqtgraph.PlotDataItem` or
            :class:`~pyqtgraph.PlotCurveItem`
        """        
        if self.curves is not None:
            for c in self.curves:
                try:
                    c.sigPlotChanged.disconnect(self.curveChanged)
                except (TypeError, RuntimeError):
                    pass

        curves = [curve1, curve2]
        for c in curves:
            if not isinstance(c, (PlotDataItem, PlotCurveItem)):
                raise TypeError("Curves must be PlotDataItem or PlotCurveItem.")
        self.curves = curves
        curve1.sigPlotChanged.connect(self.curveChanged)
        curve2.sigPlotChanged.connect(self.curveChanged)
        self.setZValue(min(curve1.zValue(), curve2.zValue())-1)
        self.curveChanged()

    def curveChanged(self) -> None:
        """
        Mark the fill path as out of date after a change of either curve.

        The path is not rebuilt here but on the next call to :meth:`path`,
        :meth:`boundingRect`, :meth:`shape` or :meth:`paint`, so that changing both
        curves before the next frame costs a single rebuild, and none while the item is
        hidden.
        """
        self._invalidatePath()

    def _invalidatePath(self) -> None:
        """
        Mark the fill path as out of date and schedule a repaint.

        The geometry change is announced to the scene only when the path was up to date:
        ``prepareGeometryChange`` must run while :meth:`boundingRect` still reports the
        current path, and once per pending rebuild is enough.
        """
        if not self._pathDirty:
            self.prepareGeometryChange()
            self._pathDirty = True
        self.update()

    def updatePath(self) -> None:
        """
        Rebuild the fill path from the data currently displayed by both curves.

        This is called automatically when the path is needed after a curve changed. It
        can be called directly to refresh the fill after modifying a curve without it
        emitting ``sigPlotChanged``.
        """
        if self.curves is None:
            path = QtGui.QPainterPath()
        else:
            curve1, curve2 = (self._curveItem(c) for c in self.curves)
            subpaths1 = self._finiteSubpaths(curve1)
            subpaths2 = self._finiteSubpaths(curve2)
            if subpaths1 is None or subpaths2 is None:
                path = self._pathFromSubpathPolygons(curve1, curve2)
            else:
                path = self._pathFromSubpaths(subpaths1, subpaths2)
        self._storePath(path)

    def _storePath(self, path: QtGui.QPainterPath) -> None:
        """
        Make `path` the fill path.

        Parameters
        ----------
        path : QPainterPath
            The new fill path.
        """
        if not self._pathDirty:
            # Explicit change: no curve change announced the geometry change yet.
            # A lazy rebuild must not do this, as it runs inside boundingRect().
            self.prepareGeometryChange()
        # a new item, rather than setPath(), skips comparing the old and new paths
        geometryItem = QtWidgets.QGraphicsPathItem(path)
        geometryItem.setPen(self._geometryItem.pen())
        self._geometryItem = geometryItem
        self._pathDirty = False
        self._shape = None

    @staticmethod
    def _curveItem(curve: PlotDataItem | PlotCurveItem) -> PlotCurveItem:
        """
        Return the item that draws the line of `curve`.

        Parameters
        ----------
        curve : PlotDataItem or PlotCurveItem
            One of the two curves of the fill.

        Returns
        -------
        PlotCurveItem
            ``curve.curve`` for a :class:`~pyqtgraph.PlotDataItem`, holding its
            displayed data, else `curve` itself.
        """
        return curve.curve if isinstance(curve, PlotDataItem) else curve

    @staticmethod
    def _finiteSubpaths(curve: PlotCurveItem) -> _Subpaths | None:
        """
        Locate the subpaths of ``curve.getPath()`` in the data of `curve`.

        The path of a curve drawn without step mode and with ``connect='all'`` holds one
        subpath through its finite points; with ``connect='finite'``, one subpath per
        run of consecutive finite points. Both are found here with numpy, without
        building the path.

        Parameters
        ----------
        curve : PlotCurveItem
            Curve whose path is described.

        Returns
        -------
        tuple of numpy.ndarray or None
            ``(x, y, starts, stops)``: subpath ``i`` is made of the points
            ``x[starts[i]:stops[i]]``, ``y[starts[i]:stops[i]]`` and has at least two
            distinct points. ``None`` when the subpaths cannot be derived from the data
            reliably: step mode, other ``connect`` values, a customized path, or
            isolated finite points and repeated points, for which ``QPainterPath``
            drops subpaths.
        """
        opts = curve.opts
        connect = opts['connect']
        if (
            opts['stepMode']
            or not isinstance(connect, str)
            or connect not in ('all', 'finite')
            or type(curve).getPath is not PlotCurveItem.getPath
            or type(curve).generatePath is not PlotCurveItem.generatePath
        ):
            return None

        x, y = curve.getData()
        noRun = np.zeros(0, dtype=np.intp)
        if x is None or y is None or len(x) == 0 or len(y) == 0:
            return np.zeros(0), np.zeros(0), noRun, noRun

        finite = None
        if connect == 'finite' or not opts['skipFiniteCheck']:
            finite = np.isfinite(x) & np.isfinite(y)
            if finite.all():
                finite = None
            elif connect == 'all':
                # non-finite points are skipped, the line joins their neighbours
                x = x[finite]
                y = y[finite]
                finite = None

        if finite is None:
            # a single subpath, provided it has two points
            if len(x) < 2:
                return x, y, noRun, noRun
            starts = np.zeros(1, dtype=np.intp)
            stops = np.full(1, len(x), dtype=np.intp)
        else:
            # one subpath per run of finite points
            edges = np.flatnonzero(np.diff(finite, prepend=False, append=False))
            starts = edges[0::2]
            stops = edges[1::2]
            if ((stops - starts) < 2).any():
                # A lone finite point followed by a non-finite one forms a degenerate
                # subpath, kept by the forward path and dropped by the reversed one.
                return None

        # QPainterPath drops consecutive points closer than a relative 1e-12
        # (qFuzzyCompare) when reversing a path, and a subpath reduced to a single
        # point disappears. Require the first two points of each subpath to be
        # clearly apart, which keeps every subpath.
        x0 = x[starts].astype(np.float64)
        x1 = x[starts + 1].astype(np.float64)
        y0 = y[starts].astype(np.float64)
        y1 = y[starts + 1].astype(np.float64)
        tolX = 1e-9 * np.maximum(1.0, np.maximum(np.abs(x0), np.abs(x1)))
        tolY = 1e-9 * np.maximum(1.0, np.maximum(np.abs(y0), np.abs(y1)))
        if not ((np.abs(x1 - x0) > tolX) | (np.abs(y1 - y0) > tolY)).all():
            return None
        return x, y, starts, stops

    def _pathFromSubpaths(
        self,
        subpaths1: _Subpaths,
        subpaths2: _Subpaths
    ) -> QtGui.QPainterPath:
        """
        Build the fill path from the subpaths of both curves, with numpy.

        The ``i``-th subpath of the first curve, followed by the ``i``-th subpath of the
        second curve in reverse order, forms the ``i``-th polygon of the fill. The
        polygon is left open, as the fill closes it implicitly.

        Parameters
        ----------
        subpaths1 : tuple of numpy.ndarray
            Subpaths of the first curve, as returned by :meth:`_finiteSubpaths`.
        subpaths2 : tuple of numpy.ndarray
            Subpaths of the second curve, as returned by :meth:`_finiteSubpaths`.

        Returns
        -------
        QPainterPath
            The fill path, with the fill rule of the item.
        """
        x1, y1, starts1, stops1 = subpaths1
        x2, y2, starts2, stops2 = subpaths2
        count = min(len(starts1), len(starts2))
        if count == 0:
            return QtGui.QPainterPath()
        starts1, stops1 = starts1[:count], stops1[:count]
        starts2, stops2 = starts2[:count], stops2[:count]

        xParts = []
        yParts = []
        for s1, e1, s2, e2 in zip(starts1, stops1, starts2, stops2):
            xParts += [x1[s1:e1], x2[s2:e2][::-1]]
            yParts += [y1[s1:e1], y2[s2:e2][::-1]]
        x = np.concatenate(xParts)
        y = np.concatenate(yParts)

        if count == 1:
            connect = 'all'
        else:
            # disconnect the last point of each polygon from the next polygon
            connect = np.ones(len(x), dtype=bool)
            connect[np.cumsum((stops1 - starts1) + (stops2 - starts2)) - 1] = False
        path = fn.arrayToQPath(x, y, connect=connect, finiteCheck=False)
        path.setFillRule(self.fillRule())
        return path

    def _pathFromSubpathPolygons(
        self,
        curve1: PlotCurveItem,
        curve2: PlotCurveItem
    ) -> QtGui.QPainterPath:
        """
        Build the fill path by pairing the subpaths of the rendered curve paths.

        This general algorithm handles every curve, in particular curves in step mode or
        with a ``connect`` array, at the cost of building both curve paths.

        Parameters
        ----------
        curve1 : PlotCurveItem
            Line of the first curve.
        curve2 : PlotCurveItem
            Line of the second curve.

        Returns
        -------
        QPainterPath
            The fill path, with the fill rule of the item.
        """
        ps1 = curve1.getPath().toSubpathPolygons()
        ps2 = curve2.getPath().toReversed().toSubpathPolygons()
        ps2.reverse()

        if len(ps1) == 0 or len(ps2) == 0:
            return QtGui.QPainterPath()

        path = QtGui.QPainterPath()
        path.setFillRule(self.fillRule())
        for p1, p2 in zip(ps1, ps2):
            path.addPolygon(p1 + p2)
        return path

    def path(self) -> QtGui.QPainterPath:
        """
        Return the fill path, rebuilding it first if a curve changed.

        Returns
        -------
        QPainterPath
            The current fill path.
        """
        if self._pathDirty:
            self.updatePath()
        return self._geometryItem.path()

    def setPath(self, path: QtGui.QPainterPath) -> None:
        """
        Replace the fill path until the next change of either curve.

        Parameters
        ----------
        path : QPainterPath
            The path to fill.
        """
        self._storePath(path)
        self.update()

    def boundingRect(self) -> QtCore.QRectF:
        """
        Return the bounding rectangle of the item, as ``QGraphicsPathItem`` does.

        Returns
        -------
        QRectF
            Bounding rectangle of the fill path and of its outline.
        """
        if self._pathDirty:
            self.updatePath()
        return self._geometryItem.boundingRect()

    def shape(self) -> QtGui.QPainterPath:
        """
        Return the shape of the item, as ``QGraphicsPathItem`` does.

        Returns
        -------
        QPainterPath
            The fill path, united with its outline when a pen is set.
        """
        if self._pathDirty:
            self.updatePath()
        if self._shape is None:
            # QGraphicsPathItem strokes the path on every call: keep the result
            self._shape = self._geometryItem.shape()
        return QtGui.QPainterPath(self._shape)

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: QtWidgets.QWidget | None = None
    ) -> None:
        """
        Draw the fill path, rebuilding it first if a curve changed.

        Parameters
        ----------
        painter : QPainter
            Painter used to draw the item.
        option : QStyleOptionGraphicsItem
            Style options of the item.
        widget : QWidget or None, optional
            Widget being painted on, by default ``None``.
        """
        if self._pathDirty:
            self.updatePath()
        painter.setPen(self.pen())
        painter.setBrush(self.brush())
        painter.drawPath(self._geometryItem.path())
        if option.state & QtWidgets.QStyle.StateFlag.State_Selected:
            # The path held by QGraphicsPathItem is empty: the base class only draws
            # the selection outline, around boundingRect().
            super().paint(painter, option, widget)
