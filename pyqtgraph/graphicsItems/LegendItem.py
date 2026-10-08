from __future__ import annotations

import math
import weakref

from .. import functions as fn
from ..icons import getGraphPixmap
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from .BarGraphItem import BarGraphItem
from .GraphicsWidget import GraphicsWidget
from .GraphicsWidgetAnchor import GraphicsWidgetAnchor
from .LabelItem import LabelItem
from .PlotDataItem import PlotDataItem
from .ScatterPlotItem import ScatterPlotItem, drawSymbol

__all__ = ['LegendItem', 'ItemSample']


class LegendItem(GraphicsWidgetAnchor, GraphicsWidget):
    """
    Displays a legend used for describing the contents of a plot.

    LegendItems are most commonly created by calling :meth:`PlotItem.addLegend
    <pyqtgraph.PlotItem.addLegend>`.

    Note that this item should *not* be added directly to a PlotItem (via
    :meth:`PlotItem.addItem <pyqtgraph.PlotItem.addItem>`). Instead, make it a
    direct descendant of the PlotItem::

        legend.setParentItem(plotItem)

    """

    sigDoubleClicked = QtCore.Signal(object, object)
    sigSampleClicked = QtCore.Signal(object)

    def __init__(self, size: tuple[float, float] | None = None,
                 offset: tuple[float, float] | None = None, horSpacing: float = 5,
                 verSpacing: float = 0, pen: object = None, brush: object = None,
                 labelTextColor: object = None, frame: bool = True,
                 labelTextSize: str = '9pt', colCount: int = 1,
                 sampleType: type | None = None, **kwargs) -> None:
        """
        Parameters
        ----------
        size : tuple of float, float, optional
            Specifies the fixed size (width, height) of the legend. If this argument
            is omitted, the legend will automatically resize to fit its contents.
        offset : tuple of float, float, optional
            Specifies the offset position relative to the legend's parent. Positive
            values offset from the left or top; negative values offset from the right
            or bottom. If offset is None, the legend must be anchored manually by
            calling anchor() or positioned by calling setPos().
        horSpacing : float, default 5
            Specifies the spacing between the line symbol and the label.
        verSpacing : float, default 0
            Specifies the spacing between individual entries of the legend
            vertically. (Can also be negative to have them really close)
        pen : object, optional
            Pen to use when drawing legend border. Any single argument accepted by
            :func:`mkPen <pyqtgraph.mkPen>` is allowed.
        brush : object, optional
            QBrush to use as legend background filling. Any single argument accepted
            by :func:`mkBrush <pyqtgraph.mkBrush>` is allowed.
        labelTextColor : object, optional
            Pen to use when drawing legend text. Any single argument accepted by
            :func:`mkPen <pyqtgraph.mkPen>` is allowed.
        frame : bool, default True
            Draw the border and the background of the legend.
        labelTextSize : str, default '9pt'
            Size to use when drawing legend text. Accepts CSS style string arguments,
            e.g. '9pt'.
        colCount : int, default 1
            Specifies the integer number of columns that the legend should be divided
            into. The number of rows will be calculated based on this argument. This
            is useful for plots with many curves displayed simultaneously.
        sampleType : type, optional
            Customizes the item sample class of the `LegendItem`.
        **kwargs
            Additional options stored in the ``opts`` dictionary.
        """
        GraphicsWidget.__init__(self)
        GraphicsWidgetAnchor.__init__(self)
        self.layout_ = QtWidgets.QGraphicsGridLayout()
        self.layout_.setVerticalSpacing(verSpacing)
        self.layout_.setHorizontalSpacing(horSpacing)

        self.setLayout(self.layout_)
        self.items = []
        self.size = size
        self.offset = offset
        self.frame = frame
        self.columnCount = colCount
        self.rowCount = 1
        # Adding or removing entries only marks the size as outdated; the size is
        # recomputed once per batch, before the next render or export at the latest
        # (see _requestSizeUpdate).
        self._sizeUpdatePending = False
        self._sizeUpdateScene = None  # weak reference, see _requestSizeUpdate
        self._sizeUpdateTimer = QtCore.QTimer(self)
        self._sizeUpdateTimer.setSingleShot(True)
        self._sizeUpdateTimer.setInterval(0)
        self._sizeUpdateTimer.timeout.connect(self._flushSizeUpdate)
        if size is not None:
            self.setGeometry(QtCore.QRectF(0, 0, self.size[0], self.size[1]))

        if sampleType is not None:
            if not issubclass(sampleType, GraphicsWidget):
                raise RuntimeError("Only classes of type `GraphicsWidgets` "
                                   "are allowed as `sampleType`")
            self.sampleType = sampleType
        else:
            self.sampleType = ItemSample

        self.opts = {
            'pen': fn.mkPen(pen),
            'brush': fn.mkBrush(brush),
            'labelTextColor': labelTextColor,
            'labelTextSize': labelTextSize,
            'offset': offset,
        }
        self.opts.update(kwargs)

    def setSampleType(self, sample):
        """Set the new sample item claspes"""
        if sample is self.sampleType:
            return

        # Clear the legend, but before create a list of items
        items = list(self.items)
        self.sampleType = sample
        self.clear()

        # Refill the legend with the item list and new sample item
        for sample, label in items:
            plot_item = sample.item
            plot_name = label.text
            self.addItem(plot_item, plot_name)

        self.updateSize()

    def offset(self):
        """Get the offset position relative to the parent."""
        return self.opts['offset']

    def setOffset(self, offset):
        """Set the offset position relative to the parent."""
        self.opts['offset'] = offset

        offset = Point(self.opts['offset'])
        anchorx = 1 if offset[0] <= 0 else 0
        anchory = 1 if offset[1] <= 0 else 0
        anchor = (anchorx, anchory)
        self.anchor(itemPos=anchor, parentPos=anchor, offset=offset)

    def pen(self):
        """Get the QPen used to draw the border around the legend."""
        return self.opts['pen']

    def setPen(self, *args, **kwargs):
        """Set the pen used to draw a border around the legend.

        Accepts the same arguments as :func:`~pyqtgraph.mkPen`.
        """
        pen = fn.mkPen(*args, **kwargs)
        self.opts['pen'] = pen

        self.update()

    def brush(self):
        """Get the QBrush used to draw the legend background."""
        return self.opts['brush']

    def setBrush(self, *args, **kwargs):
        """Set the brush used to draw the legend background.

        Accepts the same arguments as :func:`~pyqtgraph.mkBrush`.
        """
        brush = fn.mkBrush(*args, **kwargs)
        if self.opts['brush'] == brush:
            return
        self.opts['brush'] = brush

        self.update()

    def labelTextColor(self):
        """Get the QColor used for the item labels."""
        return self.opts['labelTextColor']

    def setLabelTextColor(self, *args, **kwargs):
        """Set the color of the item labels.

        Accepts the same arguments as :func:`~pyqtgraph.mkColor`.
        """
        self.opts['labelTextColor'] = fn.mkColor(*args, **kwargs)
        for sample, label in self.items:
            label.setAttr('color', self.opts['labelTextColor'])

        self.update()

    def labelTextSize(self):
        """Get the `labelTextSize` used for the item labels."""
        return self.opts['labelTextSize']

    def setLabelTextSize(self, size):
        """Set the `size` of the item labels.

        Accepts the CSS style string arguments, e.g. '8pt'.
        """
        self.opts['labelTextSize'] = size
        for _, label in self.items:
            label.setAttr('size', self.opts['labelTextSize'])

        self.update()

    def setParentItem(self, p):
        """Set the parent."""
        ret = GraphicsWidget.setParentItem(self, p)
        if self.opts['offset'] is not None:
            offset = Point(self.opts['offset'])
            anchorx = 1 if offset[0] <= 0 else 0
            anchory = 1 if offset[1] <= 0 else 0
            anchor = (anchorx, anchory)
            self.anchor(itemPos=anchor, parentPos=anchor, offset=offset)
        return ret

    def addItem(self, item: QtWidgets.QGraphicsItem, name: str) -> None:
        """
        Add a new entry to the legend.

        The size of the legend is not recomputed immediately: the update is
        coalesced with those of the other entries added in the same batch and runs
        on the next event loop iteration or before the scene is next rendered,
        whichever comes first. This keeps adding ``n`` entries linear in ``n``.
        Call :meth:`updateSize` to resize the legend immediately.

        Parameters
        ----------
        item : QtWidgets.QGraphicsItem
            A :class:`~pyqtgraph.PlotDataItem` from which the line and point style
            of the item will be determined or an instance of ItemSample (or a
            subclass), allowing the item display to be customized.
        name : str
            The title to display for this item. Simple HTML allowed.
        """
        label = LabelItem(name, color=self.opts['labelTextColor'],
                          justify='left', size=self.opts['labelTextSize'])
        if isinstance(item, self.sampleType):
            sample = item
        else:
            sample = self.sampleType(item)

        sample.sigClicked.connect(self.sigSampleClicked)

        self.items.append((sample, label))
        self._addItemToLayout(sample, label)
        self._requestSizeUpdate()

    def _addItemToLayout(self, sample, label):
        col = self.layout_.columnCount()
        row = self.layout_.rowCount()
        if row:
            row -= 1
        nCol = self.columnCount * 2
        # FIRST ROW FULL
        if col == nCol:
            for col in range(0, nCol, 2):
                # FIND RIGHT COLUMN
                if not self.layout_.itemAt(row, col):
                    break
            else:
                if col + 2 == nCol:
                    # MAKE NEW ROW
                    col = 0
                    row += 1
        self.layout_.addItem(sample, row, col, alignment=QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.layout_.addItem(label, row, col + 1)
        # Keep rowCount in sync with the number of rows if items are added
        self.rowCount = max(self.rowCount, row + 1)

    def setColumnCount(self, columnCount):
        """change the orientation of all items of the legend
        """
        if columnCount != self.columnCount:
            self.columnCount = columnCount

            self.rowCount = math.ceil(len(self.items) / columnCount)
            for i in range(self.layout_.count() - 1, -1, -1):
                self.layout_.removeAt(i)  # clear layout
            for sample, label in self.items:
                self._addItemToLayout(sample, label)
            self.updateSize()

    def getLabel(self, plotItem):
        """Return the labelItem inside the legend for a given plotItem

        The label-text can be changed via labelItem.setText
        """
        out = [(it, lab) for it, lab in self.items if it.item == plotItem]
        try:
            return out[0][1]
        except IndexError:
            return None

    def _removeItemFromLayout(self, *args):
        for item in args:
            self.layout_.removeItem(item)
            item.close()
            # Normally, the item is automatically removed from
            # its scene when it gets destroyed.
            # this doesn't happen on current versions of
            # PySide (5.15.x, 6.3.x) and results in a leak.
            scene = item.scene()
            if scene:
                scene.removeItem(item)

    def removeItem(self, item: QtWidgets.QGraphicsItem | str) -> None:
        """
        Remove one item from the legend.

        As for :meth:`addItem`, the size of the legend is updated once per batch,
        before the next render at the latest; call :meth:`updateSize` to resize the
        legend immediately.

        Parameters
        ----------
        item : QtWidgets.QGraphicsItem or str
            The item to remove or its name.
        """
        for sample, label in self.items:
            if sample.item is item or label.text == item:
                self.items.remove((sample, label))  # remove from itemlist
                self._removeItemFromLayout(sample, label)
                self._requestSizeUpdate()  # redraw box
                return  # return after first match

    def clear(self):
        """Remove all items from the legend."""
        for sample, label in self.items:
            self._removeItemFromLayout(sample, label)

        self.items = []
        self.updateSize()

    def updateSize(self) -> None:
        """
        Resize the legend to fit its entries, unless a fixed size was given.

        Any size update requested by :meth:`addItem` or :meth:`removeItem` and not
        yet performed is completed by this call.
        """
        self._sizeUpdatePending = False
        self._sizeUpdateTimer.stop()
        self._disconnectSizeUpdate()
        if self.size is not None:
            return
        height = 0
        width = 0
        for row in range(self.layout_.rowCount()):
            row_height = 0
            col_width = 0
            for col in range(self.layout_.columnCount()):
                item = self.layout_.itemAt(row, col)
                if item:
                    col_width += item.width() + 3
                    row_height = max(row_height, item.height())
            width = max(width, col_width)
            height += row_height
        self.setGeometry(0, 0, width, height)
        return

    def _requestSizeUpdate(self) -> None:
        """
        Schedule a single :meth:`updateSize` call for a batch of entry changes.

        ``updateSize`` walks every entry and relays out the whole grid, so calling it
        for each added entry made adding ``n`` entries quadratic. The update runs
        once, at the first of: the next event loop iteration (zero-delay timer), the
        scene's ``sigPrepareForPaint`` emitted before a render (image exports without
        an event loop included), or :meth:`setExportMode` (SVG exports).
        """
        if self._sizeUpdatePending:
            return
        self._sizeUpdatePending = True
        self._sizeUpdateTimer.start()
        # Connected only while an update is pending, so that renders do not call
        # back into the legend otherwise.
        scene = self.scene()
        if scene is not None and hasattr(scene, 'sigPrepareForPaint'):
            scene.sigPrepareForPaint.connect(self._flushSizeUpdate)
            self._sizeUpdateScene = weakref.ref(scene)

    def _disconnectSizeUpdate(self) -> None:
        """Disconnect the scene signal connected by :meth:`_requestSizeUpdate`."""
        sceneRef = self._sizeUpdateScene
        if sceneRef is None:
            return
        self._sizeUpdateScene = None
        scene = sceneRef()
        if scene is None:
            return
        try:
            scene.sigPrepareForPaint.disconnect(self._flushSizeUpdate)
        except (TypeError, RuntimeError):
            pass  # the scene is being deleted

    @QtCore.Slot()
    def _flushSizeUpdate(self) -> None:
        """Perform the size update requested by :meth:`_requestSizeUpdate`, if any."""
        if self._sizeUpdatePending:
            self.updateSize()

    def setExportMode(self, export: bool, opts: dict | None = None) -> None:
        """
        Complete any pending size update before the legend is exported.

        Exporters that paint items one by one, such as the SVG exporter, do not
        emit ``sigPrepareForPaint``.

        Parameters
        ----------
        export : bool
            True before exporting and False afterward.
        opts : dict, optional
            Export options, see :meth:`GraphicsItem.setExportMode
            <pyqtgraph.GraphicsItem.setExportMode>`.
        """
        if export:
            self._flushSizeUpdate()
        super().setExportMode(export, opts)

    def boundingRect(self):
        return QtCore.QRectF(0, 0, self.width(), self.height())

    def paint(self, p, *args):
        if self.frame:
            p.setPen(self.opts['pen'])
            p.setBrush(self.opts['brush'])
            p.drawRect(self.boundingRect())

    def hoverEvent(self, ev):
        ev.acceptDrags(QtCore.Qt.MouseButton.LeftButton)

    def mouseDragEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            ev.accept()
            dpos = ev.pos() - ev.lastPos()
            self.autoAnchor(self.pos() + dpos)

    def mouseDoubleClickEvent(self, ev):
        self.sigDoubleClicked.emit(self, ev)
        ev.accept()


class ItemSample(GraphicsWidget):
    """Class responsible for drawing a single item in a LegendItem (sans label)
    """

    sigClicked = QtCore.Signal(object)

    def __init__(self, item):
        GraphicsWidget.__init__(self)
        self.item = item
        self.setFixedWidth(20)
        self.setFixedHeight(20)

    def boundingRect(self):
        return QtCore.QRectF(0, 0, 20, 20)

    def paint(self, p, *args):
        opts = self.item.opts
        if opts.get('antialias'):
            p.setRenderHint(p.RenderHint.Antialiasing)

        visible = self.item.isVisible()
        if not visible:
            p.drawPixmap(QtCore.QPoint(1, 1), getGraphPixmap('invisibleEye', size=(18, 18)))
            return

        if not isinstance(self.item, ScatterPlotItem):
            p.setPen(fn.mkPen(opts['pen']))
            p.drawLine(0, 11, 20, 11)

            if (opts.get('fillLevel', None) is not None and
                    opts.get('fillBrush', None) is not None):
                p.setBrush(fn.mkBrush(opts['fillBrush']))
                p.setPen(fn.mkPen(opts['pen']))
                p.drawPolygon(QtGui.QPolygonF(
                    [QtCore.QPointF(2, 18), QtCore.QPointF(18, 2),
                     QtCore.QPointF(18, 18)]))

        symbol = opts.get('symbol', None)
        if symbol is not None:
            if isinstance(self.item, PlotDataItem):
                opts = self.item.scatter.opts
            p.translate(10, 10)
            drawSymbol(p, symbol, opts['size'], fn.mkPen(opts['pen']),
                       fn.mkBrush(opts['brush']))

        if isinstance(self.item, BarGraphItem):
            p.setBrush(fn.mkBrush(opts['brush']))
            p.drawRect(QtCore.QRectF(2, 2, 18, 18))

    def mouseClickEvent(self, event):
        """Use the mouseClick event to toggle the visibility of the plotItem
        """
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            visible = self.item.isVisible()
            self.item.setVisible(not visible)

        event.accept()
        self.update()
        self.sigClicked.emit(self.item)

