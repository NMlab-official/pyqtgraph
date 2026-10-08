__all__ = ['GraphicsObject']

from ..Qt import QtWidgets
from .GraphicsItem import GraphicsItem


class GraphicsObject(GraphicsItem, QtWidgets.QGraphicsObject):
    """
    **Bases:** :class:`GraphicsItem <pyqtgraph.GraphicsItem>`, :class:`QtWidgets.QGraphicsObject`

    Extension of QGraphicsObject with some useful methods (provided by :class:`GraphicsItem <pyqtgraph.GraphicsItem>`)
    """
    def __init__(self, *args):
        self.__inform_view_on_changes = True
        QtWidgets.QGraphicsObject.__init__(self, *args)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        GraphicsItem.__init__(self)
        
    def itemChange(self, change, value):
        ret = super().itemChange(change, value)
        if change in [
            QtWidgets.QGraphicsItem.GraphicsItemChange.ItemParentHasChanged,
            QtWidgets.QGraphicsItem.GraphicsItemChange.ItemSceneHasChanged
        ]:
            self.changeParent()
        try:
            inform_view_on_change = self.__inform_view_on_changes
        except AttributeError:
            # It's possible that the attribute was already collected when the itemChange happened
            # (if it was triggered during the gc of the object).
            pass
        else:
            if inform_view_on_change and change in [
                QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged,
                QtWidgets.QGraphicsItem.GraphicsItemChange.ItemTransformHasChanged
            ]:
                self.informViewBoundsChanged()
            elif (
                inform_view_on_change
                and change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemVisibleHasChanged
            ):
                self._informViewVisibilityChanged()
        return ret

    def _informViewVisibilityChanged(self) -> None:
        """
        Let the containing ViewBox forget the cached bounds that depend on this item.

        The bounds of an item may depend on the visibility of its children, e.g.
        :meth:`PlotDataItem.dataBounds <pyqtgraph.PlotDataItem.dataBounds>`. Unlike
        :meth:`informViewBoundsChanged`, no auto-range is requested.
        """
        if not hasattr(self, '_connectedView'):
            # GraphicsItem.__init__ has not run yet, or Python is shutting down.
            return
        view = self.getViewBox()
        if view is not None and hasattr(view, 'implements') and view.implements('ViewBox'):
            forget = getattr(view, '_forgetItemBounds', None)
            if forget is not None:
                forget(self)
