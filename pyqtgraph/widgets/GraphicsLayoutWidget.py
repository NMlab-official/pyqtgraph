__all__ = ['GraphicsLayoutWidget']

from ..graphicsItems.GraphicsLayout import GraphicsLayout
from ..Qt import QtWidgets, mkQApp
from .GraphicsView import GraphicsView


class GraphicsLayoutWidget(GraphicsView):
    """
    Convenience class consisting of a :class:`GraphicsView
    <pyqtgraph.GraphicsView>` with a single :class:`GraphicsLayout
    <pyqtgraph.GraphicsLayout>` as its central item.

    This widget is an easy starting point for generating multi-panel figures.

    This class wraps several methods from its internal GraphicsLayout:
    :func:`nextRow <pyqtgraph.GraphicsLayout.nextRow>`
    :func:`nextColumn <pyqtgraph.GraphicsLayout.nextColumn>`
    :func:`addPlot <pyqtgraph.GraphicsLayout.addPlot>`
    :func:`addViewBox <pyqtgraph.GraphicsLayout.addViewBox>`
    :func:`addItem <pyqtgraph.GraphicsLayout.addItem>`
    :func:`getItem <pyqtgraph.GraphicsLayout.getItem>`
    :func:`addLabel <pyqtgraph.GraphicsLayout.addLabel>`
    :func:`addLayout <pyqtgraph.GraphicsLayout.addLayout>`
    :func:`removeItem <pyqtgraph.GraphicsLayout.removeItem>`
    :func:`itemIndex <pyqtgraph.GraphicsLayout.itemIndex>`
    :func:`clear <pyqtgraph.GraphicsLayout.clear>`

    Parameters
    ----------
    parent : QWidget, optional
        The parent widget.
    show : bool, default False
        If True, then immediately show the widget after it is created. If the
        widget has no parent, then it will be shown inside a new window.
    size : tuple of int, optional
        ``(width, height)``. Optionally resize the widget. Note: if this widget is
        placed inside a layout, then this argument has no effect.
    title : str, optional
        If specified, then set the window title for this widget.
    useOpenGL : bool, optional
        Whether the view renders with OpenGL, see :class:`GraphicsView
        <pyqtgraph.GraphicsView>`. By default, the ``useOpenGL`` configuration
        option is used.
    **kwargs
        All extra arguments are passed to :meth:`GraphicsLayout.__init__
        <pyqtgraph.GraphicsLayout.__init__>`.

    Examples
    --------
    ::

        w = pg.GraphicsLayoutWidget()
        p1 = w.addPlot(row=0, col=0)
        p2 = w.addPlot(row=0, col=1)
        v = w.addViewBox(row=1, col=0, colspan=2)
    """
    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        show: bool = False,
        size: tuple[int, int] | None = None,
        title: str | None = None,
        useOpenGL: bool | None = None,
        **kwargs,
    ) -> None:
        mkQApp()
        # useOpenGL belongs to GraphicsView: GraphicsLayout rejects it.
        GraphicsView.__init__(self, parent, useOpenGL=useOpenGL)
        self.ci = GraphicsLayout(**kwargs)
        for n in ['nextRow', 'nextCol', 'nextColumn', 'addPlot', 'addViewBox', 'addItem', 'getItem', 'addLayout', 'addLabel', 'removeItem', 'itemIndex', 'clear']:
            setattr(self, n, getattr(self.ci, n))
        self.setCentralItem(self.ci)
        
        if size is not None:
            self.resize(*size)
            
        if title is not None:
            self.setWindowTitle(title)
            
        if show is True:
            self.show()
