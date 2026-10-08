import pytest

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

app = pg.mkQApp()

def test_basics_graphics_view():
    view = pg.GraphicsView()
    background_role = view.backgroundRole()
    assert background_role == QtGui.QPalette.ColorRole.Window

    assert view.backgroundBrush().color() == QtGui.QColor(0, 0, 0, 255)

    assert view.focusPolicy() == QtCore.Qt.FocusPolicy.StrongFocus
    assert view.transformationAnchor() == QtWidgets.QGraphicsView.ViewportAnchor.NoAnchor
    minimal_update = QtWidgets.QGraphicsView.ViewportUpdateMode.MinimalViewportUpdate
    assert view.viewportUpdateMode() == minimal_update
    assert view.frameShape() == QtWidgets.QFrame.Shape.NoFrame
    assert view.hasMouseTracking() is True

    # Default properties
    # --------------------------------------

    assert view.mouseEnabled is False
    assert view.aspectLocked is False
    assert view.autoPixelRange is True
    assert view.scaleCenter is False
    assert view.clickAccepted is False
    assert view.centralWidget is not None
    assert view._background == "default"

    # Set background color
    # --------------------------------------
    view.setBackground("w")
    assert view._background == "w"
    assert view.backgroundBrush().color() == QtCore.Qt.GlobalColor.white

    # Set anti aliasing
    # --------------------------------------
    aliasing = QtGui.QPainter.RenderHint.Antialiasing
    # Default is set to `False`
    assert view.renderHints() & aliasing != aliasing
    view.setAntialiasing(True)
    assert view.renderHints() & aliasing == aliasing
    view.setAntialiasing(False)
    assert view.renderHints() & aliasing != aliasing

    # Enable mouse
    # --------------------------------------
    view.enableMouse(True)
    assert view.mouseEnabled is True
    assert view.autoPixelRange is False
    view.enableMouse(False)
    assert view.mouseEnabled is False
    assert view.autoPixelRange is True

    # Add and remove item
    # --------------------------------------
    central_item = QtWidgets.QGraphicsWidget()
    view.setCentralItem(central_item)
    assert view.centralWidget is central_item
    # XXX: Removal of central item is not clear in code
    scene = view.sceneObj
    assert isinstance(scene, pg.GraphicsScene)
    assert central_item in scene.items()

    item = QtWidgets.QGraphicsWidget()
    assert item not in scene.items()
    view.addItem(item)
    assert item in scene.items()
    view.removeItem(item)
    assert item not in scene.items()

    # Close the graphics view
    # --------------------------------------

    view.close()
    assert view.centralWidget is None
    assert view.currentItem is None
    assert view.sceneObj is None
    assert view.closed is True


@pytest.mark.parametrize("widget_class", [pg.PlotWidget, pg.GraphicsLayoutWidget])
@pytest.mark.parametrize("use_opengl", [True, False, None])
def test_useOpenGL_is_forwarded(monkeypatch, widget_class, use_opengl):
    # record the argument instead of creating an OpenGL viewport,
    # which would require an OpenGL context
    calls = []
    monkeypatch.setattr(pg.GraphicsView, 'useOpenGL', lambda self, b=True: calls.append(b))

    kwargs = {} if use_opengl is None else {'useOpenGL': use_opengl}
    widget = widget_class(**kwargs)
    expected = pg.getConfigOption('useOpenGL') if use_opengl is None else use_opengl
    assert calls == [expected]
    widget.close()
