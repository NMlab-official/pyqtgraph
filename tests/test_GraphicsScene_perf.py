"""
Deterministic performance tests of GraphicsScene.itemsNearEvent: a single scene query,
and results identical to the former two-query implementation.
"""
import numpy as np
import pytest

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from tests.perf_helpers import count_calls, process_events, show_and_wait

app = pg.mkQApp()

IntersectsItemShape = QtCore.Qt.ItemSelectionMode.IntersectsItemShape
IntersectsItemBoundingRect = QtCore.Qt.ItemSelectionMode.IntersectsItemBoundingRect
ContainsItemShape = QtCore.Qt.ItemSelectionMode.ContainsItemShape
DescendingOrder = QtCore.Qt.SortOrder.DescendingOrder


class _Event:
    """Minimal stand-in for a HoverEvent: only the scene position is used."""

    def __init__(self, pos):
        self._pos = pos

    def scenePos(self):
        return self._pos


def reference_itemsNearEvent(scene, event, selMode=IntersectsItemShape,
                             sortOrder=DescendingOrder, hoverable=False):
    """The implementation of GraphicsScene.itemsNearEvent before it was optimized."""
    view = scene.views()[0]
    tr = view.viewportTransform()

    if hasattr(event, "buttonDownScenePos"):
        point = event.buttonDownScenePos()
    else:
        point = event.scenePos()

    def absZValue(item):
        if item is None:
            return 0
        return item.zValue() + absZValue(item.parentItem())

    items_at_point = scene.items(point, selMode, sortOrder, tr)
    items_at_point.sort(key=absZValue, reverse=True)

    r = scene._clickRadius
    items_within_radius = []
    rgn = None
    if r > 0:
        rect = view.mapToScene(QtCore.QRect(0, 0, 2 * r, 2 * r)).boundingRect()
        w = rect.width()
        h = rect.height()
        rgn = QtCore.QRectF(point.x() - w / 2, point.y() - h / 2, w, h)
        items_within_radius = scene.items(rgn, selMode, sortOrder, tr)
        items_within_radius.sort(key=absZValue, reverse=True)
        for item in items_at_point:
            if item in items_within_radius:
                items_within_radius.remove(item)

    all_items = items_at_point + items_within_radius

    selected_items = []
    for item in all_items:
        if hoverable and not hasattr(item, "hoverEvent"):
            continue
        if item.scene() is not scene:
            continue
        shape = item.shape()
        if shape is None:
            continue
        if (
            rgn is not None
            and shape.intersects(item.mapFromScene(rgn).boundingRect())
        ) or shape.contains(item.mapFromScene(point)):
            selected_items.append(item)

    return selected_items


@pytest.fixture
def mixed_scene():
    """A layout mixing curves, scatter points, lines, texts, nested and untransformable items."""
    win = pg.GraphicsLayoutWidget(size=(420, 360))
    p1 = win.addPlot(row=0, col=0)
    p2 = win.addPlot(row=1, col=0)
    p2.setXLink(p1)
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 60)
    for k in range(6):
        p1.plot(x, np.sin(x + k) * 3 + k, pen=pg.intColor(k, 6),
                symbol='o' if k % 2 else None, symbolSize=8)
    p1.addItem(pg.ScatterPlotItem(x=rng.random(30) * 10, y=rng.random(30) * 8, size=12,
                                  hoverable=True))
    line = pg.InfiniteLine(pos=4, angle=90, movable=True)
    line.setZValue(20)
    p1.addItem(line)
    p1.addItem(pg.InfiniteLine(pos=2, angle=0, movable=False), ignoreBounds=True)
    p1.addItem(pg.LinearRegionItem(values=(6, 7)))
    text = pg.TextItem('text item', anchor=(0.5, 0.5), border='w', fill=(0, 0, 0, 100))
    text.setPos(5, 2)
    p1.addItem(text)
    child = QtWidgets.QGraphicsRectItem(-2, -2, 30, 12, text)  # nested plain Qt item
    child.setZValue(-3)
    arrow = pg.ArrowItem(pos=(3, 1), angle=45, headLen=30)  # ignores transformations
    p1.addItem(arrow)
    roi = pg.RectROI([7, 0], [2, 3])
    p1.addItem(roi)
    p2.plot(x, np.cos(x), pen='y', fillLevel=0, brush=(50, 50, 200, 100))
    p2.addItem(pg.BarGraphItem(x=x[::6], height=np.cos(x[::6]), width=0.4))
    p2.addLegend().addItem(pg.PlotDataItem(pen='r'), 'legend entry')
    win.show()
    process_events(5)
    # points of interest: a grid over the whole view, plus item positions
    points = [QtCore.QPointF(px, py) for px in range(3, 420, 23) for py in range(3, 360, 19)]
    vb = p1.getViewBox()
    for pos in [(4, 0), (4, 2), (3, 1), (5, 2), (7, 0), (9, 3), (6, 1), (6.5, 0)]:
        points.append(vb.mapViewToScene(QtCore.QPointF(*pos)))
    edge = vb.sceneBoundingRect()
    for dx in (-3, -1, 0, 1, 3):
        points.append(QtCore.QPointF(edge.left() + dx, edge.center().y()))
        points.append(QtCore.QPointF(edge.center().x(), edge.bottom() + dx))
    yield win.scene(), points
    win.close()


@pytest.mark.parametrize('hoverable', [True, False])
@pytest.mark.parametrize('radius', [0, 2, 5])
def test_itemsNearEvent_matches_reference(mixed_scene, hoverable, radius):
    scene, points = mixed_scene
    scene.setClickRadius(radius)
    try:
        nonEmpty = 0
        for point in points:
            event = _Event(point)
            expected = reference_itemsNearEvent(scene, event, hoverable=hoverable)
            result = scene.itemsNearEvent(event, hoverable=hoverable)
            assert result == expected, point
            nonEmpty += len(expected) > 1
        assert nonEmpty > 20  # the comparison covers orderings, not only empty lists
    finally:
        scene.setClickRadius(2)


@pytest.mark.parametrize('selMode', [IntersectsItemBoundingRect, ContainsItemShape])
def test_itemsNearEvent_matches_reference_other_modes(mixed_scene, selMode):
    scene, points = mixed_scene
    for point in points:
        event = _Event(point)
        for hoverable in (True, False):
            expected = reference_itemsNearEvent(scene, event, selMode, hoverable=hoverable)
            assert scene.itemsNearEvent(event, selMode, hoverable=hoverable) == expected, point


def test_itemsNearEvent_many_overlapping_items():
    # More kept items than _maxItemsTestedAtPoint: a second query sorts them out.
    pw = pg.PlotWidget(size=(300, 200))
    x = np.linspace(0, 1, 20)
    for k in range(40):
        pw.plot(x, x + k * 1e-3, pen='w')
    pw.show()
    process_events(5)
    scene = pw.scene()
    vb = pw.getViewBox()
    try:
        for pos in [(0.5, 0.5), (0.2, 0.2), (0.8, 0.79), (0.5, 0.45)]:
            event = _Event(vb.mapViewToScene(QtCore.QPointF(*pos)))
            expected = reference_itemsNearEvent(scene, event)
            assert len(expected) > scene._maxItemsTestedAtPoint
            assert scene.itemsNearEvent(event) == expected
    finally:
        pw.close()


def test_itemsNearEvent_queries_scene_once(mixed_scene):
    scene, points = mixed_scene
    hovered = 0
    with count_calls(scene, 'items') as queries:
        for point in points:
            hovered += len(scene.itemsNearEvent(_Event(point), hoverable=True)) > 1
    assert hovered > 0
    assert queries.count == len(points)


# --------------------------------------------------------------------------------------
# T2.1: the scene prepares (auto-range, view transformations) before Qt computes the
# regions to repaint, so that each update is painted once.
# --------------------------------------------------------------------------------------

class _PaintLog(QtCore.QObject):
    """Event filter logging the paint events of a viewport, binding agnostic."""

    def __init__(self, log):
        super().__init__()
        self.log = log

    def eventFilter(self, obj, ev):
        if ev.type() == QtCore.QEvent.Type.Paint:
            self.log.append('paint')
        return False


def viewport_paints_per_update(widget, update, n=20, warmup=3):
    """Mean number of viewport paint events per update, events processed after each."""
    widget.show()
    process_events(5)
    for i in range(warmup):
        update(i)
        process_events()
    log = []
    paintLog = _PaintLog(log)
    widget.viewport().installEventFilter(paintLog)
    try:
        for i in range(warmup, warmup + n):
            update(i)
            process_events()
    finally:
        widget.viewport().removeEventFilter(paintLog)
    return len(log) / n


def test_streaming_with_autorange_paints_once():
    pw = pg.PlotWidget(size=(400, 300))
    curve = pw.plot(np.zeros(100))
    rng = np.random.default_rng(0)

    def update(i):
        # growing amplitude: the auto-range changes the y range on every update
        curve.setData(rng.normal(size=100) * (1 + i))

    try:
        assert viewport_paints_per_update(pw, update) <= 1.05
    finally:
        pw.close()


@pytest.mark.parametrize('axis', ['x', 'y'])
def test_pan_paints_once(axis):
    pw = pg.PlotWidget(size=(400, 300))
    pw.plot(np.random.default_rng(0).normal(size=1000))
    vb = pw.getViewBox()

    def update(i):
        vb.translateBy(**{axis: 0.5})

    try:
        assert viewport_paints_per_update(pw, update) <= 1.05
    finally:
        pw.close()


def test_linked_plots_streaming_paints_once():
    win = pg.GraphicsLayoutWidget(size=(600, 500))
    plots = []
    for row in range(3):
        p = win.addPlot(row=row, col=0)
        if plots:
            p.setXLink(plots[0])
        plots.append(p)
    rng = np.random.default_rng(1)
    curves = [p.plot(rng.normal(size=200)) for p in plots for _ in range(3)]
    line = pg.InfiniteLine(pos=0, angle=0)
    plots[0].addItem(line, ignoreBounds=True)

    def update(i):
        x = np.arange(200) + i
        for c in curves:
            c.setData(x, rng.normal(size=200))
        line.setPos(rng.normal())

    try:
        assert viewport_paints_per_update(win, update) <= 1.05
    finally:
        win.close()


def test_requested_prepare_runs_before_the_paint():
    pw = pg.PlotWidget(size=(300, 200))
    curve = pw.plot(np.arange(10.0))
    pw.show()
    process_events(5)
    scene = pw.scene()
    log = []
    scene.sigPrepareForPaint.connect(lambda: log.append('prepare'))
    paintLog = _PaintLog(log)
    pw.viewport().installEventFilter(paintLog)
    try:
        # no request: only the prepare made by GraphicsView.paintEvent, after the
        # paint event is received
        curve.curve.update()
        process_events()
        assert log == ['paint', 'prepare']

        # a request is served before the dirty items are processed
        del log[:]
        scene.requestPrepare()
        curve.curve.update()
        process_events()
        assert log[0] == 'prepare'
        assert log.count('paint') == 1
        assert not scene._prepareRequested
    finally:
        pw.viewport().removeEventFilter(paintLog)
        pw.close()


def test_autorange_is_applied_before_the_paint():
    pw = pg.PlotWidget(size=(300, 200))
    curve = pw.plot(np.arange(10.0))
    pw.show()
    process_events(5)
    vb = pw.getViewBox()
    ranges = []
    paintLog = _PaintLog([])
    original = paintLog.eventFilter

    def eventFilter(obj, ev):
        if ev.type() == QtCore.QEvent.Type.Paint:
            ranges.append(vb.viewRange()[1][1])
        return original(obj, ev)

    paintLog.eventFilter = eventFilter
    pw.viewport().installEventFilter(paintLog)
    try:
        curve.setData(np.arange(10.0) * 10)
        process_events()
        assert len(ranges) == 1
        assert ranges[0] >= 90  # the paint already shows the new auto-range
    finally:
        pw.viewport().removeEventFilter(paintLog)
        pw.close()


def test_hidden_view_defers_the_prepare():
    pw = pg.PlotWidget(size=(300, 200))
    curve = pw.plot(np.arange(10.0))
    show_and_wait(pw)
    pw.hide()
    vb = pw.getViewBox()
    before = vb.viewRange()
    calls = []
    pw.scene().sigPrepareForPaint.connect(lambda: calls.append(1))
    try:
        curve.setData(np.arange(10.0) * 10)
        process_events()
        # as before, a hidden plot does not auto-range until it is painted again
        assert calls == []
        assert vb.viewRange() == before
        show_and_wait(pw)
        assert vb.viewRange()[1][1] >= 90
    finally:
        pw.close()


@pytest.mark.skipif(
    QtWidgets.QApplication.platformName() != 'offscreen',
    reason="reads back the backing store with QScreen.grabWindow",
)
def test_single_paint_renders_like_a_full_repaint():
    # What was painted incrementally, once per update, equals a full repaint.
    win = pg.GraphicsLayoutWidget(size=(500, 400))
    p1 = win.addPlot(row=0, col=0)
    p2 = win.addPlot(row=1, col=0)
    p2.setXLink(p1)
    rng = np.random.default_rng(2)
    curves = [p1.plot(rng.normal(size=100), pen=k) for k in range(5)]
    curves.append(p2.plot(rng.random(100), fillLevel=0, brush=(50, 50, 200, 100)))
    text = pg.TextItem('label', anchor=(0, 0.5))
    p1.addItem(text)
    win.show()
    process_events(5)

    def image(qimage):
        qimage = qimage.convertToFormat(pg.QtGui.QImage.Format.Format_ARGB32)
        return pg.functions.ndarray_from_qimage(qimage).copy()

    try:
        for i in range(8):
            x = np.arange(100) + 10 * i
            for k, c in enumerate(curves):
                c.setData(x, rng.normal(size=100) * (1 + i) + k)
            text.setPos(x[-1], 0)
            process_events()
            painted = image(win.screen().grabWindow(win.winId()).toImage())
            full = image(win.grab().toImage())
            np.testing.assert_array_equal(painted, full)
    finally:
        win.close()


def test_layout_invalidated_while_preparing_is_applied_before_the_paint():
    # An item resizing itself while preparing, as an axis measuring its tick labels
    # would, relayouts the plot before the paint instead of after it.
    pw = pg.PlotWidget(size=(400, 300))
    axis = pw.getPlotItem().getAxis('left')
    axis.setWidth(40)
    curve = pw.plot(np.zeros(100))
    vb = pw.getViewBox()
    rng = np.random.default_rng(0)

    def measureLabels():
        axis.setWidth(40 + 5 * (int(vb.viewRange()[1][1]) % 3))

    pw.scene().sigPrepareForPaint.connect(measureLabels)

    def update(i):
        curve.setData(rng.normal(size=100) * (1 + i))

    widths = set()
    try:
        assert viewport_paints_per_update(pw, lambda i: (update(i), widths.add(axis.width()))) <= 1.05
        assert len(widths) == 3  # the layout did change
    finally:
        pw.close()
