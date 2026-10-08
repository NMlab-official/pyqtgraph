"""
Performance regression tests for InfiniteLine (T1.8 of PERFORMANCE_PLAN.md).

The bounds of a line are recomputed when the view, the line or its style change, and
``boundingRect()`` only reads that cache: calling ``prepareGeometryChange`` from
``boundingRect`` makes Qt re-index the item while it is being painted.
"""
import numpy as np
import pytest

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from tests.perf_helpers import count_calls, process_events, show_and_wait

app = pg.mkQApp()

# keyword arguments of the lines used by the tests, covering the fast (horizontal and
# vertical) and the generic (oblique) bounds computations
LINE_SPECS = [
    dict(pos=3, angle=0),
    dict(pos=7, angle=90, pen=pg.mkPen('g', width=5)),
    dict(pos=(5, 5), angle=45, pen='c'),
    dict(pos=(2, 8), angle=30, markers=[('<|', 0.1, 12), ('o', 0.5, 8)]),
    dict(pos=6, angle=0, markers=[('<|>', 0.3, 10), ('v', 0.9, 14)], movable=True,
         hoverPen=pg.mkPen('r', width=4)),
    dict(pos=2, angle=90, markers=[('|>', 0.2, 10), ('^', 0.7, 10)], span=(0.2, 0.8)),
    dict(pos=(1.5, 1.5), angle=135, markers=[('>|<', 0.4, 9)]),
]


def _make_view(specs=LINE_SPECS, xRange=(0, 10), yRange=(0, 10), lineClass=pg.InfiniteLine):
    """Return ``(widget, viewbox, lines)`` with the lines added to a shown ViewBox."""
    widget = pg.GraphicsLayoutWidget()
    widget.resize(320, 240)
    vb = widget.addViewBox()
    vb.setRange(xRange=xRange, yRange=yRange, padding=0)
    lines = []
    for spec in specs:
        line = lineClass(**spec)
        vb.addItem(line)
        lines.append(line)
    for line in lines:
        if line.movable:
            line.setMouseHover(True)  # drawn with its (wider) hover pen
    show_and_wait(widget)
    return widget, vb, lines


def _render(widget):
    """Render the widget and return its pixels as an array."""
    process_events()
    img = widget.grab().toImage().convertToFormat(QtGui.QImage.Format.Format_ARGB32)
    return pg.functions.ndarray_from_qimage(img).copy()


def _reference_bounds(line):
    """Bounds computed with the generic view rectangle and pixel vectors."""
    vr = line.viewRect()
    _, ortho = line.pixelVectors(QtCore.QPointF(1, 0))
    px = 0 if ortho is None else abs(ortho.y())
    pw = max(line.pen.width() / 2, line.hoverPen.width() / 2)
    marker = max([size / 2 for _, _, size in line.markers], default=0)
    w = (marker + pw + 1) * px
    left = vr.left() + vr.width() * line.span[0]
    right = vr.left() + vr.width() * line.span[1]
    return left, -w, right - left, 2 * w


def _assert_bounds_match(lines):
    for line in lines:
        br = line.boundingRect()
        expected = _reference_bounds(line)
        actual = (br.left(), br.top(), br.width(), br.height())
        assert actual == pytest.approx(expected, rel=1e-9, abs=1e-12), line.angle


def _view_changes(widget, vb):
    """Pan, zoom, invert and resize the view, processing events after each step."""
    steps = [
        lambda: vb.setRange(xRange=(1.3, 11.7), yRange=(-0.4, 9.1), padding=0),
        lambda: vb.scaleBy((0.37, 0.61)),
        lambda: vb.translateBy((2.1, -1.7)),
        lambda: widget.resize(301, 229),
        lambda: vb.invertY(True),
        lambda: vb.invertX(True),
        lambda: vb.setRange(xRange=(1e6, 1e6 + 10), yRange=(-5, 15), padding=0),
    ]
    for step in steps:
        step()
        process_events()
        yield


class _TracingLine(pg.InfiniteLine):
    """InfiniteLine recording whether prepareGeometryChange runs inside boundingRect."""

    # Overriding in a subclass rather than patching pg.InfiniteLine: PySide6 caches
    # the Python override of a C++ virtual per object, and restoring a patched
    # virtual can leave a dangling override behind (a crash in a later test).
    calls = {'inside': 0, 'outside': 0, 'boundingRect': 0}
    depth = 0

    def boundingRect(self):
        cls = _TracingLine
        cls.depth += 1
        cls.calls['boundingRect'] += 1
        try:
            return super().boundingRect()
        finally:
            cls.depth -= 1

    def prepareGeometryChange(self):
        _TracingLine.calls['inside' if _TracingLine.depth else 'outside'] += 1
        super().prepareGeometryChange()


def test_boundingRect_never_calls_prepareGeometryChange():
    _TracingLine.calls = {'inside': 0, 'outside': 0, 'boundingRect': 0}
    widget, vb, lines = _make_view(lineClass=_TracingLine)
    for _ in _view_changes(widget, vb):
        lines[0].setPos(lines[0].value() + 0.5)
        lines[2].setPos((lines[2].pos().x() + 0.25, 5))
    widget.close()
    calls = _TracingLine.calls
    assert calls['boundingRect'] > 0
    assert calls['outside'] > 0
    assert calls['inside'] == 0


def test_view_change_computes_bounds_once_per_line():
    widget, vb, lines = _make_view()
    with count_calls(pg.InfiniteLine, '_computeBoundingRect') as computes:
        vb.setRange(xRange=(2, 9), yRange=(1, 8), padding=0)
        vb.updateMatrix()  # applies the range and emits sigTransformChanged
        assert computes.count == len(lines)
        computes.reset()
        for line in lines:
            for _ in range(3):
                line.boundingRect()
        process_events()
        assert computes.count == 0
    _assert_bounds_match(lines)
    widget.close()


def test_moving_line_along_its_normal_keeps_bounds():
    # crosshair use case: the local bounds of an axis-aligned line do not depend on
    # its offset along its normal, so Qt need not be told of a geometry change
    widget, vb, lines = _make_view(specs=[dict(pos=3, angle=0), dict(pos=4, angle=90)],
                                   xRange=(0, 10), yRange=(0, 10))
    hline, vline = lines
    rects = [QtCore.QRectF(line.boundingRect()) for line in lines]
    with count_calls(pg.InfiniteLine, 'prepareGeometryChange') as prepares, \
         count_calls(pg.InfiniteLine, '_computeBoundingRect') as computes:
        for i in range(10):
            hline.setPos(3 + 0.1 * i)
            vline.setPos(4 + 0.1 * i)
            process_events()
    assert prepares.count == 0
    assert computes.count == 2 * 9  # one per actual move
    assert [line.boundingRect() for line in lines] == rects
    widget.close()


@pytest.mark.parametrize('spec_index', range(len(LINE_SPECS)))
def test_bounds_match_generic_computation(spec_index):
    widget, vb, lines = _make_view()
    line = lines[spec_index]
    _assert_bounds_match([line])
    for _ in _view_changes(widget, vb):
        _assert_bounds_match([line])
    widget.close()


def test_style_changes_update_bounds_immediately():
    widget, vb, lines = _make_view(specs=[dict(pos=3, angle=0)] * 5)
    line = lines[0]
    vr = line.viewRect()
    line.setSpan(0.25, 0.75)
    br = line.boundingRect()
    assert br.left() == pytest.approx(vr.left() + 0.25 * vr.width())
    assert br.width() == pytest.approx(0.5 * vr.width())
    height = br.height()
    line.setPen(width=9)
    assert line.boundingRect().height() > height
    height = line.boundingRect().height()
    line.addMarker('o', size=40)
    assert line.boundingRect().height() > height
    line.clearMarkers()
    assert line.boundingRect().height() == pytest.approx(height)
    _assert_bounds_match([line])
    widget.close()


def test_paint_allocates_no_points():
    widget, vb, lines = _make_view()
    img = QtGui.QImage(320, 240, QtGui.QImage.Format.Format_ARGB32)
    painter = QtGui.QPainter(img)
    try:
        with count_calls(pg.Point, '__init__') as points:
            for line in lines:
                painter.save()
                painter.setTransform(line.sceneTransform())
                line.paint(painter, None, None)
                painter.restore()
    finally:
        painter.end()
    assert points.count == 0
    widget.close()


def test_rendering_after_view_changes_matches_fresh_view():
    final_range = dict(xRange=(-1.5, 12.25), yRange=(0.75, 9.5))
    widget, vb, lines = _make_view()
    for _ in _view_changes(widget, vb):
        pass
    widget.resize(320, 240)
    vb.invertX(False)
    vb.invertY(False)
    lines[0].setPos(4.25)
    lines[1].setPos(6.5)
    vb.setRange(**final_range, padding=0)
    image = _render(widget)
    widget.close()

    specs = [dict(spec) for spec in LINE_SPECS]
    specs[0]['pos'] = 4.25
    specs[1]['pos'] = 6.5
    ref_widget, ref_vb, ref_lines = _make_view(specs=specs, **final_range)
    reference = _render(ref_widget)
    ref_widget.close()

    assert image.shape == reference.shape
    # every line is visible and was drawn
    assert (image != _render_empty(final_range)).any()
    np.testing.assert_array_equal(image, reference)


def _render_empty(final_range):
    widget = pg.GraphicsLayoutWidget()
    widget.resize(320, 240)
    vb = widget.addViewBox()
    vb.setRange(**final_range, padding=0)
    widget.show()
    image = _render(widget)
    widget.close()
    return image
