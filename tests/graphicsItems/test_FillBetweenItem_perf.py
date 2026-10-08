"""
Tests for the lazy, numpy-based path of :class:`~pyqtgraph.FillBetweenItem` (T3.3).

The rendering tests compare the item against the previous algorithm, which paired the
subpaths of both rendered curve paths, pixel by pixel.
"""
import numpy as np
import pytest

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from tests.perf_helpers import count_calls, process_events, show_and_wait

app = pg.mkQApp()

ODD_EVEN = QtCore.Qt.FillRule.OddEvenFill
WINDING = QtCore.Qt.FillRule.WindingFill

# A test case: both curves, and whether the fill is expected to be built with numpy
# (True) or by falling back to the subpaths of the rendered curve paths (False).
Curve = pg.PlotDataItem | pg.PlotCurveItem
Case = tuple[Curve, Curve, bool]


def _curveItem(curve: Curve) -> pg.PlotCurveItem:
    """The item drawing the line of `curve`."""
    return curve.curve if isinstance(curve, pg.PlotDataItem) else curve


def _legacyFillPath(fill: pg.FillBetweenItem) -> QtGui.QPainterPath:
    """Fill path built as before T3.3, from the subpaths of the rendered curve paths."""
    paths = [_curveItem(c).getPath() for c in fill.curves]
    ps1 = paths[0].toSubpathPolygons()
    ps2 = paths[1].toReversed().toSubpathPolygons()
    ps2.reverse()
    if len(ps1) == 0 or len(ps2) == 0:
        return QtGui.QPainterPath()
    path = QtGui.QPainterPath()
    path.setFillRule(fill.fillRule())
    for p1, p2 in zip(ps1, ps2):
        path.addPolygon(p1 + p2)
    return path


def _legacyItem(fill: pg.FillBetweenItem) -> QtWidgets.QGraphicsPathItem:
    """Plain QGraphicsPathItem drawing the legacy path with the style of `fill`."""
    item = QtWidgets.QGraphicsPathItem(_legacyFillPath(fill))
    item.setPen(fill.pen())
    item.setBrush(fill.brush())
    return item


def _render(item: QtWidgets.QGraphicsItem, rect: QtCore.QRectF,
            antialias: bool = True) -> QtGui.QImage:
    """Paint `item` into a 160x120 image showing `rect` of its coordinates."""
    width, height = 160, 120
    image = QtGui.QImage(width, height, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(QtCore.Qt.GlobalColor.white)
    painter = QtGui.QPainter(image)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, antialias)
    transform = QtGui.QTransform()
    transform.scale(width / rect.width(), -height / rect.height())
    transform.translate(-rect.left(), -rect.bottom())
    painter.setTransform(transform)
    option = QtWidgets.QStyleOptionGraphicsItem()
    if item.isSelected():
        option.state = option.state | QtWidgets.QStyle.StateFlag.State_Selected
    item.paint(painter, option, None)
    painter.end()
    return image


def _filledPixels(image: QtGui.QImage) -> int:
    """Number of pixels that are not white."""
    pixels = pg.functions.ndarray_from_qimage(image)
    return int(np.count_nonzero((pixels != 255).any(axis=-1)))


def _dataRect(fill: pg.FillBetweenItem) -> QtCore.QRectF:
    """Rectangle around the finite data of both curves, with a margin."""
    xs, ys = [], []
    for c in fill.curves:
        x, y = _curveItem(c).getData()
        xs.append(x[np.isfinite(x)])
        ys.append(y[np.isfinite(y)])
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    dx = (x.max() - x.min()) * 0.05 + 1e-3
    dy = (y.max() - y.min()) * 0.05 + 1e-3
    return QtCore.QRectF(x.min() - dx, y.min() - dy,
                         x.max() - x.min() + 2 * dx, y.max() - y.min() + 2 * dy)


def _assertSameRendering(fill: pg.FillBetweenItem) -> QtGui.QImage:
    """Assert `fill` renders like the legacy algorithm; return its image."""
    rect = _dataRect(fill)
    reference = _legacyItem(fill)
    for antialias in (True, False):
        image = _render(fill, rect, antialias)
        expected = _render(reference, rect, antialias)
        assert _filledPixels(expected) > 0
        assert image == expected
    assert fill.boundingRect() == reference.boundingRect()
    return image


def _usesCurvePaths(fill: pg.FillBetweenItem) -> bool:
    """Whether rebuilding the fill falls back to building the curve paths."""
    for c in fill.curves:
        _curveItem(c).path = None
    with count_calls(pg.PlotCurveItem, 'getPath') as getPath:
        fill.updatePath()
    return getPath.count > 0


def _withNaN(values: np.ndarray, index: slice | list[int] | np.ndarray) -> np.ndarray:
    """Copy of `values` with NaN at `index`."""
    values = values.astype(np.float64)
    values[index] = np.nan
    return values


X = np.linspace(0.0, 10.0, 200)
Y = np.sin(X)


def _caseSimple() -> Case:
    return pg.PlotCurveItem(X, Y + 1.5), pg.PlotCurveItem(X, Y * 0.5 - 1.0), True


def _caseCrossing() -> Case:
    return pg.PlotCurveItem(X, np.sin(X)), pg.PlotCurveItem(X, np.cos(X)), True


def _casePlotDataItems() -> Case:
    return pg.PlotDataItem(X, Y + 1.0), pg.PlotDataItem(X, Y - 1.0), True


def _caseWarmupNaN() -> Case:
    # moving average bands: undefined over the first window
    upper = _withNaN(Y + 1.0, slice(0, 19))
    lower = _withNaN(Y - 1.0, slice(0, 19))
    return pg.PlotDataItem(X, upper), pg.PlotDataItem(X, lower), True


def _caseNaNConnectAll() -> Case:
    # connect='all' joins the neighbours of non-finite points
    c1 = pg.PlotCurveItem(X, _withNaN(Y + 1.0, [40, 41, 120]))
    c2 = pg.PlotCurveItem(X, _withNaN(Y - 1.0, [80, 150, 151]))
    return c1, c2, True


def _caseCommonGaps() -> Case:
    gaps = np.r_[50:60, 130:134]
    c1 = pg.PlotCurveItem(X, _withNaN(Y + 1.0, gaps), connect='finite')
    c2 = pg.PlotCurveItem(X, _withNaN(Y - 1.0, gaps), connect='finite')
    return c1, c2, True


def _caseMisalignedGaps() -> Case:
    # different gaps: the i-th runs of finite points of both curves are paired
    c1 = pg.PlotCurveItem(X, _withNaN(Y + 1.0, [30, 31, 199]), connect='finite')
    c2 = pg.PlotCurveItem(X, _withNaN(Y - 1.0, [0, 50, 100, 101]), connect='finite')
    return c1, c2, True


def _caseIsolatedPoint() -> Case:
    # a lone finite point between two non-finite ones
    c1 = pg.PlotCurveItem(X, _withNaN(Y + 1.0, [60, 62, 140]), connect='finite')
    c2 = pg.PlotCurveItem(X, _withNaN(Y - 1.0, [60, 62, 140]), connect='finite')
    return c1, c2, False


def _caseRepeatedPoint() -> Case:
    y2 = Y - 1.0
    x2 = X.copy()
    x2[1] = x2[0]
    y2[1] = y2[0]
    return pg.PlotCurveItem(X, Y + 1.0), pg.PlotCurveItem(x2, y2), False


def _caseDifferentLengths() -> Case:
    x2 = np.linspace(2.0, 8.0, 77)
    return pg.PlotCurveItem(X, Y + 1.0), pg.PlotCurveItem(x2, np.cos(x2) - 1.0), True


def _caseLarge() -> Case:
    # more than 20000 points: chunked arrayToQPath
    x = np.linspace(0.0, 10.0, 30_000)
    y = np.cumsum(np.random.default_rng(1).standard_normal(len(x))) * 0.05
    return pg.PlotCurveItem(x, y + 1.0), pg.PlotCurveItem(x, y - 1.0), True


def _caseSkipFiniteCheck() -> Case:
    c1 = pg.PlotCurveItem(X, Y + 1.0, skipFiniteCheck=True)
    c2 = pg.PlotCurveItem(X, Y - 1.0, skipFiniteCheck=True)
    return c1, c2, True


def _caseStepCenter() -> Case:
    xe = np.linspace(0.0, 10.0, 41)
    y = np.sin(xe[:-1])
    c1 = pg.PlotCurveItem(xe, y + 1.0, stepMode='center')
    c2 = pg.PlotCurveItem(xe, y * 0.5 - 1.0, stepMode='center')
    return c1, c2, False


def _caseStepLeftRight() -> Case:
    x = np.linspace(0.0, 10.0, 40)
    c1 = pg.PlotCurveItem(x, np.sin(x) + 1.0, stepMode='left')
    c2 = pg.PlotCurveItem(x, np.cos(x) - 1.0, stepMode='right')
    return c1, c2, False


def _caseConnectPairs() -> Case:
    c1 = pg.PlotCurveItem(X, Y + 1.0, connect='pairs')
    c2 = pg.PlotCurveItem(X, Y - 1.0)
    return c1, c2, False


def _caseConnectArray() -> Case:
    connect = np.ones(len(X), dtype=bool)
    connect[[70, 140]] = False
    c1 = pg.PlotCurveItem(X, Y + 1.0, connect=connect)
    c2 = pg.PlotCurveItem(X, Y - 1.0, connect=connect)
    return c1, c2, False


CASES = {
    'simple': _caseSimple,
    'crossing': _caseCrossing,
    'PlotDataItem': _casePlotDataItems,
    'warmup NaN': _caseWarmupNaN,
    "NaN connect='all'": _caseNaNConnectAll,
    'common gaps': _caseCommonGaps,
    'misaligned gaps': _caseMisalignedGaps,
    'isolated point': _caseIsolatedPoint,
    'repeated point': _caseRepeatedPoint,
    'different lengths': _caseDifferentLengths,
    'large': _caseLarge,
    'skipFiniteCheck': _caseSkipFiniteCheck,
    "stepMode='center'": _caseStepCenter,
    "stepMode='left'/'right'": _caseStepLeftRight,
    "connect='pairs'": _caseConnectPairs,
    'connect array': _caseConnectArray,
}


@pytest.mark.parametrize('pen', [None, pg.mkPen('k', width=2)], ids=['no pen', 'pen'])
@pytest.mark.parametrize('fillRule', [ODD_EVEN, WINDING], ids=['OddEven', 'Winding'])
@pytest.mark.parametrize('case', CASES.values(), ids=CASES.keys())
def test_fill_matches_previous_algorithm(case, fillRule, pen):
    curve1, curve2, numpyPath = case()
    fill = pg.FillBetweenItem(curve1, curve2, brush=(50, 50, 200, 160), pen=pen,
                              fillRule=fillRule)
    _assertSameRendering(fill)
    assert fill.path().fillRule() == fillRule
    # the numpy path is used whenever it reproduces the subpaths of the curves
    assert _usesCurvePaths(fill) is not numpyPath


def test_fill_rule_changes_rendering():
    # a spiral winding twice around its centre: OddEven and Winding fill differently
    t = np.linspace(0.0, 4.0 * np.pi, 400)
    radius = 1.0 + 0.15 * t
    spiral = pg.PlotCurveItem(radius * np.cos(t), radius * np.sin(t))
    center = pg.PlotCurveItem(np.array([0.2, 0.0]), np.array([0.0, 0.2]))
    fill = pg.FillBetweenItem(spiral, center, brush='b')
    oddEven = _assertSameRendering(fill)
    fill.setFillRule(WINDING)
    winding = _assertSameRendering(fill)
    assert _filledPixels(winding) > _filledPixels(oddEven)
    assert not _usesCurvePaths(fill)


def test_fill_follows_curve_updates():
    curve1, curve2, _ = _caseCommonGaps()
    fill = pg.FillBetweenItem(curve1, curve2, brush='b')
    _assertSameRendering(fill)
    curve1.setData(X, Y + 2.0, connect='finite')
    curve2.setData(X[:150], Y[:150] - 0.5, connect='finite')
    _assertSameRendering(fill)
    curve2.setData(X, Y - 0.5, stepMode='right')
    _assertSameRendering(fill)


def test_empty_curves():
    fill = pg.FillBetweenItem(pg.PlotCurveItem(), pg.PlotCurveItem(X, Y))
    assert fill.path().isEmpty()
    assert fill.boundingRect() == QtCore.QRectF()
    assert pg.FillBetweenItem(None, None).path().isEmpty()


@pytest.mark.parametrize('itemClass', [pg.PlotDataItem, pg.PlotCurveItem])
def test_single_rebuild_per_frame(itemClass):
    pw = pg.PlotWidget()
    pw.resize(300, 200)
    x = np.arange(5000.0)
    y = np.cumsum(np.random.default_rng(0).standard_normal(len(x)))
    upper = itemClass(x, y + 2.0)
    lower = itemClass(x, y - 2.0)
    fill = pg.FillBetweenItem(upper, lower, brush=(50, 50, 200, 80))
    for item in (upper, lower, fill):
        pw.addItem(item)
    show_and_wait(pw)
    with count_calls(pg.FillBetweenItem, 'updatePath') as rebuilds:
        for k in range(3):
            upper.setData(x, y + 2.0 + k)
            lower.setData(x, y - 2.0 - k)
            process_events()
        assert rebuilds.count == 3
        # the path is up to date and stays cached
        assert fill.path() == _legacyFillPath(fill)
        fill.boundingRect()
        fill.shape()
        assert rebuilds.count == 3
    pw.close()


def test_no_rebuild_while_hidden():
    pw = pg.PlotWidget()
    pw.resize(300, 200)
    upper = pw.plot(X, Y + 1.0)
    lower = pw.plot(X, Y - 1.0)
    fill = pg.FillBetweenItem(upper, lower, brush=(50, 50, 200, 80))
    pw.addItem(fill)
    pw.show()
    process_events()
    fill.hide()
    process_events()
    with count_calls(pg.FillBetweenItem, 'updatePath') as rebuilds:
        for k in range(3):
            upper.setData(X, Y + 1.0 + k)
            lower.setData(X, Y - 1.0 - k)
            process_events()
        assert rebuilds.count == 0
        fill.show()
        process_events()
        assert rebuilds.count == 1
    assert fill.path() == _legacyFillPath(fill)
    pw.close()


def test_curve_change_only_marks_the_path():
    curve1, curve2, _ = _caseSimple()
    fill = pg.FillBetweenItem(curve1, curve2)
    fill.path()
    with count_calls(fill, 'prepareGeometryChange') as geometryChanges, \
            count_calls(pg.FillBetweenItem, 'updatePath') as rebuilds:
        curve1.setData(X, Y + 2.0)
        curve2.setData(X, Y - 2.0)
        assert rebuilds.count == 0
        assert geometryChanges.count == 1
        # the lazy rebuild must not announce a geometry change from boundingRect()
        rect = fill.boundingRect()
        assert rebuilds.count == 1
        assert geometryChanges.count == 1
        assert rect.top() == pytest.approx(-2.0 + Y.min(), abs=1e-6)
        # an explicit refresh announces it
        fill.updatePath()
        assert geometryChanges.count == 2


@pytest.mark.parametrize('pen', [
    None, 'r', pg.mkPen('r', width=0), pg.mkPen('r', width=3, cosmetic=False)
], ids=['None', 'cosmetic', 'width 0', 'width 3'])
def test_geometry_matches_QGraphicsPathItem(pen):
    curve1, curve2, _ = _caseSimple()
    fill = pg.FillBetweenItem(curve1, curve2, pen=pen)
    reference = _legacyItem(fill)
    assert fill.path() == reference.path()
    assert fill.boundingRect() == reference.boundingRect()
    assert fill.shape() == reference.shape()
    point = QtCore.QPointF(5.0, 0.0)
    assert fill.contains(point) == reference.contains(point)
    # changing the pen updates the geometry
    fill.setPen(width=5, cosmetic=False)
    reference.setPen(fill.pen())
    assert fill.boundingRect() == reference.boundingRect()
    assert fill.shape() == reference.shape()


def test_selected_item_draws_selection_outline():
    curve1, curve2, _ = _caseSimple()
    fill = pg.FillBetweenItem(curve1, curve2, brush='b')
    reference = _legacyItem(fill)
    for item in (fill, reference):
        item.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        item.setSelected(True)
    rect = _dataRect(fill)
    assert _render(fill, rect) == _render(reference, rect)


def test_setPath_until_next_curve_change():
    curve1, curve2, _ = _caseSimple()
    fill = pg.FillBetweenItem(curve1, curve2)
    custom = QtGui.QPainterPath()
    custom.addRect(QtCore.QRectF(0.0, 0.0, 1.0, 1.0))
    fill.setPath(custom)
    assert fill.path() == custom
    assert fill.boundingRect() == QtCore.QRectF(0.0, 0.0, 1.0, 1.0)
    curve1.setData(X, Y + 2.0)
    assert fill.path() == _legacyFillPath(fill)
