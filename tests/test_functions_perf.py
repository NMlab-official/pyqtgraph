"""
Regression tests of the private helpers that speed up ``functions.arrayToQPath``.

The faster drawing code of ``PlotCurveItem`` relies on them reproducing exactly the
paths built by ``arrayToQPath``.
"""
import numpy as np
import pytest

import pyqtgraph as pg
from pyqtgraph import functions as fn

app = pg.mkQApp()


def _path_vertices(path):
    """
    Return the coordinates and the element types of a path made of straight lines.

    Parameters
    ----------
    path : QtGui.QPainterPath
        The path.

    Returns
    -------
    x, y : np.ndarray
        Coordinates of the elements.
    types : list of bool
        True for a MoveTo element, False for a LineTo element.
    """
    n = path.elementCount()
    x = np.empty(n)
    y = np.empty(n)
    moves = []
    for i in range(n):
        element = path.elementAt(i)
        x[i] = element.x
        y[i] = element.y
        moves.append(element.isMoveTo())
    return x, y, moves


@pytest.mark.parametrize('n', [0, 1, 2, 5, 19999, 20000, 20001, 20002, 30001, 40001])
@pytest.mark.parametrize('finiteCheck, nonfinite',
                         [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize('duplicates', [False, True])
def test_arrayToQPath_all_vertices_match_path(n, finiteCheck, nonfinite, duplicates):
    rng = np.random.default_rng(n)
    x = np.arange(n, dtype=np.float64)
    y = np.round(rng.standard_normal(n), 1)
    if duplicates:
        # equal consecutive points across the chunk boundaries of _arrayToQPath_all
        for start in range(fn._ARRAYTOQPATH_CHUNKSIZE, n, fn._ARRAYTOQPATH_CHUNKSIZE):
            x[start] = x[start - 1]
            y[start] = y[start - 1]
    if nonfinite and n > 2:
        y[1::7] = np.nan
        x[3::11] = np.inf

    path = fn._arrayToQPath_all(x, y, finiteCheck)
    vx, vy = fn._arrayToQPath_all_vertices(x, y, finiteCheck)

    if len(vx) < 2:
        assert path.elementCount() == 0
        return
    px, py, moves = _path_vertices(path)
    np.testing.assert_array_equal(vx, px)
    np.testing.assert_array_equal(vy, py)
    assert moves == [True] + [False] * (len(moves) - 1)


def test_arrayToQPath_all_vertices_without_change_are_the_inputs():
    # no copy is made when all points are kept
    x = np.arange(1000, dtype=np.float64)
    y = np.sin(x)
    vx, vy = fn._arrayToQPath_all_vertices(x, y, finiteCheck=True)
    assert vx is x and vy is y


def test_arrayToQPath_all_vertices_single_point_last_chunk():
    # a single remaining point is appended to the previous chunk: connectPath would
    # ignore a chunk made of it, and the last segment would not be drawn
    n = 2 * fn._ARRAYTOQPATH_CHUNKSIZE + 1
    x = np.arange(n, dtype=np.float64)
    y = np.zeros(n)
    vx, vy = fn._arrayToQPath_all_vertices(x, y, finiteCheck=False)
    assert len(vx) == n
    assert fn.arrayToQPath(x, y).elementCount() == n


@pytest.mark.parametrize('a, b, expected', [
    ((1.0, 2.0), (1.0, 2.0), True),
    ((1.0, 2.0), (1.0 + 1e-14, 2.0), True),
    ((1.0, 2.0), (1.0 + 1e-9, 2.0), False),
    ((0.0, 0.0), (1e-13, -1e-13), True),
    ((0.0, 0.0), (1e-11, 0.0), False),
    ((np.nan, 0.0), (np.nan, 0.0), False),
])
def test_qpointf_fuzzy_equal(a, b, expected):
    from pyqtgraph.Qt import QtCore
    assert fn._qpointf_fuzzy_equal(*a, *b) is expected
    assert (QtCore.QPointF(*a) == QtCore.QPointF(*b)) is expected


# --------------------------------------------------------------------------------------
# downsample: blocks averaged by adding strided slices, as reshape and mean do
# --------------------------------------------------------------------------------------

def _reference_downsample(data, n, axis):
    nPts = data.shape[axis] // n
    shape = list(data.shape)
    shape[axis] = nPts
    shape.insert(axis + 1, n)
    sl = [slice(None)] * data.ndim
    sl[axis] = slice(0, nPts * n)
    return data[tuple(sl)].reshape(shape).mean(axis + 1)


def _downsample_data(dtype, layout, rng):
    shape = (61, 47)
    dtype = np.dtype(dtype)
    if dtype.kind == 'f':
        data = (rng.standard_normal(shape) * 1e3).astype(dtype)
        data[3, 5] = np.nan
        data[10, 2] = np.inf
    elif dtype.kind == 'b':
        data = rng.random(shape) > 0.5
    else:
        info = np.iinfo(dtype)
        data = rng.integers(info.min, info.max, shape, dtype=dtype, endpoint=True)
    if layout == 'F':
        data = np.asfortranarray(data)
    elif layout == 'strided':
        data = np.repeat(data, 2, axis=1)[:, ::2]
    return data


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'float16', 'uint8', 'uint16',
                                   'int16', 'int32', 'uint32', 'int64', 'bool'])
@pytest.mark.parametrize('n', [2, 3, 4, 5, 7, 8, 9])
@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('layout', ['C', 'F', 'strided'])
def test_downsample_identical_to_mean(dtype, n, axis, layout):
    data = _downsample_data(dtype, layout, np.random.default_rng(n * 10 + axis))
    expected = _reference_downsample(data, n, axis)
    with np.errstate(invalid='ignore'):  # inf - inf in the blocks holding both
        result = fn.downsample(data, n, axis=axis)
    assert result.dtype == expected.dtype
    np.testing.assert_array_equal(result, expected)


def test_downsample_by_slices_used_where_faster():
    image = np.zeros((40, 30), dtype=np.float32)
    # float32: any axis, fewer than 8 values per block
    assert fn._downsampleBySlicesIsExact(image, 3, 1, 10)
    assert fn._downsampleBySlicesIsExact(image, 7, 0, 5)
    assert not fn._downsampleBySlicesIsExact(image, 8, 1, 3)
    # other types: along the axis contiguous in memory only
    assert fn._downsampleBySlicesIsExact(image.astype(np.uint16), 5, 1, 6)
    assert not fn._downsampleBySlicesIsExact(image.astype(np.uint16), 5, 0, 8)
    assert fn._downsampleBySlicesIsExact(image.astype(np.float64), 3, 1, 10)
    assert not fn._downsampleBySlicesIsExact(image.astype(np.float64), 4, 1, 7)
    assert not fn._downsampleBySlicesIsExact(image.astype(np.int64), 2, 1, 15)
