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
