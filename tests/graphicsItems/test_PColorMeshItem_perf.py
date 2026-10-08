"""
Regression tests for the vectorised quad construction and drawing of PColorMeshItem.
"""
import numpy as np
import pytest

import pyqtgraph as pg
import pyqtgraph.functions as fn
from pyqtgraph.graphicsItems.PColorMeshItem import QuadInstances
from pyqtgraph.Qt import QtCore, QtGui

app = pg.mkQApp()


@pytest.mark.parametrize('nrows,ncols', [(0, 0), (1, 1), (3, 5), (7, 2)])
def test_quad_instances_share_vertices(nrows, ncols):
    quads = QuadInstances()
    quads.resize(nrows, ncols)
    points = quads.pointsarray.instances()
    polys = quads.instances()
    assert len(polys) == nrows * ncols
    for r in range(nrows):
        for c in range(ncols):
            expected = (
                points[r * (ncols + 1) + c],            # (x[r, c], y[r, c])
                points[(r + 1) * (ncols + 1) + c],      # (x[r+1, c], y[r+1, c])
                points[(r + 1) * (ncols + 1) + c + 1],  # (x[r+1, c+1], y[r+1, c+1])
                points[r * (ncols + 1) + c + 1],        # (x[r, c+1], y[r, c+1])
            )
            poly = polys[r * ncols + c]
            assert isinstance(poly, tuple)
            assert all(a is b for a, b in zip(poly, expected))

    # the quads follow the vertex memory
    if nrows and ncols:
        quads.ndarray()[:] = np.arange(2 * len(points)).reshape(-1, 2)
        assert polys[0][1] == QtCore.QPointF(2 * (ncols + 1), 2 * (ncols + 1) + 1)


@pytest.mark.parametrize('nans', [False, True])
def test_quad_colors(nans):
    rng = np.random.default_rng(0)
    nx, ny = 12, 9
    x, y = np.meshgrid(np.arange(nx + 1.0), np.arange(ny + 1.0), indexing='ij')
    z = rng.random((nx, ny))
    if nans:
        z[rng.random(z.shape) < 0.2] = np.nan
    cmap = pg.colormap.get('viridis')
    mesh = pg.PColorMeshItem(x, y, z, colorMap=cmap, levels=(0, 1), enableAutoLevels=False)
    img = QtGui.QImage(nx * 10, ny * 10, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(0)
    p = QtGui.QPainter(img)
    p.scale(10, 10)
    mesh.paint(p, None, None)
    p.end()
    pixels = fn.ndarray_from_qimage(img).astype(int)
    lut = cmap.getLookupTable(nPts=256, mode=cmap.QCOLOR)
    for i in range(nx):
        for j in range(ny):
            b, g, r, a = pixels[j * 10 + 5, i * 10 + 5]
            if np.isnan(z[i, j]):
                assert a == 0
            else:
                index = int(np.clip(np.floor(z[i, j] * 255), 0, 255))
                assert (r, g, b, a) == lut[index].getRgb()
