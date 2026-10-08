from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from .. import functions as fn
from ..colormap import ColorMap
from .. import Qt
from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject

__all__ = ['NonUniformImage']

# paint engines drawing to a pixel grid, on which the cells are drawn as one image
_PIXEL_ENGINES = (
    QtGui.QPaintEngine.Type.Raster,
    QtGui.QPaintEngine.Type.OpenGL,
    QtGui.QPaintEngine.Type.OpenGL2,
)
# transforms keeping the cells axis-aligned
_RECTILINEAR = (
    QtGui.QTransform.TransformationType.TxNone,
    QtGui.QTransform.TransformationType.TxTranslate,
    QtGui.QTransform.TransformationType.TxScale,
)
# the raster engine fills aliased rectangles at device coordinates rounded to 1/64 pixel
_SUBPIXELS = 64


def _lutToArgb(lut: npt.ArrayLike) -> np.ndarray:
    """
    Convert the entries of a lookup table to ARGB32 values.

    Parameters
    ----------
    lut : array_like
        Lookup table whose entries are accepted by :func:`~pyqtgraph.mkColor`.

    Returns
    -------
    np.ndarray
        uint32 array of the colors as ``QColor.rgba()`` values (not premultiplied).
    """
    if (
        isinstance(lut, np.ndarray)
        and lut.ndim == 2
        and lut.shape[1] in (3, 4)
        and lut.dtype.kind in 'ui'
        and (lut.size == 0 or (lut.min() >= 0 and lut.max() <= 255))
    ):
        # the common uint8 tables, converted as mkColor converts their rows
        channels = lut.astype(np.uint32)
        alpha = channels[:, 3] if lut.shape[1] == 4 else np.uint32(255)
        return (alpha << 24) | (channels[:, 0] << 16) | (channels[:, 1] << 8) | channels[:, 2]
    return np.array([fn.mkBrush(lut[i]).color().rgba() for i in range(len(lut))],
                    dtype=np.uint32)


def _premultipliedArgb(argb: np.ndarray) -> np.ndarray:
    """
    Premultiply ARGB32 values by their alpha.

    Parameters
    ----------
    argb : np.ndarray
        uint32 array of ``QColor.rgba()`` values.

    Returns
    -------
    np.ndarray
        uint32 array of the colors as ``QImage.Format_ARGB32_Premultiplied`` pixels,
        rounded as ``qPremultiply`` does.
    """
    alpha = argb >> 24
    premultiplied = argb & 0xFF000000
    for shift in (16, 8, 0):
        t = ((argb >> shift) & 0xFF) * alpha
        premultiplied |= ((t + (t >> 8) + 0x80) >> 8) << shift
    return premultiplied


def _pixelCells(start: np.ndarray, size: np.ndarray, scale: float, offset: float,
                first: int, stop: int) -> np.ndarray:
    """
    Map device pixels along one axis to the cells covering them.

    The cells are mapped to device coordinates as the raster paint engine maps aliased
    rectangles: their edges are rounded to 1/64 pixel, and a pixel belongs to the cell
    whose rounded edges enclose its center, excluding the low edge.

    Parameters
    ----------
    start : np.ndarray
        Low edge of each cell, in item coordinates, increasing.
    size : np.ndarray
        Extent of each cell, in item coordinates.
    scale : float
        Scale of the item-to-device transform along this axis.
    offset : float
        Translation of the item-to-device transform along this axis.
    first : int
        First device pixel.
    stop : int
        Device pixel after the last one.

    Returns
    -------
    np.ndarray
        Index of the cell covering each pixel of ``first .. stop - 1``, or
        ``len(start)`` for pixels outside of the cells.
    """
    n = len(start)
    a = scale * start + offset
    b = scale * (start + size) + offset  # as QRectF.right() mapped to the device
    low = np.floor(np.minimum(a, b) * _SUBPIXELS + 0.5)
    high = np.floor(np.maximum(a, b) * _SUBPIXELS + 0.5)
    if scale < 0:
        low, high = low[::-1], high[::-1]
    centers = np.arange(first, stop) * _SUBPIXELS + _SUBPIXELS // 2
    cell = np.searchsorted(low, centers) - 1
    covered = (cell >= 0) & (centers <= high[np.maximum(cell, 0)])
    if scale < 0:
        cell = n - 1 - cell
    return np.where(covered, cell, n)


class NonUniformImage(GraphicsObject):
    """
    **Bases:** :class:`GraphicsObject <pyqtgraph.GraphicsObject>`

    GraphicsObject displaying an image with non-uniform sample points. It's
    commonly used to display 2-d or slices of higher dimensional data that
    have a regular but non-uniform grid e.g. measurements or simulation results.

    Each value ``z[i, j]`` is displayed as a rectangular cell around ``(x[i], y[j])``,
    extending half way to the neighboring sample points. The colors are given by the
    levels and the lookup table, NaN values are transparent.

    On pixel-based paint devices (widgets, images), the cells visible on the device are
    drawn as a single image, regenerated only when the data, levels, lookup table or
    view change; as with :class:`~pyqtgraph.ImageItem`, cell edges are not antialiased.
    Other paint devices (e.g. SVG or PDF export) and rotated views get one rectangle per
    cell.

    Parameters
    ----------
    x : array_like
        Increasing 1-d array of the x coordinates of the samples, of length N.
    y : array_like
        Increasing 1-d array of the y coordinates of the samples, of length M.
    z : array_like
        2-d array of shape (N, M) of the sample values.
    border : QPen or None, default None
        Pen of the border drawn around the image, if any.
    """
    def __init__(self, x: npt.ArrayLike, y: npt.ArrayLike, z: npt.ArrayLike,
                 border: QtGui.QPen | None = None):

        GraphicsObject.__init__(self)

        # default colormap (black - white)
        self.cmap = ColorMap(None, [0.0, 1.0])
        self.lut = self.cmap.getLookupTable(nPts=256)

        self.levels = None
        self._levelsFromData = False  # levels computed by getLevels from the data
        self.border = border
        self._picture = None    # one rectangle per cell, for other paint devices
        self._cellColors = None  # (M, N) premultiplied ARGB32 colors of the cells
        self._rendered = None   # (view key, QImage) of the visible cells

        self.data = self._checkedData(x, y, z)
        self.update()

    def setData(self, x: npt.ArrayLike, y: npt.ArrayLike, z: npt.ArrayLike):
        """
        Replace the sample points and values.

        The lookup table, the levels set by :meth:`setLevels` and the border are kept;
        levels computed from the previous data are computed again. This is faster than
        replacing the item.

        Parameters
        ----------
        x : array_like
            Increasing 1-d array of the x coordinates of the samples, of length N.
        y : array_like
            Increasing 1-d array of the y coordinates of the samples, of length M.
        z : array_like
            2-d array of shape (N, M) of the sample values.

        Raises
        ------
        ValueError
            If the arrays have the wrong dimensions or `x` or `y` is decreasing.
        """
        data = self._checkedData(x, y, z)
        self.prepareGeometryChange()
        self.data = data
        if self._levelsFromData:
            self.levels = None
        self._invalidate()
        self.informViewBoundsChanged()
        self.update()

    @staticmethod
    def _checkedData(
        x: npt.ArrayLike, y: npt.ArrayLike, z: npt.ArrayLike
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Validate the sample points and values and convert them to float64 arrays.

        Parameters
        ----------
        x : array_like
            Increasing 1-d array of the x coordinates of the samples, of length N.
        y : array_like
            Increasing 1-d array of the y coordinates of the samples, of length M.
        z : array_like
            2-d array of shape (N, M) of the sample values.

        Returns
        -------
        tuple of np.ndarray
            The converted `x`, `y` and `z`.

        Raises
        ------
        ValueError
            If the arrays have the wrong dimensions or `x` or `y` is decreasing.
        """
        # convert to numpy arrays
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)

        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("x and y must be 1-d arrays.")

        if np.any(np.diff(x) < 0) or np.any(np.diff(y) < 0):
            raise ValueError("The values in x and y must be monotonically increasing.")

        if len(z.shape) != 2 or z.shape != (x.size, y.size):
            raise ValueError("The length of x and y must match the shape of z.")

        return x, y, z

    @property
    def picture(self) -> QtGui.QPicture | None:
        """
        QtGui.QPicture or None: The cells drawn as rectangles.

        It is only generated for paint devices that are not pixel-based. Setting it to
        None discards every cached rendering, e.g. after modifying :attr:`data` in place.
        """
        return self._picture

    @picture.setter
    def picture(self, picture: QtGui.QPicture | None):
        self._invalidate()
        self._picture = picture

    def _invalidate(self):
        """Discard the cell colors and every rendering made from them."""
        self._cellColors = None
        self._picture = None
        self._rendered = None

    def setLookupTable(self, lut: npt.ArrayLike | Callable | None, update: bool = True,
                       **kwargs):
        """
        Set the lookup table mapping the levels range to colors.

        Parameters
        ----------
        lut : array_like or callable or None
            Table of colors accepted by :func:`~pyqtgraph.mkColor` (typically an
            (N, 3) or (N, 4) uint8 array), a callable returning such a table from the
            values, or None for a grayscale ramp.
        update : bool, default True
            Repaint the item.
        **kwargs
            Unused, accepted for compatibility with :meth:`ImageItem.setLookupTable`.
        """
        self.cmap = None    # invalidate since no longer consistent with lut
        self.lut = lut
        self._invalidate()
        if update:
            self.update()

    def setColorMap(self, cmap: ColorMap):
        """
        Set the colors from a color map, sampled at 256 points.

        Parameters
        ----------
        cmap : ColorMap
            Color map.
        """
        self.setLookupTable(cmap.getLookupTable(nPts=256), update=True)
        self.cmap = cmap

    def getHistogram(self, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the histogram of the finite values.

        Parameters
        ----------
        **kwargs
            Arguments of :func:`numpy.histogram`.

        Returns
        -------
        np.ndarray, np.ndarray
            Left edges of the bins and counts. For an explanation of the return
            format, see :func:`numpy.histogram`.
        """

        z = self.data[2]
        z = z[np.isfinite(z)]
        hist = np.histogram(z, **kwargs)

        return hist[1][:-1], hist[0]

    def setLevels(self, levels: tuple[float, float] | None):
        """
        Set the values mapped to the first and last colors of the lookup table.

        Parameters
        ----------
        levels : tuple of float or None
            ``(min, max)`` levels, or None for the range of the finite values.
        """
        self.levels = levels
        self._levelsFromData = False
        self._invalidate()
        self.update()

    def getLevels(self) -> tuple[float, float]:
        """
        Return the levels, computing them from the finite values if they are not set.

        Returns
        -------
        tuple of float
            ``(min, max)`` levels.
        """
        if self.levels is None:
            z = self.data[2]
            mn, mx = z.min(), z.max()
            if not (np.isfinite(mn) and np.isfinite(mx)):
                z = z[np.isfinite(z)]
                mn, mx = z.min(), z.max()
            self.levels = mn, mx
            self._levelsFromData = True
        return self.levels

    def _colorIndices(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the lookup table index of every value.

        Returns
        -------
        index : np.ndarray
            Unsigned integer array of the shape of the values, with the lookup table
            index of every value, and the largest value of its dtype (at least the
            number of colors) for NaN values.
        colors : np.ndarray
            uint32 array of the lookup table colors, as ``QColor.rgba()`` values.
        """
        x, y, z = self.data

        # get colormap
        if callable(self.lut):
            lut = self.lut(z)
        else:
            lut = self.lut

        if lut is None:
            # lut can be None for a few reasons:
            # 1) self.lut(z) can also return None on error
            # 2) if a trivial gradient is being used, HistogramLUTItem calls
            #    setLookupTable(None) as an optimization for ImageItem
            cmap = ColorMap(None, [0.0, 1.0])
            lut = cmap.getLookupTable(nPts=256)

        # normalize and quantize
        mn, mx = self.getLevels()
        rng = mx - mn
        if rng == 0:
            rng = 1
        scale = len(lut) / rng
        index = fn.rescaleData(
            z, scale, mn, dtype=np.min_scalar_type(len(lut)), clip=(0, len(lut) - 1)
        )
        if np.isnan(z.min()):
            # all bits set at NaN values, without a slow masked assignment
            nan = np.isnan(z).astype(index.dtype)
            np.negative(nan, out=nan)
            np.bitwise_or(index, nan, out=index)
        return index, _lutToArgb(lut)

    def generatePicture(self):
        """
        Compute the colors of the cells.

        This is done automatically when the item is painted after a change of the data,
        levels or lookup table.
        """
        index, colors = self._colorIndices()
        colors = _premultipliedArgb(colors)
        # one row per y value, as the pixels of an image
        index = np.ascontiguousarray(index.T)
        if index.dtype.itemsize <= 2:
            # transparent beyond the colors, NaN values included; no bound checks
            table = np.zeros(1 << (8 * index.dtype.itemsize), dtype=np.uint32)
            table[:len(colors)] = colors
        else:
            table = np.append(colors, np.uint32(0))
            index = np.minimum(index, len(colors), out=index)
        self._cellColors = table[index]
        self._rendered = None

    def _generateVectorPicture(self) -> QtGui.QPicture:
        """
        Draw the cells as rectangles grouped by color.

        Returns
        -------
        QtGui.QPicture
            The picture of the cells, without border.
        """
        index, colors = self._colorIndices()

        # nans positions have an invalid lut index
        invalid_coloridx = len(colors)
        Z = np.minimum(index, invalid_coloridx).astype(np.intp)

        X, W, Y, H = self._cellRects()
        X, Y = np.meshgrid(X, Y, indexing='ij')
        W, H = np.meshgrid(W, H, indexing='ij')

        # pre-allocate to the largest array needed
        color_indices, counts = np.unique(Z, return_counts=True)
        rectarray = Qt.internals.PrimitiveArray(QtCore.QRectF, 4)
        rectarray.resize(counts.max())

        # sorted_indices effectively groups together the
        # (flattened) indices of the same coloridx together.
        sorted_indices = np.argsort(Z, axis=None)
        X = X.ravel()
        Y = Y.ravel()
        W = W.ravel()
        H = H.ravel()

        picture = QtGui.QPicture()
        painter = QtGui.QPainter(picture)
        painter.setPen(fn.mkPen(None))

        # draw the tiles grouped by coloridx
        offset = 0
        for coloridx, cnt in zip(color_indices, counts):
            if coloridx == invalid_coloridx:
                continue
            indices = sorted_indices[offset:offset+cnt]
            offset += cnt
            rectarray.resize(cnt)
            memory = rectarray.ndarray()
            memory[:,0] = X[indices]
            memory[:,1] = Y[indices]
            memory[:,2] = W[indices]
            memory[:,3] = H[indices]

            painter.setBrush(QtGui.QBrush(QtGui.QColor.fromRgba(int(colors[coloridx]))))
            painter.drawRects(*rectarray.drawargs())

        painter.end()
        return picture

    def _cellRects(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the positions and sizes of the cells.

        Returns
        -------
        x, width, y, height : np.ndarray
            Low edge and extent of the cells along x (N values) and y (M values).
        """
        x, y, _ = self.data

        # pad x and y so that we don't need to special-case the edges
        x = np.pad(x, 1, mode='edge')
        y = np.pad(y, 1, mode='edge')

        x = (x[:-1] + x[1:]) / 2
        y = (y[:-1] + y[1:]) / 2
        return x[:-1], np.diff(x), y[:-1], np.diff(y)

    def _renderVisibleCells(self, tr: QtGui.QTransform, left: int, right: int, top: int,
                            bottom: int) -> QtGui.QImage:
        """
        Render the cells covering a rectangle of device pixels.

        Parameters
        ----------
        tr : QtGui.QTransform
            Axis-aligned transform from item to device coordinates.
        left, right, top, bottom : int
            Device pixel columns ``left .. right - 1`` and rows ``top .. bottom - 1``.

        Returns
        -------
        QtGui.QImage
            Premultiplied ARGB32 image of the pixels.
        """
        X, W, Y, H = self._cellRects()
        columns = _pixelCells(X, W, tr.m11(), tr.dx(), left, right)
        rows = _pixelCells(Y, H, tr.m22(), tr.dy(), top, bottom)
        cells = self._cellColors
        ny, nx = cells.shape
        outsideColumns = columns == nx
        outsideRows = rows == ny
        columns[outsideColumns] = 0
        rows[outsideRows] = 0
        # gather along the axis of fewer elements first
        if ny < len(rows):
            pixels = np.take(np.take(cells, columns, axis=1), rows, axis=0)
        else:
            pixels = np.take(np.take(cells, rows, axis=0), columns, axis=1)
        # pixels outside of the cells are transparent
        if outsideColumns.any():
            pixels[:, outsideColumns] = 0
        if outsideRows.any():
            pixels[outsideRows] = 0
        return fn.ndarray_to_qimage(pixels, QtGui.QImage.Format.Format_ARGB32_Premultiplied)

    def _paintVisibleCells(self, p: QtGui.QPainter, tr: QtGui.QTransform):
        """
        Draw the visible cells as one image aligned with the device pixels.

        Parameters
        ----------
        p : QtGui.QPainter
            Painter of a pixel-based device. Its transform is reset.
        tr : QtGui.QTransform
            Axis-aligned transform from item to device coordinates.
        """
        region = tr.mapRect(self.boundingRect())
        if p.hasClipping():
            region = region.intersected(tr.mapRect(p.clipBoundingRect()))
        device = p.device()
        dpr = device.devicePixelRatioF()
        # in device pixels for widgets; may be larger than needed for images
        region = region.intersected(
            QtCore.QRectF(0, 0, device.width() * dpr, device.height() * dpr))
        left, top = int(np.floor(region.left())), int(np.floor(region.top()))
        right, bottom = int(np.ceil(region.right())), int(np.ceil(region.bottom()))
        if right <= left or bottom <= top:
            return

        key = (tr.m11(), tr.dx(), tr.m22(), tr.dy(), left, right, top, bottom)
        if self._rendered is None or self._rendered[0] != key:
            self._rendered = (key, self._renderVisibleCells(tr, left, right, top, bottom))
        image = self._rendered[1]

        # draw the image pixels onto the device pixels
        p.resetTransform()
        toLogical = p.deviceTransform().inverted()[0]
        p.drawImage(toLogical.mapRect(QtCore.QRectF(left, top, right - left, bottom - top)),
                    image)

    def paint(self, p: QtGui.QPainter, *args) -> None:
        """
        Draw the cells, as one image on pixel-based devices, and the border.

        Parameters
        ----------
        p : QtGui.QPainter
            Painter, with the item coordinate system.
        *args
            Style option and widget, unused.
        """
        if self._cellColors is None:
            self.generatePicture()
        p.save()
        tr = p.deviceTransform()
        if p.paintEngine().type() in _PIXEL_ENGINES and tr.type() in _RECTILINEAR:
            self._paintVisibleCells(p, tr)
        else:
            if self._picture is None:
                self._picture = self._generateVectorPicture()
            p.drawPicture(0, 0, self._picture)
        p.restore()

        if self.border is not None:
            p.save()
            p.setPen(self.border)
            p.setBrush(fn.mkBrush(None))
            p.drawRect(self.boundingRect())
            p.restore()

    def boundingRect(self) -> QtCore.QRectF:
        """
        Return the rectangle spanned by the sample points.

        Returns
        -------
        QtCore.QRectF
            Rectangle from ``(x[0], y[0])`` to ``(x[-1], y[-1])``.
        """
        x, y, _ = self.data
        return QtCore.QRectF(x[0], y[0], x[-1]-x[0], y[-1]-y[0])
