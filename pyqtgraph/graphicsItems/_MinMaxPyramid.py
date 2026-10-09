"""
Min/max pyramid of an array, used by the 'peak' downsampling of PlotDataItem.

This module is private: its interface may change without warning.
"""
import numpy as np

from ..util.numba_helper import getNumbaFunctions

__all__ = ['MinMaxPyramid']


class MinMaxPyramid:
    """
    Maxima and minima of the aligned cells of an array, at all power-of-2 sizes.

    Level ``k`` holds, for each cell ``j`` of ``2**k`` values, the maximum and the
    minimum of the values ``j * 2**k`` to ``(j + 1) * 2**k - 1``. Only the levels
    from :attr:`BASE` up are stored, and only complete cells: about ``len(y) / 16``
    values per extreme in total, i.e. 1/8 of the size of the data for both.

    The extremes of arbitrary blocks of ``ds`` values (:meth:`blocks`) are then
    combined from a few cells per block: computing ``n`` blocks costs ``O(n log ds)``
    instead of ``O(n * ds)``, whatever the number of values they cover. The results
    are identical to ``numpy.max`` and ``numpy.min`` over each block, including the
    propagation of ``NaN``.

    Building the pyramid costs ``O(len(y))``. With the ``useNumba`` configuration
    option, the lowest level is computed by numba. Appended values are added
    incrementally (:meth:`extend`).

    Parameters
    ----------
    y : numpy.ndarray
        One-dimensional numeric data. It must not be modified while the pyramid is
        in use.
    """

    BASE = 5  # log2 of the number of values per cell at the lowest stored level

    def __init__(self, y: np.ndarray) -> None:
        self._y = y
        # per stored level (index 0 is level BASE): buffers whose leading _len
        # entries are the extremes of the complete cells
        self._max: list[np.ndarray] = []
        self._min: list[np.ndarray] = []
        self._len: list[int] = []
        self._update()

    @property
    def y(self) -> np.ndarray:
        """numpy.ndarray: The data."""
        return self._y

    @staticmethod
    def efficientBlockSize() -> int:
        """
        Get the block size above which :meth:`blocks` is faster than a direct scan.

        A query reads about 100 values per block with numpy (2 * 32 at the block ends,
        and a few cells per level), about 50 with the numba kernel of the
        ``useNumba`` configuration option. Scanning the values of a block costs about
        0.35 ns per value (numpy 2.5, measured).

        Returns
        -------
        int
            Number of values per block.
        """
        return 256 if getNumbaFunctions() is not None else 768

    @classmethod
    def minBlockSize(cls) -> int:
        """
        Get the smallest block size served from the pyramid by :meth:`blocks`.

        Returns
        -------
        int
            Blocks of fewer values are cheaper to compute from the data directly.
        """
        return 2 << cls.BASE

    def extend(self, y: np.ndarray) -> None:
        """
        Replace the data by a longer array starting with the same values.

        Only the cells completed by the new values are computed.

        Parameters
        ----------
        y : numpy.ndarray
            The extended data. Its leading values must be equal to the current data.
        """
        self._y = y
        self._update()

    def levelExtremes(self, level: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the extremes of the complete cells of a level.

        Parameters
        ----------
        level : int
            Level, at least :attr:`BASE`. Cells hold ``2**level`` values.

        Returns
        -------
        maxima, minima : numpy.ndarray
            Extremes of the cells, as read-only views.
        """
        i = level - self.BASE
        if i < 0:
            raise ValueError(f'levels below {self.BASE} are not stored')
        if i >= len(self._len):
            empty = np.empty(0, dtype=self._y.dtype)
            return empty, empty
        mx = self._max[i][:self._len[i]]
        mn = self._min[i][:self._len[i]]
        mx.flags.writeable = False
        mn.flags.writeable = False
        return mx, mn

    def _update(self) -> None:
        """
        Compute the cells completed since the last update, at every level.
        """
        y = self._y
        size = len(y) >> self.BASE
        i = 0
        while size > 0:
            if i == len(self._len):
                self._max.append(np.empty(0, dtype=y.dtype))
                self._min.append(np.empty(0, dtype=y.dtype))
                self._len.append(0)
            start = self._len[i]
            if size > start:
                self._reserve(i, size)
                mx, mn = self._max[i], self._min[i]
                if i == 0:
                    self._computeBase(start, size)
                else:
                    below_max, below_min = self._max[i - 1], self._min[i - 1]
                    np.maximum(below_max[2 * start:2 * size:2],
                               below_max[2 * start + 1:2 * size:2], out=mx[start:size])
                    np.minimum(below_min[2 * start:2 * size:2],
                               below_min[2 * start + 1:2 * size:2], out=mn[start:size])
                self._len[i] = size
            size >>= 1
            i += 1

    def _reserve(self, i: int, size: int) -> None:
        """
        Make the buffers of a stored level hold at least `size` cells.

        Parameters
        ----------
        i : int
            Index of the stored level (0 for level :attr:`BASE`).
        size : int
            Required number of cells.
        """
        if size <= len(self._max[i]):
            return
        # grow geometrically, so that streaming appends are O(1) amortized
        capacity = size if self._len[i] == 0 else max(size, 2 * len(self._max[i]))
        for buffers in (self._max, self._min):
            grown = np.empty(capacity, dtype=self._y.dtype)
            grown[:self._len[i]] = buffers[i][:self._len[i]]
            buffers[i] = grown

    def _computeBase(self, start: int, end: int) -> None:
        """
        Compute the extremes of the cells of the lowest stored level from the data.

        Parameters
        ----------
        start, end : int
            Range of cells to compute, ``end`` excluded.
        """
        width = 1 << self.BASE
        values = self._y[start * width:end * width]
        out_max = self._max[0][start:end]
        out_min = self._min[0][start:end]
        fn_numba = getNumbaFunctions()
        if fn_numba is not None and values.dtype.kind in 'fiu':
            fn_numba.minmax_cells(values, width, out_max, out_min)
            return
        cells = values.reshape(end - start, width)
        np.max(cells, axis=1, out=out_max)
        np.min(cells, axis=1, out=out_min)

    def blocks(
        self,
        first: int,
        end: int,
        ds: int,
        out_max: np.ndarray,
        out_min: np.ndarray
    ) -> None:
        """
        Compute the extremes of a range of blocks of ``ds`` values.

        Block ``b`` covers the values ``b * ds`` to ``(b + 1) * ds - 1``. A block is
        covered by cells it fully contains, whose extremes are combined:

        * the complete cells of the level with at most ``ds / 2`` values per cell,
          read at once for all blocks;
        * where the block starts within a cell of that level, one cell per lower
          stored level, together covering the values from the start of the block to
          the next cell boundary of that level (and likewise at its end);
        * the first and the last ``2**BASE`` values of the block, read from the data.

        The covering cells may overlap, which does not change the extremes. Computing
        ``n`` blocks thus costs ``O(n log ds)`` operations, whatever the number of
        values they cover.

        Parameters
        ----------
        first : int
            First block.
        end : int
            End of the range of blocks, excluded. ``end * ds`` must not exceed the
            length of the data.
        ds : int
            Number of values per block, at least :meth:`minBlockSize`.
        out_max, out_min : numpy.ndarray
            Arrays of ``end - first`` values receiving the maxima and the minima of
            the blocks.
        """
        num = end - first
        if num <= 0:
            return
        if ds < self.minBlockSize():
            raise ValueError(f'blocks of {ds} values are not served from the pyramid')
        y = self._y
        if end * ds > len(y):
            raise ValueError('blocks extend beyond the data')
        base = self.BASE
        # level of the cells fully contained in the blocks: at most ds / 2 values
        top = min(ds.bit_length() - 2, base + len(self._len) - 1)
        cell = 1 << top
        starts = np.arange(first, end, dtype=np.int64) * ds
        stops = starts + ds

        # complete cells of the top level inside each block. Consecutive ranges are
        # separated by one straddling cell at most, which reduceat also reduces: only
        # the even segments are kept. The last range ends at the end of the slice.
        top_max, top_min = self.levelExtremes(top)
        bounds = np.empty(2 * num - 1, dtype=np.int64)
        bounds[0::2] = -(-starts // cell)
        bounds[1::2] = stops[:-1] // cell
        stop = int(stops[-1] // cell)
        np.copyto(out_max, np.maximum.reduceat(top_max[:stop], bounds)[0::2])
        np.copyto(out_min, np.minimum.reduceat(top_min[:stop], bounds)[0::2])

        # Where the block starts within a top-level cell, the values up to the end of
        # that cell are covered by the cells (start >> level) + 1 of the lower levels
        # whose bit `level` of start is 0, and by the first values of the block. The
        # cells of the other levels lie within the block as well: all are taken, which
        # avoids masking. Likewise, the cells (stop >> level) - 1 cover the values
        # from the last top-level cell boundary to the end of the block.
        for level in range(base, top):
            mx, mn = self.levelExtremes(level)
            for index in ((starts >> level) + 1, (stops >> level) - 1):
                np.maximum(out_max, mx[index], out=out_max)
                np.minimum(out_min, mn[index], out=out_min)

        width = 1 << base
        fn_numba = getNumbaFunctions()
        if fn_numba is not None and y.dtype.kind in 'fiu':
            # values from the start of each block to the next multiple of 2**BASE, and
            # from the last one to its end
            fn_numba.peak_block_ends(y, first, ds, width, out_max, out_min)
            return
        # first and last 2**BASE values of each block: strided views of the data
        stride = y.strides[0]
        for offset in (first * ds, first * ds + ds - width):
            values = np.lib.stride_tricks.as_strided(
                y[offset:], shape=(num, width), strides=(ds * stride, stride),
                writeable=False
            )
            np.maximum(out_max, values.max(axis=1), out=out_max)
            np.minimum(out_min, values.min(axis=1), out=out_min)
