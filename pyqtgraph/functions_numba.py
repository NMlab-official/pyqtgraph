import numba
import numpy as np

@numba.jit(nopython=True)
def rescale_and_lookup(data, scale, offset, lut):
    # data should be floating point and 2d
    # lut is 1d
    vmin, vmax = 0, lut.shape[0] - 1
    out = np.empty_like(data, dtype=lut.dtype)
    for (x, y) in np.nditer((data, out)):
        val = (x - offset) * scale
        val = min(max(val, vmin), vmax)
        y[...] = lut[int(val)]
    return out

@numba.jit(nopython=True)
def rescale_and_clip(data, scale, offset, vmin, vmax):
    # vmin and vmax <= 255
    out = np.empty_like(data, dtype=np.uint8)
    for (x, y) in np.nditer((data, out)):
        val = (x - offset) * scale
        val = min(max(val, vmin), vmax)
        y[...] = val
    return out

@numba.jit(nopython=True)
def numba_take(lut, data):
    # numba supports only the 1st two arguments of np.take
    return np.take(lut, data)

@numba.jit(nopython=True, nogil=True)
def _minmax_nan(data, start, stop):
    # maximum and minimum of data[start:stop], NaN propagating as with np.max
    hi = data[start]
    lo = hi
    for i in range(start + 1, stop):
        v = data[i]
        if v > hi or v != v:
            hi = v
        if v < lo or v != v:
            lo = v
    return hi, lo

@numba.jit(nopython=True, nogil=True)
def minmax_cells(data, width, out_max, out_min):
    # maximum and minimum of each cell of `width` consecutive values, NaN propagating
    # as with np.max and np.min
    for j in range(out_max.shape[0]):
        start = j * width
        hi = data[start]
        lo = hi
        # v - v is NaN for NaN and infinite values only: such cells are reduced again
        # with the NaN-propagating comparisons, which are slower
        check = hi - hi
        for i in range(start + 1, start + width):
            v = data[i]
            if v > hi:
                hi = v
            if v < lo:
                lo = v
            check += v - v
        if check != check:
            hi, lo = _minmax_nan(data, start, start + width)
        out_max[j] = hi
        out_min[j] = lo

@numba.jit(nopython=True, nogil=True)
def peak_block_ends(data, first, ds, width, out_max, out_min):
    # Combine into out_max and out_min the extremes of the values of each block of ds
    # values (block first + b for out[b]) from its start to the next multiple of width,
    # and from the last multiple of width to its end. NaN propagates as with np.max.
    for b in range(out_max.shape[0]):
        start = (first + b) * ds
        stop = start + ds
        hi = out_max[b]
        lo = out_min[b]
        nan_found = False
        nan_value = hi
        head = (start // width + 1) * width
        for i in range(start, head):
            v = data[i]
            if v > hi:
                hi = v
            if v < lo:
                lo = v
            if v != v:
                nan_found = True
                nan_value = v
        for i in range(stop // width * width, stop):
            v = data[i]
            if v > hi:
                hi = v
            if v < lo:
                lo = v
            if v != v:
                nan_found = True
                nan_value = v
        if nan_found:
            hi = nan_value
            lo = nan_value
        out_max[b] = hi
        out_min[b] = lo
