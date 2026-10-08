from types import ModuleType

import numpy

from .Qt import QtGui
from . import functions
from .util.cupy_helper import getCupy
from .util.numba_helper import getNumbaFunctions


def _apply_lut_for_uint(xp, image, lut):
    # Note: compared to makeARGB(), we have already clipped the data to range

    # if lut is 1d, then lut[image] is fastest
    # if lut is 2d, then lut.take(image, axis=0) is faster than lut[image]
    lut = _convert_2dlut_to_1dlut(xp, lut)

    if xp == numpy and (fn_numba := getNumbaFunctions()) is not None:
        # numba "take" supports only the 1st 2 arguments of np.take,
        # therefore we have to convert the lut to 1d.
        # "take" will output a c contiguous array regardless of its input.
        image = fn_numba.numba_take(lut, image)
    else:
        # advanced indexing is memory order aware.
        # its output can be either C or F contiguous.
        image = lut[image]

    if image.dtype == xp.uint32:
        # "view" requires c contiguous for numpy < 1.23
        image = xp.ascontiguousarray(image)
        image = image[..., xp.newaxis].view(xp.uint8)

    return image


def _convert_lut_to_rgba(xp, lut):
    # converts:
    #   - None to (256, 4)
    #   - uint8 (N,) to uint8 (N, 4)
    #   - uint8 (N, 1) to uint8 (N, 4)
    #   - uint8 (N, 3) to uint8 (N, 4)

    if not (
        lut is None
        or lut.ndim == 1
        or (
            lut.ndim == 2
            and lut.shape[1] in (1, 3, 4)
        )
    ):
        raise ValueError("unsupported lut shape")

    N = lut.shape[0] if lut is not None else 256

    if lut is None:
        lut = xp.arange(N, dtype=xp.uint8)

    # convert (N,) to (N, 1)
    if lut.ndim == 1:
        lut = lut[:, xp.newaxis]

    if lut.shape[1] == 4:
        return lut

    out = xp.full((N, 4), 255, dtype=xp.uint8)
    out[:, 0:3] = lut
    return out


def _convert_2dlut_to_1dlut(xp, lut):
    # converts:
    #   - uint8 (N, 1) to uint8 (N,)
    #   - uint8 (N, 3) or (N, 4) to uint32 (N,)
    # this allows faster lookup as 1d lookup is faster

    if lut.ndim == 1:
        return lut

    if lut.shape[1] == 3:  # rgb
        # convert rgb lut to rgba so that it is 32-bits
        lut = xp.column_stack([lut, xp.full(lut.shape[0], 255, dtype=xp.uint8)])
    if lut.shape[1] == 4:  # rgba
        lut = lut.view(xp.uint32)
    lut = lut.ravel()

    return lut


def _resample_lut(
    xp: ModuleType, lut: numpy.ndarray, size: int, max_deviation: int = 1
) -> numpy.ndarray | None:
    """
    Resample a lookup table to fewer entries, if this barely changes the displayed colors.

    A value normalized to ``u`` in [0, 1) by the levels is displayed with entry
    ``floor(u * N)`` of a table of N entries. Entry ``j`` of the resampled table covers
    the values ``j / size <= u < (j + 1) / size``; it is the original table linearly
    interpolated over the indices at the center of that interval. The deviation is the
    largest difference, over all values and channels, between the color of a value
    through the resampled table and its color through ``lut``.

    Parameters
    ----------
    xp : module
        Array module of ``lut``, either numpy or cupy.
    lut : numpy.ndarray or cupy.ndarray
        Lookup table of dtype uint8 and shape (N,) or (N, C), with ``N >= size``.
    size : int
        Number of entries of the resampled table.
    max_deviation : int, default 1
        Largest accepted deviation, in levels of 255.

    Returns
    -------
    numpy.ndarray or cupy.ndarray or None
        Resampled uint8 table of shape ``(size,) + lut.shape[1:]`` on the array module of
        ``lut``, or None if its deviation exceeds ``max_deviation``.
    """
    if xp != numpy:
        lut = lut.get()  # a few hundred entries: cheaper to resample on the host
    n = lut.shape[0]
    if n < size:
        raise ValueError(f"cannot resample a lookup table of {n} entries to {size} entries")
    table = lut.reshape(n, -1).astype(numpy.float64)

    pos = (numpy.arange(size) + 0.5) * (n / size) - 0.5
    numpy.clip(pos, 0, n - 1, out=pos)
    lo = pos.astype(numpy.intp)  # pos >= 0, so truncation is floor
    hi = numpy.minimum(lo + 1, n - 1)
    frac = (pos - lo)[:, numpy.newaxis]
    resampled = numpy.rint(table[lo] * (1.0 - frac) + table[hi] * frac).astype(numpy.uint8)

    # original entry k covers k / n <= u < (k + 1) / n, i.e. the resampled entries
    # first[k] to last[k] (at most two of them, since n >= size)
    k = numpy.arange(n)
    first = (k * size) // n
    last = -((-(k + 1) * size) // n) - 1
    deviation = max(
        numpy.abs(table - resampled[first]).max(),
        numpy.abs(table - resampled[last]).max(),
    )
    if deviation > max_deviation:
        return None
    return xp.asarray(resampled.reshape((size,) + lut.shape[1:]))


# index of the transparent color of Indexed8 images of float mono images with NaNs
_NAN_INDEX = 255


def _nan_index_lut(
    xp: ModuleType, lut: numpy.ndarray | None, max_deviation: int = 1
) -> tuple[numpy.ndarray, int | None] | None:
    """
    Make a lookup table of at most 255 entries, leaving index 255 free for NaN pixels.

    A table of N <= 255 entries is used as is: a value normalized to ``u`` in [0, 1) by
    the levels is displayed with entry ``floor(u * N)``. Otherwise, the value falls in
    bin ``j = floor(u * 256)`` of the 256-entry table (``lut`` itself or its
    resampled equivalent, see :func:`_resample_lut`), and two adjacent bins ``m`` and
    ``m + 1`` share one entry: bin ``j`` is displayed with entry ``j - (j > m)``. The
    pair whose merge changes the colors least is chosen, preferring ``m = 254``, which
    costs nothing at render time. Every other color is unchanged.

    Parameters
    ----------
    xp : module
        Array module of ``lut``, either numpy or cupy.
    lut : numpy.ndarray or cupy.ndarray or None
        Lookup table of dtype uint8 and shape (N,) or (N, C). None stands for the
        256-level grayscale ramp used for images without lookup table.
    max_deviation : int, default 1
        Largest accepted difference, in levels of 255 and per channel, between the
        color of any value and its color through ``lut``.

    Returns
    -------
    tuple of (numpy.ndarray or cupy.ndarray, int or None) or None
        The table of at most 255 entries and the merged bin ``m`` (None if ``lut`` is
        used as is), or None if no table meets ``max_deviation``.
    """
    if lut is None:
        lut = xp.arange(256, dtype=xp.uint8)
    n = lut.shape[0]
    if n <= _NAN_INDEX:
        return lut, None

    if n == 256:
        base = lut
    else:
        base = _resample_lut(xp, lut, 256, max_deviation)
        if base is None:
            return None
    if xp != numpy:
        lut, base = lut.get(), base.get()
    table = lut.reshape(n, -1).astype(numpy.int16)
    base = base.reshape(256, -1)

    # range of the original entries covered by each bin: entries start[j] to end[j]
    j = numpy.arange(256)
    start = (j * n) // 256
    end = -((-(j + 1) * n) // 256) - 1
    lo = numpy.minimum(numpy.minimum.reduceat(table, start, axis=0), table[end])
    hi = numpy.maximum(numpy.maximum.reduceat(table, start, axis=0), table[end])

    # merging bins m and m + 1: the middle of their color range minimizes the deviation
    pair_lo = numpy.minimum(lo[:-1], lo[1:])
    pair_hi = numpy.maximum(hi[:-1], hi[1:])
    merged_colors = numpy.rint((pair_lo + pair_hi) / 2)
    deviation = numpy.maximum(pair_hi - merged_colors, merged_colors - pair_lo).max(axis=1)
    m = _NAN_INDEX - 1
    if deviation[m] > max_deviation:
        m = int(numpy.argmin(deviation))
        if deviation[m] > max_deviation:
            return None
    merged = numpy.concatenate(
        [base[:m], merged_colors[m:m + 1].astype(numpy.uint8), base[m + 2:]]
    )
    return xp.asarray(merged.reshape((_NAN_INDEX,) + lut.shape[1:])), m


def _nan_index_mask(xp: ModuleType, image: numpy.ndarray) -> numpy.ndarray:
    """
    Return the mask of the NaN pixels of a mono image, as uint8 values ``_NAN_INDEX``.

    Parameters
    ----------
    xp : module
        Array module of ``image``, either numpy or cupy.
    image : numpy.ndarray or cupy.ndarray
        2-D floating point image.

    Returns
    -------
    numpy.ndarray or cupy.ndarray
        uint8 array of the shape of ``image``: ``_NAN_INDEX`` at NaN pixels, 0 elsewhere.
    """
    mask = xp.isnan(image).view(xp.uint8)
    mask *= _NAN_INDEX
    return mask


def _rescale_float_to_index(
    xp: ModuleType, image: numpy.ndarray, levels: numpy.ndarray, num_colors: int,
    max_index: int,
) -> numpy.ndarray:
    """
    Rescale a floating point image to uint8 color indices.

    Value ``v`` gets index ``floor((v - min) * num_colors / (max - min))`` clipped to
    ``[0, max_index]``, the indexing used by :func:`_rescale_and_lookup_float`.

    Parameters
    ----------
    xp : module
        Array module of ``image``, either numpy or cupy.
    image : numpy.ndarray or cupy.ndarray
        Floating point image.
    levels : numpy.ndarray or cupy.ndarray
        ``[min, max]`` levels.
    num_colors : int
        Number of indices the levels range is divided into.
    max_index : int
        Largest index, at most 255.

    Returns
    -------
    numpy.ndarray or cupy.ndarray
        New uint8 array of indices. NaN pixels get unspecified indices.
    """
    minVal, maxVal = levels
    rng = maxVal - minVal
    rng = 1 if rng == 0 else rng
    scale = num_colors / rng
    if xp == numpy and (fn_numba := getNumbaFunctions()) is not None:
        return fn_numba.rescale_and_clip(image, scale, minVal, 0, max_index)
    return functions.rescaleData(image, scale, minVal, dtype=xp.uint8, clip=(0, max_index))


def try_make_qimage_with_nan_index(
    image: numpy.ndarray, *, levels: numpy.ndarray | None, lut: numpy.ndarray,
    merged: int | None, nanMask: numpy.ndarray,
) -> QtGui.QImage | None:
    """
    Make an Indexed8 QImage of a float mono image, with transparent NaN pixels.

    The NaN pixels get index ``_NAN_INDEX`` (255), which is transparent in the color
    table, so that the image needs no conversion to RGBA nor alpha channel.

    Parameters
    ----------
    image : numpy.ndarray or cupy.ndarray
        2-D floating point image.
    levels : numpy.ndarray or cupy.ndarray or None
        ``[min, max]`` levels.
    lut : numpy.ndarray or cupy.ndarray
        Lookup table of at most 255 entries, as returned by :func:`_nan_index_lut`.
    merged : int or None
        Merged bin returned by :func:`_nan_index_lut` with ``lut``.
    nanMask : numpy.ndarray or cupy.ndarray
        Mask of the NaN pixels, as returned by :func:`_nan_index_mask`.

    Returns
    -------
    QtGui.QImage or None
        The Indexed8 image, or None if ``levels`` is not a single ``[min, max]`` pair.
    """
    cp = getCupy()
    xp = cp.get_array_module(image) if cp else numpy

    if levels is None:
        return None
    levels = xp.asarray(levels)
    if levels.ndim != 1:
        return None
    if lut.shape[0] > _NAN_INDEX:
        raise ValueError("lut must have at most 255 entries")

    if merged is None:
        index = _rescale_float_to_index(xp, image, levels, lut.shape[0], lut.shape[0] - 1)
    elif merged == _NAN_INDEX - 1:
        # merging the last two bins is clipping to the last entry
        index = _rescale_float_to_index(xp, image, levels, 256, _NAN_INDEX - 1)
    else:
        index = _rescale_float_to_index(xp, image, levels, 256, 255)
        # shift the bins above the merged pair down (uint8 operands: fast loop)
        xp.subtract(index, (index > merged).view(xp.uint8), out=index)
    index |= nanMask

    if xp == cp:
        index = index.get()
    index = numpy.ascontiguousarray(index)

    rgba = _convert_lut_to_rgba(xp, lut)
    ctbl = [QtGui.qRgba(*color) for color in rgba.tolist()]
    ctbl += [0] * (_NAN_INDEX + 1 - len(ctbl))  # unused entries, then transparent NaNs
    qimage = functions.ndarray_to_qimage(index, QtGui.QImage.Format.Format_Indexed8)
    qimage.setColorTable(ctbl)
    return qimage


def _rescale_and_lookup_float(xp, image, levels, lut, *, forceApplyLut):
    # It is usually more performant to _not_ apply the lut and
    # instead use it as an Indexed8 ColorTable. This is only
    # applicable if the lut has <= 256 entries.

    if forceApplyLut and lut is None:
        raise ValueError("forceApplyLut True but lut not provided")

    # Decide on maximum scaled value
    if lut is not None:
        num_colors = lut.shape[0]
        max_scale_value = num_colors
    else:
        num_colors = 256
        max_scale_value = 255.0
    dtype = xp.min_scalar_type(num_colors - 1)

    # note: "dtype == uint16" ==> lut provided ==> mono-channel image
    #       i.e. multi-channel image ==> lut is None ==> dtype == uint8
    #
    #       the library defaults to using 256-entry luts, so
    #       "dtype == uint8" is the common case

    apply_lut = forceApplyLut or dtype == xp.uint16

    minVal, maxVal = levels
    rng = maxVal - minVal
    rng = 1 if rng == 0 else rng
    offset = minVal
    scale = max_scale_value / rng

    if xp == numpy and (fn_numba := getNumbaFunctions()) is not None:
        if apply_lut:
            # this path does rescale and apply lut in one step
            lut = _convert_2dlut_to_1dlut(xp, lut)
            image = fn_numba.rescale_and_lookup(image, scale, offset, lut)
            lut = None
            if image.dtype == xp.uint32:
                # "view" requires c contiguous for numpy < 1.23
                image = xp.ascontiguousarray(image)
                image = image[..., xp.newaxis].view(xp.uint8)
        else:
            image = fn_numba.rescale_and_clip(image, scale, offset, 0, num_colors - 1)
    else:
        image = functions.rescaleData(
            image, scale, offset, dtype=dtype, clip=(0, num_colors - 1)
        )
        if apply_lut:
            image = _apply_lut_for_uint(xp, image, lut)
            lut = None

    # image is now of type uint8
    return image, lut


def _combine_levels_and_lut(xp, image, levels, lut):
    if (
        image.dtype == xp.uint16
        and levels is None
        and image.ndim == 3
        and image.shape[2] == 3
    ):
        # uint16 rgb can't be directly displayed, so make it
        # pass through effective lut processing
        levels = [0, 65535]

    if levels is None and lut is None:
        # nothing to combine
        return image, lut

    # distinguish between lut for levels and colors
    levels_lut = None
    colors_lut = lut

    eflsize = 2 ** (image.itemsize * 8)
    if levels is None:
        info = xp.iinfo(image.dtype)
        minlev, maxlev = info.min, info.max
    else:
        minlev, maxlev = levels
    levdiff = maxlev - minlev
    levdiff = 1 if levdiff == 0 else levdiff  # don't allow division by 0
    offset = minlev

    if colors_lut is None:
        scale = 255.0 / levdiff
        if image.dtype == xp.ubyte and image.ndim == 2:
            # uint8 mono image
            ind = xp.arange(eflsize)
            levels_lut = functions.rescaleData(ind, scale, offset, dtype=xp.ubyte)
            # image data is not scaled. instead, levels_lut is used
            # as (grayscale) Indexed8 ColorTable to get the same effect.
            # due to the small size of the input to rescaleData(), we
            # do not bother caching the result
            return image, levels_lut
        else:
            # uint16 mono, uint8 rgb, uint16 rgb
            # rescale image data by computation instead of by memory lookup
            if xp == numpy and (fn_numba := getNumbaFunctions()) is not None:
                image = fn_numba.rescale_and_clip(image, scale, offset, 0, 255)
            else:
                image = functions.rescaleData(image, scale, offset, dtype=xp.ubyte)
            return image, colors_lut
    else:
        num_colors = colors_lut.shape[0]
        scale = num_colors / levdiff
        lutdtype = xp.min_scalar_type(num_colors - 1)

        if image.dtype == xp.ubyte or lutdtype != xp.ubyte:
            # combine if either:
            #   1) uint8 mono image
            #   2) colors_lut has more entries than will fit within 8-bits
            ind = xp.arange(eflsize)
            levels_lut = functions.rescaleData(
                ind, scale, offset, dtype=lutdtype, clip=(0, num_colors - 1),
            )
            efflut = colors_lut[levels_lut]

            # apply the effective lut early for the following types:
            if image.dtype == xp.uint16 and image.ndim == 2:
                image = _apply_lut_for_uint(xp, image, efflut)
                efflut = None
            return image, efflut
        else:
            # uint16 image with colors_lut <= 256 entries
            # don't combine, we will use QImage ColorTable
            if xp == numpy and (fn_numba := getNumbaFunctions()) is not None:
                image = fn_numba.rescale_and_clip(image, scale, offset, 0, num_colors - 1)
            else:
                image = functions.rescaleData(
                    image, scale, offset, dtype=lutdtype, clip=(0, num_colors - 1),
                )
            return image, colors_lut


def try_make_qimage(image, *, levels, lut, transparentLocations=None):
    """
    Internal function to make an QImage from an ndarray without going
    through the full generality of makeARGB().
    Only certain combinations of input arguments are supported.
    """

    # this function assumes that image has no nans.
    # checking for nans is an expensive operation; it is expected that
    # the caller would want to cache the result rather than have this
    # function check for nans unconditionally.

    cp = getCupy()
    xp = cp.get_array_module(image) if cp else numpy

    # float images always need levels
    if image.dtype.kind == "f" and levels is None:
        return None

    if levels is not None:
        levels = xp.asarray(levels)

        # can't handle multi-channel levels
        if levels.ndim != 1:
            return None

        # if levels is provided, multi-channel images must be 3 channels only.
        # (because it doesn't make sense to scale a 4th alpha channel.)
        if image.ndim == 3 and image.shape[2] != 3:
            return None

    if lut is not None and lut.dtype != xp.uint8:
        raise ValueError("lut dtype must be uint8")

    alpha_channel_required = (
        (   # image itself has alpha channel
            image.ndim == 3
            and image.shape[2] == 4
        )
        or
        (    # lut has alpha channel
            lut is not None
            and lut.ndim == 2
            and lut.shape[1] == 4
        )
    )

    if image.dtype.kind == "f":
        if image.ndim == 2:
            # mono float images
            if transparentLocations is None:
                image, lut = _rescale_and_lookup_float(
                    xp, image, levels, lut, forceApplyLut=False
                )
                levels = None
                # on return, we will have an uint8 image.
                # lut if not None will have <= 256 entries
            else:
                # this path creates an alpha channel
                lut = _convert_lut_to_rgba(xp, lut)
                alpha_channel_required = True

                image, lut = _rescale_and_lookup_float(
                    xp, image, levels, lut, forceApplyLut=True
                )
                levels = None
                assert lut is None
                image[..., 3][transparentLocations] = 0
        else:
            # RGB float images
            # lut can only be None for RGB images
            image, lut = _rescale_and_lookup_float(
                xp, image, levels, lut, forceApplyLut=False
            )
            levels = None

            if transparentLocations is not None:
                alpha_channel_required = True
                mask = xp.full(image.shape[:2], 255, dtype=xp.uint8)
                mask[transparentLocations] = 0
                image = xp.dstack((image, mask))

    # if the image data is a small int, then we can combine levels + lut
    # into a single lut for better performance
    elif image.dtype in (xp.ubyte, xp.uint16):
        image, lut = _combine_levels_and_lut(xp, image, levels, lut)
        levels = None

    ubyte_nolvl = image.dtype == xp.ubyte and levels is None
    is_passthru8 = ubyte_nolvl and lut is None
    is_indexed8 = (
        ubyte_nolvl and image.ndim == 2 and lut is not None and lut.shape[0] <= 256
    )
    is_passthru16 = image.dtype == xp.uint16 and levels is None and lut is None
    can_grayscale16 = is_passthru16 and image.ndim == 2
    is_rgba64 = is_passthru16 and image.ndim == 3 and image.shape[2] == 4

    # bypass makeARGB for supported combinations
    supported = is_passthru8 or is_indexed8 or can_grayscale16 or is_rgba64
    if not supported:
        return None

    if xp == cp:
        image = image.get()

    # worthwhile supporting non-contiguous arrays
    image = numpy.ascontiguousarray(image)

    fmt = None
    ctbl = None
    if is_passthru8:
        # both levels and lut are None
        # these images are suitable for display directly
        if image.ndim == 2:
            fmt = QtGui.QImage.Format.Format_Grayscale8
        elif image.shape[2] == 3:
            fmt = QtGui.QImage.Format.Format_RGB888
        elif image.shape[2] == 4:
            if alpha_channel_required:
                fmt = QtGui.QImage.Format.Format_RGBA8888
            else:
                fmt = QtGui.QImage.Format.Format_RGBX8888
    elif is_indexed8:
        # levels and/or lut --> lut-only
        fmt = QtGui.QImage.Format.Format_Indexed8
        if lut.ndim == 1 or lut.shape[1] == 1:
            ctbl = [QtGui.qRgb(x, x, x) for x in lut.ravel().tolist()]
        elif lut.shape[1] == 3:
            ctbl = [QtGui.qRgb(*rgb) for rgb in lut.tolist()]
        elif lut.shape[1] == 4:
            ctbl = [QtGui.qRgba(*rgba) for rgba in lut.tolist()]
    elif can_grayscale16:
        # single channel uint16
        # both levels and lut are None
        fmt = QtGui.QImage.Format.Format_Grayscale16
    elif is_rgba64:
        # uint16 rgba
        # both levels and lut are None
        fmt = QtGui.QImage.Format.Format_RGBA64  # endian-independent
    if fmt is None:
        raise ValueError("unsupported image type")
    qimage = functions.ndarray_to_qimage(image, fmt)
    if ctbl is not None:
        qimage.setColorTable(ctbl)
    return qimage
