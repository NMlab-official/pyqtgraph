"""
Performance regression tests for ImageItem rendering.

The tests check deterministic properties (QImage formats, caches, call counts and
pixel values), never durations.
"""
import numpy as np
import pytest

import pyqtgraph as pg
import pyqtgraph.functions as fn
from pyqtgraph import functions_qimage
from pyqtgraph.Qt import QtGui
from tests.perf_helpers import count_calls

app = pg.mkQApp()

Format = QtGui.QImage.Format


def _rgba(qimage: QtGui.QImage) -> np.ndarray:
    """
    Return the non-premultiplied RGBA pixels of a QImage.

    Parameters
    ----------
    qimage : QtGui.QImage
        Image of any format.

    Returns
    -------
    np.ndarray
        Array of shape (height, width, 4) and dtype int16.
    """
    converted = qimage.convertToFormat(Format.Format_RGBA8888)
    return fn.ndarray_from_qimage(converted).astype(np.int16)


def _float_image(shape=(64, 48), seed=0) -> np.ndarray:
    """
    Return a float32 test image covering slightly more than the levels (0, 1).

    Parameters
    ----------
    shape : tuple of int, default (64, 48)
        Image shape.
    seed : int, default 0
        Seed of the random generator.

    Returns
    -------
    np.ndarray
        The image.
    """
    data = np.random.default_rng(seed).uniform(-0.1, 1.1, size=shape).astype(np.float32)
    data.flat[:1024] = np.linspace(0, 1, 1024, dtype=np.float32)  # every LUT entry
    return data


def _rendered(data: np.ndarray, lut: np.ndarray | None, levels=(0.0, 1.0)) -> QtGui.QImage:
    """
    Render an ImageItem and return its QImage.

    Parameters
    ----------
    data : np.ndarray
        Image data, row-major.
    lut : np.ndarray or None
        Lookup table.
    levels : tuple of float, default (0.0, 1.0)
        Levels.

    Returns
    -------
    QtGui.QImage
        The rendered image.
    """
    item = pg.ImageItem(axisOrder='row-major')
    item.setImage(data, levels=levels, lut=lut)
    item.render()
    return item.qimage


# --------------------------------------------------------------------------------------
# T1.7: lookup tables with more than 256 entries
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize('channels', [None, 1, 3, 4])
def test_resample_lut_shapes_and_values(channels):
    smooth = np.linspace(0, 255, 512)
    lut = np.rint(smooth if channels is None
                  else np.repeat(smooth[:, None], channels, axis=1)).astype(np.uint8)
    resampled = functions_qimage._resample_lut(np, lut, 256)
    assert resampled.dtype == np.uint8
    assert resampled.shape == (256,) + lut.shape[1:]
    # 512 -> 256 entries: each entry is the mean of the two entries it replaces
    expected = np.rint((lut[0::2].astype(float) + lut[1::2]) / 2).astype(np.uint8)
    np.testing.assert_array_equal(resampled, expected)


def test_resample_lut_rejects_sharp_tables():
    lut = np.zeros((512, 3), dtype=np.uint8)
    lut[257:] = 255  # transition in the middle of a resampled entry
    assert functions_qimage._resample_lut(np, lut, 256) is None
    lut = np.random.default_rng(0).integers(256, size=(512, 3), dtype=np.uint8)
    assert functions_qimage._resample_lut(np, lut, 256) is None


@pytest.mark.parametrize('name,npts', [('viridis', 512), ('CET-L1', 1000), ('CET-L1', 4096)])
def test_float_image_with_large_lut_uses_indexed8(name, npts):
    data = _float_image()
    lut = pg.colormap.get(name).getLookupTable(nPts=npts)
    qimage = _rendered(data, lut)
    assert qimage.format() == Format.Format_Indexed8
    assert len(qimage.colorTable()) == 256

    # colors at most 1 level away from the exact rendering through the full table
    reference = functions_qimage.try_make_qimage(data, levels=np.array([0.0, 1.0]), lut=lut)
    assert reference.format() == Format.Format_RGBX8888
    diff = np.abs(_rgba(qimage) - _rgba(reference))
    assert diff.max() <= 1


def test_float_image_with_sharp_large_lut_is_exact():
    data = _float_image()
    lut = np.random.default_rng(1).integers(256, size=(512, 3), dtype=np.uint8)
    qimage = _rendered(data, lut)
    assert qimage.format() == Format.Format_RGBX8888
    reference = functions_qimage.try_make_qimage(data, levels=np.array([0.0, 1.0]), lut=lut)
    np.testing.assert_array_equal(_rgba(qimage), _rgba(reference))


def test_resampled_lut_is_cached():
    data = _float_image()
    lut = pg.colormap.get('viridis').getLookupTable(nPts=512)
    item = pg.ImageItem(axisOrder='row-major')
    with count_calls(functions_qimage, '_resample_lut') as resample:
        for _ in range(3):
            item.setImage(data, levels=(0, 1), lut=lut)
            item.render()
        assert resample.count == 1

        # a table modified in place is resampled again
        lut[:] = lut[::-1].copy()
        item.updateImage()
        item.render()
        assert resample.count == 2
        table = item.qimage.colorTable()
        assert QtGui.QColor(table[0]).getRgb()[:3] == tuple(
            np.rint((lut[0].astype(float) + lut[1]) / 2).astype(int))

        # a new table is resampled
        item.setLookupTable(pg.colormap.get('plasma').getLookupTable(nPts=512))
        item.render()
        assert resample.count == 3


def test_large_lut_not_resampled_for_integer_images():
    data = (np.arange(64 * 48).reshape(64, 48) % 256).astype(np.uint8)
    lut = pg.colormap.get('viridis').getLookupTable(nPts=512)
    item = pg.ImageItem(axisOrder='row-major')
    with count_calls(functions_qimage, '_resample_lut') as resample:
        item.setImage(data, levels=(0, 255), lut=lut)
        item.render()
    assert resample.count == 0
    reference = functions_qimage.try_make_qimage(data, levels=np.array([0, 255]), lut=lut)
    np.testing.assert_array_equal(_rgba(item.qimage), _rgba(reference))
