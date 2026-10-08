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



# --------------------------------------------------------------------------------------
# T2.12: NaN pixels through a transparent color index
# --------------------------------------------------------------------------------------

def _float_image_with_nans(shape=(64, 48), seed=0) -> np.ndarray:
    """
    Return the image of :func:`_float_image` with NaNs outside of its value ramp.

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
    data = _float_image(shape, seed)
    data.flat[1100::7] = np.nan
    return data


def _nan_reference(data: np.ndarray, lut: np.ndarray | None, levels=(0.0, 1.0)) -> QtGui.QImage:
    """
    Render a float image with NaNs through the RGBA path with transparent locations.

    Parameters
    ----------
    data : np.ndarray
        Image data with NaNs.
    lut : np.ndarray or None
        Lookup table.
    levels : tuple of float, default (0.0, 1.0)
        Levels.

    Returns
    -------
    QtGui.QImage
        The reference RGBA image.
    """
    reference = functions_qimage.try_make_qimage(
        data, levels=np.array(levels), lut=lut, transparentLocations=np.isnan(data).nonzero()
    )
    assert reference.format() == Format.Format_RGBA8888
    return reference


def _assert_nan_indexed(qimage: QtGui.QImage, data: np.ndarray) -> None:
    """
    Check that an image is Indexed8, with the NaN pixels at the transparent index 255.

    Parameters
    ----------
    qimage : QtGui.QImage
        Rendered image of `data`.
    data : np.ndarray
        Image data with NaNs, in display order.
    """
    assert qimage.format() == Format.Format_Indexed8
    table = qimage.colorTable()
    assert len(table) == 256
    assert QtGui.qAlpha(table[255]) == 0
    np.testing.assert_array_equal(fn.ndarray_from_qimage(qimage) == 255, np.isnan(data))


def _assert_same_display(qimage: QtGui.QImage, reference: QtGui.QImage, max_diff: int) -> None:
    """
    Check that two images are displayed alike.

    Transparent pixels of `reference` must be transparent in `qimage`; the RGBA values
    of the other pixels may differ by `max_diff`.

    Parameters
    ----------
    qimage : QtGui.QImage
        Tested image.
    reference : QtGui.QImage
        Reference image.
    max_diff : int
        Largest allowed difference per channel.
    """
    tested, expected = _rgba(qimage), _rgba(reference)
    transparent = expected[..., 3] == 0
    assert (tested[transparent, 3] == 0).all()
    assert np.abs(tested[~transparent] - expected[~transparent]).max() <= max_diff


@pytest.mark.parametrize('lut', [
    pytest.param(pg.colormap.get('viridis').getLookupTable(nPts=256), id='viridis256'),
    pytest.param(pg.colormap.get('viridis').getLookupTable(nPts=512), id='viridis512'),
    pytest.param(pg.colormap.get('inferno').getLookupTable(nPts=256), id='inferno256'),
    pytest.param(pg.colormap.get('viridis').getLookupTable(nPts=256, alpha=True)[:, ::-1].copy(),
                 id='rgba256'),
    pytest.param(np.arange(256, dtype=np.uint8), id='mono256'),
    pytest.param(None, id='none'),
])
def test_nan_image_uses_transparent_index(lut):
    data = _float_image_with_nans()
    qimage = _rendered(data, lut)
    _assert_nan_indexed(qimage, data)
    _assert_same_display(qimage, _nan_reference(data, lut), 1)


def test_nan_image_with_small_lut_is_exact():
    data = _float_image_with_nans()
    lut = np.random.default_rng(2).integers(256, size=(200, 3), dtype=np.uint8)
    qimage = _rendered(data, lut)
    _assert_nan_indexed(qimage, data)
    _assert_same_display(qimage, _nan_reference(data, lut), 0)


def _alternating_lut(seed: int) -> np.ndarray:
    """
    Return a 256-entry table in which all adjacent entries differ by about 100 levels.

    Parameters
    ----------
    seed : int
        Seed of the random generator.

    Returns
    -------
    np.ndarray
        Table of shape (256, 3) and dtype uint8.
    """
    rng = np.random.default_rng(seed)
    lut = np.empty((256, 3), dtype=np.uint8)
    lut[0::2] = rng.integers(0, 100, size=(128, 3))
    lut[1::2] = rng.integers(150, 256, size=(128, 3))
    return lut


@pytest.mark.parametrize('pair', [0, 100, 253, 254])
def test_nan_index_merges_most_similar_pair(pair):
    lut = _alternating_lut(3)
    lut[pair + 1] = lut[pair]
    table, merged = functions_qimage._nan_index_lut(np, lut)
    assert merged == pair
    assert table.shape == (255, 3)
    data = _float_image_with_nans()
    qimage = _rendered(data, lut)
    _assert_nan_indexed(qimage, data)
    _assert_same_display(qimage, _nan_reference(data, lut), 0)


def test_nan_image_with_sharp_lut_falls_back_to_rgba():
    lut = _alternating_lut(4)
    assert functions_qimage._nan_index_lut(np, lut) is None
    data = _float_image_with_nans()
    qimage = _rendered(data, lut)
    assert qimage.format() == Format.Format_RGBA8888
    _assert_same_display(qimage, _nan_reference(data, lut), 0)


def test_nan_index_unsupported_levels():
    data = _float_image_with_nans()
    table, merged = functions_qimage._nan_index_lut(np, None)
    mask = functions_qimage._nan_index_mask(np, data)
    for levels in (None, np.array([[0.0, 1.0]])):
        assert functions_qimage.try_make_qimage_with_nan_index(
            data, levels=levels, lut=table, merged=merged, nanMask=mask) is None


def test_nan_mask_is_cached_and_reset():
    data = _float_image_with_nans()
    lut = pg.colormap.get('viridis').getLookupTable(nPts=256)
    item = pg.ImageItem(axisOrder='row-major')
    with count_calls(functions_qimage, '_nan_index_mask') as masks, \
            count_calls(functions_qimage, '_nan_index_lut') as luts:
        item.setImage(data, levels=(0, 1), lut=lut)
        item.render()
        # new levels: neither the mask nor the table is computed again
        item.setLevels((0.2, 0.8))
        item.render()
        assert (masks.count, luts.count) == (1, 1)
        # the locations of the NaNs are not computed on this path
        assert item._imageNanLocations is None

        # new data: new mask
        data2 = data.copy()
        data2[:5] = np.nan
        item.setImage(data2)
        item.render()
        assert masks.count == 2
        _assert_nan_indexed(item.qimage, data2)

        # new axis order: new mask, NaNs at the transposed locations
        item.setOpts(axisOrder='col-major')
        item.updateImage()
        item.render()
        assert masks.count == 3
        _assert_nan_indexed(item.qimage, data2.T)


def test_nan_image_with_downsampling():
    data = _float_image_with_nans((400, 240))
    item = pg.ImageItem(axisOrder='row-major', autoDownsample=True)
    view = pg.GraphicsView()
    viewbox = pg.ViewBox()
    view.setCentralWidget(viewbox)
    view.resize(80, 80)
    viewbox.addItem(item)
    item.setImage(data, levels=(0, 1), lut=pg.colormap.get('viridis').getLookupTable(nPts=256))
    view.show()
    app.processEvents()
    item.render()
    xds, yds = item._lastDownsample
    assert (xds, yds) != (1, 1)
    expected = fn.downsample(fn.downsample(data, xds, axis=1), yds, axis=0)
    _assert_nan_indexed(item.qimage, expected)
    view.close()
