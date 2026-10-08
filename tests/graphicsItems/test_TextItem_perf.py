"""
Performance regression tests for TextItem (T2.11 of PERFORMANCE_PLAN.md).

A TextItem placed in a ViewBox updates its anti-scaling transform from
``viewTransformChanged`` only, instead of connecting one slot per item to the
``sigPrepareForPaint`` signal of the scene. Pans do not recompute the transform, the
text offset is not recomputed on transform changes, and the anti-scaling transform
does not trigger the auto-range unless ``ensureInBounds`` is set.
"""
import numpy as np
import pytest

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from tests.perf_helpers import count_calls, process_events

app = pg.mkQApp()

# keyword arguments of the texts used by the rendering tests
TEXT_SPECS = [
    dict(text='plain', anchor=(0, 0)),
    dict(text='center', anchor=(0.5, 0.5), color='y', border='w', fill=(0, 0, 255, 100)),
    dict(text='angle 30', anchor=(1, 1), angle=30, color='g'),
    dict(text='rotate axis', anchor=(0, 1), rotateAxis=(1, 1), color='c'),
    dict(html='<b>html</b> text', anchor=(1, 0), angle=-45, rotateAxis=(0, 1)),
]
TEXT_POSITIONS = [(1, 9), (5, 5), (8, 2), (2, 3), (7, 8)]


def _paint_receivers(scene):
    """Number of slots connected to ``scene.sigPrepareForPaint``."""
    try:
        return scene.receivers(scene.sigPrepareForPaint)
    except TypeError:  # PySide6 takes the signature of the signal
        return scene.receivers(QtCore.SIGNAL('sigPrepareForPaint()'))


def _make_plot(xRange=(0, 10), yRange=(0, 10)):
    """Return ``(widget, viewbox)`` with a fixed range."""
    widget = pg.GraphicsLayoutWidget()
    widget.resize(320, 240)
    vb = widget.addViewBox()
    vb.setRange(xRange=xRange, yRange=yRange, padding=0)
    return widget, vb


def _add_scene(vb, angles=(30, 0)):
    """Add the texts, a labelled line and a text with a rotated parent to ``vb``."""
    items = []
    for spec, pos in zip(TEXT_SPECS, TEXT_POSITIONS):
        item = pg.TextItem(**spec)
        item.setPos(*pos)
        vb.addItem(item)
        items.append(item)
    line = pg.InfiniteLine(pos=(4, 4), angle=angles[0], label='label',
                           labelOpts={'rotateAxis': (1, 0), 'position': 0.6})
    vb.addItem(line)
    items.append(line.label)
    parent = QtWidgets.QGraphicsRectItem(0, 0, 1, 1)
    vb.addItem(parent)
    parent.setPos(6, 3)
    parent.setRotation(angles[1])
    child = pg.TextItem('child', anchor=(0.5, 0.5))
    child.setParentItem(parent)
    child.setPos(0.5, 0.5)
    items.append(child)
    return items, line, parent


def _render(widget):
    """Render the widget and return its pixels as an array."""
    process_events()
    img = widget.grab().toImage().convertToFormat(QtGui.QImage.Format.Format_ARGB32)
    return pg.functions.ndarray_from_qimage(img).copy()


def _text_scales(items):
    """Device scale of the text of each item (1 for unscaled text)."""
    scales = []
    for item in items:
        t = item.textItem.deviceTransform(item.getViewWidget().viewportTransform())
        scales.append((np.hypot(t.m11(), t.m12()), np.hypot(t.m21(), t.m22())))
    return np.array(scales)


def test_no_paint_signal_connection_in_viewbox():
    widget, vb = _make_plot()
    scene = widget.scene()
    widget.show()
    process_events()
    before = _paint_receivers(scene)
    texts = []
    for i in range(20):
        text = pg.TextItem(f'text {i}')
        text.setPos(i / 2, i / 2)
        vb.addItem(text)
        texts.append(text)
    corner = pg.TextItem('corner')
    corner.setParentItem(vb)
    line = pg.InfiniteLine(pos=5, angle=0, label='{value:0.1f}')
    vb.addItem(line)
    process_events()
    widget.grab()
    assert _paint_receivers(scene) == before
    widget.close()


def test_paint_signal_connection_kept_when_needed():
    widget, vb = _make_plot()
    scene = widget.scene()
    widget.show()
    process_events()
    before = _paint_receivers(scene)
    # the parent of this text can be rotated at any time
    parent = QtWidgets.QGraphicsRectItem(0, 0, 1, 1)
    vb.addItem(parent)
    text = pg.TextItem('child')
    text.setParentItem(parent)
    process_events()
    assert _paint_receivers(scene) == before + 1
    # moved into the ViewBox: the view signals cover all changes
    vb.addItem(text)
    assert _paint_receivers(scene) == before
    vb.removeItem(text)
    assert text.scene() is None
    assert _paint_receivers(scene) == before
    widget.close()

    # outside of any ViewBox
    view = pg.GraphicsView()
    view.resize(100, 100)
    before = _paint_receivers(view.scene())
    text = pg.TextItem('scene')
    view.scene().addItem(text)
    view.show()
    process_events()
    assert _paint_receivers(view.scene()) == before + 1
    view.close()


def test_line_label_paint_sync_follows_line_parent():
    widget, vb = _make_plot()
    scene = widget.scene()
    widget.show()
    process_events()
    before = _paint_receivers(scene)
    line = pg.InfiniteLine(pos=5, angle=0, label='{value:0.1f}')
    vb.addItem(line)
    assert _paint_receivers(scene) == before
    # an intermediate parent may be rotated without the label being told
    parent = QtWidgets.QGraphicsRectItem(0, 0, 1, 1)
    vb.addItem(parent)
    line.setParentItem(parent)
    assert _paint_receivers(scene) == before + 1
    line.setParentItem(vb.innerSceneItem())
    assert _paint_receivers(scene) == before
    widget.close()


def test_pan_keeps_transform_and_text_offset():
    widget, vb = _make_plot()
    texts = []
    for i in range(10):
        text = pg.TextItem(f'text {i}', anchor=(0.5, 0.5), angle=10 * i)
        text.setPos(i, i)
        vb.addItem(text)
        texts.append(text)
    widget.show()
    process_events()
    with count_calls(pg.TextItem, 'setTransform') as transforms, \
         count_calls(pg.TextItem, 'updateTextPos') as offsets:
        for i in range(5):
            vb.translateBy((0.5, -0.25))
            process_events()
        assert transforms.count == 0
        for i in range(5):
            vb.scaleBy((0.9, 0.8))
            process_events()
        # one update per text and per zoom, whatever the paint scheduling
        assert transforms.count == 5 * len(texts)
        assert offsets.count == 0
    np.testing.assert_allclose(_text_scales(texts), 1, rtol=1e-9)
    widget.close()


@pytest.mark.parametrize('ensureInBounds', [False, True])
def test_transform_update_triggers_auto_range_only_if_needed(monkeypatch, ensureInBounds):
    widget, vb = _make_plot()
    texts = []
    for i in range(10):
        text = pg.TextItem(f'text {i}', ensureInBounds=ensureInBounds)
        text.setPos(i, i)
        vb.addItem(text)
        texts.append(text)
    widget.show()
    process_events()
    notified = []
    original = pg.ViewBox.itemBoundsChanged

    def itemBoundsChanged(self, item):
        notified.append(item)
        return original(self, item)

    monkeypatch.setattr(pg.ViewBox, 'itemBoundsChanged', itemBoundsChanged)
    for i in range(3):
        vb.scaleBy((0.9, 0.8))
        process_events()
    from_texts = [item for item in notified if isinstance(item, pg.TextItem)]
    assert len(from_texts) == (3 * len(texts) if ensureInBounds else 0)
    # moving a text still updates the auto-range
    notified.clear()
    texts[0].setPos(3, 7)
    assert notified == [texts[0]]
    widget.close()


def test_line_label_follows_line_rotation_immediately():
    widget, vb = _make_plot()
    line = pg.InfiniteLine(pos=(5, 5), angle=30, label='label',
                           labelOpts={'rotateAxis': (1, 0)})
    vb.addItem(line)
    widget.show()
    process_events()
    line.setAngle(70)
    # no paint in between: the line updates its label itself
    ref = pg.TextItem('label', rotateAxis=(1, 0))
    ref.setParentItem(line)
    t, r = line.label.sceneTransform(), ref.sceneTransform()
    np.testing.assert_allclose([t.m11(), t.m12(), t.m21(), t.m22()],
                               [r.m11(), r.m12(), r.m21(), r.m22()], rtol=1e-9, atol=1e-12)
    widget.close()


@pytest.mark.parametrize('hide_parent', [False, True])
def test_hidden_text_updated_when_shown_again(hide_parent):
    # the view may change while the text, or its parent, is hidden
    widget, vb = _make_plot()
    text = pg.TextItem('hidden', anchor=(0.5, 0.5))
    vb.addItem(text)
    widget.show()
    process_events()
    hidden = vb.innerSceneItem() if hide_parent else text
    hidden.hide()
    vb.scaleBy((0.25, 0.5))
    process_events()
    hidden.show()
    np.testing.assert_allclose(_text_scales([text]), 1, rtol=1e-9)
    widget.close()


def test_rendering_after_view_changes_matches_fresh_view():
    final_range = dict(xRange=(-1.5, 12.25), yRange=(0.75, 9.5))
    widget, vb = _make_plot()
    items, line, parent = _add_scene(vb)
    widget.show()
    process_events()
    steps = [
        lambda: vb.setRange(xRange=(1.3, 11.7), yRange=(-0.4, 9.1), padding=0),
        lambda: vb.scaleBy((0.37, 0.61)),
        lambda: vb.translateBy((2.1, -1.7)),
        lambda: widget.resize(301, 229),
        lambda: vb.invertY(True),
        lambda: [line.setAngle(55), parent.setRotation(-35)],
        lambda: vb.invertX(True),
    ]
    for step in steps:
        step()
        process_events()
    widget.resize(320, 240)
    vb.invertX(False)
    vb.invertY(False)
    vb.setRange(**final_range, padding=0)
    image = _render(widget)
    scales = _text_scales(items)
    widget.close()

    ref_widget, ref_vb = _make_plot(**final_range)
    ref_items, _, _ = _add_scene(ref_vb, angles=(55, -35))
    ref_widget.show()
    reference = _render(ref_widget)
    ref_scales = _text_scales(ref_items)
    ref_widget.close()

    np.testing.assert_allclose(scales, ref_scales, rtol=1e-9)
    np.testing.assert_array_equal(image, reference)
