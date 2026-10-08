"""
Deterministic instrumentation helpers for performance regression tests.

Assertions on durations are unreliable in continuous integration. The helpers in this
module count calls instead (paints, ``setData``, bound computations, ...), so that a
performance regression shows up as a changed count.
"""
from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator

from pyqtgraph.Qt import QtCore, QtWidgets

__all__ = ['CallCounter', 'count_calls', 'process_events', 'paints_per_update']

# Wrappers of the restored counters, kept alive on purpose. PySide6 caches, per object,
# the Python override of a C++ virtual method (e.g. ``paint``) found at its first call
# from Qt: an object first called while a counter was active keeps calling the wrapper
# after the counter is restored, and freeing the wrapper then crashed PySide6.
_retired_wrappers: list[Callable] = []


class CallCounter:
    """
    Wrap a method of a class or of an object and count its calls.

    The original attribute is restored by :meth:`restore`, which
    :func:`count_calls` calls automatically.

    Parameters
    ----------
    target : type or object
        Class or instance whose method is wrapped.
    name : str
        Name of the method.
    """

    def __init__(self, target: object, name: str) -> None:
        self._target = target
        self._name = name
        self._had_own_attr = name in vars(target)
        self._original = vars(target).get(name)
        self._count = 0
        bound_or_function = getattr(target, name)
        counter = self

        def wrapper(*args, **kwargs):
            counter._count += 1
            return bound_or_function(*args, **kwargs)

        self._wrapper = wrapper
        setattr(target, name, wrapper)

    @property
    def count(self) -> int:
        """int: Number of calls observed since creation or the last :meth:`reset`."""
        return self._count

    def reset(self) -> None:
        """Set the call count back to zero."""
        self._count = 0

    def restore(self) -> None:
        """
        Restore the original attribute on the target.

        The wrapper is kept alive, as objects may still call it with PySide6 (see
        :func:`count_calls`).
        """
        _retired_wrappers.append(self._wrapper)
        if self._had_own_attr:
            setattr(self._target, self._name, self._original)
        else:
            delattr(self._target, self._name)


@contextlib.contextmanager
def count_calls(target: object, name: str) -> Iterator[CallCounter]:
    """
    Count calls of ``target.name`` within a ``with`` block.

    Parameters
    ----------
    target : type or object
        Class or instance whose method is wrapped. Wrapping a class counts the calls of
        all its instances, including calls made by Qt through virtual methods (but see
        the notes for PySide6).
    name : str
        Name of the method.

    Yields
    ------
    CallCounter
        The active counter.

    Notes
    -----
    With PySide6, the calls that Qt makes to a C++ virtual method (e.g. ``paint``)
    reach the wrapper only for objects whose first such call happens while counting:
    PySide6 caches the Python override of each object at its first call. Such objects
    keep calling the wrapper after the ``with`` block, which therefore stays alive.
    Count paints with the paint events of the viewport instead, see
    :func:`paints_per_update`.
    """
    counter = CallCounter(target, name)
    try:
        yield counter
    finally:
        counter.restore()


def process_events(passes: int = 3) -> None:
    """
    Process pending Qt events, queued calls and paint events included.

    Parameters
    ----------
    passes : int, default 3
        Number of ``processEvents`` passes.
    """
    app = QtWidgets.QApplication.instance()
    for _ in range(passes):
        app.processEvents()


class _PaintEventCounter(QtCore.QObject):
    """
    Event filter counting the paint events received by the objects it is installed on.

    An event filter sees the events of any binding, unlike a Python method patched onto
    a class after its instances were created: PySide6 does not dispatch such virtual
    method calls to the patch.
    """

    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    def eventFilter(self, obj: QtCore.QObject, ev: QtCore.QEvent) -> bool:
        """
        Count ``ev`` if it is a paint event; never filter it out.

        Parameters
        ----------
        obj : QtCore.QObject
            The object receiving the event.
        ev : QtCore.QEvent
            The event.

        Returns
        -------
        bool
            Always False: the event is delivered normally.
        """
        if ev.type() == QtCore.QEvent.Type.Paint:
            self.count += 1
        return False


def paints_per_update(widget: QtWidgets.QWidget, item_class: type | None,
                      update_fn: Callable[[int], object], n: int = 20,
                      warmup: int = 3) -> float:
    """
    Average number of times the widget is painted per update.

    The widget is shown, ``update_fn`` is called ``warmup`` times without counting, then
    ``n`` times while counting the paint events of the widget's viewport (the widget
    itself if it is not a scroll area). Events are processed after each update, as in
    an application event loop. Each viewport paint event paints all the items of the
    region to repaint, so this is the number of paints of the updated items, with any
    Qt binding.

    Parameters
    ----------
    widget : QtWidgets.QWidget
        Widget displaying the items, e.g. a ``PlotWidget``.
    item_class : type or None
        Unused, kept for compatibility. The ``paint`` calls of this class used to be
        counted, by patching the method, which PySide6 does not call for existing
        items; the paint events of the viewport are counted instead.
    update_fn : callable
        Function performing one update; receives the update index.
    n : int, default 20
        Number of counted updates.
    warmup : int, default 3
        Number of uncounted updates made first.

    Returns
    -------
    float
        Mean number of paints per update; 1.0 means each update is painted once.
    """
    if isinstance(widget, QtWidgets.QAbstractScrollArea):
        target = widget.viewport()
    else:
        target = widget
    widget.show()
    process_events(5)
    for i in range(warmup):
        update_fn(i)
        process_events()
    counter = _PaintEventCounter()
    target.installEventFilter(counter)
    try:
        for i in range(warmup, warmup + n):
            update_fn(i)
            process_events()
    finally:
        target.removeEventFilter(counter)
    return counter.count / n
