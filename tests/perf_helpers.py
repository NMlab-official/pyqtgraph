"""
Deterministic instrumentation helpers for performance regression tests.

Assertions on durations are unreliable in continuous integration. The helpers in this
module count calls instead (paints, ``setData``, bound computations, ...), so that a
performance regression shows up as a changed count.
"""
from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator

from pyqtgraph.Qt import QtWidgets

__all__ = ['CallCounter', 'count_calls', 'process_events', 'paints_per_update']


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

        setattr(target, name, wrapper)

    @property
    def count(self) -> int:
        """int: Number of calls observed since creation or the last :meth:`reset`."""
        return self._count

    def reset(self) -> None:
        """Set the call count back to zero."""
        self._count = 0

    def restore(self) -> None:
        """Restore the original attribute on the target."""
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
        all its instances, including calls made by Qt through virtual methods.
    name : str
        Name of the method.

    Yields
    ------
    CallCounter
        The active counter.
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


def paints_per_update(widget: QtWidgets.QWidget, item_class: type,
                      update_fn: Callable[[int], object], n: int = 20,
                      warmup: int = 3) -> float:
    """
    Average number of ``paint`` calls of ``item_class`` per update.

    The widget is shown, ``update_fn`` is called ``warmup`` times without counting, then
    ``n`` times while counting the paints of ``item_class``. Events are processed after
    each update, as in an application event loop.

    Parameters
    ----------
    widget : QtWidgets.QWidget
        Widget displaying the item, e.g. a ``PlotWidget``.
    item_class : type
        Class whose ``paint`` calls are counted, e.g. ``PlotCurveItem``.
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
    widget.show()
    process_events(5)
    for i in range(warmup):
        update_fn(i)
        process_events()
    with count_calls(item_class, 'paint') as paints:
        for i in range(warmup, warmup + n):
            update_fn(i)
            process_events()
    return paints.count / n
