"""asv benchmarks of ScatterPlotItem."""
import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

rng = np.random.default_rng(12345)


class TimeScatterSetData:
    """Cost of ``ScatterPlotItem.setData`` for several styling variants."""

    param_names = ["Size", "Style"]
    params = ([10_000, 100_000], ['uniform', 'QBrush list', 'tuple list', 'float sizes'])

    def setup(self, nelems: int, style: str) -> None:
        pg.mkQApp()
        self.kwargs = {'x': rng.random(nelems), 'y': rng.random(nelems), 'size': 7, 'pen': None}
        if style == 'QBrush list':
            palette = [pg.mkBrush(c) for c in ('r', 'g', 'b', 'y', 'c')]
            self.kwargs['brush'] = [palette[i] for i in rng.integers(0, 5, nelems)]
        elif style == 'tuple list':
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            self.kwargs['brush'] = [colors[i] for i in rng.integers(0, 3, nelems)]
        elif style == 'float sizes':
            self.kwargs['size'] = rng.uniform(5, 15, nelems)
        self.item = pg.ScatterPlotItem()

    def time_setData(self, nelems: int, style: str) -> None:
        self.item.setData(**self.kwargs)


class TimeScatterHitTest:
    """Cost of the first hit test after ``setData``."""

    param_names = ["Size"]
    params = ([10_000, 100_000],)

    def setup(self, nelems: int) -> None:
        pg.mkQApp()
        self.x = rng.random(nelems)
        self.y = rng.random(nelems)
        self.item = pg.ScatterPlotItem()

    def time_setData_pointsAt(self, nelems: int) -> None:
        self.item.setData(x=self.x, y=self.y, size=7)
        self.item.pointsAt(QtCore.QPointF(0.5, 0.5))
