"""asv benchmarks of the line data pipeline (PlotDataItem / PlotCurveItem)."""
import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

rng = np.random.default_rng(12345)


class TimeDisplayPipeline:
    """Cost of ``setData`` followed by the computation of the displayed data."""

    param_names = ["Size", "Mode"]
    params = ([100_000, 1_000_000], ['default', 'autoDownsample', 'clip+autoDownsample'])

    def setup(self, nelems: int, mode: str) -> None:
        pg.mkQApp()
        self.x = np.arange(nelems, dtype=np.float64)
        self.y = np.cumsum(rng.standard_normal(nelems))
        self.widget = pg.PlotWidget()
        self.widget.resize(1000, 600)
        self.item = self.widget.plot()
        if mode != 'default':
            self.item.setDownsampling(auto=True, method='peak')
        if mode == 'clip+autoDownsample':
            self.item.setClipToView(True)
            self.widget.setXRange(nelems - 5000, nelems, padding=0)

    def teardown(self, nelems: int, mode: str) -> None:
        self.widget.close()

    def time_setData_display(self, nelems: int, mode: str) -> None:
        self.item.setData(self.x, self.y)
        self.item.getData()


class TimeCurvePath:
    """Cost of building the QPainterPath of a PlotCurveItem after ``setData``."""

    param_names = ["Size"]
    params = ([100_000, 1_000_000],)

    def setup(self, nelems: int) -> None:
        pg.mkQApp()
        self.x = np.arange(nelems, dtype=np.float64)
        self.y = rng.standard_normal(nelems)
        self.curve = pg.PlotCurveItem()

    def time_setData_getPath(self, nelems: int) -> None:
        self.curve.setData(self.x, self.y)
        self.curve.getPath()

    def time_dataBounds(self, nelems: int) -> None:
        self.curve.setData(self.x, self.y)
        self.curve.dataBounds(0)
        self.curve.dataBounds(1)


class TimeViewRangeChange:
    """Cost of a vertical range change for a PlotDataItem with default options."""

    param_names = ["Size"]
    params = ([100_000, 1_000_000],)

    def setup(self, nelems: int) -> None:
        pg.mkQApp()
        self.widget = pg.PlotWidget()
        self.widget.resize(1000, 600)
        self.item = self.widget.plot(np.arange(nelems, dtype=np.float64),
                                     np.cumsum(rng.standard_normal(nelems)))
        self.vb = self.widget.getViewBox()
        self.vb.enableAutoRange(False)

    def teardown(self, nelems: int) -> None:
        self.widget.close()

    def time_pan_y(self, nelems: int) -> None:
        self.vb.translateBy(y=0.1)
        self.item.curve.getPath()
        QtCore.QCoreApplication.processEvents()
