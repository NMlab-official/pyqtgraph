"""asv benchmarks of the scene / ViewBox / AxisItem overhead around data items."""
import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui

rng = np.random.default_rng(12345)


class TimeChildrenBounds:
    """Cost of ``ViewBox.childrenBounds`` with many items."""

    param_names = ["Items"]
    params = ([20, 400],)

    def setup(self, nitems: int) -> None:
        pg.mkQApp()
        self.widget = pg.PlotWidget()
        self.widget.resize(1000, 600)
        for k in range(nitems):
            self.widget.addItem(pg.InfiniteLine(pos=k * 0.1, angle=0))
        self.widget.plot(rng.standard_normal(1000))
        self.vb = self.widget.getViewBox()

    def teardown(self, nitems: int) -> None:
        self.widget.close()

    def time_childrenBounds(self, nitems: int) -> None:
        self.vb.childrenBounds()


class TimeAxisDrawSpecs:
    """Cost of regenerating the tick specification of an axis after a range change."""

    param_names = ["Axis"]
    params = (['numeric', 'date'],)

    def setup(self, kind: str) -> None:
        pg.mkQApp()
        axis = {'bottom': pg.DateAxisItem()} if kind == 'date' else {}
        self.widget = pg.PlotWidget(axisItems=axis)
        self.widget.resize(1400, 400)
        self.widget.show()
        self.axis = self.widget.getPlotItem().getAxis('bottom')
        self.offset = 1.7e9 if kind == 'date' else 0.0
        self.k = 0
        self.image = QtGui.QImage(1400, 400, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
        self.painter = QtGui.QPainter(self.image)

    def teardown(self, kind: str) -> None:
        self.painter.end()
        self.widget.close()

    def time_range_change(self, kind: str) -> None:
        self.k += 1
        self.widget.setXRange(self.offset + self.k, self.offset + self.k + 86400, padding=0)
        self.axis.generateDrawSpecs(self.painter)
