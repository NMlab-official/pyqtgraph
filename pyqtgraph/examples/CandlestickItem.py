"""
Demonstrates CandlestickItem: OHLC candlesticks with a volume bar graph below.

A long history of one-minute candles is displayed; zoom out to see the candles
aggregated per pixel and zoom in to see individual candles. A new candle is
appended periodically, as when streaming market data.
"""

import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

app = pg.mkQApp("CandlestickItem Example")
win = pg.GraphicsLayoutWidget(show=True)
win.resize(1000, 700)
win.setWindowTitle('pyqtgraph example: CandlestickItem')

# random walk of one-minute candles
rng = np.random.default_rng(0)
count = 100_000
step = 60.0
start = 1.7e9
x = start + step * np.arange(count)
close = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, count)))
open_ = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, 2e-4, count))
high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 5e-4, count)))
low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 5e-4, count)))
volume = rng.gamma(2.0, 50.0, count)

upColor, downColor = (38, 166, 154), (239, 83, 80)
upBrush, downBrush = pg.mkBrush(upColor), pg.mkBrush(downColor)

price = win.addPlot(row=0, col=0, axisItems={'bottom': pg.DateAxisItem()})
price.addLegend()
price.showGrid(x=True, y=True, alpha=0.3)
candles = pg.CandlestickItem(x=x, open=open_, high=high, low=low, close=close,
                             upBrush=upBrush, downBrush=downBrush,
                             upPen=upColor, downPen=downColor, name='price')
price.addItem(candles)
# fit the price axis to the visible candles only
price.setAutoVisible(y=True)

volumes = win.addPlot(row=1, col=0, axisItems={'bottom': pg.DateAxisItem()})
volumes.setXLink(price)
volumes.setAutoVisible(y=True)
volumes.setMaximumHeight(200)


def volumeOpts():
    """Volume bar options, coloured like the candles."""
    rising = close >= open_
    return dict(x=x, height=volume, width=0.8 * step,
                brushes=[upBrush if up else downBrush for up in rising])


bars = pg.BarGraphItem(pen=None, **volumeOpts())
volumes.addItem(bars)

price.setXRange(x[-300], x[-1] + step, padding=0)


def appendCandle():
    """Append one random candle to the price and volume plots."""
    global x, open_, high, low, close, volume
    o = close[-1]
    c = o * np.exp(rng.normal(0, 1e-3))
    h = max(o, c) * np.exp(abs(rng.normal(0, 5e-4)))
    lo = min(o, c) * np.exp(-abs(rng.normal(0, 5e-4)))
    t = x[-1] + step
    candles.appendData(x=[t], open=[o], high=[h], low=[lo], close=[c])
    x, open_, high, low, close = (np.append(a, v) for a, v in
                                  ((x, t), (open_, o), (high, h), (low, lo), (close, c)))
    volume = np.append(volume, rng.gamma(2.0, 50.0))
    bars.setOpts(**volumeOpts())


timer = QtCore.QTimer()
timer.timeout.connect(appendCandle)
timer.start(500)

if __name__ == '__main__':
    pg.exec()
