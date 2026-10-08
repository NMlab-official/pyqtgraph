import numpy as np

import pyqtgraph as pg
from pyqtgraph.graphicsItems.PlotCurveItem import arrayToLineSegments
from tests.image_testing import assertImageApproved
from tests.perf_helpers import count_calls, process_events


def test_PlotCurveItem():
    p = pg.GraphicsLayoutWidget()
    p.resize(200, 150)
    p.ci.setContentsMargins(4, 4, 4, 4)  # default margins vary by platform
    p.show()
    v = p.addViewBox()
    data = np.array([1,4,2,3,np.inf,5,7,6,-np.inf,8,10,9,np.nan,-1,-2,0])
    c = pg.PlotCurveItem(data)
    c.setSegmentedLineMode('off')   # test images assume non-segmented-line-mode
    v.addItem(c)
    v.autoRange()
    
    # Check auto-range works. Some platform differences may be expected.
    checkRange = np.array([[-1.1457564053237301, 16.145756405323731], [-3.076811473165955, 11.076811473165955]])
    assert np.allclose(v.viewRange(), checkRange)
    
    assertImageApproved(p, 'plotcurveitem/connectall', "Plot curve with all points connected.")
    
    c.setData(data, connect='pairs')
    assertImageApproved(p, 'plotcurveitem/connectpairs', "Plot curve with pairs connected.")
    
    c.setData(data, connect='finite')
    assertImageApproved(p, 'plotcurveitem/connectfinite', "Plot curve with finite points connected.")

    c.setData(data, connect='finite', skipFiniteCheck=True)
    assertImageApproved(p, 'plotcurveitem/connectfinite', "Plot curve with finite points connected using QPolygonF.")
    c.setSkipFiniteCheck(False)
    
    c.setData(data, connect=np.array([1,1,1,0,1,1,0,0,1,0,0,0,1,1,0,0]))
    assertImageApproved(p, 'plotcurveitem/connectarray', "Plot curve with connection array.")

    p.close()


def test_arrayToLineSegments():
    # test the boundary case where the dataset consists of a single point
    xy = np.array([0.])
    parray = arrayToLineSegments(xy, xy, connect='all', finiteCheck=True)
    segs = parray.drawargs()
    assert isinstance(segs, tuple) and len(segs) in [1, 2]
    if len(segs) == 1:
        assert len(segs[0]) == 0
    elif len(segs) == 2:
        assert segs[1] == 0


def test_view_change_keeps_data_bounds_cache():
    # T1.1: panning or zooming must not recompute the O(N) data bounds
    pw = pg.PlotWidget()
    pw.resize(300, 200)
    pw.show()
    curve = pg.PlotCurveItem(np.arange(1000.), np.sin(np.arange(1000.) / 50.))
    pw.addItem(curve)
    pw.getViewBox().enableAutoRange(False)
    process_events()
    curve.boundingRect()
    vb = pw.getViewBox()
    rect_before = curve.boundingRect()
    with count_calls(np, 'nanmin') as nanmin:
        for _ in range(5):
            vb.scaleBy((0.5, 0.5))
            process_events()
    assert nanmin.count == 0
    assert curve.dataBounds(0) == (0.0, 999.0)
    # the pixel padding still follows the zoom level
    assert curve.boundingRect() != rect_before
    pw.close()
