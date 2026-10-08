from ..Qt import QtCore, QtGui, QtOpenGL, QtWidgets

import math
import warnings
import weakref

import numpy as np

from .. import Qt, debug
from .. import functions as fn
from .. import getConfigOption
from ..Qt import OpenGLConstants as GLC
from ..Qt import OpenGLHelpers
from .GraphicsObject import GraphicsObject

__all__ = ['PlotCurveItem']


class OpenGLState(QtCore.QObject):
    VERT_SRC = """
        attribute vec4 a_position;
        uniform mat4 u_mvp;
        void main() {
            gl_Position = u_mvp * a_position;
        }
    """
    FRAG_SRC = """
        uniform highp vec4 u_color;
        void main() {
            gl_FragColor = u_color;
        }
    """
    VERT_SRC_140 = """
        #version 140
        in vec4 a_position;
        uniform mat4 u_mvp;
        void main() {
            gl_Position = u_mvp * a_position;
        }
    """
    FRAG_SRC_140 = """
        #version 140
        uniform vec4 u_color;
        out vec4 fragColor;
        void main() {
            fragColor = u_color;
        }
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.context = None
        self.vbo_nbytes = 0
        self.render_cache = None
        self.m_vao = QtOpenGL.QOpenGLVertexArrayObject(self)
        self.m_vbo = QtOpenGL.QOpenGLBuffer(QtOpenGL.QOpenGLBuffer.Type.VertexBuffer)

    def setup(self, context):
        if self.context is context:
            return

        if self.context is not None:
            self.context.aboutToBeDestroyed.disconnect(self.cleanup)
            self.cleanup()

        self.context = context
        self.context.aboutToBeDestroyed.connect(self.cleanup)

        glwidget = self.parent()
        program = glwidget.retrieveProgram("PlotCurveItem")
        if program is None:
            program = QtOpenGL.QOpenGLShaderProgram()

            is_opengles = self.context.isOpenGLES()
            gl_version = self.context.format().version()

            if not is_opengles and gl_version >= (3, 1):
                vert_src = OpenGLState.VERT_SRC_140
                frag_src = OpenGLState.FRAG_SRC_140
            else:
                vert_src = OpenGLState.VERT_SRC
                frag_src = OpenGLState.FRAG_SRC

            if not program.addShaderFromSourceCode(QtOpenGL.QOpenGLShader.ShaderTypeBit.Vertex, vert_src):
                raise RuntimeError(program.log())
            if not program.addShaderFromSourceCode(QtOpenGL.QOpenGLShader.ShaderTypeBit.Fragment, frag_src):
                raise RuntimeError(program.log())
            program.bindAttributeLocation("a_position", 0)
            if not program.link():
                raise RuntimeError(program.log())
            glwidget.storeProgram("PlotCurveItem", program)

        self.m_vao.create()
        self.m_vbo.create()
        self.vbo_nbytes = 0

        self.m_vao.bind()
        self.m_vbo.bind()
        program.enableAttributeArray(0)
        program.setAttributeBuffer(0, GLC.GL_FLOAT, 0, 2)
        self.m_vbo.release()
        self.m_vao.release()

    def cleanup(self):
        # this method should restore the state back to __init__
        glwidget = self.parent()
        glwidget.makeCurrent()

        self.m_vbo.destroy()
        self.m_vao.destroy()

        self.context = None
        self.vbo_nbytes = 0
        self.render_cache = None

        glwidget.doneCurrent()

    def verticesChanged(self, curve):
        self.render_cache = None

def arrayToLineSegments(x, y, connect, finiteCheck, out=None):
    if out is None:
        out = Qt.internals.PrimitiveArray(QtCore.QLineF, 4)

    # analogue of arrayToQPath taking the same parameters
    if len(x) < 2:
        out.resize(0)
        return out

    connect_array = None
    if isinstance(connect, np.ndarray):
        # the last element is not used
        connect_array, connect = np.asarray(connect[:-1], dtype=bool), 'array'

    all_finite = True
    if finiteCheck or connect == 'finite':
        mask = np.isfinite(x) & np.isfinite(y)
        all_finite = np.all(mask)

    if connect == 'all':
        if not all_finite:
            # remove non-finite points, if any
            x = x[mask]
            y = y[mask]

    elif connect == 'finite':
        if all_finite:
            connect = 'all'
        else:
            # each non-finite point affects the segment before and after
            connect_array = mask[:-1] & mask[1:]

    elif connect == 'pairs':
        if not all_finite:
            # ensure that we have an even number of elements
            npairs = len(x) // 2
            mask = mask[:npairs*2]
            # remove pair if at least one point within pair is non-finite
            mask.reshape((-1, 2))[:] = (mask[0::2] & mask[1::2])[:, np.newaxis]
            x = x[:npairs*2][mask]
            y = y[:npairs*2][mask]

    elif connect == 'array':
        if not all_finite:
            # replicate the behavior of arrayToQPath
            backfill_idx = fn._compute_backfill_indices(mask)
            x = x[backfill_idx]
            y = y[backfill_idx]

    if connect == 'all':
        nsegs = len(x) - 1
        out.resize(nsegs)
        if nsegs:
            memory = out.ndarray()
            memory[:, 0] = x[:-1]
            memory[:, 2] = x[1:]
            memory[:, 1] = y[:-1]
            memory[:, 3] = y[1:]

    elif connect == 'pairs':
        nsegs = len(x) // 2
        out.resize(nsegs)
        if nsegs:
            memory = out.ndarray()
            memory = memory.reshape((-1, 2))
            memory[:, 0] = x[:nsegs * 2]
            memory[:, 1] = y[:nsegs * 2]

    elif connect_array is not None:
        # the following are handled here
        # - 'array'
        # - 'finite' with non-finite elements
        nsegs = np.count_nonzero(connect_array)
        out.resize(nsegs)
        if nsegs:
            memory = out.ndarray()
            memory[:, 0] = x[:-1][connect_array]
            memory[:, 2] = x[1:][connect_array]
            memory[:, 1] = y[:-1][connect_array]
            memory[:, 3] = y[1:][connect_array]

    else:
        nsegs = 0
        out.resize(nsegs)

    return out


class _VertexCache:
    """
    Per-data cache of the vertices of a curve drawn as a single polyline.

    ``PlotCurveItem.updateData`` stores new array views on every call, so the cache is
    bound to the identity of the data arrays. Only weak references to them are kept,
    so that the cache never keeps replaced data alive.

    Parameters
    ----------
    x, y : np.ndarray
        Data arrays of the curve.
    connect : str
        ``connect`` option of the curve.
    skipFiniteCheck : bool
        ``skipFiniteCheck`` option of the curve.
    """

    __slots__ = ('_xref', '_yref', '_connect', '_skipFiniteCheck',
                 'computed', 'vertices', 'polylineFilled')

    def __init__(self, x: np.ndarray, y: np.ndarray, connect: str,
                 skipFiniteCheck: bool) -> None:
        self._xref = weakref.ref(x)
        self._yref = weakref.ref(y)
        self._connect = connect
        self._skipFiniteCheck = skipFiniteCheck
        self.computed = False
        self.vertices: tuple[np.ndarray, np.ndarray] | None = None
        self.polylineFilled = False

    def matches(self, x: np.ndarray, y: np.ndarray, connect: str,
                skipFiniteCheck: bool) -> bool:
        """
        Tell whether the cache was built for these data and options.

        Parameters
        ----------
        x, y : np.ndarray
            Current data arrays of the curve.
        connect : str
            Current ``connect`` option.
        skipFiniteCheck : bool
            Current ``skipFiniteCheck`` option.

        Returns
        -------
        bool
            True if the cached values are valid.
        """
        return (
            self._xref() is x
            and self._yref() is y
            and self._connect == connect
            and self._skipFiniteCheck == skipFiniteCheck
        )


class PlotCurveItem(GraphicsObject):
    """
    Class representing a single plot curve. Instances of this class are created
    automatically as part of :class:`PlotDataItem <pyqtgraph.PlotDataItem>`; 
    these rarely need to be instantiated directly.

    Features:

      - Fast data update
      - Fill under curve
      - Mouse interaction

    =====================  ===============================================
    **Signals:**
    sigPlotChanged(self)   Emitted when the data being plotted has changed
    sigClicked(self, ev)   Emitted when the curve is clicked
    =====================  ===============================================
    """

    sigPlotChanged = QtCore.Signal(object)
    sigClicked = QtCore.Signal(object, object)

    # Private caches of the polyline drawing of paint(), created on first use.
    _vertexCache: _VertexCache | None = None
    _polyline: QtGui.QPolygonF | None = None
    _polylineMaxSize: int = 0

    def __init__(self, *args, **kwargs):
        """
        Forwards all arguments to :func:`setData <pyqtgraph.PlotCurveItem.setData>`.

        Some extra arguments are accepted as well:

        ==============  =======================================================
        **Arguments:**
        parent          The parent GraphicsObject (optional)
        clickable       If `True`, the item will emit ``sigClicked`` when it is
                        clicked on. Defaults to `False`.
        ==============  =======================================================
        """
        GraphicsObject.__init__(self, kwargs.get('parent', None))
        self.clear()

        ## this is disastrous for performance.
        #self.setCacheMode(QtWidgets.QGraphicsItem.CacheMode.DeviceCoordinateCache)

        self.metaData = {}
        self.opts = {
            'shadowPen': None,
            'fillLevel': None,
            'fillOutline': False,
            'brush': None,
            'stepMode': None,
            'name': None,
            'antialias': getConfigOption('antialias'),
            'connect': 'all',
            'mouseWidth': 8, # width of shape responding to mouse click
            'compositionMode': None,
            'skipFiniteCheck': False,
            'segmentedLineMode': getConfigOption('segmentedLineMode'),
        }
        if 'pen' not in kwargs:
            self.opts['pen'] = fn.mkPen('w')
        self.setClickable(kwargs.get('clickable', False))
        self.setData(*args, **kwargs)
        self.glstate = None

    def implements(self, interface=None):
        ints = ['plotData']
        if interface is None:
            return ints
        return interface in ints

    def name(self):
        return self.opts.get('name', None)

    def setClickable(self, s, width=None):
        """Sets whether the item responds to mouse clicks.

        The `width` argument specifies the width in pixels orthogonal to the
        curve that will respond to a mouse click.
        """
        self.clickable = s
        if width is not None:
            self.opts['mouseWidth'] = width
            self._mouseShape = None
            self._boundingRect = None

    def setCompositionMode(self, mode):
        """
        Change the composition mode of the item. This is useful when overlaying
        multiple items.
        
        Parameters
        ----------
        mode : ``QtGui.QPainter.CompositionMode``
            Composition of the item, often used when overlaying items.  Common
            options include:

            ``QPainter.CompositionMode.CompositionMode_SourceOver`` (Default)
            Image replaces the background if it is opaque. Otherwise, it uses
            the alpha channel to blend the image with the background.

            ``QPainter.CompositionMode.CompositionMode_Overlay`` Image color is
            mixed with the background color to reflect the lightness or
            darkness of the background

            ``QPainter.CompositionMode.CompositionMode_Plus`` Both the alpha
            and color of the image and background pixels are added together.

            ``QPainter.CompositionMode.CompositionMode_Plus`` The output is the
            image color multiplied by the background.

            See ``QPainter::CompositionMode`` in the Qt Documentation for more
            options and details
        """
        self.opts['compositionMode'] = mode
        self.update()

    def getData(self):
        return self.xData, self.yData

    def dataBounds(
        self,
        ax: int,
        frac: float = 1.0,
        orthoRange: tuple[float, float] | None = None
    ) -> tuple[float, float] | tuple[None, None]:
        """
        Get the range occupied by the data along an axis.

        The result is cached. The full range of the finite data (``frac >= 1`` and
        no `orthoRange`) is taken from the bounds passed by
        :class:`~pyqtgraph.PlotDataItem` when available, without scanning the data.

        Parameters
        ----------
        ax : { 0, 1 }
            The axis, 0 for `x` and 1 for `y`.
        frac : float, default 1.0
            Fraction of the data to include, centered on the median. Values of 1.0 and
            above include the full range of the finite data.
        orthoRange : tuple of float or None, default None
            Only include the data whose coordinate along the other axis lies within
            this ``(min, max)`` range.

        Returns
        -------
        tuple of float or tuple of None
            ``(min, max)``, including the fill level and the width of non-cosmetic
            pens, or ``(None, None)`` if there is no data.

        Raises
        ------
        ValueError
            Raised for an invalid `ax`.
        Exception
            Raised if `frac` is not positive.
        """
        ## Need this to run as fast as possible.
        ## check cache first:
        cache = self._boundsCache[ax]
        if cache is not None and cache[0] == (frac, orthoRange):
            return cache[1]

        if frac >= 1.0 and orthoRange is None and self._dataBoundsHint is not None:
            b = self._dataBoundsHint[ax]
        else:
            b = self._computeDataBounds(ax, frac, orthoRange)
            if b[0] is None:  # no data, not cached
                return b

        ## adjust for fill level
        if ax == 1 and self.opts['fillLevel'] not in [None, 'enclosed']:
            b = ( 
                float( min(b[0], self.opts['fillLevel']) ), 
                float( max(b[1], self.opts['fillLevel']) )
            ) # enforce float format for bounds, even if data format is different

        ## Add pen width only if it is non-cosmetic.
        pen = self.opts['pen']
        spen = self.opts['shadowPen']
        if pen is not None and not pen.isCosmetic() and pen.style() != QtCore.Qt.PenStyle.NoPen:
            b = (b[0] - pen.widthF()*0.7072, b[1] + pen.widthF()*0.7072)
        if spen is not None and not spen.isCosmetic() and spen.style() != QtCore.Qt.PenStyle.NoPen:
            b = (b[0] - spen.widthF()*0.7072, b[1] + spen.widthF()*0.7072)

        self._boundsCache[ax] = [(frac, orthoRange), b]
        return b

    def _computeDataBounds(
        self,
        ax: int,
        frac: float,
        orthoRange: tuple[float, float] | None
    ) -> tuple[float, float] | tuple[None, None]:
        """
        Scan the data for the range occupied along an axis.

        Parameters
        ----------
        ax : { 0, 1 }
            The axis, 0 for `x` and 1 for `y`.
        frac : float
            Fraction of the data to include, see :meth:`dataBounds`.
        orthoRange : tuple of float or None
            Range along the other axis, see :meth:`dataBounds`.

        Returns
        -------
        tuple of float or tuple of None
            ``(min, max)`` of the data, without fill level or pen width, or
            ``(None, None)`` if there is no data.

        Raises
        ------
        ValueError
            Raised for an invalid `ax`.
        Exception
            Raised if `frac` is not positive.
        """
        (x, y) = self.getData()
        if x is None or len(x) == 0:
            return (None, None)

        if ax == 0:
            d = x
            d2 = y
        elif ax == 1:
            d = y
            d2 = x
        else:
            raise ValueError("Invalid axis value")

        ## If an orthogonal range is specified, mask the data now
        if orthoRange is not None:
            mask = (d2 >= orthoRange[0]) * (d2 <= orthoRange[1])
            if self.opts.get("stepMode", None) == "center":
                if ax == 1:
                    mask = mask[:-1]  # len(y) == len(x) - 1 when stepMode is center
                else:
                    mask = np.concatenate([mask, [0]])
            d = d[mask]
            #d2 = d2[mask]

        if len(d) == 0:
            return (None, None)

        ## Get min/max (or percentiles) of the requested data range
        if frac >= 1.0:
            # include complete data range
            # first try faster nanmin/max function, then cut out infs if needed.
            with warnings.catch_warnings(): 
                # All-NaN data is acceptable; Explicit numpy warning is not needed.
                warnings.simplefilter("ignore")
                b = ( float(np.nanmin(d)), float(np.nanmax(d)) ) # enforce float format for bounds, even if data format is different
            if math.isinf(b[0]) or math.isinf(b[1]):
                mask = np.isfinite(d)
                d = d[mask]
                if len(d) == 0:
                    return (None, None)
                b = ( float(d.min()), float(d.max()) ) # enforce float format for bounds, even if data format is different

        elif frac <= 0.0:
            raise Exception("Value for parameter 'frac' must be > 0. (got %s)" % str(frac))
        else:
            # include a percentile of data range
            mask = np.isfinite(d)
            d = d[mask]
            if len(d) == 0:
                return (None, None)
            b = np.percentile(d, [50 * (1 - frac), 50 * (1 + frac)]) # percentile result is always float64 or larger
        return b

    def pixelPadding(self):
        pen = self.opts['pen']
        spen = self.opts['shadowPen']
        w = 0
        if  pen is not None and pen.isCosmetic() and pen.style() != QtCore.Qt.PenStyle.NoPen:
            w += pen.widthF()*0.7072
        if spen is not None and spen.isCosmetic() and spen.style() != QtCore.Qt.PenStyle.NoPen:
            w = max(w, spen.widthF()*0.7072)
        if self.clickable:
            w = max(w, self.opts['mouseWidth']//2 + 1)
        return w

    def boundingRect(self):
        if self._boundingRect is None:
            (xmn, xmx) = self.dataBounds(ax=0)
            if xmn is None or xmx is None:
                return QtCore.QRectF()
            (ymn, ymx) = self.dataBounds(ax=1)
            if ymn is None or ymx is None:
                return QtCore.QRectF()

            px = py = 0.0
            pxPad = self.pixelPadding()
            if pxPad > 0:
                # determine length of pixel in local x, y directions
                px, py = self.pixelVectors()
                try:
                    px = 0 if px is None else px.length()
                except OverflowError:
                    px = 0
                try:
                    py = 0 if py is None else py.length()
                except OverflowError:
                    py = 0

                # return bounds expanded by pixel size
                px *= pxPad
                py *= pxPad
            #px += self._maxSpotWidth * 0.5
            #py += self._maxSpotWidth * 0.5
            self._boundingRect = QtCore.QRectF(xmn-px, ymn-py, (2*px)+xmx-xmn, (2*py)+ymx-ymn)

        return self._boundingRect

    def viewTransformChanged(self):
        # Only the pixel padding of the bounding rectangle depends on the view.
        # The data bounds do not: keep their cache, recomputing them is O(N).
        self.prepareGeometryChange()
        self._boundingRect = None

    #def boundingRect(self):
        #if self._boundingRect is None:
            #(x, y) = self.getData()
            #if x is None or y is None or len(x) == 0 or len(y) == 0:
                #return QtCore.QRectF()


            #if self.opts['shadowPen'] is not None:
                #lineWidth = (max(self.opts['pen'].width(), self.opts['shadowPen'].width()) + 1)
            #else:
                #lineWidth = (self.opts['pen'].width()+1)


            #pixels = self.pixelVectors()
            #if pixels == (None, None):
                #pixels = [Point(0,0), Point(0,0)]

            #xmin = x.min()
            #xmax = x.max()
            #ymin = y.min()
            #ymax = y.max()

            #if self.opts['fillLevel'] is not None:
                #ymin = min(ymin, self.opts['fillLevel'])
                #ymax = max(ymax, self.opts['fillLevel'])

            #xmin -= pixels[0].x() * lineWidth
            #xmax += pixels[0].x() * lineWidth
            #ymin -= abs(pixels[1].y()) * lineWidth
            #ymax += abs(pixels[1].y()) * lineWidth

            #self._boundingRect = QtCore.QRectF(xmin, ymin, xmax-xmin, ymax-ymin)
        #return self._boundingRect


    def invalidateBounds(self):
        self._boundingRect = None
        self._boundsCache = [None, None]

    def _styleBoundsChanged(self) -> None:
        """
        Notify a style change that can alter the bounds of the curve.

        The pen widths enter the pixel padding and, for non-cosmetic pens, the data
        bounds; the fill level enters the data bounds. The geometry change is
        announced to the scene before the cached bounds are dropped, and the view is
        told, so that the bounds it caches per item are refreshed as well.
        """
        self.prepareGeometryChange()
        self.invalidateBounds()
        self.informViewBoundsChanged()

    def setPen(self, *args, **kwargs):
        """Set the pen used to draw the curve."""
        if args and args[0] is None:
            self.opts['pen'] = None
        else:
            self.opts['pen'] = fn.mkPen(*args, **kwargs)
        self._styleBoundsChanged()
        self.update()

    def setShadowPen(self, *args, **kwargs):
        """
        Set the shadow pen used to draw behind the primary pen.
        This pen must have a larger width than the primary
        pen to be visible. Arguments are passed to 
        :func:`mkPen <pyqtgraph.mkPen>`
        """
        if args and args[0] is None:
            self.opts['shadowPen'] = None
        else:
            self.opts['shadowPen'] = fn.mkPen(*args, **kwargs)
        self._styleBoundsChanged()
        self.update()

    def setBrush(self, *args, **kwargs):
        """
        Sets the brush used when filling the area under the curve. All 
        arguments are passed to :func:`mkBrush <pyqtgraph.mkBrush>`.
        """
        if args and args[0] is None:
            self.opts['brush'] = None
        else:
            self.opts['brush'] = fn.mkBrush(*args, **kwargs)
        self.invalidateBounds()
        self.update()

    def setFillLevel(self, level):
        """Sets the level filled to when filling under the curve"""
        self.opts['fillLevel'] = level
        self.fillPath = None
        self._fillPathList = None
        self._styleBoundsChanged()
        self.update()
        
    def setSkipFiniteCheck(self, skipFiniteCheck):
        """
        When it is known that the plot data passed to ``PlotCurveItem`` contains only finite numerical values,
        the `skipFiniteCheck` property can help speed up plotting. If this flag is set and the data contains 
        any non-finite values (such as `NaN` or `Inf`), unpredictable behavior will occur. The data might not
        be plotted, or there might be significant performance impact.
        """
        self.opts['skipFiniteCheck']  = bool(skipFiniteCheck)

    def setData(self, *args, **kwargs):
        """
        =============== =================================================================
        **Arguments:**
        x, y            (numpy arrays) Data to display
        pen             Pen to use when drawing. Any single argument accepted by
                        :func:`mkPen <pyqtgraph.mkPen>` is allowed.
        shadowPen       Pen for drawing behind the primary pen. Usually this
                        is used to emphasize the curve by providing a
                        high-contrast border. Any single argument accepted by
                        :func:`mkPen <pyqtgraph.mkPen>` is allowed.
        fillLevel       (float or None) Fill the area under the curve to
                        the specified value.
        fillOutline     (bool) If True, an outline surrounding the `fillLevel`
                        area is drawn.
        brush           Brush to use when filling. Any single argument accepted
                        by :func:`mkBrush <pyqtgraph.mkBrush>` is allowed.
        antialias       (bool) Whether to use antialiasing when drawing. This
                        is disabled by default because it decreases performance.
        stepMode        (str or None) If 'center', a step is drawn using the `x`
                        values as boundaries and the given `y` values are
                        associated to the mid-points between the boundaries of
                        each step. This is commonly used when drawing
                        histograms. Note that in this case, ``len(x) == len(y) + 1``
                        
                        If 'left' or 'right', the step is drawn assuming that
                        the `y` value is associated to the left or right boundary,
                        respectively. In this case ``len(x) == len(y)``
                        If not passed or an empty string or `None` is passed, the
                        step mode is not enabled.
        connect         Argument specifying how vertexes should be connected
                        by line segments. 
                        
                            | 'all' (default) indicates full connection. 
                            | 'pairs' draws one separate line segment for each two points given.
                            | 'finite' omits segments attached to `NaN` or `Inf` values. 
                            | For any other connectivity, specify an array of boolean values.
        compositionMode See :func:`setCompositionMode
                        <pyqtgraph.PlotCurveItem.setCompositionMode>`.
        skipFiniteCheck (bool, defaults to `False`) Optimization flag that can
                        speed up plotting by not checking and compensating for
                        `NaN` values.  If set to `True`, and `NaN` values exist, the
                        data may not be displayed or the plot may take a
                        significant performance hit.
        =============== =================================================================

        If non-keyword arguments are used, they will be interpreted as
        ``setData(y)`` for a single argument and ``setData(x, y)`` for two
        arguments.
        
        **Notes on performance:**
        
        Line widths greater than 1 pixel affect the performance as discussed in 
        the documentation of :class:`PlotDataItem <pyqtgraph.PlotDataItem>`.
        """
        self.updateData(*args, **kwargs)

    def updateData(self, *args, **kwargs) -> None:
        """
        Set the data and options of the curve.

        Parameters
        ----------
        *args
            ``(y,)`` or ``(x, y)``, see :meth:`setData`.
        **kwargs
            Data and options, see :meth:`setData`. The private keyword
            ``_dataBounds``, ``((xmin, xmax), (ymin, ymax))`` of the finite data or
            ``None``, is used by :class:`~pyqtgraph.PlotDataItem` to pass bounds it
            has already computed; it is not part of the public API.

        Raises
        ------
        Exception
            Raised if the data is not one-dimensional, is complex, or if the lengths
            of `x` and `y` do not match.
        """
        profiler = debug.Profiler()
        dataBounds = kwargs.pop('_dataBounds', None)

        if 'compositionMode' in kwargs:
            self.setCompositionMode(kwargs['compositionMode'])

        if len(args) == 1:
            kwargs['y'] = args[0]
        elif len(args) == 2:
            kwargs['x'] = args[0]
            kwargs['y'] = args[1]

        if 'y' not in kwargs or kwargs['y'] is None:
            kwargs['y'] = np.array([])
        if 'x' not in kwargs or kwargs['x'] is None:
            # np.arange(len(y)), reusing the index generated by previous calls
            n = len(kwargs['y'])
            self._indexBuffer = _arangeBuffer(self._indexBuffer, n)
            kwargs['x'] = self._indexBuffer[:n]

        for k in ['x', 'y']:
            data = kwargs[k]
            if isinstance(data, list):
                data = np.array(data)
                kwargs[k] = data
            if not isinstance(data, np.ndarray) or data.ndim > 1:
                raise Exception("Plot data must be 1D ndarray.")
            if data.dtype.kind == 'c':
                raise Exception("Can not plot complex data types.")


        profiler("data checks")

        self.yData = kwargs['y'].view(np.ndarray)
        self.xData = kwargs['x'].view(np.ndarray)
        
        self.prepareGeometryChange()
        self.invalidateBounds()
        # bounds of the finite data, known in advance or computed on demand
        self._dataBoundsHint = dataBounds
        self.informViewBoundsChanged()

        profiler('copy')

        if 'stepMode' in kwargs:
            self.opts['stepMode'] = kwargs['stepMode']

        if self.opts['stepMode'] == "center":
            if len(self.xData) != len(self.yData)+1:  ## allow difference of 1 for step mode plots
                raise Exception("len(X) must be len(Y)+1 since stepMode=True (got %s and %s)" % (self.xData.shape, self.yData.shape))
        else:
            if self.xData.shape != self.yData.shape:  ## allow difference of 1 for step mode plots
                raise Exception("X and Y arrays must be the same shape--got %s and %s." % (self.xData.shape, self.yData.shape))

        self.path = None
        self.fillPath = None
        self._fillPathList = None
        self._mouseShape = None
        self._lineSegmentsRendered = False

        if 'name' in kwargs:
            self.opts['name'] = kwargs['name']
        if 'connect' in kwargs:
            self.opts['connect'] = kwargs['connect']
        if 'pen' in kwargs:
            self.setPen(kwargs['pen'])
        if 'shadowPen' in kwargs:
            self.setShadowPen(kwargs['shadowPen'])
        if 'fillLevel' in kwargs:
            self.setFillLevel(kwargs['fillLevel'])
        if 'fillOutline' in kwargs:
            self.opts['fillOutline'] = kwargs['fillOutline']
        if 'brush' in kwargs:
            self.setBrush(kwargs['brush'])
        if 'antialias' in kwargs:
            self.opts['antialias'] = kwargs['antialias']
        if 'skipFiniteCheck' in kwargs:
            self.opts['skipFiniteCheck'] = kwargs['skipFiniteCheck']

        profiler('set')
        self.update()
        profiler('update')
        self.sigPlotChanged.emit(self)
        profiler('emit')

    @staticmethod
    def _generateStepModeData(stepMode, x, y, baseline):
        ## each value in the x/y arrays generates 2 points.
        if stepMode == "right":
            x2 = np.empty((len(x) + 1, 2), dtype=x.dtype)
            x2[:-1] = x[:, np.newaxis]
            x2[-1] = x2[-2]
        elif stepMode == "left":
            x2 = np.empty((len(x) + 1, 2), dtype=x.dtype)
            x2[1:] = x[:, np.newaxis]
            x2[0] = x2[1]
        elif stepMode in ("center", True):  ## support True for back-compat
            x2 = np.empty((len(x),2), dtype=x.dtype)
            x2[:] = x[:, np.newaxis]
        else:
            raise ValueError("Unsupported stepMode %s" % stepMode)
        if baseline is None:
            x = x2.reshape(x2.size)[1:-1]
            y2 = np.empty((len(y),2), dtype=y.dtype)
            y2[:] = y[:,np.newaxis]
            y = y2.reshape(y2.size)
        else:
            # if baseline is provided, add vertical lines to left/right ends
            x = x2.reshape(x2.size)
            y2 = np.empty((len(y)+2,2), dtype=y.dtype)
            y2[1:-1] = y[:,np.newaxis]
            y = y2.reshape(y2.size)[1:-1]
            y[[0, -1]] = baseline
        return x, y

    def generatePath(self, x, y):
        if self.opts['stepMode']:
            x, y = self._generateStepModeData(
                self.opts['stepMode'],
                x,
                y,
                baseline=self.opts['fillLevel']
            )

        return fn.arrayToQPath(
            x,
            y,
            connect=self.opts['connect'],
            finiteCheck=not self.opts['skipFiniteCheck']
        )

    def getPath(self):
        if self.path is None:
            x,y = self.getData()
            if x is None or len(x) == 0 or y is None or len(y) == 0:
                self.path = QtGui.QPainterPath()
            else:
                self.path = self.generatePath(*self.getData())
            self.fillPath = None
            self._fillPathList = None
            self._mouseShape = None

        return self.path

    def setSegmentedLineMode(self, mode):
        """
        Sets the mode that decides whether or not lines are drawn as segmented lines. Drawing lines
        as segmented lines is more performant than the standard drawing method with continuous
        lines.

        Parameters
        ----------
        mode : str
               ``'auto'`` (default) segmented lines are drawn if the pen's width > 1, pen style is a
               solid line, the pen color is opaque and anti-aliasing is not enabled.

               ``'on'`` lines are always drawn as segmented lines

               ``'off'`` lines are never drawn as segmented lines, i.e. the drawing
               method with continuous lines is used
        """
        if mode not in ('auto', 'on', 'off'):
            raise ValueError(f'segmentedLineMode must be "auto", "on" or "off", got {mode} instead')
        self.opts['segmentedLineMode'] = mode
        self.invalidateBounds()
        self.update()

    def _shouldUseDrawLineSegments(self, pen):
        mode = self.opts['segmentedLineMode']
        if mode in ('on',):
            return True
        if mode in ('off',):
            return False
        return (
            pen.widthF() > 1.0
            # non-solid pen styles need single polyline to be effective
            and pen.style() == QtCore.Qt.PenStyle.SolidLine
            # segmenting the curve slows gradient brushes, and is expected
            # to do the same for other patterns
            and pen.isSolid()   # pen.brush().style() == Qt.BrushStyle.SolidPattern
            # ends of adjacent line segments overlapping is visible when not opaque
            and pen.color().alphaF() == 1.0
            # anti-aliasing introduces transparent pixels and therefore also causes visible overlaps
            # for adjacent line segments
            and not self.opts['antialias']
        )

    def _getLineSegments(self):
        if not self._lineSegmentsRendered:
            x, y = self.getData()
            if self.opts['stepMode']:
                x, y = self._generateStepModeData(
                    self.opts['stepMode'],
                    x,
                    y,
                    baseline=self.opts['fillLevel']
                )

            self._lineSegments = arrayToLineSegments(
                x,
                y,
                connect=self.opts['connect'],
                finiteCheck=not self.opts['skipFiniteCheck'],
                out=self._lineSegments
            )

            self._lineSegmentsRendered = True

        return self._lineSegments.drawargs()

    def _getPolylineVertices(self) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Return the vertices of the curve when it is drawn as a single polyline.

        The curve is a single polyline when ``stepMode`` is off and ``connect`` is
        ``'all'``, or ``'finite'`` with finite data only. The vertices are those of the
        path built by :meth:`generatePath`. The result is cached until the data or the
        ``connect`` and ``skipFiniteCheck`` options change.

        Returns
        -------
        tuple of np.ndarray or None
            ``(x, y)`` coordinates of the vertices, or None if the curve is not a single
            polyline. Fewer than 2 vertices means that nothing is drawn.
        """
        x, y = self.xData, self.yData
        connect = self.opts['connect']
        if (
            x is None
            or y is None
            or self.opts['stepMode']
            or not isinstance(connect, str)
            or connect not in ('all', 'finite')
        ):
            return None

        skipFiniteCheck = bool(self.opts['skipFiniteCheck'])
        cache = self._vertexCache
        if cache is None or not cache.matches(x, y, connect, skipFiniteCheck):
            cache = self._vertexCache = _VertexCache(x, y, connect, skipFiniteCheck)
        if not cache.computed:
            if connect == 'all':
                cache.vertices = fn._arrayToQPath_all_vertices(
                    x, y, finiteCheck=not skipFiniteCheck)
            elif not (np.isfinite(x) & np.isfinite(y)).all():
                # 'finite' with non-finite values: several polylines
                cache.vertices = None
            elif skipFiniteCheck:
                # arrayToQPath adds the finite data as a single polygon
                cache.vertices = (x, y)
            else:
                # arrayToQPath delegates finite data to connect='all'
                cache.vertices = fn._arrayToQPath_all_vertices(x, y, finiteCheck=False)
            cache.computed = True
        return cache.vertices

    def _getPolyline(self) -> QtGui.QPolygonF:
        """
        Return the vertices of the curve as a ``QPolygonF``, filled once per data.

        The polygon is reused from one data update to the next, so that streaming data
        does not reallocate it on every frame. It is only valid when
        :meth:`_getPolylineVertices` does not return None.

        Returns
        -------
        QtGui.QPolygonF
            The polyline. Fewer than 2 points means that nothing is drawn.
        """
        cache = self._vertexCache
        if cache.polylineFilled:
            return self._polyline

        x, y = cache.vertices
        size = len(x)
        polyline = self._polyline
        if polyline is None or size < self._polylineMaxSize // 4:
            # create, or release the memory of a much longer former curve
            polyline = self._polyline = fn.create_qpolygonf(size)
            self._polylineMaxSize = size
        else:
            if hasattr(polyline, 'resize'):
                polyline.resize(size)
            else:
                polyline.fill(QtCore.QPointF(), size)
            self._polylineMaxSize = max(self._polylineMaxSize, size)
        memory = fn.ndarray_from_qpolygonf(polyline)
        memory[:, 0] = x
        memory[:, 1] = y
        cache.polylineFilled = True
        return polyline

    def _shouldUseDrawPolyline(self, painter: QtGui.QPainter, pen: QtGui.QPen,
                               antialias: bool) -> bool:
        """
        Tell whether ``pen`` can stroke the curve with ``QPainter.drawPolyline``.

        Drawing the vertices as a polyline avoids building and keeping the
        ``QPainterPath`` of the curve. It is only done when the result is
        pixel-identical to drawing the path.

        Parameters
        ----------
        painter : QtGui.QPainter
            The active painter.
        pen : QtGui.QPen
            The pen about to stroke the curve.
        antialias : bool
            Whether the painter antialiases.

        Returns
        -------
        bool
            True if the polyline can be drawn instead of the path.
        """
        if self._exportOpts is not False or self.opts['fillLevel'] is not None:
            return False
        shadowPen = self.opts['shadowPen']
        if shadowPen is not None and shadowPen.style() != QtCore.Qt.PenStyle.NoPen:
            return False
        # The raster engine strokes a polyline and the equivalent path alike, except
        # with its aliased "fast pens" (cosmetic, at most 1 px wide): for a polyline,
        # QCosmeticStroker starts the next segment from the start of a segment too
        # short to be drawn, which moves a few pixels of dense curves.
        if not (antialias or (pen.isCosmetic() and pen.widthF() > 1.0)):
            return False
        if painter.paintEngine().type() != QtGui.QPaintEngine.Type.Raster:
            return False
        # drawPath also fills the path with the brush of the painter
        if painter.brush().style() != QtCore.Qt.BrushStyle.NoBrush:
            return False
        return self._getPolylineVertices() is not None

    def _getClosingSegments(self):
        # this is only used for fillOutline
        # no point caching with so few elements generated
        segments = []
        if self.opts['fillLevel'] == 'enclosed':
            return segments

        baseline = self.opts['fillLevel']
        x, y = self.getData()
        lx, rx = x[[0, -1]]
        ly, ry = y[[0, -1]]

        if ry != baseline:
            segments.append(QtCore.QLineF(rx, ry, rx, baseline))
        segments.append(QtCore.QLineF(rx, baseline, lx, baseline))
        if ly != baseline:
            segments.append(QtCore.QLineF(lx, baseline, lx, ly))

        return segments

    def _getFillPath(self):
        if self.fillPath is not None:
            return self.fillPath

        path = QtGui.QPainterPath(self.getPath())
        self.fillPath = path
        if self.opts['fillLevel'] == 'enclosed':
            return path

        baseline = self.opts['fillLevel']
        x, y = self.getData()
        lx, rx = x[[0, -1]]
        ly, ry = y[[0, -1]]

        if ry != baseline:
            path.lineTo(rx, baseline)
        path.lineTo(lx, baseline)
        if ly != baseline:
            path.lineTo(lx, ly)

        return path

    def _shouldUseFillPathList(self, brush):
        connect = self.opts['connect']
        return (
            # not meaningful to fill disjoint lines
            isinstance(connect, str) and connect in ['all', 'finite']
            # guard against odd-ball argument 'enclosed'
            and isinstance(self.opts['fillLevel'], (int, float))
            and brush.style() == QtCore.Qt.BrushStyle.SolidPattern
        )

    def _getFillPathList(self, widget):
        if self._fillPathList is not None:
            return self._fillPathList

        x, y = self.getData()
        if self.opts['stepMode']:
            x, y = self._generateStepModeData(
                self.opts['stepMode'],
                x,
                y,
                # note that left/right vertical lines can be omitted here
                baseline=None
            )

        # Set suitable chunk size for current configuration:
        #   * Without OpenGL split in small chunks
        #   * With OpenGL split in rather big chunks
        #     Note: when OpenGL mode is enabled, we should normally be using the
        #     'paintGL' method, and should not even reach here.
        #     However, some oddball unsupported paint configurations may result in 'paintGL' not being used.
        # Values were found using 'PlotSpeedTest.py' example, see #2257.
        chunksize = 150 if not isinstance(widget, OpenGLHelpers.GraphicsViewGLWidget) else 5000

        connect_kind = self.opts['connect']
        if isinstance(connect_kind, np.ndarray):
            connect_kind = "array"

        fillLevel = self.opts['fillLevel']
        self._fillPathList = []
        sidx = []
        slen = []

        if connect_kind == "all":
            mask = np.isfinite(x) & np.isfinite(y)
            if not mask.all():
                # remove non-finite values
                x = x[mask]
                y = y[mask]
            sidx = [0]
            slen = [len(x)]

        elif connect_kind == "finite":
            isfinite = np.isfinite(x) & np.isfinite(y)
            nonfinite_locs = np.nonzero(~isfinite)[0]
            # pretend that there's a nonfinite before and after the array
            nonfinite_locs = np.concatenate(([-1], nonfinite_locs, [len(x)]))
            sidx = nonfinite_locs[:-1] + 1      # start index of segment
            slen = np.diff(nonfinite_locs) - 1  # length of segment

        for s, l in zip(sidx, slen):
            if l < 2:
                continue
            xchunk = x[s:s+l]
            ychunk = y[s:s+l]
            pathlist = self._construct_finite_segment_FillPathList(xchunk, ychunk, fillLevel, chunksize)
            self._fillPathList.extend(pathlist)

        return self._fillPathList

    def _construct_finite_segment_FillPathList(self, x, y, baseline, chunksize):
        paths = []
        offset = 0
        xybuf = np.empty((chunksize+3, 2))

        while offset < len(x) - 1:
            subx = x[offset:offset + chunksize]
            suby = y[offset:offset + chunksize]
            size = len(subx)
            xyview = xybuf[:size+3]
            xyview[:-3, 0] = subx
            xyview[:-3, 1] = suby
            xyview[-3:, 0] = subx[[-1, 0, 0]]
            xyview[-3:, 1] = [baseline, baseline, suby[0]]
            offset += size - 1  # last point is re-used for next chunk
            # data was either declared to be all-finite OR was sanitized
            path = fn._arrayToQPath_all(xyview[:, 0], xyview[:, 1], finiteCheck=False)
            paths.append(path)

        return paths

    @debug.warnOnException  ## raising an exception here causes crash
    def paint(self, p: QtGui.QPainter, opt: QtWidgets.QStyleOptionGraphicsItem,
              widget: QtWidgets.QWidget | None) -> None:
        """
        Draw the fill and the outline of the curve.

        The ``QPainterPath`` of the curve is only built when it is drawn: curves drawn
        as line segments (see :meth:`setSegmentedLineMode`) or, when the rendering is
        identical, as a polyline, do not need it.

        Parameters
        ----------
        p : QtGui.QPainter
            Painter, in item coordinates.
        opt : QtWidgets.QStyleOptionGraphicsItem
            Style options of the item.
        widget : QtWidgets.QWidget or None
            Widget being painted on, if any.
        """
        profiler = debug.Profiler()
        if self.xData is None or len(self.xData) == 0:
            return

        # opengl fill mode supports filling to a fillLevel
        # for connect="all" and connect="finite" only.
        opengl_supported_fill = (
            self.opts['fillLevel'] is None  # not filling is always supported
            or (
                isinstance(self.opts['fillLevel'], (int, float))
                and isinstance(self.opts['connect'], str)
                and self.opts['connect'] in ['all', 'finite']
                and not self.opts['fillOutline']
            )
        )

        if (
            isinstance(widget, OpenGLHelpers.GraphicsViewGLWidget)
            and opengl_supported_fill
            and not self.opts['stepMode']
        ):
            if self.glstate is None:
                self.glstate = OpenGLState(widget)
                self.sigPlotChanged.connect(self.glstate.verticesChanged)
            p.beginNativePainting()
            try:
                self.paintGL(widget)
            finally:
                p.endNativePainting()
            return

        if self._exportOpts is not False:
            aa = self._exportOpts.get('antialias', True)
        else:
            aa = self.opts['antialias']

        p.setRenderHint(p.RenderHint.Antialiasing, aa)

        cmode = self.opts['compositionMode']
        if cmode is not None:
            p.setCompositionMode(cmode)

        brush = self.opts['brush']
        do_fill = (
            self.opts['fillLevel'] is not None
            and not (brush is None or brush.style() == QtCore.Qt.BrushStyle.NoBrush)
        )
        do_fill_outline = do_fill and self.opts['fillOutline']

        path_transform = None
        if (
            self._exportOpts is not False
            and self._exportOpts.get('svgCoordinatesOffset', False)
        ):
            center = self.boundingRect().center()
            if np.isfinite(center.x()) and np.isfinite(center.y()):
                path_transform = QtGui.QTransform.fromTranslate(
                    -center.x(), -center.y()
                )
                p.save()
                p.translate(center)

        try:
            if do_fill:
                if (
                    path_transform is None
                    and self._shouldUseFillPathList(brush)
                ):
                    paths = self._getFillPathList(widget)
                else:
                    # The fill path list is a painting throughput optimization.
                    # SVG export prefers the single path while applying a
                    # path-local coordinate offset for precision.
                    paths = [self._getFillPath()]

                if path_transform is not None:
                    paths = [path_transform.map(path) for path in paths]

                profiler('generate fill path')
                for path in paths:
                    p.fillPath(path, brush)
                profiler('draw fill path')

            for pen_kind in ['shadowPen', 'pen']:
                pen = self.opts[pen_kind]
                if pen is None or pen.style() == QtCore.Qt.PenStyle.NoPen:
                    continue
                p.setPen(pen)

                if (
                    path_transform is None
                    and self._shouldUseDrawLineSegments(pen)
                ):
                    p.drawLines(*self._getLineSegments())
                    if do_fill_outline:
                        p.drawLines(self._getClosingSegments())
                elif self._shouldUseDrawPolyline(p, pen, aa):
                    polyline = self._getPolyline()
                    if len(polyline) >= 2:
                        p.drawPolyline(polyline)
                else:
                    if do_fill_outline:
                        path = self._getFillPath()
                    else:
                        path = self.getPath()

                    if path_transform is not None:
                        path = path_transform.map(path)
                    p.drawPath(path)
        finally:
            if path_transform is not None:
                p.restore()

        profiler('drawPath')

    def paintGL(self, widget):
        if (view := self.getViewBox()) is None:
            return

        x, y = self.getData()
        num_pts = len(x)
        valid_pts = num_pts

        # minimum 2 pts to draw anything
        if num_pts < 2:
            return

        glstate = self.glstate
        glstate.setup(widget.context())
        glf = widget.getFunctions()
        program = widget.retrieveProgram("PlotCurveItem")

        # OpenGL only sees the float32 version of our data, and this may cause
        # precision issues. To mitigate this, we shift the origin of our data
        # to the center of its bounds.
        # Note that xc, yc are double precision Python floats. Subtracting them
        # from the x, y ndarrays will automatically upcast the latter to double
        # precision.
        if glstate.render_cache is None:
            # the origin point is calculated once per data change.
            # once the data is uploaded, the origin point is fixed.
            center = self.boundingRect().center()
            xc, yc = center.x(), center.y()
        else:
            xc, yc, *_ = glstate.render_cache

        proj = QtGui.QMatrix4x4()
        proj.ortho(widget.rect())
        tr = self.sceneTransform()
        tr.translate(xc, yc)
        mvp = proj * QtGui.QMatrix4x4(tr)

        vbo_nbytes_needed = num_pts * 2 * 4

        connect_kind = self.opts["connect"]
        if isinstance(connect_kind, np.ndarray):
            connect_kind = "array"
            vbo_nbytes_needed = ((num_pts-1) * 2) * 2 * 4

        # filling is only supported for 'all' and 'finite'.
        # it requires an additional 2 * num_pts of storage
        # to create the triangle strip to be filled.
        fillLevel = None
        if connect_kind in ['all', 'finite']:
            if isinstance(self.opts['fillLevel'], (int, float)):
                fillLevel = float(self.opts['fillLevel'])
                vbo_nbytes_needed += (2 * num_pts) * 2 * 4

        # resize (and invalidate) gpu buffers if needed.
        # a reallocation can only occur together with a change in data.
        # i.e. reallocation ==> change in data (render_cache is None)
        if vbo_nbytes_needed != glstate.vbo_nbytes:
            glstate.m_vbo.bind()
            glstate.m_vbo.allocate(vbo_nbytes_needed)
            glstate.m_vbo.release()
            glstate.vbo_nbytes = vbo_nbytes_needed

        if glstate.render_cache is None:
            buf = None

            if connect_kind == "pairs":
                glstate.render_cache = (xc, yc, valid_pts,)

                buf = np.empty((valid_pts, 2), dtype=np.float32)
                pos = buf
                pos[:, 0] = x - xc
                pos[:, 1] = y - yc

            elif connect_kind == "all":
                if not self.opts["skipFiniteCheck"]:
                    isfinite = np.isfinite(y)
                    if x.dtype.kind == 'f':
                        isfinite &= np.isfinite(x)
                    valid_pts = np.sum(isfinite)
                glstate.render_cache = (xc, yc, valid_pts,)

                fill_pts = 0 if fillLevel is None else 2 * valid_pts
                buf = np.empty((valid_pts + fill_pts, 2), dtype=np.float32)
                pos = buf[:valid_pts, :]
                if valid_pts == num_pts:
                    pos[:, 0] = x - xc
                    pos[:, 1] = y - yc
                else:
                    pos[:, 0] = x[isfinite] - xc
                    pos[:, 1] = y[isfinite] - yc

                if fill_pts:
                    fillpos = buf[valid_pts:, :]
                    fillpos[0::2, 0] = pos[:, 0]
                    fillpos[0::2, 1] = pos[:, 1]
                    fillpos[1::2, 0] = pos[:, 0]
                    fillpos[1::2, 1] = fillLevel - yc

            elif connect_kind == "finite":
                isfinite = np.isfinite(y)
                if x.dtype.kind == 'f':
                    isfinite &= np.isfinite(x)
                nonfinite_locs = np.nonzero(~isfinite)[0]
                # pretend that there's a nonfinite before and after the array
                nonfinite_locs = np.concatenate(([-1], nonfinite_locs, [num_pts]))
                sidx = nonfinite_locs[:-1] + 1      # start index of segment
                slen = np.diff(nonfinite_locs) - 1  # length of segment
                mask = slen >= 2
                sidx = sidx[mask].tolist()
                slen = slen[mask].tolist()
                glstate.render_cache = (xc, yc, valid_pts, sidx, slen)

                fill_pts = 0 if fillLevel is None else 2 * valid_pts
                buf = np.empty((valid_pts + fill_pts, 2), dtype=np.float32)
                pos = buf[:valid_pts, :]
                pos[:, 0] = x - xc
                pos[:, 1] = y - yc

                if fill_pts:
                    fillpos = buf[valid_pts:, :]
                    fillpos[0::2, 0] = pos[:, 0]
                    fillpos[0::2, 1] = pos[:, 1]
                    fillpos[1::2, 0] = pos[:, 0]
                    fillpos[1::2, 1] = fillLevel - yc

            elif connect_kind == "array":
                mask = np.asarray(self.opts["connect"], dtype=bool)[:num_pts-1]
                valid_pts = 2 * np.sum(mask)
                glstate.render_cache = (xc, yc, valid_pts,)

                buf = np.empty((valid_pts, 2), dtype=np.float32)
                pos = buf
                xshift = x - xc
                yshift = y - yc
                pos[0::2, 0] = xshift[:-1][mask]
                pos[1::2, 0] = xshift[1:][mask]
                pos[0::2, 1] = yshift[:-1][mask]
                pos[1::2, 1] = yshift[1:][mask]

            if buf is not None:
                glstate.m_vbo.bind()
                glstate.m_vbo.write(0, buf, buf.nbytes)
                glstate.m_vbo.release()


        widget.setViewboxClip(view)

        glstate.m_vao.bind()
        program.bind()
        program.setUniformValue("u_mvp", mvp)

        # filling occurs first so that the curve outline gets painted over it.
        for brush in [self.opts["brush"]]:
            if fillLevel is None:
                continue
            if brush is None or brush.style() == QtCore.Qt.BrushStyle.NoBrush:
                continue
            program.setUniformValue("u_color", brush.color())

            glf.glEnable(GLC.GL_BLEND)
            glf.glBlendFuncSeparate(GLC.GL_SRC_ALPHA, GLC.GL_ONE_MINUS_SRC_ALPHA, 1, GLC.GL_ONE_MINUS_SRC_ALPHA)

            if connect_kind == 'all':
                *_, valid_pts = glstate.render_cache
                glf.glDrawArrays(GLC.GL_TRIANGLE_STRIP, valid_pts, 2 * valid_pts)
            elif connect_kind == 'finite':
                *_, valid_pts, sidx, slen = glstate.render_cache
                for s, l in zip(sidx, slen):
                    glf.glDrawArrays(GLC.GL_TRIANGLE_STRIP, valid_pts + 2 * s, 2 * l)

            glf.glDisable(GLC.GL_BLEND)

        # enable antialiasing if requested
        if self._exportOpts is not False:
            aa = self._exportOpts.get('antialias', True)
        else:
            aa = self.opts['antialias']
        if aa:
            glf.glEnable(GLC.GL_LINE_SMOOTH)
            glf.glEnable(GLC.GL_BLEND)
            glf.glBlendFuncSeparate(GLC.GL_SRC_ALPHA, GLC.GL_ONE_MINUS_SRC_ALPHA, 1, GLC.GL_ONE_MINUS_SRC_ALPHA)
            glf.glHint(GLC.GL_LINE_SMOOTH_HINT, GLC.GL_NICEST)
        else:
            glf.glDisable(GLC.GL_LINE_SMOOTH)

        for pen_kind in ["shadowPen", "pen"]:
            pen = self.opts[pen_kind]
            if pen is None or pen.style() == QtCore.Qt.PenStyle.NoPen:
                continue
            width = pen.widthF()
            if pen.isCosmetic() and width < 1:
                width = 1

            glf.glLineWidth(width)
            program.setUniformValue("u_color", pen.color())

            match connect_kind:
                case "pairs" | "array":
                    *_, valid_pts = glstate.render_cache
                    glf.glDrawArrays(GLC.GL_LINES, 0, valid_pts)
                case "all":
                    *_, valid_pts = glstate.render_cache
                    glf.glDrawArrays(GLC.GL_LINE_STRIP, 0, valid_pts)
                case "finite":
                    *_, sidx, slen = glstate.render_cache
                    if hasattr(glf, "glMultiDrawArrays") and not glstate.context.isOpenGLES():
                        glf.glMultiDrawArrays(GLC.GL_LINE_STRIP, sidx, slen, len(sidx))
                    else:
                        # PyQt{5,6} didn't include glMultiDrawArrays
                        for s, l in zip(sidx, slen):
                            glf.glDrawArrays(GLC.GL_LINE_STRIP, s, l)

        glstate.m_vao.release()

    def clear(self) -> None:
        """
        Remove the data and all derived caches (paths, segments, bounds).
        """
        self.xData = None  ## raw values
        self.yData = None
        # ((xmin, xmax), (ymin, ymax)) of the finite data, passed by PlotDataItem
        self._dataBoundsHint = None
        # read-only np.arange used as x when only y is given, see _arangeBuffer
        self._indexBuffer = None
        self._lineSegments = None
        self._lineSegmentsRendered = False
        self.path = None
        self.fillPath = None
        self._fillPathList = None
        self._mouseShape = None
        self._mouseBounds = None
        self._boundsCache = [None, None]
        #del self.xData, self.yData, self.xDisp, self.yDisp, self.path

    def mouseShape(self):
        """
        Return a QPainterPath representing the clickable shape of the curve

        """
        if self._mouseShape is None:
            view = self.getViewBox()
            if view is None:
                return QtGui.QPainterPath()
            stroker = QtGui.QPainterPathStroker()
            path = self.getPath()
            path = self.mapToItem(view, path)
            stroker.setWidth(self.opts['mouseWidth'])
            mousePath = stroker.createStroke(path)
            self._mouseShape = self.mapFromItem(view, mousePath)
        return self._mouseShape

    def mouseClickEvent(self, ev):
        if not self.clickable or ev.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        if self.mouseShape().contains(ev.pos()):
            ev.accept()
            self.sigClicked.emit(self, ev)



class ROIPlotItem(PlotCurveItem):
    """Plot curve that monitors an ROI and image for changes to automatically replot."""
    def __init__(self, roi, data, img, axes=(0,1), xVals=None, color=None):
        self.roi = roi
        self.roiData = data
        self.roiImg = img
        self.axes = axes
        self.xVals = xVals
        PlotCurveItem.__init__(self, self.getRoiData(), x=self.xVals, color=color)
        #roi.connect(roi, QtCore.SIGNAL('regionChanged'), self.roiChangedEvent)
        roi.sigRegionChanged.connect(self.roiChangedEvent)
        #self.roiChangedEvent()

    def getRoiData(self):
        d = self.roi.getArrayRegion(self.roiData, self.roiImg, axes=self.axes)
        if d is None:
            return
        while d.ndim > 1:
            d = d.mean(axis=1)
        return d

    def roiChangedEvent(self):
        d = self.getRoiData()
        self.updateData(d, self.xVals)


def _arangeBuffer(buffer: np.ndarray | None, n: int) -> np.ndarray:
    """
    Get a read-only buffer starting with ``np.arange(n)``.

    The buffer is reused while it is long enough, and otherwise replaced by one with
    some spare capacity, so that streaming data without `x` values does not generate
    a full index for each update. It is read-only, so that the views of it used as
    `x` data cannot alter later uses.

    Parameters
    ----------
    buffer : np.ndarray or None
        The buffer returned by the previous call, if any.
    n : int
        Number of index values needed.

    Returns
    -------
    np.ndarray
        A read-only array whose first `n` values are ``0, 1, ..., n - 1``, with the
        dtype of ``np.arange(n)``.
    """
    if buffer is not None and len(buffer) >= n:
        return buffer
    capacity = 0 if buffer is None else len(buffer)
    buffer = np.arange(max(n, capacity + capacity // 4))
    buffer.flags.writeable = False
    return buffer
