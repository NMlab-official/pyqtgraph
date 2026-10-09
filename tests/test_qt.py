import importlib
import os

import pytest

import pyqtgraph as pg

app = pg.mkQApp()

def test_isQObjectAlive():
    o1 = pg.QtCore.QObject()
    o2 = pg.QtCore.QObject()
    o2.setParent(o1)
    del o1
    assert not pg.Qt.isQObjectAlive(o2)

def test_loadUiType():
    path = os.path.dirname(__file__)
    formClass, baseClass = pg.Qt.loadUiType(os.path.join(path, 'uictest.ui'))
    w = baseClass()
    ui = formClass()
    ui.setupUi(w)
    w.show()
    app.processEvents()


@pytest.mark.skipif(not pg.Qt.QT_LIB.startswith('PySide'),
                    reason="the fallback compiles .ui files for PySide only")
def test_loadUiType_falls_back_to_uic(monkeypatch):
    # some PySide6 versions do not find their pyside6-uic script, and return None
    QtUiTools = importlib.import_module(pg.Qt.QT_LIB + '.QtUiTools')
    monkeypatch.setattr(QtUiTools, 'loadUiType', lambda uiFile: None)
    path = os.path.dirname(__file__)
    formClass, baseClass = pg.Qt.loadUiType(os.path.join(path, 'uictest.ui'))
    assert formClass.__name__ == 'Ui_Form'
    assert baseClass is pg.QtWidgets.QWidget
    w = baseClass()
    ui = formClass()
    ui.setupUi(w)
    assert isinstance(ui.widget, pg.PlotWidget)
    w.close()
