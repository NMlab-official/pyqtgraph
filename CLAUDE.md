# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Fork de pyqtgraph (`NMlab-official/pyqtgraph`, branche `master`) orienté performances pour des
applications de marchés financiers. Le travail en cours suit `PERFORMANCE_PLAN.md` (voir plus bas).

## Commandes

```bash
# Environnement : venv `.venv` à la racine (ignoré par git). numpy est la seule dépendance
# obligatoire ; une liaison Qt est requise. Dans la suite, `python` désigne le Python du venv.
python -m venv .venv
.venv/Scripts/python.exe -m pip install numpy scipy pyqt6 pytest pytest-qt pytest-xdist mypy -e .
.venv/Scripts/Activate.ps1                # activation PowerShell ; bash : source .venv/Scripts/activate

# Qt sans écran (certains tests ne tournent qu'avec offscreen ; sous Windows, voir plus bas)
export QT_QPA_PLATFORM=offscreen          # PowerShell : $env:QT_QPA_PLATFORM='offscreen'

# Tests
python -m pytest tests -q -p no:cacheprovider
python -m pytest tests/graphicsItems/test_PlotDataItem.py::test_name -q   # un seul test
python -m pytest pyqtgraph/examples -n 2       # chaque exemple tourne ~1 s, réussi s'il ne lève rien
python test.py --pyside6 tests/...             # forcer une liaison (--pyqt5 / --pyqt6 / --pyside6)
PYQTGRAPH_QT_LIB=PyQt6 python -m pytest ...    # idem par variable d'environnement
tox -e py313-pyqt6                             # matrice liaisons × Python (voir tox.ini)

# Tests d'images : revoir / accepter une image de référence (tests/images/)
PYQTGRAPH_AUDIT=1 python -m pytest tests/graphicsItems/test_ImageItem.py

# Benchmarks
python benchmarks/scenarios.py                 # scénarios S01..S15 (offscreen par défaut)
python benchmarks/scenarios.py S05 S06 --full  # sélection ; --full ajoute les tailles 1e7, --numba
asv run                                        # benchmarks asv de benchmarks/*.py (setup / time_*)

# Qualité, docs, exemples
pre-commit run --all-files    # isort, pycln, rstcheck, fins de ligne LF…
mypy                          # lancé par la CI (config dans pyproject.toml)
make -C doc html              # pip install -r doc/requirements.txt ; graphviz (`dot`) dans le PATH
python -m pyqtgraph.examples  # application de démonstration
```

- Toujours passer par `.venv` (`.venv/Scripts/python.exe`, ou `python` une fois le venv activé),
  lancé **depuis la racine du dépôt**. Le Python système (3.14) a un pyqtgraph 0.14 non éditable
  dans le `site-packages` utilisateur, qui masque le code local, et il n'a ni pytest-qt ni
  pytest-xdist.
- `.venv/Lib/site-packages/sitecustomize.py`, propre au venv et non versionné, fait
  `os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")`. Les wheels PyQt6 n'embarquent
  plus de polices : en `offscreen`, l'avertissement `QFontDatabase` ferait échouer les tests. À
  recréer avec le venv.
- Sous Windows, lancer les tests sans `offscreen` : avec, 6 tests (dockarea, exporters,
  imageview, histogramlutwidget) échouent sur l'avertissement Qt
  `This plugin does not support propagateSizeHints()`.
- `tests/test_reload.py` a besoin des `.pyc` : il échoue si `PYTHONDONTWRITEBYTECODE=1`, valeur
  présente dans l'environnement des sessions Claude Code (`Remove-Item Env:PYTHONDONTWRITEBYTECODE`).
- `filterwarnings = "error"` (pyproject) : tout avertissement Python fait échouer un test, et
  `qt_log_level_fail = "WARNING"` fait de même pour les messages Qt (pytest-qt).

## Architecture

- **Couche Qt (`pyqtgraph/Qt/`)** : choisit la liaison (`PYQTGRAPH_QT_LIB`, sinon déjà importée,
  sinon ordre PyQt6, PySide6, PyQt5) et la ré-exporte. Toujours importer `QtCore/QtGui/QtWidgets`
  depuis `pyqtgraph.Qt` (en relatif : `from ..Qt import …`), jamais depuis PyQt6/PySide6.
  Enums toujours entièrement qualifiés (`QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations`),
  car PyQt6 les impose. L'accès mémoire bas niveau aux objets Qt passe par `Qt/internals.py`
  (`PrimitiveArray`, `get_qpainterpath_element_array`) : pas de code propre à sip/shiboken ailleurs.
- **Pile de tracé 2D** : `PlotWidget` (`widgets/`, un `GraphicsView`) enveloppe un `PlotItem`
  (`graphicsItems/PlotItem/`), qui possède un `ViewBox` et des `AxisItem`. `PlotItem.plot()` crée un
  `PlotDataItem`, qui prépare les données (sous-échantillonnage, `clipToView`, log, `appendData`
  incrémental, pyramide min/max de `_MinMaxPyramid.py`) puis délègue le dessin à un `PlotCurveItem`
  (ligne) et un `ScatterPlotItem` (symboles). Le `ViewBox` gère la plage visible, l'auto-range via
  `dataBounds()`/`childrenBounds()` de ses enfants et le lien entre vues.
- **Base des éléments** : `GraphicsItem` (mixin : `viewRangeChanged`, `viewTransformChanged`,
  `pixelVectors`…), `GraphicsObject`, `GraphicsWidget`. `GraphicsScene/` remplace les événements
  souris Qt par les siens (`MouseClickEvent`, `MouseDragEvent`, `HoverEvent`).
- **Fonctions** : `functions.py` (`mkPen`, `mkBrush`, `mkColor`, `arrayToQPath`, `makeARGB`…) ;
  variantes accélérées dans `functions_numba.py` (via `util/numba_helper.getNumbaFunctions()`) et
  cupy (`util/cupy_helper.getCupy()`), actives seulement avec `useNumba` / `useCupy`.
- **Configuration globale** : `CONFIG_OPTIONS` et `setConfigOption()` dans `pyqtgraph/__init__.py`.
  Une nouvelle option : valeur par défaut + validation dans `setConfigOption` + entrée dans
  `doc/source/api_reference/config_options.rst`.
- **Autres modules** indépendants de la pile de tracé : `opengl/` (3D), `parametertree/`,
  `dockarea/`, `flowchart/`, `multiprocess/` (`RemoteGraphicsView`), `exporters/`, `jupyter/`.

### Ajouter un élément graphique

1. Module dans `pyqtgraph/graphicsItems/` avec `__all__`.
2. `from .graphicsItems.X import *` dans `pyqtgraph/__init__.py`.
3. Page `.rst` dans `doc/source/api_reference/graphicsItems/` + entrée du `toctree` de `index.rst`.
4. Exemple dans `pyqtgraph/examples/`, enregistré dans `examples_` de `examples/utils.py`
   (sinon ni l'application de démonstration ni `test_examples.py` ne le voient).

### Fichiers `.ui`

`*Template.ui` est compilé en `*Template_generic.py` (ne pas modifier à la main, exclu d'isort).
`tools/rebuildUi.py` écrit sa sortie dans le `.ui` lui-même : compiler plutôt avec
`pyuic6 X.ui -o X_generic.py`, puis remplacer l'import PyQt6 par `from ...Qt import QtCore, QtGui, QtWidgets`.

## Tests

- `tests/` reflète l'arborescence du paquet. Tests d'images : `tests/image_testing.py`
  (`assertImageApproved`), références dans `tests/images/`.
- Tests de performance (`*_perf.py`) : **jamais d'assertion sur un temps**. Ils comptent des appels
  (`paint`, `setData`, `dataBounds`…) avec `tests/perf_helpers.py` (`count_calls`, `CallCounter`,
  `show_and_wait`, `paints_per_update`). Les temps se mesurent dans `benchmarks/`.
- PySide6 met en cache, par objet, la surcharge Python d'une méthode virtuelle (`paint`…) trouvée au
  premier appel : un compteur posé après le premier rendu ne voit rien (voir `perf_helpers.py`).
- Attendre l'exposition de la fenêtre avant de mesurer ou comparer un rendu (`show_and_wait`).
- Quand un changement touche au rendu, il doit rester identique au pixel près : comparer une
  `QImage` rendue avant/après ou s'appuyer sur les tests d'images existants.

## PERFORMANCE_PLAN.md

Plan de travail en français destiné aux agents : tableau de bord des tâches T0.x à T4.x (statut à
mettre à jour à chaque tâche), scénarios S01..S15 de `benchmarks/scenarios.py`, règles communes
(§1) et pistes écartées à ne pas proposer (§8). Points clés :

- Correctif minimal, périmètre de la fiche uniquement ; aucun changement visible d'API, de valeur
  par défaut ou de rendu sauf si la fiche l'autorise.
- Mesurer avant/après avec le scénario indiqué ; écrire d'abord un test qui échoue sans le correctif.
- numpy reste la seule dépendance obligatoire ; numba et cupy restent optionnels.
- Le plan prescrit « une tâche = une branche = une PR » : dans ce dépôt, on commite directement sur
  la branche courante (`master`) sans créer de branche.

## Conventions

- API publique en camelCase, membres privés préfixés `_`, options dans le dictionnaire `opts`,
  signaux nommés `sig*`. Nouvelles classes : attributs privés exposés par des propriétés, API de
  style pyqtgraph (`setData()`, `setOpts()`, `dataBounds()`, `boundingRect()`, `paint()`).
- Indications de type complètes et docstrings numpydoc (`Parameters`, `Returns`) sur toute
  fonction nouvelle ou modifiée. Code, commentaires et docstrings en anglais.
- Python ≥ 3.12, Qt 5.15 ou ≥ 6.8 ; PyQt5, PyQt6 et PySide6 doivent tous fonctionner.
- Messages de commit en anglais, préfixés par le composant ou la tâche :
  `AxisItem: …`, `Tests: …`, `T4.4: …`.
- Pull requests rédigées en anglais, même quand la conversation est en français : titre préfixé
  comme les commits (`PlotDataItem: …`), description en sections `## What`, `## Why`,
  `## Notes for the review` (mesures, tests lancés, changements de comportement).
