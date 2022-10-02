import seaborn.objects
from seaborn._core.plot import Plot
from seaborn._core.moves import Move
from seaborn._core.scales import Scale
from seaborn._marks.base import Mark
from seaborn._stats.base import Stat


def test_objects_namespace():

    for name in dir(seaborn.objects):
        if not name.startswith("__"):
            obj = getattr(seaborn.objects, name)
            assert issubclass(obj, (Plot, Mark, Stat, Move, Scale))
