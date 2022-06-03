import inspect
from seaborn._decorators import share_init_params_with_map


def test_share_init_params_with_map():

    @share_init_params_with_map
    class Thingie:

        def map(cls, *args, **kwargs):
            return cls(*args, **kwargs)

        def __init__(self, a, b=1):
            """Make a new thingie."""
            self.a = a
            self.b = b

    thingie = Thingie.map(1, b=2)
    assert thingie.a == 1
    assert thingie.b == 2

    assert "a" in inspect.signature(Thingie.map).parameters
    assert "b" in inspect.signature(Thingie.map).parameters

    assert Thingie.map.__doc__ == Thingie.__init__.__doc__
