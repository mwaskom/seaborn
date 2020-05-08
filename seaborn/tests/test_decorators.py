import pytest
from .._decorators import _deprecate_positional_args


# This test was adapted from scikit-learn
# github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/tests/test_validation.py
def test_deprecate_positional_args_warns_for_function():

    @_deprecate_positional_args
    def f1(a, b, *, c=1, d=1):
        return a, b, c, d

    with pytest.warns(FutureWarning,
                      match=r"Pass c=3 as keyword arg\."):
        assert f1(1, 2, 3) == (1, 2, 3, 1)

    with pytest.warns(FutureWarning,
                      match=r"Pass c=3, d=4 as keyword args\."):
        assert f1(1, 2, 3, 4) == (1, 2, 3, 4)

    @_deprecate_positional_args
    def f2(a=1, *, b=1, c=1, d=1):
        return a, b, c, d

    with pytest.warns(FutureWarning,
                      match=r"Pass b=2 as keyword arg\."):
        assert f2(1, 2) == (1, 2, 1, 1)

    # The * is placed before a keyword only argument without a default value
    @_deprecate_positional_args
    def f3(a, *, b, c=1, d=1):
        return a, b, c, d

    with pytest.warns(FutureWarning,
                      match=r"Pass b=2 as keyword arg\."):
        assert f3(1, 2) == (1, 2, 1, 1)


def test_deprecate_positional_args_warns_for_class():

    class A1:
        @_deprecate_positional_args
        def __init__(self, a, b, *, c=1, d=1):
            self.a = a, b, c, d

    with pytest.warns(FutureWarning,
                      match=r"Pass c=3 as keyword arg\."):
        assert A1(1, 2, 3).a == (1, 2, 3, 1)

    with pytest.warns(FutureWarning,
                      match=r"Pass c=3, d=4 as keyword args\."):
        assert A1(1, 2, 3, 4).a == (1, 2, 3, 4)

    class A2:
        @_deprecate_positional_args
        def __init__(self, a=1, b=1, *, c=1, d=1):
            self.a = a, b, c, d

    with pytest.warns(FutureWarning,
                      match=r"Pass c=3 as keyword arg\."):
        assert A2(1, 2, 3).a == (1, 2, 3, 1)

    with pytest.warns(FutureWarning,
                      match=r"Pass c=3, d=4 as keyword args\."):
        assert A2(1, 2, 3, 4).a == (1, 2, 3, 4)
