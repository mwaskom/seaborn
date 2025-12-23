import numpy as np
import pandas as pd

import pytest


@pytest.fixture(autouse=True)
def close_figs():
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


@pytest.fixture(autouse=True)
def random_seed():
    seed = sum(map(ord, "seaborn random global"))
    np.random.seed(seed)


@pytest.fixture()
def rng():
    seed = sum(map(ord, "seaborn random object"))
    return np.random.RandomState(seed)


@pytest.fixture
def wide_df(rng):

    columns = list("abc")
    index = pd.RangeIndex(10, 50, 2, name="wide_index")
    values = rng.normal(size=(len(index), len(columns)))
    return pd.DataFrame(values, index=index, columns=columns)


@pytest.fixture
def wide_array(wide_df):

    return wide_df.to_numpy()


# TODO s/flat/thin?
@pytest.fixture
def flat_series(rng):

    index = pd.RangeIndex(10, 30, name="t")
    return pd.Series(rng.normal(size=20), index, name="s")


@pytest.fixture
def flat_array(flat_series):

    return flat_series.to_numpy()


@pytest.fixture
def flat_list(flat_series):

    return flat_series.to_list()


@pytest.fixture(params=["series", "array", "list"])
def flat_data(rng, request):

    index = pd.RangeIndex(10, 30, name="t")
    series = pd.Series(rng.normal(size=20), index, name="s")
    if request.param == "series":
        data = series
    elif request.param == "array":
        data = series.to_numpy()
    elif request.param == "list":
        data = series.to_list()
    return data


@pytest.fixture
def wide_list_of_series(rng):

    return [pd.Series(rng.normal(size=20), np.arange(20), name="a"),
            pd.Series(rng.normal(size=10), np.arange(5, 15), name="b")]


@pytest.fixture
def wide_list_of_arrays(wide_list_of_series):

    return [s.to_numpy() for s in wide_list_of_series]


@pytest.fixture
def wide_list_of_lists(wide_list_of_series):

    return [s.to_list() for s in wide_list_of_series]


@pytest.fixture
def wide_dict_of_series(wide_list_of_series):

    return {s.name: s for s in wide_list_of_series}


@pytest.fixture
def wide_dict_of_arrays(wide_list_of_series):

    return {s.name: s.to_numpy() for s in wide_list_of_series}


@pytest.fixture
def wide_dict_of_lists(wide_list_of_series):

    return {s.name: s.to_list() for s in wide_list_of_series}


@pytest.fixture
def long_df(rng):

    n = 100
    df = pd.DataFrame(dict(
        x=rng.uniform(0, 20, n).round().astype("int"),
        y=rng.normal(size=n),
        z=rng.lognormal(size=n),
        a=rng.choice(list("abc"), n),
        b=rng.choice(list("mnop"), n),
        c=rng.choice([0, 1], n, [.3, .7]),
        d=rng.choice(np.arange("2004-07-30", "2007-07-30", dtype="datetime64[Y]"), n),
        t=rng.choice(np.arange("2004-07-30", "2004-07-31", dtype="datetime64[m]"), n),
        s=rng.choice([2, 4, 8], n),
        f=rng.choice([0.2, 0.3], n),
    ))

    a_cat = df["a"].astype("category")
    new_categories = np.roll(a_cat.cat.categories, 1)
    df["a_cat"] = a_cat.cat.reorder_categories(new_categories)

    df["s_cat"] = df["s"].astype("category")
    df["s_str"] = df["s"].astype(str)

    return df


@pytest.fixture
def long_dict(long_df):

    return long_df.to_dict()


@pytest.fixture
def repeated_df(rng):

    n = 100
    return pd.DataFrame(dict(
        x=np.tile(np.arange(n // 2), 2),
        y=rng.normal(size=n),
        a=rng.choice(list("abc"), n),
        u=np.repeat(np.arange(2), n // 2),
    ))


@pytest.fixture
def null_df(rng, long_df):

    df = long_df.copy()
    for col in df:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype(float)
        idx = rng.permutation(df.index)[:10]
        df.loc[idx, col] = np.nan
    return df


@pytest.fixture
def object_df(rng, long_df):

    df = long_df.copy()
    # objectify numeric columns
    for col in ["c", "s", "f"]:
        df[col] = df[col].astype(object)
    return df


@pytest.fixture
def null_series(flat_series):

    return pd.Series(index=flat_series.index, dtype='float64')


class MockConvertibleDataFrame:
    # Mock object that is not a pandas.DataFrame but that can
    # be converted to one via the DataFrame exchange protocol
    def __init__(self, data):
        self._data = data

    def to_pandas(self, *args, **kwargs):
        if self._data is None:
            raise ValueError("Cannot convert to pandas")
        return self._data


@pytest.fixture
def mock_long_df(long_df):

    return MockConvertibleDataFrame(long_df)
