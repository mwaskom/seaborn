import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pytest


@pytest.fixture(autouse=True)
def close_figs():
    yield
    plt.close("all")


@pytest.fixture(autouse=True)
def random_seed():
    np.random.seed(47)


@pytest.fixture
def wide_df():

    columns = list("abc")
    index = pd.Int64Index(np.arange(10, 50, 2), name="wide_index")
    values = np.random.randn(len(index), len(columns))
    return pd.DataFrame(values, index=index, columns=columns)


@pytest.fixture
def wide_array(wide_df):

    # Requires panads >= 0.24
    # return wide_df.to_numpy()
    return np.asarray(wide_df)


@pytest.fixture
def flat_series():

    index = pd.Int64Index(np.arange(10, 30), name="t")
    return pd.Series(np.random.randn(20), index, name="s")


@pytest.fixture
def flat_array(flat_series):

    # Requires panads >= 0.24
    # return flat_series.to_numpy()
    return np.asarray(flat_series)


@pytest.fixture
def flat_list(flat_series):

    # Requires panads >= 0.24
    # return flat_series.to_list()
    return flat_series.tolist()


@pytest.fixture
def wide_list_of_series():

    return [pd.Series(np.random.randn(20), np.arange(20), name="a"),
            pd.Series(np.random.randn(10), np.arange(5, 15), name="b")]


@pytest.fixture
def wide_list_of_arrays(wide_list_of_series):

    # Requires pandas >= 0.24
    # return [s.to_numpy() for s in wide_list_of_series]
    return [np.asarray(s) for s in wide_list_of_series]


@pytest.fixture
def wide_list_of_lists(wide_list_of_series):

    # Requires pandas >= 0.24
    # return [s.to_list() for s in wide_list_of_series]
    return [s.tolist() for s in wide_list_of_series]


@pytest.fixture
def wide_dict_of_series(wide_list_of_series):

    return {s.name: s for s in wide_list_of_series}


@pytest.fixture
def wide_dict_of_arrays(wide_list_of_series):

    # Requires pandas >= 0.24
    # return {s.name: s.to_numpy() for s in wide_list_of_series}
    return {s.name: np.asarray(s) for s in wide_list_of_series}


@pytest.fixture
def wide_dict_of_lists(wide_list_of_series):

    # Requires pandas >= 0.24
    # return {s.name: s.to_list() for s in wide_list_of_series}
    return {s.name: s.tolist() for s in wide_list_of_series}


@pytest.fixture
def long_df():

    n = 100
    rs = np.random.RandomState()
    df = pd.DataFrame(dict(
        x=rs.randint(0, 20, n),
        y=rs.randn(n),
        a=np.take(list("abc"), rs.randint(0, 3, n)),
        b=np.take(list("mnop"), rs.randint(0, 4, n)),
        c=np.take(list([0, 1]), rs.randint(0, 2, n)),
        d=np.repeat(np.datetime64('2005-02-25'), n),
        s=np.take([2, 4, 8], rs.randint(0, 3, n)),
        f=np.take(list([0.2, 0.3]), rs.randint(0, 2, n)),
    ))
    df["s_cat"] = df["s"].astype("category")
    return df


@pytest.fixture
def repeated_df():

    n = 100
    rs = np.random.RandomState()
    return pd.DataFrame(dict(
        x=np.tile(np.arange(n // 2), 2),
        y=rs.randn(n),
        a=np.take(list("abc"), rs.randint(0, 3, n)),
        u=np.repeat(np.arange(2), n // 2),
    ))


@pytest.fixture
def missing_df():

    n = 100
    rs = np.random.RandomState()
    df = pd.DataFrame(dict(
        x=rs.randint(0, 20, n),
        y=rs.randn(n),
        a=np.take(list("abc"), rs.randint(0, 3, n)),
        b=np.take(list("mnop"), rs.randint(0, 4, n)),
        s=np.take([2, 4, 8], rs.randint(0, 3, n)),
    ))
    for col in df:
        idx = rs.permutation(df.index)[:10]
        df.loc[idx, col] = np.nan
    return df


@pytest.fixture
def null_column():

    return pd.Series(index=np.arange(20), dtype='float64')
