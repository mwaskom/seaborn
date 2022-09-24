from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Optional, Union, Mapping, Tuple, List, Dict
from collections.abc import Hashable, Iterable

from numpy import ndarray  # TODO use ArrayLike?
from pandas import DataFrame, Series, Index, Timestamp, Timedelta
from matplotlib.colors import Colormap, Normalize


ColumnName = Union[
    str, bytes, date, datetime, timedelta, bool, complex, Timestamp, Timedelta
]
Vector = Union[Series, Index, ndarray]

VariableSpec = Union[ColumnName, Vector, None]
VariableSpecList = Union[List[VariableSpec], Index, None]

DataSource = Union[DataFrame, Mapping[Hashable, Vector], None]

OrderSpec = Union[Iterable, None]  # TODO technically str is iterable
NormSpec = Union[Tuple[Optional[float], Optional[float]], Normalize, None]

# TODO for discrete mappings, it would be ideal to use a parameterized type
# as the dict values / list entries should be of specific type(s) for each method
PaletteSpec = Union[str, list, dict, Colormap, None]
DiscreteValueSpec = Union[dict, list, None]
ContinuousValueSpec = Union[
    Tuple[float, float], List[float], Dict[Any, float], None,
]
