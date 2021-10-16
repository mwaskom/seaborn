from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:

    from typing import Literal, Optional, Union, Tuple
    from collections.abc import Mapping, Hashable, Iterable
    from numpy.typing import ArrayLike
    from pandas import DataFrame, Series, Index
    from matplotlib.colors import Colormap, Normalize

    Vector = Union[Series, Index, ArrayLike]
    PaletteSpec = Union[str, list, dict, Colormap, None]
    VariableSpec = Union[Hashable, Vector, None]
    OrderSpec = Union[Series, Index, Iterable, None]  # TODO technically str is iterable
    NormSpec = Union[Tuple[Optional[float], Optional[float]], Normalize, None]
    # TODO can we better unify the VarType object and the VariableType alias?
    VariableType = Literal["numeric", "categorical", "datetime"]
    DataSource = Union[DataFrame, Mapping[Hashable, Vector], None]
