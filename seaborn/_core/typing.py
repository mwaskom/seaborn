from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:

    from typing import Optional, Union, Literal
    from collections.abc import Mapping, Hashable
    from numpy.typing import ArrayLike
    from pandas import DataFrame, Series, Index
    from matplotlib.colors import Colormap

    Vector = Union[Series, Index, ArrayLike]
    PaletteSpec = Optional[Union[str, list, dict, Colormap]]
    VariableSpec = Union[Hashable, Vector]
    VariableType = Literal["numeric", "categorical", "datetime"]
    DataSource = Union[DataFrame, Mapping[Hashable, Vector]]
