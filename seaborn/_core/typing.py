from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:

    from typing import Optional, Union
    from numpy.typing import ArrayLike
    from pandas import Series, Index
    from matplotlib.colors import Colormap

    Vector = Union[Series, Index, ArrayLike]

    PaletteSpec = Optional[Union[str, list, dict, Colormap]]

    # TODO Define the following? Would simplify a number of annotations
    # ColumnarSource = Union[DataFrame, Mapping]
