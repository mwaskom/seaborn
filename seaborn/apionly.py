import warnings
msg = (
"The seaborn.apionly module is deprecated, as seaborn no longer sets a default "
"style on import. It will be removed in a future version."
)
warnings.warn(msg, UserWarning)

from seaborn import *
reset_orig()
