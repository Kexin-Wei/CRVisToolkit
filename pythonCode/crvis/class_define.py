from pathlib import Path
import numpy as np
from typing import Union, Sequence
from typing_extensions import Literal, TypedDict

LIST_OR_NUMPY = Union[Sequence, np.ndarray]
LIST_OR_NUMPY_OR_INT = Union[Sequence, np.ndarray, int]
