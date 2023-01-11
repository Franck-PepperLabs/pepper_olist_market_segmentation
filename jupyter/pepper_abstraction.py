from typing import *
import numpy as np
import pandas as pd


"""Type hints definitions"""
T = TypeVar("T")

Single = Union[int, float, str, bool, None]
Vector = Union[
    pd.Series, pd.Index, np.ndarray,
    List[Single], Tuple[Single],
    None
]
Matrix = Union[pd.DataFrame, pd.MultiIndex, List[Tuple], Vector]
Selector = Union[Matrix, List[Matrix], Tuple[Matrix], Dict[str, Matrix]]


class DataWrapper:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __bool__(self):
        return bool(self.data)


class Matrix:  # (AbstactVector):
    def __init__(self, data):
        if isinstance(data, pd.DataFrame):
            self.data = data
        elif isinstance(data, np.ndarray):
            self.data = pd.DataFrame(data)
        else:
            raise TypeError(f"Unsupported data type for Matrix: {type(data)}")

    def __getitem__(self, key):
        return self.data.__getitem__(key)

    def __setitem__(self, key, value):
        self.data.__setitem__(key, value)

    def __repr__(self):
        return self.data.__repr__()
