from typing import *
import numpy as np
import pandas as pd


def to_ndarrays_list(vectors_list):
    ndarrays_list = []
    for v in vectors_list:
        ndarrays_list.append(np.asarray(v, dtype=object))
    return ndarrays_list


def series_to_ndarrays(series_list : List[pd.Series]):
    ndarrays_list = []
    series_names = []
    for s in series_list:
        ndarrays_list.append(s.to_numpy())
        series_names.append(s.name)
    return ndarrays_list, '(' + ', '.join(series_names) + ')'
