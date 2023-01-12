"""
This module contains functions to filter a pandas dataframe.
"""

from typing import *
from collections.abc import Iterable
from datetime import datetime, date, time
import json
import numpy as np
import pandas as pd
from pepper_matrix import (
    drop_nones,
    cartesian_product,
    trim_vector,
    zip_vectors,
    CompressedMatrix
)

"""Debug traces
"""


__debug = False


def dbg_trace(msg):
    if __debug:
        print(msg)


""" Type checking
"""


"""Type hints definitions"""

Single = Union[int, float, str, bool, None]
Vector = Union[
    pd.Series, pd.Index, np.ndarray,
    List[Single], Tuple[Single],
    None
]
Matrix = Union[pd.DataFrame, pd.MultiIndex, List[Tuple], Vector]
Selector = Union[Matrix, List[Matrix], Tuple[Matrix], Dict[str, Matrix]]

AgnosticKey = Union[int, str]
BooleanIndex = np.ndarray   # NormalizedVector[bool]


"""Type assertions
"""


def _is_int_like(x: Any) -> bool:
    """Returns True if x can be cast as an integer, False otherwise.

    Parameters:
    - x (Any): The object to test.

    Returns:
    - bool: True if x can be cast as an integer, False otherwise.

    Example:
    >>> _is_int_like(1)
    True
    >>> _is_int_like(True)
    True
    >>> _is_int_like(np.int8(1))
    True
    >>> _is_int_like(1.0)
    True
    >>> _is_int_like('2')
    True
    >>> _is_int_like('2.0')
    False
    """
    try:
        int(x)
        return True
    except (ValueError, TypeError):
        return False


def is_container(obj: Any) -> bool:
    """Check if an object is a container.

    Parameters:
        obj (Any): The object to check.

    Returns:
        bool: True if the object is a container, False otherwise.
    """
    return isinstance(obj, (list, tuple, dict))


def is_multi_index(obj: Any) -> bool:
    """Returns True if obj is a pandas MultiIndex, False otherwise.

    Parameters:
    - obj (Any): The object to test.

    Returns:
    - bool: True if obj is a pandas MultiIndex, False otherwise.

    Example:
    >>> is_multi_index(pd.MultiIndex.from_arrays([[1, 2], [3, 4]]))
    True
    >>> is_multi_index(pd.Index([[1, 3], [2, 4]]))
    False
    >>> is_multi_index(pd.Index([(1, 3), (2, 4)]))
    True
    >>> is_multi_index(pd.Index([1, 2]))
    False
    """
    return (
        obj is not None
        and isinstance(obj, pd.MultiIndex)
    )


def is_index(obj: Any) -> bool:
    return (
        obj is not None
        and isinstance(obj, pd.Index)
    )


def is_simple_index(obj: Any) -> bool:
    """Returns True if obj is a simple pandas Index, False otherwise.

    Parameters:
    - obj (Any): The object to test.

    Returns:
    - bool: True if obj is a pandas Index but not a MultiIndex,
        False otherwise.

    Example:
    >>> is_multi_index(pd.MultiIndex.from_arrays([[1, 2], [3, 4]]))
    False
    >>> is_multi_index(pd.Index([[1, 3], [2, 4]]))
    True
    >>> is_multi_index(pd.Index([(1, 3), (2, 4)]))
    False
    >>> is_multi_index(pd.Index([1, 2]))
    True
    """
    return (
        obj is not None
        and isinstance(obj, pd.Index)
        and not isinstance(obj, pd.MultiIndex)
    )


def is_dataframe(obj: Any) -> bool:
    """Returns True if obj is a pandas DataFrame, False otherwise.

    Parameters:
    - obj (Any): The object to test.

    Returns:
    - bool: True if obj is a pandas DataFrame, False otherwise.

    Example:
    >>> is_dataframe(pd.DataFrame([[1, 2], [2, 3]]))
    True
    >>> is_dataframe(pd.Series([1, 2]))
    False
    >>> is_dataframe(np.array([[1, 2], [2, 3]]))
    False
    """
    return (
        obj is not None
        and isinstance(obj, pd.DataFrame)
    )


def is_series(obj: Any) -> bool:
    """Returns True if obj is a pandas Series, False otherwise.

    Parameters:
    - obj (Any): The object to test.

    Returns:
    - bool: True if obj is a pandas Series, False otherwise.

    Example:
    >>> is_series(pd.Series([1, 2, 3]))
    True
    >>> is_series(pd.DataFrame([[1, 2], [3, 4]]))
    False
    >>> is_series(pd.Index([1, 2, 3]))
    False
    >>> is_series(None)
    False
    """
    return (
        obj is not None
        and isinstance(obj, pd.Series)
    )


def is_single(obj: Any) -> bool:
    """Check if an object is a single value.

    A single value can be an integer, a float, a string, a boolean,
    a datetime, a date, or a time.

    Parameters
        obj (Any): The object to check.

    Returns
        bool: True if the object is a single value, False otherwise.

    Example:
    >>> is_single(5)
    True
    >>> is_single(5.5)
    True
    >>> is_single('hello')
    True
    >>> is_single(True)
    True
    >>> is_single(datetime(2022, 1, 1))
    True
    >>> is_single(date(2022, 1, 1))
    True
    >>> is_single(time(12, 30))
    True
    >>> is_single([5, 6, 7])
    False
    """
    return (
        isinstance(obj, (int, float, str, bool, datetime, date, time))
        or np.isscalar(obj)
    )


def is_iterable(obj: Any) -> bool:
    """Check if an object is iterable.

    This function checks that `obj` is an instance of the `Iterable` class
    or one of its subclasses, which includes all sequence types (such as list,
    tuple, range, etc.) and some non-sequence types (such as dict, set, etc.).

    Parameters
    - obj : Any : Object to check.

    Returns
    - bool: True if `obj` is iterable, False otherwise.

    Examples
    >>> is_iterable([1, 2, 3])
    True
    >>> is_iterable((1, 2, 3))
    True
    >>> is_iterable({1, 2, 3})
    True
    >>> is_iterable(1)
    False
    >>> is_iterable(None)
    False
    """
    return isinstance(obj, Iterable)


def is_builtin_vector(obj):
    """Check if an object is a built-in 1D vector.

    This function checks that `obj` is a list, tuple, pd.Index, pd.Series,
    or ndarray of dimension 1.

    Parameters
    - obj : Any : Object to check.

    Returns
    - bool : True if `obj` is a built-in 1D vector, False otherwise.

    Examples
    >>> is_builtin_vector([1, 2, 3])
    True
    >>> is_builtin_vector(())
    True
    >>> is_builtin_vector(pd.Index([1, 2, 3]))
    True
    >>> is_builtin_vector(pd.Series([1, 2, 3]))
    True
    >>> is_builtin_vector(np.array([1, 2, 3]))
    True
    >>> is_builtin_vector(np.array([[1, 2, 3], [4, 5, 6]]))
    False
    >>> is_builtin_vector(5)
    False
    """
    return isinstance(obj, (
        list, tuple, pd.Index, pd.Series, np.ndarray
    ))


def is_list_of_uniform_tuples(obj: Any) -> bool:
    """Returns True if obj is a list of uniform tuples, False otherwise.

    A list of uniform tuples is a list where all the elements are tuples
    of the same length.

    Parameters:
    - obj (Any): The object to test.

    Returns:
    - bool: True if obj is a list of uniform tuples, False otherwise.

    Example:
    >>> is_list_of_uniform_tuples([(1, 2), (3, 4)])
    True
    >>> is_list_of_uniform_tuples([(1, 2, 3), (3, 4)])
    False
    >>> is_list_of_uniform_tuples([1, 2])
    False
    >>> is_list_of_uniform_tuples([[1, 2], [3, 4]])
    False
    """
    if not isinstance(obj, list) or len(obj) == 0:
        return False
    if not isinstance(obj[0], tuple):
        return False
    ref_length = len(obj[0])
    return all(isinstance(x, tuple) and len(x) == ref_length for x in obj)


def _is_numpy_kd_array(obj: Any, k: int) -> bool:
    """Check if an object is a NumPy kD array.

    Parameters
        obj (Any): The object to check.
        k (int): The number of dimensions.

    Returns
        bool: True if the object is a NumPy kD array, False otherwise.

    Examples:
    >>> _is_numpy_kd_array(np.array([1, 2, 3]), 1)
    True
    >>> _is_numpy_kd_array(np.array([[1, 2, 3], [4, 5, 6]]), 2)
    True
    >>> _is_numpy_kd_array(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9],\
        [10, 11, 12]]]), 3)
    True
    >>> _is_numpy_kd_array([[1, 2, 3], [4, 5, 6]], 2)
    False
    """
    return isinstance(obj, np.ndarray) and obj.ndim == k


def is_numpy_1d_array(obj: Any) -> bool:
    """Check if an object is a NumPy 1D array.

    Parameters
        obj (Any): The object to check.

    Returns
        bool: True if the object is a NumPy 1D array, False otherwise.

    Examples:
    >>> is_numpy_1d_array(np.array([1, 2, 3]))
    True
    >>> is_numpy_1d_array(np.array([[1, 2, 3], [4, 5, 6]]))
    False
    >>> is_numpy_1d_array([1, 2, 3])
    False
    >>> is_numpy_1d_array((1, 2, 3))
    False
    """
    return _is_numpy_kd_array(obj, 1)


def is_numpy_2d_array(obj: Any) -> bool:
    """Check if an object is a NumPy 2D array.

    Parameters
        obj (Any): The object to check.

    Returns
        bool: True if the object is a NumPy 2D array, False otherwise.

    Examples:
    >>> is_numpy_2d_array(np.array([1, 2, 3]))
    False
    >>> is_numpy_2d_array(np.array([[1, 2, 3], [4, 5, 6]]))
    True
    >>> is_numpy_2d_array([[1, 2, 3], [4, 5, 6]])
    False
    >>> is_numpy_2d_array(((1, 2, 3), (4, 5, 6)))
    False
    """
    return _is_numpy_kd_array(obj, 2)


def is_vector(
    obj: Any,
    of_singles=True,
    monotyped=True,
    uniform=True
) -> bool:
    """Check if an object is a vector.

    A vector can be a 1D NumPy array, a Pandas Series or Index,
    a list or tuple of single values.
    A vector can also be uniform (all elements have the same
    length) and monotyped (all elements have the same type).

    Parameters
        obj (Any): The object to check.
        of_singles (bool): Whether the elements of the list or tuple should be
            single values or not.
        monotyped (bool): Whether the elements of the list or tuple should have
            the same type or not.
        uniform (bool): Whether the elements of the list or tuple should have
            the same length (for lists and tuples of lists and tuples).

    Returns
        bool: True if the object is a vector, False otherwise.

    Examples:
    >>> is_vector(np.array([1, 2, 3]))
    True
    >>> is_vector(np.array([[1, 2], [3, 4]]))
    False
    >>> is_vector(pd.Series([1, 2, 3]))
    True
    >>> is_vector(pd.Index([1, 2, 3]))
    True
    >>> is_vector([1, 2, 3])
    True
    >>> is_vector([[1, 2], [3, 4]])
    False
    >>> is_vector([[1, 2], [3, 4]], uniform=False)
    True
    >>> is_vector((1, 2, 3))
    True
    >>> is_vector(((1, 2), (3, 4)))
    False
    >>> is_vector(((1, 2), (3, 4)), uniform=False)
    True
    """
    # If the object is None, it is not a vector
    if obj is None:
        return False

    # If the object is a NumPy array, a Pandas Series, Index,
    # MultiIndex or DataFrame, it is a monotyped and uniform vector
    if isinstance(obj, (
        pd.Index, pd.Series, np.ndarray,
        pd.MultiIndex, pd.DataFrame
    )):
        if of_singles:
            return obj.ndim == 1 and all(is_single(x) for x in obj)
        else:
            return True  # obj.ndim == 1

    # If the object is not a list or tuple, return False
    if not isinstance(obj, (list, tuple)):
        return False

    # If the list or tuple is empty, it is a vector
    if len(obj) == 0:
        return True

    # If none of the conditions 'of_singles', 'monotyped', 'uniform' are
    # specified, then the not empty list or tuple is considered a valid vector
    if not(of_singles or monotyped or uniform):
        return True

    # If the 'uniform' condition is specified, then the 'monotyped'
    # condition is also implied
    if uniform:
        monotyped = True

    # Check that all elements of the object have the same type
    if monotyped:
        # If the 'of_singles' condition is specified, then check that all
        # elements of the object are single values
        if of_singles and not is_single(obj[0]):
            return False

        # Set the reference type and element based on the first element of
        # the object
        ref_type = type(obj[0])
        ref_elt = obj[0]

        # Check that the type and element of all the elements of the
        # object match the reference type and element
        for x in obj:
            if not isinstance(x, ref_type):
                type_of_x = type(x)
                if not isinstance(ref_elt, type_of_x):
                    return False
                else:
                    ref_type = type_of_x
                    ref_elt = x

        # If the object is required to be uniform, check that all elements
        # have the same length and are vectors
        if uniform:
            if (
                is_builtin_vector(ref_elt)
                and len(set(map(len, obj))) != 1
            ):
                return False

        # If all elements have the same type and element and are required
        # to be uniform, return True
        return True

    # If the object is not required to be monotyped but made of single values,
    # check that all elements are single values
    if of_singles:
        if not all(is_single(x) for x in obj):
            return False

    # If all elements are single values and the object is not required
    # to be monotyped, return True
    return True


def is_list_of_singles(obj: Any) -> bool:
    """Check if an object is a list of single monotyped values.

    This function is an alias for the `is_vector` function with all constraint
    parameters set to True. It checks that `obj` is a list, tuple, ndarray,
    pd.Series, or pd.Index and that all its elements are single monotyped
    values.

    Parameters:
        obj (Any): The object to check.

    Returns:
        bool: True if `obj` is a list of single values, False otherwise.

    Examples:
    >>> is_list_of_singles([1, 2, 3])
    True
    >>> is_list_of_singles([1, 'a', 3])
    False
    >>> is_list_of_singles(np.array([1, 2, 3]))
    True
    >>> is_list_of_singles(pd.Series([1, 2, 3]))
    True
    >>> is_list_of_singles(pd.Index([1, 2, 3]))
    True
    """
    return is_vector(obj, of_singles=True, monotyped=True, uniform=True)


def is_monotyped_uniform_vector(obj: Any) -> bool:
    """Check if an object is a monotyped and uniform 1D NumPy array,
    Pandas Series or Index, list, or tuple.

    A monotyped and uniform vector is a list or tuple of monotyped and uniform
    elements. An element is monotyped if it has the same type as the other
    elements. An element is uniform if it is a list or tuple of the same
    length as the other elements.

    Parameters
        obj (Any): The object to check.

    Returns
        bool: True if the object is a monotyped and uniform vector,
            False otherwise.

    Examples:
    >>> is_monotyped_uniform_vector([[1, 2], [3, 4]])
    True
    >>> is_monotyped_uniform_vector([[1, 2], [3, 4, 5]])
    False
    >>> is_monotyped_uniform_vector((1, 2, 3))
    True
    >>> is_monotyped_uniform_vector(((1, 2), (3, 4)))
    True
    >>> is_monotyped_uniform_vector(((1, 2), (3, 4, 5)))
    False
    """
    return is_vector(obj, of_singles=False, monotyped=True, uniform=True)


def is_dict_of_uniform_vectors(obj: Any) -> bool:
    """Check if an object is a dictionary of uniform 1D NumPy arrays or lists.

    Parameters:
        obj (Any): The object to check.

    Returns:
        bool: True if the object is a dictionary of uniform 1D NumPy arrays or
            lists, False otherwise.

    Example:
    >>> is_dict_of_uniform_vectors({'a': [1, 2, 3], 'b': [4, 5, 6]})
    True
    >>> is_dict_of_uniform_vectors({'a': [1, 2, 3], 'b': [4, 5]})
    False
    >>> is_dict_of_uniform_vectors({'a': [1, 2, 3], 'b': [[4, 5], [6, 7]]})
    False
    >>> is_dict_of_uniform_vectors(
    >>>     {'a': np.array([1, 2, 3]), 'b': np.array([4, 5, 6])}
    >>> )
    True
    >>> is_dict_of_uniform_vectors(
    >>>     {'a': np.array([1, 2, 3]), 'b': np.array([[4, 5], [6, 7]])}
    >>> )
    False
    """
    # Check that obj is a dictionary
    if not isinstance(obj, dict):
        return False

    # The empty dictionnary is uniform
    if len(obj) == 0:
        return True

    # Check that all values of the dictionary are of the same length
    dict_values = list(obj.values())
    return is_monotyped_uniform_vector(dict_values)
    # bad (dict of singles case) return len(set(map(len, obj.values()))) == 1


def is_list_of_uniform_vectors(obj: Any) -> bool:
    # Check that obj is a list
    if not isinstance(obj, list):
        return False

    # The empty list is uniform
    if len(obj) == 0:
        return True

    # Check that all values of the list are of the same length
    # And that the list is not made of singles
    # dict_values = list(obj.values())
    return (
        is_monotyped_uniform_vector(obj)
        and not is_list_of_singles(obj)
    )


def assert_is_list_or_dict(obj, as_what: str):
    if not isinstance(obj, (dict, list)):
        raise TypeError(
            f"{as_what} must be a list or a dict, not be a {type(obj)}"
        )


def assert_is_none_or_dataframe(obj, as_what: str):
    if obj is not None and not is_dataframe(obj):
        raise TypeError(
            f"{as_what} must be a dataframe or None, not a {type(obj)}"
        )


""" Compression and zipping operations
"""


def zip_dataframe(df: pd.DataFrame) -> List[Tuple]:
    """Returns a list of tuples of the data contained in the given pandas
    DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame to zip.

    Returns:
    - List[Tuple]: A list of tuples containing the data of the DataFrame.
        Each tuple represents a row of the DataFrame.

    Example:
    >>> zip_dataframe(pd.DataFrame({'a': [1, 2], 'b': [3, 4]}))
    [(1, 3), (2, 4)]
    """
    if df is None or df.empty:
        return None
    # Assert that the input is a pandas DataFrame
    assert (
        isinstance(df, pd.DataFrame)
    ), "data should be a pandas DataFrame"
    return list(df.itertuples(index=False, name=None))


def zip_multi_index(mi: pd.MultiIndex) -> List[Tuple]:
    """Returns a list of tuples of the data contained in the given pandas
    MultiIndex.

    Parameters:
    - mi (pd.MultiIndex): The input MultiIndex to zip.

    Returns:
    - List[Tuple]: A list of tuples containing the data of the MultiIndex. Each
        tuple represents a row of the MultiIndex.

    Example:
    >>> zip_multi_index(pd.MultiIndex.from_product([[1, 2], ['a', 'b']]))
    [(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')]
    """
    if mi is None or mi.empty:
        return None
    # Assert that the input is a pandas MultiIndex
    assert (
        isinstance(mi, pd.MultiIndex)
    ), "data should be a pandas MultiIndex"

    return list(mi)


def zip_pandas_matrix(
    matrix: Union[pd.DataFrame, pd.MultiIndex]
) -> List[Tuple]:
    """Converts the given matrix into a list of tuples.

    Parameters:
    - matrix (Union[pd.DataFrame, pd.MultiIndex]): The input matrix to
        convert.

    Returns:
    - List[Tuple]: The list of tuples representing the matrix.

    Example:
    >>> zip_pandas_matrix(pd.DataFrame([[1, 2], [3, 4]]))
    [(1, 2), (3, 4)]
    """
    if matrix is None or matrix.empty:
        return None
    # Assert that the input is a pandas DataFrame or MultiIndex
    assert (
        isinstance(matrix, (pd.DataFrame, pd.MultiIndex))
    ), "data should be a pandas DataFrame or MultiIndex"

    if isinstance(matrix, pd.DataFrame):
        return zip_dataframe(matrix)

    if isinstance(matrix, pd.MultiIndex):
        return zip_multi_index(matrix)


def compress_pandas_matrix(
    matrix: Union[pd.DataFrame, pd.MultiIndex]
) -> Tuple[np.ndarray, List[str], Tuple[int, int]]:
    """Compress the given matrix by returning only the rows as a numpy array
    of tuples, the names of the columns, and the shape of the compressed
    matrix.

    Parameters:
    - matrix (Union[pd.MultiIndex, pd.DataFrame]): The input matrix to
        compress.

    Returns:
    - Tuple[np.ndarray, List[str], Tuple[int, int]]: A tuple containing the
        compressed matrix as a numpy array of tuples, the names of the
        columns, and the shape of the compressed matrix.

    Example:
    >>> compress_pandas_matrix(pd.DataFrame([[1, 2], [2, 3], [2, 3], [1, 2]]))
    (array([[1, 2], [2, 3]]), ['0', '1'], (2, 2))
    """
    # Assert that the input is a MultiIndex or a DataFrame
    assert (
        isinstance(matrix, (pd.MultiIndex, pd.DataFrame))
    ), "matrix should be a MultiIndex or a DataFrame"

    data = np.asarray(zip_pandas_matrix(matrix))
    names = (
        list(matrix.names) if isinstance(matrix, pd.MultiIndex)
        else list(matrix.columns)
    )
    shape = (
        matrix.shape if isinstance(matrix, pd.DataFrame)
        else (matrix.size, len(names))
    )

    return (data, names, shape)


"""Unique sorted reduction
"""


def reduce_pandas_vector(
    vector: Union[pd.Index, pd.Series]
) -> Tuple[np.ndarray, str, int]:
    """Reduces the given vector by sorting its values and returning only
    the unique values.

    Parameters:
    - vector (Union[pd.Index, pd.Series]): The input vector to reduce.

    Returns:
    - Tuple[np.ndarray, str, int]: A tuple containing the reduced vector as a
        numpy array, the name and the size of the reduced vector.

    Example:
    >>> reduce_pandas_vector(pd.Index([1, 2, 3, 2, 1]))
    (array([1, 2, 3]), 'index', 3)
    """
    # Assert that the input is a single-column Index or a Series
    assert (
        isinstance(vector, (pd.Index, pd.Series)) and
        not isinstance(vector, pd.MultiIndex)
    ), "vector should be a single-column Index or a Series"

    name = vector.name
    data = vector.sort_values().unique()
    return (
        np.asarray(data),
        name,
        data.size
    )


def reduce_vector(
    vector: Union[pd.Index, pd.Series, np.ndarray, list, tuple]
) -> np.ndarray:
    if vector is None:
        return None
    if isinstance(vector, (list, tuple, np.ndarray)):
        return np.unique(np.asarray(vector))
        # return np.asarray(sorted(list(set(vector))))
    if isinstance(vector, (pd.Index, pd.Series)):
        return reduce_pandas_vector(vector)[0]
    raise ValueError(f"Unknown type in reduce_vector {type(vector)}")


def reduce_pandas_matrix(
    matrix: Union[pd.DataFrame, pd.MultiIndex]
) -> np.ndarray:
    """Reduces the given matrix by returning only the unique rows as a numpy
    array.

    Parameters:
    - matrix (Union[pd.MultiIndex, pd.DataFrame]): The input matrix to reduce.

    Returns:
    - np.ndarray: The reduced matrix as a numpy array.

    Example:
    >>> reduce_pandas_matrix(pd.DataFrame([[1, 2], [2, 3], [2, 3], [1, 2]]))
    array([[1, 2], [2, 3]])
    """
    # Assert that the input is a pandas DataFrame or MultiIndex
    assert (
        isinstance(matrix, (pd.DataFrame, pd.MultiIndex))
    ), "matrix should be a pandas DataFrame or MultiIndex"

    if isinstance(matrix, pd.MultiIndex):
        matrix = matrix.unique()
        matrix = matrix.sort_values()
        return matrix.values
    elif isinstance(matrix, pd.DataFrame):
        matrix = matrix.sort_values(by=list(matrix.columns))
        matrix = matrix.drop_duplicates()
        return compress_pandas_matrix(matrix)[0]


"""Structures and compatibility analysis : labels analysis
"""


def get_vector_names(
    vectors: Union[Matrix, List[Matrix], Dict[str, Matrix]]
) -> List[Union[str, None]]:
    if vectors is None:
        return [None, ]
    if is_single(vectors):
        return [None, ]
    if is_multi_index(vectors):
        return list(vectors.names)
    if is_simple_index(vectors):
        return [vectors.name]
    if is_series(vectors):
        return [vectors.name]
    if is_dataframe(vectors):
        return list(vectors.columns)
    if isinstance(vectors, np.ndarray):
        return [None, ] * vectors.ndim
    if isinstance(vectors, tuple):
        names = []
        for child in vectors:
            child_names = get_vector_names(child)
            names.extend(child_names)
        return names
    if isinstance(vectors, dict):
        names = []
        for k, child in vectors.items():
            child_names = get_vector_names(child)
            child_names = [
                None if name is None
                else f"{k}_{name}"
                for name in child_names
            ]
            names.extend(child_names)
        return names
    if isinstance(vectors, list):
        if PandasMatrix.is_matrix(vectors):
            pm = PandasMatrix.parse(vectors)
            return pm.keys
        else:
            return get_vector_names(tuple(vectors))


def nest_named_arrays(
    obj: List[
        Union[
            pd.Series, pd.Index, pd.DataFrame, dict
        ]
    ]
) -> dict:
    """Convert a list of Series, Index, DataFrame, or dict objects into a
    nested dictionary of unique values.

    This function takes a list of pandas objects, such as Series, Index,
    or DataFrame, or dictionaries containing these objects, and returns
    a dictionary where the keys are the names of the objects, and the values
    are numpy arrays containing the unique values of these objects.
    If the input list contains dictionaries, the dictionary keys are used as
    the names of the objects and the dictionary values are recursively passed
    to this function to build the tree of named arrays.
    If the input list contains lists, the lists are recursively passed to this
    function to build the tree of named arrays. If the input list contains
    objects that are not supported by this function, a TypeError is raised.

    Args:
        - obj (List[Union[pd.Series, pd.Index, pd.DataFrame, dict]]): The list
            of pandas objects or dictionaries to convert into a tree of named
            arrays.

    Returns:
        dict: The tree of named arrays.

    Raises:
        TypeError: If an object in the input list is not a supported type.
    """
    if obj is None:
        return {}
    elif is_series(obj) or is_simple_index(obj):
        # If the object is a Series or a simple Index, return a dictionary
        # with a single key-value pair where the key is the name of the object
        # and the value is a numpy array containing the unique values of the
        # object
        return {obj.name: np.unique(obj.to_numpy())}
    elif is_dataframe(obj) or is_multi_index(obj):
        # If the object is a MultiIndex, convert it to a DataFrame
        if is_multi_index(obj):
            obj = obj.to_frame()
        names = {}
        # Iterate through the items of the DataFrame (columns and their
        # values)
        for k, s in obj.items():
            # Avoid duplicates and reduce each vector to unique
            join_vector(names, k, s.to_numpy())
        # Return the dictionary of named arrays
        return names
    # If the object is a dictionary, recursively apply this function to its
    # values
    elif isinstance(obj, dict):
        return {
            k: nest_named_arrays(o)
            for k, o in obj.items()
        }
    # If the object is a list, recursively apply this function to its elements
    # and merge the resulting dictionaries
    elif isinstance(obj, list):
        names = {}
        for o in obj:
            o_names = nest_named_arrays(o)
            for k, s in o_names.items():
                join_vector(names, k, s)
        # Return the merged dictionary of named arrays
        return names
    else:
        # Raise an error if the object is not supported
        raise TypeError(
            f"{type(obj)} cannot be part of a by_name selector"
        )


def _is_by_name_selector(obj: Any) -> bool:
    if obj is None:
        return True
    elif is_series(obj) or is_simple_index(obj):
        return True
    elif is_dataframe(obj) or is_multi_index(obj):
        return True
    elif isinstance(obj, dict):
        return all(
            _is_by_name_selector(o)
            or PandasVector.is_vector(o)
            for o in obj.values()
        )
    elif isinstance(obj, list):
        return all(_is_by_name_selector(o) for o in obj)
    else:
        return False


def _is_by_position_selector(obj: Any) -> bool:
    if obj is None:
        return True
    elif is_single(obj):
        return True
    elif PandasVector.is_vector(obj):
        return True
    elif PandasMatrix.is_matrix(obj):
        return True
    elif isinstance(obj, (tuple, list)):
        return all(_is_by_position_selector(o) for o in obj)
    else:
        return False


""" Selector normalization
"""


def join_vector(
    vectors_dict: dict,
    name: str,
    new_vector: np.ndarray
) -> None:
    """Add a new vector to the given dictionary, merging it with the existing
    vector if the name already exists in the dictionary.

    Args:
        - vectors_dict (dict): The dictionary to update.
        - name (str): The name of the new vector.
        - new_vector (ndarray): The new vector to add to the dictionary.

    Returns:
        None.
    """
    # If the name is already in the dictionary, merge the new vector with the
    # existing one using np.unique
    if name in vectors_dict:
        base_array = vectors_dict[name]
        union_array = np.union1d(base_array, new_vector)
        vectors_dict[name] = np.unique(union_array)
    # If the name is not in the dictionary, add the new vector to it
    else:
        vectors_dict[name] = np.unique(new_vector)


def expand_series_in_dataframe(s: pd.Series) -> pd.DataFrame:
    """Expand a Series of tuples into a DataFrame.

    This function expands a Series that contains tuples into a DataFrame,
    with each tuple being a row of the resulting DataFrame.

    Parameters:
        s (pd.Series): The Series to expand.

    Returns:
        pd.DataFrame: The resulting DataFrame.
    """
    # Tuplize the values in the Series
    s = s.apply(lambda x: (x, ) if is_single(x) else x)

    # Convert the tuplized Series into a DataFrame
    return pd.DataFrame(list(s), index=s.index)


def _get_mask(
    data: pd.DataFrame,
    selector: dict,
    by_name: bool,
    mask=None
) -> BooleanIndex:
    """ Recursively compute the mask to apply to a dataframe based on a nested
    dictionary representing a selector.

    This function creates a boolean mask for a DataFrame based on a nested
    dictionary selector. The keys of the dictionary represent either the names
    or the positions of the columns to filter depending on the `by_name`
    parameter. The values of the dictionary are either lists of acceptable
    values for the corresponding column, or nested dictionaries representing
    sub-selectors for columns containing tuples. If the value is a list, the
    rows where the corresponding column is in the list are kept. If the value
    is a nested dictionary, the rows where the corresponding column is in the
    set of valid values for the sub-selector are kept. The mask is created by
    performing an element-wise logical AND between the different selectors.

    Parameters:
        data (pd.DataFrame): The DataFrame to filter.
        selector (dict): The nested dictionary selector to use for filtering
            the DataFrame.
        by_name (bool): A flag indicating whether the keys of the `selector`
            dictionary represent the names or the positions of the columns to
            filter.
        mask (BooleanIndex, optional): An initial mask to use for filtering
            the DataFrame. If not provided, the mask is initialized to None.

    Returns:
        BooleanIndex: The mask for the DataFrame.
    """
    # Make a copy of the data to avoid modifying it
    _data = data.copy()

    # Iterate over the selector and apply it to the data
    for k, sub_selector in selector.items():
        # If the value is another dictionary, it means we need to recursively
        # apply the selector on a sub-DataFrame
        if isinstance(sub_selector, dict):
            # Get the column name or index corresponding to the current key
            name = k if by_name else _data.columns[k]
            # Extract the sub-DataFrame from the original data
            sub_data = _data[name]
            if isinstance(sub_data, pd.Series):
                sub_data = expand_series_in_dataframe(_data[name])
            # Recursively apply the selector on the sub-DataFrame
            mask = _get_mask(sub_data, sub_selector, by_name, mask)
        # If the value is not a dictionary, it means we need to apply the
        # selector on the current column
        else:
            # Get the column name or index corresponding to the current key
            name = k if by_name else _data.columns[k]
            # If the mask is None, we create a new one
            if mask is None:
                mask = _data[name].isin(sub_selector)
            else:
                mask &= _data[name].isin(sub_selector)
    if mask is None:
        mask = np.ones(data.shape[0], dtype=bool)
    return mask


class PandasFilter:

    def __init__(
        self,
        data: Union[None, np.ndarray, Tuple[Union[None, np.ndarray]]],
        name=None
    ):
        self._data = data
        self._name = name

    @property
    def data(self):
        return self._data

    @property
    def name(self):
        return self._name

    def __str__(self) -> str:
        def default_rep(x):
            if is_single(x):
                return str(x)
            else:
                return str(list(x))
        return json.dumps(
            self.data,
            indent=2,
            default=default_rep
        ).replace('"', '')

    def __repr__(self):
        return f'PandasFilter({self})'

    def _repr_pretty_(self, p, cycle):
        if cycle:
            return 'PandasFilter(...)'
        p.text(f'PandasFilter({self})')

    def __iter__(self):
        return iter(self.data)

    @classmethod
    def composite_from_tuple(cls, obj):
        # s = PandasSelector.parse(list(obj))
        s = PandasSelector.from_by_position_list(list(obj))
        return cls(s.data, None)

    @classmethod
    def simple_from_vector(cls, obj):
        # print('simple_from_vector', obj)
        v = PandasVector.parse(obj).unique()
        return cls(v.data, v.name)

    def is_composite(self):
        return isinstance(self.data, dict)

    def is_simple(self):
        return isinstance(self.data, np.ndarray)

    @classmethod
    def is_filter(cls, obj: Any) -> bool:
        return (
            obj is None
            or is_monotyped_uniform_vector(obj)
            or PandasSelector.is_selector(list(obj))
        )

    @classmethod
    def parse(cls, obj: Any) -> 'PandasFilter':
        if obj is None:
            return None
        if isinstance(obj, tuple):
            return cls.composite_from_tuple(obj)
        if PandasVector.is_vector(obj):
            return cls.simple_from_vector(obj)
        raise TypeError(f"Cannot parse filter data of type {type(obj)}")


class PandasSelector:

    def __init__(
        self,
        data: Union[Dict[int, PandasFilter], Dict[str, PandasFilter]],
        by_name: bool
    ):
        self._data = data
        self._by_name = by_name

    @property
    def data(self) -> Union[Dict[int, PandasFilter], Dict[str, PandasFilter]]:
        return self._data

    @property
    def by_name(self) -> bool:
        return self._by_name

    @property
    def by_position(self) -> bool:
        return not self._by_name

    def __str__(self) -> str:
        def default_rep(x):
            if is_single(x):
                return str(x)
            else:
                return str(x)
        return json.dumps(
            self.data,
            indent=2,
            default=default_rep
        ).replace('"', '')

    def __repr__(self):
        return f'PandasSelector({self})'

    def _repr_pretty_(self, p, cycle):
        if cycle:
            return 'PandasSelector(...)'
        p.text(f'PandasSelector({self})')

    @classmethod
    def from_vector(cls, v_obj) -> 'PandasSelector':
        # assert PandasVector.is_vector(v_obj)
        dbg_trace(f"PandasSelector.from_vector: {v_obj}")
        v = PandasVector.parse(v_obj)
        # Return either by position or by name filter
        if v.has_name():
            return cls({v.name: v.data}, True)
        else:
            return cls({0: v.data}, False)

    @classmethod
    def from_matrix(cls, mx_obj) -> 'PandasSelector':
        # assert PandasMatrix.is_matrix(mx_obj):
        dbg_trace(f"PandasSelector.from_matrix: {mx_obj}")
        matrix = PandasMatrix.parse(mx_obj)
        # Return either by position or by name selector
        if matrix.have_columns_names():
            return matrix.as_by_name_selector()
        else:
            return matrix.as_by_position_selector()

    @classmethod
    def from_by_position_list(
        cls,
        objs_list: List[Union[None, np.ndarray]]
    ) -> 'PandasSelector':
        """return cls(
            {i: PandasFilter(v) for i, v in enumerate(data) if v is not None}
        )"""
        dbg_trace(f"PandasSelector.from_by_position_list: {objs_list}")
        k = 0  # Position counter
        data = {}  # Result dictionary
        for obj in objs_list:
            # Discard None objects
            if obj is None:
                dbg_trace('None discarded')
                k += 1
            # CompositeFilter case
            elif isinstance(obj, tuple):
                # composite filter inserted at indice k
                data[k] = PandasFilter.parse(obj).data
                dbg_trace(f"composite filter added: {data[k]}")
                k += 1
            # PandasVector case
            elif PandasVector.is_vector(obj):
                v = PandasVector.parse(obj)
                # Insert filter by position
                part = v.as_by_position_filter(pos=k)
                dbg_trace(f"pandas vector added: {part}")
                data.update(part)
                k += 1
            # PandasMatrix case
            elif PandasMatrix.is_matrix(obj):
                matrix = PandasMatrix.parse(obj)
                # Insert filters by position
                part = matrix.as_by_position_selector(pos=k)
                dbg_trace(f"matrix part added: {part}")
                data.update(part)
                # Increment position counter
                k += len(part)
            else:
                raise TypeError(
                    f"Unkown raw selector type {type(obj)} : {obj}"
                )
        return cls(data, False)

    @classmethod
    def from_by_name_list(
        cls,
        obj: List[Union[pd.Series, pd.Index, pd.DataFrame, np.ndarray, dict]]
    ) -> 'PandasSelector':
        dbg_trace(f"PandasSelector.from_by_name_list: {obj}")
        data = nest_named_arrays(obj)
        return cls(data, True)

    @classmethod
    def from_dict(
        cls,
        objs_dict: Dict[Union[str, int], Union[None, np.ndarray]]
    ) -> 'PandasSelector':
        """return cls(
            {k: PandasFilter(v) for k, v in data.items() if v is not None}
        )"""
        dbg_trace(f"PandasSelector.from_dict: {objs_dict}")
        data = {}
        for key, obj in objs_dict.items():
            if obj is None:
                # simply discard
                pass
            elif isinstance(obj, tuple):
                # composite filter inserted
                data[key] = cls.parse(list(obj))
            elif isinstance(obj, list):
                # composite filter inserted
                data[key] = nest_named_arrays(obj)
            elif PandasVector.is_vector(obj):
                # filter inserted. Names prefixed with key_, key if unamed
                column = PandasVector.parse(obj)
                part = column.as_by_name_filter()  # prefix=key)
                # data.update(part)
                data[key] = part
            elif PandasMatrix.is_matrix(obj):
                # all filters inserted with key_ as common prefix
                # and key=1, ... key_n if unamed matrix
                matrix = PandasMatrix.parse(obj)
                part = matrix.as_by_name_selector()   # prefix=key)
                # data.update(part)
                data[key] = part
            else:
                raise TypeError(f"Unkown raw selector type {type(obj)}")
        return cls(data, True)

    @classmethod
    def is_by_name_selector(cls, obj: Any) -> bool:
        return _is_by_name_selector(obj)

    @classmethod
    def is_by_position_selector(cls, obj: Any) -> bool:
        return _is_by_position_selector(obj)

    @classmethod
    def is_selector(cls, obj: Any) -> bool:
        """Check if an object is a valid selector.

        A selector can be a matrix, a list of matrices, a tuple of matrices
        or a dictionary of matrices.

        Parameters
            obj (Any): The object to check.

        Returns
            bool: True if the object is a valid selector, False otherwise.
        """
        return (
            cls.is_by_name_selector(obj)
            or cls.is_by_position_selector(obj)
        )

    @classmethod
    def assert_is_selector(cls, obj, as_what: str):
        if not cls.is_selector(obj):
            raise TypeError(
                f"{as_what} is not a valid selector : {type(obj)}"
            )

    @classmethod
    def parse(cls, obj) -> 'PandasSelector':
        # Handle None case
        if obj is None:
            return None
        # Handle singleton cases
        if not isinstance(obj, (list, dict)):
            if PandasVector.is_vector(obj):
                return cls.from_vector(obj)
            elif PandasMatrix.is_matrix(obj):
                return cls.from_matrix(obj)
            # Tuple case : treat as a composite filter
            elif isinstance(obj, tuple):
                # return cls.parse(list(obj))
                return PandasSelector.from_by_position_list([obj])
            # implicit vector of one component case
            elif is_single(obj):
                return cls.from_vector([obj])
            # Unknown case
            else:
                raise TypeError(f"Unknown raw selector type {type(obj)}")
        # Handle collection cases
        elif isinstance(obj, list):
            if PandasSelector.is_by_name_selector(obj):
                return PandasSelector.from_by_name_list(obj)
            elif PandasSelector.is_by_position_selector(obj):
                return PandasSelector.from_by_position_list(obj)
            else:
                raise ValueError(f"{type(obj)} {obj} is not a selector")
        elif isinstance(obj, dict):
            return PandasSelector.from_dict(obj)


class PandasVector:

    def __init__(
        self,
        data: Union[np.ndarray, None],
        name=None,
        dtype=object
    ):
        assert (
            PandasVector.is_vector(data)
        ), f"{data} cannot be interpreted as a vector"
        self._data = data
        self._name = name
        self._dtype = dtype

    def __str__(self) -> str:
        return str(list(self.data))

    def __repr__(self):
        return f'PandasVector({list(self.data)})'

    def _repr_pretty_(self, p, cycle):
        if cycle:
            return 'PandasVector(...)'
        p.text(f'PandasVector({list(self.data)})')

    @property
    def data(self):
        return self._data

    @property
    def name(self):
        return self._name

    @property
    def dtype(self):
        return self._dtype

    def has_name(self):
        return self.name is not None

    def unique(self):
        dbg_trace(f"unique data {self.data}")
        return PandasVector(np.unique(self.data), self.name)

    @classmethod
    def from_series(cls, obj):
        return cls(obj.to_numpy(), obj.name)

    @classmethod
    def from_simple_index(cls, obj):
        return cls(obj.to_numpy(), obj.name)

    @classmethod
    def from_numpy_1d_array(cls, obj):
        return cls(obj, None)

    @classmethod
    def from_vector(cls, obj):
        return cls(np.asarray(obj), None)

    @classmethod
    def from_single(cls, obj):
        return cls(np.asarray([obj]), None)

    @classmethod
    def is_vector(cls, obj):
        return (
            obj is None
            or is_series(obj)
            or is_simple_index(obj)
            or is_numpy_1d_array(obj)
            or is_vector(obj)
            or is_single(obj)
        )

    @classmethod
    def parse(cls, obj):
        if obj is None:
            return None
        if is_series(obj):
            return cls.from_series(obj)
        if is_simple_index(obj):
            return cls.from_simple_index(obj)
        if is_numpy_1d_array(obj):
            return cls.from_numpy_1d_array(obj)
        if is_vector(obj):
            return cls.from_vector(obj)
        if is_single(obj):
            return cls.from_single(obj)
        raise TypeError(
            f"Cannot parse data of type {type(obj)} as a PandasColumn"
        )

    def to_series(self) -> pd.Series:
        return pd.Series(
            data=self.data,
            name=self.name,
            dtype=self.dtype
        )

    def to_simple_index(self) -> pd.Index:
        return pd.Index(
            data=self.data,
            name=self.name,
            dtype=self.dtype,
            tupleize_cols=False
        )

    def to_numpy_1d_array(self) -> np.ndarray:
        return self.data.copy()

    def is_list_of_singles(self) -> list:
        return list(self.data)

    def as_by_name_filter(self, prefix=None):
        key = self.name if prefix is None else f"{prefix}_{self.name}"
        return {key: self.unique().data}

    def as_by_position_filter(self, pos):
        return {pos: self.unique().data}


class PandasMatrix:

    def __init__(
        self,
        data: Union[pd.DataFrame, None]
    ):
        assert (
            PandasMatrix.is_matrix(data)
        ), f"{data} cannot be interpreted as a matrix"
        self._data = data

    def __str__(self) -> str:
        return self.data.__str__()

    def __repr__(self):
        return f'PandasMatrix({self.data.__repr__()})'

    def _repr_pretty_(self, p, cycle):
        if cycle:
            return 'PandasMatrix(...)'
        p.text(f'PandasMatrix({self.data.__repr__()})')

    def _repr_html_(self):
        return self.data._repr_html_()

    @property
    def data(self):
        return self._data

    @property
    def keys(self):
        return list(self._data.columns)

    @property
    def dtypes(self):
        return list(self._data.dtypes)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.data.shape

    @classmethod
    def from_dataframe(cls, obj: pd.DataFrame):
        return cls(obj.copy())

    @classmethod
    def from_index(cls, obj: pd.Index):
        return cls(obj.to_frame())

    @classmethod
    def from_tuples(cls, obj):
        df = pd.DataFrame.from_records(obj)
        df.columns = [None, ] * len(df.columns)
        return cls(df)

    @classmethod
    def from_array(cls, obj):
        df = pd.DataFrame(obj)
        df.columns = [None, ] * len(df.columns)
        return cls(df)

    @classmethod
    def from_dict(cls, obj: dict):
        return cls(pd.DataFrame.from_dict(obj, orient='columns'))

    @classmethod
    def from_list(cls, obj: dict):
        df = pd.DataFrame(obj).transpose()
        df.columns = [None, ] * len(df.columns)
        return cls(df)

    @classmethod
    def from_vector(cls, obj: PandasVector) -> "PandasMatrix":
        """Create a PandasMatrix from a PandasVector.

        This class method creates a PandasMatrix from a PandasVector `obj`.
        If all elements of `obj` are single values, a single-column matrix
        is created. Otherwise, if all elements of `obj` are uniform vectors,
        a multi-column matrix is created by zipping the elements of the vector
        to create rows of the matrix.

        Parameters
        - obj : PandasVector : PandasVector to convert to a PandasMatrix.

        Returns
        - PandasMatrix: PandasMatrix created from `obj`.

        Raises
        - ValueError: If `obj` is not a PandasVector.

        Examples
        >>> PandasMatrix.from_vector(PandasVector([1, 2, 3]))
        0
        0  1
        1  2
        2  3
        >>> PandasMatrix.from_vector(PandasVector([[1, 2], [3, 4], [5, 6]]))
        0  1
        0  1  2
        1  3  4
        2  5  6
        """
        # Check that obj is a PandasVector
        if not PandasVector.is_vector(obj):
            raise ValueError("The object is not a PandasVector.")

        of_singles = is_vector(obj, of_singles=True)
        uniform = is_vector(obj, uniform=True)

        # If all elements of the vector are single values, or are not uniform,
        # create a single-column matrix
        if of_singles or not uniform:
            # In particular if obj is a single
            if is_single(obj):
                return cls(pd.DataFrame({None: [obj]}))
            else:
                return cls(pd.DataFrame({None: obj}))

        # Otherwise, if all elements of the vector are uniform vectors,
        # create a multi-column matrix
        if uniform:
            # Zip the elements of the vector to create rows of the matrix
            rows = list(zip(*obj))
            return cls(pd.DataFrame(rows))

    @classmethod
    def is_matrix(cls, obj: Any) -> bool:
        """Check if an object is a matrix-like structure.

        This class method checks that `obj` is a Pandas DataFrame, MultiIndex,
        list of uniform tuples, NumPy 2D array, dictionary of uniform 1D NumPy
        arrays or lists, or a Pandas Vector.

        Parameters
        - obj : Any : Object to check.

        Returns
        - bool: True if `obj` is a matrix-like structure, False otherwise.

        Examples
        >>> PandasMatrix.is_matrix(pd.DataFrame([[1, 2], [3, 4]]))
        True
        >>> PandasMatrix.is_matrix([(1, 2), (3, 4)])
        True
        >>> PandasMatrix.is_matrix([[1, 2], [3, 4]])
        True
        >>> PandasMatrix.is_matrix({'a': [1, 2], 'b': [3, 4]})
        True
        >>> PandasMatrix.is_matrix(pd.MultiIndex.from_tuples([(1, 2), (3, 4)]))
        True
        >>> PandasMatrix.is_matrix(np.array([[1, 2], [3, 4]]))
        True
        >>> PandasMatrix.is_matrix([[1, 2, 3], [4, 5, 6]])
        False
        >>> PandasMatrix.is_matrix({'a': [1, 2], 'b': [3, 4, 5]})
        False
        >>> PandasMatrix.is_matrix(None)
        True
        """
        return (
            obj is None
            or is_dataframe(obj)
            or is_index(obj)
            or is_list_of_uniform_tuples(obj)
            or is_numpy_2d_array(obj)
            or is_dict_of_uniform_vectors(obj)
            or is_list_of_uniform_vectors(obj)
            or PandasVector.is_vector(obj)
        )

    @classmethod
    def assert_is_matrix(cls, obj, as_what: str):
        if not cls.is_matrix(obj):
            raise TypeError(
                f"{as_what} is not a valid matrix : {type(obj)}"
            )

    @classmethod
    def parse(cls, obj):
        if obj is None:
            return None

        # hack : (1, 2) becomes [(1, 2)] : [] denotes axis 0, () axis 1
        if (
            is_list_of_singles(obj)
            and isinstance(obj, tuple)
        ):
            obj = [obj]
        if is_dataframe(obj):
            dbg_trace('PandasMatrix.from_dataframe')
            return cls.from_dataframe(obj)
        if is_index(obj):
            dbg_trace('PandasMatrix.from_index')
            return cls.from_index(obj)
        if is_list_of_uniform_tuples(obj):
            dbg_trace('PandasMatrix.from_tuples')
            return cls.from_tuples(obj)
        if is_numpy_2d_array(obj):
            dbg_trace('PandasMatrix.from_array')
            return cls.from_array(obj)
        if is_dict_of_uniform_vectors(obj):
            dbg_trace('PandasMatrix.from_dict')
            return cls.from_dict(obj)
        if is_list_of_uniform_vectors(obj):
            dbg_trace('PandasMatrix.from_list')
            return cls.from_list(obj)
        if PandasVector.is_vector(obj):
            dbg_trace('PandasMatrix.from_vector')
            return cls.from_vector(obj)
        raise TypeError(
            f"Cannot parse data of type {type(obj)} as a PandasMatrix"
        )

    def to_dataframe(self) -> pd.DataFrame:
        return self.data.copy()

    def to_multi_index(self) -> pd.MultiIndex:
        return pd.MultiIndex.from_frame(self.data)

    def to_tuples(self) -> list:
        return self.data.to_records(index=False).tolist()

    def to_array(self) -> np.ndarray:
        return self.data.to_numpy()

    def to_dict(self) -> dict:
        return self.data.to_dict(orient='list')

    def has_key(self, key: Union[str, int]) -> bool:
        if isinstance(key, str):
            return key in self.data.columns
        else:
            return key < self.shape[1]

    def has_keys(self, keys: Union[List[str], List[int]]) -> bool:
        if keys is None or len(keys) == 0:
            return True
        if isinstance(keys[0], str):
            return all(np.isin(keys, self.names))
        else:
            return max(keys) < self.shape[1]

    def get_column(self, key: Union[str, int]) -> PandasVector:
        key = key if isinstance(key, str) else self.data.columns[key]
        return PandasVector(self.data[key])

    def update_column(self, column: PandasVector):
        if not self.has_column(column.name):
            raise ValueError(
                f"There is no column with name {column.name} in the matrix"
            )
        self.data[column.name] = column.to_series()

    def append_column(self, column: PandasVector):
        if self.has_column(column.name):
            raise ValueError(
                f"Column with name {column.name} already exists in the matrix"
            )
        self.data[column.name] = column.to_series()

    def insert_column(self, loc: int, column: PandasVector):
        if self.has_column(column.name):
            raise ValueError(
                f"Column with name {column.name} already exists in the matrix"
            )
        if not isinstance(loc, int):
            raise TypeError(f"loc ({loc}) must be an integer")
        if not 0 <= loc <= self.data.shape[1]:
            raise ValueError(
                "loc ({loc})  must be within the range [0, number of columns]"
            )
        self.data.insert(
            loc,
            column.name,
            column.to_series(),
            # allow_duplicates=False : already garded above
        )

    def row_items(self):
        return self.data.iterrows()

    def col_items(self):
        return self.data.items()

    def __iter__(self):
        return self.col_items()

    def subset(self, rows=None, cols=None):
        def _convert_to_indexer(keys, axis):
            if keys is None:
                return None
            if all(_is_int_like(k) for k in keys):
                return keys
            if axis == 0:
                return self.data.index.get_indexer(keys)
            if axis == 1:
                return self.data.columns.get_indexer(keys)
            raise ValueError(f"Invalid axis: {axis}")

        rows = _convert_to_indexer(rows, 0)
        cols = _convert_to_indexer(cols, 1)
        return PandasMatrix(self.data.iloc[rows, cols])

    def have_columns_names(self) -> bool:
        """Return whether the matrix columns are named."""
        return all(name is not None for name in self.data.columns)

    def is_unequivocal_naming(self) -> bool:
        """Return whether the matrix columns have unique names."""
        return self.data.columns.is_unique

    def repair_naming(self):
        """Repair the column names if necessary.

        This method renames duplicate column names by appending a suffix
        to the duplicate names.
        """
        # Check if repair is necessary
        if self.is_unequivocal_naming():
            return  # nothing to do
        # Create a mapping of old column names to new ones
        counts = {}
        i_first = {}
        new_names = []
        for i, name in enumerate(self.data.columns):
            if name not in counts:
                counts[name] = 1
                i_first[name] = i
                new_names[i] = name
            else:
                if counts[name] == 1:
                    new_names[i_first[name]] += '_0'
                new_names[i] = f"{name}_{counts[name]}"
                counts[name] += 1
        # Rename the columns
        self.data.rename(columns=new_names, inplace=True)

    def as_by_name_selector(self, prefix=None) -> Dict[str, Any]:
        """Return a dictionary of filters with keys as column names.

        The keys are prefixed with the given prefix.

        Parameters
            prefix (str): The prefix to use for the keys.
            Default is an empty string.

        Returns
            Dict[str, Any]: A dictionary of filters, with keys as column names
                and values as filters.
        """
        if not self.is_named():
            self.repair_naming()
        filters = {}
        for name, series in self.items():
            key = name if prefix is None else f'{prefix}_{name}'
            filters[key] = series.to_numpy()
        return filters

    def as_by_position_selector(self, pos=0) -> Dict[int, Any]:
        """Return a dictionary of filters with keys as column indices.

        The keys are indices starting from the given position.

        Parameters
            pos (int): The starting index for the keys. Default is 0.

        Returns
            Dict[int, Any]: A dictionary of filters, with keys as column
                indices and values as filters.
        """
        filters = {}
        for i, (_, series) in enumerate(self.col_items()):
            key = pos + i
            filters[key] = series.to_numpy()
        return filters

    def get_mask(self, selector: PandasSelector) -> BooleanIndex:
        if selector is None:
            return np.ones(self.shape[0], dtype=bool)
        return _get_mask(self.data, selector.data, selector.by_name)


"""Main function
"""


def filtered_copy(
    dataframe: pd.DataFrame,
    data_filter=None,
    rows_filter=None,
    cols_filter=None
) -> pd.DataFrame:
    """Create a filtered copy of a DataFrame.

    This function creates a filtered copy of the input DataFrame based on the
    given index and frame selectors. If the index or frame selector is
    None, the corresponding rows or columns are not filtered.

    Parameters:
        data (pd.DataFrame): The DataFrame to filter.
        index_selector (Selector): The index selector to use for filtering the
            rows of the DataFrame Index (MultiIndex) depending on some of its
            columns.
        frame_selector (Selector): The frame selector to use for filtering
            the rows of the DataFrame depending on some of its columns.

    Returns:
        pd.DataFrame: The filtered copy of the input DataFrame.
    """
    # Check that the input arguments are acceptable
    PandasMatrix.assert_is_matrix(dataframe, "`dataframe` argument")
    PandasSelector.assert_is_selector(data_filter, "`data_filter` argument")
    PandasSelector.assert_is_selector(rows_filter, "`rows_filter` argument")
    PandasSelector.assert_is_selector(cols_filter, "`cols_filter` argument")

    _data = PandasMatrix.parse(dataframe)
    _rows = PandasMatrix.parse(dataframe.index)
    _cols = PandasMatrix.parse(dataframe.columns)

    _data_filter = PandasSelector.parse(data_filter)
    _rows_filter = PandasSelector.parse(rows_filter)
    _cols_filter = PandasSelector.parse(cols_filter)

    data = _data.get_mask(_data_filter)
    rows = _rows.get_mask(_rows_filter)
    cols = _cols.get_mask(_cols_filter)

    if __debug:
        show_mask(data, 'data')
        show_mask(rows, 'rows')
        show_mask(cols, 'cols')

    return dataframe.loc[rows & data, cols].copy()


def show_mask(mask, name='', indent=0):
    indent = '\t' * indent
    if name != '':
        name = " `" + name + "`"
    if mask is None:
        print(f"{indent}Boolean selection mask{name}: None")
    else:
        print(f"{indent}Boolean selection mask{name}:")
        print(f"{indent}\t# Total : {len(mask)}")
        print(f"{indent}\t# True : {mask.sum()}")
        print(f"{indent}\t# False : {(~mask).sum()}")


def selector_diagnostic(
    data: pd.DataFrame,
    selector: Selector,
    name: str,
    display_: bool = False
) -> None:
    print("Selector '" + name + "'", end='')
    if display_:
        print(':', selector)
    else:
        print()

    print('\tSelector is', 'None' if selector is None else 'not None')

    mode = _is_by_name_selector(selector)
    print('\tSelection will be by', 'name' if mode is None else 'position')

    is_a = PandasSelector.is_selector(selector)
    print('\tSelector is', 'consistent' if is_a else 'unconsistent')
    # print('\tSelector type :', type_)

    # check_selector_compatibility_v1(data, selector, mode, on_index)
    # print('\tSelector is compatible with the dataframe', target_name)
    # TODO : développement de l'intersection

    # norm_selector = normalize_selector_v1(selector, mode)
    norm_selector = PandasSelector.parse(selector)
    print(
        '\tNormalized selector :',
        norm_selector if display_
        else str(norm_selector)[:20] + '...' + str(norm_selector)[-20:]
    )

    _data = PandasMatrix.parse(data)
    mask = _data.get_mask(norm_selector)
    show_mask(mask, name, indent=1)


def filtered_copy_inspection(
    data: pd.DataFrame,
    data_filter=None,
    rows_filter=None,
    cols_filter=None,
    display_=False
):
    if data_filter is not None or display_:
        selector_diagnostic(data, data_filter, 'data filter', display_)
    if rows_filter is not None or display_:
        selector_diagnostic(data.index, rows_filter, 'rows filter', display_)
    if cols_filter is not None or display_:
        selector_diagnostic(data.columns, cols_filter, 'cols filter', display_)
