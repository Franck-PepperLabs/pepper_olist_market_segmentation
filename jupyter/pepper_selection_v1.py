"""
This module contains functions to filter a pandas dataframe.
Prototype version
"""

from typing import *
import warnings
import numpy as np
import pandas as pd


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


"""Type hints definitions
"""

T = TypeVar("T")

Single = Union[int, float, str, bool, None]
Vector = Union[
    pd.Series, pd.Index, np.ndarray,
    List[Single], Tuple[Single],
    None
]
Matrix = Union[pd.DataFrame, pd.MultiIndex, List[Tuple], Vector]
Selector = Union[Matrix, List[Matrix], Tuple[Matrix], Dict[str, Matrix]]

NormalizedVector = Union[List[T], np.ndarray, None]
NormalizedSelector = Union[Dict[str, np.ndarray], List[np.ndarray]]

AgnosticKey = Union[int, str]
BooleanIndex = NormalizedVector[bool]

AndAppendFunction = Callable[
    [
        Union[pd.MultiIndex, pd.DataFrame],
        BooleanIndex,
        AgnosticKey,
        NormalizedVector
    ],
    BooleanIndex
]


def is_single_v1(obj):
    return isinstance(obj, (int, float, str, bool))


def is_monotyped_vector_v1(vector: Vector, check_len=False) -> bool:
    """Check if a vector is monotyped.

    DEPRECADTED Use `is_vector` instead

    A vector is monotyped if all its elements have the same type.

    Parameters
        vector (Vector): The vector to check.

    Returns
        bool: True if the vector is monotyped, False otherwise.
    """
    warnings.warn(
        "`is_monotyped_vector_v1` is deprecated and will be removed "
        "in a future version : use `is_vector` instead.",
        DeprecationWarning
    )
    if vector is None or len(vector) == 0:
        return True
    if isinstance(vector, (pd.Series, np.ndarray)):
        return True
    element_type = type(vector[0])
    element_len = None
    if check_len:
        try:
            element_len = len(vector[0])
        except TypeError:
            check_len = False
    for e in vector:
        if type(e) != element_type:
            return False
        if check_len and len(e) != element_len:
            return False
    return True


def is_vector_v1(vector: Vector) -> Tuple[bool, str]:
    """Check if an object is a vector.

    DEPRECATED Use `is_vector` instead

    A vector is a pd.Series, a pd.Index or a list of single values, all with
    the same type. A list of tuples is not considered as a vector.

    Parameters
        vector (Vector): The object to check.

    Returns
        Tuple[bool, str]: A tuple where the first element is a boolean
            indicating if the object is a vector, and the second element is a
            string describing the type of vector. Possible values are 'None',
            'pd.Series', 'pd.Index', 'MonotypedList[Single]',
            'MonotypedList[Tuple]', 'MultitypedList', 'Unknown'.
    """
    warnings.warn(
        "`is_vector_v1` is deprecated and will be removed "
        "in a future version : use `is_vector` instead.",
        DeprecationWarning
    )
    if vector is None:
        return True, 'None'
    if isinstance(vector, pd.Series):
        return True, 'pd.Series'
    if isinstance(vector, pd.Index):
        return True, 'pd.Index'
    # TODO : add np.ndarray and tuple cases
    if isinstance(vector, list):
        if is_monotyped_vector_v1(vector):
            if isinstance(vector, tuple):
                return False, 'MonotypedList[Tuple]'
            else:
                if len(vector) == 0 or is_single_v1(vector[0]):
                    return True, 'MonotypedList[Single]'
                else:
                    return False, 'MonotypedList[~(Single)]'
        else:
            return False, 'MultitypedList'
    return False, 'Unknown'


def matrix_to_vector_v1(matrix: Matrix) -> np.ndarray:
    """Convert a matrix-like object to a single vector.

    Args:
        - matrix (Matrix): The matrix to convert.

    Returns:
        np.ndarray: The converted vector.
    """
    # Return None if matrix is None
    if matrix is None:
        return None

    # Convert pd.Series and pd.Index to ndarray
    if isinstance(matrix, (pd.Series, pd.Index)):
        return matrix.to_numpy()

    # Convert pd.MultiIndex to flat ndarray
    if isinstance(matrix, pd.MultiIndex):
        return matrix.to_flat_index().to_numpy()

    # Convert pd.DataFrame to record ndarray
    if isinstance(matrix, pd.DataFrame):
        return np.ndarray(matrix.to_records())

    # Convert list to ndarray
    if isinstance(matrix, list):
        return np.ndarray(matrix)


def convert_matrix_v1(
    matrix: Matrix,
    vectors_dict: dict
) -> bool:
    """Convert a matrix-like object to a single vector and add it to the given
    dictionary.

    Args:
        - matrix (Matrix): The matrix to convert.
        - vectors_dict (dict): The dictionary to update.

    Returns:
        bool: True if the matrix was successfully converted, False otherwise.
    """
    # If matrix is None, return True
    if matrix is None:
        return True

    # If matrix is a Series or Index, add its name and data to the dictionary
    if isinstance(matrix, (pd.Series, pd.Index)):
        join_vector(vectors_dict, matrix.name, matrix.to_numpy())
        return True

    # If matrix is a DataFrame, add the name and data of each series to the
    # dictionary
    if isinstance(matrix, pd.DataFrame):
        for name, series in matrix.items():
            join_vector(vectors_dict, name, series.to_numpy())
        return True

    # If matrix is a MultiIndex, add the name and data of each level to the
    # dictionary
    if isinstance(matrix, pd.MultiIndex):
        for name in matrix.names:
            vector = matrix.get_level_values(name).to_numpy()
            join_vector(vectors_dict, name, vector)
        return True

    # If matrix is none of the above, return False
    return False


def is_name_list_v1(names: List[Union[str, None]]) -> Tuple[bool, int]:
    """Check if a list of strings is valid.

    A valid list does not contain any None element.

    Parameters
        names (List[str]): The list of strings to check.

    Returns
        Tuple[bool, int]: If the list is valid, returns True and its length.
            If the list is invalid, returns a tuple with a boolean value of
            False and the position of the first None element in the list.
    """
    if names is None or len(names) == 0:
        return True, 0
    if not isinstance(names, list):
        return False, -1
    for i, name in enumerate(names):
        if name is None or not isinstance(name, str):
            return False, i
    return True, len(names)


""" Built-in alternative to test :
if names is None or len(names) == 0:
    return True, 0
if not all(isinstance(name, str) for name in names):
    return (
        False,
        next(
            i for i,
            name in enumerate(names)
            if name is not None
            and not isinstance(name, str)
        )
    )
if any(name is None for name in names):
    return (
        False,
        next(
            i for i,
            name in enumerate(names)
            if name is None
        )
    )
return True, len(names)"""


def is_id_list_v1(names: Union[List[str], None]) -> bool:
    """Check if a list of strings is a valid list of unique ids.

    A valid list of ids does not contain any repeated element.
    This function assumes that the list has already been checked for validity
    with the `is_name_list` function.

    Parameters
        names (List[str]): The list of strings to check.

    Returns
        bool: True if the list is a valid list of unique ids, False otherwise.
    """
    if names is None:
        return True
    if not is_name_list_v1(names):
        return False
    return len(names) == len(set(names))


def is_by_name_selector_v1(selector: Selector) -> bool:
    """Determine if `selector` is a by-name selector.

    This function checks if `selector` is a by-name selector. A by-name
    selector is an object that specifies rows or columns by their names,
    rather than by their position.

    Parameters
        selector: The object to check.

    Returns
        bool: True if `selector` is a by-name selector, False otherwise.

    Examples
    >>> is_by_name_selector_v1(['a', 'b', 'c'])
    True
    >>> is_by_name_selector_v1(['a', None, 'c'])
    False
    >>> is_by_name_selector_v1({'a': 0, 'b': 1, 'c': 2})
    True
    >>> is_by_name_selector_v1(pd.Index(['a', 'b', 'c']))
    True
    >>> is_by_name_selector_v1(
    >>>     pd.MultiIndex.from_tuples([('a', 'x'), ('b', 'y'), ('c', 'z')])
    >>> )
    True
    >>> is_by_name_selector_v1(None)
    False
    >>> is_by_name_selector_v1((1, 2, 3))
    False
    """
    if selector is None:
        return None
    if isinstance(selector, pd.MultiIndex):
        is_nl, _ = is_name_list_v1(list(selector.names))
        return is_nl
    if isinstance(selector, pd.Index):
        is_nl, _ = is_name_list_v1([selector.name])
        return is_nl
    if isinstance(selector, pd.DataFrame):
        is_nl, _ = is_name_list_v1(list(selector.columns))
        return is_nl
    if isinstance(selector, pd.Series):
        is_nl, _ = is_name_list_v1([selector.name])
        return is_nl
    if is_vector_v1(selector)[0]:
        return False
    if isinstance(selector, tuple):
        return False
    if isinstance(selector, dict):
        return True
    if isinstance(selector, list):
        is_nl, _ = is_name_list_v1(get_vector_names_v1(selector))
        return is_nl


def is_matrix_v1(matrix: Matrix) -> Tuple[bool, str]:
    """Check if an object is a matrix.

    DEPRECATED - Use `PandasMatrix.is_matrix` instead

    A matrix is a pd.DataFrame, a pd.MultiIndex or a list of tuples with the
    same length, all with the same type. A single value, a list of single
    values or a list of lists of different lengths are not considered as a
    matrix.

    Parameters
        matrix (Matrix): The object to check.

    Returns
        Tuple[bool, str]: A tuple where the first element is a boolean
            indicating if the object is a matrix, and the second element is a
            string describing the type of matrix. Possible values are 'None',
            'pd.DataFrame', 'pd.MultiIndex', 'MonotypedList[Tuple]', 'Vector',
            'Unknown'.
    """
    warnings.warn(
        "`is_matrix_v1` is deprecated and will be removed "
        "in a future version : use `PandasMatrix.is_matrix` instead.",
        DeprecationWarning
    )
    if matrix is None:
        return True, 'None'
    if isinstance(matrix, pd.DataFrame):
        return True, 'pd.DataFrame'
    if isinstance(matrix, pd.MultiIndex):
        return True, 'pd.MultiIndex'
    is_vect, type_ = is_vector_v1(matrix)
    if is_vect:
        return True, 'Vector[' + type_ + ']'
    if isinstance(matrix, list):
        if is_monotyped_vector_v1(matrix):
            if isinstance(matrix[0], tuple):
                return True, 'MonotypedList[Tuple]'
            else:
                type_ = str(type(matrix[0]))
                return False, 'MonotypedList[' + type_ + ']'
        else:
            return False, 'MultitypedList'
    return False, 'Unknown'


def is_selector_v1(
    selector: Selector,
    name: str = '',
    verbose: bool = False
) -> Tuple[bool, str]:
    """Check if an object is a valid selector.

    DEPRECATED - Use `PandasSelector.Selector.is_selector` instead

    A selector can be a matrix, a list of matrices, a tuple of matrices or a
    dictionary of matrices.

    Parameters
        selector (Selector): The object to check.
        name (str): The name of the object, used in the returned message.
        verbose (bool): Whether to print detailed messages or not.

    Returns
        Tuple[bool, str]: A tuple where the first element is a boolean
            indicating if the object is a valid selector, and the second
            element is a string describing the type of selector. Possible
            values are 'None', 'Matrix', 'List[Matrix]', 'Tuple[Matrix]',
            'Dict[Matrix]', 'Unknown'.
    """
    warnings.warn(
        "`is_selector_v1` is deprecated and will be removed "
        "in a future version : use `PandasSelector.is_selector` instead.",
        DeprecationWarning
    )
    if selector is None:
        return True, f"Selector '{name}' is None"

    # Root container
    if isinstance(selector, (list, tuple, dict)):
        container_type = type(selector).__name__
        iterator = (
            selector.items() if isinstance(selector, dict)
            else enumerate(selector)
        )
        for k, item in iterator:
            is_a, type_ = is_matrix_v1(item)
            if not is_a:
                if verbose:
                    print(
                        f"is_a_selector({name}) : ✘ {container_type} item "
                        f"({k}) is not a Matrix :", type_
                    )
                return (
                    False,
                    f"{name} is not a Selector : {container_type}[~Matrix]"
                )
        return (
            True,
            f"Selector '{name}' is a {len(selector)}-{container_type}[Matrix]"
        )

    is_a, type_ = is_matrix_v1(selector)
    if is_a:
        if verbose:
            print(f"is_a_selector({name}) : ✔ Matrix :", type_)
        return True, f"Selector '{name}' is a Matrix"
    else:
        if verbose:
            print(f"is_a_selector({name}) :", type_)

    return False, f"{name} is not a Selector : Unknown"


def check_arguments_integrity_v1(
    data: Any,
    index_selector: Any,
    columns_selector: Any
) -> None:
    """Check the integrity of the arguments of a function.

    DEPRECATED Use `check_arguments_integrity` instead.

    This function checks that the `data` argument is a DataFrame, and that the
    `index_selector` and `columns_selector` arguments are valid selectors. If
    any of these conditions is not met, it raises a TypeError.

    Parameters
        data (Any): The data to check.
        index_selector (Any): The index selector to check.
        columns_selector (Any): The columns selector to check.

    Raises
        TypeError: If any of the arguments is not of the correct type.
    """
    warnings.warn(
        "`check_arguments_integrity_v1` is deprecated and will be removed "
        "in a future version : use `check_arguments_integrity` instead.",
        DeprecationWarning
    )
    if data is not None and not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"`data` argument must be a dataframe or None, not a {type(data)}"
        )

    is_a, type_ = is_selector_v1(index_selector, 'index_selector')
    if not is_a:
        raise TypeError(
            f"`index_selector` argument is not a valid selector : {type_}"
        )

    is_a, type_ = is_selector_v1(columns_selector, 'columns_selector')
    if not is_a:
        raise TypeError(
            f"`columns_selector` argument is not a valid selector : {type_}"
        )


""" Args integrity
"""


def get_index_names_v1(
    data: Union[pd.DataFrame, None]
) -> Union[List[str], None]:
    """Get level names in the index of a dataframe.

    DEPRECATED Use `PandasMatrix.names` instead

    Parameters
        data (pd.DataFrame): The dataframe to get the index level names from.

    Returns
        Union[List[str], None]: The list of level names in the index of the
            dataframe. None if the dataframe is None.
    """
    if data is None:
        return None
    if isinstance(data.index, pd.Index):
        return [data.index.name]
    if isinstance(data.index, pd.MultiIndex):
        return list(data.index.names)


def get_columns_names_v1(data: pd.DataFrame) -> Union[List[str], None]:
    """Get column names in a dataframe.

    DEPRECATED Use `PandasMatrix.names` instead

    Parameters
        data (pd.DataFrame): The dataframe to get the column names from.

    Returns
        Union[List[str], None]: The list of columns names in the dataframe.
            None if the dataframe is None.
    """
    if data is None:
        return None
    return list(data.columns)


def get_vector_names_v1(
    vectors: Union[Matrix, List[Matrix], Dict[str, Matrix]]
) -> List[Union[str, None]]:
    """Get the names of the vectors in a matrix.

    DEPRECATED Use `get_vector_names` instead

    Parameters
        vectors (Union[Matrix, List[Matrix], Dict[str, Matrix]]): The matrix
            or list of matrices for which to get the names of the vectors. If
            a dictionary, the keys will be used as the names of the vectors.

    Returns
        List[str]: A list of the names of the vectors.
    """
    warnings.warn(
        "`get_vector_names_v1` is deprecated and will be removed "
        "in a future version : use `get_vector_names` instead.",
        DeprecationWarning
    )
    if vectors is None:
        return []
    if isinstance(vectors, pd.MultiIndex):
        return list(vectors.names)
    if isinstance(vectors, pd.Index):
        return [vectors.name]
    if isinstance(vectors, dict):
        return list(vectors.keys())
    if isinstance(vectors, (list, tuple)):
        vector_names = []
        for v in vectors:
            if v is None:
                vector_names += [None]
            elif isinstance(v, list):
                if not v:
                    vector_names += [None]
                else:
                    if isinstance(v[0], tuple):
                        vector_names += [None] * len(v[0])
                    else:
                        vector_names += [None]
            elif isinstance(v, pd.MultiIndex):
                vector_names += list(v.names)
            elif isinstance(v, pd.Index):
                vector_names += [v.name]
            else:
                vector_names += [v.name]
        return vector_names


def are_names_in_v1(ref_names_container, check_if_in_names_list):
    """Check if all the names in check_if_in_names_list are in
    ref_names_container.

    DEPRECATED See `PandasMatrix.has_columns` instead.

    This function checks if all the names in `check_if_in_names_list` are in
    `ref_names_container`.

    Parameters
        ref_names_container: An object containing a list of names to check
            against.
        check_if_in_names_list: A list of names to check if they are in
            `ref_names_container`.

    Returns
        bool: True if all the names in `check_if_in_names_list` are in
            `ref_names_container`, False otherwise.
    """
    ref_names = list(ref_names_container)
    ref_names = [name for name in ref_names if name is not None]
    return len(set(check_if_in_names_list) - set(ref_names)) == 0


def are_index_names_v1(data, names):
    """Check if all the names in names are in the index of data.

    DEPRECATED See `PandasMatrix.has_columns` instead.

    This function checks if all the names in `names` are in the index of
    `data`.

    Parameters
        data (pd.DataFrame): The data to check the index names against.
        names (List[str]): A list of names to check if they are in the index
            of `data`.

    Returns
        bool: True if all the names in `names` are in the index of `data`,
            False otherwise.
    """
    return are_names_in_v1(data.index.names, names)


def are_columns_names_v1(data, names):
    """Check if all the names in names are in the columns of data.

    DEPRECATED See `PandasMatrix.has_columns` instead.

    This function checks if all the names in `names` are in the columns of
    `data`.

    Parameters
        data (pd.DataFrame): The data to check the columns names against.
        names (List[str]): A list of names to check if they are in the columns
            of `data`.

    Returns
        bool: True if all the names in `names` are in the columns of `data`,
            False otherwise.
    """
    return are_names_in_v1(data.columns, names)


def get_index_nlevels_v1(data: Union[pd.DataFrame, None]) -> int:
    """Get the number of levels in the index of a dataframe.

    Parameters
        data (pd.DataFrame): The dataframe to get the index nlevels from.

    Returns
        int: The number of levels in the index of the dataframe.
    """
    if data is None or data.empty:
        return 0
    if isinstance(data.index, pd.MultiIndex):
        return data.index.nlevels
    if isinstance(data.index, pd.Index):
        return 1


def get_columns_nlevels_v1(data: Union[pd.DataFrame, None]) -> int:
    """Get the number of levels in the columns of a dataframe.

    Parameters
        data (pd.DataFrame): The dataframe to get the columns nlevels from.

    Returns
        int: The number of levels in the columns of the dataframe.
    """
    if data is None or data.empty:
        return 0
    return data.shape[1]


def get_selector_nlevels_v1(selector):
    """Get the number of levels in a by-position selector.

    DEPRECATED use `get_selector_nlevels` instead

    Parameters
        selector: The by-position selector to get the nlevels from.

    Returns
        int: The number of levels in the selector.

    TODO : Future enhancement : count the None (or equivalent) on the right
        and substract
    """
    if selector is None:
        return 0
    if isinstance(selector, pd.Index):
        return 1
    if isinstance(selector, pd.MultiIndex):
        return selector.nlevels
    if isinstance(selector, tuple):
        return len(selector)
    if isinstance(selector, list):
        nlevels = 0
        for item in selector:
            if item is None:
                nlevels += 1
            elif isinstance(item, pd.Index):
                nlevels += 1
            elif isinstance(item, pd.MultiIndex):
                nlevels += item.nlevels
            elif isinstance(item, list):
                if len(item) == 0:
                    nlevels += 1
                else:
                    if isinstance(item[0], tuple):
                        nlevels += len(item[0])
                    else:
                        nlevels += 1
        return nlevels
    return -1


def get_selector_nlevels_v2(selector):
    """Get the number of levels in a by-position selector.

    Count the Nones on the right and substract it from the result.

    Parameters
        selector: The by-position selector to get the nlevels from.

    Returns
        int: The number of levels in the selector.
    """
    if selector is None:
        return 0
    if isinstance(selector, pd.MultiIndex):
        return selector.nlevels
    if isinstance(selector, pd.Index):
        return 1
    if isinstance(selector, tuple):
        return 1   # (...) is the same as [(...)]
    if isinstance(selector, list):
        nlevels = 0
        n_right_nones = 0
        for item in selector:
            if item is None:
                nlevels += 1
                n_right_nones += 1
            elif isinstance(item, pd.MultiIndex):
                nlevels += item.nlevels
                n_right_nones = 0
            elif isinstance(item, pd.Index):
                nlevels += 1
                n_right_nones = 0
            elif isinstance(item, tuple):
                nlevels += 1
                n_right_nones = 0
            elif isinstance(item, list):
                nlevels += 1
                n_right_nones = 0
            else:
                raise TypeError(
                    "List items type must be one of None, pd.Index,"
                    " pd.MultiIndex, tuple or list, but not "
                    f"{str(type(item))}"
                )
        return nlevels - n_right_nones
    raise TypeError(
        "Selector type must be one of None, pd.Index,"
        " pd.MultiIndex, tuple or list, but not "
        f"{str(type(selector))}"
    )


def check_selector_compatibility_v1(
    data: pd.DataFrame,
    selector,
    by_name: bool,
    on_index: bool
):
    """
    Check if a selector is compatible with a DataFrame.

    DEPRECATED Use `PandasMatrix.assert_applicable(PandasSelector)` instead

    This function checks if the selector is a valid selector for the given
    DataFrame. If the selector is a by-name selector, the function checks that
    all the names in the selector are present in the DataFrame's index level
    names or columns. If the index selector is a by-position selector, the
    function checks that the number of levels in the selector is not greater
    than the number of levels in the DataFrame's index or columns.

    Parameters:
        data (pd.DataFrame): The DataFrame to check against.
        selector (Selector): The selector to check.
        by_name (bool): A boolean indicating if the selector is a by-name
            selector or a by-position selector.
        on_index (bool): A boolean indicating if the selector targets the
            dataframe index or the dataframe columns.

    Raises:
        ValueError: If the selector is not compatible with the DataFrame.
    """
    # If selector is None, we don't need to check anything.
    if selector is None:
        return None

    # Get the necessary parameters for the check based on whether the selector
    # targets the index or the columns of the DataFrame.
    def get_params(on_index):
        if on_index:
            return (
                get_index_names_v1,
                are_index_names_v1,
                get_index_nlevels_v1,
                'index_selector',
                'index'
            )
        else:
            return (
                get_columns_names_v1,
                are_columns_names_v1,
                get_columns_nlevels_v1,
                'columns_selector',
                'columns'
            )

    # Define variables based on whether the selector is applied to the index
    # or to the columns of the dataframe
    (
        data_names_getter,
        data_names_checker,
        data_nlevels_getter,
        selector_name,
        target_name
    ) = get_params(on_index)

    # If the selector is a by-name selector, we check that all the names in
    # the selector are present in the DataFrame's index level names or
    # columns.
    if by_name:
        # Get the names of the index levels or columns in the DataFrame.
        target_names = data_names_getter(data)
        # Get the names of the index levels or columns in the selector.
        selector_names = get_vector_names_v1(selector)
        # Check if all the names in the selector are present in the DataFrame.
        if not data_names_checker(data, selector_names):
            # If the check fails, we raise a ValueError with a message
            # indicating which selector names are not present in the
            # DataFrame.
            raise ValueError(
                f"`{selector_name}` names {selector_names} not all"
                f" in data {target_name} names {target_names}"
            )
    # If the selector is a by-position selector, we check that the number of
    # levels in the selector is not greater than the number of levels in the
    # DataFrame's index or columns.
    else:
        # Get the number of levels in the index or columns of the DataFrame.
        target_nlevels = data_nlevels_getter(data)
        # Get the number of levels in the index or columns of the selector.
        selector_nlevels = get_selector_nlevels_v1(selector)
        # Check if the number of levels in the selector is greater than the
        # number of levels in the DataFrame.
        if selector_nlevels > target_nlevels:
            # If the check fails, we raise a ValueError with a message
            # indicating the number of levels in the selector and the number
            # of levels in the DataFrame.
            raise ValueError(
                f"`{selector_name}` nlevels ({selector_nlevels}) >"
                f" data {target_name} nlevels ({target_nlevels})"
            )


def compute_mask_v1(
    selector: NormalizedSelector,
    data_size: int,
    and_append: AndAppendFunction,
    target: Union[pd.MultiIndex, pd.DataFrame]
) -> BooleanIndex:
    """Compute a boolean mask for selecting rows or columns in a dataframe.

    Args:
        selector (NormalizedSelector): The normalized selector to use for
            filtering.
        data (pd.DataFrame): The dataframe to filter.
        and_append (AndAppendFunction): The function to use for updating the
            boolean mask.
        target (Union[pd.MultiIndex, pd.DataFrame]): The target index or
            dataframe against whiwh to evaluate the boolean mask update

    Returns:
        BooleanIndex: The boolean mask to use for filtering the rows or
            columns of the dataframe.
    """
    mask = np.ones(data_size, dtype=bool)
    if isinstance(selector, dict):
        for k, v in selector.items():
            mask &= and_append(target, mask, k, v)
    else:
        for i, v in enumerate(selector):
            mask &= and_append(target, mask, i, v)
    return mask


def index_mask_and_append_v1(
    index: pd.MultiIndex,
    mask: BooleanIndex,
    key: AgnosticKey,
    vector: NormalizedVector
) -> BooleanIndex:
    """Compute the boolean mask for the rows of a DataFrame with a MultiIndex.

    This function is used in `compute_mask` to compute the boolean mask for
    selecting rows in a DataFrame with a MultiIndex. It appends the boolean
    mask for the rows corresponding to the given `vector` for the index level
    specified by `key` to the given `mask`. If `vector` is None, the
    corresponding mask is set to True.

    Args:
        index (pd.MultiIndex): The MultiIndex of the DataFrame.
        mask (BooleanIndex): The boolean mask to update.
        key (AgnosticKey): The key specifying the index level to use.
        vector (NormalizedVector): The vector of values to use for filtering
            the rows.

    Returns:
        BooleanIndex: The updated boolean mask.
    """
    # If a vector is None, it is interpreted as a slicing
    # operator, so we set the corresponding mask to True.
    return (
        mask if vector is None
        else mask & index.get_level_values(key).isin(vector)
    )


def columns_mask_and_append_v1(
    data: pd.DataFrame,
    mask: BooleanIndex,
    key: AgnosticKey,
    vector: NormalizedVector
) -> BooleanIndex:
    """Update the boolean mask for selecting columns in a dataframe.

    Args:
        data (pd.DataFrame): The dataframe to filter.
        mask (BooleanIndex): The boolean mask to update.
        key (AgnosticKey): The key (column name or position) to use for
            filtering.
        vector (NormalizedVector): The column values to include in the mask.

    Returns:
        BooleanIndex: The updated boolean mask.
    """
    # If a vector is None, it is interpreted as a slicing
    # operator, so we set the corresponding mask to True.
    key = key if isinstance(key, str) else data.columns[key]

    return (
        mask if vector is None
        else mask & data[key].isin(vector)
    )


def get_index_boolean_mask_v1(
    data: pd.DataFrame,
    selector: NormalizedSelector
) -> BooleanIndex:
    """Return a boolean mask for selecting rows in the given dataframe.

    Args:
        - data (pd.DataFrame): The dataframe to filter.
        - selector (NormalizedSelector): The normalized index selector.

    Returns:
        np.ndarray: The boolean mask to use for filtering the rows of the
        dataframe.
    """
    if data is None:
        return None

    if selector is None:
        return np.ones(data.shape[0], dtype=bool)

    if not isinstance(selector, (dict, list)):
        raise TypeError(
            f"Selector is not a NormalizedSelector : {type(selector)}"
        )

    if isinstance(data.index, pd.MultiIndex):
        return compute_mask_v1(
            selector,
            data.shape[0],
            index_mask_and_append_v1,
            data.index
        )

    if isinstance(data.index, pd.Index):
        vector = (
            selector[0] if isinstance(selector, list)   # first list element
            else list(selector.items())[0][1]           # first dict element
        )
        return data.index.isin(vector)

    raise ValueError(f"Unsupported index type: {type(data.index)}")


def get_columns_boolean_mask_v1(
    data: pd.DataFrame,
    selector: NormalizedSelector
) -> BooleanIndex:
    """Return a boolean mask for selecting rows in the given dataframe.

    Args:
        - data (pd.DataFrame): The dataframe to filter.
        - selector (NormalizedSelector): The normalized index selector.

    Returns:
        BooleanIndex: The boolean mask to use for filtering the rows of the
        dataframe.
    """
    if data is None:
        return None

    if selector is None:
        return np.ones(data.shape[0], dtype=bool)

    return compute_mask_v1(
        selector,
        data.shape[0],
        columns_mask_and_append_v1,
        data
    )


def get_vectors_list_v1(selector: Selector) -> List[np.ndarray]:
    """Convert a selector to a list of vectors.

    Args:
        - selector (Selector): The selector to convert.

    Returns:
        List[np.ndarray]: The list of vectors obtained from the selector.
    """
    vectors = []
    if selector is None:
        return None
    elif isinstance(selector, (pd.DataFrame, pd.MultiIndex)):
        for _, series in selector.items():
            vectors.append(series.to_numpy())
    elif isinstance(selector, (pd.Series, pd.Index)):
        vectors.append(selector.to_numpy())
    # elif is_vector(selector)[0]:
    #    vectors.append(np.asarray(selector))
    elif isinstance(selector, (list, tuple)):
        for item in selector:
            if isinstance(item, (list, tuple)):
                vectors.append(np.asarray(item))
            else:
                vectors.append(item)
    # vectors.extend(get_vectors_list(matrix))
    else:
        raise TypeError(
            f"The given selector of type {type(selector)} is not of a"
            " supported type (pd.Series, pd.Index, pd.DataFrame,"
            " pd.MultiIndex, list, tuple)"
        )
    return vectors


def get_vectors_dict_v1(
    selector: Selector,
    vectors_dict: dict = {}
) -> dict:
    """"Convert a selector object to a dictionary of vectors, merging them if
    their name already exists in the dictionary.

    Args:
        - selector (Selector): The matrix or collection of matrices to convert.
        - vectors_dict (dict): The dictionary to update, set by default to {}.

    Returns:
        dict: The updated dictionary.
    """
    if not isinstance(vectors_dict, dict):
        raise TypeError("The given vectors_dict is not of type dict")
    # Simple element case
    if convert_matrix_v1(selector, vectors_dict):
        pass
    # Collections cases
    elif isinstance(selector, list):
        for named_matrix in selector:
            convert_matrix_v1(named_matrix, vectors_dict)
    elif isinstance(selector, dict):
        for name, matrix in selector.items():
            join_vector(vectors_dict, name, matrix_to_vector_v1(matrix))
    else:
        raise TypeError(
            "The given selector is not of a supported type (pd.Series, "
            "pd.Index, pd.DataFrame, pd.MultiIndex, list, dict)"
        )
    return vectors_dict


def normalize_selector_v1(selector, by_name):
    """Normalize a selector.

    This function normalizes a selector by converting it to a dictionary
    if it is a by-name selector, or to a list if it is a by-position
    selector.

    Parameters
        selector: The selector to normalize.
        by_name (bool): A boolean indicating if the selector is a
            by-name selector or a by-position selector.

    Returns
        Union[Dict[AgnosticKey, NormalizedVector],
              List[NormalizedVector]]: The normalized selector.
    """
    if selector is None:
        return None
    return (
        get_vectors_dict_v1(selector) if by_name
        else get_vectors_list_v1(selector)
    )


def apply_selectors_v1(
    data: pd.DataFrame,
    index_selector: Selector,
    columns_selector: Selector
) -> pd.DataFrame:
    """Apply index and columns selectors to a DataFrame.

    This function applies the index and columns selectors to the given
    DataFrame to create a filtered copy of the DataFrame. If either of the
    selectors is None, it is not applied. If both selectors are provided,
    the resulting filtered copy includes only rows and columns that match
    both selectors. If only one selector is provided, the resulting filtered
    copy includes only rows or columns that match the selector.

    Parameters
        data (pd.DataFrame): The dataframe to filter.
        index_selector (Selector): The index selector to apply.
        columns_selector (Selector): The columns selector to apply.

    Returns
        pd.DataFrame: A filtered copy of the dataframe.
    """
    if index_selector is None and columns_selector is None:
        # If no selectors are provided, return a copy of the original dataframe
        return data.copy()

    # Initialize the boolean mask to None
    mask = None
    if (
        index_selector is not None
        and columns_selector is not None
    ):
        # If both selectors are provided, get the boolean mask for both
        # selectors and compute the intersection of the two masks using the
        # "&" operator
        mask = (
            get_index_boolean_mask_v1(data, index_selector)
            & get_columns_boolean_mask_v1(data, columns_selector)
        )
    elif index_selector is not None:
        # If only the index selector is provided, get the boolean mask for it
        mask = get_index_boolean_mask_v1(data, index_selector)
    else:
        # If only the columns selector is provided, get the boolean mask for it
        mask = get_columns_boolean_mask_v1(data, columns_selector)

    # Use the boolean mask to filter the rows and columns of the dataframe
    # and return a copy of the resulting filtered dataframe
    return data[mask].copy()


"""Main function
"""


def filtered_copy_v1(
    data: pd.DataFrame,
    index_selector=None,
    columns_selector=None
):
    """Create a filtered copy of a DataFrame.

    This function creates a filtered copy of the input DataFrame based on the
    given index and columns selectors. If the index or columns selector is
    None, the corresponding rows or columns are not filtered.

    Parameters:
        data (pd.DataFrame): The DataFrame to filter.
        index_selector (Selector): The index selector to use for filtering the
            rows of the DataFrame.
        columns_selector (Selector): The columns selector to use for filtering
            the columns of the DataFrame.

    Returns:
        pd.DataFrame: The filtered copy of the input DataFrame.
    """
    # Check that the input arguments are of the correct type
    check_arguments_integrity_v1(data, index_selector, columns_selector)

    # Check if the index and columns selectors are by-name or by-position
    # selectors
    index_mode = is_by_name_selector_v1(index_selector)
    columns_mode = is_by_name_selector_v1(columns_selector)

    # Check if the index and columns selectors are compatible with the input
    # DataFrame
    check_selector_compatibility_v1(data, index_selector, index_mode, True)
    check_selector_compatibility_v1(
        data, columns_selector, columns_mode, False
    )

    # Normalize the index and columns selectors by converting them to
    # dictionaries if they are by-name selectors, or lists if they are by-
    # position selectors
    index_selector = normalize_selector_v1(index_selector, index_mode)
    columns_selector = normalize_selector_v1(columns_selector, columns_mode)

    # Create the filtered copy of the DataFrame using the normalized selectors
    return apply_selectors_v1(data, index_selector, columns_selector)


def step_by_step_filtered_copy_v1(
    data: pd.DataFrame,
    index_selector=None,
    columns_selector=None
):
    print('Given arguments: ', end='')
    print(
        '_' if data is None else 'data',
        '_' if index_selector is None else 'index_selector',
        '_' if columns_selector is None else 'columns_selector',
        sep=', '
    )

    # Check that the input arguments are of the correct type
    check_arguments_integrity_v1(data, index_selector, columns_selector)
    print('✔ arguments integrity validated')

    # Check if the index and columns selectors are by-name or by-position
    # selectors
    index_mode = is_by_name_selector_v1(index_selector)
    print('⇒ index selection by', 'name' if index_mode else 'position')
    columns_mode = is_by_name_selector_v1(columns_selector)
    print('⇒ columns selection by', 'name' if columns_mode else 'position')

    # Check if the index and columns selectors are compatible with the input
    # DataFrame
    check_selector_compatibility_v1(data, index_selector, index_mode, True)
    print('✔ index selector is compatible with the dataframe index')
    check_selector_compatibility_v1(
        data, columns_selector, columns_mode, False
    )
    print('✔ columns selector is compatible with the dataframe columns')

    # Normalize the index and columns selectors by converting them to
    # dictionaries if they are by-name selectors, or lists if they are by-
    # position selectors
    index_selector = normalize_selector_v1(index_selector, index_mode)
    print('✔ index selector normalized')
    columns_selector = normalize_selector_v1(columns_selector, columns_mode)
    print('✔ columns selector normalized')

    # Create the filtered copy of the DataFrame using the normalized selectors
    return apply_selectors_v1(data, index_selector, columns_selector)


def selector_diagnostic_v1(
    data: pd.DataFrame,
    selector: Selector,
    name: str,
    on_index: bool = True,
    display_: bool = True
) -> None:
    target_name = 'index' if on_index else 'columns'
    print("Selector '" + name + "' (on " + target_name + ")", end='')
    if display_:
        print(':', selector)
    else:
        print()

    print('\tSelector is', 'None' if selector is None else 'not None')

    mode = is_by_name_selector_v1(selector)
    print('\tSelection will be by', 'name' if mode is None else 'position')

    is_a, type_ = is_selector_v1(selector, 'selector')
    print('\tSelector is', 'consistent' if is_a else 'unconsistent')
    print('\tSelector type :', type_)

    check_selector_compatibility_v1(data, selector, mode, on_index)
    print('\tSelector is compatible with the dataframe', target_name)

    norm_selector = normalize_selector_v1(selector, mode)
    print(
        '\tNormalized selector :',
        norm_selector if display_
        else str(norm_selector)[:20] + '...' + str(norm_selector)[-20:]
    )

    mask = (
        get_index_boolean_mask_v1(data, norm_selector) if on_index
        else get_columns_boolean_mask_v1(data, norm_selector)
    )
    print(
        '\tBoolean selection mask : # True (', mask.sum(),
        '), # False (', (~mask).sum(), ')',
        sep=''
    )


def filtered_copy_inspection_v1(
    data,
    index_selector,
    columns_selector,
    index_selector_name='index_selector',
    columns_selector_name='columns_selector',
    display_=True
):
    selector_diagnostic_v1(
        data=data,
        selector=index_selector,
        name=index_selector_name,
        on_index=True,
        display_=display_
    )
    selector_diagnostic_v1(
        data=data,
        selector=columns_selector,
        name=columns_selector_name,
        on_index=False,
        display_=display_
    )


""" Intermediary version (1.5)
"""


"""def get_selector_type_v15(selector):
    "" "Determine the type of a selector.

    A selector can be either a by-name selector or a by-position selector.
    This function determines the type of the selector.

    Parameters
        selector: The object to check.

    Returns
        str: A string indicating the type of the selector.
            Possible values are 'by_name' and 'by_position'.
    " ""
    if selector is None or isinstance(selector, tuple):
        return 'by_position'
    if isinstance(selector, dict):
        return 'by_name'
    if isinstance(selector, (pd.Index, pd.MultiIndex, pd.Series, list)):
        is_nl, _ = is_name_list(get_vector_names(selector))
        return 'by_name' if is_nl else 'by_position'
    return 'unknown'
    """


"""def check_arguments_integrity_v15(
    data: Any,
    index_selector: Any,
    columns_selector: Any
) -> None:
    "" "Check the integrity of the arguments of a function.

    This function checks that the `data` argument is a DataFrame, and that the
    `index_selector` and `columns_selector` arguments are valid selectors. If
    any of these conditions is not met, it raises a TypeError.

    Parameters
        data (Any): The data to check.
        index_selector (Any): The index selector to check.
        columns_selector (Any): The columns selector to check.

    Raises
        TypeError: If any of the arguments is not of the correct type.
    "" "
    assert_is_none_or_dataframe(data, "`data` argument")
    assert_is_selector(index_selector, "`index_selector` argument")
    assert_is_selector(columns_selector, "`columns_selector` argument")
    """


""" Boolean index building
"""


"""def zip_isin_v15(zipped_matrix, selector):
    "" "Return a boolean mask indicating which rows of `zipped_matrix` match
    the elements of the cartesian product of `selector`.

    Args:
        - zipped_matrix (List[Tuple]): A list of tuples.
        - selector (Union[List, Tuple]): A list or tuple of lists or tuples,
          representing the dimensions of `zipped_matrix`. A `None` element
          indicates that the corresponding dimension should be discarded.

    Returns:
        np.ndarray: The boolean mask indicating which rows of `zipped_matrix`
        match the elements of the cartesian product of `selector`.

    Examples:
    >>> zip_isin([(1, 2, 3), (1, 3, 5), (1, 4, 6)], [(1,), None, (3, 5)])
    array([ True,  True, False])
    >>> zip_isin([(1, 2, 3), (1, 3, 5), (1, 4, 6)], [(1,), (3,), (3, 5)])
    array([False,  True, False])
    "" "
    discard_mask = get_discard_mask_v2(selector)
    zipped_matrix = discard_subcolumns(zipped_matrix, discard_mask)
    product_selector = cartesian_product(drop_nones(selector))
    return np.isin(zipped_matrix, product_selector).all(axis=1)"""


"""def index_vmask_update(
    vmask: BooleanIndex,
    index: pd.MultiIndex, key: AgnosticKey,  # target couple
    filter_
) -> BooleanIndex:
    "" "Compute the boolean mask for the rows of a DataFrame with a MultiIndex.

    This function is used in `compute_mask` to compute the boolean mask for
    selecting rows in a DataFrame with a MultiIndex. It appends the boolean
    mask for the rows corresponding to the given `vector` for the index level
    specified by `key` to the given `mask`. If `vector` is None, the
    corresponding mask is set to True.

    Args:
        vmask (BooleanIndex): The boolean mask to update.
        index (pd.MultiIndex): The MultiIndex of the DataFrame.
        key (AgnosticKey): The key specifying the index level to use.
        vector (NormalizedVector): The vector of values to use for filtering
            the rows.

    Returns:
        BooleanIndex: The updated boolean mask.
    "" "
    # If a filter is None, it is interpreted as a slicing
    # operator, so we set the corresponding mask to True.
    if filter_ is None:
        return vmask

    # The agnostic key is ok in its both formats (code or name)
    # with get_level_values
    target = index.get_level_values(key)
    if isinstance(filter_, tuple):  # is composite_filter(filter)
        # If vector is a tuple, we use zip_isin to filter the rows
        # using the corresponding level of the MultiIndex
        zipped_target = zip_vectors(target)
        return vmask & zip_isin(zipped_target, filter_)
    else:
        # Otherwise, we use the isin method of the MultiIndex
        return vmask & target.isin(filter_)"""


"""def data_vmask_update(
    vmask: BooleanIndex,
    data: pd.DataFrame, key: AgnosticKey,  # target couple
    filter_
) -> BooleanIndex:
    # If a filter is None, it is interpreted as a slicing
    # operator, so we set the corresponding mask to True.
    if filter_ is None:
        return vmask

    key = key if isinstance(key, str) else data.columns[key]

    target = data[key]
    if isinstance(filter_, tuple):  # is composite_filter(filter)
        # If vector is a tuple, we use zip_isin to filter the rows
        # using the corresponding level of the MultiIndex
        zipped_target = zip_vectors(target)
        return vmask & zip_isin(zipped_target, filter_)
    else:
        # Otherwise, we use the isin method of the MultiIndex
        return vmask & target.isin(filter_)"""


"""class AgnosticKey:

    def __init__(self, key: Union[str, int]):
        self.key = key

    def is_id(self):
        return isinstance(self.key, int)

    def is_name(self):
        return isinstance(self.key, str)"""

"""class PandasFilter:


    def as_simple_filter_data(self) -> Union[None, np.ndarray]:
        if self.is_simple():
            return self.data
        raise ValueError("Filter is not a simple filter")

    def as_composite_filter_data(self) -> Tuple[Union[None, np.ndarray]]:
        if self.is_composite():
            return self.data
        raise ValueError("Filter is not a composite filter")

    @staticmethod
    def _convert_raw_filter_data(
        raw_filter: Union[None, np.ndarray, Tuple[Union[None, np.ndarray]]]
    ) -> 'PandasFilter':
        ""Convert raw filter data to a Filter instance.

        Args:
            raw_filter (Union[None, np.ndarray, \
                Tuple[Union[None, np.ndarray]]]):
                The raw filter data to convert.

        Returns:
            Filter: The converted Filter instance.
        ""
        if isinstance(raw_filter, tuple):
            if all(isinstance(f, np.ndarray) for f in raw_filter):
                return PandasFilter(raw_filter)
            elif all(f is None for f in raw_filter):
                return PandasFilter(raw_filter)
            else:
                raise ValueError(
                    "Invalid composite filter data: %s" % raw_filter
                )
        elif isinstance(raw_filter, np.ndarray):
            return PandasFilter(raw_filter)
        elif raw_filter is None:
            return PandasFilter(raw_filter)
        else:
            raise ValueError(
                "Invalid filter data: %s" % raw_filter
            )

    def from_raw_filter_data(cls, raw_filter_data: Any) -> 'PandasFilter':
        ""Create a Filter object from raw filter data.

        Args:
            raw_filter_data: The raw filter data. Can be any type of data.

        Returns:
            Filter: The Filter object created from the raw filter data.
        ""
        # Convert raw filter data to Filter data
        filter_data = cls._convert_raw_filter_data(raw_filter_data)
        # Create and return Filter object
        return cls(filter_data)
"""

"""def _matrix_to_selector_data(matrix: Union[pd.DataFrame, pd.MultiIndex]):
    # on retourne un dictionnaire de ndarrays réduits
    # dont les clés sont les noms de cols
    return PandasMatrix(matrix).to_np_uniques_dict()"""


"""class PandasVector:

    def apply_filter(self, filter_: PandasFilter):
        if filter_ is None:
            return np.ones(self.data.size, dtype=bool)
        if filter_.is_composite():
            zipped_target = zip_vectors(self.data)
            return zip_isin(zipped_target, filter_.data)
        if filter_.is_simple():
            return self.data.isin(filter_.data)
        raise TypeError(f"Unknown filter type {type(filter_)}")
"""

"""class PandasMatrix:

    @classmethod
    def is_it_a_matrix(cls, obj):
        if obj is None:
            return "Object is None, then not a matrix"
        if isinstance(obj, (pd.MultiIndex, pd.DataFrame)):
            return f"Matrix object is a {type(obj).__name__}"
        if is_list_of_uniform_tuples(obj):
            return "Matrix object is a list of uniform tuples"
        if is_numpy_2d_array(obj):
            return "Matrix object is a NumPy 2D array"
        if PandasVector.is_vector(obj):
            return "Matrix object is a PandasColumn"
        return f"Object is not a matrix : Unknown type {type(obj)}"


    def is_matrix(cls, obj):
        if obj is None:
            return False
        if isinstance(obj, (pd.MultiIndex, pd.DataFrame)):
            return True
        if is_list_of_uniform_tuples(obj):
            return True
        if is_numpy_2d_array(obj):
            return True
        if PandasColumn.is_column(obj):
            return True
        return False

    def update_vmask(
        self,
        vmask: BooleanIndex,
        key: AgnosticKey,
        filter_: PandasFilter
    ) -> BooleanIndex:
        "" "Update the vertical mask of the matrix using the given filter.

        Args:
            vmask (BooleanIndex): The boolean mask to update.
            key (AgnosticKey): The key specifying the index level or column to
                use.
            filter_ (Filter): The filter to apply to the index level or column.

        Returns:
            BooleanIndex: The updated boolean mask.
        "" "
        return vmask & self.get_column(key).apply_filter(filter_)


    def assert_applicable(self, selector: PandasSelector) -> None:
        pass

    def assert_applicable_by_name(self, selector: PandasSelector) -> None:
        if not self.has_keys(selector.keys):
            # If the check fails, we raise a ValueError with a message
            # indicating which selector names are not present in the
            # DataFrame.
            raise ValueError(
                f"`{selector.name}` names {selector.keys} not all"
                f" in data {self.name} names {self.names}"
            )

"""


"""def compute_mask_v2(
    selector: NormalizedSelector,
    data_size: int,
    and_append: AndAppendFunction,
    target: PandasMatrix
) -> BooleanIndex:
    "" "Compute a boolean mask for selecting rows or columns in a dataframe.

    Args:
        selector (NormalizedSelector): The normalized selector to use for
            filtering.
        data (pd.DataFrame): The dataframe to filter.
        and_append (AndAppendFunction): The function to use for updating the
            boolean mask.
        target (Union[pd.MultiIndex, pd.DataFrame]): The target index or
            dataframe against whiwh to evaluate the boolean mask update

    Returns:
        BooleanIndex: The boolean mask to use for filtering the rows or
            columns of the dataframe.
    "" "
    mask = np.ones(data_size, dtype=bool)
    if isinstance(selector, dict):
        for k, v in selector.items():
            mask &= and_append(target, mask, k, v)
    else:
        for i, v in enumerate(selector):
            mask &= and_append(target, mask, i, v)
    return mask"""


"""def discard_subcolumns(matrix_data, discard_mask):
    "" "
    Return a list of tuples with the elements of `matrix_data` whose
    corresponding element in `discard_mask` is `False`.

    Args:
        - matrix_data (List[Tuple]): The list of tuples to filter.
        - discard_mask (List[Union[bool, Tuple[bool]]]): The boolean mask to
        use for filtering the elements of `matrix_data`.
    Returns:
        List[Tuple]: The list of tuples with the elements of `matrix_data`
        whose corresponding element in `discard_mask` is `False`.
    "" "
    return (
        CompressedMatrix(matrix_data)
        .discard_columns(discard_mask)
        .data
    )"""


"""def get_index_boolean_mask_v2(
    index: pd.Index,
    selector: NormalizedSelector
) -> BooleanIndex:
    "" "Return a boolean mask for selecting rows in the given dataframe.
    This is the version 2 of the function, which uses the `zip_isin` function
    rather than the `isin` function from NumPy.

    Args:
        data (pd.DataFrame): The dataframe to filter.
        selector (Union[List, Tuple]): The index selector.
            This can be a list or a tuple of lists or tuples.
        multi_index (bool, optional): If `True`, `selector` is assumed to be
            a multi-index selector. Defaults to `False`.

    Returns:
        np.ndarray: The boolean mask to use for filtering the rows of the
            dataframe.

    Examples:
    >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
    >>> get_index_boolean_mask_v2(df, (1, 2))
    array([False,  True, False])
    >>> get_index_boolean_mask_v2(df, [1, 2])
    array([False,  True, False])
    >>> get_index_boolean_mask_v2(df, [(1, 2), (2, 3)])
    array([False,  True,  True])
    >>> get_index_boolean_mask_v2(df, [(1, 2), (2, 3)], multi_index=True)
    array([False, False, False])
    "" "

    if index is None or index.size == 0:
        return None

    if selector is None:
        return np.ones(index.size, dtype=bool)

    assert_is_list_or_dict(selector, 'NormalizedSelector')

    # index = data.index
    # discard_mask = get_discard_mask(selector)

    if is_multi_index(index):
        return compute_mask(
            selector,
            index.size,
            index_mask_and_append,
            index
        )

    if is_simple_index(index):
        "" "sub_selector = selector[0]
        sub_mask = discard_mask[0]
        sub_index = discard_subcolumns(index, sub_mask)
        sub_selector_product = cartesian_product(drop_nones(sub_selector))
        return np.isin(sub_index, sub_selector_product).flatten()"" "
        return zip_isin(index, selector[0])

    raise ValueError(f"Unsupported index type: {type(index)}")"""


"""def get_discard_mask(
    normalized_selector: List[Union[None, np.ndarray, Tuple[None, np.ndarray]]]
) -> List[Union[bool, Tuple[bool]]]:
    "" "Creates a discard mask from the given normalized selector.

    A discard mask is a list of booleans indicating which elements of the
    normalized selector should be discarded. An element is discarded if it is
    None or if it is a tuple containing only Nones.

    Parameters:
    - normalized_selector (List): The input normalized selector.

    Returns:
    - List: The discard mask.

    Example:
    >>> get_discard_mask([
    >>>    (None, np.array([1, 2]), None),
    >>>    None,
    >>>    np.array([1]),
    >>>    (None, None),
    >>>    np.array([1, 2])
    >>> ])
    [(True, False, True), True, False, (True, True), False]
    "" "
    discard_mask = []
    for i in normalized_selector:
        if i is None:
            discard_mask.append(True)
        elif not isinstance(i, tuple):
            discard_mask.append(False)
        else:
            sub_mask = tuple(j is None for j in i)
            discard_mask.append(sub_mask)
    return discard_mask"""

"""
# version récursive et multidimensionnelle
def get_discard_mask_v2(
    selector: Union[List, Tuple]
) -> List[Union[bool, Tuple[bool]]]:
    "" "Return a nested boolean mask for selecting elements in a list of lists
    or tuples.

    Args:
        - selector (Union[List, Tuple]): The list or tuple of lists or tuples
        to filter.
    Returns:
        List[Union[bool, Tuple[bool]]]: The boolean mask to use for filtering
        the elements of the list or tuple.
    "" "
    discard_mask = []
    for child in selector:
        if child is None:
            discard_mask.append(True)
        elif not isinstance(child, tuple):
            discard_mask.append(False)
        else:
            discard_mask.append(get_discard_mask_v2(child))

    if isinstance(selector, tuple):
        return tuple(discard_mask)
    else:
        return discard_mask"""


"""def reduce_discard_mask(
    discard_mask: List[Union[bool, List[bool], Tuple[bool]]]
) -> List[Union[bool, List[bool], Tuple[bool]]]:
    "" "Reduces the given discard mask by removing trailing False values.

    Parameters:
    - discard_mask (List): The input discard mask.

    Returns:
    - List: The reduced discard mask.

    Example:
    >>> reduce_discard_mask([(1, 0, 1, 0, 0), 0, 1, (1, 0), 0, 0])
    [(1, 0, 1), 0, 1, (1,)]
    "" "
    if discard_mask is None:
        return None
    discard_mask = list(trim_vector(discard_mask, False))
    new_discard_mask = []
    for item in discard_mask:
        if isinstance(item, tuple):
            new_discard_mask.append(trim_vector(item, False))
        elif isinstance(item, list):
            new_discard_mask.append(list(trim_vector(item, False)))
        else:
            new_discard_mask.append(item)
    return new_discard_mask"""

"""


def extract_names(selector: PandasSelector) -> List[str]:
    "" "Extract the names from a selector.

    This function extracts the names from a selector, whether it is a
    dictionary, an object of type `Index` or `MultiIndex`, or a list.

    Parameters
        selector: The object to extract the names from.

    Returns
        List[str]: A list of the names extracted from the selector.
    "" "
    if isinstance(selector, dict):
        return list(selector.keys())
    if isinstance(selector, (pd.Index, pd.MultiIndex)):
        return list(selector.names)
    if isinstance(selector, list):
        return get_vector_names(selector)
    return []
"""


"""
def get_vectors_list(selector: Selector) -> List[np.ndarray]:
    "" "Convert a selector to a list of vectors.

    Args:
        - selector (Selector): The selector to convert.

    Returns:
        List[np.ndarray]: The list of vectors obtained from the selector.
    "" "
    vectors = []
    if selector is None:
        return None
    elif isinstance(selector, (pd.DataFrame, pd.MultiIndex)):
        vectors.extend(
            [reduce_vector(series) for _, series in selector.items()]
        )
    elif isinstance(selector, (pd.Series, pd.Index)):
        vectors.append(reduce_vector(selector))
    elif isinstance(selector, tuple):
        vectors.append(tuple(reduce_vector(item) for item in selector))
    elif isinstance(selector, list):
        for item in selector:
            if item is None:
                vectors.append(None)
            elif isinstance(item, (pd.DataFrame, pd.MultiIndex)):
                vectors.extend(
                    [reduce_vector(series) for _, series in item.items()]
                )
            elif isinstance(item, (pd.Series, pd.Index)):
                vectors.append(reduce_vector(item))
            elif isinstance(item, tuple):
                vectors.append(tuple(reduce_vector(e) for e in item))
            elif isinstance(item, list):
                vectors.append(reduce_vector(item))
            else:
                raise TypeError(f"Unknown list item type {type(item)}")
    else:
        raise TypeError(
            f"The given selector of type {type(selector)} is not of a"
            " supported type (pd.Series, pd.Index, pd.DataFrame,"
            " pd.MultiIndex, list, tuple)"
        )
    return vectors
"""
