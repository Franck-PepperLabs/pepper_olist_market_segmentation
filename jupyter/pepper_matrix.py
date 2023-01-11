import copy
import operator
import itertools
from typing import *
import json


def rindex(vector, value):
    if vector is None or value not in vector:
        return None
    return len(vector) - operator.indexOf(reversed(vector), value) - 1


def drop_nones(vector):
    """
    Return a tuple with the elements of `vector` that are not `None`.

    Args:
        - vector (Union[List, Tuple]): The list or tuple to filter.
    Returns:
        Tuple: The tuple with the elements of `vector` that are not `None`.

    Examples:
    >>> drop_nones(())
    ()
    >>> drop_nones((None))
    None
    >>> drop_nones((None,))
    ()
    >>> drop_nones((None, None))
    ()
    >>> drop_nones((1,))
    (1,)
    >>> drop_nones((1, None))
    (1,)
    >>> drop_nones((1, None, 2))
    (1, 2)
    >>> drop_nones((None, None, 1, None, 2, None, 3, None, None))
    (1, 2, 3)
    """
    if vector is None:
        return None
    return tuple(x for x in vector if x is not None)


def cartesian_product(vectors_list):
    """Return the cartesian product of the input list of iterables.

    Args:
    - vectors_list (List[Iterable]): The list of iterables to compute the
        cartesian product of.

    Returns:
    - List[Tuple]: The cartesian product of the input list of iterables.

    Examples:
    >>> cartesian_product([])
    []
    >>> cartesian_product([[1]])
    [(1,)]
    >>> cartesian_product([[1, 2]])
    [(1,), (2,)]
    >>> cartesian_product([[1, 2], [3, 4]])
    [(1, 3), (1, 4), (2, 3), (2, 4)]
    >>> cartesian_product([[1, 2], [3, 4], [5, 6]])
    [(1, 3, 5), (1, 3, 6), (1, 4, 5), (1, 4, 6), (2, 3, 5), (2, 3, 6),
     (2, 4, 5), (2, 4, 6)]
    """
    return list(itertools.product(*vectors_list))


def zip_vectors(vectors_list: List[List]) -> List[Tuple]:
    """Create a list of tuples by combining the elements of the given lists
    element-wise.

    Args:
    - vectors_list (List[List]): The lists to combine element-wise.

    Returns:
    - List[Tuple]: A list of tuples where each tuple contains the i-th
        element of each list in vectors_list.

    Examples:
    >>> zip_vectors([[1, 2, 3], [4, 5, 6]])
    [(1, 4), (2, 5), (3, 6)]
    >>> zip_vectors([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    [(1, 4, 7), (2, 5, 8), (3, 6, 9)]
    >>> zip_vectors([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    [(1, 4, 7, 10), (2, 5, 8, 11), (3, 6, 9, 12)]
    """
    return list(zip(*vectors_list))


def shift_vector(
    vector: Tuple[Any],
    on_left: bool,
    n_steps: int,
    value: Optional[Any] = None
) -> Tuple[Any]:
    """Shifts the given vector to a specified direction by padding it with a
    specified value.

    Parameters:
    - vector (Tuple[Any]): The input vector to shift.
    - on_left (bool): Whether to shift the vector on the left. If True, shift
        on the left. Otherwise, shifts on the right.
    - n_steps (positive int): number of steps
    - value (Optional[Any]): The value to pad the vector with. If not
        specified, the default value is None.

    Returns:
    - Tuple[Any]: The shifted vector.

    Example:
    >>> shift_vector((1, 2, 3), True, 0, 0)
    (1, 2, 3)
    >>> shift_vector((1, 2, 3), False, 2, 0)
    (0, 0, 1, 2, 3)
    """
    if n_steps < 0:
        raise ValueError(f"n_steps must be a positive integer, not {n_steps}")
    if n_steps == 0:
        return vector
    padding = [value, ] * n_steps
    if on_left:
        return tuple(list(vector) + padding)
    else:
        return tuple(padding + list(vector))


def left_shift_vector(
    vector: Tuple[Any],
    n_steps: int,
    value: Optional[Any] = None
) -> Tuple[Any]:
    """Shifts the given vector to the left by padding it with a specified
    value.

    Parameters:
    - vector (Tuple[Any]): The input vector to shift.
    - value (Optional[Any]): The value to pad the vector with. If not
        specified, the default value is None.

    Returns:
    - Tuple[Any]: The left-shifted vector.

    Example:
    >>> left_shift_vector((1, 2, 3), 2)
    (1, 2, 3, None, None)
    >>> left_shift_vector((1, 2, 3), 2, 1)
    (1, 2, 3, 1, 1)
    """
    return shift_vector(vector, True, n_steps, value)


def right_shift_vector(
    vector: Tuple[Any],
    n_steps: int,
    value: Optional[Any] = None
) -> Tuple[Any]:
    """Shifts the given vector to the right by padding it with a specified
    value.

    Parameters:
    - vector (Tuple[Any]): The input vector to shift.
    - value (Optional[Any]): The value to pad the vector with. If not
        specified, the default value is None.

    Returns:
    - Tuple[Any]: The right-shifted vector.

    Example:
    >>> right_shift_vector((1, 2, 3), 2, 0)
    (0, 0, 1, 2, 3)
    """
    return shift_vector(vector, False, n_steps, value)


def _left_trailing_len(vector, value):
    if vector is None:
        return 0
    count = 0
    if value is None:
        for e in vector:
            if e is None:
                count += 1
            else:
                return count
    else:
        for e in vector:
            if e == value:
                count += 1
            else:
                return count
    return count


def _right_trailing_len(vector, value):
    return _left_trailing_len(vector[::-1], value)


def left_trim_vector(
    vector: Tuple[Any],
    value: Optional[Any] = None
) -> Tuple[Any]:
    """Trims the given vector by removing a specified value from the left
    side.

    Parameters:
    - vector (Tuple[Any]): The input vector to trim.
    - value (Optional[Any]): The value to remove from the vector. If not
        specified, the default value is None.

    Returns:
    - Tuple[Any]: The left-trimmed vector.

    Example:
    >>> left_trim_vector((5, 5, 1, 2, 3, 4, 5, 5), value=5)
    (1, 2, 3, 4, 5, 5)
    """
    if vector is None:
        return None
    trailing_len = _left_trailing_len(vector, value)
    return tuple(vector[trailing_len:])


def right_trim_vector(
    vector: Tuple[Any],
    value: Optional[Any] = None
) -> Tuple[Any]:
    """Trims the given vector by removing a specified value from the right
    side.

    Parameters:
    - vector (Tuple[Any]): The input vector to trim.
    - value (Optional[Any]): The value to remove from the vector. If not
        specified, the default value is None.

    Returns:
    - Tuple[Any]: The left-trimmed vector.

    Example:
    >>> left_trim_vector((5, 5, 1, 2, 3, 4, 5, 5), value=5)
    (5, 5, 1, 2, 3, 4)
    """
    if vector is None:
        return None
    trailing_len = _right_trailing_len(vector, value)
    if trailing_len == 0:
        return vector
    return tuple(vector[:-trailing_len])


def trim_vector(
    vector: Tuple[Any],
    value: Optional[Any] = None,
) -> Tuple[Any]:
    """Trims the given vector by removing the specified value on the both
    sides.

    Parameters:
    - vector (Tuple[Any]): The input vector to trim.
    - value (Optional[Any]): The value to remove from the vector. If not
        specified, the default value is None.

    Returns:
    - Tuple[Any]: The trimmed vector.

    Example:
    >>> trim_vector((1, 2, 3, None, None))
    (1, 2, 3)
    >>> trim_vector((5, 5, 1, 2, 3, 5, 5), 5)
    (1, 2, 3)
    """
    return right_trim_vector(left_trim_vector(vector, value), value)


class Vector(tuple):
    def __init__(self, data: Tuple[Any]):
        self.data = data

    def __lshift__(self, n_steps: int) -> "Vector":
        return Vector(left_shift_vector(self, n_steps))

    def __rshift__(self, n_steps: int) -> "Vector":
        return Vector(right_shift_vector(self, n_steps))

    def trim(self):
        return Vector(trim_vector(self))


"""def _extend_row_with_trailing_nones(row, n_nones):
    # use left_shift_vector(row, n_nones) instead
    return row + (None,) * n_nones"""


def _count_trailing_nones_in_row(row):
    if row is None:
        return 0
    count = 0
    for e in row[::-1]:
        if e is None:
            count += 1
        else:
            return count
    return count


class CompressedMatrix:

    def __init__(self, data, uniformity_guaranteed=False):
        self.data = list(data)
        self._uniformity_guaranteed = uniformity_guaranteed
        if uniformity_guaranteed:
            self._is_uniform = True

    def __copy__(self):
        return CompressedMatrix(self.data, self._uniformity_guaranteed)

    def __deepcopy__(self, memo):
        return CompressedMatrix(
            copy.deepcopy(self.data, memo),
            self._uniformity_guaranteed
        )

    def __str__(self) -> str:
        return str(self.data)

    def __repr__(self):
        return f'CompressedMatrix({self.data})'

    def _repr_pretty_(self, p, cycle):
        if cycle:
            return 'CompressedMatrix(...)'
        p.text(f'CompressedMatrix({self.data})')

    @property
    def is_uniform(self):
        if not self._uniformity_guaranteed:
            self._is_uniform = self.is_width_uniform()
            if self._is_uniform:
                self._uniformity_guaranteed
        return self._is_uniform

    @is_uniform.setter
    def is_uniform(self, value):
        self._uniformity_guaranteed = value
        self._is_uniform = value

    def transpose(self):
        return CompressedMatrix(list(zip(*self.data)))

    def get_max_row_length(self):
        if self.data is None or len(self.data) == 0:
            return 0
        matrix_width = 0
        for row in self.data:
            row_len = len(row)
            if row_len > matrix_width:
                matrix_width = row_len
        return matrix_width

    def width(self):
        return self.get_max_row_length()

    def height(self):
        if self.data is None:
            return 0
        return len(self.data)

    def is_width_uniform(self):
        if self.data is None or len(self.data) == 0:
            return True
        matrix_width = self.data[0]
        for row in self.data[1:]:
            row_len = len(row)
            if row_len != matrix_width:
                return False
        return True

    def uniformize(self):
        if self._uniformity_guaranteed:
            return self

        self._is_uniform = self.is_width_uniform()
        if self._is_uniform:
            self._uniformity_guaranteed = True
            return self

        if self.data is not None and len(self.data) > 0:
            max_row_length = self.get_max_row_length()
            new_matrix = []
            for row in self.data:
                row_trailling_len = max_row_length - len(row)
                if row_trailling_len:
                    row = left_shift_vector(row, row_trailling_len)
                new_matrix.append(row)
        return CompressedMatrix(new_matrix, uniformity_guaranteed=True)

    def trim(self):
        """Removes trailing `None` elements from the right side of each row in
        the matrix.

        Returns:
            CompressedMatrix: A new `CompressedMatrix` object with trailing
                None` elements removed from each row.
        """
        # Return a new CompressedMatrix object with the updated data
        return CompressedMatrix([right_trim_vector(row) for row in self.data])

    def transpose_with_homogenization(self):
        return self.uniformize().transpose()

    def discard_columns(self, discard_mask, verbose=False):
        """Discard columns from a compressed matrix.

        Parameters
        ----------
        discard_mask : list of bool
            A list of boolean values indicating which columns to discard (True)
            and which to keep (False). If the length of the mask is greater
            than the width of the matrix, the excess values are ignored.
        verbose : bool, optional
            If set to True, the function will print intermediate steps for
            debugging purposes.

        Returns
        -------
        CompressedMatrix
            A new CompressedMatrix object with the specified columns removed.

        """
        # If the matrix is not uniform, we guarantee uniformity by creating a
        # new object
        if not self._uniformity_guaranteed:
            self = self.uniformize()
        if verbose:
            print('uniformized matrix:', self)

        # Transpose the matrix to transform the problem of column removal
        # into a problem of row removal
        t_matrix = self.transpose()
        if verbose:
            print('transposed matrix:', t_matrix)

        # Measure the height of the transposed matrix (i.e. the width of the
        # original matrix) and the length of the discard mask
        t_matrix_len = t_matrix.height()
        discard_mask_len = len(discard_mask)
        if verbose:
            print(
                't_matrix_len:', t_matrix_len,
                'discard_mask_len:', discard_mask_len
            )

        # If the height of the transposed matrix is greater than the size of
        # the discard mask we store the rows beyond the size of the mask
        # car elles ne seront pas affectées, c'est-à-dire préservées et
        # restituées telles quelles
        t_data_remainder = None
        t_data_filtered = None
        if t_matrix_len > discard_mask_len:
            remainder_len = t_matrix_len - discard_mask_len
            t_data_remainder = t_matrix.data[-remainder_len:]
            t_data_filtered = t_matrix.data[:-remainder_len]
            if verbose:
                print('t_data_remainder:', t_data_remainder)
        else:
            t_data_filtered = t_matrix.data.copy()

        # TODO : EN - si la taille du masque est supérieure à la hauteur
        #  de la matrice transposée
        # le contenu du masque au delà de la de cette hauteur est sans effet
        if t_matrix_len < discard_mask_len:
            discard_mask = discard_mask[:t_matrix_len]

        # On filtre les colonnes à conserver en fonction du masque
        # de suppression
        t_data_filtered = [
            t_data_filtered[i]
            for i, discard in enumerate(discard_mask)
            if not discard
        ]
        if verbose:
            print('discarded t_matrix:', t_data_filtered)

        if t_data_filtered is None and t_data_remainder is None:
            return CompressedMatrix([])

        if t_data_filtered is None:
            t_data_filtered = t_data_remainder
        else:
            if t_data_remainder is not None:
                t_data_filtered += t_data_remainder

        t_matrix_filtered = CompressedMatrix(
            t_data_filtered,
            uniformity_guaranteed=True
        )

        matrix_filtered = t_matrix_filtered.transpose()

        return matrix_filtered.trim()


""" Dict of tuple keys
"""


def _is_single(x):
    return not isinstance(x, tuple)


def tree_to_flatten_dict(d: dict) -> dict:
    """
    Transforms a nested dictionary (tree) into a flatten dictionary
    (dict with tuple keys). The tuple keys represent the path from
    the root of the tree to the corresponding leaf.

    Args:
        d (dict): the input nested dictionary
    Returns:
        dict: the flatten dictionary
    """
    flat_dict = {}
    for key, obj in d.items():
        if isinstance(obj, dict):
            # Recursively flatten the children dictionaries
            child_flat_dict = tree_to_flatten_dict(obj)
            # Update the flat_dict with the children key/value pairs,
            # with the current key as a prefix
            flat_dict.update(
                {(key, k): v for k, v in child_flat_dict.items()}
            )
        else:
            # Leaf node, add the key/value pair to the flat_dict
            flat_dict[key] = obj
    return flat_dict


def max_vector_key_len(d: dict) -> int:
    """
    Finds the maximum length of the tuple keys in the input dictionary.

    Args:
        d (dict): the input dictionary
    Returns:
        int: the maximum length of the tuple keys
    """
    max_k_len = 0
    for k in d.keys():
        k_len = 0
        if _is_single(k):
            # Single item tuple, length is 1
            k_len = 1
        else:
            # Tuple of multiple items, length is the number of items
            k_len = len(k)
        if k_len > max_k_len:
            max_k_len = k_len
    return max_k_len


def uniformize_keys(d: dict) -> dict:
    """
    Transforms a dictionary with tuple keys of varying lengths into
    a dictionary with tuple keys of the same length, by padding the
    shorter keys with `None` values.

    Args:
        d (dict): the input dictionary with tuple keys
    Returns:
        dict: the dictionary with tuple keys of the same length
    """
    width = max_vector_key_len(d)
    new_dict = {}
    for k, v in d.items():
        new_k = None
        if _is_single(k):
            # If k is a single item tuple, pad it with None values
            # to the desired width
            new_k = left_shift_vector((k, ), width - 1)
        else:
            # Pad the tuple with None values to the desired width
            new_k = left_shift_vector(k, width - len(k))
        new_dict[new_k] = v
    return new_dict


def right_trim_keys(d: dict) -> dict:
    """
    Trim right all tuple keys of the input dictionary, removing empty elements.

    Args:
        d (dict): the input dictionary with tuple keys
    Returns:
        dict: the output dictionary with tuple keys trimmed to the right
    """
    new_dict = {}
    for k, v in d.items():
        # Trim the key
        new_k = right_trim_vector(k)
        # Add the key/value pair to the new dictionary
        new_dict[new_k] = v
    return new_dict


def flatten_to_tree_dict(d: dict) -> dict:
    """
    Transforms a flatten dictionary (dict with tuple keys) into a nested
    dictionary (tree). The tuple keys are assumed to represent the path
    from the root of the tree to the corresponding leaf.

    Args:
        d (dict): the input flatten dictionary
    Returns:
        dict: the nested dictionary
    """
    # Initialize the new dictionary and a dict to store the children
    # dictionaries
    new_dict = {}
    children_dicts = {}
    # Iterate over the key/value pairs in the input dictionary
    for k, v in d.items():
        if _is_single(k):
            # Key is a single element tuple, add it as a key in the
            # new dictionary
            new_dict[k] = v
        elif len(k) == 1:
            # Key is a tuple with a single element, add it as a key
            # in the new dictionary
            new_dict[k[0]] = v
        else:
            # Key is a tuple with more than one element, add it as
            # a key in the new dictionary with the first element
            # as the key and the rest of the elements as a new
            # tuple key in a nested dictionary
            if k[0] not in new_dict:
                # If the first element of the tuple is not already
                # a key in the new dictionary, it is added as an
                # empty dictionary.
                new_dict[k[0]] = {}
                children_dicts[k[0]] = new_dict[k[0]]
            # Add the key/value pair to the children dictionary,
            # with the current key as a prefix
            new_dict[k[0]][k[1:]] = v
    # Iterate over the children dictionaries
    for k, cd in children_dicts.items():
        # Recursively transform the children dictionaries into nested
        # dictionaries
        new_dict[k] = flatten_to_tree_dict(cd)
    return new_dict


def dict_index__str__(tree_index: dict) -> str:
    def default_rep(x):
        if _is_single(x):
            return str(x)
        else:
            return str(list(x))

    def strize_keys(tree_dict: dict):
        new_dict = {}
        for k, v in tree_dict.items():
            new_dict[str(k)] = (
                strize_keys(v)
                if isinstance(v, dict)
                else v
            )
        return new_dict

    return json.dumps(
        strize_keys(tree_index),
        indent=2,
        default=default_rep)


def permut(cols, perm):
    return [cols[j] for j in perm]


def permut_cols(data, perm):
    return data[permut(data.columns, perm)]


def move_col_to(data, col_name, pos):
    cur_pos = data.columns.get_loc(col_name)
    if cur_pos == pos:
        return data  # identity
    n = len(data.columns)
    perm = i = j = None
    i, j = (pos, cur_pos) if pos < cur_pos else (cur_pos, pos)
    before = list(range(0, i))
    after = list(range(j+1, n))
    between = list(range(i+1, j))
    perm = before + [j] + [i] + between + after
    return permut_cols(data, perm)
