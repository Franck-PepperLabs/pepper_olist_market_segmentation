import timeit
from typing import Any, Callable, Tuple
import pandas as pd
import numpy as np


def do_unitary_tests(f, tests_io_list, equals_func=lambda x, y: x == y):
    ok_count, ko_count = 0, 0
    args_list = [t[:-1] for t in tests_io_list]
    outputs_list = [t[-1] for t in tests_io_list]
    for i, (args, awaited_output) in enumerate(zip(args_list, outputs_list)):
        real_output = f(*args)
        if not equals_func(real_output, awaited_output):
            ko_count += 1
            print(f"test {i}: {f.__name__}({args})")
            print(f"\treturned ✘ {real_output}")
            print(f"\tand not  ✔ {awaited_output}")
        else:
            ok_count += 1

    if ko_count == 0:
        print(f'The whole {ok_count} tests are successful!')
    else:
        def is_are(ko_count):
            return 'is' if ko_count == 1 else 'are'
        print(
            f"{ok_count} tests are successful "
            f"but {ko_count} {is_are(ko_count)} not."
        )


def zip_functions(*funcs: Callable) -> Callable:
    """Create a new function that returns a tuple
    of the return values of the given functions.

    The new function will accept the same arguments as the given functions,
    and will return a tuple of their return values.
    If a function raises an exception,
    the exception will be raised by the new function.

    Parameters
    - *funcs : Callable : Functions to be combined.

    Returns
    - Callable : The new function.

    Examples
    >>> def f1(x: int) -> int:
    ...     return x + 1
    >>> def f2(x: int) -> str:
    ...     return str(x)
    >>> g = zip_functions(f1, f2)
    >>> g(1)
    (2, '1')
    """
    def zipped_function(*args: Any, **kwargs: Any) -> Tuple:
        return tuple(f(*args, **kwargs) for f in funcs)
    return zipped_function


def test_f_with_args(f, args, test_name=None, repeat=1_000, show_result=False):
    if test_name is None:
        test_name = f.__name__
    if not isinstance(args, tuple) or len(args) == 0:
        args = (args, )
    res = f(*args)
    time = timeit.timeit(lambda: f(*args), number=repeat)
    print(f'{test_name} ({round(time, 4)})')
    if show_result:
        print('\t', res)
    return res, time


def benchmarck(functions_list, args, repeat=1_000, show_result=False):
    for f in functions_list:
        test_f_with_args(f, args, repeat=repeat, show_result=show_result)


""" Pandas test datasets
"""


def get_example_dataset_1():
    return pd.DataFrame.from_dict({
        'index': [('a', 'b'), ('a', 'c')],
        'columns': [('x', 1), ('y', 2)],
        'data': [[1, 3], [2, 4]],
        'index_names': ['n1', 'n2'],
        'column_names': ['z1', 'z2']
    }, orient='tight')


def get_example_dataset_2():
    return pd.DataFrame.from_dict({
        'row_1': [3, 2, 1, 0],
        'row_2': ['a', 'b', 'c', 'd']
    }, orient='index', columns=['A', 'B', 'C', 'D'])


def get_monster_row_index():
    return pd.MultiIndex.from_product([
        ['A', 'B'],
        [1, 2, 3],
        [True, False],
        [(2, 3), (5, 7), (11, 13)],
        [(0), (1, ), (1, 2), (3, 4, 5), (6, 7, 8, 9)]
    ], names=['L', 'int', 'bool', 'primes', 'irr_list'])


def get_monster_col_index():
    return pd.MultiIndex.from_product([
        ['A', 'B'],
        [(None, None), (None, 1), (1, None), (1, 1)],
        [(np.nan,), (np.nan, np.nan)]
    ], names=['L', 'n1', 'nan'])


def get_monster_dataframe():
    monster_data = pd.DataFrame(
        index=get_monster_row_index(),
        columns=get_monster_col_index()
    )
    for i in range(monster_data.shape[0]):
        for j in range(monster_data.shape[1]):
            i_j = i + j
            i_j = tuple(k + (j % 3) for k in range(i % 5))
            monster_data.iloc[i, j] = i_j

    return monster_data
