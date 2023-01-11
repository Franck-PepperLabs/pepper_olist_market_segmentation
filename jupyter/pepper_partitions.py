import random
import numpy.random
import numpy as np


rng = np.random.default_rng()
dbg = False


def dbg_trace(msg):
    if dbg:
        print(msg)


def constant_size_partition(size, n):
    """Generate a list of n sizes, with a total sum of size,
    such that each size is size/n (+ 1 if necessary for size % n elements)"""
    # if n > size:
    #    raise ValueError(f"n ({n}) cannot be > size ({size})")
    m = size // n
    r = size % n
    sizes = np.asarray([m] * n)
    rem = np.asarray([1] * r + [0] * (n - r))
    random.shuffle(rem)
    sizes += rem
    return sizes


# TODO : amélioration possible : le correcteur est constant pur
# TODO : Compléter la partie maths + ajouter un facteur de pente autre que 1
# TODO : Puis la dernière étape sera de sortir de la seule progression linéaire
def uniform_size_partition(size, n, p=1):
    """
    Generate a list of n sizes, with a total sum of size,
    such that each size is uniformly distributed between
    min_size and max_size/n.

    Parameters
    ----------
    size: int
        The total size of the partition.
    n: int
        The number of elements in the partition.
    p: int, optional
        The maximum deviation from the mean size for each element.
        Default is 1.

    Returns
    -------
    sizes: List[int]
        A list of n sizes with a total sum of size, and a uniform distribution
        between min_size and max_size/n.

    Raises
    ------
    Warning
        If n is greater than size or p is greater than m - 1.

    Examples
    --------
    >>> uniform_size_partition(10, 3)
    [3, 3, 4]
    >>> uniform_size_partition(10, 3, 2)
    [2, 4, 4]
    """
    # if n > size:
    #    raise ValueError(f"n ({n}) cannot be > size ({size})")

    m = size / n
    # min_size = int(m - p) + 1
    # max_size = int(m + p)
    if p > m - 1:
        dbg_trace(f"p ({p}) shouldn't be > (size / n) - 1 ({m - 1})")

    sizes = constant_size_partition(size, n)
    n_mod_parts = 2 * p + 1
    mod = np.asarray(list(range(-p, p+1)) * ((n // n_mod_parts) + 1))
    mod = mod[:n]
    zero = sum(mod)
    # print('mod', mod, sum(mod))
    if zero != 0:
        z = abs(zero)
        rem = constant_size_partition(z, n)
        if zero > 0:
            rem = -rem
        # print('rem', rem)
        mod += rem
    random.shuffle(mod)
    return sizes + mod


def normal_size_partition(size, n, p=1, sigma=1):
    """
    Generate a list of n sizes, with a total sum of size,
    such that each size is distributed normally around the mean size,
    with a maximum deviation of s.

    Parameters
    ----------
    size: int
        The total size of the partition.
    n: int
        The number of elements in the partition.
    p: int, optional
        The maximum deviation from the mean size for each element.
        Default is 1.
    sigma: float, optional
        The standard deviation of the normal distribution. Default is 1.

    Returns
    -------
    sizes: List[int]
        A list of n sizes with a total sum of size, and a normal distribution
        around the mean size, with a maximum deviation of p.

    Raises
    ------
    Warning
        If n is greater than size or p is greater than m - 1.

    Examples
    --------
    >>> normal_size_partition(10, 3)
    [3, 3, 4]
    >>> normal_size_partition(10, 3, 2, 1)
    [1, 5, 4]
    """
    if n > size:
        raise ValueError(f"n ({n}) cannot be > size ({size})")

    m = size // n
    # p = 3 * sigma
    # min_size = int(m - p) + 1
    # max_size = int(m + p)
    # print('min', min_size, 'max', max_size)
    # if p > m - 1:
    #    raise ValueError(f"3 * sigma + 1 ({p + 1}) cannot be > m ({m})")

    sizes = constant_size_partition(size, n)
    # n_mod_parts = 2 * p + 1
    # mod = np.asarray(list(range(-p, p+1)) * ((n // n_mod_parts) + 1))
    mod = numpy.clip(
        ((p / (3 * sigma)) * np.random.normal(0, sigma, size=n)).astype(int),
        -(m - 1), (m - 1)
    )
    print('mod', mod, sum(mod))
    # mod = mod[:n]
    zero = sum(mod)
    if zero != 0:
        z = abs(zero)
        # pb si z > n, on déborde => correctifs avec des 2, etc
        # mais cela signifie qu'on est à saturation car sigma est trop grand
        # ce qui entraine une augmentation significative des valeurs au delà
        # de l'intervalle
        # rem =  np.asarray([-1 if zero > 0 else 1] * z + [0] * (n - z))
        # random.shuffle(rem)
        # mod += rem
        rem = constant_size_partition(z, n)
        if zero > 0:
            rem = -rem
        # print('rem', rem)
        mod += rem
    random.shuffle(mod)
    return sizes + mod
