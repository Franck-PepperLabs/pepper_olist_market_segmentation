"""
This module contains functions to load and analyze data tables
for a e-commerce dataset.
"""

from typing import *
import re
from sys import getsizeof
from datetime import *
import time
import requests
from unidecode import unidecode
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_samples,
    silhouette_score,
    davies_bouldin_score,
)
from IPython.display import display
from pepper_commons import format_iB, bold, print_subtitle
from pepper_selection import filtered_copy
from pepper_cache import Cache


def load_table(name):
    """Load a data table from a file.
    """
    filepath = f'../data/olist_{name}_dataset.csv'
    data = pd.read_csv(filepath, dtype=object)
    data.columns.name = 'raw_' + name
    return data


def load_product_categories_table():
    filepath = '../data/product_category_name_translation.csv'
    data = pd.read_csv(filepath, dtype=object)
    data.columns.name = 'raw_product_categories'
    return data


def table_content_analysis(data):
    """Perform a content analysis of a data table.
    """
    # Print basic infos about the data table
    print_subtitle('basic infos')
    print(bold('dimensions'), ':', data.shape)
    print(bold('size'), ':', *format_iB(getsizeof(data)))

    # Print detailed info about the data table
    print(bold('info'), ':')
    data.info()

    # Print statistical summary of the data table
    print(bold('stats'), ':')
    display(data.describe(include='all').T)

    # Print a sample of the data table
    print(bold('content'), ':')
    display(data)


# as the tables are small we preload them in the raw object format
_raw_customers = load_table('customers')
_raw_geolocations = load_table('geolocation')
_raw_order_items = load_table('order_items')
_raw_order_payments = load_table('order_payments')
_raw_order_reviews = load_table('order_reviews')
_raw_orders = load_table('orders')
_raw_products = load_table('products')
_raw_sellers = load_table('sellers')
_raw_product_categories = load_product_categories_table()


# globals for pk-indexed unitary tables caching
_cached_order_items = None
_cached_order_items_mi = None
_cached_orders = None
_cached_customer_orders = None
_cached_customers = None
_cached_products = None
_cached_sellers = None
_cached_order_payments = None
_cached_order_reviews = None
_cached_product_categories = None
_cached_geolocations = None


# globals for pk-indexed merged tables caching
_cached_categorized_products = None
_cached_products_sales = None
_cached_sellers_sales = None
_cached_sales = None

_cached_orders_reviews = None
_cached_customers_orders = None
_cached_orders_payments = None

_cached_customers_orders_payments = None
_cached_customers_orders_reviews = None
_cached_orders_payments_reviews = None

_cached_order_details = None
_cached_sales_details = None

_cached_all_in_one = None

# _cached_order_payments_by_order = None
# _cached_order_payments_by_customer = None


def _is_identifier(identifier: str) -> bool:
    return re.search('^[A-Za-z_][A-Za-z0-9_]*', identifier)


def _set_global_variable(var_name: str, value):
    """
    Set a global variable with the given name and value.

    Parameters:
        - var_name (str): The name of the variable to set.
            The variable name must be a non-empty string containing
            only alphanumeric characters.
        - value: The value to set the variable to.

    Raises:
        - ValueError: if `var_name` is not a valid string
        - NameError: if a variable with the given `var_name` does not exist
            in the global scope
    """
    if not (isinstance(var_name, str) and _is_identifier(var_name)):
        raise ValueError(f"`var_name` {var_name} is not a valid identifier")
    if var_name in globals():
        globals()[var_name] = value
    else:
        raise NameError(f"`var_name` {var_name} not found in globals")


def _get_cache(table_name: str) -> Union[pd.DataFrame, None]:
    """
    Get the cache of a table.

    Parameters:
        - table_name (str): The name of the table.
            The table name must be a non-empty string.

    Returns:
        - Union[pd.DataFrame, None]: The cache of the table if it exists
            and it is a DataFrame, None otherwise

    Raises:
        - ValueError: if `table_name` is not a valid string
        - NameError: if a variable with the name '_cached_' + table_name
            does not exist in the global scope
        - TypeError: if the cache is not None or a pd.DataFrame
    """
    if not (isinstance(table_name, str) and _is_identifier(table_name)):
        raise ValueError(
            f"`table_name` {table_name} is not a valid identifier"
        )
    cache_name = '_cached_' + table_name
    if cache_name not in globals():
        raise NameError(f"{cache_name} not found in globals")
    cache = globals()[cache_name]
    if not (cache is None or isinstance(cache, pd.DataFrame)):
        raise TypeError(
            f"{cache_name} is not a DataFrame"
            f"(its type is {type(cache)})"
        )
    return cache


def _set_cache(table_name: str, value: Union[pd.DataFrame, None]) -> None:
    """
    Set the cache of a table.

    Parameters:
        - table_name (str): The name of the table.
            The table name must be a non-empty string containing only
            alphanumeric characters.
        - value: The value to set the cache to. The value must be None
            or a pd.DataFrame

    Returns:
        The cache.

    Raises:
        - ValueError: if `table_name` is not a valid string
        - TypeError: if the `value` is not None or a pd.DataFrame
    """
    if not (isinstance(table_name, str) and _is_identifier(table_name)):
        raise ValueError(
            f"`table_name` {table_name} is not a valid identifier"
        )
    if not (value is None or isinstance(value, pd.DataFrame)):
        raise TypeError("`value` is not None or a DataFrame")
    cache_name = '_cached_' + table_name
    _set_global_variable(cache_name, value)
    return value


def _get_cache_loader(table_name: str) -> Callable[[], Type[pd.DataFrame]]:
    """
    Get the cache loader function of a table.

    Parameters:
        - table_name (str): The name of the table.
            The table name must be a non-empty string containing only
            alphanumeric characters.

    Returns:
        - Callable[[], Type[pd.DataFrame]]: The cache loader function.

    Raises:
        - ValueError: if `table_name` is not a valid string
        - NameError: if a variable with the name '_load_cached_' + table_name
            does not exist in the global scope
    """
    if not (isinstance(table_name, str) and _is_identifier(table_name)):
        raise ValueError(
            f"`table_name` {table_name} is not a valid identifier"
        )
    loader_name = '_load_cached_' + table_name
    if loader_name not in globals():
        raise NameError(f"{loader_name} not found in globals")
    loader = globals()[loader_name]
    if not isinstance(loader, Callable):
        raise TypeError(
            f"{loader_name} is not a Callable"
            f"(its type is {type(loader)})"
        )
    return loader


def _init_cache(table_name: str) -> None:
    """
    Initialize the cache for a table.
    If the cache for the table does not exist, it is loaded using
    the corresponding cache loader.

    Parameters:
    - table_name (str): The name of the table.
        The table name must be a non-empty string containing
        only alphanumeric characters.

    Returns:
        The cache.

    Raises:
    - ValueError: if `table_name` is not a valid string
    - NameError: if a variable with the name '_cached_' + table_name
        or '_load_cached_' + table_name  does not exist in the global scope
    """
    cache = _get_cache(table_name)
    cache_loader = _get_cache_loader(table_name)
    if cache is None:
        cache = _set_cache(table_name, cache_loader())
    return cache


""" Raw tables with object dtypes
"""


def filter_by_indices(
    data: pd.DataFrame,
    filter_columns: Dict[str, Optional[Iterable]]
) -> pd.DataFrame:
    """Filter a data table by multiple indices.

    Args:
        data (pd.DataFrame): The data table to filter.
        filter_columns (dict): A dictionary mapping column names to indices.
            Rows in the data table that do not have values in the indices
            for the respective columns will be filtered out.

    Returns:
        pd.DataFrame: The data table with rows filtered by the indices.
    """
    # Create a boolean mask that is True for every row
    mask = pd.Series(True, index=data.index)

    # Iterate through the columns to filter by, and update the mask
    # if an index is provided
    for column, index in filter_columns.items():
        if index is not None:
            mask &= data[column].isin(index)

    # Return the data table with the rows that match the mask
    return data[mask]


def get_raw_order_items_v1() -> pd.DataFrame:
    """Deprecated: use `get_raw_order_items` instead.
    Returns the raw order items data table.
    """
    return _raw_order_items.copy()


def get_raw_order_items(
    orders_index: Optional[Iterable] = None,
    products_index: Optional[Iterable] = None,
    sellers_index: Optional[Iterable] = None
) -> pd.DataFrame:
    """Returns the raw order items data table.

    Args:
        orders_index (iterable, optional): The orders index to filter the
            raw data table by.
        products_index (iterable, optional): The products index to filter the
            raw data table by.
        sellers_index (iterable, optional): The sellers index to filter the
            raw data table by.

    Returns:
        pd.DataFrame: The raw order items data table,
            optionally filtered by the orders, products, and sellers indices.
    """
    return filter_by_indices(
        data=_raw_order_items.copy(),
        filter_columns={
            'order_id': orders_index,
            'product_id': products_index,
            'seller_id': sellers_index
        }
    )


def get_raw_orders_v1() -> pd.DataFrame:
    """Deprecated: use `get_raw_orders` instead.
    Returns the raw orders data table.
    """
    return _raw_orders.copy()


def get_raw_orders(
    orders_index: Optional[Iterable] = None,
    customers_index: Optional[Iterable] = None
) -> pd.DataFrame:
    """Returns the raw orders data table.

    Args:
        orders_index (iterable, optional): The orders index to filter the
            raw data table by.
        customers_index (iterable, optional): The customers index to filter
            the raw data table by.

    Returns:
        pd.DataFrame: The raw orders data table,
            optionally filtered by the orders and customers indices.
    """
    return filter_by_indices(
        data=_raw_orders.copy(),
        filter_columns={
            'order_id': orders_index,
            'customer_id': customers_index
        }
    )


def get_raw_customers_v1() -> pd.DataFrame:
    """Deprecated: use `get_raw_customers` instead.
    Returns the raw customers data table.
    """
    return _raw_customers.copy()


def get_raw_customers(
    customers_index: Optional[Iterable] = None,
    unique_customers_index: Optional[Iterable] = None
) -> pd.DataFrame:
    """Returns the raw customers data table.

    Args:
        customers_index (iterable, optional): The customers index to filter
            the raw data table by.
        unique_customers_index (iterable, optional): The unique customers
            index to filter the raw data table by.

    Returns:
        pd.DataFrame: The raw customers data table,
            optionally filtered by the customers and unique customers indices.
    """
    return filter_by_indices(
        data=_raw_customers.copy(),
        filter_columns={
            'customer_id': customers_index,
            'customer_unique_id': unique_customers_index
        }
    )


def get_raw_products_v1() -> pd.DataFrame:
    """Deprecated: use `get_raw_products` instead.
    Returns the raw products data table.
    """
    return _raw_products.copy()


def get_raw_products(
    products_index: Optional[Iterable] = None
) -> pd.DataFrame:
    """Returns the raw products data table.

    Args:
        products_index (iterable, optional): The products index to filter the
            raw data table by.

    Returns:
        pd.DataFrame: The raw products data table,
            optionally filtered by the products index.
    """
    return filter_by_indices(
        data=_raw_products.copy(),
        filter_columns={
            'product_id': products_index
        }
    )


def get_raw_sellers_v1() -> pd.DataFrame:
    """Deprecated: use `get_raw_sellers` instead.
    Returns the raw sellers data table.
    """
    return _raw_sellers.copy()


def get_raw_sellers(
    sellers_index: Optional[Iterable] = None
) -> pd.DataFrame:
    """Returns the raw sellers data table.

    Args:
        sellers_index (iterable, optional): The sellers index to filter the
            raw data table by.

    Returns:
        pd.DataFrame: The raw sellers data table,
            optionally filtered by the sellers index.
    """
    return filter_by_indices(
        data=_raw_sellers.copy(),
        filter_columns={
            'product_id': sellers_index
        }
    )


def get_raw_order_payments_v1() -> pd.DataFrame:
    """Deprecated: use `get_raw_order_payments` instead.
    Returns the raw order payments data table.
    """
    return _raw_order_payments.copy()


def get_raw_order_payments(
    orders_index: Optional[Iterable] = None
) -> pd.DataFrame:
    """Returns the raw order payments data table.

    Args:
        orders_index (iterable, optional): The orders index to filter the
            raw data table by.

    Returns:
        pd.DataFrame: The raw order payments data table,
            optionally filtered by the orders index.
    """
    return filter_by_indices(
        data=_raw_order_payments.copy(),
        filter_columns={
            'order_id': orders_index
        }
    )


def get_raw_order_reviews_v1() -> pd.DataFrame:
    """Deprecated: use `get_raw_order_reviews` instead.
    Returns the raw order reviews data table.
    """
    return _raw_order_reviews.copy()


def get_raw_order_reviews(
    orders_index: Optional[Iterable] = None
) -> pd.DataFrame:
    """Returns the raw order reviews data table.

    Args:
        orders_index (iterable, optional): The orders index to filter the
            raw data table by.

    Returns:
        pd.DataFrame: The raw order reviews data table,
            optionally filtered by the orders index.
    """
    return filter_by_indices(
        data=_raw_order_reviews.copy(),
        filter_columns={
            'order_id': orders_index
        }
    )


def get_raw_product_categories():
    """Returns the raw product categories data table.

    Returns:
        pd.DataFrame: The raw product categories data table.
    """
    return _raw_product_categories.copy()


def get_raw_geolocations():
    """Returns the raw geolocations data table.

    Returns:
        pd.DataFrame: The raw geolocations data table.
    """
    return _raw_geolocations.copy()


"""Tables indexed by their primary key (simple or compound)
"""


def remove_columns_prefixes(
    data: pd.DataFrame,
    prefix: str,
    start_indice: int = 0,
    new_cols_name: Optional[str] = None
) -> None:
    """Remove the given prefix from the dataframe column names.

    Args:
        data (pd.DataFrame): The dataframe whose column names to modify.
        prefix (str): The prefix to remove from the column names.
        start_indice (int, optional): The index of the first column name to
            remove the prefix from. Default is 0.
        new_cols_name (str, optional): The new name for the dataframe column
            names. If not provided, the old column name (if it exists and
            starts with "raw_") will be used.

    Returns:
        None: The dataframe column names are modified in place.
    """
    i = len(prefix)  # The number of characters of the prefix to remove
    old_cols_name = data.columns.name  # Save the old columns name for later

    # Remove the prefix from the column names
    if start_indice == 0:
        data.columns = (
            [c[i:] for c in data.columns]
        )
    else:
        data.columns = (
            list(data.columns[:start_indice])
            + [c[i:] for c in data.columns[start_indice:]]
        )

    # Set the new columns name
    if new_cols_name is None:
        if old_cols_name is not None and old_cols_name.startswith('raw_'):
            new_cols_name = old_cols_name[4:]
    if new_cols_name is not None:
        data.columns.name = new_cols_name


def set_pk_index(
        data: pd.DataFrame,
        pk_columns: Union[str, List[str]],
        pk_name: Optional[str] = None
):
    """Sets the primary key index of the data table.

    Args:
        data (pd.DataFrame): The data table.
        pk_columns (Union[str, List[str]]): The primary key column(s). If a
            single string, it will be used as the primary key index directly.
            If a list, the primary key index will be a tuple of the values in
            the list.
        pk_name (str, optional): The name of the primary key index. If not
            provided, the primary key index name will be the concatenation of
            the primary key column names in parentheses if pk_columns is a
            list, or the single column name if pk_columns is a string.

    Returns:
        None: inplace.
    """
    # If the primary key has only one column, set it as the index directly
    if isinstance(pk_columns, str):
        data.set_index(pk_columns, drop=True, inplace=True)
    elif len(pk_columns) == 1:
        data.set_index(pk_columns[0], drop=True, inplace=True)
    else:
        # If the primary key name is not provided, create it from the
        # primary key column names
        if pk_name is None:
            #pk_name = '(' + ', '.join(pk_columns) + ')'
            pk_name=tuple(pk_columns)

        # Create a Series with the primary key values as tuples
        pk = pd.Series(
            list(zip(*[data[col] for col in pk_columns]))
        ).rename(pk_name)

        # Set the primary key Series as the index of the data table
        data.set_index(pk, inplace=True)
        data.drop(columns=pk_columns, inplace=True)


def set_pk_multi_index(
        data: pd.DataFrame,
        pk_columns: Union[str, List[str]],
        pk_name: Optional[str] = None
):
    """Sets the primary key index of the data table.

    Args:
        data (pd.DataFrame): The data table.
        pk_columns (Union[str, List[str]]): The primary key column(s). If a
            single string, it will be used as the primary key index directly.
            If a list, the primary key index will be a tuple of the values in
            the list.
        pk_name (str, optional): The name of the primary key index. If not
            provided, the primary key index name will be the concatenation of
            the primary key column names in parentheses if pk_columns is a
            list, or the single column name if pk_columns is a string.

    Returns:
        None: inplace.
    """
    # If the primary key has only one column, set it as the index directly
    if isinstance(pk_columns, str):
        data.set_index(pk_columns, drop=True, inplace=True)
    elif len(pk_columns) == 1:
        data.set_index(pk_columns[0], drop=True, inplace=True)
    else:
        # If the primary key name is not provided, create it from the
        # primary key column names
        if pk_name is None:
            # pk_name = '(' + ', '.join(pk_columns) + ')'
            pk_name=tuple(pk_columns)

        """# Create a Series with the primary key values as tuples
        pk = pd.Series(
            list(zip(*[data[col] for col in pk_columns]))
        ).rename(pk_name)"""

        # Set the primary key Series as the index of the data table
        data.set_index(pk_columns, inplace=True, drop=True)
        # data.index.name = pk_name : not copied ?!?


def cast_columns(
    data: pd.DataFrame,
    cols: Union[str, List[str]],
    dtype: Type
):
    """Casts the specified columns in the dataframe to the specified dtype.

    Args:
        data (pd.DataFrame): The dataframe.
        cols (Union[str, list]): The column name(s) to cast. If a single
            string, it will be used as the column name to cast. If a list,
            all the columns in the list will be cast.
        dtype (Type): The type to cast the columns to (e.g. int, float, etc).

    Returns:
        None: inplace.

    Example:
        # Cast the weight, length, height and width columns to float
        >>> cast_columns(products,
        >>>     ['weight_g', 'length_cm', 'height_cm', 'width_cm']
        >>> , float)
        >>>
        >>> # Cast the 'price' column to float
        >>> cast_columns(products, 'price', float)

        # Cast the 'shipping_limit_date' column to 'datetime64[ns]'
        >>> cast_columns(order_items, 'shipping_limit_date', 'datetime64[ns]')

        # Cast the 'price' and 'freight_value' columns to float
        >>> cast_columns(order_items, ['price', 'freight_value'], float)
    """
    if isinstance(cols, str):
        cols = [cols]

    # Check if the s Series can be converted to the target type
    def is_castable(s: pd.Series, dtype: Type) -> bool:
        try:
            s.astype(dtype)
            return True
        except ValueError:
            return False

    for col in cols:
        if not is_castable(data[col], dtype):
            raise ValueError(
                f"Cannot cast column '{col}' to type '{dtype}'. "
                f"Some values are not castable."
            )

    # Proceed the cast
    data[cols] = data[cols].astype(dtype, copy=False)


# Experimental
# TODO : complete it, update the _load_foo functions,
# compare (tests), remove old versions

def _load_pk_indexed_table(
    get_raw_table,
    pk_index_params,
    global_cache_name
):
    # Retrieve the raw data table
    data = get_raw_table()

    # Set the the index of the data table
    set_pk_index(data=data, *pk_index_params)

    # Remove the prefix from the columns of the data table
    remove_columns_prefixes(data, 'seller_')

    # Save to the cache global variable
    _set_global_variable(global_cache_name, data)

    return data.copy()  # TODO : remove after filtered_copy bypassing tests


def get_order_items_v1(index=None) -> pd.DataFrame:
    """Get the order items pk-indexed data table."""
    order_items = get_raw_order_items()
    pk = pd.Series(list(zip(
        order_items.order_id,
        order_items.order_item_id
    ))).rename('(order_id, order_item_id)')
    order_items = order_items.set_index(pk)
    order_items = order_items.drop(columns=['order_id', 'order_item_id'])
    order_items.columns.name = 'order_items'
    return order_items if index is None else order_items.loc[index]


def _load_cached_order_items():
    # Retrieve the raw order items data table,
    # optionally filtered by the orders, products and sellers indexes
    oi = get_raw_order_items()

    # Set the ('order_id', 'order_item_id') columns as the index of the
    # order items data table
    set_pk_index(oi, ['order_id', 'order_item_id'])

    # Set the new columns name
    oi.columns.name = 'order_items'

    # Cast the 'shipping_limit_date' column to 'datetime64[ns]'
    cast_columns(
        oi,
        'shipping_limit_date',
        'datetime64[ns]'
    )

    # Cast the 'price' and 'freight_value' columns to float
    cast_columns(oi, ['price', 'freight_value'], float)

    # Save to the cache global variable
    global _cached_order_items
    _cached_order_items = oi

    return oi.copy()  # TODO : remove after filtered_copy bypassing tests


def get_order_items(
    orders_index: Optional[Iterable] = None,
    products_index: Optional[Iterable] = None,
    sellers_index: Optional[Iterable] = None
) -> pd.DataFrame:
    """Returns the pk-indexed order items data table.

    A cached table is constructed the first time this function is called.
    Subsequent calls return an optionally filtered copy of tis cached table.

    Args:
        orders_index (iterable, optional): The orders index to filter the
            pk-indexed data table by.
        products_index (iterable, optional): The products index to filter the
            pk-indexed data table by.
        sellers_index (iterable, optional): The sellers index to filter the
            pk-indexed data table by.

    Returns:
        pd.DataFrame: The raw order items data table,
            optionally filtered by the orders, products, and sellers indices.
    """
    # If the _cached_order_items table is not set, set it
    global _cached_order_items
    if _cached_order_items is None:
        _load_cached_order_items()

    return filtered_copy(
        _cached_order_items,
        rows_filter=(orders_index, None),
        data_filter=[products_index, sellers_index]
    )


def _load_cached_order_items_mi():
    # Retrieve the raw order items data table,
    # optionally filtered by the orders, products and sellers indexes
    oi_mi = get_raw_order_items()

    # Set the ('order_id', 'order_item_id') columns as the index of the
    # order items data table
    set_pk_multi_index(oi_mi, ['order_id', 'order_item_id'])

    # Set the new columns name
    oi_mi.columns.name = 'order_items'

    # Cast the 'shipping_limit_date' column to 'datetime64[ns]'
    cast_columns(oi_mi, 'shipping_limit_date', 'datetime64[ns]')

    # Cast the 'price' and 'freight_value' columns to float
    cast_columns(oi_mi, ['price', 'freight_value'], float)

    # Save to the cache global variable
    global _cached_order_items_mi
    _cached_order_items_mi = oi_mi

    return oi_mi.copy()  # TODO : remove after filtered_copy bypassing tests


def get_order_items_multi_index(
    orders_index: Optional[Iterable] = None,
    products_index: Optional[Iterable] = None,
    sellers_index: Optional[Iterable] = None
) -> pd.DataFrame:
    """Returns the pk-indexed order items data table.

    A cached table is constructed the first time this function is called.
    Subsequent calls return an optionally filtered copy of tis cached table.

    Args:
        orders_index (iterable, optional): The orders index to filter the
            pk-indexed data table by.
        products_index (iterable, optional): The products index to filter the
            pk-indexed data table by.
        sellers_index (iterable, optional): The sellers index to filter the
            pk-indexed data table by.

    Returns:
        pd.DataFrame: The raw order items data table,
            optionally filtered by the orders, products, and sellers indices.
    """
    # If the _cached_order_items table is not set, set it
    global _cached_order_items_mi
    if _cached_order_items_mi is None:
        _load_cached_order_items_mi()

    return filtered_copy(
        _cached_order_items_mi,
        rows_filter=(orders_index, None),
        data_filter=[products_index, sellers_index]
    )


def get_orders_v1(index=None) -> pd.DataFrame:
    """Get the orders pk-indexed data table."""
    orders = get_raw_orders()
    orders = orders.set_index('order_id', drop=True)
    orders = orders.drop(columns='customer_id')
    i = len('order_')
    orders.columns = [c[i:] for c in orders.columns]
    orders.columns.name = 'orders'
    orders = orders.astype({
        'purchase_timestamp': 'datetime64[ns]',
        'approved_at': 'datetime64[ns]',
        'delivered_carrier_date': 'datetime64[ns]',
        'delivered_customer_date': 'datetime64[ns]',
        'estimated_delivery_date': 'datetime64[ns]'
    })

    return orders if index is None else orders.loc[index]


def _load_cached_orders():
    # Retrieve the raw orders data table
    o = get_raw_orders()

    # Set the 'order_id' column as the index of the orders data table
    set_pk_index(o, 'order_id')

    # Drop the 'customer_id' column which is redundant with 'order_id'
    o.drop(columns='customer_id', inplace=True)

    # Remove the 'order_' prefix from the column names
    remove_columns_prefixes(o, 'order_')

    # Cast the date columns (all from indice 1) to 'datetime64[ns]'
    cast_columns(o, o.columns[1:], 'datetime64[ns]')

    # Save to the cache global variable
    global _cached_orders
    _cached_orders = o

    return o.copy()  # TODO : remove after filtered_copy bypassing tests


def get_orders(
    orders_index: Optional[Iterable] = None
) -> pd.DataFrame:
    """Returns the pk-indexed orders data table.

    A cached table is constructed the first time this function is called.
    Subsequent calls return an optionally filtered copy of tis cached table.

    Args:
        orders_index (iterable, optional): The orders index to filter the
            pk-indexed data table by.
    Returns:
        pd.DataFrame: The pk-indexed orders data table,
            optionally filtered by the orders indices.
    """
    # If the _cached_orders table is not set, set it
    global _cached_orders
    if _cached_orders is None:
        _load_cached_orders()

    return filtered_copy(
        _cached_orders,
        rows_filter=orders_index
    )


def get_customer_orders_v1(index=None) -> pd.DataFrame:
    """Get the customer orders pk-indexed data table."""
    customers = pd.merge(
        get_raw_customers(),
        get_raw_orders()[['order_id', 'customer_id']],
        how='outer', on='customer_id'
    )

    customers = customers.set_index('order_id', drop=True)
    customers = customers.drop(columns='customer_id')
    customers = customers.rename(
        columns={'customer_unique_id': 'customer_id'}
    )
    i = len('customer_')
    customers.columns = (
        [customers.columns[0]]
        + [c[i:] for c in customers.columns[1:]]
    )
    customers.columns.name = 'customers_orders'

    return customers if index is None else customers.loc[index]


def _load_cached_customer_orders():
    # Retrieve the raw customers and orders data table
    # and merge them on customer_id key
    co = pd.merge(
        get_raw_customers(),
        get_raw_orders()[['order_id', 'customer_id']],
        how='inner', on='customer_id'
    )

    # Set the 'order_id' column as the index of the customer orders data
    # table
    set_pk_index(co, 'order_id')

    # Remove the useless `customer_id` key
    co.drop(columns='customer_id', inplace=True)

    # Rename the 'real' customer ID into `customer_id`
    co.rename(
        columns={'customer_unique_id': 'customer_id'},
        inplace=True
    )

    # Remove the 'customer_' prefix from the column names
    remove_columns_prefixes(
        data=co,
        prefix='customer_',
        start_indice=1,
        new_cols_name='customers_orders'
    )

    # Normalize the city names in the data table
    normalize_city_names(co)

    # Save to the cache global variable
    global _cached_customer_orders
    _cached_customer_orders = co

    return co.copy()  # TODO : remove after filtered_copy bypassing tests


def get_customer_orders(
    orders_index: Optional[Iterable] = None,
    customers_index: Optional[Iterable] = None
) -> pd.DataFrame:
    """Returns the customer orders pk-indexed data table.

    A cached table is constructed the first time this function is called.
    Subsequent calls return an optionally filtered copy of tis cached table.

    Args:
        orders_index (iterable, optional): The orders index to filter
            the raw data table by.
        customers_index (iterable, optional): The customers index to filter
            the raw data table by.

    Returns:
        pd.DataFrame: The customer orders data table,
            optionally filtered by the orders and customers indices.
    """
    # If the _cached_customer_orders table is not set, set it
    global _cached_customer_orders
    if _cached_customer_orders is None:
        _load_cached_customer_orders()
    return filtered_copy(
        _cached_customer_orders,
        rows_filter=orders_index,
        data_filter=customers_index
    )


def get_customers_v1(index=None):
    """Get the customers pk-indexed data table."""
    customer_orders = get_customer_orders_v1().reset_index()
    customer_orders = normalize_city_names_v1(customer_orders)
    customers = customer_orders.groupby(by='customer_id').agg(list)
    customers.columns.name = 'customers'
    return customers if index is None else customers.loc[index]


def _load_cached_customers():
    # Retrieve the pk-indexed customer orders data table
    co = get_customer_orders().reset_index()

    # Group it by the `customer_id` key
    c = co.groupby(by='customer_id').agg(tuple)
    c.columns.name = 'customers'

    # Save to the cache global variable
    global _cached_customers
    _cached_customers = c

    return c.copy()  # TODO : remove after filtered_copy bypassing tests


def get_customers(
    customers_index: Optional[Iterable] = None,
    orders_index: Optional[Iterable] = None
) -> pd.DataFrame:
    """Returns the customers pk-indexed data table.

    A cached table is constructed the first time this function is called.
    Subsequent calls return an optionally filtered copy of tis cached table.

    Args:
        customers_index (iterable, optional): The customers index to filter
            the raw data table by.
        orders_index (iterable, optional): The orders index to filter
            the raw data table by.

    Returns:
        pd.DataFrame: The customer orders data table,
            optionally filtered by the orders and customers indices.
    """
    # If the _cached_customers table is not set, set it
    global _cached_customers
    if _cached_customers is None:
        _load_cached_customers()

    return filtered_copy(
        _cached_customers,
        rows_filter=customers_index,
        data_filter=orders_index
    )


def get_products_v1(index=None):
    """Get the products pk-indexed data table."""
    products = get_raw_products()
    products = products.set_index('product_id', drop=True)
    i = len('product_')
    products.columns = [c[i:] for c in products.columns]
    products.columns.name = 'products'
    products = products.astype({
        'weight_g': float,
        'length_cm': float,
        'height_cm': float,
        'width_cm': float
    })
    return products if index is None else products.loc[index]


def _load_cached_products():
    # Retrieve the raw products data table,
    # optionally filtered by the products index
    p = get_raw_products()

    # Set the 'product_id' column as the index of the products data table
    set_pk_index(p, 'product_id')

    # Remove the 'products_' prefix from the columns of the products data
    # table
    remove_columns_prefixes(p, 'product_')

    # Cast the weight, length, height and width columns to float
    physical_features = ['weight_g', 'length_cm', 'height_cm', 'width_cm']
    cast_columns(p, physical_features, float)

    # Save to the cache global variable
    global _cached_products
    _cached_products = p

    return p.copy()  # TODO : remove after filtered_copy bypassing tests


def get_products(
    products_index: Optional[Iterable] = None
) -> pd.DataFrame:
    """Returns the products pk-indexed data table.

    A cached table is constructed the first time this function is called.
    Subsequent calls return an optionally filtered copy of tis cached table.

    Args:
        products_index (iterable, optional): The products index to filter the
            pk-indexed data table by.

    Returns:
        pd.DataFrame: The pk-indexed products data table,
            optionally filtered by the products index.
    """
    # If the _cached_products table is not set, set it
    global _cached_products
    if _cached_products is None:
        _load_cached_products()

    return filtered_copy(
        _cached_products,
        rows_filter=products_index
    )


def get_sellers_v1(index=None):
    """Get the sellers pk-indexed data table."""
    sellers = get_raw_sellers()
    sellers = sellers.set_index('seller_id', drop=True)
    i = len('seller_')
    sellers.columns = [c[i:] for c in sellers.columns]
    sellers.columns.name = 'sellers'
    return sellers if index is None else sellers.loc[index]


def _load_cached_sellers():
    # Retrieve the raw sellers data table
    s = get_raw_sellers()

    # Set the 'seller_id' column as the index of the sellers data table
    set_pk_index(s, 'seller_id')

    # Remove the 'seller_' prefix from the columns of the sellers data
    # table
    remove_columns_prefixes(s, 'seller_')

    # Normalize the city names in the sellers data table
    normalize_city_names(s)

    # Save to the cache global variable
    global _cached_sellers
    _cached_sellers = s

    return s.copy()  # TODO : remove after filtered_copy bypassing tests


def get_sellers(
    sellers_index: Optional[Iterable] = None
) -> pd.DataFrame:
    """Returns the pk-indexed sellers data table.

    A cached table is constructed the first time this function is called.
    Subsequent calls return an optionally filtered copy of tis cached table.

    Args:
        sellers_index (iterable, optional): The sellers index to filter the
            pk-indexed data table by.

    Returns:
        pd.DataFrame: The pk-indexed sellers data table,
            optionally filtered by the sellers index.
    """
    # If the _cached_sellers table is not set, set it
    global _cached_sellers
    if _cached_sellers is None:
        _load_cached_sellers()

    return filtered_copy(
        _cached_sellers,
        rows_filter=sellers_index
    )


def get_order_payments_v1(orders_index=None):
    """Get the order pk-indexed payments data table."""
    order_payments = get_raw_order_payments(orders_index=orders_index)
    pk = pd.Series(list(zip(
        order_payments.order_id,
        order_payments.payment_sequential
    ))).rename('(order_id, sequential)')
    order_payments = order_payments.set_index(pk)
    order_payments = order_payments.drop(
        columns=['order_id', 'payment_sequential']
    )
    i = len('payment_')
    order_payments.columns = [c[i:] for c in order_payments.columns]
    order_payments.columns.name = 'order_payments'
    order_payments = order_payments.astype({
        'value': float,
    })

    return (
        order_payments if orders_index is None
        else order_payments.loc[orders_index]
    )


def _load_cached_order_payments():
    # Retrieve the raw order payments data table,
    # optionally filtered by the orders index
    op = get_raw_order_payments()

    # Set the ('order_id', 'payment_sequential') columns as the index of
    # the order payments data table
    set_pk_index(
        data=op,
        pk_columns=['order_id', 'payment_sequential'],
        pk_name=('order_id', 'sequential')
    )

    # Remove the 'payment_' prefix from the columns of the sellers data
    # table
    remove_columns_prefixes(op, 'payment_')

    # Cast the value column to float
    cast_columns(op, ['value'], float)

    # Save to the cache global variable
    global _cached_order_payments
    _cached_order_payments = op

    return op.copy()  # TODO : remove after filtered_copy bypassing tests


def get_order_payments(
    orders_index: Optional[Iterable] = None
) -> pd.DataFrame:
    """Returns the pk-indexed order payments data table.

    A cached table is constructed the first time this function is called.
    Subsequent calls return an optionally filtered copy of tis cached table.

    Args:
        orders_index (iterable, optional): The orders index to filter the
            pk-indexed data table by.

    Returns:
        pd.DataFrame: The pk-indexed order payments data table,
            optionally filtered by the orders index.
    """
    # If the _cached_order_payments table is not set, set it
    global _cached_order_payments
    if _cached_order_payments is None:
        _load_cached_order_payments()

    return filtered_copy(
        _cached_order_payments,
        rows_filter=(orders_index, None)
    )


def get_order_reviews_v1(index=None):
    """Get the order reviews pk-indexed data table."""
    order_reviews = get_raw_order_reviews()
    pk = pd.Series(list(zip(
        order_reviews.order_id,
        order_reviews.review_id
    ))).rename('(order_id, review_id)')
    order_reviews = order_reviews.set_index(pk)
    order_reviews = order_reviews.drop(
        columns=['order_id', 'review_id']
    )
    i = len('review_')
    order_reviews.columns = [c[i:] for c in order_reviews.columns]
    order_reviews.columns.name = 'order_reviews'
    return order_reviews if index is None else order_reviews.loc[index]


def _load_cached_order_reviews():
    # Retrieve the raw sellers data table,
    # optionally filtered by the orders index
    orw = get_raw_order_reviews()

    # Set the ('order_id', 'review_id') columns as the index of the
    # order reviews data table
    set_pk_index(orw, ['order_id', 'review_id'])

    # Remove the 'review_' prefix from the columns of the sellers data
    # table
    remove_columns_prefixes(orw, 'review_')

    # Cast the creation_date and answer_timestamp column to
    # 'datetime64[ns]'
    cast_columns(
        orw,
        ['creation_date', 'answer_timestamp'],
        'datetime64[ns]'
    )

    # Cast the score column to int
    cast_columns(orw, ['score'], int)

    # Save to the cache global variable
    global _cached_order_reviews
    _cached_order_reviews = orw

    return orw.copy()  # TODO : remove after filtered_copy bypassing tests


def get_order_reviews(
    orders_index: Optional[Iterable] = None
) -> pd.DataFrame:
    """Returns the pk-indexed order reviews data table.

    A cached table is constructed the first time this function is called.
    Subsequent calls return an optionally filtered copy of tis cached table.

    Args:
        orders_index (iterable, optional): The orders index to filter the
            pk-indexed data table by.

    Returns:
        pd.DataFrame: The pk-indexed order reviews data table,
            optionally filtered by the orders index.
    """
    # If the _cached_order_reviews table is not set, set it
    global _cached_order_reviews
    if _cached_order_reviews is None:
        _load_cached_order_reviews()

    return filtered_copy(
        _cached_order_reviews,
        rows_filter=(orders_index, None)
    )


def get_product_categories_v1(index=None):
    """Get the product categories pk-indexed data table."""
    products = get_products_v1(index=index_of_documented_products_v1())
    counts = products.category_name.value_counts()
    counts = counts.reset_index()
    counts.columns = ['category_name', 'products_count']
    categories = get_raw_product_categories()
    categories.columns = ['category_name', 'category_name_EN']
    categories = pd.merge(categories, counts, how='outer', on='category_name')
    categories.columns.name = 'product_categories'
    categories.index.name = 'product_category_id'
    categories.loc[71, 'category_name_EN'] = \
        'kitchen_portables_and_food_preparators'
    categories.loc[72, 'category_name_EN'] = 'pc_gamer'
    return categories if index is None else categories.loc[index]


def _load_product_categories():
    # Get the index of all documented products (with no NA category_name)
    documented_products_index = index_of_documented_products()

    # Retrieved the documented products table filtered by the optional
    # product_index
    products = get_products(
        products_index=documented_products_index
    )

    # Build the empirical distribution of product categories
    counts = products.category_name.value_counts()
    counts = counts.reset_index()
    counts.columns = ['category_name', 'products_count']

    # Retrieve the raw product categories data table
    categories = get_raw_product_categories()
    categories.columns = ['category_name', 'category_name_EN']

    # Merge the raw data table with the empirical distribution
    pc = pd.merge(
        categories,
        counts,
        how='outer',
        on='category_name'
    )

    # Set the name of the columns and index
    pc.columns.name = 'product_categories'
    pc.index.name = 'product_category_id'

    # Complete the english translation of unreferenced categories
    pc.loc[71, 'category_name_EN'] = \
        'kitchen_portables_and_food_preparators'
    pc.loc[72, 'category_name_EN'] = 'pc_gamer'

    # Save to the cache global variable
    global _cached_product_categories
    _cached_product_categories = pc

    return pc.copy()  # TODO : remove after filtered_copy bypassing tests


def get_product_categories() -> pd.DataFrame:
    """
    Returns the pk-indexed product categories data table.

    The data table is constructed the first time this function is called.
    Subsequent calls return a copy of the data table.

    Returns:
        pd.DataFrame: The pk-indexed product categories data table.
    """
    # If the cached _product_categories is not set, set it
    global _cached_product_categories
    if _cached_product_categories is None:
        _load_product_categories()

    return _cached_product_categories.copy()


def get_geolocations_v1(index=None):
    """Get the geolocations pk-indexed data table."""
    geolocations = get_raw_geolocations()
    geolocations.index.name = 'geolocation_id'
    i = len('geolocation_')
    geolocations.columns = [c[i:] for c in geolocations.columns]
    geolocations.columns.name = 'geolocations'
    # Permutation of columns
    cols = geolocations.columns
    new_cols = list(cols[1:3]) + [cols[0]] + list(cols[3:])
    geolocations = geolocations[new_cols]
    return geolocations if index is None else geolocations.loc[index]


def _load_cached_geolocations():
    # Retrieve the raw geolocations data table
    g = get_raw_geolocations()

    # Fix the default positional index as the pk
    g.index.name = 'geolocation_id'

    # Remove the 'geolocation_' prefix from the column names
    remove_columns_prefixes(g, 'geolocation_')

    # Permute the columns to follow the standard order
    cols = g.columns
    new_cols = list(cols[1:3]) + [cols[0]] + list(cols[3:])
    g = g[new_cols]

    # Cast numerical objects into float :
    # The conversion to float loses precision with only
    #  5 significant digits
    # against the 15 recorded. This corresponds to a metric precision of
    # the order of a meter, which is more than enough in the context of
    # the application.
    cast_columns(g, ['lat', 'lng'], 'float64')

    # Normalize the city names in the data table
    normalize_city_names(g)

    # Save to the cache global variable
    global _cached_geolocations
    _cached_geolocations = g

    return g.copy()  # TODO : remove after filtered_copy bypassing tests


def get_geolocations() -> pd.DataFrame:
    """Retrieve and preprocess the geolocations data table.

    This function retrieves the raw geolocations data table, fixes the
    primary key (pk) index, removes the 'geolocation_' prefix from the
    column names, permutes the columns to follow the standard order,
    casts numerical objects into float, and normalizes the city names.

    The data table is constructed the first time this function is called.
    Subsequent calls return a copy of the data table.

    Returns:
        pd.DataFrame: The preprocessed, pk-indexed geolocations data table.
    """
    # If _cached_geolocations is not set, set it
    global _cached_geolocations
    if _cached_geolocations is None:
        _load_cached_geolocations()

    return _cached_geolocations.copy()


""" Special cases indexes
"""


def index_of_delivered_orders_v1(index=None) -> pd.Index:
    """Returns the index of orders that have been delivered.
    """
    orders = get_orders_v1(index=index)
    return orders[orders.status == 'delivered'].index


def index_of_undelivered_orders_v1(index=None) -> pd.Index:
    """Returns the index of orders that have not been delivered.
    """
    orders = get_orders_v1(index=index)
    return orders[orders.status != 'delivered'].index


def index_of_delivered_orders(
    orders_index: Optional[Iterable] = None
) -> pd.Index:
    """Returns the index of orders that have been delivered.
    """
    orders = get_orders(orders_index=orders_index)
    return orders[orders.status == 'delivered'].index


def index_of_undelivered_orders(
    orders_index: Optional[Iterable] = None
) -> pd.Index:
    """Returns the index of orders that have not been delivered.
    """
    orders = get_orders(orders_index=orders_index)
    return orders[orders.status != 'delivered'].index


def index_of_unpaid_orders_v1(index=None):
    """Returns the index of orders that have not been paid.
    """
    # Calculate the set difference between the set of unique order ids
    # and the set of unique order ids that have been paid
    return pd.Index(list(
        set(get_raw_orders_v1().order_id.unique())
        - set(get_raw_order_payments_v1().order_id.unique())
    ))


def customer_location_counts_v1(index=None):
    """Returns the customer location counts.

    DEPRECATED: use `get_customer_location_counts()` instead.
    """
    customer_orders = get_customer_orders_v1(index=index)
    customer_locs = customer_orders.drop_duplicates()
    return customer_locs.customer_id.value_counts()


def get_customer_location_counts(
    customers_index: Optional[Iterable] = None
) -> pd.Series:
    """Returns the customer location counts.
    """
    customer_orders = get_customer_orders(
        customers_index=customers_index
    )
    customer_locs = customer_orders.drop_duplicates()
    return customer_locs.customer_id.value_counts()


def index_of_sedentary_customers_v1(index=None):
    """Returns the index of customers associated with a single location.

    DEPRECATED: use `index_of_sedentary_customers()` instead.

    Args:
        index (iterable, optional): The orders index to filter the data
            table by.

    Returns:
        pd.Index: The index of customers associated with a single location.
    """
    counts = customer_location_counts_v1(index=index)
    return counts[counts == 1].index.rename('customer_id')


def index_of_sedentary_customers(
    customers_index: Optional[Iterable] = None
) -> pd.Index:
    """Returns the index of customers associated with a single location.

    Args:
        customers_index (iterable, optional): The customers index to filter
            the data table by.

    Returns:
        pd.Index: The index of customers associated with a single location.
    """
    counts = get_customer_location_counts(
        customers_index=customers_index
    )
    return counts[counts == 1].index


def index_of_nomadic_customers_v1(index=None):
    """Returns the index of customers associated with many locations.

    Args:
        orders_index (iterable, optional): The orders index to filter the data
            table by.

    Returns:
        pd.Index: The index of customers associated with many locations.
    """
    counts = customer_location_counts_v1(index=index)
    return counts[counts > 1].index.rename('customer_id')


def index_of_nomadic_customers(
    customers_index: Optional[Iterable] = None
) -> pd.Index:
    """Returns the index of customers associated with many location.

    Args:
        customers_index (iterable, optional): The customers index to filter
            the data table by.

    Returns:
        pd.Index: The index of customers associated with many location.
    """
    counts = get_customer_location_counts(
        customers_index=customers_index
    )
    return counts[counts > 1].index


def index_of_dimensioned_products_v1(index=None):
    """Returns the index of products that have physical features.

    DEPRECATED: use `index_of_dimensioned_products()` instead.

    Args:
        index (iterable, optional): The products index to filter the data
            table by.

    Returns:
        pd.Index: The index of products that have physical features.
    """
    products = get_products_v1(index=index)
    # Get products where the 'weight_g' column is not null
    bindex = products.weight_g.notna()
    products_subset = products[bindex]
    return products_subset.index


def index_of_dimensioned_products(
    products_index: Optional[Iterable] = None
) -> pd.Index:
    """Returns the index of products that have physical features.

    Args:
        products_index (iterable, optional): The products index to filter the
            data table by.

    Returns:
        pd.Index: The index of products that have physical features.
    """
    # Retrieve the products pk-indexed table
    products = get_products(products_index=products_index)
    # Filter products where the 'weight_g' column is not null
    return products[products.weight_g.notna()].index


def index_of_undimensioned_products_v1(index=None):
    """Returns the index of products that do not have physical features.

    DEPRECATED: use `index_of_undimensioned_products()` instead.

    Args:
        index (iterable, optional): The products index to filter the data
            table by.

    Returns:
        pd.Index: The index of products that do not have physical features.
    """
    products = get_products_v1(index=index)
    # Get products where the 'weight_g' column is null
    bindex = products.weight_g.isna()
    products_subset = products[bindex]
    return products_subset.index


def index_of_undimensioned_products(
    products_index: Optional[Iterable] = None
) -> pd.Index:
    """Returns the index of products that do not have physical features.

    Args:
        index (iterable, optional): The products index to filter the data
            table by.

    Returns:
        pd.Index: The index of products that do not have physical features.
    """
    # Retrieve the products pk-indexed table
    products = get_products(products_index=products_index)
    # Filter products where the 'weight_g' column is null
    return products[products.weight_g.isna()].index


def index_of_documented_products_v1(index=None):
    """Returns the index of products that have marketing features.

    DEPRECATED: use `index_of_documented_products()` instead.

    Args:
        index (iterable, optional): The products index to filter the data
            table by.

    Returns:
        pd.Index: The index of products that have marketing features.
    """
    products = get_products_v1(index=index)
    # Get products where the 'category_name' column is not null
    bindex = products.category_name.notna()
    products_subset = products[bindex]
    return products_subset.index


def index_of_documented_products(
    products_index: Optional[Iterable] = None
) -> pd.Index:
    """Returns the index of products that have marketing features.

    Args:
        products_index (iterable, optional): The products index to filter the
            data table by.

    Returns:
        pd.Index: The index of products that have marketing features.
    """
    # Retrieve the products pk-indexed table
    products = get_products(products_index=products_index)
    # Filter products where the 'category_name' column is not null
    return products[products.category_name.notna()].index


def index_of_undocumented_products_v1(index=None):
    """Returns the index of products that do not have marketing features.

    DEPRECATED: use `index_of_undocumented_products()` instead.

    Args:
        index (iterable, optional): The products index to filter the data
            table by.

    Returns:
        pd.Index: The index of products that do not have marketing features.
    """
    # Retrieve the products pk-indexed table
    products = get_products_v1(index=index)
    # Get products where the 'category_name' column is null
    bindex = products.category_name.isna()
    products_subset = products[bindex]
    return products_subset.index


def index_of_undocumented_products(
    products_index: Optional[Iterable] = None
) -> pd.Index:
    """Returns the index of products that do not have marketing features.

    Args:
        products_index (iterable, optional): The products index to filter the
            data table by.

    Returns:
        pd.Index: The index of products that do not have marketing features.
    """
    # Retrieve the products pk-indexed table
    products = get_products(products_index=products_index)
    # Filter products where the 'category_name' column is null
    return products[products.category_name.isna()].index


def index_of_fully_qualified_products_v1(index=None):
    """Returns the index of products with all
    physical and marketing features provided.
    """
    return (
        index_of_dimensioned_products_v1(index=index)
        .intersection(index_of_documented_products_v1(index=index))
    )


def index_of_fully_qualified_products(
    products_index: Optional[Iterable] = None
) -> pd.Index:
    """Returns the index of products with all
    physical and marketing features provided.
    """
    # Retrieve the products pk-indexed table
    products = get_products(products_index=products_index)
    # Select products where
    # both 'category_name' and 'weight_g' columns are not null
    return products[
        products.category_name.notna()
        & products.weight_g.notna()
    ].index


def index_of_unknown_products_v1(index=None):
    """Returns the index of products that have no features.
    DEPRECATED
    """
    return (
        index_of_undimensioned_products_v1(index=index)
        .intersection(index_of_undocumented_products_v1(index=index))
    )


def index_of_unknown_products(
    products_index: Optional[Iterable] = None
) -> pd.Index:
    """Returns the index of products that have no features.
    """
    # Retrieve the products pk-indexed table
    products = get_products(products_index=products_index)
    # Select products where
    # both 'category_name' and 'weight_g' columns are null
    return products[
        products.category_name.isna()
        & products.weight_g.isna()
    ].index


def index_of_sellers_from_state(
    state: str,
    sellers_index: Optional[Iterable] = None
) -> pd.Index:
    """Returns the index of products that have no features.
    """
    # Retrieve the sellers pk-indexed table
    sellers = get_sellers(sellers_index=sellers_index)
    # Select sellers where 'state' is state
    return sellers[sellers.state == state].index


""" Merging
"""


def _expand_index(data: pd.DataFrame) -> pd.DataFrame:
    data.index = pd.MultiIndex.from_tuples(
        list(data.index),
        names=data.index.name
    )
    return data


def merge_col_indexes(*args: List[pd.DataFrame]) -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples(
        [
            (df.columns.name, col)
            for df in args
            for col in df.columns
        ],
        names=['object', 'features']
    )


def wrapped_merge(
    x: pd.DataFrame,
    y: pd.DataFrame,
    **merge_kwargs
) -> pd.DataFrame:
    xy = pd.merge(x, y, **merge_kwargs)
    xy = xy[list(x.columns) + list(y.columns)]
    xy.columns = merge_col_indexes(x, y)
    return xy


def _load_categorized_products():
    categories = get_product_categories().set_index('category_name')
    products = get_products().reset_index()
    categories.columns.name = 'categories'
    data = wrapped_merge(
        categories, products,
        left_index=True, right_on='category_name',
        how='outer'
    )

    data.set_index(
        [
            ('products', 'category_name'),
            ('products', 'product_id')
        ],
        inplace=True
    )
    data.index.names = [
        ('_ident_', 'category_name'),
        ('_ident_', 'product_id')
    ]
    data.index.name = 'categorized_products'
    data.columns.name = data.index.name

    return data


def _load_products_sales():
    products = get_products()
    sales = _expand_index(get_order_items())
    sales.columns.name = 'sales'
    data = wrapped_merge(
        products, sales,
        left_index=True, right_on='product_id',
        how='outer'
    )

    data.set_index(('sales', 'product_id'), append=True, inplace=True)
    data.index.names = [
        ('_ident_', 'order_id'),
        ('_ident_', 'order_item_id'),
        ('_ident_', 'product_id')
    ]
    data.index.name = 'products_sales'
    data.columns.name = data.index.name

    return data


def _load_sellers_sales():
    sellers = get_sellers()
    sales = _expand_index(get_order_items())
    sales.columns.name = 'sales'
    data = wrapped_merge(
        sellers, sales,
        left_index=True, right_on='seller_id',
        how='outer'
    )

    data.set_index(('sales', 'seller_id'), append=True, inplace=True)
    data.index.names = [
        ('_ident_', 'order_id'),
        ('_ident_', 'order_item_id'),
        ('_ident_', 'seller_id')
    ]
    data.index.name = 'sellers_sales'
    data.columns.name = data.index.name

    return data


def _load_sales():
    orders = get_orders()
    sales = _expand_index(get_order_items()).reset_index(1)
    sales.columns.name = 'sales'
    data = wrapped_merge(
        sales, orders,
        on='order_id',
        how='outer'
    )

    data.set_index(
        ('sales', 'order_item_id'),
        append=True, inplace=True
    )
    data.index.names = [
        ('_ident_', 'order_id'),
        ('_ident_', 'order_item_id')
    ]
    data.index.name = 'sales'
    data.columns.name = data.index.name

    return data


def _load_cached_orders_reviews():
    reviews = _expand_index(get_order_reviews()).reset_index(1)
    orders = get_orders()
    reviews.columns.name = 'reviews'
    data = wrapped_merge(
        orders, reviews,
        left_on='order_id', right_index=True,
        how='outer'
    )

    data.set_index(
        ('reviews', 'review_id'),
        append=True, inplace=True
    )
    data.index.names = [
        ('_ident_', 'order_id'),
        ('_ident_', 'review_id')
    ]
    data.index.name = 'orders_reviews'
    data.columns.name = data.index.name

    return data


def _load_cached_customers_orders():
    customers = get_customer_orders()
    orders = get_orders()
    customers.columns.name = 'customers'
    data = wrapped_merge(
        customers,
        orders,
        on='order_id',
        how='outer'
    )

    data.set_index(
        ('customers', 'customer_id'),
        append=True, inplace=True
    )
    data.index.names = [
        ('_ident_', 'order_id'),
        ('_ident_', 'customer_id')
    ]
    data.index.name = 'orders_reviews'
    data.columns.name = data.index.name

    return data


def _load_cached_orders_payments():
    payments = _expand_index(get_order_payments()).reset_index(1)
    orders = get_orders()
    payments.columns.name = 'payments'
    data = wrapped_merge(
        payments, orders,
        left_index=True, right_on='order_id',
        how='outer'
    )

    data.set_index(
        ('payments', 'sequential'),
        append=True, inplace=True
    )
    data.index.names = [
        ('_ident_', 'order_id'),
        ('_ident_', 'sequential')
    ]
    data.index.name = 'orders_payments'
    data.columns.name = data.index.name

    return data


def _load_cached_customers_orders_payments():
    c = get_customer_orders()
    o = get_orders()
    p = _expand_index(get_order_payments()).reset_index(1)

    c.columns.name = 'customers'
    p.columns.name = 'payments'

    data = pd.merge(c, o, on='order_id', how='outer')
    data = pd.merge(data, p, on='order_id', how='outer')

    data.columns = merge_col_indexes(c, o, p)

    data.set_index(
        [
            ('customers', 'customer_id'),
            ('payments', 'sequential')
        ],
        append=True, inplace=True
    )
    data.index.names = [
        ('_ident_', 'order_id'),
        ('_ident_', 'customer_id'),
        ('_ident_', 'sequential')
    ]
    data.index.name = 'COP'
    data.columns.name = data.index.name

    return data


def _load_cached_customers_orders_reviews():
    o = get_orders()
    c = get_customer_orders()
    r = _expand_index(get_order_reviews()).reset_index(1)

    c.columns.name = 'customers'
    r.columns.name = 'reviews'

    data = pd.merge(c, o, on='order_id', how='outer')
    data = pd.merge(data, r, on='order_id', how='outer')

    data.columns = merge_col_indexes(c, o, r)

    data.set_index(
        [
            ('customers', 'customer_id'),
            ('reviews', 'review_id')
        ],
        append=True, inplace=True
    )
    data.index.names = [
        ('_ident_', 'order_id'),
        ('_ident_', 'customer_id'),
        ('_ident_', 'review_id')
    ]
    data.index.name = 'COR'
    data.columns.name = data.index.name

    return data


def _load_cached_orders_payments_reviews():
    o = get_orders()
    p = _expand_index(get_order_payments()).reset_index(1)
    r = _expand_index(get_order_reviews()).reset_index(1)

    p.columns.name = 'payments'
    r.columns.name = 'reviews'

    data = pd.merge(o, p, on='order_id', how='outer')
    data = pd.merge(data, r, on='order_id', how='outer')

    data.columns = merge_col_indexes(o, p, r)

    data.set_index(
        [
            ('payments', 'sequential'),
            ('reviews', 'review_id')
        ],
        append=True, inplace=True
    )
    data.index.names = [
        ('_ident_', 'order_id'),
        ('_ident_', 'sequential'),
        ('_ident_', 'review_id')
    ]
    data.index.name = 'OPR'
    data.columns.name = data.index.name

    return data


def _load_cached_order_details():
    c = get_customer_orders()
    o = get_orders()
    p = _expand_index(get_order_payments()).reset_index(1)
    r = _expand_index(get_order_reviews()).reset_index(1)

    c.columns.name = 'customers'
    p.columns.name = 'payments'
    r.columns.name = 'reviews'

    data = pd.merge(c, o, on='order_id', how='outer')
    data = pd.merge(data, p, on='order_id', how='outer')
    data = pd.merge(data, r, on='order_id', how='outer')

    data.columns = merge_col_indexes(c, o, p, r)

    data.set_index(
        [
            ('customers', 'customer_id'),
            ('payments', 'sequential'),
            ('reviews', 'review_id')
        ],
        append=True, inplace=True
    )
    data.index.names = [
        ('_ident_', 'order_id'),
        ('_ident_', 'customer_id'),
        ('_ident_', 'sequential'),
        ('_ident_', 'review_id')
    ]
    data.index.name = 'OPR'
    data.columns.name = data.index.name

    return data


def _load_cached_sales_details():
    i = _expand_index(get_order_items()).reset_index()
    s = get_sellers()
    c = get_product_categories().set_index('category_name')
    p = get_products().reset_index().set_index('category_name')

    c.columns.name = 'categories'
    i.columns.name = 'sales'

    cp = pd.merge(c, p, on='category_name', how='outer')
    cp.reset_index('category_name', inplace=True)
    c.reset_index('category_name', inplace=True)
    p.set_index('product_id', inplace=True)
    data = pd.merge(i, s, on='seller_id', how='outer')
    data = pd.merge(data, cp, on='product_id', how='outer')

    data.columns = merge_col_indexes(i, s, c, p)

    data.set_index(
        [
            ('sales', 'order_id'),
            ('sales', 'order_item_id'),
            ('sales', 'product_id'),
            ('sales', 'seller_id')
        ],
        inplace=True
    )
    data.index.names = [
        ('_ident_', 'order_id'),
        ('_ident_', 'order_item_id'),
        ('_ident_', 'product_id'),
        ('_ident_', 'seller_id')
    ]
    data.index.name = 'ISCP'
    data.columns.name = data.index.name

    return data


def _load_cached_all_in_one():
    od = get_order_details()
    sd = get_sales_details()
    sd.reset_index(inplace=True)
    sd.set_index(('_ident_', 'order_id'), inplace=True)
    od.reset_index(inplace=True)
    od.set_index(('_ident_', 'order_id'), inplace=True)
    data = pd.merge(sd, od, on=[('_ident_', 'order_id')], how='outer')
    data.set_index([
            ('_ident_', 'order_item_id'),
            ('_ident_', 'product_id'),
            ('_ident_', 'seller_id'),     
            ('_ident_', 'customer_id'),
            ('_ident_', 'sequential'),
            ('_ident_', 'review_id'),
        ],
        append=True, inplace=True
    )
    return data


# TODO : test métier de filtered copy avec ce cas :
# j'ai obtenu un plantage avec (None, documented_products)
def get_categorized_products(
    categories_index: Optional[Iterable] = None,
    products_index: Optional[Iterable] = None
) -> pd.DataFrame:
    categorized_products = Cache.init(
        'categorized_products',
        _load_categorized_products
    )
    return filtered_copy(
        categorized_products,
        rows_filter=(categories_index, products_index)
    )


def get_products_sales(
    orders_index: Optional[Iterable] = None,
    products_index: Optional[Iterable] = None
) -> pd.DataFrame:
    products_sales = Cache.init(
        'products_sales',
        _load_products_sales
    )
    return filtered_copy(
        products_sales,
        rows_filter=(orders_index, None, products_index)
    )


def get_sellers_sales(
    orders_index: Optional[Iterable] = None,
    sellers_index: Optional[Iterable] = None
) -> pd.DataFrame:
    sellers_sales = Cache.init('sellers_sales', _load_sellers_sales)
    return filtered_copy(
        sellers_sales,
        rows_filter=(orders_index, None, sellers_index)
    )


def get_sales(
    orders_index: Optional[Iterable] = None,
    products_index: Optional[Iterable] = None,
    sellers_index: Optional[Iterable] = None
) -> pd.DataFrame:
    sales = Cache.init('sales', _load_sales)
    return filtered_copy(
        sales,
        rows_filter={('_ident_', 'order_id'): orders_index},
        data_filter={
            ('sales', 'product_id'): products_index,
            ('sales', 'seller_id'): sellers_index
        }
    )


def get_customers_orders_payments():
    return _init_cache('customers_orders_payments').copy()


def get_customers_orders_reviews():
    return _init_cache('customers_orders_reviews').copy()


def get_orders_payments_reviews():
    return _init_cache('orders_payments_reviews').copy()


def get_orders_reviews():
    return _init_cache('orders_reviews').copy()


def get_customers_orders():
    return _init_cache('customers_orders').copy()


def get_orders_payments():
    return _init_cache('orders_payments').copy()


def get_order_details():
    return _init_cache('order_details').copy()


def get_sales_details():
    return _init_cache('sales_details').copy()


def get_all_in_one():
    return _init_cache('all_in_one').copy()


"""def get_payment_types():
    "" "Get the unique payment types in the order payments data table.
    Returns:
        numpy.ndarray: The unique payment types.
    "" "
    return _raw_order_payments.payment_type.unique()
"""


"""def get_merged_data_v1():
    " ""
    Merge several data tables to create a comprehensive dataset.

    DEPRECATED : use `get_all_in_one` instead

    Returns:
    pandas.DataFrame: The merged dataset.
    "" "
    m = get_raw_order_items()
    m = pd.merge(m, get_raw_orders(), how='outer', on='order_id')
    m = pd.merge(m, get_raw_products(), how='outer', on='product_id')
    m = pd.merge(m, get_raw_sellers(), how='outer', on='seller_id')
    m = pd.merge(m, get_raw_customers(), how='outer', on='customer_id')
    m = pd.merge(m, get_raw_order_payments(), how='outer', on='order_id')
    m = pd.merge(m, get_raw_order_reviews(), how='outer', on='order_id')
    return m"""


""" Customer centric DB refactoring


        TODO : à refactorer


"""


def get_customer_order_counts(index=None):
    """Get the customer order counts feature series."""
    customers = get_customers_v1(index=index)
    return customers.order_id.apply(len)


def customer(customers, cu_id):
    """
    Get a customer by unique ID.

    Parameters:
    customers (pandas.DataFrame): The customers data table.
    cu_id (str): The unique ID of the customer to get.

    Returns:
    pandas.DataFrame: The customer with the specified unique ID.
    """
    return customers[customers.customer_unique_id == cu_id]


def customer_states(customer):
    """
    Get the unique states for a customer table subset.

    Parameters:
    customer (pandas.DataFrame): The customer data.

    Returns:
    numpy.ndarray: The unique states for selected customers.
    """
    return customer.customer_state.unique()


def customer_cities(customer, state):
    """
    Get the unique cities for a customer in a given state.

    Parameters:
    customer (pandas.DataFrame): The customer data.
    state (str): The state to filter the cities by.

    Returns:
    numpy.ndarray: The unique cities for the customer in the specified state.
    """
    return customer[
        customer.customer_state == state
    ].customer_city.unique()


def customer_zips(customer, state, city):
    """
    Get the unique ZIP codes for a customer in a given state and city.

    Parameters:
    customer (pandas.DataFrame): The customer data.
    state (str): The state to filter the ZIP codes by.
    city (str): The city to filter the ZIP codes by.

    Returns:
    numpy.ndarray:
        The unique ZIP codes for the customer in the specified state and city.
    """
    return customer[
        (customer.customer_state == state) &
        (customer.customer_city == city)
    ].customer_zip_code_prefix.unique()


def customer_ids(customer, state, city, zip_code):
    """
    Get the customer IDs for a customer in a given state, city, and ZIP code.

    Parameters:
    customer (pandas.DataFrame): The customer data.
    state (str): The state to filter the customer IDs by.
    city (str): The city to filter the customer IDs by.
    zip_code (str): The ZIP code to filter the customer IDs by.

    Returns:
    tuple:
        The customer IDs for the customer in the specified state, city,
        and ZIP code.
    """
    return tuple(
        customer[
            (customer.customer_state == state) &
            (customer.customer_city == city) &
            (customer.customer_zip_code_prefix == zip_code)
        ].customer_id
    )


def customer_locations(customer):
    """
    Get the locations (states, cities, and ZIP codes) for a customer.

    Parameters:
    customer (pandas.DataFrame): The customer data.

    Returns:
    dict:
        A dictionary of the customer's locations, organized by state, city,
        and ZIP code.
    """
    return {
        state: {
            city: {
                zip_code: customer_ids(customer, state, city, zip_code)
                for zip_code in customer_zips(customer, state, city)
            } for city in customer_cities(customer, state)
        } for state in customer_states(customer)
    }


def test_customer_locations():
    """
    Test the customer_locations function.
    """
    customers = get_raw_customers()
    cu_id = 'fe59d5878cd80080edbd29b5a0a4e1cf'
    c = customer(customers, cu_id)
    c_locations = customer_locations(c)
    display(c_locations)


def get_unique_customers():
    """Get a DataFrame of unique customers and their locations.

    Returns:
        pandas.DataFrame: A DataFrame of unique customers and their locations.
    """
    return pd.DataFrame(
        get_raw_customers()
        .groupby(by=['customer_unique_id'], group_keys=True)
        .apply(customer_locations),
        columns=['locations']
    )


""" Aggregating of payments (by order then by customer)
"""


"""def get_aggregated_order_payments_v1():
    "" "Aggregate the order payments data table by order ID.
    DEPRECATED : used `get_order_payments_by_order` instead
    "" "
    # Load the order payments data table
    op = get_raw_order_payments_v1()
    op.payment_value = op.payment_value.astype(float)
    op.payment_sequential = op.payment_sequential.astype(int)

    # Sort the data table by order ID and payment sequential number
    op = op.sort_values(
        by=['order_id', 'payment_sequential']
    )

    # Group the data table by order ID and aggregate the payment data
    op_gpby = (
        op
        .groupby(by='order_id')
        .aggregate(tuple)
    )

    # Add columns for the payment count and total value
    op_gpby.insert(
        0, 'payment_count',
        op_gpby.payment_sequential.apply(lambda x: len(x))
    )
    op_gpby.insert(
        1, 'payment_total',
        op_gpby.payment_value.apply(lambda x: sum(x))
    )

    return op_gpby"""


"""def get_aggregated_order_payments_v2(
    orders_index: Optional[Iterable] = None
) -> pd.DataFrame:
    "" "Aggregate the order payments data table by order ID.

    DEPRECATED : used `get_order_payments_by_order` instead

    Args:
        orders_index (iterable, optional): The orders index to filter the
            raw data table by.

    Returns:
        pd.DataFrame: The order payments data table aggregated by order ID,
            optionally filtered by the orders index.
    "" "
    op = get_raw_order_payments(orders_index=orders_index)
    op.payment_value = op.payment_value.astype(float)
    op.payment_sequential = op.payment_sequential.astype(int)

    # Sort the data table by order ID and payment sequential number
    op = op.sort_values(
        by=['order_id', 'payment_sequential']
    )

    # Group the data table by order ID and aggregate the payment data
    op_gpby = (
        op
        .groupby(by='order_id')
        .aggregate(tuple)
    )

    # Add columns for the payment count and total value
    op_gpby.insert(
        0, 'count',
        op_gpby.payment_sequential.apply(lambda x: len(x))
    )
    op_gpby.insert(
        1, 'sum',
        op_gpby.payment_value.apply(lambda x: sum(x))
    )

    op_gpby.columns.name = 'aggregated_order_payments'
    op_gpby.columns = [
        'count',
        'sum',
        'sequence',
        'types',
        'installments',
        'values'
    ]

    return op_gpby"""


"""def _load_cached_order_payments_by_order_v1():
    "" "Aggregate the order payments data table by order ID.

    Returns:
        pd.DataFrame: The order payments data table aggregated by order ID
    "" "
    op = _expand_index(get_order_payments())
    op.reset_index(1, inplace=True)

    # Sort the data table by order ID and payment sequential number
    op.sort_values(by=['order_id', 'sequential'], inplace=True)

    # Group the data table by order ID and aggregate the payment data
    op_by_o = op.groupby(by='order_id').aggregate(tuple)

    # Add columns for the payment count and total value
    op_by_o.insert(0, 'count', op_by_o.sequential.apply(len))
    op_by_o.insert(1, 'sum', op_by_o.value.apply(sum))

    op_by_o.columns.name = 'order_payments_by_order'

    return op_by_o"""


def _load_order_payments_by_order():
    """Aggregate the order payments data table by order ID.

    Returns:
        pd.DataFrame: The order payments data table aggregated by order ID
    """
    cop = get_customers_orders_payments().reset_index()
    cop = cop[[
        ('_ident_', 'order_id'),
        ('_ident_', 'customer_id'),
        ('_ident_', 'sequential'),
        ('orders', 'purchase_timestamp'),
        ('payments', 'type'),
        ('payments', 'installments'),
        ('payments', 'value'),
    ]]
    cop.columns = [
        'order_id', 'customer_id', 'sequential',
        'purchase_date',
        'type', 'installments', 'value'
    ]
    op_by_o = (
        cop.groupby(by=['order_id', 'customer_id'])
        .agg({
            'value': ['count', 'sum', 'min', 'max', 'mean'],
            'purchase_date': ['min', 'max'],
            'sequential': tuple,
            'type': tuple,
            'installments': tuple
        })
    )
    op_by_o.reset_index(1, inplace=True)
    return op_by_o


def get_order_payments_by_order(
    orders_index: Optional[Iterable] = None
) -> pd.DataFrame:
    """Aggregate the order payments data table by order ID.

    Args:
        orders_index (iterable, optional): The orders index to filter the
            raw data table by.

    Returns:
        pd.DataFrame: The order payments data table aggregated by order ID,
            optionally filtered by the orders index.
    """
    data = Cache.init(
        'order_payments_by_order',
        _load_order_payments_by_order
    )
    return filtered_copy(data, rows_filter=orders_index)


"""def get_customer_order_payments_v1():
    "" "Get a DataFrame of customer, order and payment data.

    DEPRECATED : use `get_order_payments_by_customer` instead
    "" "
    customers = get_raw_customers()[['customer_id', 'customer_unique_id']]
    orders = get_raw_orders()[
        ['order_id', 'customer_id', 'order_purchase_timestamp']
    ]
    "" "orders.order_purchase_timestamp = (
        orders.order_purchase_timestamp
        .astype('datetime64[ns]')
    )"" "
    agg_order_payments = get_aggregated_order_payments_v1()[['payment_total']]
    m = pd.merge(customers, orders, how='outer', on='customer_id')
    m = pd.merge(m, agg_order_payments, how='outer', on='order_id')
    m = m.sort_values(by='order_purchase_timestamp', ascending=False)
    m = m.drop(columns=['customer_id'])
    m = m.set_index('order_id')
    return m"""


"""def _load_cached_order_payments_by_customer_v1():
    cop = get_customers_orders_payments().reset_index()
    cop = cop[[
        ('_ident_', 'customer_id'),
        ('orders', 'purchase_timestamp'),
        ('payments', 'value')
    ]]

    cop.columns = ['customer_id', 'purchase_date', 'value']

    op_by_c = (
        cop.groupby('customer_id')
        .agg({
            'value': ['count', 'sum', 'min', 'max', 'mean'],
            'purchase_date': ['min', 'max']
        })
    )

    return op_by_c"""


def _load_order_payments_by_customer():
    op_by_o = get_order_payments_by_order()
    op_by_o.reset_index(inplace=True)

    op_by_c = op_by_o.groupby(by='customer_id').agg({
        ('order_id', ''): tuple,
        ('value', 'count'): ['count', 'sum'],
        ('value', 'sum'): ['sum', 'min', 'max', 'mean'],
        ('purchase_date', 'min'): 'min',
        ('purchase_date', 'max'): 'max',
        ('sequential', 'tuple'): tuple,
        ('type', 'tuple'): tuple,
        ('installments', 'tuple'): tuple
    })

    op_by_c.columns = [
        'order_id_tuple', 'orders_count',
        'payments_count', 'payments_total',
        'order_payment_min', 'order_payment_max', 'order_payment_mean',
        'first_purchase_date', 'last_purchase_date',
        'sequentials_tuple', 'types_tuple', 'installments_tuple'
    ]

    return op_by_c


def get_order_payments_by_customer(
    customers_index: Optional[Iterable] = None
) -> pd.DataFrame:
    data = Cache.init(
        'order_payments_by_customer',
        _load_order_payments_by_customer
    )
    return filtered_copy(data, rows_filter=customers_index)


""" By date filtering
"""


def get_last_order_date_v1():
    """Get the last order date.
    DEPRECATED
    """
    return (
        get_raw_orders()
        .order_purchase_timestamp
        .astype('datetime64[ns]')
        .max()
    )


def get_first_order_date_v1():
    """Get the first order date.
    DEPERECATED
    """
    return (
        get_raw_orders()
        .order_purchase_timestamp
        .astype('datetime64[ns]')
        .min()
    )


def _event_col(
    event: str
) -> str:
    events_dict = {
        'purchase': 'purchase_timestamp',
        'approval': 'approved_at',
        'carrier_delivery': 'delivered_carrier_date',
        'customer_delivery': 'delivered_customer_date',
        'estimated_delivery': 'estimated_delivery_date'
    }
    if event is None:
        return events_dict['purchase']
    if event in events_dict:
        return events_dict[event]
    return ''


def get_first_order_date(
    orders_index: Optional[Iterable] = None,
    event: Optional[str] = None
) -> datetime:
    """Get the first order date.
    """
    return get_orders(orders_index)[_event_col(event)].min()


def get_last_order_date(
    orders_index: Optional[Iterable] = None,
    event: Optional[str] = None
) -> datetime:
    """Get the last order date.
    """
    return get_orders(orders_index)[_event_col(event)].max()


def get_order_ages_v0(now):
    """Get the ages of all orders at a given time.

    DEPRECATED

    Args:
        now (datetime): The reference time for calculating the ages.

    Returns:
        Series: A Series with the order ages, indexed by order id.
    """
    return now - (
        get_raw_orders()
        .set_index('order_id')
        .order_purchase_timestamp
        .astype('datetime64[ns]')
        .sort_values(ascending=False)
        .rename('order_age')
    )


def get_order_ages_v1(
        from_date=get_first_order_date(),
        to_date=get_last_order_date()
):
    """
    Return the age of orders placed between the given
    `from_date` and `to_date` dates.
    If no dates are given, the function will use
    the first and last order dates in the orders table.
    """
    ord = get_raw_orders()
    is_ord_between = (
        (from_date <= ord.order_purchase_timestamp)
        & (ord.order_purchase_timestamp <= to_date)
    )
    ordb = ord[is_ord_between]
    return to_date - (
        ordb
        .set_index('order_id')
        .order_purchase_timestamp
        .astype('datetime64[ns]')
        .sort_values(ascending=False)
        .rename('order_age')
    )


def get_order_event_ages(
    present_date: Optional[datetime] = datetime.now(),
    orders_index: Optional[Iterable] = None,
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    event: Optional[str] = None
) -> pd.Series:
    """
    Return the age of orders placed between the given
    `from_date` and `to_date` dates.
    If no dates are given, the function will use
    the first and last order dates in the orders table.
    """
    orders = get_orders(orders_index=orders_index)

    if from_date is None:
        from_date = get_first_order_date(
            orders_index=orders_index,
            event=event
        )

    if to_date is None:
        to_date = get_last_order_date(
            orders_index=orders_index,
            event=event
        )

    event_dates = orders[_event_col(event)]

    between_dates = (
        (from_date <= event_dates)
        & (event_dates <= to_date)
    )

    event_dates = event_dates[between_dates]

    return (present_date - event_dates).sort_values(ascending=False)


""" Derived features : order times
"""


def get_order_times_v0():
    """
    Get a DataFrame of order time data.

    DEPRECATED

    Returns:
    pandas.DataFrame: A DataFrame of order time data.
    """
    orders = get_raw_orders()
    return pd.concat([
        orders[['order_id', 'customer_id', 'order_status']],
        (
            orders.order_delivered_customer_date.astype('datetime64[ns]')
            - orders.order_purchase_timestamp.astype('datetime64[ns]')
        ).rename('order_processing_times'),
        (
            orders.order_estimated_delivery_date.astype('datetime64[ns]')
            - orders.order_purchase_timestamp.astype('datetime64[ns]')
        ).rename('order_processing_estimated_times'),
        (
            orders.order_estimated_delivery_date.astype('datetime64[ns]')
            - orders.order_delivered_customer_date.astype('datetime64[ns]')
        ).rename('order_delivery_advance_times'),
        (
            orders.order_approved_at.astype('datetime64[ns]')
            - orders.order_purchase_timestamp.astype('datetime64[ns]')
        ).rename('order_approval_times'),
        (
            orders.order_delivered_carrier_date.astype('datetime64[ns]')
            - orders.order_approved_at.astype('datetime64[ns]')
        ).rename('order_carrier_delivery_times'),
        (
            orders.order_delivered_customer_date.astype('datetime64[ns]')
            - orders.order_delivered_carrier_date.astype('datetime64[ns]')
        ).rename('order_customer_delivery_times'),
    ], axis=1)


def get_order_times_v1(index=None):
    """Get a DataFrame of order times data.

    DEPRECATED
    """
    orders = get_orders(index=index)
    return pd.concat([
        orders.status,
        (
            orders.delivered_customer_date
            - orders.purchase_timestamp
        ).rename('processing_times'),
        (
            orders.estimated_delivery_date
            - orders.purchase_timestamp
        ).rename('processing_estimated_times'),
        (
            orders.estimated_delivery_date
            - orders.delivered_customer_date
        ).rename('delivery_advance_times'),
        (
            orders.approved_at
            - orders.purchase_timestamp
        ).rename('approval_times'),
        (
            orders.delivered_carrier_date
            - orders.approved_at
        ).rename('carrier_delivery_times'),
        (
            orders.delivered_customer_date
            - orders.delivered_carrier_date
        ).rename('customer_delivery_times'),
    ], axis=1)


def get_order_times(
    orders_index: Optional[Iterable] = None
) -> pd.DataFrame:
    """Get a DataFrame of order times data.
    """
    orders = get_orders(orders_index=orders_index)

    def time_diff(data, from_date, to_date, name):
        return (data[to_date] - data[from_date]).rename(name)

    t1 = 'purchase_timestamp'
    t2 = 'approved_at'
    t3 = 'delivered_carrier_date'
    t4 = 'delivered_customer_date'
    t5 = 'estimated_delivery_date'

    dt12 = 'approval_time'
    dt13 = 'carrier_delivering_time'
    dt14 = 'customer_delivering_time'
    dt15 = 'processing_estimated_time'
    dt23 = 'approval_to_carrier_delivery_time'
    dt24 = 'approval_to_customer_delivery_time'
    dt25 = 'approval_to_estimated_delivery_time'
    dt34 = 'transit_time'
    dt35 = 'estimated_transit_time'
    dt45 = 'delivery_advance_time'

    return pd.concat([
        time_diff(orders, t1, t2, dt12),
        time_diff(orders, t1, t3, dt13),
        time_diff(orders, t1, t4, dt14),
        time_diff(orders, t1, t5, dt15),
        time_diff(orders, t2, t3, dt23),
        time_diff(orders, t2, t4, dt24),
        time_diff(orders, t2, t5, dt25),
        time_diff(orders, t3, t4, dt34),
        time_diff(orders, t3, t5, dt35),
        time_diff(orders, t4, t5, dt45),
    ], axis=1)


""" Derived features : R, F, M
"""


def _index_args(kwargs: dict) -> dict:
    return {
        k: v for k, v in kwargs.items()
        if k.endswith('_index')
    }


def _event_dating_args(kwargs: dict) -> dict:
    dating_args = ['present_date', 'from_date', 'to_date', 'event']
    return {
        k: v for k, v in kwargs.items()
        if k in dating_args
    }


def get_customer_order_recency_v1(
        from_date=get_first_order_date(),
        to_date=get_last_order_date()
):
    cop = get_customer_order_payments_v1()
    order_age = get_order_ages_v1(get_last_order_date())
    copa = pd.concat([cop, order_age], axis=1)
    is_copa_between = (
        (from_date <= copa.order_purchase_timestamp)
        & (copa.order_purchase_timestamp <= to_date)
    )
    copab = copa[is_copa_between]

    customer_recency = copab.drop_duplicates(subset='customer_unique_id')
    customer_recency = customer_recency.set_index('customer_unique_id')
    return customer_recency
   

"""def get_customer_order_recency(
    customers_index: Optional[Iterable] = None,
    orders_index: Optional[Iterable] = None,
    present_date: Optional[datetime] = datetime.now(),
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    event: Optional[str] = None
) -> pd.DataFrame:

    kwargs = locals()
    customer_payments = get_customer_payments(_index_args(kwargs))
    order_ages = get_order_event_ages(_event_dating_args(kwargs))

    event_dates = orders[_event_col(event)]

    between_dates = (
        (from_date <= event_dates)
        & (event_dates <= to_date)
    )

    event_dates = event_dates[between_dates]"""



def get_customer_order_freqs_and_amount(
        from_date=get_first_order_date(),
        to_date=get_last_order_date()
):
    """
    Returns a DataFrame containing the customer
    unique id and the order recency for each customer.

    The order recency is the age (in days)
    of the most recent order made by the customer.

    The `from_date` and `to_date` parameters are optional and can be used
    to filter the orders considered to compute the order recency.
    They default to the minimum and maximum order purchase dates
    in the data, respectively.

    The `from_date` and `to_date` parameters must be strings
    in the ISO 8601 format, such as "2022-12-18".
    """
    cop = get_customer_order_payments_v1()
    is_cop_between = (
        (from_date <= cop.order_purchase_timestamp)
        & (cop.order_purchase_timestamp <= to_date)
    )
    copb = cop[is_cop_between]
    gpby = copb.groupby(by='customer_unique_id').agg({
        'order_purchase_timestamp': 'count',
        'payment_total': 'sum',
    })
    gpby.columns = ['order_count', 'order_amount']
    gpby = gpby.sort_values(by='order_count', ascending=False)
    return gpby


def get_customer_RFM(
        from_date=get_first_order_date(),
        to_date=get_last_order_date()
):
    """
    Return a dataframe containing
    RFM (Recency, Frequency, Monetary) values for each customer.

    The from_date and to_date parameters can be used to specify the period
    over which the RFM values are computed.

    Returns:
    A dataframe with columns ['R', 'F', 'M']
    containing the RFM values for each customer.
    """
    cor = get_customer_order_recency_v1(from_date, to_date)
    cofa = get_customer_order_freqs_and_amount(from_date, to_date)
    crfm = pd.merge(
        cor[['order_age']], cofa,
        how='outer', on='customer_unique_id'
    )
    crfm.columns = ['R', 'F', 'M']
    oneday = pd.Timedelta(days=1)
    crfm.R = crfm.R / oneday
    return crfm


def get_product_physical_features(
    products_index: Optional[Iterable] = None
) -> pd.DataFrame:
    products = get_products(products_index=products_index)
    volume = (
        products.length_cm
        * products.height_cm
        * products.width_cm
    ).rename('volume_cm^3')
    density = (
        products.weight_g / volume
    ).rename('density_g_cm^-3')
    return pd.concat([volume, density], axis=1)


""" Plotting
"""


def plot_clusters_2d_v1(x, y, title, xlabel, ylabel, clu_labels):
    """
    Plots a 2D scatter plot.

    Parameters:
    x (array-like): The x coordinates of the points in the scatter plot.
    y (array-like): The y coordinates of the points in the scatter plot.
    title (str): The title of the plot.
    xlabel (str): The label for the x axis.
    ylabel (str): The label for the y axis.
    clu_labels (array-like): The cluster labels for each point.

    Returns:
    None
    """
    plt.scatter(x, y, c=clu_labels)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plot_clusters_2d(ax, title, xy, xy_labels, xy_clu_centers, clu_labels):
    """
    Create a scatter plot for the given data, with labels for each cluster.
    Parameters

    ax: matplotlib.axes.Axes
    The matplotlib Axes to draw the plot on.
    title: str
    The title of the plot.
    xy: tuple of two numpy.ndarray
    The two arrays representing the x and y coordinates of the data points.
    xy_labels: tuple of two str
    The labels for the x and y coordinates.
    xy_clu_centers: tuple of two numpy.ndarray
    The two arrays representing the x and y coordinates of the cluster centers.
    clu_labels: numpy.ndarray
    The array of labels for each data point, indicating its cluster membership.
    """
    n_clusters = len(np.unique(clu_labels))
    colors = cm.nipy_spectral(clu_labels.astype(float) / n_clusters)
    ax.scatter(
        xy[0], xy[1],
        marker='.', s=30, lw=0, alpha=0.7,
        c=colors, edgecolor='k',
    )

    # Clusters labeling
    ax.scatter(
        xy_clu_centers[0], xy_clu_centers[1],
        marker='o', c='white', alpha=1, s=200,
        edgecolor='k',
    )
    for i, (c_x, c_y) in enumerate(zip(xy_clu_centers[0], xy_clu_centers[1])):
        ax.scatter(
            c_x, c_y,
            marker=f'${i}$', alpha=1, s=50,
            edgecolor='k'
        )

    ax.set_title(title)
    ax.set_xlabel(xy_labels[0])
    ax.set_ylabel(xy_labels[1])


def plot_clusters_3d_v1(xyz, xyz_labels, clu_labels, title, figsize=(13, 13)):
    """Plots a 3D scatter plot of the given data points and their labels.

    Parameters
    ----------
    xyz : list
        List of three 1D numpy arrays containing
        the x, y, and z coordinates of the data points.
    xyz_labels : list
        List of three strings representing the labels for
        the x, y, and z axes.
    clu_labels : 1D numpy array
        Array containing the labels for each data point.
    title : str
        Title for the plot.
    figsize : tuple, optional
        Figure size for the plot (width, height), by default (13, 13)

    Returns
    -------
    None
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d', elev=10, azim=140)
    ax.scatter(
        xyz[0], xyz[1], xyz[2],
        c=clu_labels, cmap=cm.Set1_r, edgecolor='k'
    )
    ax.set_xlabel(xyz_labels[0])
    ax.set_ylabel(xyz_labels[1])
    ax.set_zlabel(xyz_labels[2])
    ax.set_title(title)


def plot_clusters_3d(ax, title, xyz, xyz_labels, clu_labels):
    """Plots a 3D scatter plot of the given data points and their labels.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object to plot on.
    title : str
        Title for the plot.
    xyz : list
        List of three 1D numpy arrays containing
        the x, y, and z coordinates of the data points.
    xyz_labels : list
        List of three strings representing the labels for
        the x, y, and z axes.
    clu_labels : 1D numpy array
        Array containing the labels for each data point.

    Returns
    -------
    None
    """
    n_clusters = len(np.unique(clu_labels))
    colors = cm.nipy_spectral(clu_labels.astype(float) / n_clusters)
    ax.scatter(xyz[0], xyz[1], xyz[2], c=colors)
    ax.set_xlabel(xyz_labels[0])
    ax.set_ylabel(xyz_labels[1])
    ax.set_zlabel(xyz_labels[2])
    ax.set_title(title)


def plot_silhouette(ax, silhouette_avg, silhouette_values, clu_labels):
    """
    Plots the silhouette plot for the given data.
    Parameters
    ----------
    ax: Matplotlib axis object
        The axis to draw the plot on.
    silhouette_avg: float
        The average silhouette score.
    silhouette_values: numpy array
        The silhouette scores for each sample.
    clu_labels: numpy array
        The cluster labels for each sample.
    Returns
    ----------
    None
    """
    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    # ax.set_xlim([-0.1, 1])
    # The (n_clusters + 1) * 10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    n_clusters = len(np.unique(clu_labels))
    ax.set_ylim([0, len(silhouette_values) + (n_clusters + 1) * 10])

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = silhouette_values[clu_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax.set_title("SILHOUETTE")
    ax.set_xlabel("Silhouette values")
    ax.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])


def plot_kmeans_rfm_clusters(
        rfm, rfm_labels, rfm_centers,
        clu_labels, slh_avg, slh_vals
):
    """Plot the K-Means clustering results for the RFM features.

    Parameters:
    - rfm (list): The RFM features as a list of 3 elements (r, f, m).
    - rfm_labels (list): The RFM feature labels as a list of 3 elements
      (r_label, f_label, m_label).
    - rfm_centers (list): The RFM feature clusters centers as a list of 3
      elements (r_centers, f_centers, m_centers).
    - clu_labels (array): The cluster labels for each sample.
    - slh_avg (float): The average silhouette score for all the samples.
    - slh_vals (array): The silhouette score for each sample.
    """
    n_clusters = len(np.unique(clu_labels))
    r, f, m = rfm[0], rfm[1], rfm[2]
    r_label, f_label, m_label = rfm_labels[0], rfm_labels[1], rfm_labels[2]
    r_centers, f_centers, m_centers = (
        rfm_centers[0], rfm_centers[1], rfm_centers[2]
    )

    fig = plt.figure(figsize=(15, 7))

    ax1 = plt.subplot2grid(
        (2, 4), (0, 0),
        colspan=2, rowspan=2,
        projection='3d', elev=10, azim=140
    )
    ax2 = plt.subplot2grid((2, 4), (0, 2))
    ax3 = plt.subplot2grid((2, 4), (0, 3))
    ax4 = plt.subplot2grid((2, 4), (1, 2))
    ax5 = plt.subplot2grid((2, 4), (1, 3))

    plot_clusters_3d(
        ax=ax1,
        title=f'RMF 3D',
        xyz=[r, m, f],
        xyz_labels=[r_label, m_label, f_label],
        clu_labels=clu_labels,
    )

    plot_silhouette(ax2, slh_avg, slh_vals, clu_labels)

    ax3.semilogy()
    plot_clusters_2d(
        ax3, 'RM',
        xy=[r, m], xy_labels=[r_label, m_label],
        xy_clu_centers=[r_centers, m_centers],
        clu_labels=clu_labels
    )

    plot_clusters_2d(
        ax4, 'FR',
        xy=[f, r], xy_labels=[f_label, r_label],
        xy_clu_centers=[f_centers, r_centers],
        clu_labels=clu_labels
    )

    ax5.semilogy()
    plot_clusters_2d(
        ax5, 'FM',
        xy=[f, m], xy_labels=[f_label, m_label],
        xy_clu_centers=[f_centers, m_centers],
        clu_labels=clu_labels
    )

    plt.tight_layout()

    plt.suptitle(
        f'{n_clusters}-Means clusters',
        fontsize=14,
        fontweight='bold',
        y=1.05,
    )
    plt.show()


def plot_kmeans_rfm_clusters_v2(
        rfm, rfm_labels,
        clu_labels, clu_centers,
        slh_avg, slh_vals
):
    """Plot the K-Means clustering results for the RFM features.

    Parameters:
    - rfm (list): The RFM features as a list of 3 elements (r, f, m).
    - rfm_labels (list): The RFM feature labels as a list of 3 elements
      (r_label, f_label, m_label).
    - rfm_centers (list): The RFM feature clusters centers as a list of 3
      elements (r_centers, f_centers, m_centers).
    - clu_labels (array): The cluster labels for each sample.
    - slh_avg (float): The average silhouette score for all the samples.
    - slh_vals (array): The silhouette score for each sample.
    """
    n_clusters = len(np.unique(clu_labels))
    r, f, m = rfm[0], rfm[1], rfm[2]
    r_label, f_label, m_label = rfm_labels[0], rfm_labels[1], rfm_labels[2]
    r_centers, f_centers, m_centers = (
        clu_centers[:, 0],
        clu_centers[:, 1],
        clu_centers[:, 2],
    )

    plt.figure(figsize=(15, 7))

    ax1 = plt.subplot2grid(
        (2, 4), (0, 0),
        colspan=2, rowspan=2,
        projection='3d', elev=10, azim=140
    )
    ax2 = plt.subplot2grid((2, 4), (0, 2))
    ax3 = plt.subplot2grid((2, 4), (0, 3))
    ax4 = plt.subplot2grid((2, 4), (1, 2))
    ax5 = plt.subplot2grid((2, 4), (1, 3))

    plot_clusters_3d(
        ax=ax1,
        title=f'RMF 3D',
        xyz=[r, m, f],
        xyz_labels=[r_label, m_label, f_label],
        clu_labels=clu_labels,
    )

    plot_silhouette(ax2, slh_avg, slh_vals, clu_labels)

    ax3.semilogy()
    plot_clusters_2d(
        ax3, 'RM',
        xy=[r, m], xy_labels=[r_label, m_label],
        xy_clu_centers=[r_centers, m_centers],
        clu_labels=clu_labels
    )

    plot_clusters_2d(
        ax4, 'FR',
        xy=[f, r], xy_labels=[f_label, r_label],
        xy_clu_centers=[f_centers, r_centers],
        clu_labels=clu_labels
    )

    ax5.semilogy()
    plot_clusters_2d(
        ax5, 'FM',
        xy=[f, m], xy_labels=[f_label, m_label],
        xy_clu_centers=[f_centers, m_centers],
        clu_labels=clu_labels
    )

    plt.tight_layout()

    plt.suptitle(
        f'{n_clusters}-Means clusters',
        fontsize=14,
        fontweight='bold',
        y=1.05,
    )
    plt.show()


def kmeans_clustering(crfm, k, normalize=False):
    """
    Perform K-Means clustering on customer RFM data.
    Parameters
    crfm: Pandas DataFrame
        DataFrame with columns 'R', 'F', and 'M' representing
        recency, frequency, and monetary value of customer orders.
    k: int
        Number of clusters to form.
    normalize: Boolean
        If True (defaut is False), crfm is normalized before
        performing K-Means clustering

    Returns
    kmeans: sklearn.cluster.KMeans
        KMeans model with the best parameters.
    clu_labels: numpy.ndarray
        Array of shape (n_samples,) containing the cluster labels
        for each point in the input data.
    rfm: tuple
        Tuple of Pandas Series ('R', 'F', 'M') representing
        recency, frequency, and monetary value of customer orders.
    rfm_labels: tuple
        Tuple of str ('Recency', 'Frequency', 'Monetary') representing
        the names of the columns in rfm.
    rfm_centers: tuple
        Tuple of numpy.ndarray containing
        the coordinates of the cluster centers.
    km_t: float
        Time taken to fit and predict with the KMeans model.
    """
    km_t = -time.time()
    # Normalize the data
    crfm_scaled = crfm
    if normalize:
        scaler = StandardScaler()
        crfm_scaled = scaler.fit_transform(crfm)
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit_predict(crfm_scaled)
    km_t += time.time()
    clu_labels = kmeans.labels_
    # clu_centers = kmeans.cluster_centers_
    # rfm = r, f, m = crfm.R, crfm.F, crfm.M
    rfm = crfm.R, crfm.F, crfm.M
    # rfm_labels = r_label, f_label, m_label = \
    # 'Recency', 'Frequency', 'Monetary'
    rfm_labels = 'Recency', 'Frequency', 'Monetary'
    """rfm_centers = r_centers, f_centers, m_centers = (
        clu_centers[:, 0],
        clu_centers[:, 1],
        clu_centers[:, 2],
    )"""
    rfm_centers = get_centers(crfm, clu_labels)
    return kmeans, clu_labels, rfm, rfm_labels, rfm_centers, km_t


def kmeans_clustering_v2(crfm, k, normalize=False):
    km_t = -time.time()
    # Normalize the data
    X = crfm
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(crfm)
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit_predict(X)
    km_t += time.time()
    clu_labels = kmeans.labels_
    # clu_centers = kmeans.cluster_centers_
    # rfm = r, f, m = crfm.R, crfm.F, crfm.M
    # rfm = crfm.R, crfm.F, crfm.M
    # rfm_labels = r_label, f_label, m_label = \
    # 'Recency', 'Frequency', 'Monetary'
    # rfm_labels = 'Recency', 'Frequency', 'Monetary'
    """rfm_centers = r_centers, f_centers, m_centers = (
        clu_centers[:, 0],
        clu_centers[:, 1],
        clu_centers[:, 2],
    )"""
    clu_centers = get_centers_v2(crfm, clu_labels)
    return kmeans, clu_labels, clu_centers, km_t


def kmeans_analysis(crfm, k):
    """Perform k-means clustering analysis on
    a customer RFM (recency, frequency, monetary value) dataset.

    Parameters:

        crfm (DataFrame):
            RFM dataset with customer-level scores
            for recency, frequency, and monetary value

        k (int):
            Number of clusters to use in k-means clustering

    Returns:

        tuple:
            A tuple containing the silhouette average, k-means fit time,
            and silhouette compute time
    """
    (
        _, clu_labels, rfm, rfm_labels, rfm_centers, km_t
    ) = kmeans_clustering(crfm, k)
    slh_t = -time.time()
    slh_avg = silhouette_score(crfm, clu_labels)
    slh_vals = silhouette_samples(crfm, clu_labels)
    slh_t += time.time()
    plot_kmeans_rfm_clusters(
        rfm, rfm_labels, rfm_centers,
        clu_labels, slh_avg, slh_vals)
    plt.show()
    print('silhouette average :', round(slh_avg, 3))
    print('k-means fit time :', round(km_t, 3))
    print('silouhette compute time :', round(slh_t, 3))
    return slh_avg, km_t, slh_t


def classes_labeling_v1(rfm, classes_def):
    """Assign a label to each row of rfm
    based on the classes definition in `classes_def`.

    Parameters
    ----------
    rfm : pd.DataFrame
        DataFrame with columns 'R', 'F', 'M' containing the RFM features.
    classes_def : dict
        Dictionary containing the classes definition.
        The keys are the class ids and the values are dictionaries
        containing the lower and upper bounds for 'R' and 'M' features.

    Returns
    -------
    pd.Series
        Series of class labels with the same index as rfm.

    Example
    -------
    classes_def = {
        0: {'R': [0, np.inf], 'M': [600, np.inf]},
        1: {'R': [0, 300], 'M': [0, 600]},
        2: {'R': [300, np.inf], 'M': [0, 600]},
    }
    cla_labels = classes_labeling(crfm_1, classes_def)
    display(cla_labels)
    """
    label = pd.Series(data=-1, index=rfm.index, name='label')
    for c_id in classes_def:
        c_def = classes_def[c_id]
        c_bindex = (
            ((c_def['R'][0] <= rfm.R) & (rfm.R < c_def['R'][1]))
            & ((c_def['M'][0] <= rfm.M) & (rfm.M < c_def['M'][1]))
        )
        label[c_bindex] = c_id
    return label


def cluster_to_abstract_class(
    cluster: List[List[float]]
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    # Initialiser les intervalles minimaux et maximaux à des valeurs extrêmes
    min_intervals = [float("inf"), float("inf"), float("inf")]
    max_intervals = [float("-inf"), float("-inf"), float("-inf")]

    # Pour chaque point du cluster
    for point in cluster:
        # Mettre à jour les intervalles minimaux et maximaux si nécessaire
        for i in range(3):
            min_intervals[i] = min(min_intervals[i], point[i])
            max_intervals[i] = max(max_intervals[i], point[i])

    # Renvoyer les intervalles sous forme de tuples
    return tuple(zip(min_intervals, max_intervals))


def insert_cl_labels(crfm, cl_labels, name='cl'):
    return pd.concat([
        pd.Series(cl_labels, index=crfm.index, name=name),
        crfm
    ], axis=1)


def get_cl_counts(crfm_labeled, name='cl'):
    return crfm_labeled[name].value_counts()


def get_cluster(crfm_labeled, clu_index):
    clu = crfm_labeled[crfm_labeled.k_clu == clu_index]
    return clu[['R', 'F', 'M']]


def get_abstracted_classes(crfm, clu_labels):
    k_values = np.unique(clu_labels)
    crfm_labeled = insert_cl_labels(crfm, clu_labels)
    return {
        k: cluster_to_abstract_class(get_cluster(crfm_labeled, k).values)
        for k in k_values
    }


def clusters_business_analysis(crfm, k, classes_def):
    """Perform business analysis on clusters generated by k-Means clustering.

    Parameters:

        crfm (pandas.DataFrame):
            data to cluster, with columns 'R', 'F', 'M'
        k (int):
            number of clusters to generate
        classes_def (dict):
            manual labeling of the data, in the form of nested dictionaries
        mapping class labels to criteria on features 'R' and 'M'
        in the form of ranges in the form:
        {
        class_label_1: {'R': [min_R, max_R], 'M': [min_M, max_M]},
        class_label_2: {'R': [min_R, max_R], 'M': [min_M, max_M]},
        ...
        }

    Returns:

        None
    """
    # k-Means clustering
    (
        kmeans, clu_labels, rfm, rfm_labels, rfm_centers, km_t
    ) = kmeans_clustering(crfm, k)
    r_label, f_label, m_label = rfm_labels
    # Labeling
    print_subtitle('Labeling')
    crfm_labeled = pd.concat([
        pd.Series(clu_labels, index=crfm.index, name='k_clu'),
        crfm
    ], axis=1)
    display(crfm_labeled)
    # Cluster cardinalities
    print_subtitle('Cluster cardinalities')
    clu_counts = crfm_labeled.k_clu.value_counts()
    display(clu_counts)
    # Cluster per feature stats
    print_subtitle('Cluster per feature stats')
    gpby = crfm_labeled.groupby(by='k_clu').agg(
        ['min', 'max', 'mean', 'median', 'std', 'skew', pd.Series.kurt]
    )
    print(r_label)
    display(gpby.R)
    print(m_label)
    display(gpby.M)
    print(f_label + ' (less pertinent)')
    display(gpby.F)

    # Turnover
    print_subtitle('Turnover')
    m_labeled = crfm_labeled[['k_clu', 'M']]
    m_gpby = m_labeled.groupby(by='k_clu').agg(
        [
            'count', 'min', 'max', 'sum', 'mean',
            'median', 'std', 'skew', pd.Series.kurt
        ]
    )
    turnover_abs = m_gpby.M['sum'].rename('toa')
    turnover_rel = (turnover_abs / turnover_abs.sum()).rename('tor')
    turnover = pd.concat([turnover_abs, turnover_rel], axis=1)
    display(turnover)
    display(m_gpby.M)
    # Hypercubic roughing of the domain
    print_subtitle('Hypercubic roughing of the domain')
    gpby_3 = crfm_labeled.groupby(by='k_clu').agg(
        ['min', 'max']
    )
    display(gpby_3)
    # Manual sterotyping
    import numpy as np
    print_subtitle('Manual sterotyping')
    cla_labels = classes_labeling_v1(crfm, classes_def)
    crfm_mlabeled = pd.concat([
        pd.Series(cla_labels, index=crfm.index, name='k_cla'),
        crfm
    ], axis=1)
    display(crfm_mlabeled)
    # Compare cluster (machine learning) | classes (manual)
    print_subtitle('Compare cluster (machine learning) | classes (manual)')
    cla_counts = crfm_mlabeled.k_cla.value_counts()
    # display(cla_counts)
    cl_comp = pd.concat([clu_counts, cla_counts], axis=1)
    display(cl_comp)


# geodata bonus

# def select_region(
#   geolocation: pd.DataFrame,
#   lng_limits: Tuple[float, float],
#   lat_limits: Tuple[float, float]) -> pd.DataFrame:
def select_region(geolocation, lng_limits, lat_limits):
    """Select the geolocations inside the specified region defined by the
    longitude and latitude limits.

    Parameters
    ----------
    geolocation : pd.DataFrame
        A dataframe with 'lng' and 'lat' columns, representing the geolocations
        to filter.
    lng_limits : tuple of float
        A tuple of two floats defining the minimum and maximum longitude
        limits of the region.
    lat_limits : tuple of float
        A tuple of two floats defining the minimum and maximum latitude
        limits of the region.

    Returns
    -------
    pd.DataFrame
        The subset of `geolocation` with geolocations inside the specified
        region.
    """
    lng_limits = (
        (lng_limits[0] < geolocation.lng)
        & (geolocation.lng < lng_limits[1])
    )
    lat_limits = (
        (lat_limits[0] < geolocation.lat)
        & (geolocation.lat < lat_limits[1])
    )
    return geolocation[lng_limits & lat_limits]


def select_outof_region(geolocation, lng_limits, lat_limits):
    """Select the geolocations outside the specified region defined by the
    longitude and latitude limits.

    Parameters
    ----------
    geolocation : pd.DataFrame
        A dataframe with 'lng' and 'lat' columns, representing the geolocations
        to filter.
    lng_limits : tuple of float
        A tuple of two floats defining the minimum and maximum longitude
        limits of the region.
    lat_limits : tuple of float
        A tuple of two floats defining the minimum and maximum latitude
        limits of the region.

    Returns
    -------
    pd.DataFrame
        The subset of `geolocation` with geolocations outside the specified
        region.
    """
    lng_limits = (
        (lng_limits[0] < geolocation.lng)
        & (geolocation.lng < lng_limits[1])
    )
    lat_limits = (
        (lat_limits[0] < geolocation.lat)
        & (geolocation.lat < lat_limits[1])
    )
    return geolocation[~(lng_limits & lat_limits)]


"""Scraping"""


def get_brazil_states():
    """Get the dataframe containing the list of states in Brazil
    from Wikipedia.
    """
    # get the HTML content
    url = 'https://en.wikipedia.org/wiki/Federative_units_of_Brazil#List'
    response = requests.get(url)
    # print(response.status_code)

    # parse data from the html into a beautifulsoup object
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'class': 'wikitable sortable'})

    # turn it into dataframe
    return pd.read_html(str(table))[0]


def scrap_brazil_municipalities():
    """Get the dataframe containing the list of municipalities in Brazil
    from Wikipedia.
    """
    # get the HTML content
    url = 'https://en.wikipedia.org/wiki/List_of_municipalities_of_Brazil'
    response = requests.get(url)
    # print(response.status_code)

    # parse data from the html into a beautifulsoup object
    soup = BeautifulSoup(response.text, 'html.parser')
    data_list = []
    for table in soup.find_all('table', {'class': 'wikitable sortable'}):
        # Find the h2 (Brazil's region)
        #      and h3 (Brazil's state) elements preceding the table
        h2 = table.find_previous('h2')
        h3 = table.find_previous('h3')
        # Extract the text of the h2 and h3 element
        region_name = h2.text[:-len('[edit]')]
        state_name_and_code = h3.text[:-len('[edit]')]
        state_code = state_name_and_code[-3:-1]
        state_name = state_name_and_code[:-5]
        # Read the table and turn it into dataframe
        data = pd.read_html(str(table))[0]
        # Insert region and state data to the dataframe
        data.insert(0, 'CO', state_code)
        data.insert(0, 'State', state_name)
        data.insert(0, 'Region', region_name)
        # Add the dataframe to the list
        data_list += [data]

    # merge into a unique dataframe
    all = pd.concat(data_list, axis=0, ignore_index=True)

    # Add a column 'is_state_capital' and remove (State Capital) from name
    is_state_capital = all.Municipality.str.endswith('(State Capital)')
    all['is_state_capital'] = is_state_capital
    fixed_names = all.Municipality.str[:-len(' (State Capital)')]
    all.loc[is_state_capital, 'Municipality'] = fixed_names
    return all


def get_municipalities(municipalities=None):
    if municipalities is None:
        municipalities = scrap_brazil_municipalities()
    return (
        municipalities[['CO', 'Municipality']]
        .drop_duplicates()
        .sort_values(by='CO')
        .reset_index(drop=True)
    )


def remove_accents(s):
    return unidecode(s)


def get_municipality_states(municipalities=None):
    if municipalities is None:
        municipalities = scrap_brazil_municipalities()
    municipality_states = (
        municipalities[['CO', 'State']]
        .drop_duplicates()
        .sort_values(by='CO')
        .reset_index(drop=True)
    )
    municipality_states.columns = ['2A', 'Stãte']
    municipality_states['State'] = (
        municipality_states.Stãte.apply(remove_accents)
    )
    return municipality_states


def load_brazil_zip_codes():
    """
    0. country code      : iso country code, 2 characters
    1. postal code       : varchar(20)
    2. place name        : varchar(180)
    3. admin name1       : 1. order subdivision (state) varchar(100)
    4. admin code1       : 1. order subdivision (state) varchar(20)
    _5. admin name2       : 2. order subdivision (county/province) varchar(100)
    6. admin code2       : 2. order subdivision (county/province) varchar(20)
    _7. admin name3       : 3. order subdivision (community) varchar(100)
    _8. admin code3       : 3. order subdivision (community) varchar(20)
    9. latitude          : estimated latitude (wgs84)
    10. longitude         : estimated longitude (wgs84)
    11. accuracy          : accuracy of lat/lng from
                            1=estimated, 4=geonameid,
                            6=centroid of addresses or shape
    """
    # Raw loading
    zips = pd.read_csv(
        f'../data/geonames_BR/BR.txt',
        sep='\t',
        header=None,
        dtype=object
    )
    zips = zips.dropna(axis=1, how='all')
    zips.columns.name = 'zip_codes'
    zips.columns = [
        'country_code', 'postal_code', 'place_name',
        'admin name1', 'admin_code1', 'admin code2',
        'latitude', 'longitude', 'accuracy'
    ]
    # Feature engineering
    zips = zips.drop(columns=['country_code', 'accuracy'])  # rmv cst cols
    zips = zips.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)
    return zips


def get_zip_code_states(zip_codes=None):
    if zip_codes is None:
        zip_codes = load_brazil_zip_codes()
    zip_code_states = (
        zip_codes[['admin_code1', 'admin name1']]
        .drop_duplicates()
        .sort_values(by='admin name1')
        .reset_index(drop=True)
    )
    zip_code_states.columns = ['2D', 'State']
    return zip_code_states


def get_states_encoding_table():
    municipality_states = get_municipality_states()
    zip_code_states = get_zip_code_states()
    states = pd.merge(municipality_states, zip_code_states, on='State')
    states = states[['2D', '2A', 'Stãte', 'State']]
    return states


def load_brazil_zip_codes_compl_v1():

    def zip_range_extract(x):
        return x[1:10], x[11:-1]

    # Raw loading
    zips = pd.read_csv(
        f'../data/datasets_BR/br-city-codes.csv',
        dtype=object
    )

    # Feature engineering
    zips.columns.name = 'zip_codes_compl'

    # Drop constant columns
    zips = zips.drop(
        columns=['creation', 'extinction', 'notes']
    )
    # zips = zips.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)
    zip_ranges = zips.postalCode_ranges
    zips.postalCode_ranges = zip_ranges.apply(zip_range_extract)
    return zips


def load_brazil_zip_codes_compl():
    # Load the CSV file with the Brazilian zip code information
    zips = pd.read_csv(
        f'../data/datasets_BR/br-city-codes.csv',
        dtype=object
    )

    # Replace the separator "] [" with "☗"
    zips['postalCode_ranges'] = (
        zips['postalCode_ranges']
        .str.replace(r'] [', '☗', regex=False)
    )

    # Remove the leading "[" and trailing "]"
    zips['postalCode_ranges'] = zips['postalCode_ranges'].str[1:-1]

    # Split the postalCode_ranges column into a list of ranges
    zips['postalCode_ranges'] = zips['postalCode_ranges'].str.split('☗')

    # Explode the postalCode_ranges column
    # to duplicate rows with multiple ranges
    zips = zips.explode('postalCode_ranges')

    # Remove excess characters from the postal code ranges
    def extract_codes(x):
        return x[:9], x[10:]

    zips['postalCode_ranges'] = zips['postalCode_ranges'].apply(extract_codes)

    # Remove empty columns
    zips = zips.drop(columns=['creation', 'extinction', 'notes'])
    zips.columns.name = 'zip_codes_compl'

    return zips


def get_zip_ranges():

    def expand_zip_range(s):
        return (
            s.apply(lambda x: x[0][:5]).rename('range_from'),
            s.apply(lambda x: x[1][:5]).rename('range_to')
        )

    zip_compls = load_brazil_zip_codes_compl()

    zip_ranges = pd.concat(
        list(expand_zip_range(zip_compls.postalCode_ranges))
        + [zip_compls.name, zip_compls.state],
        axis=1
    )

    # Sort the dataframe according to the range_from column
    zip_ranges = zip_ranges.sort_values(by='range_from')
    return zip_ranges


def get_zip_ranges_holes_table(ranges=None):
    """Lists 'holes' in the list of postcode ranges.
    """
    if ranges is None:
        ranges = get_zip_ranges()

    ranges['gap_before'] = (
        ranges.range_from.astype(float)
        - ranges.range_to.astype(float).shift(1) - 1
    )

    ranges['gb_from'] = ranges['gb_to'] = np.nan

    has_gap_before = ranges.gap_before > 0

    preceded_by_gap = ranges[has_gap_before]

    ranges.loc[has_gap_before, 'gb_from'] = (
        preceded_by_gap.range_from.astype(float)
        - preceded_by_gap.gap_before
    ).map(lambda x: f'{int(x):05d}')

    ranges.loc[has_gap_before, 'gb_to'] = (
        preceded_by_gap.range_from.astype(float)
        - 1
    ).map(lambda x: f'{int(x):05d}')

    ranges['gap_after'] = (
        ranges.range_from.shift(-1).astype(float)
        - ranges.range_to.astype(float) - 1
    )

    ranges['ga_from'] = ranges['ga_to'] = np.nan

    has_gap_after = ranges.gap_after > 0

    followed_by_gap = ranges[has_gap_after]

    ranges.loc[has_gap_after, 'ga_from'] = (
        followed_by_gap.range_to.astype(float)
        + 1
    ).map(lambda x: f'{int(x):05d}')

    ranges.loc[has_gap_after, 'ga_to'] = (
        followed_by_gap.range_to.astype(float)
        + followed_by_gap.gap_after
    ).map(lambda x: f'{int(x):05d}')

    gap_ranges_before = ranges[ranges.gap_before > 0]
    gap_ranges_before = gap_ranges_before[[
        'gb_from', 'gb_to', 'name', 'state', 'range_from', 'range_to'
    ]]
    gap_ranges_before.columns = [
        'from', 'to', 'after_name', 'after_state', 'after_from', 'after_to'
    ]

    gap_ranges_after = ranges[ranges.gap_after > 0]
    gap_ranges_after = gap_ranges_after[[
        'ga_from', 'ga_to', 'name', 'state', 'range_from', 'range_to'
    ]]
    gap_ranges_after.columns = [
        'from', 'to', 'before_name', 'before_state', 'before_from', 'before_to'
    ]

    gap_merged = pd.merge(
        gap_ranges_after,
        gap_ranges_before,
        on=['from', 'to']
    )

    states_equal = gap_merged.before_state == gap_merged.after_state

    gap_merged_short = gap_merged[['from', 'to']].copy()
    gap_merged_short['state'] = gap_merged.before_state.where(
        states_equal,
        gap_merged.before_state + '_' + gap_merged.after_state
    )
    gap_merged_short['before_name'] = gap_merged.before_name
    gap_merged_short['after_name'] = gap_merged.after_name

    return gap_merged_short


def get_zip_gap_ranges(gap_merged_short=None):
    """hack to use get_names on unsolved cases.
    """
    if gap_merged_short is None:
        gap_merged_short = get_zip_ranges_holes_table()

    gap_ranges = pd.concat(
        [
            gap_merged_short[['from', 'to']],
            gap_merged_short[
                ['state', 'before_name', 'after_name']
            ].apply(tuple, axis=1).rename('name')
        ], axis=1
    )

    gap_ranges.columns = ['range_from', 'range_to', 'name']

    return gap_ranges


""" Geolocations cleaning
"""


def get_zcs_reduction(data: pd.DataFrame) -> pd.DataFrame:
    """Returns a reduced version of the given zip-codes-states data table,
    containing only the columns 'zip_code_prefix', 'city', and 'state', with
    duplicates removed.

    Args:
        data (pd.DataFrame): The zip-codes-states data table to reduce.

    Returns:
        pd.DataFrame: A reduced version of the input data table, with
        duplicates removed.
    """
    return (
        data[['zip_code_prefix', 'city', 'state']]
        .drop_duplicates()
        .reset_index(drop=True)
    )


def get_name(zip_code, zip_ranges_table):
    # Use binary search to find the row that corresponds to zip_code
    left = 0
    right = len(zip_ranges_table) - 1
    while left <= right:
        mid = (left + right) // 2
        if zip_code < zip_ranges_table.iloc[mid]['range_from']:
            right = mid - 1
        elif zip_code > zip_ranges_table.iloc[mid]['range_to']:
            left = mid + 1
        else:
            # If zip_code is within range_from and range_to,
            # return the associated name
            return zip_ranges_table.iloc[mid]['name']
    # If no row is found, return None
    return None


def get_names_v1(zip_codes, zip_ranges_table):
    """Bad perf version"""
    return zip_codes.apply(
        get_name,
        zip_ranges_table=zip_ranges_table
    )


def resolve_names(zip_codes: pd.Series, ranges: pd.DataFrame) -> pd.Series:
    """Resolves city names for the given zip codes using the given zip ranges.

    Args:
        zip_codes (pd.Series): The zip codes to resolve city names for.
        ranges (pd.DataFrame): The zip ranges data frame containing at least
            'range_from', 'range_to', and 'name' columns.

    Returns:
        pd.Series: The city names corresponding to the given zip codes.
    """
    # Sort zip_codes in ascending order
    zip_codes = zip_codes.sort_values()

    # Get the sizes of the ranges table
    # and the range_from, range_to, and name columns
    m = len(ranges)
    range_from = list(ranges.range_from)
    range_to = list(ranges.range_to)
    range_name = list(ranges.name)

    # Initialize the result with an empty series
    names = []

    # Initialize the range index to the first row of the ranges table
    j = 0

    # For each zip code in the zip_codes table
    for zip_code in list(zip_codes):
        # While the range index is less than the length of the ranges table
        # and the zip code is greater than the end of the range
        while j < m and zip_code > range_to[j]:
            # Move to the next range
            j += 1

        # If we found a range containing the zip code
        if j < m and zip_code >= range_from[j]:
            # Add the name of the range to the result
            names += [range_name[j]]
        else:
            # If no range was found, add a NaN value to the result
            names += [np.nan]

    # Return the series of names with the same index as the input zip_codes
    names = pd.Series(names, index=zip_codes.index, name='names', dtype=object)

    # Sort the names by index
    names = names.sort_index()
    return names


def normalize_city_names_v1(raw_zcs_table: pd.DataFrame) -> pd.DataFrame:
    """Returns the raw zip-code-city-state table with normalized city names.

    Args:
        raw_zcs_table (pd.DataFrame): The raw zip-code-city-state table.

    Returns:
        pd.DataFrame: The raw zip-code-city-state table with normalized
        city names.
    """
    # Reduce the table to only the zip_code_prefix, city and state columns
    raw_zcs_part = get_zcs_reduction(raw_zcs_table)

    # Load the zip ranges table
    zip_ranges_table = get_zip_ranges()

    # Add the municipality column with resolved names and NaN otherwise
    raw_zcs_part['municipality'] = resolve_names(
        raw_zcs_part.zip_code_prefix,
        zip_ranges_table
    )

    # Process unresolved rows
    unresolved_rows = raw_zcs_part.municipality.isna()
    if unresolved_rows.sum() > 0:
        # Load the zip gap ranges table
        gap_ranges = get_zip_gap_ranges()

        # Get unresolved rows
        unresolved_raw_zcs_part = raw_zcs_part[unresolved_rows]

        # Remove the municipality column and add a new one
        unresolved_raw_zcs_part = unresolved_raw_zcs_part.drop(
            columns='municipality'
        )
        unresolved_raw_zcs_part['municipality'] = resolve_names(
            unresolved_raw_zcs_part.zip_code_prefix,
            gap_ranges
        )
        raw_zcs_part.loc[unresolved_rows, 'municipality'] = (
            unresolved_raw_zcs_part.city.str.title()
            + '@'
            + unresolved_raw_zcs_part.municipality.apply(
                lambda x: '[' + x[1] + '(' + x[0] + ')' + x[2] + ']'
            )
        )

    # Intégration des résultats : merge pour compléter raw_zcs_table avec la
    # colonne municipality
    raw_zcs_table_normalized = pd.merge(
        raw_zcs_table,
        raw_zcs_part,
        on=['zip_code_prefix', 'city', 'state']
    )

    raw_zcs_table_normalized.city = raw_zcs_table_normalized.municipality
    raw_zcs_table_normalized = raw_zcs_table_normalized.drop(
        columns='municipality'
    )
    raw_zcs_table_normalized.index.name = raw_zcs_table.index.name

    return raw_zcs_table_normalized


def assert_data_schema(data: pd.DataFrame, schema_cols: List[str]) -> None:
    """Checks that the given data table has the expected columns.

    Args:
        data (pd.DataFrame): The data table to check.
        schema_cols (List[str]): The expected columns of the data table.

    Raises:
        ValueError: If the data table does not have the expected columns.
    """
    real_cols = list(data.columns)
    data_name = data.columns.name
    ok = True
    for col in schema_cols:
        ok = ok and (col in real_cols)
    if not ok:
        raise ValueError(f'{data_name} must have columns {str(schema_cols)}')


def resolve_names_outof_zip_range(
    unresolved_zcs: pd.DataFrame,
    zip_gap_ranges: Optional[pd.DataFrame] = None
) -> None:
    """Resolves city names for zip codes that fall in a gap between zip ranges.

    Args:
        unresolved_zcs (pd.DataFrame): The data frame containing unresolved
            city names and their corresponding zip codes, states, and zip
            gap ranges. This data frame should have at least 'zip_code_prefix',
            'city', and 'state' columns.
        zip_gap_ranges (pd.DataFrame, optional): The zip gap ranges table to
            use for resolving city names. This table should contain at least
            'range_from', 'range_to', and 'name' columns. If not provided, the
            `get_zip_gap_ranges` function will be called to retrieve it.

    Returns:
        pd.DataFrame: The data frame with the city names resolved and
            formatted as
            'CITY@[CITY_BEFORE(STATE_BEFORE[_STATE_AFTER])CITY_AFTER]'.
    """
    # Load the zip gap ranges table if it is not provided
    if zip_gap_ranges is None:
        zip_gap_ranges = get_zip_gap_ranges()

    # Assure that the zip_ranges DataFrame has the expected columns
    assert_data_schema(zip_gap_ranges, ['range_from', 'range_to', 'name'])

    # Add the 'municipality' column with names from the gap ranges table
    resolved_names = resolve_names(
        unresolved_zcs.zip_code_prefix,
        zip_gap_ranges
    )

    # Format the unresolved city names as
    # 'CITY@[CITY_BEFORE(STATE_BEFORE[_STATE_AFTER])CITY_AFTER]'
    return (
        unresolved_zcs.city.str.title()
        + '@'
        + resolved_names.apply(
            lambda x: '[' + x[1] + '(' + x[0] + ')' + x[2] + ']'
        )
    )


def normalize_city_names(
    zcs_data: pd.DataFrame,
    zip_ranges: Optional[pd.DataFrame] = None,
    zip_gap_ranges: Optional[pd.DataFrame] = None,
) -> None:
    """Normalizes city names in the given zip-city-state data table.

    Args:
        zcs_data (pd.DataFrame): The zip-city-state data table to
            normalize.
        zip_ranges (pd.DataFrame, optional): The zip ranges table to use
            for resolving city names. This table should contain at least
            'range_from', 'range_to', and 'name' columns. If not provided, the
            `get_zip_ranges_table` function will be called to retrieve it.
        zip_gap_ranges (pd.DataFrame, optional): The zip gap ranges table to
            use for resolving city names for zip codes that fall in a gap
            between zip ranges. This table should contain at least
            'range_from', 'range_to', and 'name' columns. If not provided, the
            `get_zip_gap_ranges` function will be called to retrieve it.

    Returns:
        None: This function modifies the `zcs_data` DataFrame inplace.
    """
    # Assure that the zcs_data DataFrame has the expected columns
    assert_data_schema(zcs_data, ['zip_code_prefix', 'city', 'state'])

    # Reduce the raw_zcs_table to the zip_code_prefix, city, and state columns
    # and remove duplicates
    zcs = get_zcs_reduction(zcs_data)

    # Load the zip ranges table if it is not provided
    if zip_ranges is None:
        zip_ranges = get_zip_ranges()

    # Assure that the zip_ranges DataFrame has the expected columns
    assert_data_schema(zip_ranges, ['range_from', 'range_to', 'name'])

    # Add the municipality column with resolved names and `NA` for unresolved
    # names
    zcs['new_city'] = resolve_names(
        zcs.zip_code_prefix,
        zip_ranges
    )

    # Process unresolved rows
    unresolved_rows = zcs.new_city.isna()

    # If there are unresolved rows
    if unresolved_rows.sum() > 0:
        # Format the unresolved city names as
        # 'CITY@[CITY_BEFORE(STATE_BEFORE[STATE_AFTER])CITY_AFTER]'
        zcs.loc[unresolved_rows, 'new_city'] = (
            resolve_names_outof_zip_range(
                zcs[unresolved_rows],
                zip_gap_ranges
            )
        )

    # Save the pk-index (primary key index) by resetting the index
    zcs_data.reset_index(inplace=True)

    # Create a positional index by resetting the index again
    zcs_data.reset_index(inplace=True)

    # Merge the normalized city names into the original `zcs_data` dataframe
    new_zcs_data = pd.merge(
        zcs_data,
        zcs,
        on=['zip_code_prefix', 'city', 'state']
    )

    # Sort `new_zcs_data` by the positional index
    # to align with the original order of `zcs_data`
    new_zcs_data.sort_values(by='index', inplace=True)

    # Reset the original index of `zcs_data`
    zcs_data.set_index(zcs_data.columns[1], inplace=True)

    # Drop the positional index column
    zcs_data.drop(columns='index', inplace=True)

    # Overwrite the city column in `zcs_data` with the normalized city names
    zcs_data.city = new_zcs_data.new_city.values


""" Entities and Relationships analysis
"""


def count_of_objets_A_by_objet_B(
    table: pd.DataFrame,
    col_A: str, col_B: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Counts the number and frequency of values of column `col_A`
    in `table` by values of column `col_B`.

    Parameters:
    - table (pd.DataFrame): input data.
    - col_A (str): name of the column to count values in `table`.
    - col_B (str): name of the column to group by.

    Returns:
    - count_freq (pd.DataFrame):
        count and frequency of `col_A` values by `col_B` values.
    - gpby (pd.DataFrame):
        result of the grouping.
    """
    gpby = table[[col_A, col_B]].groupby(by=col_B).count()
    count = gpby[col_A].value_counts().rename('count')
    freq = gpby[col_A].value_counts(normalize=True).rename('freq')
    count_freq = pd.concat([count, freq], axis=1)
    count_freq.index.name = f'{col_A} by {col_B}'
    count_freq['count'] = count_freq['count'].astype(int)
    return count_freq, gpby


def out_of_intersection(
   table_A: pd.DataFrame,
   table_B: pd.DataFrame, pk_name: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find the primary keys that are in one table but not the other.

    Parameters:
    - table_A (pd.DataFrame): First table.
    - table_B (pd.DataFrame): Second table.
    - pk_name (str): Name of the primary key column.

    Returns:
    - pk_A (np.ndarray): Primary keys of `table_A`.
    - pk_B (np.ndarray): Primary keys of `table_B`.
    - pk_A_not_B (np.ndarray):
        Primary keys of `table_A` that are not in `table_B`.
    - pk_B_not_A (np.ndarray):
        Primary keys of `table_B` that are not in `table_A`.
    """
    pk_A = table_A[pk_name].unique()
    pk_B = table_B[pk_name].unique()
    pk_A_not_B = np.array(list(set(pk_A) - set(pk_B)))
    pk_B_not_A = np.array(list(set(pk_B) - set(pk_A)))
    return pk_A, pk_B, pk_A_not_B, pk_B_not_A


def print_out_of_intersection(
    table_A: pd.DataFrame,
    table_B: pd.DataFrame,
    pk_name: str
) -> None:
    """Print the number and percentage of primary keys
    that are in one table but not the other.

    Parameters:
    - table_A (pd.DataFrame): First table.
    - table_B (pd.DataFrame): Second table.
    - pk_name (str): Name of the primary key column.
    """
    (
        pk_A, pk_B, pk_A_not_B, pk_B_not_A
    ) = out_of_intersection(table_A, table_B, pk_name)
    name_A = table_A.columns.name
    name_B = table_B.columns.name
    print(f'|{name_A}.{pk_name}| :', len(pk_A))
    print(f'|{name_B}.{pk_name}| :', len(pk_B))
    print(
        f'|{name_A}.{pk_name} \\ {name_B}.{pk_name}| :',
        len(pk_A_not_B),
        '(' + str(round(100 * len(pk_A_not_B) / len(pk_A), 3)) + '%)'
    )
    print(
        f'|{name_B}.{pk_name} \\ {name_A}.{pk_name}| :',
        len(pk_B_not_A),
        '(' + str(round(100 * len(pk_B_not_A) / len(pk_B), 3)) + '%)'
    )


def display_relation_arities(
    table_A: pd.DataFrame, pk_A: str,
    table_B: pd.DataFrame, fk_B: str,
    verbose: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, float, float, float, float]:
    """Compute and display statistics about the relation between two tables.

    Parameters:
    - table_A (pd.DataFrame): First table.
    - table_B (pd.DataFrame): Second table.
    - pk_A (str): Primary key column in `table_A`.
    - fk_B (str): Foreign key column in `table_B`.
    - verbose (bool, optional):
        Whether to print the statistics (default is False).
    """
    ab, _ = count_of_objets_A_by_objet_B(table_A, pk_A, fk_B)
    ba, _ = count_of_objets_A_by_objet_B(table_A, fk_B, pk_A)

    name_A = table_A.columns.name
    name_B = table_B.columns.name

    _, _, pk_A_not_B, pk_B_not_A = out_of_intersection(table_A, table_B, fk_B)

    ab_min = 0 if len(pk_B_not_A) > 0 else ab.index.min()
    ab_max = ab.index.max()
    ba_min = 0 if len(pk_A_not_B) > 0 else ba.index.min()
    ba_max = ba.index.max()

    print(
        'relation arities : '
        f'[{name_A}]({ab_min}..{ab_max})'
        f'--({ba_min}..{ba_max})[{name_B}]'
    )

    ab = ab.T
    ab.insert(0, 'sum', ab.T.sum())

    ba = ba.T
    ba.insert(0, 'sum', ba.T.sum())

    if verbose:
        display(ab)
        display(ba)

    return ab, ba, ab_min, ab_max, ba_min, ba_max


def get_centers(
    crfm: pd.DataFrame,
    cl_labels: List[int]
) -> Tuple[List[float], List[float], List[float]]:
    # def get_centers(crfm, cl_labels):
    """Compute the centers of the clusters in the RFM space.

    Parameters:
    - crfm (pd.DataFrame): RFM data.
    - cl_labels (List[int]): Cluster labels for each observation in `crfm`.

    Returns:
    - r_centers (List[float]): R-coordinates of the cluster centers.
    - f_centers (List[float]): F-coordinates of the cluster centers.
    - m_centers (List[float]): M-coordinates of the cluster centers.
    """
    crfm_cl_labeled = pd.concat([
        pd.Series(cl_labels, index=crfm.index, name='k_cl'),
        crfm
    ], axis=1)
    cl_means = crfm_cl_labeled.groupby(by='k_cl').mean()
    cl_centers = cl_means.values
    return (
        cl_centers[:, 0],
        cl_centers[:, 1],
        cl_centers[:, 2],
    )


def get_centers_v2(crfm, cl_labels):
    """Compute the centers of the clusters in the RFM space.

    Parameters:
    - crfm (pd.DataFrame): RFM data.
    - cl_labels (List[int]): Cluster labels for each observation in `crfm`.
    """
    crfm_cl_labeled = pd.concat([
        pd.Series(cl_labels, index=crfm.index, name='k_cl'),
        crfm
    ], axis=1)
    cl_means = crfm_cl_labeled.groupby(by='k_cl').mean()
    return cl_means.values


def plot_kmeans_rfm_clusters_and_classes(
    crfm: pd.DataFrame,
    rfm_labels: (Tuple[str, str, str]),
    clu_labels: List[int],
    cla_labels: List[int]
) -> Tuple[List[float], List[float], List[float]]:
    # def plot_kmeans_rfm_clusters_and_classes(
    #    crfm, rfm_labels,
    #    clu_labels, cla_labels
    # ):
    """Plot the RFM space with the K-Means clusters and the abstracted classes.

    Parameters:
    - crfm (pd.DataFrame): RFM data.
    - rfm_labels (Tuple[str, str, str]): Labels for the R, F, and M dimensions.
    - clu_labels (List[int]): Cluster labels for each observation in `crfm`.
    - cla_labels (List[int]): Class labels for each observation in `crfm`.
    """
    n_cl = clu_labels.nunique()   # len(np.unique(clu_labels))
    r, f, m = crfm.R, crfm.F, crfm.M
    r_label, m_label = rfm_labels[0], rfm_labels[2]
    clu_r_centers, _, clu_m_centers = get_centers(crfm, clu_labels)
    cla_r_centers, _, cla_m_centers = get_centers(crfm, cla_labels)

    fig = plt.figure(figsize=(10, 5))

    ax1 = plt.subplot2grid((1, 2), (0, 0))
    ax2 = plt.subplot2grid((1, 2), (0, 1))

    ax1.semilogy()
    plot_clusters_2d(
        ax1, f'RM ({n_cl} clusters)',
        xy=[r, m], xy_labels=[r_label, m_label],
        xy_clu_centers=[clu_r_centers, clu_m_centers],
        clu_labels=clu_labels
    )
    ax2.semilogy()
    plot_clusters_2d(
        ax2, f'RM ({n_cl} classes)',
        xy=[r, m], xy_labels=[r_label, m_label],
        xy_clu_centers=[cla_r_centers, cla_m_centers],
        clu_labels=cla_labels
    )

    plt.tight_layout()

    plt.suptitle(
        f'{n_cl}-Means clusters and abstracted classes',
        fontsize=14,
        fontweight='bold',
        y=1.05,
    )
    plt.show()


def select_k_with_anova(
    crfm,
    k_min=2, k_max=20,
    metric='inertia',
    verbose=False
):
    """
    Select the optimal number of clusters using ANOVA and the elbow method.
    Parameters
    ----------
    crfm: Pandas DataFrame
        DataFrame with columns 'R', 'F', and 'M' representing
        recency, frequency, and monetary value of customer orders.
    k_min: int, optional (default=2)
        Minimum number of clusters to test.
    k_max: int, optional (default=20)
        Maximum number of clusters to test.
    metric: str, optional (default='inertia')
        Metric to use for the ANOVA. Can be either 'inertia' or 'silhouette'.
    verbose: bool, optional (default=True)
        Whether to print the time taken to fit and predict
        with the KMeans model for each value of k.
    Returns
    -------
    k: int
        Optimal number of clusters.
    """
    # Normalize the data
    scaler = StandardScaler()
    crfm_scaled = scaler.fit_transform(crfm)

    # Create a list of k values to test
    k_values = list(range(k_min, k_max+1))

    # Initialize lists to store the scores
    anova_scores = []

    # Loop over k values
    for k in k_values:
        # Create a KMeans model with k clusters
        kmeans, clu_labels, _, _, _, km_t = kmeans_clustering(crfm, k)
        if verbose:
            print(
                f'Time for kmeans_clustering with k={k} :',
                round(km_t, 3), 's'
            )

        # Calculate the anova score for the current model
        if metric == 'inertia':
            score = kmeans.inertia_
        elif metric == 'silhouette':
            score = silhouette_score(crfm_scaled, clu_labels)
        else:
            raise ValueError('Invalid metric')

        # Append the score to the list
        anova_scores.append(score)

    # Plot the scores
    plt.plot(k_values, anova_scores)
    plt.xlabel('Number of clusters')
    plt.xticks(k_values)
    if metric == 'inertia':
        plt.ylabel('Inertia')
    elif metric == 'silhouette':
        plt.ylabel('Silhouette Score')
    plt.title('ANOVA with Elbow Method')
    plt.show()

    # Select the optimal k using the elbow method
    # TODO : find a formal way to identify the inflection point
    """if metric == 'inertia':
        k = k_values[anova_scores.index(min(anova_scores))]
    else :
        k = k_values[anova_scores.index(max(anova_scores))]
    return k"""


def select_k_with_davies_bouldin(X, k_min=2, k_max=20):
    """Calculate the Davies-Bouldin index for each value of k
    and plot the results as a bar chart.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data to cluster.
    k_min, k_max : int
        The minimum and maximum number of clusters to consider.
    """
    # Create a list of k values to test
    k_values = list(range(k_min, k_max+1))

    scores = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        scores.append(davies_bouldin_score(X, kmeans.labels_))

    # Sort the scores in descending order
    scores_sorted = sorted(scores)

    # Plot the bar chart
    plt.bar(k_values, scores)
    plt.xlabel("Number of clusters")
    plt.ylabel("Davies-Bouldin index")
    plt.xticks(k_values)
    plt.ylim(bottom=.5)

    # Mark the three best values of k with red bars
    for i in range(3):
        best_k = scores.index(scores_sorted[i]) + k_min
        plt.bar(best_k, scores[best_k-k_min], color='green')

    plt.show()
