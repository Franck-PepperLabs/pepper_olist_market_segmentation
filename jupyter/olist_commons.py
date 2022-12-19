"""
This module contains functions to load and analyze data tables
for a e-commerce dataset.
"""

from typing import List, Tuple
from sys import getsizeof
import time
import requests
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


def load_table(name):
    """
    Load a data table from a file.

    Parameters:
    name (str): The name of the table to load.

    Returns:
    pandas.DataFrame: The loaded data table.
    """
    return pd.read_csv(f'../data/olist_{name}_dataset.csv')


def load_product_category_name_translation():
    return pd.read_csv(f'../data/product_category_name_translation.csv')


def table_content_analysis(data):
    """
    Perform a content analysis of a data table.

    Parameters:
    data (pandas.DataFrame): The data table to analyze.

    Returns:
    None
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
_customers = load_table('customers').astype(object)
_geolocation = load_table('geolocation').astype(object)
_order_items = load_table('order_items').astype(object)
_order_payments = load_table('order_payments').astype(object)
_order_reviews = load_table('order_reviews').astype(object)
_orders = load_table('orders').astype(object)
_products = load_table('products').astype(object)
_sellers = load_table('sellers').astype(object)
_cats = load_product_category_name_translation()


def get_customers():
    """
    Get the customers data table.

    Returns:
    pandas.DataFrame: The customers data table.
    """
    return _customers.copy()


def get_geolocation():
    return _geolocation.copy()


def get_order_items():
    return _order_items.copy()


def get_order_payments():
    return _order_payments.copy()


def get_order_reviews():
    return _order_reviews.copy()


def get_orders():
    return _orders.copy()


def get_products():
    return _products.copy()


def get_sellers():
    return _sellers.copy()


def get_cats():
    return _cats.copy()


def get_payment_types():
    """
    Get the unique payment types in the order payments data table.

    Returns:
    numpy.ndarray: The unique payment types.
    """
    return _order_payments.payment_type.unique()


def get_merged_data():
    """
    Merge several data tables to create a comprehensive dataset.

    Returns:
    pandas.DataFrame: The merged dataset.
    """
    m = get_order_items()
    m = pd.merge(m, get_orders(), how='outer', on='order_id')
    m = pd.merge(m, get_products(), how='outer', on='product_id')
    m = pd.merge(m, get_sellers(), how='outer', on='seller_id')
    m = pd.merge(m, get_customers(), how='outer', on='customer_id')
    m = pd.merge(m, get_order_payments(), how='outer', on='order_id')
    m = pd.merge(m, get_order_reviews(), how='outer', on='order_id')
    return m


""" Unique customers
"""


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
    Get the unique states for a customer.

    Parameters:
    customer (pandas.DataFrame): The customer data.

    Returns:
    numpy.ndarray: The unique states for the customer.
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
    customers = get_customers()
    cu_id = 'fe59d5878cd80080edbd29b5a0a4e1cf'
    c = customer(customers, cu_id)
    c_locations = customer_locations(c)
    display(c_locations)


def get_unique_customers():
    """
    Get a DataFrame of unique customers and their locations.

    Returns:
    pandas.DataFrame: A DataFrame of unique customers and their locations.
    """
    return pd.DataFrame(
        get_customers()
        .groupby(by=['customer_unique_id'], group_keys=True)
        .apply(customer_locations),
        columns=['locations']
    )


def get_aggregated_order_payments():
    """
    Aggregate the order payments data table by order ID.

    Returns:
    pandas.DataFrame: The aggregated order payments data table.
    """
    # Load the order payments data table
    op = get_order_payments()

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

    return op_gpby


def get_order_times():
    """
    Get a DataFrame of order time data.

    Returns:
    pandas.DataFrame: A DataFrame of order time data.
    """
    orders = get_orders()
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


def get_customer_order_payment():
    """
    Get a DataFrame of customer, order and payment data.

    Returns:
    pandas.DataFrame: A DataFrame of customer, order and payment data.
    """
    customers = get_customers()[['customer_id', 'customer_unique_id']]
    orders = get_orders()[
        ['order_id', 'customer_id', 'order_purchase_timestamp']
    ]
    orders.order_purchase_timestamp = (
        orders.order_purchase_timestamp
        .astype('datetime64[ns]')
    )
    agg_order_payments = get_aggregated_order_payments()[['payment_total']]
    m = pd.merge(customers, orders, how='outer', on='customer_id')
    m = pd.merge(m, agg_order_payments, how='outer', on='order_id')
    m = m.sort_values(by='order_purchase_timestamp', ascending=False)
    m = m.drop(columns=['customer_id'])
    m = m.set_index('order_id')
    return m


def get_last_order_date():
    """Get the last order date.

    Returns:
        datetime: The last order date.
    """
    return get_orders().order_purchase_timestamp.astype('datetime64[ns]').max()


def get_first_order_date():
    """Get the first order date.

    Returns:
        datetime: The first order date.
    """
    return get_orders().order_purchase_timestamp.astype('datetime64[ns]').min()


def get_order_ages(now):
    """Get the ages of all orders at a given time.

    Args:
        now (datetime): The reference time for calculating the ages.

    Returns:
        Series: A Series with the order ages, indexed by order id.
    """
    return now - (
        get_orders()
        .set_index('order_id')
        .order_purchase_timestamp
        .astype('datetime64[ns]')
        .sort_values(ascending=False)
        .rename('order_age')
    )


def get_order_ages_2(
        from_date=get_first_order_date(),
        to_date=get_last_order_date()
):
    """
    Return the age of orders placed between the given
    `from_date` and `to_date` dates.
    If no dates are given, the function will use
    the first and last order dates in the orders table.
    """
    ord = get_orders()
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


def get_customer_order_recency(
        from_date=get_first_order_date(),
        to_date=get_last_order_date()
):
    cop = get_customer_order_payment()
    order_age = get_order_ages(get_last_order_date())
    copa = pd.concat([cop, order_age], axis=1)
    is_copa_between = (
        (from_date <= copa.order_purchase_timestamp)
        & (copa.order_purchase_timestamp <= to_date)
    )
    copab = copa[is_copa_between]

    customer_recency = copab.drop_duplicates(subset='customer_unique_id')
    customer_recency = customer_recency.set_index('customer_unique_id')
    return customer_recency


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
    cop = get_customer_order_payment()
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
    cor = get_customer_order_recency(from_date, to_date)
    cofa = get_customer_order_freqs_and_amount(from_date, to_date)
    crfm = pd.merge(
        cor[['order_age']], cofa,
        how='outer', on='customer_unique_id'
    )
    crfm.columns = ['R', 'F', 'M']
    oneday = pd.Timedelta(days=1)
    crfm.R = crfm.R / oneday
    return crfm


def get_unpaid_order_ids():
    """Returns a list of order ids for orders that have not been paid.

    Returns:
        pd.Index: the indices of unpaid orders.
    """
    # Calculate the set difference between the set of unique order ids
    # and the set of unique order ids that have been paid
    return pd.Index(list(
        set(get_orders().order_id.unique())
        - set(get_order_payments().order_id.unique())
    ))


def get_without_physical_features_product_ids():
    """Returns a list of product ids for products
    that do not have physical features.

    Returns:
        pd.Index: the indices of products without physical features.
    """
    products = get_products()
    # Get products where the 'product_weight_g' column is null
    bindex = products.product_weight_g.isna()
    without_physical_features_products = products[bindex]
    return pd.Index(list(without_physical_features_products.product_id))


def get_without_marketing_features_product_ids():
    """Returns a list of product ids for products
    that do not have marketing features.

    Returns:
        pd.Index: the indices of products without marketing features.
    """
    products = get_products()
    # Get products where the 'product_category_name' column is null
    bindex = products.product_category_name.isna()
    without_marketing_features_products = products[bindex]
    return pd.Index(list(without_marketing_features_products.product_id))


def get_unknown_product_ids():
    """Get the product ids for products with both missing physical features
    and missing marketing features.

    Returns:
    pd.Index: product ids of products with both missing physical features
    and missing marketing features.
    """
    return (
        get_without_physical_features_product_ids()
        .intersection(get_without_marketing_features_product_ids())
    )


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


def kmeans_clustering(crfm, k):
    """
    Perform K-Means clustering on customer RFM data.
    Parameters
    crfm: Pandas DataFrame
        DataFrame with columns 'R', 'F', and 'M' representing
        recency, frequency, and monetary value of customer orders.
    k: int
        Number of clusters to form.

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
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit_predict(crfm)
    km_t += time.time()
    clu_labels = kmeans.labels_
    clu_centers = kmeans.cluster_centers_
    rfm = r, f, m = crfm.R, crfm.F, crfm.M
    rfm_labels = r_label, f_label, m_label = 'Recency', 'Frequency', 'Monetary'
    rfm_centers = r_centers, f_centers, m_centers = (
        clu_centers[:, 0],
        clu_centers[:, 1],
        clu_centers[:, 2],
    )
    return kmeans, clu_labels, rfm, rfm_labels, rfm_centers, km_t


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


def classes_labeling(rfm, classes_def):
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
    cla_labels = classes_labeling(crfm, classes_def)
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


def get_brazil_states():
    """
    Get the dataframe containing the list of states in Brazil from Wikipedia.

    Returns:
        pd.DataFrame: The dataframe containing the list of states in Brazil.
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


def count_of_objets_A_by_objet_B(
    table: pd.DataFrame,
    col_A: str, col_B: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # def count_of_objets_A_by_objet_B(table, col_A, col_B):
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
    # def out_of_intersection(table_A, table_B, pk_name):
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
    """
    pk_A = table_A[pk_name].unique()
    pk_B = table_B[pk_name].unique()
    pk_A_not_B = np.array(list(set(pk_A) - set(pk_B)))
    pk_B_not_A = np.array(list(set(pk_B) - set(pk_A)))"""
    pk_A = table_A[pk_name].unique()
    pk_B = table_B[pk_name].unique()
    pk_A_not_B = np.setdiff1d(pk_A, pk_B)
    pk_B_not_A = np.setdiff1d(pk_B, pk_A)
    return pk_A, pk_B, pk_A_not_B, pk_B_not_A


def print_out_of_intersection(
    table_A: pd.DataFrame,
    table_B: pd.DataFrame,
    pk_name: str
) -> None:
    # def print_out_of_intersection(table_A, table_B, pk_name):
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
    table_A: pd.DataFrame,
    table_B: pd.DataFrame,
    pk_A: str, fk_B: str,
    verbose: bool = False
) -> None:
    # def display_relation_arities(table_A, table_B,
    # pk_A, fk_B, verbose=False):
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


def get_centers(
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
        Whether to print the time taken to fit and predict with the KMeans model for each value of k.
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
            print(f'Time for kmeans_clustering with k={k} :', round(km_t, 3), 's')
        
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
    """
    Calculate the Davies-Bouldin index for each value of k and plot the results as a bar chart.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data to cluster.
    k_min, k_max : int
        The minimum and maximum number of clusters to consider.

    Returns
    -------
    None
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
