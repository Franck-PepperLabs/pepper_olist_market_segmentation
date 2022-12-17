from sys import getsizeof
import time
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from IPython.display import display
from pepper_commons import format_iB, bold, print_subtitle


def load_table(name):
    return pd.read_csv(f'../data/olist_{name}_dataset.csv')


def load_product_category_name_translation():
    return pd.read_csv(f'../data/product_category_name_translation.csv')


def table_content_analysis(data):
    print_subtitle('basic infos')
    print(bold('dimensions'), ':', data.shape)
    print(bold('size'), ':', *format_iB(getsizeof(data)))
    print(bold('info'), ':')
    data.info()
    print(bold('stats'), ':')
    display(data.describe(include='all').T)
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
    return _order_payments.payment_type.unique()


# TODO : Pascal : il y un param indicator qui ajoute
# une colonne de méta donnée sur la nature du join
def get_merged_data():
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
    return customers[customers.customer_unique_id == cu_id]


def customer_states(customer):
    return customer.customer_state.unique()


def customer_cities(customer, state):
    return customer[
        customer.customer_state == state
    ].customer_city.unique()


def customer_zips(customer, state, city):
    return customer[
        (customer.customer_state == state) &
        (customer.customer_city == city)
    ].customer_zip_code_prefix.unique()


def customer_ids(customer, state, city, zip_code):
    return tuple(
        customer[
            (customer.customer_state == state) &
            (customer.customer_city == city) &
            (customer.customer_zip_code_prefix == zip_code)
        ].customer_id
    )


def customer_locations(customer):
    return {
        state: {
            city: {
                zip_code: customer_ids(customer, state, city, zip_code)
                for zip_code in customer_zips(customer, state, city)
            } for city in customer_cities(customer, state)
        } for state in customer_states(customer)
    }


def test_customer_locations():
    customers = get_customers()
    cu_id = 'fe59d5878cd80080edbd29b5a0a4e1cf'
    c = customer(customers, cu_id)
    c_locations = customer_locations(c)
    display(c_locations)


def get_unique_customers():
    return pd.DataFrame(
        get_customers()
        .groupby(by=['customer_unique_id'], group_keys=True)
        .apply(customer_locations),
        columns=['locations']
    )


def get_aggregated_order_payments():
    op = get_order_payments()
    op = op.sort_values(
        by=['order_id', 'payment_sequential']
    )
    op_gpby = (
        op
        .groupby(by='order_id')
        .aggregate(tuple)
    )
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
    return get_orders().order_purchase_timestamp.astype('datetime64[ns]').max()


def get_first_order_date():
    return get_orders().order_purchase_timestamp.astype('datetime64[ns]').min()


def get_order_ages(now):
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
    return pd.Index(list(
        set(get_orders().order_id.unique())
        - set(get_order_payments().order_id.unique())
    ))


def get_without_physical_features_product_ids():
    products = get_products()
    bindex = products.product_weight_g.isna()
    without_physical_features_products = products[bindex]
    return pd.Index(list(without_physical_features_products.product_id))


def get_without_marketing_features_product_ids():
    products = get_products()
    bindex = products.product_category_name.isna()
    without_marketing_features_products = products[bindex]
    return pd.Index(list(without_marketing_features_products.product_id))


def get_unknown_product_ids():
    return (
        get_without_physical_features_product_ids()
        .intersection(get_without_marketing_features_product_ids())
    )


def plot_clusters_2d_v1(x, y, title, xlabel, ylabel, clu_labels):
    plt.scatter(x, y, c=clu_labels)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plot_clusters_2d(ax, title, xy, xy_labels, xy_clu_centers, clu_labels):
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
    n_clusters = len(np.unique(clu_labels))
    colors = cm.nipy_spectral(clu_labels.astype(float) / n_clusters)
    ax.scatter(xyz[0], xyz[1], xyz[2], c=colors)
    ax.set_xlabel(xyz_labels[0])
    ax.set_ylabel(xyz_labels[1])
    ax.set_zlabel(xyz_labels[2])
    ax.set_title(title)


def plot_silhouette(ax, silhouette_avg, silhouette_values, clu_labels):
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

def select_region(geolocation, lng_limits, lat_limits):
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
    # get the HTML content
    url = 'https://en.wikipedia.org/wiki/Federative_units_of_Brazil#List'
    response = requests.get(url)
    # print(response.status_code)

    # parse data from the html into a beautifulsoup object
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'class': 'wikitable sortable'})

    # turn it into dataframe
    return pd.read_html(str(table))[0]


def count_of_objets_A_by_objet_B(table, col_A, col_B):
    gpby = table[[col_A, col_B]].groupby(by=col_B).count()
    count = gpby[col_A].value_counts().rename('count')
    freq = gpby[col_A].value_counts(normalize=True).rename('freq')
    count_freq = pd.concat([count, freq], axis=1)
    count_freq.index.name = f'{col_A} by {col_B}'
    count_freq['count'] = count_freq['count'].astype(int)
    return count_freq, gpby


def out_of_intersection(table_A, table_B, pk_name):
    pk_A = table_A[pk_name].unique()
    pk_B = table_B[pk_name].unique()
    pk_A_not_B = np.array(list(set(pk_A) - set(pk_B)))
    pk_B_not_A = np.array(list(set(pk_B) - set(pk_A)))
    return pk_A, pk_B, pk_A_not_B, pk_B_not_A


def print_out_of_intersection(table_A, table_B, pk_name):
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


def display_relation_arities(table_A, table_B, pk_A, fk_B, verbose=False):
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
