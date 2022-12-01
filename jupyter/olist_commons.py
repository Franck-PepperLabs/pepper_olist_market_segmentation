import pandas as pd

"""olist_tables = [
    'customers',
    'geolocation',
    'order_items',
    'order_payments',
    'order_reviews',
    'orders',
    'products',
    'sellers',
]"""

def load_table(name):
    return pd.read_csv(f'../data/olist_{name}_dataset.csv')

def load_product_category_name_translation():
    return pd.read_csv(f'../data/product_category_name_translation.csv')

from pepper_commons import *
from sys import getsizeof
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



# as the tables are small we preload them
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


import pandas as pd
def get_merged_data():
    m = get_order_items()
    m = pd.merge(m, get_orders(), how='outer', on='order_id')
    m = pd.merge(m, get_products(), how='outer', on='product_id')
    m = pd.merge(m, get_sellers(), how='outer', on='seller_id')
    m = pd.merge(m, get_customers(), how='outer', on='customer_id')
    m = pd.merge(m, get_order_payments(), how='outer', on='order_id')    
    m = pd.merge(m, get_order_reviews(), how='outer', on='order_id')    
    return m


""" Unique customers """

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
        state:{
            city:{
                zip_code:customer_ids(customer, state, city, zip_code)
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
    op_gpby = (op
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
    orders = get_orders()[['order_id', 'customer_id', 'order_purchase_timestamp']]
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

def get_customer_order_freqs_and_amount(
    from_date=get_first_order_date(),
    to_date = get_last_order_date()
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
