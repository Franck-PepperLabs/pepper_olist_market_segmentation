olist_tables = [
    'customers',
    'geolocation',
    'order_items',
    'order_payments',
    'order_reviews',
    'orders',
    'products',
    'sellers',
]

import pandas as pd
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

