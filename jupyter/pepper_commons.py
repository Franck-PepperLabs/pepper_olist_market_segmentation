# pretty printing
bold = lambda s: '\033[1m' + str(s) + '\033[0m'
italic = lambda s: '\033[3m' + str(s) + '\033[0m'
cyan = lambda s : '\033[36m' + str(s) + '\033[0m'
magenta = lambda s : '\033[35m' + str(s) + '\033[0m'
red = lambda s : '\033[31m' + str(s) + '\033[0m'
green = lambda s : '\033[32m' + str(s) + '\033[0m'

def print_title(txt):
    print(bold(magenta('\n' + txt.upper())))

def print_subtitle(txt):
    print(bold(cyan('\n' + txt)))


# memory assets
# from sys import getsizeof
KiB = 2**10
MiB = KiB * KiB
GiB = MiB * KiB
TiB = GiB * KiB
def format_iB(n_bytes):
    if n_bytes < KiB:
        return n_bytes, 'iB'
    elif n_bytes < MiB:
        return round(n_bytes / KiB, 3), 'KiB'
    elif n_bytes < GiB:
        return round(n_bytes / MiB, 3), 'MiB'
    elif n_bytes < TiB:
        return round(n_bytes / GiB, 3), 'GiB'
    else:
        return round(n_bytes / TiB), 'TiB'


import pandas as pd
def display_head_and_tail(data, n=5):
    cdots_row = data.head(1).copy().apply(lambda x: '...')  # TODO : find a better way
    cdots_row.index = ['...']
    _data = pd.concat([data.head(n), cdots_row, data.tail(n)])
    display(_data)

def value_counts_and_freqs(data, label):
    s = data[label]
    freqs = pd.DataFrame({
        'count': s.value_counts(normalize=False),
        'freq': s.value_counts(normalize=True),
    })
    freqs.index.names = [label]
    return freqs

import matplotlib.pyplot as plt
def plot_value_freqs(data, label, c='blue'):
    freq = value_counts_and_freqs(data, label).sort_values(by='count').freq
    freq.plot.barh(color=c)
    plt.show()

def discrete_stats(data, name): # TODO : add an indice based on Lorenz curve
    """[count, unique_count, na_count, filling_rate, variety_rate] as [n, n_u, n_na, fr, vr] for each var in data"""
    n = data.count()
    n_u = data.nunique()
    n_na = data.isna().sum()
    stats = pd.DataFrame({
        'n': n,
        'n_u': n_u,
        'n_na': n_na,
        'fr': n / (n + n_na),
        'vr': n_u / n,
    }, index=data.columns)
    stats.index.names = [name]
    return stats
