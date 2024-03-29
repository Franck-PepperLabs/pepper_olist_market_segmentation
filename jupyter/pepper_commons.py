import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from scipy.stats import chi2_contingency, entropy


# pretty printing
def bold(s):
    return '\033[1m' + str(s) + '\033[0m'


def italic(s):
    return '\033[3m' + str(s) + '\033[0m'


def cyan(s):
    return '\033[36m' + str(s) + '\033[0m'


def magenta(s):
    return '\033[35m' + str(s) + '\033[0m'


def red(s):
    return '\033[31m' + str(s) + '\033[0m'


def green(s):
    return '\033[32m' + str(s) + '\033[0m'


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
    """AI is creating summary for format_iB

    Args:
        n_bytes ([type]): [description]

    Returns:
        [type]: [description]
    """
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


def display_head_and_tail(data, n=5):
    # TODO : find a better way
    cdots_row = data.head(1).copy().apply(lambda x: '...')
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


def plot_value_freqs(data, label, c='blue'):
    freq = value_counts_and_freqs(data, label).sort_values(by='count').freq
    freq.plot.barh(color=c)
    plt.show()


def plot_freqs(data, label, min=0, max=1, c='blue'):
    freq = value_counts_and_freqs(data, label).sort_values(by='count').freq
    freq = freq[(min <= freq) & (freq <= max)]
    freq.plot.barh(color=c)
    plt.show()


def lorenz_curve(v):
    """Calculate the Lorenz curve for a variable.

    Args:
        v: 1D array-like containing the values of the variable.

    Returns:
        tuple: A tuple containing two 1D array-like objects representing the
            Lorenz curve. The first array-like contains the x values, and the
            second array-like contains the y values.
    """
    # Sort the data in ascending order
    v = sorted(v)

    # Calculate the cumulative sum of the data
    cum_sum = [sum(v[:i+1]) for i in range(len(v))]

    # Normalize the cumulative sum by the total sum of the data
    norm_cum_sum = [x / cum_sum[-1] for x in cum_sum]

    # Calculate the fraction of the population at each cumulative sum
    pop_fraction = [i / len(v) for i in range(len(v))]

    return (pop_fraction, norm_cum_sum)


def gini(v):
    """Calculate the Gini coefficient for a variable.

    Args:
        v: 1D array-like containing the values of the variable.

    Returns:
        float: The Gini coefficient.
    """
    # Calculate the Lorenz curve
    x, y = lorenz_curve(v)

    # Calculate the area under the Lorenz curve
    auc = sum([(x[i+1] - x[i]) * y[i] for i in range(len(x) - 1)])

    # Calculate the Gini coefficient
    gini = 1 - 2 * auc

    return gini


# TODO : gini est plutôt utilisé pour la distribution continue
# Pour la distribution catégorielle, c'est plutôt le chi2 et l'entroprie
def empirical_dist_gini(v):
    return gini(v.value_counts())


def categorical_stats(data, name):

    # PB : VisibleDeprecationWarning:
    #   Creating an ndarray from ragged nested sequences
    #   (which is a list-or-tuple of lists-or-tuples-or ndarrays
    #   with different lengths or shapes) is deprecated.
    #   If you meant to do this, you must specify 'dtype=object'
    #   when creating the ndarray. arr_value = np.asarray(value)

    """Calculate chi-squared test of independence and entropy for each
    categorical variable in data.

    Args:
        data: DataFrame containing categorical variables.
        name: String representing the name of the data.

    Returns:
        DataFrame: A DataFrame containing the chi-squared statistic, p-value,
            and entropy for each categorical variable in data.
    """
    stats = pd.DataFrame(columns=['chi2', 'p', 'entropy'], index=data.columns)
    stats.index.name = name

    for col in data.columns:
        # Calculate chi-squared statistic and p-value
        ct = pd.crosstab(
            index=data[col], columns='count'
        )   # .values.astype(object).astype(float)
        chi2, p, _, _ = chi2_contingency(ct)

        # Calculate entropy
        freq = ct / ct.sum()
        entropy_ = entropy(freq)

        stats.loc[col] = (chi2, p, entropy_)

    return stats


def discrete_stats(data, name=None):
    """[count, unique_count, na_count, filling_rate, variety_rate]
    as [n, n_u, n_na, fr, vr] for each var in data
    """
    n = data.count()
    n_u = data.nunique()
    n_na = data.isna().sum()
    stats = pd.DataFrame({
        'n': n,
        'n_u': n_u,
        'n_na': n_na,
        'Filling rate': n / (n + n_na),
        'Shannon entropy': n_u / n,
        'dtypes': data.dtypes
    }, index=data.columns)
    if name is not None:
        stats.index.names = [name]

    return stats


def plot_discrete_stats(stats, precision=.1):
    table_name = stats.index.name
    filling_rate = stats[['Filling rate', ]].copy()
    na_rate = 1 - filling_rate['Filling rate']
    filling_rate.insert(0, 'NA_', na_rate)
    filling_rate = filling_rate * 100
    filling_rate.columns = ['NA', 'Filled']

    shannon_entropy = stats['Shannon entropy']
    shannon_entropy = np.maximum(shannon_entropy * 100, precision)

    # Create stacked bar chart
    ax1 = filling_rate.plot(kind='bar', stacked=True, color=['lightcoral', 'lightgreen'])
    legend1 = ax1.legend(['NA', 'Filled'], loc='upper left', bbox_to_anchor=(1, 1))
    plt.gca().add_artist(legend1)

    # Add scatter plot for Shannon entropy
    ax2 = plt.scatter(
        np.arange(len(filling_rate)),  # x-coordinates
        shannon_entropy,               # y-coordinates
        s=200,                         # size of the points
        color='black'
    )
    plt.legend([ax2], ['Shannon entropy'], loc='upper left', bbox_to_anchor=(1, .8))

    plt.yscale('log')
    plt.ylim(precision, 100)

    # Axis titles
    plt.ylabel('Filling rate & Shannon entropy')
    plt.xlabel('')

    # Rotate x-axis labels
    plt.xticks(rotation=30, ha='right')

    # Add overall title
    plt.title(f'Discrete statistics of `{table_name}` table', fontsize=16)

    plt.savefig(
        f'../img/Filling rate & Shannon entropy of `{table_name}`.png',
        facecolor='white',
        bbox_inches='tight',
        dpi=300   # x 2
    )

    plt.show()


def show_discrete_stats(data, name=None):
    stats = discrete_stats(data, name=name)
    display(stats)
    plot_discrete_stats(stats)
