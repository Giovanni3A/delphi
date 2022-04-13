import numpy as np
import pandas as pd
from datetime import timedelta


def apply_time_frequency(ts: pd.Series, window: str, window_size: int):
    '''
    Auxiliar function that applies multiple time frequencies index.

    Parameters
    ----------
    ts: pandas.Series
        Timeseries that will be tranformed into time frequency indexes

    window: str
        Window frequency.
        Y -> year
        M -> month
        W -> week
        D -> day
        H -> hour
        m -> minute
        s -> second

    window_size: int
        How many windows to aggregate into one index.

    Returns
    -------
    w: pd.Series
        Index series.

    Example
    -------
    >>> ts = pd.to_datetime(pd.Series([
            '2022-01-01', '2022-01-04', '2022-01-10', '2022-12-31'
        ]))
    >>> apply_time_frequency(ts, 'W', 1)
        0     0.0
        1     1.0
        2     2.0
        3    52.0
        dtype: float64
    '''
    # apply window unit

    if window == 'Y':
        w = ts.dt.year - ts.dt.year.min()

    elif window == 'M':
        months = ts.dt.month
        years = ts.dt.year
        min_year = years.min()
        w = 12*(years - min_year) + months - 1

    elif window == 'W':
        min_day = ts.min().replace(hour=0, minute=0, second=0)
        min_day = min_day - timedelta(days=min_day.weekday())
        w = (ts - min_day).dt.total_seconds() // (7*24*60*60)

    elif window == 'D':
        min_day = ts.min().replace(hour=0, minute=0, second=0)
        w = (ts - min_day).dt.total_seconds() // (24*60*60)

    elif window == 'H':
        min_hour = ts.min().replace(hour=0, minute=0, second=0)
        w = (ts - min_hour).dt.total_seconds() // (60*60)

    elif window == 'm':
        min_minute = ts.min().replace(hour=0, minute=0, second=0)
        w = (ts - min_minute).dt.total_seconds() // (60)

    elif window == 's':
        min_second = ts.min().replace(second=0)
        w = (ts - min_second).dt.total_seconds()

    # window_size indicates how many consecutive windows should be aggregated
    return w // window_size


def calculate_sazonality(
    ts: pd.Series,
    sazonality_type: str,
    window: int,
    frequency: int
):
    '''
    Calculates sazonality index from multiple time windows and frequencies.

    Parameters
    ----------
    ts: pandas.Series
        Timeseries to be transformed into sazonality indexes.
        MUST BE sorted.

    sazonality_type: str
        Sazonality frequency type
            Y -> year
            M -> month
            W -> week
            D -> day
            H -> hour
            m -> minute
            S -> second

    window: (int, List[int])
        Sazonality window

    frequency: int
        Sazonality frequency

        eg: If sazonality is between 12 months in a year:
            sazonality_type = 'Y'
            window = 1
            frequency = 12

    Returns
    -------
    sazon_idx: pandas.Series
        Sazonality indexes

    Example
    -------
    >>> ts = pd.to_datetime(pd.Series([
            '2022-01-01', '2022-01-04', '2022-01-10', '2022-12-31'
        ]))
    >>> calculate_sazonality(ts, 'D', 1, 7)
        0     0
        1     3
        2     2
        3     0
        dtype: float64
    '''
    unitary_sazonality = apply_time_frequency(
        ts,
        sazonality_type,
        1
    )
    # if unique frequency, apply basic rule
    if type(window) is int:
        sazon_idx = (unitary_sazonality // window) % (frequency // window)

    # if variable frequencies, calculate indexes accordingly
    else:
        sazon_n = unitary_sazonality % frequency
        sazon_idx = pd.Series(np.nan, index=sazon_n.index)

        for i_w, w in enumerate(np.cumsum(window)):
            sazon_idx.loc[(sazon_n < w) & sazon_idx.isna()] = i_w

    return sazon_idx.astype(int)
