import pandas as pd


def process_inputs(price_series: pd.Series, window_length: int) -> pd.DataFrame:
    """
    Creates sequences consisting of data from across a moving window. For example, given a window length of 10,
    sequences will span indices 1-10, 2-11, 3-12, etc. The sequence is scaled by the value just before the first
    value in the moving window. So for the sequence that spans 1-10, it'll be scaled by the value in index 0.
    :param price_series: The stock price data to extract sequences from.
    :param window_length: The size of the moving window.
    :return: Pandas DataFrame where each row contains a sequence, and the index refers to the most recent input date,
    a.k.a. the reference date.
    """
    dataframes = []
    for i in range(window_length):
        dataframes.append(price_series.shift(i).to_frame(f"T - {i}"))

    df = pd.concat(reversed(dataframes), axis=1)

    # Remove level values by scaling by starting price
    df = df.divide(price_series.shift(window_length), axis='rows') - 1.
    return df.dropna()


def process_targets(perf_series: pd.Series) -> pd.Series:
    """
    Creates targets consisting of data 2 days after the reference date (i.e. the most recent input's date)
    :param perf_series: The stock price returns data to extract targets from.
    :return: A series where the values consist of returns 2 days after the reference dates, and where the index consists
    of the reference dates.
    """
    return perf_series.pct_change().shift(-2).dropna()
