import pandas as pd


def process_inputs(perf_series: pd.Series, window_length: int) -> pd.DataFrame:
    dataframes = []
    for i in range(window_length):
        dataframes.append(perf_series.shift(i).to_frame(f"T - {i}"))

    df = pd.concat(reversed(dataframes), axis=1)

    # Remove level values by scaling by starting price
    df = df.divide(perf_series.shift(window_length), axis='rows') - 1.
    return df.dropna()


def process_targets(perf_series: pd.Series) -> pd.Series:
    """
    Creates targets consisting of data 2 days after the reference date (i.e. the most recent input's date)
    :param perf_series: The stock price returns data to extract targets from.
    :return: A series where the values consist of returns 2 days after the reference dates, and where the index consists
    of the reference dates.
    """
    return perf_series.pct_change().shift(-2).dropna()
