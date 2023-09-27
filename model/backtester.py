import numpy as np
import pandas as pd


def simple_backtest(results_df: pd.DataFrame):
    # Calculate cumulative returns from following the model
    model_results_df = results_df.copy()
    model_results_df.loc[model_results_df['Forecast'] <= 0.] = 0.
    model_series = (model_results_df['Actual'] + 1.).cumprod() - 1.
    model_cumulative_return = model_series.iloc[-1]
    model_stdev = model_results_df['Actual'].std() * np.sqrt(252.)

    # Calculate cumulative returns from buying and holding
    buy_hold_series = (results_df['Actual'] + 1.).cumprod() - 1.
    buy_hold_cumulative_return = buy_hold_series.iloc[-1]
    buy_hold_stdev = results_df['Actual'].std() * np.sqrt(252.)

    print(f"Model cumulative returns: {model_cumulative_return:.4f}, Buy and hold: {buy_hold_cumulative_return:.4f}")
    print(f"Model standard deviation: {model_stdev:.4f}, Buy and hold: {buy_hold_stdev:.4f}")
