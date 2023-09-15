import datetime

import pytz as pytz
import yfinance as yf
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

from model.helpers import train, predict
from model.preprocessors import process_inputs, process_targets

import pandas as pd


TRAINING_CUTOFF = datetime.datetime(2020, 1, 1, tzinfo=pytz.timezone('America/New_York'))

if __name__ == "__main__":
    # Download price histories from Yahoo Finance
    spy = yf.Ticker("SPY")
    price_series = spy.history(period='max')['Close'].dropna()

    price_series.to_pickle('prices.pkl')

    price_series = pd.read_pickle('prices.pkl')

    x_df = process_inputs(price_series, window_length=10)
    y_series = process_targets(price_series)

    # Only keep rows in which we have both inputs and data.
    common_index = x_df.index.intersection(y_series.index)
    x_df, y_series = x_df.loc[common_index], y_series.loc[common_index]

    # Isolate training data
    training_x_series = x_df.loc[x_df.index < TRAINING_CUTOFF]
    training_y_series = y_series.loc[y_series.index < TRAINING_CUTOFF]

    trained_model = train(training_x_series, training_y_series)

    # Isolate test data
    test_x_series = x_df.loc[x_df.index >= TRAINING_CUTOFF]
    actual_series = y_series.loc[y_series.index >= TRAINING_CUTOFF]

    forecast_series = predict(trained_model, test_x_series)
    results_df = forecast_series.to_frame('Forecast').join(actual_series.to_frame('Actual')).dropna()

    # Evaluate forecasts
    results_df.plot.scatter(x='Actual', y='Forecast')
    plt.show()
    print(f"R Squared: {r2_score(results_df['Actual'], results_df['Forecast']):.4f}, "
          f"Mean Absolute Error: {mean_absolute_error(results_df['Actual'], results_df['Forecast']):.4f}")
