import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import persim
import yfinance as yf
from ordpy import weighted_permutation_entropy
from ripser import Rips

plt.rcParams.update({"text.usetex": False})

# Define index names and date range
index_names = ["^GSPC", "^DJI", "^IXIC", "^RUT"]
start_date_string = "1997-01-01"
# start_date_string = "2018-01-01"
end_date_string = "2024-04-21"

# Download and prepare data
raw_data = yf.download(index_names, start=start_date_string, end=end_date_string)
df_close = raw_data["Adj Close"].dropna(axis="rows")
P = df_close.to_numpy()
r = np.log(np.divide(P[1:], P[:-1]))

# Handle NaN values that might appear after log return calculation
r = np.nan_to_num(r)  # Replace NaNs with zero (or you might choose to drop them)


# Define and compute Wasserstein distances and permutation entropy
rips = Rips(maxdim=2)
w = 20
n = len(raw_data) - (2 * w) + 1
wasserstein_dists = np.zeros((n, 1))
perm_entropy = np.zeros(n)
hawkes_values = np.zeros(n)


# Define the Hawkes process function
def hawkes_process(data, decay):
    alpha = np.exp(-decay)
    output = np.zeros_like(data)
    output[0] = data[0]
    for t in range(1, len(data)):
        output[t] = alpha * output[t - 1] + (1 - alpha) * data[t]
    return output


for i in range(n):
    dgm1 = rips.fit_transform(r[i : i + w])
    dgm2 = rips.fit_transform(r[i + w + 1 : i + (2 * w) + 1])
    wasserstein_dists[i] = persim.wasserstein(dgm1[0], dgm2[0])
    # # Calculate permutation entropy for the window
    # perm_entropy[i] = permutation_entropy(r[i:i+(2*w)+1], dx=1, dy=1, taux=1, tauy=1, normalized=True)
    # Calculate permutation entropy for the window, ensuring data is appropriately shaped
    flat_data = r[i : i + (2 * w) + 1].flatten()  # Flatten the data
    perm_entropy[i] = weighted_permutation_entropy(flat_data, dx=10, normalized=True)
    hawkes_values[i] = hawkes_process(flat_data, decay=0.1)[-1]


# Define functions to calculate drawdowns and find peaks
def calculate_drawdowns(series):
    cum_max = series.cummax()
    return (series - cum_max) / cum_max


def find_peaks(series):
    return series[(series.shift(1) < series) & (series.shift(-1) < series)]


# Calculate significant drawdowns and find peaks
sp500_drawdowns = calculate_drawdowns(df_close["^GSPC"])
peaks = find_peaks(df_close["^GSPC"])

# Plotting
fig, ax1 = plt.subplots(figsize=(18, 8), dpi=80)
color = "tab:blue"
ax1.set_xlabel("Date")
ax1.set_ylabel("S&P 500 (scaled)", color=color)
ax1.plot(
    raw_data.index[w : n + w],
    df_close.iloc[w : n + w, 0] / max(df_close.iloc[w : n + w, 0]),
    color=color,
)
ax1.tick_params(axis="y", labelcolor=color)

print("checking r : ", r)
print("check wasserstein_dists: ", wasserstein_dists)
print("check perm_entropy: ", perm_entropy)

ax2 = ax1.twinx()
color = "tab:red"
ax2.set_ylabel("Metrics", color=color)
ax2.plot(
    raw_data.index[w : n + w],
    wasserstein_dists,
    color=color,
    label="Wasserstein distances",
    linewidth=0.8,
    alpha=0.7,
)
ax2.plot(
    raw_data.index[w : n + w],
    perm_entropy,
    color="tab:green",
    label="Permutation Entropy",
    linewidth=0.8,
    alpha=0.7,
)
plt.plot(
    raw_data.index[w : n + w],
    hawkes_values,
    color="orange",
    label="Hawkes Process Output",
    linewidth=0.8,
    alpha=0.7,
)
ax2.tick_params(axis="y", labelcolor=color)

# Draw horizontal line at Wasserstein distance 0.75
ax2.axhline(y=0.075, color='gray', linestyle='--', linewidth=1.5, label='Threshold at 0.075')
ax2.axhline(y=0.1, color='gray', linestyle='--', linewidth=1.5, label='Threshold at 0.1')
ax2.axhline(y=0.125, color='gray', linestyle='--', linewidth=1.5, label='Threshold at 0.125')


last_marked = None
marked_DD_dates = []
# Iterate over each drawdown
for date, value in sp500_drawdowns.items():
    if value < -0.15:  # Check for significant drawdowns
        if last_marked is None or (date - last_marked) > pd.Timedelta(
            "180D"
        ):  # Apply a cool-off period of 6 months
            # Find the last peak before the significant drawdown
            if not peaks[peaks.index < date].empty:
                last_peak_date = peaks[peaks.index < date].idxmax()
                # Calculate the value at the peak for plotting
                peak_value = df_close["^GSPC"].loc[last_peak_date] / max(
                    df_close["^GSPC"]
                )
                # Draw a vertical line from the peak to the x-axis
                ax1.axvline(
                    x=last_peak_date,
                    color="red",
                    linestyle="--",
                    linewidth=0.8,
                    label="Last Peak Before Drawdown" if last_marked is None else "",
                )
                # Update the last marked variable
                last_marked = date
                if last_peak_date not in marked_DD_dates:
                    marked_DD_dates.append(last_peak_date)

# fig.tight_layout()
# plt.title("Homology, Permutation Entropy, and Market Drawdowns with Peaks")
# plt.legend(loc="upper left")
# plt.savefig(
#     "combined_metrics_and_drawdowns.png",
#     dpi="figure",
#     format=None,
#     metadata=None,
#     bbox_inches="tight",
#     pad_inches=0.1,
#     facecolor="white",
#     edgecolor="auto",
# )
# plt.show()

import pandas as pd

def last_peak_before_drawdown(df_close, wasserstein_dists, index_dates, window=pd.Timedelta('30 days')):
    """
    Identify the last peak before significant drawdowns and extract surrounding data for Wasserstein distances.

    :param df_close: DataFrame containing the adjusted closing prices of the stock.
    :param wasserstein_dists: Array of Wasserstein distances calculated previously.
    :param index_dates: Dates corresponding to the entries in df_close and wasserstein_dists.
    :param window: The range of days before and after the peak date to include in the output.
    :return: A dictionary containing the peak dates and their surrounding Wasserstein distance values.
    """
    peaks = df_close[(df_close.shift(1) < df_close) & (df_close.shift(-1) < df_close)]
    drawdowns = calculate_drawdowns(df_close)

    results = {}
    
    # Filter significant drawdowns
    significant_drawdowns = drawdowns[drawdowns < -0.15]

    for drawdown_date in significant_drawdowns.index:
        # Find the last peak before the drawdown
        prior_peaks = peaks[peaks.index < drawdown_date]
        if not prior_peaks.empty:
            last_peak_date = prior_peaks.idxmax()

            # Find the corresponding index for the peak date in the full dataset
            peak_index = index_dates.get_loc(last_peak_date)
            
            # Calculate the start and end indices based on the window
            start_index = max(0, peak_index - window.days)
            end_index = min(len(wasserstein_dists), peak_index + window.days + 1)

            # Extract the Wasserstein distances around the peak date
            surrounding_dates = index_dates[start_index:end_index]
            surrounding_wassersteins = wasserstein_dists[start_index:end_index]

            # Store the results
            results[last_peak_date] = {
                'dates': surrounding_dates,
                'wasserstein_distances': surrounding_wassersteins
            }

    return results

# Usage example
index_dates = pd.DatetimeIndex(raw_data.index)  # Assuming 'raw_data.index' holds the dates
results = last_peak_before_drawdown(df_close["^GSPC"], wasserstein_dists, index_dates)
print("#####")
print(results)




import pandas as pd
from datetime import timedelta

# def filter_predictions(predictions, cool_off_period='30D'):
#     cool_off = pd.Timedelta(cool_off_period)
#     last_prediction_date = None
#     filtered_prediction_dates = []

#     for date in predictions[predictions].index:
#         if last_prediction_date is None or date > last_prediction_date + cool_off:
#             filtered_prediction_dates.append(date)
#             last_prediction_date = date

#     return pd.DatetimeIndex(filtered_prediction_dates)

def evaluate_predictions(df, predictions, actual_downturn_dates, threshold, window=30):
    """
    Evaluate prediction accuracy for market downturns based on Wasserstein distances.
    
    :param df: DataFrame with date indices and Wasserstein distances.
    :param predictions: Series with calculated Wasserstein distances.
    :param actual_downturn_dates: Series or list of dates when actual downturns occurred.
    :param threshold: Numeric, the threshold for what constitutes a prediction.
    :param window: Integer, number of days before and after the prediction to look for an actual downturn.
    :return: Dictionary with precision and recall values.
    """
    # Generate prediction flags
    prediction_flags = predictions > threshold
    prediction_dates = prediction_flags[prediction_flags].index

    cool_off_period='180D'
    cool_off = pd.Timedelta(cool_off_period)
    last_prediction_date = None
    filtered_prediction_dates = []

    for date in prediction_dates:
        if last_prediction_date is None or date > last_prediction_date + cool_off:
            filtered_prediction_dates.append(date)
            last_prediction_date = date

    prediction_dates =  pd.DatetimeIndex(filtered_prediction_dates)

    print("***")
    print("prediction_dates: ", prediction_dates)
    print("actual_downturn_dates: ", actual_downturn_dates)
    print("***")
    for date in prediction_dates:
        print("predicted date: ", date)
        ax1.axvline(
                        x=date,
                        color="green",
                        linestyle="--",
                        linewidth=1.0,
                        label="Predicted Downturn" if last_marked is None else "",
                    )
    # Prepare to track hits and false alarms
    hits = 0
    false_alarms = 0

    predict_then_go_down = 0
    predict_then_go_up = 0

    # Buffer period to check for downturn around each prediction
    buffer = timedelta(days=window)

    # Check each prediction
    for prediction_date in prediction_dates:
        # Check if there's an actual downturn within the window of this prediction
        if any((actual_downturn_dates >= prediction_date - buffer) & (actual_downturn_dates <= prediction_date + buffer)):
            print("prediction_date: ", prediction_date, " HIT")
            hits += 1
        else:
            print("prediction_date: ", prediction_date, " MISS")
            false_alarms += 1
        price_at_prediction     = df["^GSPC"].loc[prediction_date]
        if (prediction_date + buffer).weekday() < 5:
            try:
                price_after_prediction  = df["^GSPC"].loc[prediction_date + buffer]
            except:
                try:
                    price_after_prediction  = df["^GSPC"].loc[prediction_date + buffer + timedelta(days=1)]
                except:
                    try:
                        price_after_prediction  = df["^GSPC"].loc[prediction_date + buffer + timedelta(days=2)]
                    except:
                        try:
                            price_after_prediction  = df["^GSPC"].loc[prediction_date + buffer + timedelta(days=3)]
                        except:
                            try:
                                price_after_prediction  = df["^GSPC"].loc[prediction_date + buffer + timedelta(days=4)]
                            except:
                                price_after_prediction  = df["^GSPC"].loc[prediction_date + buffer + timedelta(days=5)]
        elif (prediction_date + buffer).weekday() == 5:
            try:
                price_after_prediction  = df["^GSPC"].loc[prediction_date + buffer + timedelta(days=2)]
            except:
                price_after_prediction  = df["^GSPC"].loc[prediction_date + buffer + timedelta(days=3)]
        elif (prediction_date + buffer).weekday() == 6:
            try:
                price_after_prediction  = df["^GSPC"].loc[prediction_date + buffer + timedelta(days=1)]
            except:
                price_after_prediction  = df["^GSPC"].loc[prediction_date + buffer + timedelta(days=2)]
        print("price_at_prediction: ", price_at_prediction)
        print("price_after_prediction: ", price_after_prediction)
        if price_after_prediction < price_at_prediction:
            predict_then_go_down += 1
        else:
            predict_then_go_up += 1
        print("***")

    # Calculate precision and recall
    if hits + false_alarms > 0:
        precision = hits / (hits + false_alarms)
    else:
        precision = 0

    if len(actual_downturn_dates) > 0:
        recall = hits / len(actual_downturn_dates)
    else:
        recall = 0

    simple_dd_accuracy = predict_then_go_down / len(prediction_dates)

    return {'precision': precision, 'recall': recall, 'simple_dd_accuracy': simple_dd_accuracy}

marked_DD_dates = pd.Series(marked_DD_dates)
print("check marked_DD_dates: ", marked_DD_dates)

wasserstein_series = pd.Series(wasserstein_dists.flatten(), index=raw_data.index[w:n+w])
actual_downturn_dates = pd.Series(df_close.index[sp500_drawdowns < -0.15])  # Assuming significant downturn is defined as -15% drawdown

results = evaluate_predictions(df_close, wasserstein_series, marked_DD_dates, threshold=0.12)
print("Precision:", results['precision'])
print("Recall:", results['recall'])
print("simple_dd_accuracy:", results['simple_dd_accuracy'])




fig.tight_layout()
plt.title("Homology, Permutation Entropy, and Market Drawdowns with Peaks")
plt.legend(loc="upper left")
plt.savefig(
    "combined_metrics_and_drawdowns.png",
    dpi="figure",
    format=None,
    metadata=None,
    bbox_inches="tight",
    pad_inches=0.1,
    facecolor="white",
    edgecolor="auto",
)
plt.show()



