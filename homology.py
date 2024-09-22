import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ripser import Rips
import persim
from ordpy import weighted_permutation_entropy

plt.rcParams.update({
    "text.usetex": False
})

# Define index names and date range
index_names = ['^GSPC', '^DJI', '^IXIC', '^RUT']
# start_date_string = "1997-01-01"
start_date_string = "2018-01-01"
end_date_string = "2024-04-21"

# Download and prepare data
raw_data = yf.download(index_names, start=start_date_string, end=end_date_string)
df_close = raw_data['Adj Close'].dropna(axis='rows')
P = df_close.to_numpy()
r = np.log(np.divide(P[1:], P[:-1]))

# Handle NaN values that might appear after log return calculation
r = np.nan_to_num(r)  # Replace NaNs with zero (or you might choose to drop them)


# Define and compute Wasserstein distances and permutation entropy
rips = Rips(maxdim=2)
w = 30
n = len(raw_data)-(2*w)+1
wasserstein_dists = np.zeros((n,1))
perm_entropy = np.zeros(n)

for i in range(n):
    dgm1 = rips.fit_transform(r[i:i+w])
    dgm2 = rips.fit_transform(r[i+w+1:i+(2*w)+1])
    wasserstein_dists[i] = persim.wasserstein(dgm1[0], dgm2[0])
    # # Calculate permutation entropy for the window
    # perm_entropy[i] = permutation_entropy(r[i:i+(2*w)+1], dx=1, dy=1, taux=1, tauy=1, normalized=True)
    # Calculate permutation entropy for the window, ensuring data is appropriately shaped
    flat_data = r[i:i+(2*w)+1].flatten()  # Flatten the data
    perm_entropy[i] = weighted_permutation_entropy(flat_data, dx=10, normalized=True)

# Define functions to calculate drawdowns and find peaks
def calculate_drawdowns(series):
    cum_max = series.cummax()
    return (series - cum_max) / cum_max

def find_peaks(series):
    return series[(series.shift(1) < series) & (series.shift(-1) < series)]

# Calculate significant drawdowns and find peaks
sp500_drawdowns = calculate_drawdowns(df_close['^GSPC'])
peaks = find_peaks(df_close['^GSPC'])

# Plotting
fig, ax1 = plt.subplots(figsize=(18, 8), dpi=80)
color = 'tab:blue'
ax1.set_xlabel('Date')
ax1.set_ylabel('S&P 500 (scaled)', color=color)
ax1.plot(raw_data.index[w:n+w], df_close.iloc[w:n+w,0]/max(df_close.iloc[w:n+w,0]), color=color)
ax1.tick_params(axis='y', labelcolor=color)

print("checking r : ", r)
print("check wasserstein_dists: ", wasserstein_dists)
print("check perm_entropy: ", perm_entropy)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Metrics', color=color)
ax2.plot(raw_data.index[w:n+w], wasserstein_dists, color=color, label='Wasserstein distances')
ax2.plot(raw_data.index[w:n+w], perm_entropy, color='tab:green', label='Permutation Entropy')
ax2.tick_params(axis='y', labelcolor=color)

last_marked = None
# Iterate over each drawdown
for date, value in sp500_drawdowns.items():
    if value < -0.15:  # Check for significant drawdowns
        if last_marked is None or (date - last_marked) > pd.Timedelta('180D'):  # Apply a cool-off period of 6 months
            # Find the last peak before the significant drawdown
            if not peaks[peaks.index < date].empty:
                last_peak_date = peaks[peaks.index < date].idxmax()
                # Calculate the value at the peak for plotting
                peak_value = df_close['^GSPC'].loc[last_peak_date] / max(df_close['^GSPC'])
                # Draw a vertical line from the peak to the x-axis
                ax1.axvline(x=last_peak_date, color='red', linestyle='--', linewidth=2, label='Last Peak Before Drawdown' if last_marked is None else "")
                # Update the last marked variable
                last_marked = date

fig.tight_layout()
plt.title('Homology, Permutation Entropy, and Market Drawdowns with Peaks')
plt.legend(loc='upper left')
plt.savefig("combined_metrics_and_drawdowns.png", dpi='figure', format=None, metadata=None, bbox_inches='tight', pad_inches=0.1, facecolor='white', edgecolor='auto')
plt.show()
