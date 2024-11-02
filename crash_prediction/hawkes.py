import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import persim
import yfinance as yf
from ordpy import weighted_permutation_entropy
from ripser import Rips

plt.rcParams.update({"text.usetex": False})


# Define the Hawkes process function
def hawkes_process(data, decay):
    alpha = np.exp(-decay)
    output = np.zeros_like(data)
    output[0] = data[0]
    for t in range(1, len(data)):
        output[t] = alpha * output[t - 1] + (1 - alpha) * data[t]
    return output


# Download data
index_names = ["^GSPC", "^DJI", "^IXIC", "^RUT"]
start_date = "2018-01-01"
end_date = "2024-04-21"
raw_data = yf.download(index_names, start=start_date, end=end_date)
df_close = raw_data["Adj Close"].dropna()

# Prepare data
log_returns = np.log(df_close / df_close.shift(1)).dropna()
P = log_returns.to_numpy()

# Compute Wasserstein distances and permutation entropy
rips = Rips(maxdim=2)
w = 20
n = len(P) - (2 * w) + 1
wasserstein_dists = np.zeros(n)
perm_entropy = np.zeros(n)
hawkes_values = np.zeros(n)

# Analysis loop
for i in range(n):
    window = P[i : i + (2 * w)]
    dgm1 = rips.fit_transform(window[:w])
    dgm2 = rips.fit_transform(window[w:])
    wasserstein_dists[i] = persim.wasserstein(dgm1[0], dgm2[0])
    flat_window = window.flatten()
    perm_entropy[i] = weighted_permutation_entropy(flat_window, dx=10, normalized=True)
    hawkes_values[i] = hawkes_process(window[:, 0], decay=0.1)[-1]

# Plotting results
plt.figure(figsize=(15, 10))
dates = df_close.index[w : (w + n)]
plt.plot(dates, df_close.iloc[w : (w + n), 0], label="S&P 500 Close", color="gray")
plt.plot(dates, wasserstein_dists, label="Wasserstein Distances", color="blue")
plt.plot(dates, perm_entropy, label="Permutation Entropy", color="green")
plt.plot(dates, hawkes_values, label="Hawkes Process Output", color="red")
plt.title("Comparison of Metrics")
plt.xlabel("Date")
plt.ylabel("Metric Value")
plt.legend()
plt.show()
