from ordpy import permutation_entropy, weighted_permutation_entropy

# Example time series data
data = [4, 7, 9, 10, 6, 2, 1, 8, 3, 5]

# Calculate standard permutation entropy
pe = permutation_entropy(data, dx=3, dy=1, normalized=True)
print("Permutation Entropy:", pe)

# Calculate weighted permutation entropy
wpe = weighted_permutation_entropy(data, dx=3, dy=1, normalized=True)
print("Weighted Permutation Entropy:", wpe)
