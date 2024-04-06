from scipy.stats import entropy
distribution = [1/4, 1/4, 1/4, 1/4]
entropy_value = entropy(distribution, base=2)
print(f"Entropy: {entropy_value:.2f} bits")