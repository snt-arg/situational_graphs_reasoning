import numpy as np
import matplotlib.pyplot as plt

# Define the input range [0, 1]
x = np.linspace(0, 1, 200)

# Exponential output range bounds
a, b = 0.0001, 10

# Different steepness parameters to explore
k_values = [0.5, 10, 100, 50, 1.5, 2, 5]

# Set up plots
fig, axes = plt.subplots(2, 1, figsize=(10, 10))

# Plot exponential mappings
for k in k_values:
    mapped_x = a * (b / a) ** (x ** k)
    axes[0].plot(x, mapped_x, label=f'k={k}')

axes[0].set_title('Exponential Mapping [0, 1] → [0.0001, 10] with Varying Steepness')
axes[0].set_xlabel('Input [0, 1]')
axes[0].set_ylabel('Mapped Output')
axes[0].grid(True)
axes[0].legend()

# Plot inverse mappings (1 / mapped_x)
for k in k_values:
    mapped_x = a * (b / a) ** (x ** k)
    inverse_x = 1 / mapped_x
    axes[1].plot(x, inverse_x, label=f'k={k}')

axes[1].set_title('Inverse Mapping (1 / mapped_x) with Varying Steepness')
axes[1].set_xlabel('Input [0, 1]')
axes[1].set_ylabel('Inverse Output')
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.show()
