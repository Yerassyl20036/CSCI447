import numpy as np
import matplotlib.pyplot as plt


def load_data(filename):
    data = np.loadtxt(filename)  # Load space-separated values
    return data.T  # Transpose to get d-by-N shape

# Function to plot 2D coordinates
def plot_data(data, title):
    plt.figure(figsize=(6,6))
    plt.plot(data[0, :], data[1, :], '.', markersize=10)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title(title)
    plt.grid(True)
    plt.show()

# Z-score normalization
def z_score_normalization(data):
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    return (data - mean) / std

# Min-Max normalization
def min_max_normalization(data):
    min_vals = np.min(data, axis=1, keepdims=True)
    max_vals = np.max(data, axis=1, keepdims=True)
    return (data - min_vals) / (max_vals - min_vals)

# Robust scaling
def robust_scaling(data):
    median = np.median(data, axis=1, keepdims=True)
    q1 = np.percentile(data, 25, axis=1, keepdims=True)
    q3 = np.percentile(data, 75, axis=1, keepdims=True)
    iqr = q3 - q1
    return (data - median) / iqr

# Main function
def main():
    filename = 'handpositions.txt'  # Update with the correct file path
    
    # Task 3: Load data
    data = load_data(filename)
    plot_data(data, 'Original Data')
    
    # Task 4: Z-score normalization
    z_score_data = z_score_normalization(data)
    plot_data(z_score_data, 'Z-Score Normalized Data')
    
    # Task 5: Min-Max normalization
    min_max_data = min_max_normalization(data)
    plot_data(min_max_data, 'Min-Max Normalized Data')
    
    # Task 6: Robust scaling
    robust_scaled_data = robust_scaling(data)
    plot_data(robust_scaled_data, 'Robust Scaled Data')

if __name__ == "__main__":
    main()