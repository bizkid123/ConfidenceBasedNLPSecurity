from matplotlib import pyplot as plt
import numpy as np
import math
import pandas as pd


def plot_mse_predictions(test_y, predictions):
    base_y = test_y[test_y == 0]
    adversarial_y = test_y[test_y == 1]

    base_predictions = predictions[test_y == 0]
    adversarial_predictions = predictions[test_y == 1]

    # Plot distribution of differences
    plt.hist(
        [abs(base_predictions[i] - base_y[i]) for i in range(len(base_y))],
        bins=50,
        color="blue",
    )
    plt.title("Difference between predictions and actual values for base examples")

    plt.show()

    plt.hist(
        [
            abs(adversarial_predictions[i] - adversarial_y[i])
            for i in range(len(base_y), len(base_y) + len(test_y))
        ],
        bins=50,
        color="yellow",
    )
    plt.title(
        "Difference between predictions and actual values for adversarial examples"
    )

    plt.show()


def plot_mse_vs_confidence(test_y, predictions, confidence):
    base_y = test_y[test_y == 0]
    adversarial_y = test_y[test_y == 1]

    base_predictions = predictions[test_y == 0]
    adversarial_predictions = predictions[test_y == 1]

    base_confidence = confidence[test_y == 0]
    adversarial_confidence = confidence[test_y == 1]

    # Plot confidence vs. difference,
    plt.scatter(
        base_confidence,
        [abs(base_predictions[i] - base_y[i]) for i in range(len(base_y))],
        color="blue",
    )
    plt.title("Confidence vs. difference for base examples")

    plt.show()

    plt.scatter(
        adversarial_confidence,
        [
            abs(adversarial_predictions[i] - adversarial_y[i])
            for i in range(len(adversarial_y))
        ],
        color="yellow",
    )

    plt.title("Confidence vs. difference for adversarial examples")

    plt.show()


def plot_wdr(data, first_n=512, color="blue", title=""):
    # Your data: A 2D array where each row is an instance of the distribution
    data = np.stack(data)  # Replace this with your actual data

    data = data[:, :first_n]

    # Calculate mean and standard deviation across instances
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    # Create x-axis values
    x_values = np.arange(1, len(data[0]) + 1)

    # Plot the mean and standard deviation
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, mean, color=color, label="Mean")
    plt.fill_between(
        x_values,
        mean - std,
        mean + std,
        color=color,
        alpha=0.2,
        label="Standard Deviation",
    )

    plt.xlabel("Ordered Points")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_confidence_histogram(confidence, name="", bins=50):
    plt.hist(confidence, bins=bins)
    plt.title("Confidence Histogram for " + name)
    plt.show()


def better_plot_wdr(
    data,
    percentiles=[0, 1, 5, 10, 25, 75, 90, 95, 99, 100],
    first_n=512,
    color="blue",
    title="",
    y_range=None,
    alphas=None,
):
    data = np.stack(data)  # Replace this with your actual data
    data = data[:, :first_n]

    percentile_values = [np.percentile(data, p, axis=0) for p in percentiles]
    # Increasing then decreasing alphas
    alpha_lin = np.linspace(0.05, 0.5, len(percentiles) // 2)
    alpha_lin = np.concatenate([alpha_lin, alpha_lin[::-1]])

    # percentile_values = np.percentile(data, 25, axis=0)
    x_values = np.arange(1, data.shape[1] + 1)

    plt.figure(figsize=(10, 6))

    # Plot line for mean

    mean = np.mean(data, axis=0)
    plt.plot(x_values, mean, color=color, label="Mean")

    # Plot dotted lines for min and max
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    plt.plot(x_values, data_min, color=color, linestyle="--", label="Min")
    plt.plot(x_values, data_max, color=color, linestyle="--", label="Max")

    for i, (p_low, p_high) in enumerate(zip(percentiles[:-1], percentiles[1:])):
        # Calculate alpha based on percentile difference
        if alphas is None:
            alpha = (1 - 1 / math.e ** (1 * ((p_high - p_low) / 100))) ** (1 / 2)
            alpha += alpha_lin[i]
            alpha /= 2
        else:
            alpha = alphas[i]

        # Fill the percentile range
        plt.fill_between(
            x_values,
            percentile_values[i],
            percentile_values[i + 1],
            color=color,
            alpha=alpha,
            label=f"{p_low}-{p_high} Percentile Range",
        )
    # Set y limits
    if y_range is not None:
        plt.ylim(y_range)

    plt.xlabel("nth WDR")
    plt.ylabel("Value")
    plt.title(title)
    # Legend in top right corner
    plt.legend(loc="upper right")
    plt.show()
