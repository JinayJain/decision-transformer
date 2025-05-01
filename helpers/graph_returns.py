import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score


def main():
    with open("eval_returns.json", "r") as f:
        data = json.load(f)

    returns = np.linspace(0.0, 200.0, 21)

    means = data["means"]
    medians = data["medians"]
    stds = data["stds"]

    # Calculate the line of best fit (linear regression)
    coefficients = np.polyfit(returns, means, 1)
    poly = np.poly1d(coefficients)
    y_pred = poly(returns)
    r_squared = r2_score(means, y_pred)

    # Plot the scatter and the line of best fit
    plt.scatter(returns, means, label="Means")
    plt.plot(
        returns,
        y_pred,
        color="red",
        label=f"Line of Best Fit: y = {coefficients[0]:.2f}x + {coefficients[1]:.2f}\nR² = {r_squared:.2f}",
    )

    # Add labels, title, and legend
    plt.xlabel("Targeted Returns")
    plt.ylabel("Average Reward per Episode")
    plt.title("Decision Transformer: Avg Reward vs Target Return")
    plt.legend()
    plt.grid(True)
    plt.savefig("returns.png")


if __name__ == "__main__":
    main()
