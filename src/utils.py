import numpy as np
import matplotlib.pyplot as plt

def plot_contour(f_func, x_range, y_range, history=None, levels=50, title=""):
    """
    Plots the 2D contour lines of f_func over a grid defined by x_range and y_range,
    and optionally overlays the optimization path from 'history'.

    Args:
        f_func: function that takes (x, hessian_flag) and returns (f, grad, hess).
                In contour plotting, only f(x, False) is used.
        x_range: tuple (x_min, x_max)
        y_range: tuple (y_min, y_max)
        history: optimization history dictionary with keys "locations" → list of 2D points
        levels: number of contour levels
        title: string title for the plot
    """
    x_min, x_max = x_range
    y_min, y_max = y_range

    # Prepare grid
    N = 200
    X = np.linspace(x_min, x_max, N)
    Y = np.linspace(y_min, y_max, N)
    XX, YY = np.meshgrid(X, Y)
    Z = np.zeros_like(XX)

    # Compute f on the grid
    for i in range(N):
        for j in range(N):
            x_ij = np.array([XX[i, j], YY[i, j]])
            f_ij, _, _ = f_func(x_ij, hessian=False)
            Z[i, j] = f_ij

    # Contour plot
    plt.figure()
    contour_set = plt.contour(XX, YY, Z, levels=levels, cmap="viridis")
    plt.clabel(contour_set, inline=True, fontsize=8)
    plt.title(title)
    plt.xlabel("x₁")
    plt.ylabel("x₂")

    # If history is provided, overlay the path
    if history is not None and "locations" in history:
        locs = np.array(history["locations"])
        if locs.shape[1] == 2:
            plt.plot(locs[:, 0], locs[:, 1], 'r.-', label="Path")
            plt.scatter(locs[0, 0], locs[0, 1], c='green', marker='o', label="Start")
            plt.scatter(locs[-1, 0], locs[-1, 1], c='blue', marker='x', label="End")
            plt.legend()

    plt.show()


def plot_function_values(history, title="Function Value vs Iteration"):
    """
    Plots the objective value at each iteration from the history dictionary.

    Args:
        history: dictionary with key "values" → list of function values
        title: title for the plot
    """
    if history is None or "values" not in history:
        raise ValueError("History must be provided with key 'values'.")

    values = history["values"]
    iters = list(range(len(values)))

    plt.figure()
    plt.plot(iters, values, 'b.-')
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("f(x)")
    plt.yscale("log")  # Often useful for optimization logs
    plt.grid(True, which="both", ls="--")
    plt.show()
