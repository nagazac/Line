import numpy as np
import matplotlib.pyplot as plt

# Import get_history from your optimization module
from src.unconstrained_min import get_history

def plot_contour_two(
    f_func,
    x_range,
    y_range,
    hist_gd,
    hist_nt,
    levels=50,
    title="",
    gdl_color="red",
    ntl_color="blue"
):
    """
    Draw a single contour plot of f_func over the grid defined by x_range × y_range,
    then overlay two optimization paths: one for GD (in gdl_color) and one for NT (in ntl_color).

    Args:
        f_func: function taking (x, hessian_flag) → (f, grad, hess). Only f(x, False) is used.
        x_range: tuple (x_min, x_max)
        y_range: tuple (y_min, y_max)
        hist_gd: history dict for GD (keys "locations" & "values")
        hist_nt: history dict for NT (same structure)
        levels: number of contour levels (or explicit level sequence)
        title: string for the plot title
        gdl_color: color name or code for the GD path (default "red")
        ntl_color: color name or code for the NT path (default "blue")
    """
    x_min, x_max = x_range
    y_min, y_max = y_range

    # Create a dense grid
    N = 200
    X = np.linspace(x_min, x_max, N)
    Y = np.linspace(y_min, y_max, N)
    XX, YY = np.meshgrid(X, Y)
    ZZ = np.zeros_like(XX)

    # Evaluate f on the grid
    for i in range(N):
        for j in range(N):
            xij = np.array([XX[i, j], YY[i, j]])
            fij, _, _ = f_func(xij, hessian=False)
            ZZ[i, j] = fij

    # Draw contour
    plt.figure()
    cs = plt.contour(XX, YY, ZZ, levels=levels, cmap="viridis")
    plt.clabel(cs, inline=True, fontsize=8)
    plt.title(title)
    plt.xlabel("x₁")
    plt.ylabel("x₂")

    # Overlay the GD path in gdl_color
    if hist_gd is not None and "locations" in hist_gd and len(hist_gd["locations"]) > 0:
        locs_gd = np.array(hist_gd["locations"])
        if locs_gd.ndim == 2 and locs_gd.shape[1] == 2:
            plt.plot(
                locs_gd[:, 0],
                locs_gd[:, 1],
                color=gdl_color,
                linestyle='-',
                marker='.',
                label="GD Path"
            )
            plt.scatter(
                locs_gd[0, 0],
                locs_gd[0, 1],
                c='green',
                marker='o',
                s=50,
                label="GD Start"
            )
            plt.scatter(
                locs_gd[-1, 0],
                locs_gd[-1, 1],
                c=gdl_color,
                marker='x',
                s=60,
                label="GD End"
            )

    # Overlay the NT path in ntl_color
    if hist_nt is not None and "locations" in hist_nt and len(hist_nt["locations"]) > 0:
        locs_nt = np.array(hist_nt["locations"])
        if locs_nt.ndim == 2 and locs_nt.shape[1] == 2:
            plt.plot(
                locs_nt[:, 0],
                locs_nt[:, 1],
                color=ntl_color,
                linestyle='-',
                marker='.',
                label="NT Path"
            )
            plt.scatter(
                locs_nt[0, 0],
                locs_nt[0, 1],
                c='cyan',
                marker='o',
                s=50,
                label="NT Start"
            )
            plt.scatter(
                locs_nt[-1, 0],
                locs_nt[-1, 1],
                c=ntl_color,
                marker='x',
                s=60,
                label="NT End"
            )

    plt.legend()
    plt.show()


def plot_function_values_two(
    hist_gd,
    hist_nt,
    title="Function Value vs Iteration",
    gdl_color="red",
    ntl_color="blue"
):
    """
    Plot f(x) vs iteration index for both GD and NT on the same log-scale figure.

    Args:
        hist_gd: history dict for GD with key "values"
        hist_nt: history dict for NT with key "values"
        title: plot title
        gdl_color: color for GD curve (default "red")
        ntl_color: color for NT curve (default "blue")
    """
    if hist_gd is None or "values" not in hist_gd:
        raise ValueError("hist_gd must be provided with key 'values'.")
    if hist_nt is None or "values" not in hist_nt:
        raise ValueError("hist_nt must be provided with key 'values'.")

    vals_gd = hist_gd["values"]
    vals_nt = hist_nt["values"]
    iters_gd = list(range(len(vals_gd)))
    iters_nt = list(range(len(vals_nt)))

    plt.figure()
    plt.semilogy(
        iters_gd,
        vals_gd,
        color=gdl_color,
        linestyle='-',
        marker='.',
        label="GD f(x)"
    )
    plt.semilogy(
        iters_nt,
        vals_nt,
        color=ntl_color,
        linestyle='-',
        marker='.',
        label="NT f(x)"
    )
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("f(x) [log scale]")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()
