import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

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


def plot_qp_region_and_path(path, x_star):
    """
    Plot the 3D simplex feasible region and the central path for the QP example.

    Parameters:
    - path: array of shape (K,3) with the central path outer iterates
    - x_star: array of shape (3,) with the final solution
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Simplex vertices
    verts = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # Plot simplex as a triangular surface
    ax.plot_trisurf(verts[:,0], verts[:,1], verts[:,2], color='lightgray', alpha=0.5)
    # Plot central path points
    ax.plot(path[:,0], path[:,1], path[:,2], '-o', label='central path')
    # Highlight final solution
    ax.scatter(x_star[0], x_star[1], x_star[2], color='red', s=50, label='solution')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('QP Feasible Region & Central Path')
    ax.legend()
    plt.show()


def plot_lp_region_and_path(path, x_star):
    """
    Plot the 2D polygon feasible region and the central path for the LP example.

    Parameters:
    - path: array of shape (K,2) with the central path outer iterates
    - x_star: array of shape (2,) with the final solution
    """
    fig, ax = plt.subplots()
    # Define polygon vertices of the feasible region
    verts = np.array([[1,0], [2,0], [2,1], [0,1]])
    poly = Polygon(verts, facecolor='lightgray', alpha=0.5)
    ax.add_patch(poly)
    # Plot central path points
    ax.plot(path[:,0], path[:,1], '-o', label='central path')
    # Highlight final solution
    ax.scatter(x_star[0], x_star[1], color='red', s=50, label='solution')
    ax.set_xlim(0, 2.1)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('LP Feasible Region & Central Path')
    ax.set_aspect('equal', 'box')
    ax.legend()
    plt.show()


def plot_objective_vs_outer(obj_vals, title='Objective vs Outer Iteration'):
    """
    Plot the objective value against outer iteration number.

    Parameters:
    - obj_vals: array-like of objectives at each outer iteration
    - title: title string for the plot
    """
    plt.figure()
    plt.plot(np.arange(len(obj_vals)), obj_vals, '-o')
    plt.xlabel('Outer iteration')
    plt.ylabel('Objective value')
    plt.title(title)
    plt.show()