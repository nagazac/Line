import unittest
import numpy as np
from colorama import Fore, Style, init

from src.unconstrained_min import minimize, get_history
from src.utils import plot_contour, plot_function_values
from tests.examples import (
    q_circle,
    q_ellipse_axis,
    q_ellipse_rotated,
    rosenbrock,
    linear_func,
    smooth_corner
)

# Initialize Colorama
init(autoreset=True)


class TestUnconstrainedMinimization(unittest.TestCase):
    """
    For each example function, run both Gradient Descent (GD) and Newton's Method (NT).
    Print final location, final value, and success flag using Colorama. 
    Generate two plots per example:
      1. 2D contour + iteration paths of both methods.
      2. Function value vs. iteration (log-scale) for both methods on the same axes.
    """

    def run_example(self, func, x0, tol_obj, tol_param, max_iter_gd, max_iter_nt, 
                     contour_title, fv_title, xlims, ylims, levels):
        """
        Helper to run minimize() twice (GD and NT) on a given func,
        print results, then plot contour + paths and function values.
        
        Args:
            func: Example function handle (signature: f(x, hessian=False) → (f, grad, hess))
            x0: 1D numpy array, initial guess
            tol_obj: float, objective-tolerance (10⁻¹²)
            tol_param: float, parameter-tolerance (10⁻⁸)
            max_iter_gd: int, max iterations for GD
            max_iter_nt: int, max iterations for NT
            contour_title: str, title for contour plot
            fv_title: str, title for function-value plot
            xlims: tuple (x_min, x_max) for contour axes
            ylims: tuple (y_min, y_max) for contour axes
            levels: int or sequence of contour levels
        """
        # -- Run Gradient Descent --
        xg, fg, success_gd = minimize(f=func, x0=x0, algo="grad", obj_tol=tol_obj, param_tol=tol_param, max_iter=max_iter_gd)
        hist_gd = get_history()

        print(
            Fore.BLUE + "GD Final for " + func.__name__ + ":" + Style.RESET_ALL +
            f" x* = {xg},  f(x*) = {fg:.6e},  Success = {success_gd}"
        )

        # -- Run Newton's Method --
        xn, fn, success_nt = minimize(f=func, x0=x0, algo="newton", obj_tol=tol_obj, param_tol=tol_param, max_iter=max_iter_nt)
        hist_nt = get_history()

        print(
            Fore.MAGENTA + "NT Final for " + func.__name__ + ":" + Style.RESET_ALL +
            f" x* = {xn},  f(x*) = {fn:.6e},  Success = {success_nt}"
        )

        # -- Combine histories for plotting --
        # We want two distinct paths on the same contour: GD in red, NT in blue.
        # The plotting utility was designed to accept one history at a time,
        # so we’ll call it twice on the same axes manually.

        # Create a contour with both paths overlaid
        # First, prepare the grid limits using provided xlims, ylims, levels.
        # Then manually overlay the two histories.
        import matplotlib.pyplot as plt

        # Contour of the objective
        X_min, X_max = xlims
        Y_min, Y_max = ylims
        N_grid = 200
        X = np.linspace(X_min, X_max, N_grid)
        Y = np.linspace(Y_min, Y_max, N_grid)
        XX, YY = np.meshgrid(X, Y)
        ZZ = np.zeros_like(XX)

        # Evaluate func value on the grid
        for i in range(N_grid):
            for j in range(N_grid):
                pts = np.array([XX[i, j], YY[i, j]])
                val_ij, _, _ = func(pts, hessian=False)
                ZZ[i, j] = val_ij

        plt.figure()
        cs = plt.contour(XX, YY, ZZ, levels=levels, cmap="viridis")
        plt.clabel(cs, inline=True, fontsize=8)
        plt.title(contour_title)
        plt.xlabel("x₁")
        plt.ylabel("x₂")

        # Plot GD path in red
        locs_gd = np.array(hist_gd["locations"])
        plt.plot(
            locs_gd[:, 0],
            locs_gd[:, 1],
            'r.-',
            label="GD Path"
        )
        plt.scatter(
            locs_gd[0, 0], locs_gd[0, 1],
            c='green', marker='o', s=50,
            label="GD Start"
        )
        plt.scatter(
            locs_gd[-1, 0], locs_gd[-1, 1],
            c='red', marker='x', s=60,
            label="GD End"
        )

        # Plot NT path in blue
        locs_nt = np.array(hist_nt["locations"])
        plt.plot(
            locs_nt[:, 0],
            locs_nt[:, 1],
            'b.-',
            label="NT Path"
        )
        plt.scatter(
            locs_nt[0, 0], locs_nt[0, 1],
            c='cyan', marker='o', s=50,
            label="NT Start"
        )
        plt.scatter(
            locs_nt[-1, 0], locs_nt[-1, 1],
            c='blue', marker='x', s=60,
            label="NT End"
        )

        plt.legend()
        plt.show()

        # -- Plot function values vs iteration on same figure (log-scale) --
        plt.figure()
        iters_gd = list(range(len(hist_gd["values"])))
        iters_nt = list(range(len(hist_nt["values"])))
        plt.semilogy(
            iters_gd,
            hist_gd["values"],
            'r.-',
            label="GD f(x)"
        )
        plt.semilogy(
            iters_nt,
            hist_nt["values"],
            'b.-',
            label="NT f(x)"
        )
        plt.title(fv_title)
        plt.xlabel("Iteration")
        plt.ylabel("f(x) [log scale]")
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.show()

    def test_q_circle(self):
        """
        f(x) = x₁² + x₂²
        Initial x0 = [1, 1], tolerances = 1e-12, 1e-8, max_iter=100.
        Contour limits chosen to enclose the path observed in history.
        """
        func = q_circle
        x0 = np.array([1.0, 1.0])
        tol_obj = 1e-12
        tol_param = 1e-8
        max_iter = 100

        # After experimenting, the GD and NT paths remain within [-1.1, 1.1] in both dims.
        xlims = (-1.1, 1.1)
        ylims = (-1.1, 1.1)
        levels = np.linspace(0, 2, 30)

        self.run_example(
            func=func,
            x0=x0,
            tol_obj=tol_obj,
            tol_param=tol_param,
            max_iter_gd=max_iter,
            max_iter_nt=max_iter,
            contour_title="Contour of Quad Circle with GD & NT Paths",
            fv_title="Quad Circle: f(x) vs Iteration",
            xlims=xlims,
            ylims=ylims,
            levels=levels
        )

    def test_q_ellipse_axis(self):
        """
        f(x) = x₁² + 100 x₂²
        Initial x0 = [1, 1], tolerances = 1e-12, 1e-8, max_iter=100.
        Contour limits chosen to enclose the narrow, tall ellipse path. 
        """
        func = q_ellipse_axis
        x0 = np.array([1.0, 1.0])
        tol_obj = 1e-12
        tol_param = 1e-8
        max_iter = 100

        # After experimenting, GD moves mostly horizontally then slowly vertically.
        # Newton jumps near the bottom. Paths stay within [-1.1, 1.1] × [-0.15, 0.15].
        xlims = (-1.1, 1.1)
        ylims = (-0.15, 0.15)
        levels = np.logspace(-3, 2, 40)

        self.run_example(
            func=func,
            x0=x0,
            tol_obj=tol_obj,
            tol_param=tol_param,
            max_iter_gd=max_iter,
            max_iter_nt=max_iter,
            contour_title="Contour of Quad Ellipse (Axis Aligned) with Paths",
            fv_title="Quad Ellipse Axis: f(x) vs Iteration",
            xlims=xlims,
            ylims=ylims,
            levels=levels
        )

    def test_q_ellipse_rotated(self):
        """
        f(x) = xᵀ Q x, where Q is the rotated ellipse matrix.
        Initial x0 = [1, 1], tolerances = 1e-12, 1e-8, max_iter=100.
        Contour limits roughly [-1.2, 1.2] × [-1.2, 1.2] to see diagonal orientation.
        """
        func = q_ellipse_rotated
        x0 = np.array([1.0, 1.0])
        tol_obj = 1e-12
        tol_param = 1e-8
        max_iter = 100

        xlims = (-1.2, 1.2)
        ylims = (-1.2, 1.2)
        levels = np.logspace(-3, 2, 40)

        self.run_example(
            func=func,
            x0=x0,
            tol_obj=tol_obj,
            tol_param=tol_param,
            max_iter_gd=max_iter,
            max_iter_nt=max_iter,
            contour_title="Contour of Quad Ellipse (Rotated) with Paths",
            fv_title="Quad Ellipse Rotated: f(x) vs Iteration",
            xlims=xlims,
            ylims=ylims,
            levels=levels
        )

    def test_rosenbrock(self):
        """
        Rosenbrock: f(x) = 100(x2 - x1^2)^2 + (1 - x1)^2 (banana-shaped).
        Initial x0 = [-1, 2], tolerances = 1e-12, 1e-8.
        Use max_iter=10000 for GD (§7c :contentReference[oaicite:2]{index=2}) and max_iter=100 for NT.
        Contour limits chosen so that the narrow valley from [-1,2] → [1,1] is visible.
        """
        func = rosenbrock
        x0_gd = np.array([-1.0, 2.0])
        x0_nt = np.array([-1.0, 2.0])  # Same starting point
        tol_obj = 1e-12
        tol_param = 1e-8
        max_iter_gd = 10000
        max_iter_nt = 100

        # The Rosenbrock valley extends roughly from x in [-1.5, 2], y in [-0.5, 2.5].
        xlims = (-1.5, 2.0)
        ylims = (-0.5, 2.5)
        levels = np.logspace(-1, 3, 50)

        self.run_example(
            func=func,
            x0=x0_gd,
            tol_obj=tol_obj,
            tol_param=tol_param,
            max_iter_gd=max_iter_gd,
            max_iter_nt=max_iter_nt,
            contour_title="Rosenbrock Contour with GD & NT Paths",
            fv_title="Rosenbrock: f(x) vs Iteration",
            xlims=xlims,
            ylims=ylims,
            levels=levels
        )

    def test_linear_function(self):
        """
        Linear f(x) = [1,2]ᵀ · x. Initial x0 = [1,1], tolerances=1e-12, 1e-8, max_iter=100.
        Contour lines are straight lines; level sets might be too simple, but we choose a region that
        shows descent along the gradient direction. 
        """
        func = linear_func
        x0 = np.array([1.0, 1.0])
        tol_obj = 1e-12
        tol_param = 1e-8
        max_iter = 100

        # In a linear function, GD just moves straight down in direction −[1,2], NT is undefined Hessian (zero),
        # so it also effectively moves like GD. The path resides within [0, 1] × [−1, 1].
        xlims = (0.0, 1.0)
        ylims = (-1.0, 1.0)
        levels = np.linspace(-1.0, 3.0, 20)

        self.run_example(
            func=func,
            x0=x0,
            tol_obj=tol_obj,
            tol_param=tol_param,
            max_iter_gd=max_iter,
            max_iter_nt=max_iter,
            contour_title="Linear Function Contour with Descents",
            fv_title="Linear Function: f(x) vs Iteration",
            xlims=xlims,
            ylims=ylims,
            levels=levels
        )

    def test_smooth_corner(self):
        """
        f(x) = exp(x1+3x2−0.1) + exp(x1−3x2−0.1) + exp(−x1−0.1). Initial x0 = [1,1], tol=1e-12,1e-8, max_iter=100.
        Contours form smoothed corner triangles; choose limits to show the “corner” area near (0,0).
        """
        func = smooth_corner
        x0 = np.array([1.0, 1.0])
        tol_obj = 1e-12
        tol_param = 1e-8
        max_iter = 100

        # From experimentation, iterates stay in roughly [-1, 2] × [-1, 1].
        xlims = (-1.0, 2.0)
        ylims = (-1.0, 1.0)
        levels = np.logspace(-1, 2, 40)

        self.run_example(
            func=func,
            x0=x0,
            tol_obj=tol_obj,
            tol_param=tol_param,
            max_iter_gd=max_iter,
            max_iter_nt=max_iter,
            contour_title="Smooth Corner Contour with GD & NT Paths",
            fv_title="Smooth Corner: f(x) vs Iteration",
            xlims=xlims,
            ylims=ylims,
            levels=levels
        )


if __name__ == "__main__":
    unittest.main()
