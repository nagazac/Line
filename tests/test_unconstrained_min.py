import unittest
import numpy as np
from colorama import Fore, Style, init

from src.unconstrained_min import minimize, get_history
from src.utils import plot_contour_two, plot_function_values_two
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
    Use utils.plot_contour_two and utils.plot_function_values_two to visualize.
    """

    def run_example(
        self,
        func,
        x0,
        tol_obj,
        tol_param,
        max_iter_gd,
        max_iter_nt,
        contour_title,
        fv_title,
        xlims,
        ylims,
        levels,
        formula_str
    ):
        """
        Run minimize() twice (GD and NT) on func, print results, then plot:

          (1) Contour with both paths overlaid
          (2) f(x) vs iteration (log scale) with both histories.

        Args:
            func: The test function (returns f, grad, hess).
            x0: initial guess as numpy array
            tol_obj: objective tolerance (1e-12)
            tol_param: parameter/gradient tolerance (1e-8)
            max_iter_gd: max iterations for GD
            max_iter_nt: max iterations for NT
            contour_title: base title (we’ll append the formula)
            fv_title: base title for function-value plot
            xlims: (x_min, x_max)
            ylims: (y_min, y_max)
            levels: contour levels (array or int)
            formula_str: string representing the analytic formula, to embed in the title
        """
        # Run Gradient Descent
        iterg, xg, fg, success_gd = minimize(
            f=func,
            x0=x0,
            algo="grad",
            obj_tol=tol_obj,
            param_tol=tol_param,
            max_iter=max_iter_gd
        )
        hist_gd = get_history()

        print(
            Fore.BLUE + f"GD Final for {func.__name__}:" + Style.RESET_ALL +
            f"Iteration number = {iterg}," +
            f" x* = {xg},  f(x*) = {fg:.6e},  Success = {success_gd}"
        )

        # Run Newton’s Method
        itern, xn, fn, success_nt = minimize(
            f=func,
            x0=x0,
            algo="newton",
            obj_tol=tol_obj,
            param_tol=tol_param,
            max_iter=max_iter_nt
        )
        hist_nt = get_history()

        print(
            Fore.MAGENTA + f"NT Final for {func.__name__}:" + Style.RESET_ALL +
            f" Iteration number = {itern}," +
            f" x* = {xn},  f(x*) = {fn:.6e},  Success = {success_nt}"
        )

        # Build titles including the formula
        full_contour_title = f"{contour_title}\n({formula_str})"
        full_fv_title = f"{fv_title}\n({formula_str})"

        # Plot both histories on the same contour
        plot_contour_two(
            func,
            x_range=xlims,
            y_range=ylims,
            hist_gd=hist_gd,
            hist_nt=hist_nt,
            levels=levels,
            title=full_contour_title,
            gdl_color="red",
            ntl_color="blue"
        )

        # Plot f(x) vs iteration for both methods
        plot_function_values_two(
            hist_gd=hist_gd,
            hist_nt=hist_nt,
            title=full_fv_title,
            gdl_color="red",
            ntl_color="blue"
        )

    def test_q_circle(self):
        """
        f(x) = x₁² + x₂²
        Initial x0 = [1, 1], tol=1e-12,1e-8, max_iter=100.
        """
        func = q_circle
        x0 = np.array([1.0, 1.0])
        tol_obj = 1e-12
        tol_param = 1e-8
        max_iter = 100

        xlims = (-1.1, 1.1)
        ylims = (-1.1, 1.1)
        levels = np.linspace(0, 2, 30)
        formula = "f(x) = x₁² + x₂²"

        self.run_example(
            func=func,
            x0=x0,
            tol_obj=tol_obj,
            tol_param=tol_param,
            max_iter_gd=max_iter,
            max_iter_nt=max_iter,
            contour_title="Quad Circle Contour with GD & NT Paths",
            fv_title="Quad Circle: f(x) vs Iteration",
            xlims=xlims,
            ylims=ylims,
            levels=levels,
            formula_str=formula
        )

    def test_q_ellipse_axis(self):
        """
        f(x) = x₁² + 100 x₂²
        Initial x0 = [1, 1], tol=1e-12,1e-8, max_iter=100.
        """
        func = q_ellipse_axis
        x0 = np.array([1.0, 1.0])
        tol_obj = 1e-12
        tol_param = 1e-8
        max_iter = 100

        xlims = (-1.1, 1.1)
        ylims = (-0.15, 0.15)
        levels = np.logspace(-3, 2, 40)
        formula = "f(x) = x₁² + 100 x₂²"

        self.run_example(
            func=func,
            x0=x0,
            tol_obj=tol_obj,
            tol_param=tol_param,
            max_iter_gd=max_iter,
            max_iter_nt=max_iter,
            contour_title="Quad Ellipse (Axis Aligned) Contour with Paths",
            fv_title="Quad Ellipse Axis: f(x) vs Iteration",
            xlims=xlims,
            ylims=ylims,
            levels=levels,
            formula_str=formula
        )

    def test_q_ellipse_rotated(self):
        """
        f(x) = xᵀ Q x, Q = Rᵀ diag(100,1) R, R = [[√3/2, -1/2],[1/2, √3/2]]
        Initial x0 = [1, 1], tol=1e-12,1e-8, max_iter=100.
        """
        func = q_ellipse_rotated
        x0 = np.array([1.0, 1.0])
        tol_obj = 1e-12
        tol_param = 1e-8
        max_iter = 100

        xlims = (-1.2, 1.2)
        ylims = (-1.2, 1.2)
        levels = np.logspace(-3, 2, 40)
        formula = ("f(x) = xᵀ Q x, Q = Rᵀ diag(100,1) R\n"
                   "R = [[√3/2, -1/2], [1/2, √3/2]]")

        self.run_example(
            func=func,
            x0=x0,
            tol_obj=tol_obj,
            tol_param=tol_param,
            max_iter_gd=max_iter,
            max_iter_nt=max_iter,
            contour_title="Quad Ellipse (Rotated) Contour with Paths",
            fv_title="Quad Ellipse Rotated: f(x) vs Iteration",
            xlims=xlims,
            ylims=ylims,
            levels=levels,
            formula_str=formula
        )

    def test_rosenbrock(self):
        """
        f(x) = 100 (x₂ – x₁²)² + (1 – x₁)²
        Initial x0 = [-1, 2], tol=1e-12,1e-8. GD max_iter=10000, NT max_iter=100.
        """
        func = rosenbrock
        x0 = np.array([-1.0, 2.0])
        tol_obj = 1e-12
        tol_param = 1e-8
        max_iter_gd = 10000
        max_iter_nt = 100

        xlims = (-1.5, 2.0)
        ylims = (-0.5, 2.5)
        levels = np.logspace(-1, 3, 50)
        formula = "f(x) = 100 (x₂ – x₁²)² + (1 – x₁)²"

        self.run_example(
            func=func,
            x0=x0,
            tol_obj=tol_obj,
            tol_param=tol_param,
            max_iter_gd=max_iter_gd,
            max_iter_nt=max_iter_nt,
            contour_title="Rosenbrock Contour with GD & NT Paths",
            fv_title="Rosenbrock: f(x) vs Iteration",
            xlims=xlims,
            ylims=ylims,
            levels=levels,
            formula_str=formula
        )

    def test_linear_function(self):
        """
        f(x) = [-6, 8]ᵀ x
        Initial x0 = [1, 1], tol=1e-12,1e-8, max_iter=100.
        """
        func = linear_func
        x0 = np.array([1.0, 1.0])
        tol_obj = 1e-12
        tol_param = 1e-8
        max_iter = 100

        xlims = (0.0, 1.0)
        ylims = (-1.0, 1.0)
        levels = np.linspace(-1.0, 3.0, 20)
        formula = "f(x) = [-6, 8]ᵀ x = -6·x₁ + 8·x₂"

        self.run_example(
            func=func,
            x0=x0,
            tol_obj=tol_obj,
            tol_param=tol_param,
            max_iter_gd=max_iter,
            max_iter_nt=max_iter,
            contour_title="Linear Function Contour with GD & NT Paths",
            fv_title="Linear Function: f(x) vs Iteration",
            xlims=xlims,
            ylims=ylims,
            levels=levels,
            formula_str=formula
        )

    def test_smooth_corner(self):
        """
        f(x) = e^{x₁ + 3 x₂ – 0.1} + e^{x₁ – 3 x₂ – 0.1} + e^{–x₁ – 0.1}
        Initial x0 = [1, 1], tol=1e-12,1e-8, max_iter=100.
        """
        func = smooth_corner
        x0 = np.array([1.0, 1.0])
        tol_obj = 1e-12
        tol_param = 1e-8
        max_iter = 100

        xlims = (-1.0, 2.0)
        ylims = (-1.0, 1.0)
        levels = np.logspace(-1, 2, 40)
        formula = ("f(x) = e^{x₁ + 3 x₂ – 0.1} + e^{x₁ – 3 x₂ – 0.1} + e^{–x₁ – 0.1}")

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
            levels=levels,
            formula_str=formula
        )


if __name__ == "__main__":
    unittest.main()
