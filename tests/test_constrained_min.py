import unittest
import numpy as np
from colorama import Fore, Style, init

from src.constrained_min import interior_pt
from src.utils import plot_objective_vs_outer, plot_qp_region_and_path, plot_lp_region_and_path
from tests.examples import (
    qp_obj,
    qp_constraints,
    get_qp_data,
    lp_obj,
    lp_constraints,
    get_lp_data
)

# Initialize Colorama
init(autoreset=True)


class TestConstrainedMinimization(unittest.TestCase):
    """Test cases for constrained minimization using interior point method."""

    def test_quadratic_programming(self):
        """Test quadratic programming with interior point method."""
        print(Fore.GREEN + "Running quadratic programming test...")
        A, b, x0 = get_qp_data()
        ineq_constraints = qp_constraints()
        x_star, central_path, obj_vals = interior_pt(
            qp_obj, ineq_constraints, A, b, x0,
            mu=10.0, tol=1e-8, bt_alpha=0.25, bt_beta=0.5
        )
        
        print(Fore.GREEN + "Quadratic programming test completed.")
        # Plot the results
        plot_qp_region_and_path(central_path, x_star)
        plot_objective_vs_outer(obj_vals, title='QP Objective vs Outer Iteration')
        return x_star, central_path, obj_vals

    def test_linear_programming(self):
        """Test linear programming with interior point method."""
        print(Fore.GREEN + "Running linear programming test...")
        A, b, x0 = get_lp_data()
        ineq_constraints = lp_constraints()
        x_star, central_path, obj_vals = interior_pt(
            lp_obj, ineq_constraints, A, b, x0,
            mu=10.0, tol=1e-8, bt_alpha=0.25, bt_beta=0.5
        )
        
        print(Fore.GREEN + "Linear programming test completed.")
        # Plot the results
        plot_lp_region_and_path(central_path, x_star)
        plot_objective_vs_outer(obj_vals, title='LP Objective vs Outer Iteration')
        return x_star, central_path, obj_vals