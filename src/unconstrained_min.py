import numpy as np
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Global dictionary to store optimization history
# Keys: "locations" → list of numpy arrays; "values" → list of scalars
history = {"locations": [], "values": []}

def get_history():
    """Retrieve the optimization history.
        Dictionary with keys:
        locations" (list of numpy arrays) and "values" (list of scalars).
    """
    return history

def minimize(f, x0, method="grad", obj_tol=1e-12, param_tol=1e-8, max_iter=100):
    """
    Minimize a function using either gradient descent or Newton's method.

    Args:
        f (callable): Signature f(x, hessian=False) → (f_val, grad, hess_or_None).
                       If hessian=True, it must return the Hessian matrix as well.
        x0 (array-like): Initial guess.
        method (str): "grad" for gradient descent, or "newton" for Newton's method.
        obj_tol (float): Convergence tolerance on |f_new – f_old|.
        param_tol (float): Convergence tolerance on ||grad|| or ||x_new – x_old||.
        max_iter (int): Maximum number of iterations.

    Returns:
        x_final (ndarray): The final iterate.
        f_final (float): The function value at x_final.
        success (bool): True if converged, False if reached max_iter without meeting tolerances.
    """
    global history
    history = {"locations": [], "values": []}

    # Convert x0 to a flat numpy array
    x = np.array(x0, dtype=float).flatten()

    # Evaluate f, grad, hessian (only need hessian if method="newton")
    if method == "newton":
        f_val, grad, hess = f(x, hessian=True)
    else:
        f_val, grad, _ = f(x, hessian=False)
        hess = None

    # Initialize history
    history["locations"].append(x.copy())
    history["values"].append(f_val)

    # Print iteration 0
    print(Fore.YELLOW + f"Iteration 0: x = {x}, f(x) = {f_val:.6e}" + Style.RESET_ALL)

    iteration = 0
    success = False

    if method == "grad":
        # --------------------------
        # Gradient Descent Loop
        # --------------------------
        while iteration < max_iter:
            # Descent direction is -grad
            d = -grad

            # Backtracking to find alpha that satisfies Armijo: f(x+αd) ≤ f_val + c α (gradᵀ d)
            alpha = backtrack(f, x, f_val, grad, d, c=0.01, b=0.5, a=1.0)

            # Update x
            x_new = x + alpha * d

            # Evaluate f, grad at the new x (no Hessian needed)
            f_new, grad_new, _ = f(x_new, hessian=False)

            # Record history
            history["locations"].append(x_new.copy())
            history["values"].append(f_new)

            iteration += 1
            print(Fore.YELLOW + f"Iteration {iteration}: x = {x_new}, f(x) = {f_new:.6e}" + Style.RESET_ALL)

            # Check convergence:
            if abs(f_new - f_val) < obj_tol or np.linalg.norm(x_new - x) < param_tol:
                success = True
                print(Fore.GREEN + f"Converged in {iteration} iterations." + Style.RESET_ALL)
                return x_new, f_new, success

            # Otherwise prepare for next iteration
            x = x_new
            f_val = f_new
            grad = grad_new

        # If we exit the loop, we hit max_iter
        return x, f_val, success

    elif method == "newton":
        # --------------------------
        # Newton's Method Loop
        # --------------------------
        while iteration < max_iter:
            if hess is None:
                raise ValueError("Hessian must be supplied for Newton's method.")

            # Solve for Newton direction: H d = -grad
            try:
                d = -np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                # If Hessian is singular or badly conditioned, fallback to steepest descent
                d = -grad

            # Backtracking with directional derivative gradᵀ d
            alpha = backtrack(f, x, f_val, grad, d, c=0.01, b=0.5, a=1.0)

            # Update x
            x_new = x + alpha * d

            # Evaluate f, grad, hess at x_new
            f_new, grad_new, hess_new = f(x_new, hessian=True)

            # Record history
            history["locations"].append(x_new.copy())
            history["values"].append(f_new)

            iteration += 1
            print(Fore.YELLOW + f"Iteration {iteration}: x = {x_new}, f(x) = {f_new:.6e}" + Style.RESET_ALL)

            # Convergence checks:
            if abs(f_new - f_val) < obj_tol or np.linalg.norm(x_new - x) < param_tol:
                success = True
                print(Fore.GREEN + f"Converged in {iteration} iterations." + Style.RESET_ALL)
                return x_new, f_new, success

            # Prepare for next iteration
            x = x_new
            f_val = f_new
            grad = grad_new
            hess = hess_new

        # If we exit the loop, we hit max_iter
        return x, f_val, success

    else:
        raise ValueError(f"Unknown method '{method}' (must be 'grad' or 'newton')")


def backtrack(f, x, fx, grad, d, c=0.01, b=0.5, a=1.0):
    """
    Backtracking line search (Armijo condition).
    Args:
        f (callable): signature f(x, hessian=False) → (f_val, grad, _).
        x (ndarray): current point.
        fx (float): f(x).
        grad (ndarray): gradient at x, i.e. ∇f(x).
        d (ndarray): descent direction (should satisfy grad.T @ d < 0).
        c (float): Armijo parameter, e.g. 0.01.
        b (float): shrink factor, e.g. 0.5.
        a (float): initial α, e.g. 1.0.
    Returns:
        α (float): a step length satisfying
            f(x + α d) ≤ f(x) + c α (gradᵀ d).
    """
    alpha = a
    dir_d = np.dot(grad, d)  # This must be < 0 for a true descent direction.
    if dir_d >= 0:
        # Not a descent direction – force a tiny step
        return 1e-8

    while alpha > 1e-10:
        x_trial = x + alpha * d
        f_trial, _, _ = f(x_trial, hessian=False)
        # Armijo condition:
        if f_trial <= fx + c * alpha * dir_d:
            return alpha
        alpha *= b

    # If α got very small without satisfying Armijo, just return the small α
    return alpha