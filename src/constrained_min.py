import numpy as np

def interior_pt(func, ineq_constraints, A, b, x0,
                mu=10.0, tol=1e-8, bt_alpha=0.25, bt_beta=0.5):
    """
    Interior point method (log barrier + constrained Newton).
    Solves: min f0(x) s.t. fi(x) <= 0, Ax = b

    - func(x) -> (f0, grad0, hess0)
    - ineq_constraints: list of fi(x)->(fi, gradi, hessi)
    - A @ x = b  (A: p×n, b: p,)
    - x0: strictly feasible starting point
    Returns (x_star, central_path, obj_vals)
    """
    m = len(ineq_constraints)
    n = x0.size
    x = x0.copy()
    t = 1.0
    count = 0
    
    central_path = []
    obj_vals     = []

    def barrier_obj(x, t):
        # compute f_bar, grad_bar, hess_bar
        f0, g0, H0 = func(x)
        phi = 0.0
        g_phi = np.zeros(n)
        H_phi = np.zeros((n,n))

        for fi in ineq_constraints:
            v, gi, Hi = fi(x)
            phi   += -np.log(-v)
            g_phi += -1.0/v * gi
            H_phi += (1.0/v**2)*np.outer(gi,gi) + (-1.0/v)*Hi

        f_bar  = t*f0 + phi
        g_bar  = t*g0 + g_phi
        H_bar  = t*H0 + H_phi
        return f_bar, g_bar, H_bar

    def solve_kkt(H, A, g):
        # Solve [H  A^T; A 0] [p; w] = [g; 0]
        p_eq = A.shape[0]
        KKT  = np.block([
            [H,      A.T],
            [A, np.zeros((p_eq,p_eq))]
        ])
        rhs = np.concatenate([g, np.zeros(p_eq)])
        sol = np.linalg.solve(KKT, rhs)
        p   = sol[:n]
        w   = sol[n:]
        return p, w

    # outer loop: stop when m/t < tol
    while m/t > tol:
        # --- solve the equality-constrained problem for this t (interior Newton) ---
        for _ in range(100):
            count += 1
            f_bar, g_bar, H_bar = barrier_obj(x, t)
            # constrained Newton step
            p, _ = solve_kkt(H_bar, A, -g_bar)
            # Newton decrement
            lambda2 = -g_bar.dot(p)
            if lambda2/2.0 <= tol:
                break

            # backtracking line search
            alpha = 1.0
            while True:
                x_new = x + alpha * p
                # maintain domain fi(x_new) < 0
                if all(fi(x_new)[0] < 0 for fi in ineq_constraints):
                    f_new, _, _ = barrier_obj(x_new, t)
                    if f_new <= f_bar + bt_alpha*alpha*g_bar.dot(p):
                        break
                alpha *= bt_beta

            x = x + alpha*p

        # record the final point for this t
        central_path.append(x.copy())
        obj_vals.append(func(x)[0])
        t *= mu

    # Print Solutions 
    print(f"Converged in {count} iterations.")
    print(f"Final point: {x}")
    print(f"Objective value: {func(x)[0]}")
    # Print constraint values at the final point
    print("Final constraints values at x:")
    
    for fi,i in zip(ineq_constraints, range(m)):
        v, _, _ = fi(x)
        print(f"Constraint {i+1} value: {v}")
    

    return x, np.array(central_path), np.array(obj_vals)
