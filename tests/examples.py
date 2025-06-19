import numpy as np

def q_circle(x, hessian=False):
    """
    f(x) = xᵀ I x = x₁² + x₂²
    """
    x = np.array(x, dtype=float).flatten()
    f = x[0]**2 + x[1]**2
    g = 2 * x
    if hessian:
        h = 2 * np.eye(2)
    else:
        h = None
    return f, g, h

def q_ellipse_axis(x, hessian=False):
    """
    f(x) = xᵀ diag([1, 100]) x
         = x₁² + 100 x₂²
    """
    x = np.array(x, dtype=float).flatten()
    f = x[0]**2 + 100.0 * x[1]**2
    g = np.array([2.0 * x[0], 200.0 * x[1]])
    if hessian:
        h = np.diag([2.0, 200.0])
    else:
        h = None
    return f, g, h

def q_ellipse_rotated(x, hessian=False):
    """
    f(x) = xᵀ Q x, where
    R = [ [√3/2, -0.5],
          [ 0.5, √3/2] ]
    D = diag([100, 1])
    Q = Rᵀ D R
    """
    x = np.array(x, dtype=float).flatten()
    # Rotation matrix R
    R = np.array([[np.sqrt(3)/2, -0.5],
                  [0.5,            np.sqrt(3)/2]])
    D = np.diag([100.0, 1.0])
    Q = R.T @ D @ R

    f = x.T @ Q @ x
    g = 2.0 * Q @ x
    if hessian:
        h = 2.0 * Q
    else:
        h = None
    return f, g, h

def rosenbrock(x, hessian=False):
    """
    f(x) = 100 (x₂ - x₁²)² + (1 - x₁)²
    """
    x = np.array(x, dtype=float).flatten()
    x1, x2 = x[0], x[1]
    f = 100.0 * (x2 - x1**2)**2 + (1.0 - x1)**2

    # Gradient
    df_dx1 = -400.0 * x1 * (x2 - x1**2) - 2.0 * (1.0 - x1)
    df_dx2 = 200.0 * (x2 - x1**2)
    g = np.array([df_dx1, df_dx2])

    if hessian:
        # Hessian entries
        d2f_dx1_dx1 = 1200.0 * x1**2 - 400.0 * x2 + 2.0
        d2f_dx1_dx2 = -400.0 * x1
        d2f_dx2_dx1 = -400.0 * x1
        d2f_dx2_dx2 = 200.0
        h = np.array([[d2f_dx1_dx1, d2f_dx1_dx2],
                      [d2f_dx2_dx1, d2f_dx2_dx2]])
    else:
        h = None

    return f, g, h

def linear_func(x, hessian=False):
    """
    f(x) = aᵀ x, with a chosen as [-6, 8]ᵀ (non-zero vector).
    """
    x = np.array(x, dtype=float).flatten()
    a = np.array([-6.0, 8.0])
    f = a.T @ x
    g = a.copy()
    if hessian:
        h = np.zeros((2, 2))
    else:
        h = None
    return f, g, h


def smooth_corner(x, hessian=False):
    """
    f(x₁, x₂) = exp(x₁ + 3 x₂ - 0.1) + exp(x₁ - 3 x₂ - 0.1) + exp(-x₁ - 0.1)
    """
    x = np.array(x, dtype=float).flatten()
    x1, x2 = x[0], x[1]
    t1 = np.exp(x1 + 3.0*x2 - 0.1)
    t2 = np.exp(x1 - 3.0*x2 - 0.1)
    t3 = np.exp(-x1 - 0.1)

    f = t1 + t2 + t3

    # Gradient
    df_dx1 = t1 + t2 - t3
    df_dx2 = 3.0 * t1 - 3.0 * t2
    g = np.array([df_dx1, df_dx2])

    if hessian:
        d2f_dx1_dx1 = t1 + t2 + t3
        d2f_dx1_dx2 = 3.0 * t1 - 3.0 * t2
        d2f_dx2_dx1 = 3.0 * t1 - 3.0 * t2
        d2f_dx2_dx2 = 9.0 * t1 + 9.0 * t2
        h = np.array([[d2f_dx1_dx1, d2f_dx1_dx2],
                      [d2f_dx2_dx1, d2f_dx2_dx2]])
    else:
        h = None

    return f, g, h


# Quadratic Program (QP):
def qp_obj(x):
    # f(x) = x^2 + y^2 + (z+1)^2
    f = x[0]**2 + x[1]**2 + (x[2] + 1)**2
    grad = np.array([2*x[0], 2*x[1], 2*(x[2] + 1)])
    hess = 2 * np.eye(3)
    return f, grad, hess

def qp_constraints():
    # x >= 0, y >= 0, z >= 0 captured as -x <= 0, -y <= 0, -z <= 0
    return [
        lambda x: (-x[0], np.array([-1.0,  0.0,  0.0]), np.zeros((3,3))),
        lambda x: (-x[1], np.array([ 0.0, -1.0,  0.0]), np.zeros((3,3))),
        lambda x: (-x[2], np.array([ 0.0,  0.0, -1.0]), np.zeros((3,3)))
    ]

# Equality constraint: x + y + z = 1
def get_qp_data():
    A_qp = np.array([[1.0, 1.0, 1.0]])
    b_qp = np.array([1.0])
    x0_qp = np.array([0.1, 0.2, 0.7])  # strictly feasible start
    return A_qp, b_qp, x0_qp

# Linear Program (LP): maximize x+y over polygon
def lp_obj(x):
    # max x+y <=> min -(x+y)
    f = -(x[0] + x[1])
    grad = np.array([-1.0, -1.0])
    hess = np.zeros((2,2))
    return f, grad, hess

def lp_constraints():
    return [
        # y ≥ –x + 1  ⇔  –(y + x – 1) ≤ 0
        lambda x: (-(x[1] + x[0] - 1.0), np.array([-1.0, -1.0]), np.zeros((2,2))),
        # y ≤ 1       ⇔  y – 1 ≤ 0
        lambda x: ( x[1] - 1.0,            np.array([ 0.0,  1.0]), np.zeros((2,2))),
        # x ≤ 2       ⇔  x – 2 ≤ 0
        lambda x: ( x[0] - 2.0,            np.array([ 1.0,  0.0]), np.zeros((2,2))),
        # y ≥ 0       ⇔  –y ≤ 0
        lambda x: (-x[1],                  np.array([ 0.0, -1.0]), np.zeros((2,2)))
    ]

# No equality constraints for LP

def get_lp_data():
    A_lp = np.zeros((0,2))
    b_lp = np.zeros((0,))
    x0_lp = np.array([0.5, 0.75])   # strictly feasible start
    return A_lp, b_lp, x0_lp


