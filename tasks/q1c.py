"""
Q1(c): Equality-Constrained Newton's Method

This script demonstrates equality-constrained Newton's method for solving:
    minimize f0(x1, x2) = log(e^x1 + e^x2) + 0.5*(x1^2 + x2^2)
    subject to 0.5*x1 + x2 = 1

The optimization uses the KKT Newton step with backtracking line search
that maintains feasibility. The initial point is computed as the minimum-norm
feasible point: x0 = A^T(AA^T)^{-1}b.

The results include:
- Initial minimum-norm feasible point x0
- Optimal solution x_star
- Function value at the optimum
- Verification of constraint satisfaction
- Number of iterations
- Contour plot with constraint line saved as 'plots/q1c_contours.png'
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from scipy.optimize import minimize
from src.optimizer import newton_eq
from src.functions import f0, grad_f0, hess_f0
from src.plotting import plot_convergence_constrained


def verify_kkt_stationarity(x_star, A, grad_f, nu_star, tol=1e-6):
    """
    Verify KKT stationarity condition: ∇f(x*) + A^T ν* = 0

    Args:
        x_star: Optimal point
        A: Constraint matrix (p × n)
        grad_f: Gradient function
        nu_star: Dual variable from KKT solve
        tol: Tolerance for stationarity check

    Returns:
        stationarity_residual: ||∇f(x*) + A^T ν*||
        is_stationary: Whether KKT stationarity is satisfied
    """
    grad_at_star = grad_f(x_star)

    # Verify stationarity: ∇f(x*) + A^T ν* should be zero
    stationarity_residual = np.linalg.norm(grad_at_star + A.T @ nu_star)
    is_stationary = stationarity_residual < tol

    return stationarity_residual, is_stationary


def main():
    """Run equality-constrained Newton's method on Q1(c) and generate visualizations."""

    # Define constraint: 0.5*x1 + x2 = 1
    A = np.array([[0.5, 1.0]])
    b = np.array([1.0])

    # Compute minimum-norm feasible starting point: x0 = A^T(AA^T)^{-1}b
    AAT = A @ A.T  # This is a scalar for p=1
    AAT_inv = 1.0 / AAT  # Inverse for scalar
    x0 = A.T @ AAT_inv @ b
    x0 = x0.flatten()  # Ensure it's a 1D array

    # Verify initial feasibility
    constraint_residual_x0 = np.linalg.norm(A @ x0 - b)
    print("=" * 60)
    print("Q1(c): Equality-Constrained Newton's Method")
    print("=" * 60)
    print(f"\nConstraint: {A[0, 0]:.1f}*x1 + {A[0, 1]:.1f}*x2 = {b[0]:.1f}")
    print(f"\nInitial point (minimum-norm feasible):")
    print(f"x0 = {x0}")
    print(f"f0(x0) = {f0(x0):.6f}")
    print(f"Constraint residual: ||Ax0 - b|| = {constraint_residual_x0:.2e}")
    print("=" * 60)

    # Solve using equality-constrained Newton's method
    x_star, num_it, xs, nu_star = newton_eq(
        f=f0,
        grad=grad_f0,
        hess=hess_f0,
        A=A,
        b=b,
        x0=x0,
        alpha=0.1,
        beta=0.5,
        eps=1e-6
    )

    # Verify final constraint satisfaction
    constraint_residual_final = np.linalg.norm(A @ x_star - b)

    # Print results
    print(f"\nOptimization Results:")
    print("=" * 60)
    print(f"x_star:              {x_star}")
    print(f"f0(x_star):          {f0(x_star):.6f}")
    print(f"A @ x_star:          {(A @ x_star)[0]:.10f}")
    print(f"Target value (b):    {b[0]:.10f}")
    print(f"Constraint residual: {constraint_residual_final:.2e}")
    print(f"Number of iterations: {num_it}")
    print("=" * 60)

    # Show iterate history
    print(f"\nIterate History:")
    print("=" * 60)
    for i, x in enumerate(xs):
        print(f"Iteration {i}: x = {x}, f0(x) = {f0(x):.6f}, Ax = {(A @ x)[0]:.6f}")
    print("=" * 60)

    # Verify KKT stationarity conditions using the stored dual variable
    print(f"\nKKT Stationarity Verification:")
    print("=" * 60)
    stationarity_residual, is_stationary = verify_kkt_stationarity(
        x_star, A, grad_f0, nu_star, tol=1e-6
    )
    print(f"Dual variable (ν*):           {nu_star}")
    print(f"Gradient at x*:               {grad_f0(x_star)}")
    print(f"A^T ν*:                       {(A.T @ nu_star).flatten()}")
    print(f"∇f(x*) + A^T ν*:              {(grad_f0(x_star) + A.T @ nu_star).flatten()}")
    print(f"Stationarity residual:        {stationarity_residual:.2e}")
    print(f"KKT stationarity satisfied:   {is_stationary}")
    print("=" * 60)

    # Compare with SciPy reference optimizer
    print(f"\nSciPy Reference Optimizer:")
    print("=" * 60)

    # Define constraint in SciPy format
    def constraint_eq(x):
        return (A @ x - b)[0]

    constraints = {'type': 'eq', 'fun': constraint_eq}

    # Run SciPy optimizer
    scipy_result = minimize(
        fun=f0,
        x0=x0,
        method='SLSQP',
        jac=grad_f0,
        constraints=constraints,
        options={'ftol': 1e-9, 'disp': False}
    )

    x_scipy = scipy_result.x
    f_scipy = scipy_result.fun

    # Compute differences
    solution_diff = np.linalg.norm(x_star - x_scipy)
    objective_diff = abs(f0(x_star) - f_scipy)

    print(f"SciPy solution:              {x_scipy}")
    print(f"SciPy f0(x):                 {f_scipy:.6f}")
    print(f"SciPy constraint residual:   {abs(constraint_eq(x_scipy)):.2e}")
    print(f"SciPy iterations:            {scipy_result.nit}")
    print(f"\nComparison:")
    print(f"||x_custom - x_scipy||:      {solution_diff:.2e}")
    print(f"|f_custom - f_scipy|:        {objective_diff:.2e}")
    print("=" * 60)

    # Create contour plot with constraint line
    plot_convergence_constrained(
        f=f0,
        xs=xs,
        x0=x0,
        x_star=x_star,
        A=A,
        b=b,
        x_range=(-2, 3),
        y_range=(-2, 3),
        levels=40,
        title="Equality-Constrained Newton's Method (Q1c)",
        output_file='plots/q1c_contours.png'
    )


if __name__ == "__main__":
    main()
