"""
Unconstrained Newton's Method

This script demonstrates Newton's method for minimizing:
    f0(x1, x2) = log(e^x1 + e^x2) + 0.5*(x1^2 + x2^2)

The optimization is performed from the initial point x0 = [-1.5, 2.5]
with parameters eps=1e-6, alpha=0.1, beta=0.5.

The results include:
- Optimal solution x_star
- Function value at the optimum
- Number of iterations
- Contour plot with convergence path saved as 'plots/newton_contours.png'
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from scipy.optimize import minimize
from src.optimizer import newton
from src.functions import f0, grad_f0, hess_f0
from src.plotting import plot_convergence


def main():
    """Run Newton's method on f0 and generate visualizations."""

    # Starting point
    x0 = np.array([-1.5, 2.5])

    # Solve using Newton's method with specified parameters
    x_star, num_it, xs = newton(x0, f0, grad_f0, hess_f0, eps=1e-6, alpha=0.1, beta=0.5)

    # Get reference solution from scipy.optimize
    result_scipy = minimize(
        f0,
        x0,
        method='Newton-CG',
        jac=grad_f0,
        hess=hess_f0,
        options={'xtol': 1e-6, 'disp': False}
    )

    # Print results
    print("=" * 60)
    print("Newton's Method Results")
    print("=" * 60)
    print(f"x_star:              {x_star}")
    print(f"f0(x_star):          {f0(x_star):.6f}")
    print(f"Number of iterations: {num_it}")
    print("=" * 60)
    print("\nReference Solution (scipy.optimize):")
    print("=" * 60)
    print(f"x_star (scipy):      {result_scipy.x}")
    print(f"f0(x_star):          {result_scipy.fun:.6f}")
    print(f"Number of iterations: {result_scipy.nit}")
    print(f"Success:             {result_scipy.success}")
    print("=" * 60)
    print("\nComparison:")
    print("=" * 60)
    print(f"Solution difference: {np.linalg.norm(x_star - result_scipy.x):.2e}")
    print(f"Objective difference: {abs(f0(x_star) - result_scipy.fun):.2e}")
    print("=" * 60)

    # Create contour plot
    plot_convergence(
        f=f0,
        xs=xs,
        x0=x0,
        x_star=x_star,
        x_range=(-2, 3),
        y_range=(-2, 3),
        levels=40,
        title="Newton's Method Convergence Path",
        output_file='plots/newton_contours.png'
    )


if __name__ == "__main__":
    main()
