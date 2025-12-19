"""
4M17 Coursework - Question 2(e): Hard-Margin SVM using Barrier Method

Train a hard-margin SVM on the provided data using the barrier method solver from Q1.

Primal formulation:
    min_{w,b} 1/2 ||w||_2^2
    s.t. y_i(w^T x_i + b) >= 1, i=1,...,m

Reformulated for barrier method (fi(x) <= 0):
    min_{w,b} 1/2 ||w||_2^2
    s.t. 1 - y_i(w^T x_i + b) <= 0, i=1,...,m
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from optimizer import newton_eq


def load_data(filepath):
    """Load training data from CSV."""
    df = pd.read_csv(filepath)
    X = df[['x_1', 'x_2']].values  # (m, 2)
    y = df['y'].values  # (m,)
    return X, y


def barrier_method_svm(X, y, t0=1.0, mu=10.0, eps_outer=1e-6,
                       alpha=0.1, beta=0.5, eps_inner=1e-6, max_outer_it=100):
    """
    Solve hard-margin SVM using barrier method.

    Variables: x = [w1, w2, b]^T where w = [w1, w2]^T

    Objective: f0(x) = 1/2 * (w1^2 + w2^2)
    Constraints: fi(x) = 1 - y_i(w^T x_i + b) <= 0, i=1,...,m

    Returns:
        x_star: optimal [w1, w2, b]
        iteration_count: number of outer iterations
        history: list of iterates at each outer iteration
    """
    m = len(y)
    n = 3  # dimension: [w1, w2, b]

    # Objective: f0(x) = 1/2 * ||w||^2
    def f0(x):
        w = x[:2]
        return 0.5 * np.dot(w, w)

    def grad0(x):
        g = np.zeros(3)
        g[:2] = x[:2]  # gradient w.r.t. w
        # gradient w.r.t. b is 0
        return g

    def hess0(x):
        H = np.zeros((3, 3))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        # Hessian w.r.t. b is 0
        return H

    # Inequality constraints: fi(x) = 1 - y_i(w^T x_i + b) <= 0
    def make_constraint_funcs(i):
        """Create constraint functions for i-th data point."""
        xi = X[i]
        yi = y[i]

        def fi(x):
            w = x[:2]
            b = x[2]
            return 1.0 - yi * (np.dot(w, xi) + b)

        def grad_fi(x):
            g = np.zeros(3)
            g[:2] = -yi * xi  # gradient w.r.t. w
            g[2] = -yi        # gradient w.r.t. b
            return g

        def hess_fi(x):
            # Second derivative is zero (linear constraint)
            return np.zeros((3, 3))

        return fi, grad_fi, hess_fi

    # Create all constraint functions
    fis = []
    grads = []
    hesses = []
    for i in range(m):
        fi, grad_fi, hess_fi = make_constraint_funcs(i)
        fis.append(fi)
        grads.append(grad_fi)
        hesses.append(hess_fi)

    # No equality constraints
    A = np.zeros((0, n))
    b_eq = np.zeros(0)

    # Find strictly feasible starting point
    x0 = find_feasible_start(X, y)

    print(f"Starting point: w = [{x0[0]:.4f}, {x0[1]:.4f}], b = {x0[2]:.4f}")
    print(f"Initial objective: f0(x0) = {f0(x0):.6f}")

    # Verify strict feasibility
    max_constraint = max(fi(x0) for fi in fis)
    print(f"Max constraint value: {max_constraint:.6e} (must be < 0)")
    assert max_constraint < 0, "Starting point is not strictly feasible!"

    # Barrier method
    t = t0
    x = x0.copy()
    history = [x.copy()]

    print("\n" + "="*70)
    print(f"{'Iter':<6} {'t':<12} {'m/t':<12} {'f0(x)':<12} {'||w||':<12}")
    print("="*70)

    for outer_it in range(max_outer_it):
        # Construct barrier objective: φ_t(x) = t*f0(x) - Σ log(-fi(x))
        def phi_t(x):
            val = t * f0(x)
            for fi in fis:
                fi_val = fi(x)
                if fi_val >= 0:
                    return np.inf
                val -= np.log(-fi_val)
            return val

        def grad_phi_t(x):
            g = t * grad0(x)
            for i, fi in enumerate(fis):
                fi_val = fi(x)
                if fi_val >= 0:
                    return np.full(n, np.nan)
                g -= grads[i](x) / fi_val
            return g

        def hess_phi_t(x):
            H = t * hess0(x)
            for i, fi in enumerate(fis):
                fi_val = fi(x)
                if fi_val >= 0:
                    return np.full((n, n), np.nan)
                grad_fi_val = grads[i](x)
                H -= hesses[i](x) / fi_val
                H += np.outer(grad_fi_val, grad_fi_val) / (fi_val ** 2)
            return H

        # Centering step: minimize φ_t(x) using Newton's method
        x_new, it_inner, xs_inner, _ = newton_eq(
            phi_t, grad_phi_t, hess_phi_t, A, b_eq, x,
            alpha=alpha, beta=beta, eps=eps_inner, max_it=50
        )

        x = x_new
        history.append(x.copy())

        # Compute current values
        w_norm = np.linalg.norm(x[:2])
        gap = m / t

        print(f"{outer_it+1:<6} {t:<12.2e} {gap:<12.2e} {f0(x):<12.6f} {w_norm:<12.6f}")

        # Check stopping criterion: m/t <= eps_outer
        if gap <= eps_outer:
            print("="*70)
            print(f"Converged: m/t = {gap:.2e} <= {eps_outer:.2e}")
            break

        # Increase t
        t *= mu
    else:
        print("="*70)
        print("Warning: Maximum iterations reached")

    return x, outer_it + 1, history


def find_feasible_start(X, y):
    """
    Find a strictly feasible starting point for the SVM problem.

    Strategy: Use a simple perceptron-like approach to find a separating hyperplane,
    then scale it to ensure strict feasibility with margin.
    """
    m = len(y)

    # Start with a simple linear separator using least squares
    # We want to find w, b such that y_i(w^T x_i + b) > 1

    # Simple heuristic: use class centroids
    pos_idx = y == 1
    neg_idx = y == -1

    centroid_pos = np.mean(X[pos_idx], axis=0)
    centroid_neg = np.mean(X[neg_idx], axis=0)

    # Normal vector points from negative to positive centroid
    w_init = centroid_pos - centroid_neg
    w_init = w_init / (np.linalg.norm(w_init) + 1e-10)

    # Find b such that the hyperplane passes between centroids
    b_init = -0.5 * np.dot(w_init, centroid_pos + centroid_neg)

    # Scale to ensure strict feasibility: y_i(w^T x_i + b) >= 1 + slack
    x_init = np.array([w_init[0], w_init[1], b_init])

    margins = y * (X @ x_init[:2] + x_init[2])
    min_margin = np.min(margins)

    if min_margin <= 1.0:
        # Scale w and b to ensure all margins > 1
        scale = 1.5 / min_margin if min_margin > 0 else 2.0
        x_init[:2] *= scale
        x_init[2] *= scale

    # Verify strict feasibility again
    margins = y * (X @ x_init[:2] + x_init[2])
    assert np.all(margins > 1.0), f"Failed to find feasible start: min margin = {np.min(margins)}"

    return x_init


def scipy_reference_solver(X, y):
    """
    Solve the same SVM problem using scipy's SLSQP optimizer for verification.
    """
    def objective(x):
        w = x[:2]
        return 0.5 * np.dot(w, w)

    def jac(x):
        g = np.zeros(3)
        g[:2] = x[:2]
        return g

    # Constraints: y_i(w^T x_i + b) >= 1
    constraints = []
    for i in range(len(y)):
        def make_constraint(i=i):
            return {
                'type': 'ineq',
                'fun': lambda x, i=i: y[i] * (np.dot(x[:2], X[i]) + x[2]) - 1.0
            }
        constraints.append(make_constraint())

    # Use the same starting point
    x0 = find_feasible_start(X, y)

    result = minimize(
        objective,
        x0,
        method='SLSQP',
        jac=jac,
        constraints=constraints,
        options={'ftol': 1e-9, 'disp': False}
    )

    return result.x, result.fun, result.success


def classify_point(w, b, x):
    """Classify a point using the trained SVM."""
    score = np.dot(w, x) + b
    prediction = 1 if score >= 0 else -1
    return prediction, score


def plot_svm_results(X, y, w, b, test_points, save_path='plots/q2e_svm.png'):
    """
    Plot the training data, decision boundary, and margins with professional styling
    matching the existing plot aesthetics.
    """
    # Set professional plot styling (matching existing plots)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['figure.titlesize'] = 12
    plt.rcParams['lines.linewidth'] = 1.5

    fig, ax = plt.subplots(figsize=(8, 6))

    # Compute plot range
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x_vals = np.linspace(x_min, x_max, 500)

    # w1*x1 + w2*x2 + b = 0  =>  x2 = -(w1*x1 + b)/w2
    y_decision = -(w[0] * x_vals + b) / w[1]

    # Plot margin boundaries: w^T x + b = ±1
    y_margin_pos = -(w[0] * x_vals + b - 1) / w[1]
    y_margin_neg = -(w[0] * x_vals + b + 1) / w[1]

    # Shade the margin region with very subtle color
    ax.fill_between(x_vals, y_margin_neg, y_margin_pos,
                    alpha=0.08, color='purple', zorder=1)

    # Plot margin lines - dashed purple lines
    ax.plot(x_vals, y_margin_pos, color='#9B59B6', linewidth=2,
            linestyle='--', alpha=0.8, zorder=8,
            label=r'$\mathbf{w}^\top\mathbf{x} + b = \pm 1$')
    ax.plot(x_vals, y_margin_neg, color='#9B59B6', linewidth=2,
            linestyle='--', alpha=0.8, zorder=8)

    # Plot decision boundary - solid black line
    ax.plot(x_vals, y_decision, color='black', linewidth=2.5,
            linestyle='-', alpha=0.9, zorder=9,
            label=r'$\mathbf{w}^\top\mathbf{x} + b = 0$')

    # Plot training points with clean styling
    pos_idx = y == 1
    neg_idx = y == -1

    ax.scatter(X[pos_idx, 0], X[pos_idx, 1], c='#E74C3C', marker='o', s=100,
               label='Class $+1$', edgecolors='darkred', linewidth=1.2,
               zorder=10, alpha=0.85)
    ax.scatter(X[neg_idx, 0], X[neg_idx, 1], c='#3498DB', marker='s', s=100,
               label='Class $-1$', edgecolors='darkblue', linewidth=1.2,
               zorder=10, alpha=0.85)

    # Plot test points with distinct markers
    test_colors = {'(-2, 1)': None, '(3, 0)': None}
    for idx, (pt, (x_test, y_test)) in enumerate(test_points.items()):
        pred, score = classify_point(w, b, np.array([x_test, y_test]))
        if pred == 1:
            marker = '^'
            color = '#C0392B'  # Darker red
            edge_color = 'darkred'
        else:
            marker = 'v'
            color = '#2980B9'  # Darker blue
            edge_color = 'darkblue'

        ax.scatter(x_test, y_test, c=color, marker=marker, s=180,
                  edgecolors=edge_color, linewidth=1.8,
                  label=f'Test {pt}', zorder=15, alpha=0.9)

    # Axis labels with LaTeX formatting
    ax.set_xlabel(r'$x_1$', fontsize=11)
    ax.set_ylabel(r'$x_2$', fontsize=11)
    ax.set_title('Hard-Margin SVM', fontsize=12, fontweight='bold', pad=10)

    # Set axis limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Clean legend without shadow or fancy boxes - fully opaque, on top
    legend = ax.legend(loc='upper right', framealpha=1, edgecolor='black', fancybox=False, facecolor='white')
    legend.set_zorder(100)

    # Add subtle grid matching the style of other plots
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5, color='gray')

    # White background
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # Tight layout
    plt.tight_layout()

    # Save the figure with high DPI
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nPlot saved to {save_path}")
    plt.close(fig)


def main():
    print("="*70)
    print("4M17 Coursework - Question 2(e): Hard-Margin SVM")
    print("="*70)

    # Load data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'part_e_training_data.csv')
    X, y = load_data(data_path)
    m = len(y)

    print(f"\nLoaded {m} training samples")
    print(f"  Positive class (+1): {np.sum(y == 1)} samples")
    print(f"  Negative class (-1): {np.sum(y == -1)} samples")

    # Solve using barrier method
    print("\n" + "="*70)
    print("Solving with Barrier Method")
    print("="*70)

    x_star, n_iter, history = barrier_method_svm(
        X, y,
        t0=1.0,
        mu=10.0,
        eps_outer=1e-6,
        alpha=0.1,
        beta=0.5,
        eps_inner=1e-6,
        max_outer_it=100
    )

    w_star = x_star[:2]
    b_star = x_star[2]
    margin = 1.0 / np.linalg.norm(w_star)

    print("\n" + "="*70)
    print("Barrier Method Results")
    print("="*70)
    print(f"Optimal w = [{w_star[0]:.6f}, {w_star[1]:.6f}]")
    print(f"Optimal b = {b_star:.6f}")
    print(f"||w||_2   = {np.linalg.norm(w_star):.6f}")
    print(f"Margin r  = 1/||w||_2 = {margin:.6f}")
    print(f"Objective = 1/2||w||^2 = {0.5 * np.dot(w_star, w_star):.6f}")
    print(f"Outer iterations: {n_iter}")

    # Verify constraints
    margins = y * (X @ w_star + b_star)
    print(f"\nConstraint verification:")
    print(f"  Min margin: {np.min(margins):.6f} (should be >= 1.0)")
    print(f"  Max margin: {np.max(margins):.6f}")
    print(f"  All constraints satisfied: {np.all(margins >= 1.0 - 1e-6)}")

    # Solve using scipy for verification
    print("\n" + "="*70)
    print("Scipy Reference Solver (SLSQP)")
    print("="*70)

    x_scipy, obj_scipy, success_scipy = scipy_reference_solver(X, y)
    w_scipy = x_scipy[:2]
    b_scipy = x_scipy[2]
    margin_scipy = 1.0 / np.linalg.norm(w_scipy)

    print(f"Optimal w = [{w_scipy[0]:.6f}, {w_scipy[1]:.6f}]")
    print(f"Optimal b = {b_scipy:.6f}")
    print(f"||w||_2   = {np.linalg.norm(w_scipy):.6f}")
    print(f"Margin r  = 1/||w||_2 = {margin_scipy:.6f}")
    print(f"Objective = {obj_scipy:.6f}")
    print(f"Success: {success_scipy}")

    # Compare solutions
    print("\n" + "="*70)
    print("Comparison: Barrier Method vs Scipy")
    print("="*70)
    print(f"||w_barrier - w_scipy||   = {np.linalg.norm(w_star - w_scipy):.2e}")
    print(f"|b_barrier - b_scipy|     = {abs(b_star - b_scipy):.2e}")
    print(f"|obj_barrier - obj_scipy| = {abs(0.5*np.dot(w_star, w_star) - obj_scipy):.2e}")

    relative_error = np.linalg.norm(x_star - x_scipy) / np.linalg.norm(x_scipy)
    print(f"Relative solution error   = {relative_error:.2e}")

    if relative_error < 1e-4:
        print("\n✓ Solutions match! Barrier method implementation is correct.")
    else:
        print("\n✗ Warning: Solutions differ significantly.")

    # Classify test points
    print("\n" + "="*70)
    print("Test Point Classification")
    print("="*70)

    test_points = {
        '(-2, 1)': (-2.0, 1.0),
        '(3, 0)': (3.0, 0.0)
    }

    for name, (x1, x2) in test_points.items():
        x_test = np.array([x1, x2])
        pred, score = classify_point(w_star, b_star, x_test)
        print(f"Point {name}:")
        print(f"  Decision function: {score:.6f}")
        print(f"  Predicted class: {pred:+d}")

    # Plot results
    print("\n" + "="*70)
    print("Generating plot...")
    print("="*70)

    plot_svm_results(X, y, w_star, b_star, test_points)

    print("\n" + "="*70)
    print("Question 2(e) Complete!")
    print("="*70)


if __name__ == '__main__':
    main()
