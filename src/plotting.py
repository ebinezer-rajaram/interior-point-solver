"""
Visualization utilities for optimization algorithms.

This module provides functions to visualize optimization convergence,
including contour plots with iterate paths.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


# Set professional plot styling
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['lines.linewidth'] = 1.5


def plot_convergence(f, xs, x0, x_star, x_range=(-2, 3), y_range=(-2, 3),
                     levels=40, title='Optimization Convergence Path',
                     output_file='plots/convergence.png'):
    """
    Create a professional contour plot showing the optimization convergence path.

    Parameters
    ----------
    f : callable
        Objective function; f(x) -> scalar.
    xs : list of ndarray
        Full iterate history from the optimization algorithm.
    x0 : ndarray, shape (2,)
        Initial starting point.
    x_star : ndarray, shape (2,)
        Final optimal point.
    x_range : tuple, optional
        Range for x-axis (x1 values).
    y_range : tuple, optional
        Range for y-axis (x2 values).
    levels : int, optional
        Number of contour levels to plot.
    title : str, optional
        Title for the plot.
    output_file : str, optional
        Filename to save the plot (relative to project root).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Define high-resolution grid for contour plot
    x1_range = np.linspace(x_range[0], x_range[1], 500)
    x2_range = np.linspace(y_range[0], y_range[1], 500)
    X1, X2 = np.meshgrid(x1_range, x2_range)

    # Compute f over the grid
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = f(np.array([X1[i, j], X2[i, j]]))

    # Create the plot with professional styling
    fig, ax = plt.subplots(figsize=(12, 9))

    # Use filled contours for better visualization
    contourf = ax.contourf(X1, X2, Z, levels=levels, cmap='RdYlBu_r', alpha=0.8)
    contours = ax.contour(X1, X2, Z, levels=levels, colors='black',
                          alpha=0.2, linewidths=0.5)

    # Add colorbar with better formatting
    cbar = plt.colorbar(contourf, ax=ax, label=r'$f_0(x_1, x_2)$', pad=0.02)
    cbar.ax.yaxis.set_label_coords(3.5, 0.5)

    # Overlay iterate path with enhanced visibility
    xs_array = np.array(xs)

    # Draw path line
    ax.plot(xs_array[:, 0], xs_array[:, 1],
            color='white', linewidth=3.5, alpha=0.8, zorder=10)
    ax.plot(xs_array[:, 0], xs_array[:, 1],
            color='#FF6B35', linewidth=2.5, linestyle='-',
            alpha=1.0, zorder=11, label='Optimization path')

    # Mark iterate points
    ax.scatter(xs_array[:, 0], xs_array[:, 1],
              s=80, c='white', edgecolors='#FF6B35',
              linewidths=2, zorder=12, alpha=0.9)

    # Add iteration numbers to points
    for i, (x1, x2) in enumerate(xs_array):
        ax.annotate(f'{i}', xy=(x1, x2), xytext=(5, 5),
                   textcoords='offset points', fontsize=9,
                   color='#2C3E50', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='none', alpha=0.7), zorder=13)

    # Mark starting point with distinct style
    ax.scatter(x0[0], x0[1], s=200, marker='o',
              c='#2ECC71', edgecolors='white', linewidths=2.5,
              label='Starting point', zorder=14)

    # Mark optimal point with distinct style
    ax.scatter(x_star[0], x_star[1], s=300, marker='*',
              c='#E74C3C', edgecolors='white', linewidths=2,
              label='Optimal solution', zorder=15)

    # Axis labels with LaTeX formatting
    ax.set_xlabel(r'$x_1$', fontsize=13)
    ax.set_ylabel(r'$x_2$', fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)

    # Enhanced legend
    ax.legend(loc='upper right', framealpha=0.95, shadow=True,
             fancybox=True, edgecolor='gray', borderpad=1)

    # Set aspect ratio
    ax.set_aspect('equal', adjustable='box')

    # Add subtle grid
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

    # Tight layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nContour plot saved as '{output_file}'")

    plt.close(fig)

    return fig


def plot_convergence_constrained(f, xs, x0, x_star, A, b, x_range=(-2, 3), y_range=(-2, 3),
                                   levels=40, title='Constrained Optimization Convergence Path',
                                   output_file='plots/convergence_constrained.png'):
    """
    Create a contour plot showing constrained optimization convergence with constraint line.

    Parameters
    ----------
    f : callable
        Objective function; f(x) -> scalar.
    xs : list of ndarray
        Full iterate history from the optimization algorithm.
    x0 : ndarray, shape (2,)
        Initial starting point.
    x_star : ndarray, shape (2,)
        Final optimal point.
    A : ndarray, shape (p, n)
        Constraint matrix.
    b : ndarray, shape (p,)
        Constraint right-hand side.
    x_range : tuple, optional
        Range for x-axis (x1 values).
    y_range : tuple, optional
        Range for y-axis (x2 values).
    levels : int, optional
        Number of contour levels to plot.
    title : str, optional
        Title for the plot.
    output_file : str, optional
        Filename to save the plot (relative to project root).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Define high-resolution grid for contour plot
    x1_range = np.linspace(x_range[0], x_range[1], 500)
    x2_range = np.linspace(y_range[0], y_range[1], 500)
    X1, X2 = np.meshgrid(x1_range, x2_range)

    # Compute f over the grid
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = f(np.array([X1[i, j], X2[i, j]]))

    # Create the plot with professional styling
    fig, ax = plt.subplots(figsize=(12, 9))

    # Use filled contours for better visualization
    contourf = ax.contourf(X1, X2, Z, levels=levels, cmap='RdYlBu_r', alpha=0.8)
    contours = ax.contour(X1, X2, Z, levels=levels, colors='black',
                          alpha=0.2, linewidths=0.5)

    # Add colorbar with better formatting
    cbar = plt.colorbar(contourf, ax=ax, label=r'$f_0(x_1, x_2)$', pad=0.02)
    cbar.ax.yaxis.set_label_coords(3.5, 0.5)

    # Plot constraint line: Ax = b
    # For 2D case: a1*x1 + a2*x2 = b  =>  x2 = (b - a1*x1) / a2
    if A.shape[0] == 1 and A.shape[1] == 2:  # Single constraint in 2D
        a1, a2 = A[0]
        b_val = b[0]

        # Generate points along the constraint line
        x1_constraint = np.linspace(x_range[0], x_range[1], 500)
        x2_constraint = (b_val - a1 * x1_constraint) / a2

        # Plot constraint line
        ax.plot(x1_constraint, x2_constraint,
                color='#9B59B6', linewidth=3.5, linestyle='--',
                alpha=0.9, zorder=9, label=f'Constraint: {a1:.1f}$x_1$ + {a2:.1f}$x_2$ = {b_val:.1f}')

    # Overlay iterate path with enhanced visibility
    xs_array = np.array(xs)

    # Draw path line
    ax.plot(xs_array[:, 0], xs_array[:, 1],
            color='white', linewidth=3.5, alpha=0.8, zorder=10)
    ax.plot(xs_array[:, 0], xs_array[:, 1],
            color='#FF6B35', linewidth=2.5, linestyle='-',
            alpha=1.0, zorder=11, label='Optimization path')

    # Mark iterate points
    ax.scatter(xs_array[:, 0], xs_array[:, 1],
              s=80, c='white', edgecolors='#FF6B35',
              linewidths=2, zorder=12, alpha=0.9)

    # Add iteration numbers to points
    for i, (x1, x2) in enumerate(xs_array):
        ax.annotate(f'{i}', xy=(x1, x2), xytext=(5, 5),
                   textcoords='offset points', fontsize=9,
                   color='#2C3E50', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='none', alpha=0.7), zorder=13)

    # Mark starting point with distinct style
    ax.scatter(x0[0], x0[1], s=200, marker='o',
              c='#2ECC71', edgecolors='white', linewidths=2.5,
              label='Starting point', zorder=14)

    # Mark optimal point with distinct style
    ax.scatter(x_star[0], x_star[1], s=300, marker='*',
              c='#E74C3C', edgecolors='white', linewidths=2,
              label='Optimal solution', zorder=15)

    # Axis labels with LaTeX formatting
    ax.set_xlabel(r'$x_1$', fontsize=13)
    ax.set_ylabel(r'$x_2$', fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)

    # Enhanced legend
    ax.legend(loc='upper right', framealpha=0.95, shadow=True,
             fancybox=True, edgecolor='gray', borderpad=1)

    # Set aspect ratio
    ax.set_aspect('equal', adjustable='box')

    # Add subtle grid
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

    # Tight layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nConstrained contour plot saved as '{output_file}'")

    plt.close(fig)

    return fig
