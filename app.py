import re
import math
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import sympy as sp
from scipy.interpolate import pade

# =========================
# Utility Functions
# =========================


def refine_x_values_around_singularities(x_values, threshold=1e-3):
    """
    Generate a refined set of x-values avoiding regions near tan singularities.
    """
    x_scale = max(x_values)
    singularities = [
        (2 * n + 1) * np.pi / 2
        for n in range(int(-x_scale / np.pi), int(x_scale / np.pi))
    ]

    refined_x_values = []
    for i in range(len(x_values) - 1):
        refined_x_values.append(x_values[i])
        for singularity in singularities:
            if x_values[i] < singularity < x_values[i + 1]:
                refined_x_values.extend(
                    [singularity - threshold, singularity + threshold]
                )

    refined_x_values.append(x_values[-1])
    return np.array(refined_x_values)


def compute_pade_approximation(taylor_coeffs, n, m):
    """Compute Pade approximation with degree n for numerator and m for denominator."""
    numerical_coeffs = [float(coeff.evalf()) for coeff in taylor_coeffs]
    try:
        p, q = pade(numerical_coeffs, m)
        return p, q
    except np.linalg.LinAlgError:
        st.error(
            "Failed to compute the Pade approximation due to numerical instability. Consider changing Order of Approximation."
        )
        return None, None


def generate_pade_expression(p, q):
    """Generate Pade approximation expression from polynomial coefficients."""

    def format_term(coeff, i):
        if i == 0:
            return f"{coeff:.1f}"
        elif i == 1:
            return f"{coeff:.1f}x"
        else:
            return f"{coeff:.1f}x^{i}"

    num_expr = " ".join(
        [
            f"{'+' if coeff >= 0 else '-'} {format_term(abs(coeff), i)}"
            for i, coeff in enumerate(p.coeffs[::-1])
            if coeff != 0
        ]
    ).lstrip("+ ")

    den_expr = " ".join(
        [
            f"{'+' if coeff >= 0 else '-'} {format_term(abs(coeff), i)}"
            for i, coeff in enumerate(q.coeffs[::-1])
            if coeff != 0
        ]
    ).lstrip("+ ")

    return f"({num_expr}) / ({den_expr})"


def is_valid_expression(expr):
    """Validate the given expression based on allowed functions."""
    x = sp.symbols("x")
    allowed_functions = [sp.sin, sp.cos, sp.tan, sp.log, sp.exp]

    try:
        parsed_expr = sp.sympify(expr, locals={"x": x})
        if any(symbol != x for symbol in parsed_expr.free_symbols):
            return False

        atoms = parsed_expr.atoms(sp.Function, sp.Pow, sp.Mul, sp.Add)
        return all(
            (isinstance(atom, sp.Function) and atom.func in allowed_functions)
            or not isinstance(atom, sp.Function)
            for atom in atoms
        )

    except Exception:
        return False


def evaluate_expression(expr, x_val):
    """Evaluate the given expression at a particular x-value."""
    x = sp.symbols("x")
    if not is_valid_expression(expr):
        return "Invalid function."

    try:
        f = sp.lambdify(
            x,
            expr,
            modules=[
                {
                    "sin": np.sin,
                    "cos": np.cos,
                    "tan": np.tan,
                    "log": np.log,
                    "exp": np.exp,
                },
                "numpy",
            ],
        )
        return f(x_val)
    except Exception as e:
        return str(e)


# =========================
# Streamlit Application
# =========================


def main():
    st.title("Compare Taylor and Pade Approximation")

    func_str = st.text_input(
        "Enter your function in terms of x (e.g., sin(x) or x**2):"
    )
    taylor_order = st.sidebar.number_input(
        "Order of Approximation:", min_value=0, max_value=20, value=2, step=1
    )
    show_taylor = st.sidebar.checkbox("Show Taylor's Approximation", value=True)
    show_pade = st.sidebar.checkbox("Show Pade's Approximation", value=True)

    # Validations
    if not isinstance(taylor_order, int) or not (0 <= taylor_order <= 20):
        st.error("Please enter a valid Taylor order (integer between 0 and 20).")
        return

    x_scale = st.number_input("X-axis scale:", value=10.0)
    y_scale = st.number_input("Y-axis scale:", value=10.0)

    if func_str:
        original_x_values = np.linspace(-x_scale, x_scale, 400)
        x_values = refine_x_values_around_singularities(original_x_values)
        y_values = [evaluate_expression(func_str, val) for val in x_values]

        if isinstance(y_values[0], (int, float, np.number)):
            # Displaying the plot
            plot_approximations(
                x_values,
                y_values,
                func_str,
                taylor_order,
                show_taylor,
                show_pade,
                x_scale,
                y_scale,
            )
        else:
            st.error(y_values[0])


def plot_approximations(
    x_values, y_values, func_str, taylor_order, show_taylor, show_pade, x_scale, y_scale
):
    """Plot original function, Taylor, and Pade approximations."""
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, label=func_str)

    x_symbol = sp.symbols("x")
    func_sympy = sp.sympify(func_str)
    taylor_expansion = func_sympy.series(x_symbol, 0, taylor_order + 1).removeO()

    if show_taylor:
        st.sidebar.write(f"Taylor's Approximation (Order {taylor_order}):")
        st.sidebar.latex(sp.latex(taylor_expansion))

        taylor_func = sp.lambdify(x_symbol, taylor_expansion, "numpy")
        y_taylor = taylor_func(x_values)
        plt.plot(
            x_values, y_taylor, label=f"Taylor Approximation (Order {taylor_order})"
        )

    if show_pade:
        taylor_coeffs = [
            taylor_expansion.coeff(x_symbol, i) for i in range(taylor_order + 1)
        ]
        n = taylor_order // 2
        m = taylor_order - taylor_order // 2
        p, q = compute_pade_approximation(taylor_coeffs, n, m)

        if p is not None:
            st.sidebar.write(
                f"Pade's Approximation (Numerator Order {n}, Denominator Order {m}:"
            )
            st.sidebar.latex(generate_pade_expression(p, q))

            pade_func = lambda x: np.polyval(p.coeffs, x) / np.polyval(q.coeffs, x)
            y_pade = pade_func(x_values)
            plt.plot(
                x_values,
                y_pade,
                label=f"Pade's Approximation (Numerator Order {n}, Denominator Order {m}:",
            )

    plt.title("Function and its Approximations")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim([-y_scale, y_scale])
    plt.xlim([-x_scale, x_scale])
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    st.pyplot(plt)


if __name__ == "__main__":
    main()
