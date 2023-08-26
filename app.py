import re
import math
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import sympy as sp
from scipy.interpolate import pade
from sympy.core.function import PoleError


def refine_x_values_around_singularities(x_values, threshold=1e-3):
    """Generate a refined set of x-values avoiding regions near tan singularities."""
    refined_x_values = []
    singularities = [
        (2 * n + 1) * np.pi / 2
        for n in range(int(-x_scale / np.pi), int(x_scale / np.pi))
    ]

    for i in range(len(x_values) - 1):
        refined_x_values.append(x_values[i])
        for singularity in singularities:
            # If the current x value and next x value bracket a singularity
            if x_values[i] < singularity < x_values[i + 1]:
                refined_x_values.append(singularity - threshold)
                refined_x_values.append(singularity + threshold)
    refined_x_values.append(x_values[-1])

    return np.array(refined_x_values)


def compute_pade(taylor_coeffs, n, m):
    """Compute Pade approximation with degree n for numerator and m for denominator."""
    # Convert symbolic coefficients to numerical values
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
    x = sp.symbols("x")
    allowed_functions = [sp.sin, sp.cos, sp.tan, sp.log, sp.exp]

    try:
        # Parse the expression
        parsed_expr = sp.sympify(expr, locals={"x": x})

        # Check for unwanted symbols
        for symbol in parsed_expr.free_symbols:
            if symbol != x:
                return False

        # Validate against allowed functions and operators
        atoms = parsed_expr.atoms(sp.Function, sp.Pow, sp.Mul, sp.Add)
        for atom in atoms:
            if isinstance(atom, sp.Function) and atom.func not in allowed_functions:
                return False

        return True
    except Exception as e:
        return False


def evaluate_expression(expr, x_val):
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


# Streamlit code
st.title("Compare Taylor and Pade Approximation")

# Taking user input for the function
func_str = st.text_input("Enter your function in terms of x (e.g., sin(x) or x**2):")

# User input for Taylor approximation order
taylor_order = st.sidebar.number_input(
    "Order of Approximation:", min_value=0, max_value=20, value=2, step=1
)


# Checkboxes to show Taylor's and Pade's approximation
show_taylor = st.sidebar.checkbox("Show Taylor's Approximation", value=True)
show_pade = st.sidebar.checkbox("Show Pade's Approximation", value=True)


# Ensure Taylor order is a valid integer
if not isinstance(taylor_order, int) or taylor_order < 0 or taylor_order > 20:
    st.error("Please enter a valid Taylor order (integer between 0 and 20).")
    taylor_order = None  # Invalidate the Taylor order to prevent further processing

# User inputs for x and y scales
x_scale = st.number_input("X-axis scale:", value=10.0)
y_scale = st.number_input("Y-axis scale:", value=10.0)

if func_str:
    original_x_values = np.linspace(-x_scale, x_scale, 400)
    x_values = refine_x_values_around_singularities(original_x_values)
    y_values = [evaluate_expression(func_str, val) for val in x_values]
    if any(isinstance(val, complex) for val in y_values):
        st.warning("The function produces complex values and cannot be plotted.")
    else:
        if isinstance(y_values[0], (int, float, np.number)):
            plt.figure(figsize=(8, 6))
            plt.plot(x_values, y_values, label=func_str)

            x_symbol = sp.symbols("x")
            func_sympy = sp.sympify(func_str)
            taylor_expansion = None
            try:
                taylor_expansion = func_sympy.series(
                    x_symbol, 0, taylor_order + 1
                ).removeO()
            except PoleError:
                st.error(
                    f"Cannot expand {func_str} around x=0. Consider a horizontal shift (e.g., use `x-a` instead of `x`)"
                )
            # If the Taylor expansion is a constant function around zero
            if taylor_expansion is not None and taylor_expansion.is_number:
                st.warning(
                    "The Taylor expansion for this function around x=0 is a constant. "
                    "Consider a horizontal shift (e.g., use `x-a` instead of `x`) "
                    "for a more meaningful Taylor approximation."
                )
            # If user selects Taylor's approximation
            if (
                show_taylor
                and taylor_order is not None
                and taylor_expansion is not None
                and not taylor_expansion.is_number
            ):
                # Display the Taylor's approximation expression on the sidebar
                st.sidebar.write(f"Taylor's Approximation (Order {taylor_order}):")
                st.sidebar.latex(sp.latex(taylor_expansion))

                taylor_func = sp.lambdify(x_symbol, taylor_expansion, "numpy")
                y_taylor_result = taylor_func(x_values)

                # Check if y_taylor_result is scalar and, if so, convert it to an array
                if np.isscalar(y_taylor_result):
                    y_taylor = np.full(x_values.shape, y_taylor_result)
                else:
                    y_taylor = y_taylor_result

                plt.plot(
                    x_values,
                    y_taylor,
                    label=f"Taylor Approximation (Order {taylor_order})",
                )

            # If user selects Pade approximation
            if (
                show_pade
                and taylor_order is not None
                and taylor_expansion is not None
                and not taylor_expansion.is_number
            ):
                taylor_coeffs = [
                    taylor_expansion.coeff(x_symbol, i) for i in range(taylor_order + 1)
                ]

                n = taylor_order // 2
                m = taylor_order - taylor_order // 2
                try:
                    p, q = compute_pade(taylor_coeffs, n, m)
                    if p is not None and q is not None:
                        # Create callable polynomial functions for p and q
                        p_func = np.poly1d(p.coeffs)
                        q_func = np.poly1d(q.coeffs)

                        # Evaluate the Pade approximation
                        y_pade = [p_func(val) / q_func(val) for val in x_values]
                        plt.plot(
                            x_values,
                            y_pade,
                            label=f"Pade Approximation (Numerator Order {n}, Denominator Order {m})",
                        )
                        pade_expression = generate_pade_expression(p, q)
                        st.sidebar.write(
                            f"Pade Approximation (Numerator Order {n}, Denominator Order {m}):"
                        )
                        st.sidebar.latex(pade_expression)
                except TypeError:
                    st.warning(
                        "Pade approximation resulted in complex coefficients and cannot be plotted."
                    )

            # Center the coordinate plane around (0, 0) and set the scale
            plt.xlim([-x_scale, x_scale])
            plt.ylim([-y_scale, y_scale])
            plt.axhline(0, color="black", linewidth=0.5)
            plt.axvline(0, color="black", linewidth=0.5)
            plt.title("y = f(x)")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)
        else:
            st.error(y_values[0])
