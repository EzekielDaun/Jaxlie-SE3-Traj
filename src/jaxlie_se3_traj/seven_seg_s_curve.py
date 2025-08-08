from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Float


@jax.jit
def __7_seg_params_from_m(m: float, rho: float):
    """Given plateau slope m and shape ratio rho, compute 7-seg parameters.
    Returns (alpha, beta, gamma, J, t1).
    Assumptions: normalized total time = 1, total displacement = 1 (center-symmetric).
    VALID INPUTS (must be enforced by caller): 1 < m <= 2, 0 < rho <= 0.5.
    """
    # implied left-transition duration from the unit-time/unit-displacement closure
    t1 = 1.0 - 1.0 / m  # in (0, 0.5]
    alpha = rho * t1
    beta = t1 - 2.0 * alpha
    gamma = 1.0 - 2.0 * t1
    # jerk chosen to achieve plateau slope m with zero accel at t1
    J = m / (alpha * (alpha + beta))
    return alpha, beta, gamma, J, t1


@jax.jit
def __7_seg_s_curve_left_jerk(x: Float, m: float, rho: float):
    """Left half [0,0.5] of a symmetric jerk-limited 7-segment S-curve.
    Primary control is the plateau slope m; rho splits the left transition.
    """
    alpha, beta, _, J, t1_total = __7_seg_params_from_m(m, rho)
    a1 = J * alpha
    v1 = 0.5 * J * alpha**2
    y1 = (1.0 / 6.0) * J * alpha**3

    # segment-2 precomputes
    v2 = v1 + a1 * beta
    y2_base = y1 + v1 * beta + 0.5 * a1 * beta**2

    def seg1(x):
        # 0 <= x <= alpha (jerk +J)
        return (1.0 / 6.0) * J * x**3

    def seg2(x):
        # alpha <= x <= alpha+beta (const acceleration a1)
        t = x - alpha
        return y1 + v1 * t + 0.5 * a1 * t**2

    def seg3(x):
        # alpha+beta <= x <= t1_total (jerk -J to zero accel)
        tau = x - (alpha + beta)
        return y2_base + v2 * tau + 0.5 * a1 * tau**2 - (1.0 / 6.0) * J * tau**3

    # position at x=t1_total
    y3 = y2_base + v2 * alpha + 0.5 * a1 * alpha**2 - (1.0 / 6.0) * J * alpha**3

    def seg4(x):
        # t1_total <= x <= 0.5, constant slope m
        return y3 + m * (x - t1_total)

    # piecewise select
    y = jnp.where(
        x <= alpha,
        seg1(x),
        jnp.where(
            x <= alpha + beta, seg2(x), jnp.where(x <= t1_total, seg3(x), seg4(x))
        ),
    )
    return y


@jax.jit
def __7_seg_s_curve_jerk_sym_impl(x: Float, m: float, rho: float):
    xr = jnp.where(x <= 0.5, x, 1.0 - x)
    left_val = __7_seg_s_curve_left_jerk(xr, m, rho)
    return jnp.where(x <= 0.5, left_val, 1.0 - left_val)


@partial(jax.jit, static_argnums=(1, 2))
def seven_seg_s_curve(x: Float, m: float, rho: float):
    """Validated wrapper for jerk7 S-curve.
    Requirements: 1 < m <= 2 and 0 < rho <= 0.5. Raises ValueError if violated.
    """
    if not (m > 1.0 and m <= 2.0):
        raise ValueError(f"m must satisfy 1 < m <= 2 (got {m})")
    if not (rho > 0.0 and rho <= 0.5):
        raise ValueError(f"rho must satisfy 0 < rho <= 0.5 (got {rho})")
    return __7_seg_s_curve_jerk_sym_impl(x, m, rho)


if __name__ == "__main__":
    # Jerk-limited 7-segment S-curve only (monotone by construction)
    m = 1.25  # plateau speed (normalized). Feasible range for unit time/length: [1, 2]
    rho = (
        1.0 / 3.0
    )  # shape split in the left transition: alpha=rho*t1, beta=(1-2*rho)*t1
    x = jnp.linspace(0.0, 1.0, 1600)

    # curve
    y = seven_seg_s_curve(x, m, rho)

    # derivatives up to 5th
    def _vmapped_derivatives(f, x, m, rho):
        g1 = jax.grad(lambda z: f(z, m, rho))
        g2 = jax.grad(g1)
        g3 = jax.grad(g2)
        g4 = jax.grad(g3)
        g5 = jax.grad(g4)
        return (
            jax.vmap(g1)(x),
            jax.vmap(g2)(x),
            jax.vmap(g3)(x),
            jax.vmap(g4)(x),
            jax.vmap(g5)(x),
        )

    dy, d2y, d3y, d4y, d5y = _vmapped_derivatives(seven_seg_s_curve, x, m, rho)

    # Diagnostics: compute t1, alpha, beta, gamma, J from (m, rho)
    t1 = float(1.0 - 1.0 / m)
    alpha = float(rho * t1)
    beta = float(t1 - 2.0 * alpha)
    gamma = float(1.0 - 2.0 * t1)
    J = float(m / (alpha * (alpha + beta)))
    print(
        f"[jerk7] t1={t1:.6g}, alpha={alpha:.6g}, beta={beta:.6g}, gamma={gamma:.6g}, J={J:.6g}, rho={float(rho):.6g}"
    )
    print(f"[jerk7] plateau window: [{t1:.6g}, {1.0 - t1:.6g}]")

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 18))

    # Row 1: y(x)
    ax = plt.subplot(6, 1, 1)
    ax.plot(x, y, label="Jerk7(m)")
    ax.set_title("Jerk7(m) S-curve")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.legend()

    # Row 2: dy/dx
    ax = plt.subplot(6, 1, 2)
    ax.plot(x, dy, label="dy/dx")
    ax.set_title("First derivative")
    ax.set_xlabel("x")
    ax.set_ylabel("dy/dx")
    ax.grid(True)
    ax.legend()

    # Row 3: d2y/dx2 (acceleration)
    ax = plt.subplot(6, 1, 3)
    ax.plot(x, d2y, label="d2y/dx2")
    ax.set_title("Second derivative (accel)")
    ax.set_xlabel("x")
    ax.set_ylabel("d2y/dx2")
    ax.grid(True)
    ax.legend()

    # Row 4: d3y/dx3 (jerk)
    ax = plt.subplot(6, 1, 4)
    ax.plot(x, d3y, label="d3y/dx3")
    ax.set_title("Third derivative (jerk)")
    ax.set_xlabel("x")
    ax.set_ylabel("d3y/dx3")
    ax.grid(True)
    ax.legend()

    # Row 5: d4y/dx4
    ax = plt.subplot(6, 1, 5)
    ax.plot(x, d4y, label="d4y/dx4")
    ax.set_title("Fourth derivative")
    ax.set_xlabel("x")
    ax.set_ylabel("d4y/dx4")
    ax.grid(True)
    ax.legend()

    # Row 6: d5y/dx5
    ax = plt.subplot(6, 1, 6)
    ax.plot(x, d5y, label="d5y/dx5")
    ax.set_title("Fifth derivative")
    ax.set_xlabel("x")
    ax.set_ylabel("d5y/dx5")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()
