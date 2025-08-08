from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxlie import SE3
from jaxtyping import Float

from example_utils import render_trajectory
from jaxlie_se3_traj import (
    MultiplyDirection,
    SE3DeltaTiltTorsionFullSpace,
    SE3Trajectory,
)
from jaxlie_se3_traj.bump_method import linear

if __name__ == "__main__":
    common_factor = 5e3
    params = {
        "bump_function": linear,
        "tilt_angle_amplitude": jnp.pi / 2,
        "torsion_angle_amplitude": jnp.deg2rad(2.5),
        "tilt_angle_frequency": common_factor,
        "torsion_angle_frequency": common_factor * 2**0.5,
        "tilt_direction_wrap_frequency": common_factor * 3**0.5,
    }

    trajectory = SE3Trajectory.from_deltas(
        (
            SE3DeltaTiltTorsionFullSpace(
                direction=MultiplyDirection.RIGHT, duration=40.0, **params
            ),
        )
    )

    def fn(
        s: float,
        bump_function: Callable[[Float], Float],
        tilt_direction_wrap_frequency: float,
        tilt_angle_amplitude: float,
        tilt_angle_frequency: float,
        torsion_angle_amplitude: float,
        torsion_angle_frequency: float,
    ):
        tilt_direction_angle = tilt_direction_wrap_frequency * 2 * jnp.pi * s

        bumped_s = bump_function(s)

        tilt_angle = (
            tilt_angle_amplitude
            * (jnp.cos(tilt_angle_frequency / 2 * 2 * jnp.pi * bumped_s**2) - 1)
            / -2
        )
        torsion_angle = torsion_angle_amplitude * jnp.sin(
            torsion_angle_frequency * 2 * jnp.pi * bumped_s**2
        )
        return jnp.array([torsion_angle, tilt_angle, tilt_direction_angle]).flatten()

    ret = jax.vmap(partial(fn, **params))(
        jnp.linspace(0.0, 1.0, num=int(common_factor * 5e0))
    )
    print(ret)

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")
    z = np.rad2deg(ret[:, 0])
    r = np.rad2deg(ret[:, 1])
    theta = ret[:, 2]
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    ax1.plot(x, y, z, marker=".", linestyle="None", alpha=(alpha := 0.1))

    data = np.array([x, y, z]).T
    data_diff = np.diff(data, axis=0)
    data_diff_normed = data_diff / np.linalg.norm(data_diff, axis=1, keepdims=True)
    ax2.plot(
        data_diff_normed[:, 0],
        data_diff_normed[:, 1],
        data_diff_normed[:, 2],
        marker=".",
        linestyle="None",
        alpha=alpha,
    )
    plt.show()
    plt.close()

    render_trajectory(
        lambda t: SE3.from_translation(jnp.array([0, 0, 0.5])) @ trajectory(t),
        (0.0, trajectory.total_time),
        fps=24,
    )
