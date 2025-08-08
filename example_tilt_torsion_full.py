from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxlie import SE3

from example_utils import render_trajectory
from jaxlie_se3_traj import (
    BumpMethod,
    MultiplyDirection,
    SE3DeltaTiltTorsionFullSpace,
    SE3Trajectory,
    bump_function,
)

if __name__ == "__main__":
    params = {
        "bump_method": BumpMethod.LINEAR,
        "tilt_angle_amplitude": jnp.pi / 2,
        "torsion_angle_amplitude": jnp.deg2rad(5),
        "tilt_angle_frequency": 5e1,
        "torsion_angle_frequency": 3e2 * 2**0.5,
        "tilt_direction_wrap_frequency": 2e2 * 3**0.5,
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
        bump_method: BumpMethod,
        tilt_direction_wrap_frequency: float,
        tilt_angle_amplitude: float,
        tilt_angle_frequency: float,
        torsion_angle_amplitude: float,
        torsion_angle_frequency: float,
    ):
        tilt_direction_angle = tilt_direction_wrap_frequency * 2 * jnp.pi * s

        bumped_s = bump_function(bump_method)(s)

        tilt_angle = (
            tilt_angle_amplitude
            * (jnp.cos(tilt_angle_frequency / 2 * 2 * jnp.pi * bumped_s**2) - 1)
            / -2
        )
        torsion_angle = torsion_angle_amplitude * jnp.sin(
            torsion_angle_frequency * 2 * jnp.pi * bumped_s**2
        )
        return jnp.array([torsion_angle, tilt_angle, tilt_direction_angle]).flatten()

    ret = jax.vmap(partial(fn, **params))(jnp.linspace(0.0, 1.0, num=int(1e5)))
    print(ret)

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")
    z = np.rad2deg(ret[:, 0])
    r = np.rad2deg(ret[:, 1])
    theta = ret[:, 2]
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    ax1.plot(x, y, z, marker=".", linestyle="None", alpha=0.01)

    data = np.array([x, y, z]).T
    data_diff = np.diff(data, axis=0)
    data_diff_normed = data_diff / np.linalg.norm(data_diff, axis=1, keepdims=True)
    ax2.plot(
        data_diff_normed[:, 0],
        data_diff_normed[:, 1],
        data_diff_normed[:, 2],
        marker=".",
        linestyle="None",
        alpha=0.01,
    )
    plt.show()
    plt.close()

    render_trajectory(
        lambda t: SE3.from_translation(jnp.array([0, 0, 0.5])) @ trajectory(t),
        (0.0, trajectory.total_time),
        fps=24,
    )
