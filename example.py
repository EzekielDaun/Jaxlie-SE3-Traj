import jax.numpy as jnp
from jaxlie import SE3

from example_utils import render_trajectory
from jaxlie_se3_traj import (
    BumpMethod,
    MultiplyDirection,
    SE3DeltaLog,
    SE3DeltaTiltTorsionRolling,
    SE3DeltaTiltTorsionSpiral,
    SE3Trajectory,
)

if __name__ == "__main__":
    trajectory = SE3Trajectory.from_deltas(
        (
            SE3DeltaTiltTorsionSpiral(
                MultiplyDirection.RIGHT,
                8.0,
                BumpMethod.SEVEN_SEG_S_DEFAULT,
                jnp.pi / 2,
                2,
            ),
            SE3DeltaLog(
                MultiplyDirection.LEFT,
                1.0,
                BumpMethod.CUBIC,
                tuple(jnp.array([0.0, 0.0, 0.0, 0, 0, jnp.pi / 2])),
            ),
            SE3DeltaLog(
                MultiplyDirection.LEFT,
                1.0,
                BumpMethod.CUBIC,
                tuple(jnp.array([0.0, 0.0, 0.5, 0, 0, 0])),
            ),
            SE3DeltaLog(
                MultiplyDirection.LEFT,
                1.0,
                BumpMethod.LINEAR,
                tuple(jnp.array([0.0, 0.0, 0.0, 0, tilt_angle := jnp.pi / 2, 0])),
            ),
            SE3DeltaTiltTorsionRolling(
                MultiplyDirection.RIGHT,
                3.0,
                BumpMethod.TRIG_SIN_INTEGRAL,
                tilt_angle=tilt_angle,
            ),
            SE3DeltaLog(
                MultiplyDirection.LEFT,
                1.0,
                BumpMethod.QUINTIC,
                tuple(jnp.array([0.0, 0.0, 0.0, 0, -tilt_angle, 0])),
            ),
            SE3DeltaLog(
                MultiplyDirection.RIGHT,
                1.0,
                BumpMethod.TRIG_COS_EASE,
                tuple(jnp.array([0.0, 0.0, -0.5, 0, 0, 0])),
            ),
        ),
    )

    for t in jnp.linspace(0.0, trajectory.total_time, num=10):
        print(f"t={t:.2f}, T={trajectory(t)}")

    render_trajectory(
        lambda t: SE3.from_translation(jnp.array([0, 0, 0.5])) @ trajectory(t),
        t_range=(0.0, trajectory.total_time),
    )
