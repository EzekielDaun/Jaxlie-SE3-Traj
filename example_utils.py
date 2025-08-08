from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from jaxlie import SE3
from jaxtyping import Float
from mpl_toolkits.mplot3d import Axes3D


def plot_se3(ax, T: SE3, scale: float = 0.5):
    origin = T.translation()
    R = T.rotation().as_matrix()

    colors = ["r", "g", "b"]
    for i in range(3):
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            R[0, i],
            R[1, i],
            R[2, i],
            length=scale,
            color=colors[i],
        )


def render_trajectory(
    trajectory: Callable[[Float], SE3], t_range: tuple[float, float], fps=24
):
    fig = plt.figure(figsize=(12, 8))
    ax1: Axes3D = fig.add_subplot(221, projection="3d")
    ax1.view_init(elev=30, azim=-60)
    ax2: Axes3D = fig.add_subplot(222, projection="3d")
    ax2.view_init(elev=90, azim=-90)
    ax3: Axes3D = fig.add_subplot(223, projection="3d")
    ax3.view_init(elev=0, azim=-90)
    ax4: Axes3D = fig.add_subplot(224, projection="3d")
    ax4.view_init(elev=0, azim=0)

    times = np.linspace(*t_range, num=int((t_range[1] - t_range[0]) * fps))
    positions = []

    for t in times:
        T = trajectory(t)
        positions.append(T.translation())

    positions = np.stack(positions)
    size = np.max(np.abs([positions.min(), positions.max()]))

    for t in times:
        for ax in (ax1, ax2, ax3, ax4):
            ax.clear()
            ax.set_xlim(-size, size)
            ax.set_ylim(-size, size)
            ax.set_zlim(-size, size)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

            T = trajectory(t)
            plot_se3(ax, T)

        fig.canvas.draw()
        fig.tight_layout()
        plt.pause(1 / fps)

    plt.close(fig)
