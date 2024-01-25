import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.animation import FuncAnimation

def plot_cuboid(ax, position, size, colors):
    # Define the vertices of the cuboid
    vertices = np.array([
        [position[0], position[1], position[2]],
        [position[0] + size[0], position[1], position[2]],
        [position[0] + size[0], position[1] + size[1], position[2]],
        [position[0], position[1] + size[1], position[2]],
        [position[0], position[1], position[2] + size[2]],
        [position[0] + size[0], position[1], position[2] + size[2]],
        [position[0] + size[0], position[1] + size[1], position[2] + size[2]],
        [position[0], position[1] + size[1], position[2] + size[2]],
    ])

    # Create bar3d for each face
    for i in range(0, len(vertices), 4):
        x = [vertices[i][0], vertices[i + 1][0], vertices[i + 2][0], vertices[i + 3][0], vertices[i][0]]
        y = [vertices[i][1], vertices[i + 1][1], vertices[i + 2][1], vertices[i + 3][1], vertices[i][1]]
        z = [vertices[i][2], vertices[i + 1][2], vertices[i + 2][2], vertices[i + 3][2], vertices[i][2]]
        ax.bar3d(x, y, z, size[0], size[1], size[2], shade=True, color=colors[i // 4])

def update(frame):
    ax.cla()  # Clear the current axes
    plot_cuboid(ax, cuboid_position, cuboid_size, gradient_color)
    ax.set_xlim([0, 20])
    ax.set_ylim([0, 20])
    ax.set_zlim([0, 10])
    azimuth = frame % 360
    elevation = 20
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_axis_off()
    
    print(f"Frame: {frame}, Elevation: {elevation}, Azimuth: {azimuth}")
    
    if frame >= 360:
        animation.event_source.stop()  # Stop the animation after 360 frames

if __name__ == "__main__":
    # Create a figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set the cuboid position and size
    cuboid_position = (0, 0, 0)
    cuboid_size = (10, 10, 1)

    # Set gradient color
    gradient_color = ['deepskyblue', 'deepskyblue', 'deepskyblue', 'deepskyblue', 'deepskyblue', 'deepskyblue']

    # Set axis limits
    ax.set_xlim([0, 20])
    ax.set_ylim([0, 20])
    ax.set_zlim([0, 10])

    # Create animation
    animation = FuncAnimation(fig, update, frames=np.arange(0, 361, 1), interval=50)

    # Show the plot
    plt.show()
