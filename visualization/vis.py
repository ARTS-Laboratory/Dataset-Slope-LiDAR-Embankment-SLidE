import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import laspy
import json

#file_path = '../Data/Site-3/Fall 2021/laz/terryF21_S3_groundex.laz'
file_path = '../Data/Site-3/Summer 2021/laz/terryS21_groundex_seg.laz'

with laspy.open(file_path) as file:
    las = file.read()

x = las.x
y = las.y
z = las.z

# Extracts color information (scaled to 0-1 for matplotlib)
red = las.red / np.max(las.red)
green = las.green / np.max(las.green)
blue = las.blue / np.max(las.blue)
colors = np.vstack((red, green, blue)).T

range_x = np.ptp(x)  # Peak to peak (max - min) for x
range_y = np.ptp(y)  # Peak to peak (max - min) for y
range_z = np.ptp(z)  # Peak to peak (max - min) for z

# Avoid division by zero in case of flat point clouds in any dimension
range_x = max(range_x, 0.1)
range_y = max(range_y, 0.1)
range_z = max(range_z, 0.1)

# Creates a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(x, y, z, c=colors, marker='.', s=1)

# Sets labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_box_aspect((range_x, range_y, range_z))


#Set limits if you want to focus on a specific area
# ax.set_xlim([min_x, max_x])
# ax.set_ylim([min_y, max_y])
# ax.set_zlim([min_z, max_z])

plt.show()
