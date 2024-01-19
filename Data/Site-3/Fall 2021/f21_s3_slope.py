import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import laspy
import json

file_path = 'slope.las'

with laspy.open(file_path) as file:
    las = file.read()

for dimension in las.point_format.dimensions:
    print(dimension.name)

intensities = np.array(las.intensity)
return_numbers = np.array(las.return_number)
num_returns = np.array(las.number_of_returns)
scan_dir_flags = np.array(las.scan_direction_flag)
edge_flags = np.array(las.edge_of_flight_line)
classifications = np.array(las.classification)
synthetics = np.array(las.synthetic)
key_points = np.array(las.key_point)
withhelds = np.array(las.withheld)
scan_angles = np.array(las.scan_angle_rank)
user_data = np.array(las.user_data)
point_src_ids = np.array(las.point_source_id)
reds = np.array(las.red)
greens = np.array(las.green)
blues = np.array(las.blue)

las_data = {
    'X': las.x.array.tolist(),
    'Y': las.y.array.tolist(),
    'Z': las.z.array.tolist(),
    'intensity': intensities.tolist(),
    'return_number': return_numbers.tolist(),
    'number_of_returns': num_returns.tolist(),
    'scan_direction_flag': scan_dir_flags.tolist(),
    'edge_of_flight_line': edge_flags.tolist(),
    'classification': classifications.tolist(),
    'synthetic': synthetics.tolist(),
    'key_point': key_points.tolist(),
    'withheld': withhelds.tolist(),
    'scan_angle_rank': scan_angles.tolist(),
    'user_data': user_data.tolist(),
    'point_source_id': point_src_ids.tolist(),
    'red': reds.tolist(),
    'green': greens.tolist(),
    'blue': blues.tolist()
}

with open('slope.json', 'w') as json_file:
    json.dump(las_data, json_file, indent=4)

print("JSON file created successfully.")

# Uncomment all below to plot with mpl,
# though I would not recommend it, since this particular
# dataset contains ~3.1 million x, y, and z points each.

# x = las.x
# y = las.y
# z = las.z

# # Extracts color information (scaled to 0-1 for matplotlib)
# red = las.red / np.max(las.red)
# green = las.green / np.max(las.green)
# blue = las.blue / np.max(las.blue)
# colors = np.vstack((red, green, blue)).T

# # Creates a 3D scatter plot
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# scatter = ax.scatter(x, y, z, c=colors, marker='.', s=1)

# # Sets labels
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# Set limits if you want to focus on a specific area
# ax.set_xlim([min_x, max_x])
# ax.set_ylim([min_y, max_y])
# ax.set_zlim([min_z, max_z])

# plt.show()
