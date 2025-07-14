# Section 1: Generate and Visualize Mildly Curved Embankment-Like Surface with Abnormalities

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Create grid for a gently curved embankment surface
x = np.linspace(0, 40, 80)  # length
y = np.linspace(0, 50, 60)  # width
x, y = np.meshgrid(x, y)

# Gently sloped surface with slight curvature (like windshield)
a, b, c = 0.1, 0.03, -0.002  # slope + mild curvature
z_base = a * x + b * y + c * (x**2 + y**2)

# Add a hump and a cavity
z = z_base + \
    1 * np.exp(-((x - 25)**2 + (y - 30)**2) / 3) - \
    1 * np.exp(-((x - 15)**2 + (y - 10)**2) / 6)

X = np.column_stack((x.flatten(), y.flatten(), z.flatten()))
np.save("synthetic_curved_surface.npy", X)  # Save for next section

# Plot the surface
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='terrain', edgecolor='k', alpha=0.6)
ax.set_title("Synthetic Embankment-Like Surface with Hump and Cavity")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.tight_layout()
plt.show()

#%% Step 2

from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import DBSCAN
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import pandas as pd
import laspy
import numpy as np
import matplotlib.pyplot as plt


# Load the synthetic surface
X = np.load("C:/Users/golzardm/Documents/Dataset-Slope-LiDAR-Embankment-SLidE/visualization/synthetic_curved_surface.npy")
x, y, z = X[:, 0], X[:, 1], X[:, 2]

# Step 1: Fit a smooth surface using polynomial regression (RANSAC for robustness)
degree = 2
poly = PolynomialFeatures(degree=degree)
model = make_pipeline(poly, RANSACRegressor(LinearRegression(), residual_threshold=0.1))
model.fit(X[:, :2], z)

# Step 2: Predict the smooth surface and calculate residuals
z_pred = model.predict(X[:, :2])
residuals = z - z_pred

# Step 3: Threshold the residuals to detect abnormalities
threshold = 0.25  # Sensitivity threshold
abnormal_indices = np.where(np.abs(residuals) > threshold)[0]
abnormal_points = X[abnormal_indices]

# Step 4: Apply clustering (DBSCAN) to the abnormal points to isolate connected regions
db = DBSCAN(eps=1.0, min_samples=5).fit(abnormal_points[:, :2])
labels = db.labels_
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

# Step 5: Plot detected abnormalities with cluster labels

# Plot Format
plt.rcParams.update({
    'image.cmap': 'viridis',
    'font.serif': [
        'Times New Roman', 'Times', 'DejaVu Serif', 'Bitstream Vera Serif',
        'Computer Modern Roman', 'New Century Schoolbook', 'Century Schoolbook L',
        'Utopia', 'ITC Bookman', 'Bookman', 'Nimbus Roman No9 L', 'Palatino',
        'Charter', 'serif'
    ],
    'font.family': 'serif',
    'font.size': 14,
})

# Step 5: Plot detected abnormalities with cluster labels
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(abnormal_points[:, 0], abnormal_points[:, 1], abnormal_points[:, 2], 
                     c=labels, cmap='viridis', s=6)
ax.set_title(f"Detected Abnormalities via Residuals + DBSCAN Clustering ({num_clusters} clusters)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.tight_layout()
plt.show()

# ✅ Step 6: Save abnormalities to LAS file
header = laspy.LasHeader(point_format=3, version="1.2")
header.x_scale = header.y_scale = header.z_scale = 0.001
header.x_offset = header.y_offset = header.z_offset = 0.0

las = laspy.LasData(header)
las.x = abnormal_points[:, 0]
las.y = abnormal_points[:, 1]
las.z = abnormal_points[:, 2]
las.write("abnormalities_ransac_dbscan.las")

print("✅ Abnormalities saved to 'abnormalities_ransac_dbscan.las'")

# ✅ Step 7: Print fit evaluation metrics
r2 = r2_score(z, z_pred)
std_residual = np.std(residuals)

print("\nPolynomial Fit Evaluation:")
print(f"Degree: {degree}")
print(f"R² Score: {r2:.5f}")
print(f"Residual Std: {std_residual:.5f}")
