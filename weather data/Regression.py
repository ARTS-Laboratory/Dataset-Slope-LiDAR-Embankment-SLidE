import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# plot Format
plt.rcParams.update({
    'image.cmap': 'viridis',
    'font.serif': [
        'Times New Roman', 'Times', 'DejaVu Serif', 'Bitstream Vera Serif',
        'Computer Modern Roman', 'New Century Schoolbook', 'Century Schoolbook L',
        'Utopia', 'ITC Bookman', 'Bookman', 'Nimbus Roman No9 L', 'Palatino',
        'Charter', 'serif'
    ],
    'font.family': 'serif',
    'font.size': 11,
})

# Load the CSV file
file_path = 'PRISM_ppt_tmean_tdmean_2.csv'
data = pd.read_csv(file_path, skiprows=10)

X = data[['Temperature', 'Rainfall']].values
y = data['Soil_Moisture'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
poly = PolynomialFeatures(degree=3, include_bias=True)  # degree of regressor
X_poly = poly.fit_transform(X_scaled)

param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1]}
ridge = Ridge()
grid_search = GridSearchCV(ridge, param_grid, cv=7)
grid_search.fit(X_poly, y)
best_alpha = grid_search.best_params_['alpha']
print(f"Best alpha: {best_alpha}")

# Fit final model with best alpha
final_ridge = Ridge(alpha=best_alpha)
final_ridge.fit(X_poly, y)

# Fix other feature at mean for Temperature plot
X_mean = np.mean(X_scaled, axis=0)
X_test_temp = np.tile(X_mean, (200, 1))
X_test_temp[:, 0] = np.linspace(X_scaled[:, 0].min(), X_scaled[:, 0].max(), 200)  # Vary Temperature
X_test_temp_poly = poly.transform(X_test_temp)
y_pred_temp = final_ridge.predict(X_test_temp_poly)

plt.figure()
plt.plot(X_test_temp[:, 0], y_pred_temp, label='Regression', color='blue')
plt.scatter(X_scaled[:, 0], y, color='red', alpha=0.5, label='Data Points')  # Add scattered points
plt.xlabel('Temperature (scaled)')
plt.ylabel('Soil Moisture')
plt.title('Soil Moisture vs. Temperature (Rainfall fixed at mean)')
plt.grid(True, linestyle='--', alpha=0.7)  
plt.legend()
plt.show()

# Fix other feature at mean for Rainfall plot
X_test_rain = np.tile(X_mean, (200, 1))
X_test_rain[:, 1] = np.linspace(X_scaled[:, 1].min(), X_scaled[:, 1].max(), 200)  # Vary Rainfall
X_test_rain_poly = poly.transform(X_test_rain)
y_pred_rain = final_ridge.predict(X_test_rain_poly)

plt.figure()
plt.plot(X_test_rain[:, 1], y_pred_rain, label='Regression', color='green')
plt.scatter(X_scaled[:, 1], y, color='red', alpha=0.5, label='Data Points')  # Add scattered points
plt.xlabel('Rainfall (scaled)')
plt.ylabel('Soil Moisture')
plt.title('Soil Moisture vs. Rainfall (Temperature fixed at mean)')
plt.grid(True, linestyle='--', alpha=0.7)  
plt.legend()
plt.show()

# Create a grid of Temperature and Rainfall
T = np.linspace(X_scaled[:, 0].min(), X_scaled[:, 0].max(), 50)
R = np.linspace(X_scaled[:, 1].min(), X_scaled[:, 1].max(), 50)
T, R = np.meshgrid(T, R)
X_test_3d = np.c_[T.ravel(), R.ravel()]
X_test_3d_scaled = scaler.transform(X_test_3d)  # Rescale test data
X_test_3d_poly = poly.transform(X_test_3d_scaled)
y_pred_3d = final_ridge.predict(X_test_3d_poly)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(T, R, y_pred_3d.reshape(T.shape), cmap='viridis')
ax.set_xlabel('Temperature (scaled)')
ax.set_ylabel('Rainfall (scaled)')
ax.set_zlabel('Soil Moisture')
plt.title('Soil Moisture vs. Temperature and Rainfall')
plt.grid(True, linestyle='--', alpha=0.7)  # Add grid
plt.show()


# additional computation for further modification

y_pred_temp_all = final_ridge.predict(X_poly)
epsilon = y - y_pred_temp_all
plt.figure()
plt.scatter(X_scaled[:, 0], epsilon, color='purple', alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Temperature (scaled)')
plt.ylabel('Residual')
plt.title('Residuals vs. Temperature')
plt.show()

# Soil Moisture vs. Temperature (Colored by Temperature)
plt.figure()
plt.scatter(data['Temperature'], y, c=data['Temperature'], cmap='viridis', alpha=0.5)
plt.colorbar(label='Temperature')
plt.xlabel('Temperature (raw)')
plt.ylabel('Soil Moisture')
plt.title('Soil Moisture vs. Temperature (Colored by Temperature)')
plt.show()

# Soil Moisture vs. Rainfall (Colored by Rainfall)
plt.figure()
plt.scatter(data['Rainfall'], y, c=data['Rainfall'], cmap='viridis', alpha=0.5)
plt.colorbar(label='Rainfall')
plt.xlabel('Rainfall (raw)')
plt.ylabel('Soil Moisture')
plt.title('Soil Moisture vs. Rainfall (Colored by Rainfall)')
plt.show()


# Example: Feed new data
new_data = np.array([
    [25.0, 12.0],  # Temperature, Rainfall
    [28.0, 15.0],
    [22.0, 10.0]
])
new_data_scaled = scaler.transform(new_data)
new_data_poly = poly.transform(new_data_scaled)
predictions = final_ridge.predict(new_data_poly)
print("Predicted Soil Moisture for new data:", predictions)