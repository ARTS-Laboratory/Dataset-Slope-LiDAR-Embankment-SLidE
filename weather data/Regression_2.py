import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

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
    'font.size': 11,
})

# Load the CSV file
file_path = 'PRISM_ppt_tmean_tdmean_2.csv'
data = pd.read_csv(file_path, skiprows=10)

X = data[['Rainfall']].values  # Use only Rainfall as the feature
y = data['Soil_Moisture'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Prompt user for polynomial degree
degree = int(input("Enter the polynomial degree (1-5): "))

# Fit model with user-chosen degree
poly = PolynomialFeatures(degree=degree, include_bias=True)
X_poly = poly.fit_transform(X_scaled)
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
ridge = Ridge()
grid_search = GridSearchCV(ridge, param_grid, cv=5)
grid_search.fit(X_poly, y)
best_alpha = grid_search.best_params_['alpha']
print(f"Best alpha: {best_alpha}")

# Fit final model with best alpha
final_ridge = Ridge(alpha=best_alpha)
final_ridge.fit(X_poly, y)

# Calculate and print MSE
y_pred = final_ridge.predict(X_poly)
mse = mean_squared_error(y, y_pred)
print(f"MSE for degree {degree}: {mse:.4f}")

# Predict with new rainfall data
new_rainfall = np.array([[8.0], [12.0], [14.0]])  # New rainfall values (2D array)
new_rainfall_scaled = scaler.transform(new_rainfall)
new_rainfall_poly = poly.transform(new_rainfall_scaled)
predictions_scaled = final_ridge.predict(new_rainfall_poly)
print("Predicted Soil Moisture for new rainfall data (scaled context):", predictions_scaled)

# Inverse transform to get unscaled predictions
# Since y is not scaled, predictions are in the original scale
predictions_unscaled = predictions_scaled  # No inverse scaling needed for y
new_rainfall_unscaled = new_rainfall  # Unscaled new rainfall

# Visualize the fit - Scaled Plot
X_test_rain = np.linspace(X_scaled.min(), X_scaled.max(), 100).reshape(-1, 1)
X_test_rain_poly = poly.transform(X_test_rain)
y_pred_rain_scaled = final_ridge.predict(X_test_rain_poly)

plt.figure()
plt.scatter(X_scaled, y, color='red', alpha=0.5, label='Training Data')
plt.plot(X_test_rain, y_pred_rain_scaled, color='blue', label='Regression')
plt.scatter(new_rainfall_scaled, predictions_scaled, color='green', marker='x', label='New Predictions')
plt.xlabel('Rainfall (scaled)')
plt.ylabel('Soil Moisture (scaled)')
plt.title('Scaled: Soil Moisture vs. Rainfall')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Visualize the fit - Unscaled Plot
X_test_rain_unscaled = np.linspace(data['Rainfall'].min(), data['Rainfall'].max(), 100).reshape(-1, 1)
X_test_rain_unscaled_scaled = scaler.transform(X_test_rain_unscaled)
X_test_rain_unscaled_poly = poly.transform(X_test_rain_unscaled_scaled)
y_pred_rain_unscaled = final_ridge.predict(X_test_rain_unscaled_poly)

plt.figure()
plt.scatter(data['Rainfall'].values, y, color='red', alpha=0.5, label='Training Data')
plt.plot(X_test_rain_unscaled, y_pred_rain_unscaled, color='blue', label='Regression')
plt.scatter(new_rainfall_unscaled, predictions_unscaled, color='green', marker='x', label='New Predictions')
plt.xlabel('Rainfall (unscaled)')
plt.ylabel('Soil Moisture (unscaled)')
plt.title('Unscaled: Soil Moisture vs. Rainfall')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()