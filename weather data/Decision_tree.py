import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
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

# Define X with both features and y
X = data[['Rainfall', 'Temperature']].values
y = data['Soil_Moisture'].values

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Prompt user for max_depth (optional tuning)
max_depth = int(input("Enter max depth for Random Forest (e.g., 5-10): "))

# Initialize and fit Random Forest
rf = RandomForestRegressor(n_estimators=100, max_depth=max_depth, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
rf.fit(X_train, y_train)

# Predict and calculate MSE
y_train_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
print(f"Training MSE: {train_mse:.4f}")
print(f"Test MSE: {test_mse:.4f}")

# Feature importance
print("Feature Importances: Rainfall, Temperature")
print(rf.feature_importances_)

# Predict with new data
new_data = np.array([[14.0, 25.0]])  # Example: rainfall=14, temperature=25
new_data_scaled = scaler.transform(new_data)
prediction = rf.predict(new_data_scaled)
print("Random Forest Prediction for new data (rainfall=14, temperature=25):", prediction)

# Visualize the fit - 2D plot with color by Temperature
plt.figure()
scatter = plt.scatter(X_scaled[:, 0], y, c=X_scaled[:, 1], cmap='viridis', alpha=0.5, label='Training Data')
plt.colorbar(scatter, label='Temperature (scaled)')
plt.scatter(new_data_scaled[:, 0], prediction, color='green', marker='x', label='New Prediction')
plt.xlabel('Rainfall (scaled)')
plt.ylabel('Soil Moisture (scaled)')
plt.title('Soil Moisture vs. Rainfall (Colored by Temperature)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Visualize the fit - Unscaled Plot
X_test_rain_unscaled = np.linspace(data['Rainfall'].min(), data['Rainfall'].max(), 100).reshape(-1, 1)
X_test_temp_unscaled = np.linspace(data['Temperature'].min(), data['Temperature'].max(), 100).reshape(-1, 1)
X_test_unscaled = np.column_stack((X_test_rain_unscaled, X_test_temp_unscaled))
X_test_unscaled_scaled = scaler.transform(X_test_unscaled)
y_pred_unscaled = rf.predict(X_test_unscaled_scaled)

plt.figure()
plt.scatter(data['Rainfall'].values, y, c=data['Temperature'].values, cmap='viridis', alpha=0.5, label='Training Data')
plt.colorbar(label='Temperature (unscaled)')
plt.plot(X_test_rain_unscaled, y_pred_unscaled, color='blue', label='Regression')
plt.scatter(14.0, prediction, color='green', marker='x', label='New Prediction')  # Unscaled new data point
plt.xlabel('Rainfall (unscaled)')
plt.ylabel('Soil Moisture (unscaled)')
plt.title('Unscaled: Soil Moisture vs. Rainfall (Colored by Temperature)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
