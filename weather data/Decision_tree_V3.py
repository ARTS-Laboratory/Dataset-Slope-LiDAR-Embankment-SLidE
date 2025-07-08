import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.tree import _tree

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
max_depth = int(input("Enter max depth for Decision Tree (e.g., 3-5): "))

# Initialize and fit Decision Tree
dt = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
dt.fit(X_train, y_train)

# Predict and calculate MSE
y_train_pred = dt.predict(X_train)
y_test_pred = dt.predict(X_test)
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
print(f"Training MSE: {train_mse:.4f}")
print(f"Test MSE: {test_mse:.4f}")

# Feature importance
importances = dt.feature_importances_
feature_names = ['Rainfall', 'Temperature']
print("Feature Importances:")
for name, importance in zip(feature_names, importances):
    print(f"{name}: {importance:.4f} ({importance*100:.1f}%)")

# Visualize feature importance
plt.figure(figsize=(8, 6))
plt.bar(feature_names, importances, color=['#1f77b4', '#ff7f0e'])
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance in Decision Tree')
plt.ylim(0, 1)  # Importance ranges from 0 to 1
for i, v in enumerate(importances):
    plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
plt.show()

# Predict with new data
new_data = np.array([[14.0, 25.0]])  # Example: rainfall=14, temperature=25
new_data_scaled = scaler.transform(new_data)
prediction = dt.predict(new_data_scaled)
print(f"New data (unscaled): {new_data[0]}")
print(f"New data (scaled): {new_data_scaled[0]}")
print(f"Decision Tree Prediction for new data: {prediction[0]}")

# Function to trace the path through the tree
def trace_tree_path(tree, feature_names, x):
    node_indicator = tree.decision_path(x)
    leaf_id = tree.apply(x)
    node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]

    path = []
    for node_id in node_index:
        if tree.tree_.feature[node_id] != _tree.TREE_UNDEFINED:  # Not a leaf
            feature = feature_names[tree.tree_.feature[node_id]]
            threshold = tree.tree_.threshold[node_id]
            value = x[0, tree.tree_.feature[node_id]]
            direction = "left" if value <= threshold else "right"
            path.append(f"Node {node_id}: {feature} <= {threshold:.4f} -> {direction}")
        else:
            path.append(f"Leaf Node {node_id}: value = {tree.tree_.value[node_id][0][0]:.4f}")
    return path

# Trace the path for the new data
path = trace_tree_path(dt, ['Rainfall', 'Temperature'], new_data_scaled)
print("Path through the tree:")
for step in path:
    print(step)

# Visualize the decision tree with parent and child nodes
plt.figure(figsize=(20,10))
plot_tree(dt, feature_names=['Rainfall', 'Temperature'], 
          impurity=False, 
          filled=True, 
          rounded=True, 
          label='all', 
          proportion=True,
          max_depth=max_depth)
plt.title("Decision Tree for Soil Moisture Prediction")
plt.show()

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
y_pred_unscaled = dt.predict(X_test_unscaled_scaled)

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

