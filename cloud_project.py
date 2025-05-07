import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
import matplotlib.pyplot as plt
## Loading & Analyzing the Data
# File path to the uploaded Excel file
file_path = "C:/Users/Admin/Desktop/water_level.csv"

# Loads the dataset
File "/mount/src/cloud--app/cloud_project.py"
data = pd.read_csv(file_path)
data.shape
display(data)
data.SiteNo.value_counts()
# Group the data by 'SiteNo' and aggregate features
grouped_data = data.groupby('SiteNo').agg({
    'Original Value': 'mean',
    'Depth to Water Below Surface': 'mean',
    'Water level ': 'mean'  # Target column
}).reset_index()
print("\nGrouped Data (Preview):")
print(grouped_data.head())
## Modeling
# Define features and target
features = ['Original Value', 'Depth to Water Below Surface']  # Exclude 'SiteNo' as it's an identifier
target = 'Water level '

# Extract features (X) and target (y)
X = grouped_data[features].values
y = grouped_data[target].values
X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
std_scaler = StandardScaler()

X_trained_scaled = std_scaler.fit_transform(X_train)
X_test_scaled = std_scaler.transform(X_test)

X_trained_scaled.shape
# Build the ANN model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(64, activation='relu'),  # Input layer
    tf.keras.layers.Dense(32, activation='relu'),                              # Hidden layer
    tf.keras.layers.Dense(1)                              # Output layer
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
# 5. Train the model
history = model.fit(X_trained_scaled, y_train, epochs=100, batch_size=8, validation_split=0.2, verbose=1)


plt.plot(history.history["mae"], label="Training Loss")
plt.plot(history.history["val_mae"], label="Validation Loss")

plt.legend()
plt.title("Learning Curve")
plt.show()
### Performing Modeling Performance
y_pred_test = model.predict(X_test_scaled)
mae_test = mean_absolute_error(y_test, y_pred_test)

mae_test
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='True Groundwater Levels', color='blue')
plt.plot(y_pred_test, label='Predicted Groundwater Levels', color='red', linestyle='dashed')
plt.legend()
plt.xlabel('Sample Index')
plt.ylabel('Groundwater Level')
plt.title('True vs Predicted Groundwater Levels')
plt.show()
