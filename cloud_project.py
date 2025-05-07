import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
import matplotlib.pyplot as plt


def main():
    st.title("Groundwater Level Prediction Using ANN")

    uploaded_file = st.file_uploader("C:/Users/Admin/Documents/Ambu[1]/Ambu/WATERLEVEL.xlsx", type=["xlsx"])

    if uploaded_file is not None:
        data = pd.read_excel(uploaded_file)
        st.subheader("Raw Data Preview")
        st.write(data.head())

        # Grouping data
        grouped_data = data.groupby('SiteNo').agg({
            'Original Value': 'mean',
            'Depth to Water Below Surface': 'mean',
            'Water level ': 'mean'  # Target column
        }).reset_index()

        st.subheader("Grouped Data Preview")
        st.write(grouped_data.head())

        # Feature selection
        features = ['Original Value', 'Depth to Water Below Surface']
        target = 'Water level '

        X = grouped_data[features].values
        y = grouped_data[target].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(2,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=8, validation_split=0.2, verbose=0)

        # Plot learning curve
        st.subheader("Training History")
        fig, ax = plt.subplots()
        ax.plot(history.history["mae"], label="Training MAE")
        ax.plot(history.history["val_mae"], label="Validation MAE")
        ax.legend()
        ax.set_title("Learning Curve")
        st.pyplot(fig)

        # Evaluation
        y_pred_test = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred_test)

        st.subheader(f"Mean Absolute Error on Test Data: {mae:.2f}")

        # True vs Predicted Plot
        st.subheader("True vs Predicted Groundwater Levels")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(y_test, label='True', color='blue')
        ax2.plot(y_pred_test, label='Predicted', color='red', linestyle='dashed')
        ax2.legend()
        ax2.set_xlabel('Sample Index')
        ax2.set_ylabel('Water Level')
        ax2.set_title('True vs Predicted Water Levels')
        st.pyplot(fig2)

    else:
        st.info("Please upload the Excel file to proceed.")


if __name__ == "__main__":
    main()
