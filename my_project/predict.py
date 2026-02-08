import pandas as pd
import numpy as np
import joblib

print("Loading saved model...")
model = joblib.load('models/best_model.pkl')
scaler_X = joblib.load('models/scaler_X.pkl')
scaler_y = joblib.load('models/scaler_y.pkl')

def predict_next_24h(last_24h_data):
    input_data = np.array(last_24h_data).reshape(1, -1)
    input_scaled = scaler_X.transform(input_data)
    prediction_scaled = model.predict(input_scaled)
    prediction = scaler_y.inverse_transform(prediction_scaled)
    return prediction[0]

df = pd.read_csv('data/aquaPDataset.csv', sep=';')
last_24_hours = df['flow_meter1'].tail(24).values

print("Predicting the next 24 hours...\n")
future_24h = predict_next_24h(last_24_hours)

print("Results:")
print("=" * 50)
for i, value in enumerate(future_24h, 1):
    print(f"Hour {i:2d}: {value:6.2f} L/s")

results_df = pd.DataFrame({
    'Hour': range(1, 25),
    'Predicted_Flow': future_24h
})

results_df.to_csv('results/predictions.csv', index=False)
print("Results saved to: results/predictions.csv")
