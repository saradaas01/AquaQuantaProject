import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import KNNImputer
from sklearn.multioutput import MultiOutputRegressor
import joblib
import os

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

print("Loading dataset...")
df = pd.read_csv('data/aquaPDataset.csv', sep=';')

def remove_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    data[column] = data[column].mask((data[column] < lower) | (data[column] > upper), np.nan)
    return data

print("Cleaning data and handling outliers...")
df = remove_outliers(df, 'flow_meter1')

imputer = KNNImputer(n_neighbors=3)
df['flow_meter1'] = imputer.fit_transform(df[['flow_meter1']])



lookback = 24
forecast_horizon = 24

def create_sequences(data, lookback, horizon):
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback:i+lookback+horizon])
    return np.array(X), np.array(y)

print("Generating time series sequences...")
flow_data = df['flow_meter1'].values
X, y = create_sequences(flow_data, lookback, forecast_horizon)


split_point = int(len(X) * 0.8)
X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = y[:split_point], y[split_point:]

print("Scaling data...")
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)


print("\nStarting model training...\n")


print("Training SVR model...")
svr_model = MultiOutputRegressor(SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1))
svr_model.fit(X_train_scaled, y_train_scaled)
y_pred_svr = scaler_y.inverse_transform(svr_model.predict(X_test_scaled))


print("Training KNN model...")
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train_scaled)
y_pred_knn = scaler_y.inverse_transform(knn_model.predict(X_test_scaled))


print("Training MLP model...")
mlp_model = MLPRegressor(hidden_layer_sizes=(48, 92), activation='relu', max_iter=1000, random_state=42)
mlp_model.fit(X_train_scaled, y_train_scaled)
y_pred_mlp = scaler_y.inverse_transform(mlp_model.predict(X_test_scaled))


def evaluate(y_true, y_pred, model_name):
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true.flatten(), y_pred.flatten())
    
    print("\n==================================================")
    print(f"{model_name} Performance:")
    print("==================================================")
    print(f"MAPE: {mape:.2f}%")
    print(f"MAE:  {mae:.2f}")
    print(f"R2:   {r2:.3f}")
    
    return {'mape': mape, 'mae': mae, 'r2': r2}

results = {}
results['SVR'] = evaluate(y_test, y_pred_svr, "SVR")
results['KNN'] = evaluate(y_test, y_pred_knn, "KNN")
results['MLP'] = evaluate(y_test, y_pred_mlp, "MLP")


best_model_name = min(results, key=lambda x: results[x]['mape'])
print(f"\nBest model: {best_model_name}")


print(f"\nSaving the best model: {best_model_name}...")
if best_model_name == 'SVR':
    joblib.dump(svr_model, 'models/best_model.pkl')
    joblib.dump(y_pred_svr, 'models/best_predictions.pkl')
elif best_model_name == 'KNN':
    joblib.dump(knn_model, 'models/best_model.pkl')
    joblib.dump(y_pred_knn, 'models/best_predictions.pkl')
else:
    joblib.dump(mlp_model, 'models/best_model.pkl')
    joblib.dump(y_pred_mlp, 'models/best_predictions.pkl')

joblib.dump(scaler_X, 'models/scaler_X.pkl')
joblib.dump(scaler_y, 'models/scaler_y.pkl')
joblib.dump(y_test, 'models/y_test.pkl')


with open('models/model_info.txt', 'w', encoding='utf-8') as f:
    f.write(f"Best Model: {best_model_name}\n")
    f.write(f"MAPE: {results[best_model_name]['mape']:.2f}%\n")
    f.write(f"MAE: {results[best_model_name]['mae']:.2f}\n")
    f.write(f"R2: {results[best_model_name]['r2']:.3f}\n")

print("Training and saving process completed successfully!")
