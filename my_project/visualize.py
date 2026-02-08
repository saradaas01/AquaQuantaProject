import matplotlib.pyplot as plt
import joblib
import numpy as np

y_test = joblib.load('models/y_test.pkl')
y_pred = joblib.load('models/best_predictions.pkl')

plt.rcParams['font.family'] = 'Arial'

plt.figure(figsize=(15, 6))
sample_size = 168
y_test_sample = y_test[:sample_size].flatten()
y_pred_sample = y_pred[:sample_size].flatten()

plt.plot(y_test_sample, label='Actual', linewidth=2, color='blue')
plt.plot(y_pred_sample, label='Predicted', linewidth=2, color='red', linestyle='--')
plt.xlabel('Time (hours)', fontsize=12)
plt.ylabel('Flow (L/s)', fontsize=12)
plt.title('Actual vs Predicted Water Flow', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/comparison.png', dpi=300)
print("Saved: results/comparison.png")

errors = y_test.flatten() - y_pred.flatten()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
plt.xlabel('Error (L/s)')
plt.ylabel('Frequency')
plt.title('Error Distribution')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2)

plt.subplot(1, 2, 2)
plt.scatter(y_test.flatten(), y_pred.flatten(), alpha=0.3, s=10)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--',
    linewidth=2,
    label='Perfect Prediction'
)
plt.xlabel('Actual (L/s)')
plt.ylabel('Predicted (L/s)')
plt.title('Actual vs Predicted Scatter')
plt.legend()

plt.tight_layout()
plt.savefig('results/error_analysis.png', dpi=300)
print("Saved: results/error_analysis.png")

plt.show()
