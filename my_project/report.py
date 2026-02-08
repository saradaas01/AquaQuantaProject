import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime

print("=" * 80)
print(" " * 25 + "DETAILED MODEL REPORT")
print("=" * 80)

# Load data
print("\nLoading data...")
y_test = joblib.load('models/y_test.pkl')
y_pred = joblib.load('models/best_predictions.pkl')

# Calculate metrics
print("Calculating metrics...")
errors = y_test.flatten() - y_pred.flatten()
abs_errors = np.abs(errors)
percentage_errors = (errors / (y_test.flatten() + 1e-10)) * 100

mape = np.mean(np.abs(percentage_errors))
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test.flatten(), y_pred.flatten())

# Build report
report = f"""
{'=' * 80}
        Water Flow Prediction Model Evaluation Report
{'=' * 80}

Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'=' * 80}
1. Overall Performance Summary
{'=' * 80}

Key Metrics:
-------------------------------------------------------------------------------
  - MAPE (Mean Absolute Percentage Error):    {mape:>8.2f}%
  - MAE  (Mean Absolute Error):               {mae:>8.2f} L/s
  - RMSE (Root Mean Square Error):            {rmse:>8.2f} L/s
  - R²   (Coefficient of Determination):      {r2:>8.3f}
-------------------------------------------------------------------------------

Performance Assessment:
"""

if mape < 10 and r2 > 0.85:
    assessment = "Excellent - Model is ready for production use"
elif mape < 15 and r2 > 0.75:
    assessment = "Good - Can be used with caution"
else:
    assessment = "Needs improvement - Retraining is recommended"

report += f"  {assessment}\n\n"

report += f"""
{'=' * 80}
2. Detailed Error Analysis
{'=' * 80}

Error Statistics:
-------------------------------------------------------------------------------
  - Mean Error:                              {errors.mean():>8.2f} L/s
  - Error Standard Deviation:                {errors.std():>8.2f} L/s
  - Max Over-prediction Error:               {errors.max():>8.2f} L/s
  - Max Under-prediction Error:              {errors.min():>8.2f} L/s
  - Mean Absolute Error:                     {abs_errors.mean():>8.2f} L/s
  - Median Absolute Error:                   {np.median(abs_errors):>8.2f} L/s
-------------------------------------------------------------------------------

Error Distribution by Range:
-------------------------------------------------------------------------------
"""

ranges = [
    (0, 2, "Excellent"),
    (2, 5, "Good"),
    (5, 10, "Acceptable"),
    (10, float('inf'), "Poor")
]

for low, high, quality in ranges:
    if high == float('inf'):
        count = np.sum(abs_errors >= low)
        range_str = f"  - {quality:10} ({low}+ L/s)"
    else:
        count = np.sum((abs_errors >= low) & (abs_errors < high))
        range_str = f"  - {quality:10} ({low}-{high} L/s)"

    percentage = (count / len(abs_errors)) * 100
    bar = "#" * int(percentage / 2)
    report += f"{range_str:35}: {count:>6} ({percentage:>5.1f}%) {bar}\n"

report += f"-------------------------------------------------------------------------------\n\n"

report += f"""
{'=' * 80}
3. Temporal Performance Analysis
{'=' * 80}

Model performance across test days:
-------------------------------------------------------------------------------
"""

num_days = min(7, len(y_test))
for day in range(num_days):
    day_actual = y_test[day]
    day_pred = y_pred[day]
    day_error = day_actual - day_pred
    day_mape = np.mean(np.abs(day_error / (day_actual + 1e-10))) * 100
    day_mae = np.mean(np.abs(day_error))

    report += (
        f"  Day {day + 1:2d} | "
        f"MAPE: {day_mape:>6.2f}% | "
        f"MAE: {day_mae:>6.2f} L/s | "
    )

    if day_mape < 10:
        report += "Excellent\n"
    elif day_mape < 15:
        report += "Good\n"
    else:
        report += "Poor\n"

report += f"-------------------------------------------------------------------------------\n\n"

report += f"""
{'=' * 80}
4. Critical Time Period Analysis
{'=' * 80}

Worst 5 periods by error magnitude:
-------------------------------------------------------------------------------
"""

worst_indices = np.argsort(abs_errors)[-5:][::-1]
for rank, idx in enumerate(worst_indices, 1):
    actual = y_test.flatten()[idx]
    predicted = y_pred.flatten()[idx]
    error = errors[idx]
    pct_error = percentage_errors[idx]

    report += (
        f"  {rank}. Period {idx:>6} | "
        f"Actual: {actual:>6.2f} | "
        f"Predicted: {predicted:>6.2f} | "
        f"Error: {error:>+7.2f} ({pct_error:>+6.2f}%)\n"
    )

report += f"-------------------------------------------------------------------------------\n\n"

report += f"""
{'=' * 80}
5. Recommendations
{'=' * 80}
"""

recommendations = []

if mape > 10:
    recommendations.append("  - Collect more data to improve accuracy")
if abs_errors.max() > 20:
    recommendations.append("  - Large errors detected; check for outliers")
if r2 < 0.85:
    recommendations.append("  - Consider more advanced models (LSTM, Transformer)")
if errors.std() > 5:
    recommendations.append("  - High error variance indicates instability")

if not recommendations:
    recommendations.append("  - Model performance is excellent; no critical actions required")

report += "\n".join(recommendations)

report += f"""

{'=' * 80}
6. Suggested Next Steps
{'=' * 80}

  1. Monitor model performance on new data regularly
  2. Retrain the model monthly or when data behavior changes
  3. Document all model updates and changes
  4. Implement an alert system for large errors (>15 L/s)
  5. Continuously compare predictions with actual values

{'=' * 80}
End of Report
{'=' * 80}
"""

print(report)

report_file = 'results/detailed_report.txt'
with open(report_file, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"\nReport saved to: {report_file}")

print("\nCreating detailed Excel file...")
excel_file = 'results/analysis.xlsx'

with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
    summary_df = pd.DataFrame({
        'Metric': ['MAPE', 'MAE', 'RMSE', 'R²'],
        'Value': [f'{mape:.2f}%', f'{mae:.2f} L/s', f'{rmse:.2f} L/s', f'{r2:.3f}'],
        'Evaluation': [
            'Excellent' if mape < 10 else 'Good' if mape < 15 else 'Poor',
            'Excellent' if mae < 5 else 'Good' if mae < 10 else 'Poor',
            'Excellent' if rmse < 7 else 'Good' if rmse < 12 else 'Poor',
            'Excellent' if r2 > 0.85 else 'Good' if r2 > 0.75 else 'Poor'
        ]
    })
    summary_df.to_excel(writer, sheet_name='Overall Performance', index=False)

    detailed_df = pd.DataFrame({
        'Period': range(1, len(y_test.flatten()) + 1),
        'Actual_Value': y_test.flatten(),
        'Predicted_Value': y_pred.flatten(),
        'Error': errors,
        'Absolute_Error': abs_errors,
        'Percentage_Error': percentage_errors
    })
    detailed_df.to_excel(writer, sheet_name='Detailed Predictions', index=False)

    stats_df = pd.DataFrame({
        'Statistic': [
            'Mean Error',
            'Standard Deviation',
            'Max Over-prediction',
            'Max Under-prediction',
            'Mean Absolute Error',
            'Median Absolute Error'
        ],
        'Value (L/s)': [
            errors.mean(),
            errors.std(),
            errors.max(),
            errors.min(),
            abs_errors.mean(),
            np.median(abs_errors)
        ]
    })
    stats_df.to_excel(writer, sheet_name='Statistics', index=False)

print(f"Excel file saved to: {excel_file}")

print("\n" + "=" * 80)
print("Detailed report generated successfully.")
print("=" * 80)
