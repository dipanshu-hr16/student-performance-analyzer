import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ── Load & clean data
data = pd.read_csv("student_data.csv")
data = data.dropna()
data['marks'] = np.clip(data['marks'], 0, 100)

# ── Features and target
X = data[['study_hours', 'attendance']]
y = data['marks']

# ── Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Normalize features using training stats
X_train_mean = X_train.mean()
X_train_std  = X_train.std()

X_train_scaled = (X_train - X_train_mean) / X_train_std
X_test_scaled  = (X_test  - X_train_mean) / X_train_std

# ── Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# ── Evaluate
predictions = np.clip(model.predict(X_test_scaled), 0, 100)
mae = mean_absolute_error(y_test, predictions)
r2  = r2_score(y_test, predictions)

print("=" * 40)
print("   Student Performance Analyzer")
print("=" * 40)
print(f"  Mean Absolute Error : {mae:.2f}")
print(f"  R² Score            : {r2:.4f}")
print("=" * 40)

# ── Predict for custom input
print("\nEnter student details to predict marks:\n")

try:
    study_hours = float(input("  Study hours per day (0–12) : "))
    attendance  = float(input("  Attendance percentage (0–100) : "))
except ValueError:
    print("\n  Error: Please enter valid numeric values.")
    exit()

if not (0 <= study_hours <= 12):
    print("\n  Invalid study hours! Enter a value between 0 and 12.")
elif not (0 <= attendance <= 100):
    print("\n  Invalid attendance! Enter a value between 0 and 100.")
else:
    input_df     = pd.DataFrame([[study_hours, attendance]],
                                columns=['study_hours', 'attendance'])
    input_scaled = (input_df - X_train_mean) / X_train_std

    raw_result = model.predict(input_scaled)[0]
    result     = float(np.clip(raw_result, 0, 100))

    print(f"\n  Predicted Marks : {result:.2f} / 100")

    if result >= 85:
        grade = "A — Excellent"
    elif result >= 70:
        grade = "B — Good"
    elif result >= 55:
        grade = "C — Average"
    elif result >= 40:
        grade = "D — Below Average"
    else:
        grade = "F — Needs Improvement"

    print(f"  Grade           : {grade}")
    print()
