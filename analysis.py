import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load data
data = pd.read_csv("student_data.csv")
data = data.dropna()
data['marks'] = np.clip(data['marks'], 0, 100)

X = data[['study_hours', 'attendance']]
y = data['marks']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_mean = X_train.mean()
X_std  = X_train.std()

model = LinearRegression()
model.fit((X_train - X_mean) / X_std, y_train)

y_pred = np.clip(model.predict((X_test - X_mean) / X_std), 0, 100)

mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

print(f"Mean Absolute Error : {mae:.2f}")
print(f"R2 Score            : {r2:.4f}")

# Plots
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle('Student Performance Analyzer', fontsize=16, fontweight='bold')

# 1. Study Hours vs Marks
axes[0, 0].scatter(data['study_hours'], data['marks'], color='steelblue', alpha=0.5, s=20)
axes[0, 0].set_xlabel('Study Hours')
axes[0, 0].set_ylabel('Marks')
axes[0, 0].set_title('Study Hours vs Marks')
axes[0, 0].grid(True)

# 2. Attendance vs Marks
axes[0, 1].scatter(data['attendance'], data['marks'], color='tomato', alpha=0.5, s=20)
axes[0, 1].set_xlabel('Attendance (%)')
axes[0, 1].set_ylabel('Marks')
axes[0, 1].set_title('Attendance vs Marks')
axes[0, 1].grid(True)

# 3. Actual vs Predicted
axes[1, 0].scatter(y_test, y_pred, color='mediumseagreen', alpha=0.6, s=20)
axes[1, 0].plot([0, 100], [0, 100], 'r--', linewidth=1.5, label='Perfect fit')
axes[1, 0].set_xlabel('Actual Marks')
axes[1, 0].set_ylabel('Predicted Marks')
axes[1, 0].set_title('Actual vs Predicted')
axes[1, 0].legend()
axes[1, 0].grid(True)

# 4. Marks Distribution
axes[1, 1].hist(data['marks'], bins=25, color='mediumpurple', edgecolor='white')
axes[1, 1].axvline(data['marks'].mean(), color='red', linestyle='--',
                   label=f"Mean: {data['marks'].mean():.1f}")
axes[1, 1].set_xlabel('Marks')
axes[1, 1].set_ylabel('Number of Students')
axes[1, 1].set_title('Marks Distribution')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('analysis_report.png', dpi=130, bbox_inches='tight')
print("Saved analysis_report.png")
plt.show()
