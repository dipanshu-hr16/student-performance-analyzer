# Student Performance Analyzer

A machine learning project that predicts student marks based on study hours and attendance using Linear Regression.

---

## Project Structure

```
student-performance-analyzer/
│
├── model.py              # Train model + predict marks for custom input
├── analysis.py           # Generate graphs and model report
├── student_data.csv      # Dataset (621 students)
└── README.md
```

---

## How It Works

The model takes two inputs:
- **Study hours per day** (0 – 12)
- **Attendance percentage** (0 – 100)

And predicts the student's expected marks out of 100.

Marks are calculated based on:
- Study hours contribute **~60%** of the score
- Attendance contributes **~40%** of the score
- Realistic noise is added to reflect natural variation

---

## Model Performance

| Metric | Value |
|---|---|
| Algorithm | Linear Regression |
| R² Score | 0.9792 |
| Mean Absolute Error | 1.98 marks |
| Dataset Size | 621 students |
| Train / Test Split | 80% / 20% |

---

## Sample Predictions

| Study Hours | Attendance | Predicted Marks |
|---|---|---|
| 12 | 100% | 99.7 / 100 |
| 12 | 97% | 98.5 / 100 |
| 10 | 95% | 87.8 / 100 |
| 8 | 80% | 71.9 / 100 |
| 5 | 60% | 49.1 / 100 |
| 3 | 40% | 31.3 / 100 |

---

## Setup & Usage

**1. Clone the repository**
```bash
git clone https://github.com/your-username/student-performance-analyzer.git
cd student-performance-analyzer
```

**2. Install dependencies**
```bash
pip install numpy pandas scikit-learn matplotlib
```

**3. Run the predictor**
```bash
python model.py
```

**4. Generate analysis graphs**
```bash
python analysis.py
```

---

## Graphs Generated

Running `analysis.py` produces a full report image (`analysis_report.png`) with 6 panels:

- Actual vs Predicted marks scatter plot
- Residuals distribution
- Study Hours vs Marks (colored by attendance)
- Attendance vs Marks (colored by study hours)
- Marks distribution across the dataset
- Model summary with coefficients and metrics

---

## Tech Stack

- **Python 3**
- **NumPy** — feature normalization and array operations
- **Pandas** — data loading and preprocessing
- **Scikit-learn** — Linear Regression, train/test split, evaluation metrics
- **Matplotlib** — data visualization

---

## Grade Scale

| Marks | Grade |
|---|---|
| 85 – 100 | A — Excellent |
| 70 – 84 | B — Good |
| 55 – 69 | C — Average |
| 40 – 54 | D — Below Average |
| 0 – 39 | F — Needs Improvement |
