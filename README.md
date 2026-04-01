# ML Practice 1 - Iris Dataset Classification

## Overview

A hands-on machine learning project using the classic Iris dataset to practise the full ML workflow, from data loading and exploration through to model training, evaluation, and visualisation. Two classifiers are compared: Logistic Regression and K-Nearest Neighbours (KNN).

---

## Dataset

**Source:** Scikit-learn built-in (`sklearn.datasets.load_iris`)

| Property | Detail |
|---|---|
| Samples | 150 |
| Features | 4 (sepal length, sepal width, petal length, petal width) |
| Classes | 3 (setosa, versicolor, virginica) |
| Class balance | 50 samples per class |
| Missing values | None |

---

## Workflow

### 1. Data Loading & Exploration
- Loaded the Iris dataset using `load_iris()`
- Created a pandas DataFrame with named feature columns
- Inspected shape, data types, and summary statistics (`df.info()`, `df.describe()`)
- Checked class distribution with `value_counts()`

### 2. Preprocessing
- Split features (`X`) and labels (`y`)
- Applied an 80/20 train/test split (`random_state=42` for reproducibility)
- Standardised features using `StandardScaler` — fit on training data only, applied to both sets

### 3. Model Training
Two models were trained on the same data:
- **Logistic Regression** (`max_iter=200`)
- **K-Nearest Neighbours** (`n_neighbors=5`)

### 4. Evaluation
Both models were evaluated using:
- Classification report (precision, recall, F1-score, support)
- Confusion matrix (visualised with `ConfusionMatrixDisplay`)

### 5. Visualisation
- Pairplot (`seaborn`) to inspect feature separability across species

---

## Results

| Model | Accuracy |
|---|---|
| Logistic Regression | 100% |
| KNN | 100% |

Both models achieved perfect classification on the 30-sample test set. This is expected for the Iris dataset — petal features provide very clean separation between classes, making it an ideal learning dataset.

---

## Key Observations from the Pairplot

- **Setosa** is completely separable from the other two species in petal measurements
- **Versicolor** and **Virginica** overlap slightly but are largely distinguishable
- **Petal length vs petal width** is the most informative feature pair
- **Sepal measurements** show more overlap between classes

---

## Libraries Used

```
scikit-learn
pandas
matplotlib
seaborn
```

---

## How to Run

```bash
pip install scikit-learn pandas matplotlib seaborn
```

Open `ML_PRACTICE_1_IRIS.ipynb` in VS Code and run all cells sequentially.
