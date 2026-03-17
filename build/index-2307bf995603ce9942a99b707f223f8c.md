---
title: Pertemuan 4 - KNN Imputation on Iris Dataset
---

# Pertemuan 4 - KNN Imputation on Iris Dataset

Referensi: [Mulaab - Data Mining](https://mulaab.github.io/datamining/)

---

## Python Implementation
[Open in Google Colab](https://colab.research.google.com/drive/1iRM9_cHJJNJ8uz4DSCpCuJit78LJ0jFo?usp=sharing)

### KNN Imputation on Iris Dataset
```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.impute import KNNImputer

# Load iris data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Take only 5 rows
df = df.head(5)
print("Original Data:")
print(df)

# Randomly delete some values
np.random.seed(42)
df_missing = df.copy()
df_missing.iloc[3, 2] = np.nan  # delete row 3, petal length
df_missing.iloc[1, 0] = np.nan  # delete row 1, sepal length

print("\nData with Missing Values:")
print(df_missing)

print("\nMissing Value Locations:")
print(df_missing.isnull().sum())

# KNN Imputation
imputer = KNNImputer(n_neighbors=2)
df_filled = pd.DataFrame(imputer.fit_transform(df_missing), columns=iris.feature_names)

print("\nAfter KNN Imputation:")
print(df_filled)

# Show only what changed
print("\nFilled Values:")
for col in df.columns:
    for idx in df_missing[df_missing[col].isnull()].index:
        print(f"  Row {idx}, '{col}': {df_filled.iloc[idx][col]:.4f}")
```
```
Original Data:
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
0                5.1               3.5                1.4               0.2
1                4.9               3.0                1.4               0.2
2                4.7               3.2                1.3               0.2
3                4.6               3.1                1.5               0.2
4                5.0               3.6                1.4               0.2

Data with Missing Values:
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
0                5.1               3.5                1.4               0.2
1                NaN               3.0                1.4               0.2
2                4.7               3.2                1.3               0.2
3                4.6               3.1                NaN               0.2
4                5.0               3.6                1.4               0.2

Missing Value Locations:
sepal length (cm)    1
sepal width (cm)     0
petal length (cm)    1
petal width (cm)     0
dtype: int64

After KNN Imputation:
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
0               5.10               3.5               1.40               0.2
1               4.65               3.0               1.40               0.2
2               4.70               3.2               1.30               0.2
3               4.60               3.1               1.35               0.2
4               5.00               3.6               1.40               0.2

Filled Values:
  Row 1, 'sepal length (cm)': 4.6500
  Row 3, 'petal length (cm)': 1.3500

```


## Iris Data Imputation in Orange

### Step 1: Load Data
- Add **File** widget and import `IRIS.csv`.

### Step 2: Check Missing Values (Before)
- Connect **File → Feature Statistics**
- Result: Observe the "Missing" column showing empty values (e.g., in `petal_width`).

![Showing missing values](before.png)
### Step 3: Set Target Variable
- Connect **File → Select Columns**
- Move `species` to the **Target** box.
- Keep numeric attributes in **Features**.

![Select columns](select_col.png)

### Step 4: Impute Missing Data
- Connect **Select Columns → Impute**
- Settings: Select **Model-based imputer**

![Impute](impute.png)


### Step 5: Verify Results (After)
- Connect **Impute → Feature Statistics**
- Result: The "Missing" column should now show **0** for all features.

![Showing missing values are filled](after.png)

### Workflow
![Workflow](workflow.png)
### File Download

- {download}`IRIS.csv <IRIS.csv>`
- {download}`Orange File <pertemuan_4.ows>`