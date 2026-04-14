---
title: Principal Component Analysis
---

# Principal Component Analysis

### 1. Overview
Principal Component Analysis (PCA) is a dimensionality reduction technique used to simplify complex datasets. It transforms high-dimensional data into a lower-dimensional space by finding new, uncorrelated variables called Principal Components (PCs).
* **Purpose:** Reduce data size, visualize high-dimensional data, and remove noise.
* **Goal:** Maintain maximum variation (variance) of the original data in the reduced space.
* **Math:** Uses Eigenvalues and Eigenvectors to identify the most important data directions.

---

### 2. Methodology
Process the Iris dataset in KNIME using these steps:

1. **Load Data:** Use a **CSV Reader** or **Excel Reader** node to import `iris.xlsx`.
2. **Partitioning:** Use a **Table Partitioner** node to split data (e.g., 50% training, 50% testing).
3. **Normalization:** Add a **Normalizer** node. In the selector, include: *Sepal Length, Sepal Width, Petal Length, and Petal Width*.
4. **PCA Computation:** Attach a **PCA Compute** node to the training data. Select the four numerical columns in the selector.
5. **PCA Transformation:** Use a **PCA Apply** node to transform the testing data based on the computation results.
6. **Classification:** Connect the **PCA Apply** output to a **K Nearest Neighbor (kNN)** node.
7. **Evaluation:** Use a **Scorer** node to generate the Confusion Matrix.

---

### 3. Result

#### Confusion Matrix Screenshot
```{image} confusion_matrix.png
:alt: matrix screenshot
:width: 60%
```

---

### 4. Conclusion
The 4-dimensional Iris dataset was successfully reduced using PCA. The accuracy results in the Confusion Matrix demonstrate that the Principal Components captured the necessary variation to distinguish flower species even with fewer dimensions.