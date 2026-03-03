---
title: Pertemuan 3 - Mengukur Jarak
---

# Pertemuan 3 - Mengukur Jarak

Referensi: [Mulaab - Data Mining](https://mulaab.github.io/datamining/)

---
<!--
## Materi Utama

- Hubungan Data Mining dengan Data Science
- Peran Data Scientist
- Tools: Python, R, Jupyter Notebook
- Perbedaan Data Mining, Data Science, Machine Learning

---

## Catatan Kuliah

*Tambahkan catatan kamu di sini menggunakan Markdown.*

---

## Implementasi

:::{note}
Tambahkan link Google Colab kamu:
[Buka di Google Colab](https://colab.research.google.com/)
:::

```python
# Tambahkan kode implementasi di sini
```

---

## Screenshot & Lampiran

Letakkan gambar di folder `pertemuan3/` lalu tampilkan dengan:

```
{image} nama_gambar.png
:alt: Deskripsi
:width: 100%
```

Untuk file download, letakkan file di folder `pertemuan3/` lalu:

```
{download}`Nama file <nama_file.csv>`
```
-->

## Measuring Distance with Mixed Data Types

In data mining, we often encounter datasets containing **"Mixed Type"** attributes, where Nominal, Binary, Numeric, and Ordinal data are all present in one table. To calculate the total distance between two objects, we cannot use a single formula like Euclidean distance alone. Instead, we must process each attribute based on its type and then combine them.

The general approach is to calculate the dissimilarity for each attribute $f$ ($d_{ij}(f)$) and then average them. The rules for each type are:

- **Nominal & Binary Attributes**: If the values for two objects are the same, the distance is `0`. If they are different, the distance is `1`.
- **Numeric Attributes**: Since numeric data (like salary or age) can have very different scales, we must first normalize them using Z-score or Min-Max scaling to bring them into a comparable range.
- **Ordinal Attributes**: These have a specific order. Replace the labels with their ranks, map these ranks to a range between 0 and 1, and then treat the result as numeric data.

By combining these individual distances, we get a final dissimilarity value that accurately reflects how different two records are, even when their features are of different types.

---

## Python Implementation

### 1. Min-Max Normalization

This technique scales numeric data into a fixed range of $[0, 1]$.
```python
def min_max_technique(list_data):
    min_data = min(list_data)
    max_data = max(list_data)
    for x in list_data:
        val = (x - min_data) / (max_data - min_data)
        print(f"{x} : {val}")

min_max_technique([1200, 1500, 1000, 1800])
```

**Output:**
```
1200 : 0.25
1500 : 0.625
1000 : 0.0
1800 : 1.0
```

---

### 2. Z-Score Standardization

This method uses the mean ($\mu$) and standard deviation ($\sigma$) to normalize numeric data.
```python
import statistics as st

def z_score_technique(list_data):
    mean_data = st.mean(list_data)
    sv_data = st.stdev(list_data)
    for x in list_data:
        val = (x - mean_data) / sv_data
        print(f"{x} : {val}")

z_score_technique([64, 70, 72, 68, 76])
```

**Output:**
```
64 : -1.3416407864998738
70 : 0.0
72 : 0.4472135954999579
68 : -0.4472135954999579
76 : 1.3416407864998738
```

---

### 3. Ordinal Attribute Mapping

This function replaces labels with ranks and maps them to a $[0, 1]$ interval.
```python
def normalize_ordinal_ranks(labels_list):
    n = len(labels_list)
    for index, label in enumerate(labels_list):
        rank = index + 1
        val = (rank - 1) / (n - 1)
        print(f"{label} : {val}")

normalize_ordinal_ranks(['lowest', 'low', 'high', 'highest'])
```
```
lowest : 0.0
low : 0.3333333333333333
high : 0.6666666666666666
highest : 1.0
```