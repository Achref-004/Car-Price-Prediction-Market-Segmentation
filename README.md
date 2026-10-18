<div align="center">

# Car Price Prediction & Market Segmentation

### Linear Regression, Principal Component Analysis, and Hierarchical Clustering

An educational machine-learning notebook that cleans used-car listings, predicts vehicle prices with linear regression, and explores market segments using PCA and hierarchical clustering.

</div>

---

## Table of Contents

- [Project Overview](#project-overview)
- [Implemented Workflow](#implemented-workflow)
- [Dataset](#dataset)
- [Exploratory Analysis and Visualizations](#exploratory-analysis-and-visualizations)
- [Models and Methodology](#models-and-methodology)
- [Evaluation](#evaluation)
- [Clustering and Market Segmentation](#clustering-and-market-segmentation)
- [Technologies](#technologies)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [Running the Notebook](#running-the-notebook)
- [Reproducibility and Data Requirements](#reproducibility-and-data-requirements)
- [Limitations and Future Improvements](#limitations-and-future-improvements)
- [Author](#author)

## Project Overview

This project analyses a used-car listings dataset and demonstrates two related machine-learning tasks:

1. **Price prediction:** fit linear regression models to estimate the `Price` column.
2. **Market segmentation:** reduce the feature space with Principal Component Analysis (PCA), then group vehicles with hierarchical clustering.

The project is delivered as a Jupyter notebook. It includes data inspection, feature preparation, regression evaluation, residual analysis, a PCA comparison, and visual summaries of the clusters.

## Implemented Workflow

### Data loading and inspection

- Loads `car_price_prediction.csv` into a pandas DataFrame.
- Reports the dataset dimensions, data types, column names, and missing-value counts.
- Inspects the numeric distributions and special values such as `-` in `Levy` and the `km` suffix in `Mileage`.

### Data preparation

The notebook's `prepare_data` function:

- Converts `Levy`, `Mileage`, `Engine volume`, and `Cylinders` to numeric values.
- Removes the `ID` column.
- Applies explicit filters (`Price < 100000`, `Mileage < 500000`, and `Engine volume < 10`), then filters numeric columns using the IQR rule.
- Imputes remaining numeric missing values with the median.
- Fills missing categorical values with `Unknown` and replaces categories occurring in less than 1% of the data with `Other`.
- One-hot encodes categorical features with `drop_first=True`.

The notebook's saved output reports **19,237 rows and 18 columns** before preparation, and **8,653 rows and 67 columns** afterwards. The target is `Price`; the remaining 66 columns are model features.

### Regression and segmentation

- Trains a full-feature linear regression model using an 80/20 train/test split.
- Evaluates it with a five-fold cross-validation.
- Standardizes the prepared features, applies PCA, and trains another linear regression model using only the first two components.
- Uses Ward hierarchical clustering on those two components and assigns three clusters for visualization and profiling.

## Dataset

The notebook expects a file named `car_price_prediction.csv`. The dataset is **not included in this repository**, and the notebook does not record a public download URL.

The notebook currently loads the CSV using a machine-specific Windows path:

```python
df = pd.read_csv(r"C:\Users\ASUS\Downloads\ML\car_price_prediction.csv")
```

To run it on another machine, place the CSV in the repository root and change the line to:

```python
df = pd.read_csv("car_price_prediction.csv")
```

The notebook references these columns:

`ID`, `Price`, `Levy`, `Manufacturer`, `Model`, `Prod. year`, `Category`, `Leather interior`, `Fuel type`, `Engine volume`, `Mileage`, `Cylinders`, `Gear box type`, `Drive wheels`, `Doors`, `Wheel`, `Color`, and `Airbags`.

## Exploratory Analysis and Visualizations

The notebook includes:

- Dataset dimensions, data types, missing-value checks, and descriptive statistics.
- A plot of residuals against predictions and a residual histogram.
- A bar chart of the ten largest absolute linear-regression coefficients.
- PCA explained-variance output.
- A Ward-linkage dendrogram and a PC1-versus-PC2 scatter plot colored by cluster.
- Cluster price and manufacturer charts.
- A comparison chart of the regression models' R² scores.

## Models and Methodology

| Model or analysis | Configuration in the notebook | Input |
|---|---|---|
| Full-feature regression | `LinearRegression()`; 80/20 split with `random_state=42` | All 66 prepared features |
| Full-feature cross-validation | `KFold(n_splits=5, shuffle=True, random_state=42)` | All prepared features |
| PCA regression | `StandardScaler`, then `PCA`; `LinearRegression()` on PC1 and PC2; 80/20 split with `random_state=42` | First two principal components |
| PCA cross-validation | Five-fold `KFold` using the same split configuration | First two principal components |
| Hierarchical clustering | Ward linkage; `AgglomerativeClustering(n_clusters=3, linkage="ward")` | PC1 and PC2 |

The first two principal components explain **11.04%** of the variance in the standardized features (PC1: 5.91%; PC2: 5.13%), according to the notebook's saved output.

## Evaluation

The regression models are evaluated with R², root mean squared error (RMSE), and mean absolute error (MAE). The values below are the outputs saved in the notebook; they may change if the input data or preprocessing is changed.

| Model | Evaluation | R² | RMSE | MAE |
|---|---|---:|---:|---:|
| All features | 80/20 holdout | 0.5166 | 8,387.43 | 6,479.42 |
| All features | 5-fold cross-validation | 0.5290 ± 0.0175* | 8,155.39 | Not reported |
| PCA (PC1 + PC2) | 80/20 holdout | 0.1323 | 11,237.20 | 9,184.75 |
| PCA (PC1 + PC2) | 5-fold cross-validation | 0.1349 ± 0.0197* | 11,052.52 | Not reported |

The full-feature model performs better than the two-component PCA model in the saved results. The notebook's cross-validation R² interval is printed as **mean ± two standard deviations**, not as a confidence interval.

## Clustering and Market Segmentation

The notebook builds a Ward dendrogram, then fits a three-cluster agglomerative model to PC1 and PC2. Its saved output reports the following cluster counts and mean prices:

| Cluster | Rows assigned | Reported mean `Price` |
|---:|---:|---:|
| 0 | 4,109 | 17,516.35 |
| 1 | 2,625 | 17,148.60 |
| 2 | 1,919 | 16,758.40 |

**Important:** the cluster profile statistics should not be treated as reliable yet. The notebook assigns cluster labels to the first 8,653 rows of the original, unfiltered DataFrame instead of preserving and using the row indices of the records that survived cleaning. The saved cluster output also reports mileage averages above the `< 500000` preprocessing filter, which is a sign that the cluster labels and vehicle records are misaligned. The cluster counts above describe the generated labels, but the price and other segment profiles need to be recalculated after fixing that alignment.

## Technologies

The notebook uses:

- Python
- NumPy
- pandas
- Matplotlib
- Seaborn
- scikit-learn
- SciPy
- Jupyter Notebook or JupyterLab

## Project Structure

The repository currently contains:

```text
.
├── Projet ML.ipynb
└── README.md
```

The required CSV dataset is not tracked in this repository.

## Installation and Setup

Create and activate a virtual environment, then install the notebook dependencies.

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib seaborn scikit-learn scipy jupyter
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib seaborn scikit-learn scipy jupyter
```

## Running the Notebook

1. Put `car_price_prediction.csv` in the repository root.
2. Update the `pd.read_csv` path as described in [Dataset](#dataset).
3. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

   Or start JupyterLab:

   ```bash
   jupyter lab
   ```

4. Open `Projet ML.ipynb` and run the cells from top to bottom.

## Reproducibility and Data Requirements

- The CSV file must have the columns used by the notebook and is not supplied here.
- Train/test splits use `random_state=42` and a 20% test set.
- Cross-validation uses five shuffled folds with `random_state=42`.
- The hierarchical clustering uses Ward linkage and three clusters.
- The saved results were generated with a particular local copy of the dataset; the notebook does not pin package versions or document the dataset source.
- Preprocessing, scaling, and PCA are fitted before the train/test split or cross-validation. This can leak information from evaluation data into training and make the reported scores optimistic; see [Limitations and Future Improvements](#limitations-and-future-improvements).

## Limitations and Future Improvements

- Replace the hard-coded local CSV path with a portable path or a configurable data location, and document the dataset source and license.
- Split the data before fitting preprocessing steps. Put imputation, encoding, scaling, and PCA into scikit-learn `Pipeline`/`ColumnTransformer` objects so they are refitted inside each training fold.
- Preserve row indices through filtering and clustering, then attach cluster labels to the matching records before profiling segments.
- Evaluate how many PCA components are needed to retain a useful share of the variance; the first two currently retain only 11.04%.
- Compare against regularized regression and non-linear models, and report reproducible cross-validation metrics for MAE as well as RMSE and R².
- Add a small executable script or tests for the preprocessing and evaluation steps if the project is intended for repeated use.

## Author

- **Name:** Achref Bejaoui
- **GitHub:** [Achref-004](https://github.com/Achref-004)
- **LinkedIn:** [achref-bejaoui00](https://linkedin.com/in/achref-bejaoui00)
