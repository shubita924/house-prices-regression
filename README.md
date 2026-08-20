# House Prices — Advanced Regression Techniques

## Competition overview

The goal of Kaggle's [House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) competition is to predict residential house prices from a range of features (area, location, quality, and so on). Submissions are scored by Root Mean Squared Error on a logarithmic scale (log RMSE).

**Result: Kaggle public score ~0.135.**

## Repository structure

```
proj1/
├── experiment.ipynb   # EDA, preprocessing, feature engineering, model experiments
├── inference.ipynb    # prediction with the best model, submission generation
└── README.md
```

| File | Description |
|---|---|
| `experiment.ipynb` | Data analysis, preprocessing, feature engineering, and model experiments |
| `inference.ipynb` | Applying the final model to the test data and generating the submission |

## Cleaning and preprocessing

### Missing value analysis

The percentage of missing values was computed per column:

```python
na_percent = (train_df.isnull().sum() / len(train_df)) * 100
```

### Dropping high-missingness columns

Columns with ≥ 40% missing values were removed entirely.

Rationale: when most of a column is empty, imputing it often adds noise and degrades the model.

### Encoding categorical columns

Categorical columns were converted to numeric codes with `factorize`, assigning each category a unique integer.

### Imputing numeric NA values

For `LotFrontage`, `GarageYrBlt`, and `MasVnrArea`, the column median was used:

```python
median = X_train[col].median()
```

The median was chosen because it is more robust to outliers than the mean.

## Feature engineering

### Target transformation

```python
y = np.log1p(train_df["SalePrice"])
```

Rationale: prices are right-skewed, and the competition scores on log RMSE — so transforming the target aligns the training objective with the evaluation metric and materially improves the score.

### Transforming skewed features

Skewed columns were identified and log-transformed:

```python
skewed_feats = X_train[numeric_feats].skew()
np.log1p()
```

The aim is to compress large values, reduce the influence of outliers, and give the features a better-behaved distribution.

## Feature selection

### Correlation filter (threshold = 0.8)

Columns very strongly correlated with one another were removed.

Rationale: high correlation means redundant information, and dropping it reduces overfitting.

## Improved categorical handling

### Ordinal encoding (quality features)

The following columns have a natural ordering, so they were encoded by rank rather than arbitrarily:

`ExterQual`, `BsmtQual`, `KitchenQual`, `GarageQual`

```python
{"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}
```

### Handling missing categories

```python
fillna("None")
```

This makes `"None"` a category in its own right rather than a gap.

### One-hot encoding

Nominal categories were expanded with `pd.get_dummies()`, then the train, validation, and test frames were reconciled with `align()` so all three share the same column set.

## Model

```python
xgb.XGBRegressor(
    n_estimators=1000,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8
)
```

**Why XGBoost:** it performs well on tabular data, selects informative features automatically, and is robust to noise.

### Evaluation

RMSE on the validation set:

```python
np.sqrt(mean_squared_error(y_val, preds))
```

### Submission

Since the target was log-transformed, predictions are inverted before submission:

```python
test_preds = np.expm1(model.predict(test_df))
```

## Result

Kaggle public score: **~0.135**
