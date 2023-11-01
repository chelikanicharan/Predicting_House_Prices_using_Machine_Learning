# Predicting_House_Prices_using_Machine_Learning

This project focuses on predicting house prices using machine learning techniques. The goal is to build a model that accurately estimates the price of a house based on various features, providing valuable insights for buyers and sellers in the real estate market.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Data Exploration](#data-exploration)
- [Data Preprocessing](#data-preprocessing)
- [Model Building](#model-building)

## Overview
House price prediction is a critical task in the real estate industry. This project utilizes machine learning to develop predictive models, assisting users in making informed decisions related to property values.

## Dataset
The dataset used for this project contains various attributes, including:
- Avg. Area Income
- Avg. Area House Age
- Avg. Area Number of Rooms
- Avg. Area Number of Bedrooms
- Area Population
- Price
- Address

You can access the dataset [https://www.kaggle.com/datasets/vedavyasv/usa-housing]

## Prerequisites
Before running the code, ensure that you have the following libraries installed:

- Pandas
- NumPy
- Seaborn
- Matplotlib
- Scikit-learn

```bash
pip install pandas numpy seaborn matplotlib scikit-learn xgboost
```

## Getting Started
1. Load the dataset using the provided link or replace it with your dataset.
2. Explore the data to understand its structure, including data types and any missing values.

## Data Exploration
- Utilize data visualization to gain insights into feature relationships and distributions.
- Generate histograms, box plots, pair plots, and correlation heatmaps.

## Data Preprocessing
- Handle missing data if necessary.
- Standardize or normalize the data.
- Encode categorical variables if applicable.

## Model Building
This project includes building multiple regression models:
- Linear Regression
- Support Vector Regressor
- Lasso Regression
- Random Forest Regressor

For each model:
1. Train the model.
2. Make predictions.
3. Evaluate the model using R-squared, mean absolute error, and mean squared error.

## 1.Introduction
    Briefly explain the importance of predicting house prices. State the purpose and scope of the document.
  # Importing Libraries and Dataset:
           Matplotlib 
           Seaborn
           Pandas
## 2.Data Preprocessing:
```bash
obj = (dataset.dtypes == 'object')
object_cols = list(obj[obj].index)
print("Categorical variables:",len(object_cols))

int_ = (dataset.dtypes == 'int')
num_cols = list(int_[int_].index)
print("Integer variables:",len(num_cols))

fl = (dataset.dtypes == 'float')
fl_cols = list(fl[fl].index)
print("Float variables:",len(fl_cols))
```
## 3.Exploratory Data Analysis:
```bash
plt.figure(figsize=(12, 6))
sns.heatmap(dataset.corr(),
cmap = 'BrBG',
fmt = '.2f',
linewidths = 2,
annot = True)
```
## 4.Data Cleaning:
```bash
dataset.drop(['Id'],axis=1,inplace=True)
```
## 5.OneHotEncoder â€“ For Label categorical features:
```bash
from sklearn.preprocessing import OneHotEncoder
s = (new_dataset.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables:")
print(object_cols)
print('No. of. categorical features: ', 
len(object_cols))
```
## 6.Splitting Dataset into Training and Testing:
 ```bash
 X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, 
 random_state=101)
 Y_train.head()
```
## 7.Standardizing the data:
```bash
Sc = StandardScaler()
X_train_scal = sc.fit_transform(X_train)
X_test_scal = sc.fit_transform(X_test)
```
## 8.Predicting Prices:
```bash
Prediction1 = model_lr.predict(X_test_scal)
```
## 9.Evaluation of Predicted Data:
```bash
Plt.figure(figsize=(12,6))
Plt.plot(np.arange(len(Y_test)), Y_test, label=â€™Actual Trendâ€™)
Plt.plot(np.arange(len(Y_test)), Prediction1, label=â€™Predicted Trendâ€™)
Plt.xlabel(â€˜Dataâ€™)
Plt.ylabel(â€˜Trendâ€™)
Plt.legend()
Plt.title(â€˜Actual vs Predictedâ€™)
```
