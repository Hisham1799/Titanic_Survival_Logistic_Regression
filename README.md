# Titanic Disaster Survival Prediction Using Logistic Regression

This project involves predicting the survival of passengers on the Titanic using a Logistic Regression model. By analyzing various features from the Titanic dataset, we aim to identify factors that influenced survival and build a predictive model.

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Libraries Used](#libraries-used)
4. [Data Exploration](#data-exploration)
5. [Data Cleaning](#data-cleaning)
6. [Feature Engineering](#feature-engineering)
7. [Model Building](#model-building)
8. [Evaluation](#evaluation)
9. [Results](#results)
10. [How to Run](#how-to-run)
11. [References](#references)

## Introduction

The Titanic disaster is one of the most infamous shipwrecks in history. In this project, we use machine learning techniques to predict whether a passenger survived the Titanic disaster based on features such as age, sex, passenger class, and others. Logistic Regression is employed due to its effectiveness in binary classification tasks.

## Dataset

The dataset used for this project is the well-known Titanic dataset, which can be found on [Kaggle's Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic) competition. It contains data about passengers, such as their age, gender, class, etc., and whether they survived the disaster.

- **Total number of samples**: 891
- **Features**: 11 (after preprocessing)
- **Target variable**: `Survived` (0 = No, 1 = Yes)

## Libraries Used

- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical operations.
- `seaborn`: For data visualization.
- `matplotlib`: For plotting graphs.
- `scikit-learn`: For implementing the machine learning model.

## Data Exploration

We performed an initial exploration of the dataset to understand its structure and contents:

1. **Viewing the dataset**: Used `head()` to get a quick look at the first few rows of the data.
2. **Data information**: Used `info()` and `describe()` to get insights into the data types and summary statistics.
3. **Visualization**: Used `seaborn` count plots to understand the distribution of survival among different categories (e.g., male vs. female).

Example visualization code:

```python
# Plot showing survival counts
sns.countplot(x='Survived', data=titanic_data)
```

## Data Cleaning

Data cleaning is a crucial step to handle missing values and irrelevant features:

1. **Missing Values**: Identified missing values using `isna().sum()` and visualized them with a heatmap.
2. **Filling Missing Values**: Filled missing values in the `Age` column with the mean age.
3. **Dropping Irrelevant Features**: Dropped features like `Cabin`, which had too many missing values, and other non-numeric features that weren't essential for prediction.

```python
# Fill missing age values with the mean age
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)
```

## Feature Engineering

Converted non-numeric data into numeric form for model processing:

- Used `pd.get_dummies()` to convert the `Sex` column into a numerical format.
- Dropped columns that were not useful for the prediction model.

```python
# Convert the 'Sex' column to numerical
titanic_data['Gender'] = pd.get_dummies(titanic_data['Sex'], drop_first=True)
```

## Model Building

We built the model using Logistic Regression:

1. **Data Splitting**: Split the data into training and testing sets using `train_test_split()`.
2. **Model Training**: Trained the Logistic Regression model using the training set.
3. **Prediction**: Made predictions on the test set.

```python
# Train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Fit Logistic Regression
lr = LogisticRegression()
lr.fit(x_train, y_train)
```

## Evaluation

Evaluated the model using:

- **Confusion Matrix**: To understand the true positives, true negatives, false positives, and false negatives.
- **Classification Report**: Provided precision, recall, and F1-score for the model.

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, predict))
```

## Results

The Logistic Regression model provided reasonable accuracy and insight into the factors affecting survival:

- **Precision**: The model's ability to correctly identify those who survived.
- **Recall**: The model's ability to capture all actual survivors.
- **F1 Score**: The balance between precision and recall.

**Note**: Accuracy could potentially be improved by incorporating more features or using more complex models.

## How to Run

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/titanic-survival-logistic-regression.git
    ```

2. Install the required Python libraries:

    ```bash
    pip install pandas numpy seaborn matplotlib scikit-learn
    ```

3. Run the Jupyter Notebook:

    ```bash
    jupyter notebook
    ```

4. Open the `Titanic_Survival_Logistic_Regression.ipynb` file and execute the cells.

## References

- Kaggle Titanic Competition: [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)
- Scikit-Learn Documentation: [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

---

This README provides a comprehensive overview of the Titanic survival prediction project using Logistic Regression. For detailed implementation, visualizations, and insights, refer to the project notebook.
