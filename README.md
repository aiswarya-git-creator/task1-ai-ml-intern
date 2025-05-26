# Titanic Data Cleaning & Visualization
project demonstrates how to clean and preprocess the Titanic dataset, visualize outliers using boxplots, and remove them using the IQR method.

## Features
- Load Titanic CSV dataset
- Fill missing values (Age and Embarked)
- Drop columns with too many missing values (Cabin)
- Encode categorical columns (`Sex`, `Embarked`)
- Scale features using MinMaxScaler
- Create boxplots for `Age` and `Fare`
- Add helpful annotations to boxplots
- Remove outliers using IQR method

## Files
- `Titanic-Dataset.csv`: Dataset file
- `task1.py`: Python script with full implementation
- `README.md`: This file

## How to Run

1. Install the required libraries:
   ```bash
   pip install pandas numpy seaborn matplotlib scikit-learn
