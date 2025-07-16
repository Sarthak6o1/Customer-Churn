# Customer Churn Prediction Project

This project develops a neural network model to predict customer churn based on the dataset `/kaggle/input/customer-chrun/Churn_Modelling.csv`. The workflow includes data preprocessing, model construction using Keras, and hyperparameter tuning via GridSearchCV and RandomizedSearchCV.

## Dataset
The dataset contains customer demographics and banking information, including:

- CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
- Target variable: `Exited` (1 if customer left, 0 otherwise)

## Project Overview

1. **Load Data**  
   Reads CSV directly from Kaggle input.

2. **Preprocessing**  
   - Drops irrelevant columns (`RowNumber`, `CustomerId`, `Surname`)  
   - One-hot encodes categorical columns (`Geography`, `Gender`)  
   - Standardizes numerical features  

3. **Model Development**  
   - Builds a flexible Keras Sequential model with parameterized layers, neurons, dropout, activation, optimizer, and learning rate  
   - Compiles model with binary crossentropy loss and accuracy metrics

4. **Hyperparameter Tuning**  
   - Uses `GridSearchCV` for exhaustive search on a fixed parameter grid  
   - Uses `RandomizedSearchCV` for more efficient search across distributions  
   - Tunes batch size, epochs, optimizer, learning rate, activation, neurons, dropout, and number of layers  

5. **Evaluation**  
   - Reports best hyperparameters and accuracy from both searches  
   - Evaluates best models on test set for final accuracy

## How to Run

1. Clone or download this repo.  
2. Ensure Kaggle dataset is accessible at `/kaggle/input/customer-chrun/Churn_Modelling.csv`.  
3. Install required libraries:

numpy
pandas
scikit-learn
tensorflow
matplotlib
scipy
