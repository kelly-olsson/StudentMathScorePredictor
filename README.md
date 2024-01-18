# Students Performance in Math Exams - Predictive Analysis

## Introduction
This repository contains an analysis of the "Students Performance in Exams" dataset, focusing on predicting math scores. The analysis includes machine learning techniques for feature selection and evaluation of predictive algorithms.

## Dataset Overview
The dataset includes student demographics and academic scores across various subjects: [Kaggle Dataset](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)

Columns include: 
- `gender`: Student's gender
- `race/ethnicity`: Categorized race/ethnicity of the student
- `parental level of education`: Highest education level obtained by the student's parents
- `lunch`: Type of lunch received (standard or free/reduced)
- `test preparation course`: Completion status of test preparation course
- `math score`: Score in the math exam
- `reading score`: Score in the reading exam
- `writing score`: Score in the writing exam

## Key Techniques and Processes
1. **Data Preprocessing**:
   - Conversion of string columns to numeric using `LabelEncoder`.
   - Identification and handling of outliers based on z-scores.
   - Imputation of missing values and data normalization.

2. **Feature Selection**:
   - Utilization of techniques like RFE, FFS, and Chi-Square Test to identify significant predictors for math scores.

3. **Model Development**:
   - **Ordinary Least Squares (OLS) Model**: Statistical approach for linear regression, including cross-validation and performance metrics (RMSE, MAE, R-squared).
   - **Neural Network Model**: Developed using Keras with early stopping and model checkpointing, optimized through GridSearchCV.
   - **Stacked Model**: Combination of various base models to improve prediction accuracy. 

4. **Hyperparameter Tuning**:
   - Exhaustive search over specified parameter values for models like RandomForestRegressor, AdaBoostRegressor, DecisionTreeRegressor, etc.

## Repository Structure
- `DS_Exams_Report.pdf`: Detailed report of the exploratory data analysis.
- `Production.py`: Python script for production-level model predictions.
- `TrainingScript.py`: Python script used for training the models.
- `train.csv`: Training dataset used for model development.
- `test.csv`: Test dataset used for evaluating the model.
- `BinaryFolder/`: Directory containing serialized models and scaler objects.
  - `base_model1.pkl` to `base_model7.pkl`: Serialized base models for the stacked model.
  - `m1_ols_model`: Serialized Ordinary Least Squares (OLS) model.
  - `m2_nn_model.h5`: Serialized Neural Network model in HDF5 format.
  - `stacked_model`: Serialized stacked model combining various base models.
  - `sc_x.pkl`: Serialized scaler object for data normalization.

## Objectives
The goal is to utilize statistical and machine learning methods to predict math scores based on various student attributes.

## Key Findings
- Significant predictors include reading and writing scores, gender, and test preparation course status.
- The stacked model showed the best performance in predicting math scores.

## Conclusion
This project demonstrates the application of feature selection, model tuning, and ensemble methods in predictive modeling, crucial for roles in data analysis and data quality assurance.

Dataset Source: [Kaggle Dataset](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
