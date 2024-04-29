# Students Performance in Math Exams - Predictive Analysis

## Introduction
This repository showcases a comprehensive predictive analysis of student performance in math exams, utilizing advanced machine learning techniques for feature engineering, model development, and evaluation.

## Dataset
The project uses the "Students Performance in Exams" dataset from Kaggle, which includes demographics and scores across various subjects. Key attributes include gender, race/ethnicity, parental education, lunch type, test preparation, and academic scores. The dataset is available [here](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams).

## Technical Overview

### Data Preprocessing
- **Normalization and Encoding**: Applied `LabelEncoder` and standard scaling to transform categorical data into a machine-readable format.
- **Outlier Management**: Identified and handled outliers using z-scores to ensure robust model performance.

### Feature Selection
- **Statistical Tests & Recursive Feature Elimination**: Employed Chi-Square tests and RFE to determine significant predictors, enhancing model accuracy.

### Model Development & Evaluation
- **Ordinary Least Squares (OLS)**: Implemented with cross-validation to estimate model parameters and assess performance using RMSE, MAE, and R-squared metrics.
- **Neural Networks**: Configured and optimized a neural network in Keras, employing techniques like early stopping to prevent overfitting.
- **Ensemble Modeling**: Developed a stacked model that combines multiple base models to leverage their strengths and improve accuracy.

### Hyperparameter Tuning
- **GridSearchCV**: Performed an exhaustive search over specified model parameters to find the most optimal settings, particularly for complex models like RandomForest and AdaBoost.

## Repository Structure
- `DS_Exams_Report.pdf`: A detailed analytical report.
- `Production.py`: Script for deploying models into production.
- `TrainingScript.py`: Contains the complete workflow for training models.
- `data/`: Includes `train.csv` for training and `test.csv` for model validation.
- `models/`: Contains serialized models (`*.pkl`, `*.h5`) for production use.
- `visualizations/`: Contains code for visualizations employed in Exploratory Data Analysis in `DS_Exams_Report.pdf`

## Highlights
- **Predictive Power**: Demonstrated strong predictive performance, especially with the stacked model approach.
- **Insights**: Identified critical predictors impacting math scores, providing actionable insights into student performance.

## Conclusion
This project illustrates the application of sophisticated statistical techniques and machine learning algorithms to predict educational outcomes, showcasing skills crucial for advanced data analysis roles.

## Acknowledgements
Data provided by [Kaggle](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams).
