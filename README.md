# Titanic Survival Prediction

## Project Overview
This project involves preprocessing the Titanic dataset and training a Random Forest classifier to predict passenger survival.

## Structured Code
The code is organized in a Jupyter notebook (`Data_preprocessing_model_training.ipynb`) that includes:
- Data loading and exploration
- Stratified splitting of the dataset
- Custom transformers for data preprocessing
- Pipeline creation for preprocessing steps
- Feature scaling
- Model training with hyperparameter tuning using GridSearchCV
- Model evaluation on the test set

## Preprocessing Steps
- Load the dataset from a CSV file.
- Perform stratified shuffle split to maintain the distribution of 'Survived', 'Pclass', and 'Sex' in train and test sets.
- Visualize the distribution of key features in training and testing sets.
- Impute missing values in the 'Age' column using mean imputation.
- One-hot encode categorical features 'Embarked' and 'Sex'.
- Drop irrelevant features such as 'Name', 'Ticket', 'Cabin', and original categorical columns.
- Scale features using StandardScaler for better model performance.

## Model Selection
- A Random Forest classifier is used for prediction.
- Hyperparameters such as number of estimators, maximum depth, and minimum samples split are tuned using GridSearchCV with 3-fold cross-validation.
- The best estimator from the grid search is selected as the final model.

## Performance Analysis
- The model is evaluated on the test set.
- The accuracy score on the test set is reported as the performance metric.

## How to Run
1. Ensure `tested.csv` is in the working directory.
2. Open and run the Jupyter notebook `Data_preprocessing_model_training.ipynb`.
3. The notebook will output the test set accuracy after training and evaluation.

## Dependencies
- numpy
- pandas
- matplotlib
- scikit-learn

Install dependencies using:
```
pip install numpy pandas matplotlib scikit-learn
