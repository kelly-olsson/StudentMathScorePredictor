
import time
import numpy as np
import pandas as pd
import statsmodels.api as sm
import pickle
from math import sqrt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import RFE, SelectKBest, chi2, f_regression
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scikeras.wrappers import KerasRegressor
from keras.layers import Dense, Dropout
from keras.models import Sequential, load_model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2
from matplotlib import pyplot

df = pd.read_csv('train.csv')

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# print the number of unique values in each column
# print(df.nunique())

# Function to convert string columns to numerics
def convert_string_columns_to_numeric(df):
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])
    return df

# Convert string columns to numerics
df = convert_string_columns_to_numeric(df)

# Check for missing data and see other metrics
# print(df.describe().T)

print(df)

###### Uncomment to View Outliers ##########

# # select columns with numerical data
# num_cols = ['math score', 'reading score', 'writing score']
# data = df[num_cols]
#
# # calculate z-scores for each value in the dataset
# z_scores = np.abs((data - data.mean()) / data.std())
#
# # identify outliers using a threshold of 3 standard deviations from the mean
# outliers = df[(z_scores >= 3).any(axis=1)]
#
# # # print the columns that contain outliers in each row
# for idx, row in outliers.iterrows():
#     print(f"Row {idx}:")
#     for col in num_cols:
#         if z_scores.loc[idx, col] >= 3:
#             print(f"   Column '{col}' is an outlier with z-score {z_scores.loc[idx, col]}")

####### End of Outlier Calculation Section ########

########### Feature Selection ##########

# # Separate target variable from predictors
# X = df.drop('math score', axis=1)
# y = df['math score']
#
# # Feature selection with RFE
# rfe_selector = RFE(estimator=LinearRegression(), n_features_to_select=3)
# rfe_selector.fit(X, y)
# rfe_support = rfe_selector.get_support()
# rfe_features = X.loc[:, rfe_support].columns.tolist()
# print(f"RFE Features: {rfe_features}")
#
# # Feature selection with FFS
# ffs_selector = SelectKBest(score_func=f_regression, k=3)
# ffs_selector.fit(X, y)
# ffs_support = ffs_selector.get_support()
# ffs_features = X.loc[:, ffs_support].columns.tolist()
# print(f"FFS Features: {ffs_features}")
#
# # Feature selection with Chi-Square
# chi2_selector = SelectKBest(score_func=chi2, k=3)
# chi2_selector.fit(X, y)
# chi2_support = chi2_selector.get_support()
# chi2_features = X.loc[:, chi2_support].columns.tolist()
# print(f"Chi-Square Features: {chi2_features}")




########### First Model: OLS ##########

# Statistically significant features found
features = ['writing score', 'reading score', 'test preparation course']

# PART B
X = df[features]

# Scale the features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)
pickle.dump(scaler, open('BinaryFolder/sc_x.pkl', 'wb'))

# Adding an intercept *** This is required ***. Don't forget this step.
# The intercept centers the error residuals around zero
# which helps to avoid over-fitting.
X_ols = sm.add_constant(X_scaled)
y = df['math score']

# Set up cross-fold validation
kfold = KFold(n_splits=5, shuffle=True)

rmse_scores = []
mae_scores = []
r2_scores = []

# Perform cross-fold validation
for train_index, test_index in kfold.split(X_ols):
    X_train, X_test = X_ols.iloc[train_index], X_ols.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = sm.OLS(y_train, X_train).fit()
    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    rmse_scores.append(rmse)
    mae_scores.append(mae)
    r2_scores.append(r2)

# Fit the model and print the summary
ols_model = sm.OLS(y, X_ols)
ols_model_fitted = ols_model.fit()
with open('BinaryFolder/m1_ols_model', 'wb') as files:
    pickle.dump(ols_model_fitted, files)

# Load saved model
with open('BinaryFolder/m1_ols_model', 'rb') as f:
    loadedModel = pickle.load(f)

print(loadedModel.summary())

# Calculate the average and standard deviation of RMSE, MAE, and R-squared across all cross-validation folds
average_rmse = np.mean(rmse_scores)
std_rmse = np.std(rmse_scores)
average_mae = np.mean(mae_scores)
std_mae = np.std(mae_scores)
average_r2 = np.mean(r2_scores)
std_r2 = np.std(r2_scores)

print(f"Average RMSE: {average_rmse}, Standard deviation: {std_rmse}")
print(f"Average MAE: {average_mae}, Standard deviation: {std_mae}")
print(f"Average R^2: {average_r2}, Standard deviation: {std_r2}\n")





########### Second Model: NN ##########

# Split the data.
X_train, X_temp, y_train, y_temp = train_test_split(X,
                                                    y, test_size=0.3, random_state=0)
X_test, X_val, y_test, y_val = train_test_split(X_temp,
                                                y_temp, test_size=0.5, random_state=0)

normSizeEvaluations = []

# Define the model.
model = Sequential()
model.add(Dense(10, input_dim=3, activation='relu',
                kernel_initializer='he_uniform',
                kernel_regularizer=l2(0.02))) # Added L2 regularization
model.add(Dense(10, activation='relu',
                kernel_initializer='he_uniform')) # hidden layer
model.add(Dense(10, activation='relu',
                kernel_initializer='he_uniform'))
model.add(Dense(1, activation='linear'))

# Compile the model.
model.compile(loss='mean_squared_error',
              optimizer=SGD(learning_rate=0.00001, momentum=0.9))

# reshape 1d arrays to 2d arrays
y_train = np.array(y_train.values.tolist())
y_test = np.array(y_test.values.tolist())
y_val = np.array(y_val.values.tolist())
trainy = y_train.reshape(len(y_train), 1)
testy = y_test.reshape(len(y_test), 1)
valy = y_val.reshape(len(y_val), 1)

# configure early stopping
es = EarlyStopping(monitor='val_loss', patience=300)
mc = ModelCheckpoint('BinaryFolder/m2_nn_model.h5', monitor='val_loss', mode='max', verbose=1,
                     save_best_only=True)

history = model.fit(X_train, y_train, epochs=2000, verbose=0, validation_data=(X_val, valy), callbacks=[es,mc])

# Load the pickled model
nn_model = load_model('BinaryFolder/m2_nn_model.h5')

# Set up the KFold cross-validator
kf = KFold(n_splits=5, random_state=0, shuffle=True)
X_list = np.array(X.values.tolist())
y_list = np.array(y.values.tolist())

rmse_scores = []
mae_scores = []
r2_scores = []
best_rmse = float('inf')
best_history = None

for train_index, test_index in kf.split(X_list):
    X_train, X_test = X_list[train_index], X_list[test_index]
    y_train, y_test = y_list[train_index], y_list[test_index]

    # Evaluate the loaded model on the fold
    y_pred = nn_model.predict(X_test)

    rmse = sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    rmse_scores.append(rmse)
    mae_scores.append(mae)
    r2_scores.append(r2)

    if rmse < best_rmse:
        best_rmse = rmse
        best_history = history

# Calculate average and standard deviation for each metric
avg_rmse = np.mean(rmse_scores)
std_rmse = np.std(rmse_scores)
avg_mae = np.mean(mae_scores)
std_mae = np.std(mae_scores)
avg_r2 = np.mean(r2_scores)
std_r2 = np.std(r2_scores)

print("Average RMSE: {:.3f}, Standard deviation: {:.3f}".format(avg_rmse, std_rmse))
print("Average MAE: {:.3f}, Standard deviation: {:.3f}".format(avg_mae, std_mae))
print("Average R^2: {:.3f}, Standard deviation: {:.3f}".format(avg_r2, std_r2))

# Plot the loss during training for the best RMSE model
pyplot.title('Mean Squared Error - Best RMSE Model')
pyplot.plot(best_history.history['loss'], label='train')
pyplot.plot(best_history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()



### Grid-Searching for Optimized Hyperparameters for NN (Model 2)

# # Function to create the model
# def create_model(nodes=25, learning_rate=0.01, layers=1, activation='relu', kernel_initializer='he_uniform'):
#     model = Sequential()
#     model.add(Dense(nodes, input_dim=3, activation=activation, kernel_initializer=kernel_initializer))
#
#     for _ in range(layers - 1):
#         model.add(Dense(nodes, activation=activation, kernel_initializer=kernel_initializer))
#
#     model.add(Dense(1, activation='linear'))
#
#     optimizer = SGD(learning_rate=learning_rate + 1e-8, momentum=0.9, clipvalue=0.5)
#     model.compile(loss='mean_squared_error', optimizer=optimizer)
#
#     return model
#
#
# # Function to perform grid search
# def perform_grid_search(trainX, trainy):
#     param_grid = {
#         'model__nodes': [10, 25, 50],
#         'model__learning_rate': [0.001, 0.01, 0.1],
#         'model__layers': [1, 2, 3],
#         'model__activation': ['relu', 'tanh'],
#         'model__kernel_initializer': ['he_uniform', 'glorot_uniform']
#     }
#
#     model = KerasRegressor(model=create_model, epochs=200, verbose=0)
#     grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
#     grid_result = grid.fit(trainX, trainy)
#
#     return grid_result
#
#
# # Load and preprocess the data
# X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=0)
# X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)
#
# # reshape 1d arrays to 2d arrays
# y_train = np.array(y_train.values.tolist())
# y_test = np.array(y_test.values.tolist())
# trainy = y_train.reshape(len(y_train), 1)
# testy = y_test.reshape(len(y_test), 1)
#
# scaler = StandardScaler()
# scaler.fit(trainy)
# trainy = scaler.transform(trainy)
# testy = scaler.transform(testy)
#
# xscaler = StandardScaler()
# xscaler.fit(X_train)
# trainX = xscaler.transform(X_train)
# testX = xscaler.transform(X_test)
#
# # Perform the grid search
# start_time = time.time()
# grid_result = perform_grid_search(trainX, trainy)
# end_time = time.time()
#
# # Calculate the time it took
# elapsed_time = end_time - start_time
#
# # Print the results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))
#
# # Print the elapsed time
# print("Elapsed time: %.2f seconds" % elapsed_time)
#
# # Took about 18 minutes to run
# # Best hyperparams result: 0.076292 using {'model__activation': 'relu', 'model__kernel_initializer': 'he_uniform', 'model__layers': 1, 'model__learning_rate': 0.001, 'model__nodes': 10}

###### End Hyperparameter Grid Searching Section  #########






####### Third Model: Stacked Model ##########

def k_fold_train_validate(models, X_stack_train, y_stack_train, k=5):
    kf = KFold(n_splits=k)
    model_metrics = {model.__class__.__name__: [] for model in models}
    stacked_model_name = 'STACKED MODEL LinearRegression'
    model_metrics[stacked_model_name] = []

    trained_base_models = {model.__class__.__name__: [] for model in models}

    for train_index, val_index in kf.split(X_stack_train):
        X_train, X_val = X_stack_train.iloc[train_index], X_stack_train.iloc[val_index]
        y_train, y_val = y_stack_train.iloc[train_index], y_stack_train.iloc[val_index]

        dfPredictions = pd.DataFrame()

        # Train and validate base models
        for model in models:
            model.fit(X_train, y_train)

            predictions = model.predict(X_val)

            # Calculate metrics
            rmse = round(np.sqrt(mean_squared_error(y_val, predictions)), 3)
            mae = round(mean_absolute_error(y_val, predictions), 3)
            r2 = round(r2_score(y_val, predictions), 3)

            model_metrics[model.__class__.__name__].append((rmse, mae, r2))

            colName = str(len(dfPredictions.columns))
            dfPredictions[colName] = predictions

            # Store trained base models
            trained_base_models[model.__class__.__name__].append(model)

        # Train and validate stacked model
        stacked_model = fitStackedModel(dfPredictions, y_val)
        stacked_predictions = stacked_model.predict(dfPredictions)

        # Calculate metrics
        rmse = round(np.sqrt(mean_squared_error(y_val, stacked_predictions)), 3)
        mae = round(mean_absolute_error(y_val, stacked_predictions), 3)
        r2 = round(r2_score(y_val, stacked_predictions), 3)

        model_metrics[stacked_model_name].append((rmse, mae, r2))

    return model_metrics, trained_base_models


def getUnfitModels():
    models = list()
    models.append(Ridge(alpha=10, fit_intercept=True, max_iter=500, positive=False, solver='sag', tol=0.01))
    models.append(Lasso(alpha=0.01, fit_intercept=True, max_iter=1500, positive=False, selection='random', tol=0.001,
                        warm_start=False))
    models.append(DecisionTreeRegressor(max_depth=None, max_features=None, min_samples_leaf=1, min_samples_split=2))
    models.append(AdaBoostRegressor(learning_rate=0.01, loss='linear', n_estimators=50))
    models.append(RandomForestRegressor(max_depth=4, max_features=1.0, min_samples_leaf=15, min_samples_split=15,
                                        n_estimators=1600))
    models.append(GradientBoostingRegressor(learning_rate=0.01, max_depth=5, min_samples_leaf=2, n_estimators=200)),
    models.append(SVR(C=100, epsilon=1, kernel='poly'))
    return models


def fitStackedModel(X_stacked, y_stacked):
    model = LinearRegression()
    model.fit(X_stacked, y_stacked)
    with open('BinaryFolder/stacked_model', 'wb') as f:
        pickle.dump(model, f)
    return model

print("\n***Starting Stacked Model Creation")
unfit_models = getUnfitModels()
model_metrics, base_models = k_fold_train_validate(unfit_models, X, y)

for model_name, metrics in model_metrics.items():
    avg_rmse = round(float(np.mean([m[0] for m in metrics])), 3)
    std_rmse = round(float(np.std([m[0] for m in metrics])), 3)
    avg_mae = round(float(np.mean([m[1] for m in metrics])), 3)
    std_mae = round(float(np.std([m[1] for m in metrics])), 3)
    avg_r2 = round(float(np.mean([m[2] for m in metrics])), 3)
    std_r2 = round(float(np.std([m[2] for m in metrics])), 3)

    print(f"\n*** {model_name} Metrics (Average over {len(metrics)} folds):")
    print(f"  Average RMSE: {avg_rmse}, Standard deviation: {std_rmse}")
    print(f"  Average MAE: {avg_mae}, Standard deviation: {std_mae}")
    print(f"  Average R-squared: {avg_r2}, Standard deviation: {std_r2}")

# Save base models
for i, (model_name, models) in enumerate(base_models.items()):
    # Get the corresponding RMSE values for each model
    rmse_values = [metrics[0] for metrics in model_metrics[model_name]]

    # Find the index of the model with the lowest RMSE
    best_model_idx = np.argmin(rmse_values)

    # Select the best model according to the lowest RMSE
    best_model = models[best_model_idx]

    # Create a file name for the model
    file_name = f"BinaryFolder/base_model{i + 1}.pkl"

    # Save the trained model to a pickle file
    with open(file_name, 'wb') as f:
        pickle.dump(best_model, f)

# # Grid Search to find optimal properties for RandomForestRegressor/AdaBoostRegressor/DecisionTreeRegressor/LinearRegression/GradientBoost/SVR
#
# # # Define the parameter grid for RandomForestRegressor
# # param_grid = {
# #     'n_estimators': [400, 800, 1600],
# #     'max_depth': [4,6, None],
# #     'min_samples_split': [15],
# #     'min_samples_leaf': [15],
# #     'max_features': [1.0]
# # }
#
# # # Define the parameter grid for LinearRegression
# # param_grid = {
# #     'fit_intercept': [True, False],
# #     'copy_X': [True, False],
# #     'positive': [True, False],
# #     'n_jobs': [-1]
# # }
#
# # # Parameter grid for AdaBoostRegressor
# # param_grid = {
# #     'n_estimators': [50, 100, 200],
# #     'learning_rate': [0.01, 0.1, 1],
# #     'loss': ['linear', 'square', 'exponential']
# # }
#
# # # Parameter grid for DecisionTreeRegressor
# # param_grid = {
# #     'max_depth': [None, 10, 20, 30],
# #     'min_samples_split': [2, 5, 10],
# #     'min_samples_leaf': [1, 2, 5],
# #     'max_features': [None, 1.0, 'sqrt', 'log2']
# # }
#
# # # Parameter grid for Lasso Regression
# # param_grid = {
# #     'alpha': [0.001, 0.01, 0.1, 1, 10],
# #     'fit_intercept': [True, False],
# #     'max_iter': [500, 1000, 1500],
# #     'tol': [1e-4, 1e-3, 1e-2],
# #     'warm_start': [True, False],
# #     'positive': [True, False],
# #     'selection': ['cyclic', 'random']
# # }
#
# # # Parameter grid for Ridge Regression
# # param_grid = {
# #     'alpha': [0.001, 0.01, 0.1, 1, 10],
# #     'fit_intercept': [True, False],
# #     'max_iter': [500, 1000, 1500],
# #     'tol': [1e-4, 1e-3, 1e-2],
# #     'positive': [False],
# #     'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
# # }
#
# # # Parameter grid for GradientBoostingRegressor
# # param_grid = {
# #     'n_estimators': [50, 100, 200],
# #     'learning_rate': [0.01, 0.1, 0.2],
# #     'max_depth': [3, 4, 5],
# #     'min_samples_split': [2, 5, 10],
# #     'min_samples_leaf': [1, 2, 5]
# # }
#
# # Parameter grid for SVR
# param_grid = {
#     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
#     'C': [0.1, 1, 10, 100],
#     'epsilon': [0.01, 0.1, 1]
# }
#
# # # Create a random forest classifier object
# # rf = RandomForestRegressor()
#
# # # Create a LinearRegression object
# # lr = LinearRegression()
#
# # # Create a AdaBoostRegressor object
# # ab = AdaBoostRegressor()
#
# # # Create a DecisionTreeRegressor object
# # dtr = DecisionTreeRegressor()
#
# # # Create a Lasso Regression object
# # ls = Lasso()
#
# # # Create a Ridge Regression object
# # rd = Ridge()
#
# # # Create a Gradient Boosting Regressor object
# # gbr = GradientBoostingRegressor()
#
# # # Create a Support Vector Regression object
# svr = SVR()
#
# # Create a grid search object
# grid_search = GridSearchCV(svr, param_grid=param_grid, cv=5, n_jobs=-1)
#
# # Fit the grid search object to the data
# grid_search.fit(X, y)
#
# # Print the best hyperparameters
# # Optimum hyperparameters found for RandomForestRegressor: {'max_depth': 4, 'max_features': 1.0, 'min_samples_leaf': 15, 'min_samples_split': 15, 'n_estimators': 1600}
# # Optimum hyperparameters found for LinearRegression: {'copy_X': True, 'fit_intercept': True, 'n_jobs': -1, 'positive': False}
# # Optimum hyperparameters found for AdaBoostRegressor: {'learning_rate': 0.01, 'loss': 'linear', 'n_estimators': 50}
# # Optimum hyperparameters found for DecisionTreeRegressor: {'max_depth': None, 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 2}
# # Optimum hyperparameters found for Lasso Regression: {'alpha': 0.01, 'fit_intercept': True, 'max_iter': 1500, 'positive': False, 'selection': 'random', 'tol': 0.001, 'warm_start': False}
# # Optimum hyperparameters found for Ridge Regression: {'alpha': 10, 'fit_intercept': True, 'max_iter': 500, 'positive': False, 'solver': 'sag', 'tol': 0.01}
# # Optimum hyperparameters found for Gradient Boosting Regression: {'learning_rate': 0.01, 'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
# # Optimum hyperparameters found for Support Vector Regression: {'C': 100, 'epsilon': 1, 'kernel': 'poly'}
# print(grid_search.best_params_)