
import pandas as pd
import statsmodels.api as sm
import pickle
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model

df = pd.read_csv('test.csv')

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Function to convert string columns to numerics
def convert_string_columns_to_numeric(df):
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])
    return df

# Convert string columns to numerics
df = convert_string_columns_to_numeric(df)

# Statistically significant features found
features = ['writing score', 'reading score', 'test preparation course']

# Define X
X = df[features]
y = df['math score']




#### Model 1: OLS Predictions

# Load the OLS model
with open("BinaryFolder/m1_ols_model", "rb") as file:
    ols_model = pickle.load(file)

# Load the scalar
with open("BinaryFolder/sc_x.pkl", "rb") as file:
    scalar = pickle.load(file)

# Scale the input data using the loaded scalar
X_scaled = scalar.transform(X)

X_ols = sm.add_constant(X_scaled)

# Make predictions using the loaded OLS model
predictions = ols_model.predict(X_ols)

# Print the predictions
print("*** Model 1: OLS Model - Predictions:\n\n", predictions)



### Model 2: NN Predictions

# Load the neural network model
nn_model = load_model("BinaryFolder/m2_nn_model.h5")

# Make predictions using the loaded neural network model
predictions = nn_model.predict(X)

# Print the predictions
print("\n*** Model 2: NN Model - Predictions:\n\n", predictions)



### Model 3: Stacked Model Predictions

# Load base models from their pickle files
base_models = []
for i in range(1, 8):
    file_name = f"BinaryFolder/base_model{i}.pkl"
    with open(file_name, "rb") as f:
        model = pickle.load(f)
        base_models.append(model)

# Make predictions with each base model and store them in a new DataFrame
base_predictions = pd.DataFrame()
for i, model in enumerate(base_models):
    predictions = model.predict(X)
    base_predictions[str(i)] = predictions

# Load the stacked model from its pickle file
with open("BinaryFolder/stacked_model", "rb") as f:
    stacked_model = pickle.load(f)

# Make predictions with the stacked model using the base models' predictions
stacked_predictions = stacked_model.predict(base_predictions)

# Print the predictions
print("\n*** Model 3: Linear Regression Stacked Model - Predictions:\n\n", stacked_predictions)
