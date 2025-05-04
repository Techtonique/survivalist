import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegressionCV
from survivalist.datasets import load_whas500, load_veterans_lung_cancer, load_gbsg2
from survivalist.survstack import SurvStacker
from survivalist.nonparametric import kaplan_meier_estimator

import pandas as pd

def _encode_categorical_columns(df, categorical_columns=None):
    """
    Automatically identifies categorical columns and applies one-hot encoding.

    Parameters:
    - df (pd.DataFrame): The input DataFrame with mixed continuous and categorical variables.
    - categorical_columns (list): Optional list of column names to treat as categorical.

    Returns:
    - pd.DataFrame: A new DataFrame with one-hot encoded categorical columns.
    """
    # Automatically identify categorical columns if not provided
    if categorical_columns is None:
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Apply one-hot encoding to the identified categorical columns
    df_encoded = pd.get_dummies(df, columns=categorical_columns)

    # Convert boolean columns to integer (0 and 1)
    bool_columns = df_encoded.select_dtypes(include=['bool']).columns.tolist()
    df_encoded[bool_columns] = df_encoded[bool_columns].astype(int)

    return df_encoded

# Load and prepare the WHAS500 dataset
X, y = load_whas500()
print("X.head()", X.head())
X = X.astype(float).to_numpy()  # Convert to numpy array

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

event_time = [y[1] for y in y_test]
event_status = [y[0] for y in y_test]
km = kaplan_meier_estimator(event_status, event_time,
                            conf_type="log-log")

print("X_train.shape", X_train.shape)
print("X_test.shape", X_test.shape)
print("y_train.shape", y_train.shape)
print("y_test.shape", y_test.shape)

# Define time points for evaluation
times = np.linspace(0, 2000, 100)

# Initialize models with specified time points
survstacker_rf = SurvStacker(
    clf=RandomForestClassifier(n_estimators=100, random_state=42))
survstacker_et = SurvStacker(
    clf=ExtraTreesClassifier(random_state=42))
#coxph = CoxPHSurvivalAnalysis()

# Fit models
survstacker_rf.fit(X_train, y_train)
survstacker_et.fit(X_train, y_train)
#coxph.fit(X_train, y_train)

# Get predictions for first two test samples
X_samples = X_test[:2]

# Get survival functions
surv_rf = survstacker_rf.predict_survival_function(X_samples)
surv_et = survstacker_et.predict_survival_function(X_samples)

# Print diagnostics
print("Time points shape:", times.shape)
print("Survival function shape:", surv_rf.shape)
print("First few survival probabilities:", surv_rf[:, :20])
print("Monotonic decreasing check:", np.all(np.diff(surv_rf[0]) <= 0))
print("Monotonic decreasing check:", np.all(np.diff(surv_et[0]) <= 0))

# Plot survival functions
#plt.figure(figsize=(10, 6))
plt.plot(surv_rf[0, :], label='Random Forest')
plt.plot(surv_et[0, :], label='Logistic Regression')
plt.title('Survival Function Estimates')
plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.legend()
plt.grid()
plt.show()


X, y = load_veterans_lung_cancer()
print("\n X.head()", X.head())
X = _encode_categorical_columns(X) # Convert categorical columns to one-hot encoding
X = X.astype(float).to_numpy()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

event_time = [y[1] for y in y_test]
event_status = [y[0] for y in y_test]
km = kaplan_meier_estimator(event_status, event_time,
                            conf_type="log-log")

print("X_train.shape", X_train.shape)
print("X_test.shape", X_test.shape)
print("y_train.shape", y_train.shape)
print("y_test.shape", y_test.shape)

# Define time points for evaluation
times = np.linspace(0, 2000, 100)

# Initialize models with specified time points
survstacker_rf = SurvStacker(
    clf=RandomForestClassifier(n_estimators=100, random_state=42))
survstacker_et = SurvStacker(
    clf=ExtraTreesClassifier(random_state=42))
#coxph = CoxPHSurvivalAnalysis()

# Fit models
survstacker_rf.fit(X_train, y_train)
survstacker_et.fit(X_train, y_train)
#coxph.fit(X_train, y_train)

# Get predictions for first two test samples
X_samples = X_test[:2]

# Get survival functions
surv_rf = survstacker_rf.predict_survival_function(X_samples)
surv_et = survstacker_et.predict_survival_function(X_samples)

# Print diagnostics
print("Time points shape:", times.shape)
print("Survival function shape:", surv_rf.shape)
print("First few survival probabilities:", surv_rf[:, :20])
print("Monotonic decreasing check:", np.all(np.diff(surv_rf[0]) <= 0))
print("Monotonic decreasing check:", np.all(np.diff(surv_et[0]) <= 0))

# Plot survival functions
#plt.figure(figsize=(10, 6))
plt.plot(surv_rf[0, :], label='Random Forest')
plt.plot(surv_et[0, :], label='Logistic Regression')
plt.title('Survival Function Estimates')
plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.legend()
plt.grid()
plt.show()


X, y = load_gbsg2()
print("\n X.head()", X.head())
X = _encode_categorical_columns(X) # Convert categorical columns to one-hot encoding
X = X.astype(float).to_numpy()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

event_time = [y[1] for y in y_test]
event_status = [y[0] for y in y_test]
km = kaplan_meier_estimator(event_status, event_time,
                            conf_type="log-log")

print("X_train.shape", X_train.shape)
print("X_test.shape", X_test.shape)
print("y_train.shape", y_train.shape)
print("y_test.shape", y_test.shape)

# Define time points for evaluation
times = np.linspace(0, 2000, 100)

# Initialize models with specified time points
survstacker_rf = SurvStacker(
    clf=RandomForestClassifier(n_estimators=100, random_state=42))
survstacker_et = SurvStacker(
    clf=ExtraTreesClassifier(random_state=42))
#coxph = CoxPHSurvivalAnalysis()

# Fit models
survstacker_rf.fit(X_train, y_train)
survstacker_et.fit(X_train, y_train)
#coxph.fit(X_train, y_train)

# Get predictions for first two test samples
X_samples = X_test[:2]

# Get survival functions
surv_rf = survstacker_rf.predict_survival_function(X_samples)
surv_et = survstacker_et.predict_survival_function(X_samples)

# Print diagnostics
print("Time points shape:", times.shape)
print("Survival function shape:", surv_rf.shape)
print("First few survival probabilities:", surv_rf[:, :20])
print("Monotonic decreasing check:", np.all(np.diff(surv_rf[0]) <= 0))
print("Monotonic decreasing check:", np.all(np.diff(surv_et[0]) <= 0))

# Plot survival functions
#plt.figure(figsize=(10, 6))
plt.plot(surv_rf[0, :], label='Random Forest')
plt.plot(surv_et[0, :], label='Logistic Regression')
plt.title('Survival Function Estimates')
plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.legend()
plt.grid()
plt.show()

