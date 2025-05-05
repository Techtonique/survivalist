import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegressionCV
from survivalist.datasets import load_whas500, load_veterans_lung_cancer, load_gbsg2
from survivalist.survstack import SurvStacker

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

def analyze_survival_dataset(X, y, dataset_name):
    """Analyze a survival dataset using Random Forest and Extra Trees models"""
    # Data preparation
    X = X.astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize models
    survstacker_rf = SurvStacker(
        clf=RandomForestClassifier(n_estimators=100, random_state=42),
        type_sim="kde",
        loss = "ipcwls"
    )
    survstacker_et = SurvStacker(
        clf=ExtraTreesClassifier(random_state=42),
        type_sim="kde",
        loss = "ipcwls"
    )
    
    # Fit models
    survstacker_rf.fit(X_train, y_train)
    survstacker_et.fit(X_train, y_train)
    
    # Get survival function predictions
    surv_funcs_rf = survstacker_rf.predict_survival_function(X_test[:3])
    surv_funcs_et = survstacker_et.predict_survival_function(X_test[:3])

    print(f"Survival functions for {dataset_name} dataset:")    
    print("surv_funcs_rf", surv_funcs_rf)
    print("len(surv_funcs_rf)", len(surv_funcs_rf))
    print("surv_funcs_et", surv_funcs_et)
    print("len(surv_funcs_et)", len(surv_funcs_et))

# Analyze WHAS500 dataset
print("Analyzing WHAS500 dataset...")
X, y = load_whas500()
analyze_survival_dataset(X, y, "WHAS500")

# Analyze Veterans Lung Cancer dataset
print("\nAnalyzing Veterans Lung Cancer dataset...")
X, y = load_veterans_lung_cancer()
X = _encode_categorical_columns(X)
analyze_survival_dataset(X, y, "Veterans")

# # Analyze GBSG2 dataset
# print("\nAnalyzing GBSG2 dataset...")
# X, y = load_gbsg2()
# X = _encode_categorical_columns(X)
# analyze_survival_dataset(X, y, "GBSG2")

