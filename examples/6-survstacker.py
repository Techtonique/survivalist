import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
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
        clf=RandomForestClassifier(n_estimators=100, random_state=42)
    )
    survstacker_et = SurvStacker(
        clf=ExtraTreesClassifier(random_state=42)
    )
    
    # Fit models
    survstacker_rf.fit(X_train, y_train)
    survstacker_et.fit(X_train, y_train)
    
    # Get survival function predictions
    surv_funcs_rf = survstacker_rf.predict_survival_function(X_test[:2], return_array=False)
    surv_funcs_et = survstacker_et.predict_survival_function(X_test[:2], return_array=False)
    
    # Print performance scores
    print(f"\n{dataset_name} Dataset Results:")
    print(f"Random Forest C-index: {survstacker_rf.score(X_test, y_test):.3f}")
    print(f"Extra Trees C-index: {survstacker_et.score(X_test, y_test):.3f}")
    
    # Plot survival functions
    plt.figure(figsize=(10, 5))
    
    # Plot RF predictions
    plt.subplot(1, 2, 1)
    for i, fn in enumerate(surv_funcs_rf):
        plt.step(fn.x, fn(fn.x), where="post", label=f"Patient {i+1}")
    plt.title(f"{dataset_name}: Random Forest")
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    
    # Plot ET predictions
    plt.subplot(1, 2, 2)
    for i, fn in enumerate(surv_funcs_et):
        plt.step(fn.x, fn(fn.x), where="post", label=f"Patient {i+1}")
    plt.title(f"{dataset_name}: Extra Trees")
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Analyze WHAS500 dataset
print("Analyzing WHAS500 dataset...")
X, y = load_whas500()
analyze_survival_dataset(X, y, "WHAS500")

# Analyze Veterans Lung Cancer dataset
print("\nAnalyzing Veterans Lung Cancer dataset...")
X, y = load_veterans_lung_cancer()
X = _encode_categorical_columns(X)
analyze_survival_dataset(X, y, "Veterans")

# Analyze GBSG2 dataset
print("\nAnalyzing GBSG2 dataset...")
X, y = load_gbsg2()
X = _encode_categorical_columns(X)
analyze_survival_dataset(X, y, "GBSG2")

