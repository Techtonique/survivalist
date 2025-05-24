import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
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
        clf=RandomForestClassifier(random_state=42),
        type_sim="kde",
        loss = "ipcwls"
    )
    survstacker_et = SurvStacker(
        clf=ExtraTreesClassifier(random_state=42),
        type_sim="kde",
        loss = "ipcwls"
    )

    survstacker_lr = SurvStacker(
        clf=LogisticRegression(),
        type_sim="kde",
        loss = "ipcwls"
    )
    
    # Fit models
    survstacker_rf.fit(X_train, y_train)
    survstacker_et.fit(X_train, y_train)
    survstacker_lr.fit(X_train, y_train)
    
    # Get survival function predictions with confidence intervals
    surv_funcs_rf = survstacker_rf.predict_survival_function(X_test[:3], level=95)
    surv_funcs_et = survstacker_et.predict_survival_function(X_test[:3], level=95)
    surv_funcs_lr = survstacker_lr.predict_survival_function(X_test[:3], level=95)

    print(f"Survival functions for {dataset_name} dataset:")    
    
    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['blue', 'red', 'green']
    
    # Plot RF predictions
    ax1.set_title(f'Random Forest - {dataset_name}')
    for i, (mean, lower, upper) in enumerate(zip(
        surv_funcs_rf.mean, surv_funcs_rf.lower, surv_funcs_rf.upper)):
        times = mean.x
        ax1.step(times, mean.y, where="post", label=f'Patient {i+1}', color=colors[i])
        ax1.fill_between(times, lower.y, upper.y, alpha=0.2, color=colors[i], step="post")
    ax1.grid(True)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Survival Probability')
    ax1.set_ylim(0, 1)  # Set y-axis limits
    ax1.legend()
    
    # Plot ET predictions
    ax2.set_title(f'Extra Trees - {dataset_name}')
    for i, (mean, lower, upper) in enumerate(zip(
        surv_funcs_et.mean, surv_funcs_et.lower, surv_funcs_et.upper)):
        times = mean.x
        ax2.step(times, mean.y, where="post", label=f'Patient {i+1}', color=colors[i])
        ax2.fill_between(times, lower.y, upper.y, alpha=0.2, color=colors[i], step="post")
    ax2.grid(True)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Survival Probability')
    ax2.set_ylim(0, 1)  # Set y-axis limits
    ax2.legend()

    # Plot LR predictions
    ax3.set_title(f'Logistic Regression - {dataset_name}')
    for i, (mean, lower, upper) in enumerate(zip(
        surv_funcs_lr.mean, surv_funcs_lr.lower, surv_funcs_lr.upper)):
        times = mean.x
        ax3.step(times, mean.y, where="post", label=f'Patient {i+1}', color=colors[i])
        ax3.fill_between(times, lower.y, upper.y, alpha=0.2, color=colors[i], step="post")
    ax3.grid(True)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Survival Probability')
    ax3.set_ylim(0, 1)  # Set y-axis limits
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(f'survival_curves_{dataset_name}.png')
    plt.close()

# Import necessary libraries

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

