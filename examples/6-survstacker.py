import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegressionCV
from survivalist.datasets import load_whas500
from survivalist.survstack import SurvStacker

# Load and prepare the WHAS500 dataset
X, y = load_whas500()
X = X.astype(float).to_numpy()  # Convert to numpy array

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

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
