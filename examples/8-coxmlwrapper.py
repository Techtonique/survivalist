
from survivalist.coxml import CoxMLWrapper
from sklearn.linear_model import RidgeCV

obj = CoxMLWrapper(base_model=RidgeCV())

results_df = obj.run_comparison_study(n_samples=250, test_size=0.2, random_state=42)

print(results_df)