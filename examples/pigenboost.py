import matplotlib.pyplot as plt
import nnetsauce as ns
from sksurv.datasets import load_whas500
from sksurv.ensemble import PIComponentwiseGenGradientBoostingSurvivalAnalysis
from sklearn.linear_model import RidgeCV
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split

X, y = load_whas500()
X = X.astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

estimator = PIComponentwiseGenGradientBoostingSurvivalAnalysis(regr = RidgeCV(), 
                                                               loss="coxph")

estimator.fit(X_train, y_train)

surv_funcs = estimator.predict_survival_function(X.iloc[:1])

print(surv_funcs)

print("surv_funcs.mean", surv_funcs.mean)

print("surv_funcs.lower", surv_funcs.lower)

print("surv_funcs.upper", surv_funcs.upper)

# for fn in surv_funcs:
#     plt.step(fn.x, fn(fn.x), where="post")
#     plt.ylim(0, 1)
#     plt.show()

# for fn in surv_funcs2:
#     plt.step(fn.x, fn(fn.x), where="post")
#     plt.ylim(0, 1)
#     plt.show()
