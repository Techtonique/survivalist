import matplotlib.pyplot as plt
import nnetsauce as ns
from sksurv.datasets import load_whas500
from sksurv.ensemble import ComponentwiseGenGradientBoostingSurvivalAnalysis
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor

X, y = load_whas500()
X = X.astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

estimator = ComponentwiseGenGradientBoostingSurvivalAnalysis(regr = RidgeCV(), 
                                                             loss="coxph")
estimator2 = ComponentwiseGenGradientBoostingSurvivalAnalysis(regr = ExtraTreeRegressor(), 
                                                              loss="coxph")

estimator.fit(X_train, y_train)
estimator2.fit(X_train, y_train)

print(estimator.score(X_test, y_test))
print(estimator2.score(X_test, y_test))

surv_funcs = estimator.predict_survival_function(X_test.iloc[:2])
surv_funcs2 = estimator2.predict_survival_function(X_test.iloc[:2])

print(surv_funcs[1])
print(surv_funcs2[1])

for fn in surv_funcs:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.ylim(0, 1)
    plt.show()

for fn in surv_funcs2:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.ylim(0, 1)
    plt.show()
