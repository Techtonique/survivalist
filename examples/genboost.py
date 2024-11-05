import matplotlib.pyplot as plt
from sksurv.datasets import load_whas500
from sksurv.ensemble import ComponentwiseGenGradientBoostingSurvivalAnalysis
from sklearn.linear_model import RidgeCV
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor

X, y = load_whas500()
X = X.astype(float)

estimator = ComponentwiseGenGradientBoostingSurvivalAnalysis(regr = RidgeCV(), loss="coxph")
estimator2 = ComponentwiseGenGradientBoostingSurvivalAnalysis(regr = ExtraTreeRegressor(), loss="coxph")
estimator3 = ComponentwiseGenGradientBoostingSurvivalAnalysis(regr = ExtraTreesRegressor(), 
                                                              n_estimators=1, loss="coxph")

estimator.fit(X, y)
estimator2.fit(X, y)
estimator3.fit(X, y)

print(estimator)
print(estimator2)
print(estimator3)

surv_funcs = estimator.predict_survival_function(X.iloc[:2])
surv_funcs2 = estimator2.predict_survival_function(X.iloc[:2])
surv_funcs3 = estimator3.predict_survival_function(X.iloc[:2])

print(surv_funcs[1])
print(surv_funcs2[1])
print(surv_funcs3[1])

for fn in surv_funcs:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.ylim(0, 1)
    plt.show()

for fn in surv_funcs2:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.ylim(0, 1)
    plt.show()

for fn in surv_funcs3:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.ylim(0, 1)
    plt.show()
