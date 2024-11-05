import matplotlib.pyplot as plt
from sksurv.datasets import load_whas500
from sksurv.custom import SurvivalCustom
from sksurv.tree import SurvivalTree
from sklearn.linear_model import Ridge, MultiTaskElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from time import time

X, y = load_whas500()
X = X.astype(float)

estimator = SurvivalCustom(regr=RandomForestRegressor())
estimator2 = SurvivalTree()
estimator3 = SurvivalCustom(regr=MLPRegressor())

start = time()
estimator.fit(X, y)
print("Time to fit RandomForestRegressor: ", time() - start)
start = time()
estimator2.fit(X, y)
print("Time to fit SurvivalTree: ", time() - start)
start = time()
estimator3.fit(X, y)
print("Time to fit MLPRegressor: ", time() - start)

X_people = X.iloc[:1]
surv_funcs = estimator.predict_survival_function(X_people)
surv_funcs2 = estimator2.predict_survival_function(X_people)
surv_funcs3 = estimator3.predict_survival_function(X_people)

print(surv_funcs)
print(surv_funcs2)
print(surv_funcs3)

for fn in surv_funcs:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.ylim(0, 1)
    plt.show()
