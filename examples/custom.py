import matplotlib.pyplot as plt
from sksurv.datasets import load_whas500
from sksurv.custom import SurvivalCustom
from sksurv.tree import SurvivalTree
from sklearn.linear_model import Ridge, MultiTaskElasticNet

X, y = load_whas500()
X = X.astype(float)

estimator = SurvivalCustom()
estimator2 = SurvivalTree()

estimator.fit(X, y)
estimator2.fit(X, y)

surv_funcs2 = estimator2.predict_survival_function(X.iloc[:2])
print(surv_funcs2[1])
surv_funcs = estimator.predict_survival_function(X.iloc[:2])
print(surv_funcs[1])


for fn in surv_funcs:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.ylim(0, 1)
    plt.show()
