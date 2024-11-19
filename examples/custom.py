import matplotlib.pyplot as plt
import numpy as np
from sksurv.datasets import load_whas500, load_veterans_lung_cancer
from sksurv.custom import SurvivalCustom
from sksurv.tree import SurvivalTree
from sklearn.linear_model import Ridge, MultiTaskElasticNet, RidgeCV, ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import brier_score, integrated_brier_score
from time import time

X, y = load_whas500()
X = X.astype(float)

estimator = SurvivalCustom(regr=RandomForestRegressor())
estimator2 = SurvivalTree()
estimator3 = CoxPHSurvivalAnalysis(ties="efron")
estimator4 = SurvivalCustom(regr=RidgeCV())

start = time()
estimator.fit(X.iloc[:100,:], y[:100])
print("Time to fit RandomForestRegressor: ", time() - start)
start = time()
estimator2.fit(X.iloc[:100,:], y[:100])
print("Time to fit SurvivalTree: ", time() - start)
start = time()
estimator3.fit(X.iloc[:100,:], y[:100])
print("Time to fit CoxPHSurvivalAnalysis: ", time() - start)
start = time()
estimator4.fit(X.iloc[:100,:], y[:100])
print("Time to fit RidgeCV: ", time() - start)


X_people = X.iloc[101:103]
surv_funcs = estimator.predict_survival_function(X_people)
preds = [fn(200) for fn in surv_funcs]
surv_funcs2 = estimator2.predict_survival_function(X_people)
preds2 = [fn(200) for fn in surv_funcs2]
surv_funcs3 = estimator3.predict_survival_function(X_people)
preds3 = [fn(200) for fn in surv_funcs3]
surv_funcs4 = estimator4.predict_survival_function(X_people)
preds4 = [fn(200) for fn in surv_funcs4]

times, score = brier_score(y[:100], y[101:103], preds, 200)
print(score)
times, score = brier_score(y[:100],  y[101:103], preds2, 200)
print(score)
times, score = brier_score(y[:100],  y[101:103], preds3, 200)
print(score)
times, score = brier_score(y[:100],  y[101:103], preds4, 200)
print(score)

times_ = np.linspace(200, 300)
preds = np.asarray([[fn(t) for t in times_] for fn in surv_funcs])
score = integrated_brier_score(y[:100], y[101:103], preds, np.linspace(200, 300))
print(score)
preds2 = np.asarray([[fn(t) for t in times_] for fn in surv_funcs2])
score = integrated_brier_score(y[:100], y[101:103], preds2, np.linspace(200, 300))
print(score)
preds3 = np.asarray([[fn(t) for t in times_] for fn in surv_funcs3])
score = integrated_brier_score(y[:100], y[101:103], preds3, np.linspace(200, 300))
print(score)
preds4 = np.asarray([[fn(t) for t in times_] for fn in surv_funcs4])
score = integrated_brier_score(y[:100], y[101:103], preds4, np.linspace(200, 300))
print(score)

for fn in surv_funcs:
    plt.step(fn.x, fn(fn.x), where="post")
    plt.ylim(0, 1)
    plt.show()
