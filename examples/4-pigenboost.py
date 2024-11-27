import matplotlib.pyplot as plt
import nnetsauce as ns
from survivalist.datasets import load_whas500
from survivalist.ensemble import PIComponentwiseGenGradientBoostingSurvivalAnalysis
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

print("estimator.predict(X_test)", estimator.predict(X_test))


estimator = PIComponentwiseGenGradientBoostingSurvivalAnalysis(regr = RidgeCV(), 
                                                               loss="coxph", 
                                                               type_pi="kde")

estimator.fit(X_train, y_train)

surv_funcs = estimator.predict_survival_function(X.iloc[:1])

print(surv_funcs)
print("surv_funcs.mean", surv_funcs.mean)
print("surv_funcs.lower", surv_funcs.lower)
print("surv_funcs.upper", surv_funcs.upper)

estimator = PIComponentwiseGenGradientBoostingSurvivalAnalysis(regr = RidgeCV(), 
                                                               loss="coxph", 
                                                               type_pi="bootstrap")

estimator.fit(X_train, y_train)

surv_funcs = estimator.predict_survival_function(X.iloc[:1])

print(surv_funcs)
print("surv_funcs.mean", surv_funcs.mean)
print("surv_funcs.lower", surv_funcs.lower)
print("surv_funcs.upper", surv_funcs.upper)

estimator = PIComponentwiseGenGradientBoostingSurvivalAnalysis(regr = RidgeCV(), 
                                                               loss="coxph", 
                                                               type_pi="ecdf")

estimator.fit(X_train, y_train)

surv_funcs = estimator.predict_survival_function(X.iloc[:1])

print(surv_funcs)
print("surv_funcs.mean", surv_funcs.mean)
print("surv_funcs.lower", surv_funcs.lower)
print("surv_funcs.upper", surv_funcs.upper)


