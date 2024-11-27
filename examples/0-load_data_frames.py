from sksurv.datasets import load_whas500, load_gbsg2, load_veterans_lung_cancer, \
    load_aids, load_breast_cancer, load_flchain

X, y = load_whas500()

#print(X.head())
#print(y[5:])

df = load_whas500(as_frame=True)
print(df.frame.head())

X, y = load_gbsg2()

#print(X.head())
#print(y[5:])

df = load_gbsg2(as_frame=True)
print(df.frame.head())

X, y = load_veterans_lung_cancer()

#print(X.head())
#print(y[5:])

df = load_veterans_lung_cancer(as_frame=True)
print(df.frame.head())

X, y = load_aids()

#print(X.head())
#print(y[5:])

df = load_aids(as_frame=True)
print(df.frame.head())

X, y = load_breast_cancer()

#print(X.head())
#print(y[5:])

df = load_breast_cancer(as_frame=True)
print(df.frame.head())

X, y = load_flchain()

#print(X.head())
#print(y[5:])

df = load_flchain(as_frame=True)
print(df.frame.head())