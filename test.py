import pandas as pd
import functools
import operator

l = []
a = pd.Series([True,False])
b= pd.Series([True,True])

l.append(a)
l.append(b)

print(functools.reduce(operator.__or__,l))

from catboost import CatBoostClassifier