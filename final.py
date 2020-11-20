# CS 301 Final Project

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LinearRegression

# Task 1 (visualization should be in separate file?)
data = pd.read_csv("fetal_health-1.csv")
X = data.drop('fetal_health', axis=1)
y = data['fetal_health']

class_count = {}

for i in y:
    if i in class_count:
        class_count[i] += 1
    else:
        class_count[i] = 1

print(class_count)

# Task 2


# Task 3


# Task 4


# Task 5


# Task 6