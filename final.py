# CS 301 Final Project

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
from scipy import stats

# Write to file.
def output(*argv):
    s = ""
    for arg in argv:
        s += str(arg) + " "
    s += '\n'
    f = open('results.txt', 'a')
    f.write(s)
    f.write('\n')
    f.close()
    print(s)

# Task 1 (visualization should be in separate file - bar graph?)
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

# Task 1.5 Handle data imbalance problem - over or undersampling. There's probably an API function made for this.


# Task 2 involves pearson coefficient. Find attributes correlated to outcome



# Task 3 Create two different models using the most appropriate features found in Task 2


# Task 4 Visually present the confusion matrix


# Task 5 see homework 3


# Task 6

for k in [5, 10, 15]:
    pass
