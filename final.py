# CS 301 Final Project

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy import stats
from yellowbrick.classifier import PrecisionRecallCurve


# Write to file.
def output(*argv):
    s = ""
    for arg in argv:
        s += str(arg) + " "
    f = open('results.txt', 'a')
    f.write(s)
    f.write('\n')
    f.close()
    print(s)


# Wipe file
f = open('results.txt', 'w')
f.close()

# Load Data
data = pd.read_csv("fetal_health-1.csv")
X = data.drop('fetal_health', axis=1)
y = data['fetal_health']

# Task 1
class_count = {}

for i in y:
    if i in class_count:
        class_count[i] += 1
    else:
        class_count[i] = 1

print(class_count)

bar_x = ['1', '2', '3']
bar_y = [class_count[1], class_count[2], class_count[3]]

plt.bar(bar_x, bar_y, color='red')
plt.xlabel("Class")
plt.ylabel("Number of Instances")
plt.title("Class Imbalance")
#plt.savefig('Class Imbalance')
plt.show()

# Task 2 Correlation Analysis
attr_scores = {}
for attr in X.columns:
        # pearson[0] is pearson coeffecient, pearson[1] is p value
        pearson = stats.pearsonr(data[attr], y)
        coeff = pearson[0]
        p_val = pearson[1]
        output("Attribute: ", attr, "Pearson coefficient:", coeff, 'p-value:', p_val)

        attr_scores[attr] = abs(coeff)

        #if (abs(coeff) > p_val):
                #print("Significant")

        for crit_val in [0.90, 0.95]:
                if (p_val <= 1 - crit_val):
                        output("Significant at", crit_val)
output("")

# Print Top 10 attributes from most to least correlated
sorted_attr = {k: v for k, v in sorted(attr_scores.items(), key=lambda item: item[1], reverse=True)}
output("Top 10 most correlated attributes by correlation coefficient")
for index, s in enumerate(sorted_attr):
    output(s, attr_scores[s])
    if index > 9:
        break
output("")

################################################################
# # Task 1.5 Handle data imbalance problem - over or undersampling. May need to oversample
# add2 = []
# add3 = []
# for i in range(len(data)):
#     if data.iloc[i]["fetal_health"] == 2:
#         add2.append(data.iloc[i])
#     elif data.iloc[i]["fetal_health"] == 3:
#         add3.append(data.iloc[i])
#
# diff2 = class_count[1] - class_count[2]
# diff3 = class_count[1] - class_count[3]
#
# for i in range(diff2):
#     data = data.append(add2[i % len(add2)])
#
# for i in range(diff3):
#     data = data.append(add3[i % len(add3)])
#
# X = data.drop('fetal_health', axis=1)
# y = data['fetal_health']
#########################################################

# Task 2.5 - drop non-correlated attributes
X = X.drop('histogram_number_of_peaks', axis=1)
X = X.drop('histogram_number_of_zeroes', axis=1)

# split into train-test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=27)

# Oversample test data
#################################################
add2 = []
add3 = []
y2 = []
y3 = []

for i in range(len(X_train)):
    if y_train.iloc[i] == 2:
        add2.append(X_train.iloc[i])
        y2.append(y_train.iloc[i])
    elif y_train.iloc[i] == 3:
        add3.append(X_train.iloc[i])
        y3.append(y_train.iloc[i])

class_count = {}

for i in y_train:
    if i in class_count:
        class_count[i] += 1
    else:
        class_count[i] = 1

diff2 = class_count[1] - class_count[2]
diff3 = class_count[1] - class_count[3]

for i in range(diff2):
    X_train = X_train.append(add2[i % len(add2)])
    y_train = y_train.append(pd.Series([2]))

for i in range(diff3):
    X_train = X_train.append(add3[i % len(add3)])
    y_train = y_train.append(pd.Series([3]))
#############################################################

# Task 3 Create two different models using the most appropriate features found in Task 2
models = {GaussianNB(): 'Bayesian',
          RandomForestClassifier(): 'RandomForest',
          DecisionTreeClassifier(criterion="entropy", random_state=100,
          max_depth=3, min_samples_leaf=5): 'Decision Tree',
          KNeighborsClassifier(n_neighbors=4): 'KNeighbor'}

for model in models:
    model.fit(X_train, y_train)

# Evaluate models
for model in models:
    output('-'*5, models[model], '-'*5)
    y_pred = model.predict(X_test)

    # Task 5 Score models
    output("F1 Score: ", f1_score(y_test, y_pred, average='macro'))
    output("Precision: ", precision_score(y_test, y_pred, average='macro'))
    output("Recall: ", recall_score(y_test, y_pred, average='macro'))

    pred_prob = model.predict_proba(X_test)
    auc_score = roc_auc_score(y_test, pred_prob, multi_class='ovo')
    output('Area under ROC curve:', auc_score)

    # Task 4 Visually present the confusion matrix
    plot_confusion_matrix(model, X_test, y_test)
    plt.title("Confusion Matrix - " + models[model])
    #plt.savefig(models[model] + '-confusion-matrix')
    plt.show()

    viz = PrecisionRecallCurve(
        RandomForestClassifier(n_estimators=10),
        per_class=True,
        cmap="Set1"
    )
    viz.fit(X_train, y_train)
    viz.score(X_test, y_pred)
    viz.show()

# Task 6 K Means clustering
m = 7
n = 12
for k in [5, 10, 15]:
    cluster_data = X.iloc[:, [m,n]]
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, random_state=0)
    kmeans.fit(cluster_data)
    pred_y = kmeans.fit_predict(cluster_data)

    # plot original data
    plt.scatter(cluster_data.iloc[:,0], cluster_data.iloc[:,1], c=pred_y, cmap=plt.cm.Paired)
    # plot clusters
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')

    plt.title(str(k) + " clusters")
    plt.xlabel(X.columns[m])
    plt.ylabel(X.columns[n])
    plt.legend()
    plt.savefig(str(k) + 'kmeans')
    plt.show()