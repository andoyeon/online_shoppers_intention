from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest


# Importing data set
online_shoppers_intention = pd.read_csv('online_shoppers_intention.csv')
dataset = online_shoppers_intention
print('shape:', dataset.shape)
m = dataset.shape[0]
print(dataset.info())
print(dataset.head())
print(dataset.describe())
features = list(dataset.columns)
print('features:', features)

# Revenue graph
Revenue = dataset.loc[:, 'Revenue']


# Dividing X and y
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
print('X =', X[:5])
print('y =', y[:5])

# Checking missing data
dataset.isnull().sum()

# Handling catagorical data
month_name = set(dataset.loc[:, 'Month'])
# print('Month:', month_name)
visitortype = set(dataset.loc[:, 'VisitorType'])
# print('VisitorType:', visitortype)

# print(list(dataset['Month']))
print(Counter(dataset['Browser']))
print(Counter(dataset['OperatingSystems']))
print(Counter(dataset['Region']))

month = []
visitorType = []
weekend = []
browser = []
operatingSystems = []
region = []
for i in features:
    if i == 'Month':
        data = list(dataset[i])
        for j in range(m):
            if data[j] == 'Jan':
                month.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            elif data[j] == 'Feb':
                month.append([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            elif data[j] == 'Mar':
                month.append([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            elif data[j] == 'Apr':
                month.append([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
            elif data[j] == 'May':
                month.append([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
            elif data[j] == 'June':
                month.append([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
            elif data[j] == 'Jul':
                month.append([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
            elif data[j] == 'Aug':
                month.append([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
            elif data[j] == 'Sep':
                month.append([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
            elif data[j] == 'Oct':
                month.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
            elif data[j] == 'Nov':
                month.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
            elif data[j] == 'Dec':
                month.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    elif i == 'VisitorType':
        data = list(dataset[i])
        for j in range(m):
            if data[j] == 'New_Visitor':
                visitorType.append([1, 0, 0])
            elif data[j] == 'Returning_Visitor':
                visitorType.append([0, 1, 0])
            elif data[j] == 'Other':
                visitorType.append([0, 0, 1])

    elif i == 'Weekend':
        data = list(dataset[i])
        for j in range(m):
            if data[j]:
                weekend.append([1, 0])
            elif data[j] == False:
                weekend.append([0, 1])

    elif i == 'Browser':
        data = list(dataset[i])
        for j in range(m):
            if data[j] == 1:
                browser.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            elif data[j] == 2:
                browser.append([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            elif data[j] == 3:
                browser.append([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            elif data[j] == 4:
                browser.append([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            elif data[j] == 5:
                browser.append([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
            elif data[j] == 6:
                browser.append([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
            elif data[j] == 7:
                browser.append([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
            elif data[j] == 8:
                browser.append([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
            elif data[j] == 9:
                browser.append([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
            elif data[j] == 10:
                browser.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
            elif data[j] == 11:
                browser.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
            elif data[j] == 12:
                browser.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
            elif data[j] == 13:
                browser.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    elif i == 'OperatingSystems':
        data = list(dataset[i])
        for j in range(m):
            if data[j] == 1:
                operatingSystems.append([1, 0, 0, 0, 0, 0, 0, 0])
            elif data[j] == 2:
                operatingSystems.append([0, 1, 0, 0, 0, 0, 0, 0])
            elif data[j] == 3:
                operatingSystems.append([0, 0, 1, 0, 0, 0, 0, 0])
            elif data[j] == 4:
                operatingSystems.append([0, 0, 0, 1, 0, 0, 0, 0])
            elif data[j] == 5:
                operatingSystems.append([0, 0, 0, 0, 1, 0, 0, 0])
            elif data[j] == 6:
                operatingSystems.append([0, 0, 0, 0, 0, 1, 0, 0])
            elif data[j] == 7:
                operatingSystems.append([0, 0, 0, 0, 0, 0, 1, 0])
            elif data[j] == 8:
                operatingSystems.append([0, 0, 0, 0, 0, 0, 0, 1])

    elif i == 'Region':
        data = list(dataset[i])
        for j in range(m):
            if data[j] == 1:
                region.append([1, 0, 0, 0, 0, 0, 0, 0, 0])
            elif data[j] == 2:
                region.append([0, 1, 0, 0, 0, 0, 0, 0, 0])
            elif data[j] == 3:
                region.append([0, 0, 1, 0, 0, 0, 0, 0, 0])
            elif data[j] == 4:
                region.append([0, 0, 0, 1, 0, 0, 0, 0, 0])
            elif data[j] == 5:
                region.append([0, 0, 0, 0, 1, 0, 0, 0, 0])
            elif data[j] == 6:
                region.append([0, 0, 0, 0, 0, 1, 0, 0, 0])
            elif data[j] == 7:
                region.append([0, 0, 0, 0, 0, 0, 1, 0, 0])
            elif data[j] == 8:
                region.append([0, 0, 0, 0, 0, 0, 0, 1, 0])
            elif data[j] == 9:
                region.append([0, 0, 0, 0, 0, 0, 0, 0, 1])

print(dataset.columns)
# print(month[:5])
# print(weekend[:5])
# What differ?
# print(dataset.iloc[:, -1][:5])
# print(dataset.iloc[:, 17:18][:5])

X_0 = dataset.iloc[:, 0:10].values
X_1 = dataset.iloc[:, 14:15].values  # traffic type
y_ = dataset.iloc[:, 17:18].values
X_ = np.append(X_0, month, axis=1)
X_ = np.append(X_, operatingSystems, axis=1)
X_ = np.append(X_, browser, axis=1)
X_ = np.append(X_, region, axis=1)
X_ = np.append(X_, X_1, axis=1)
X_ = np.append(X_, visitorType, axis=1)
X_ = np.append(X_, weekend, axis=1)
df_ = np.append(X_, y_, axis=1)
# print('df_ shape:', df_.shape)


featureNames = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
                'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay',
                'Month1', 'Month2', 'Month3', 'Month4', 'Month5', 'Month6', 'Month7','Month8', 'Month9', 'Month10',
                'Month11', 'Month12',
                'os1', 'os2', 'os3', 'os4', 'os5', 'os6', 'os7', 'os8',
                'browser1', 'browser2', 'browser3', 'browser4', 'browser5', 'browser6', 'browser7', 'browser8',
                'browser9', 'browser10', 'browser11', 'browser12', 'browser13',
                'region1', 'region2', 'region3', 'region4', 'region5', 'region6', 'region7', 'region8', 'region9',
                'TrafficType', 'VisitorType1', 'VisitorType2', 'VisitorType3', 'Weekend1', 'Weekend2', 'Revenue']
df = pd.DataFrame(df_, columns=featureNames)
# print(df.head(1))

# Dividing X and y
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
print('X =', X[:5])
print('y =', y[:5])

bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X, y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Specs', 'Score']
print(featureScores.nlargest(10, 'Score'))
