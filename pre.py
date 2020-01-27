import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
month = []
visitorType = []
weekend = []
browser = []
operatingSystems = []
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
            if data[j] == True:
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

# print(month[:5])
# print(weekend[:5])
# What differ?
# print(dataset.iloc[:, -1][:5])
# print(dataset.iloc[:, 17:18][:5])

X_0 = dataset.iloc[:, 0:10].values
X_1 = dataset.iloc[:, 13:15].values
y_ = dataset.iloc[:, 17:18].values
X_ = np.append(X_0, month, axis=1)
X_ = np.append(X_0, operatingSystems, axis=1)
X_ = np.append(X_0, browser, axis=1)
X_ = np.append(X_, X_1, axis=1)
X_ = np.append(X_, visitorType, axis=1)
X_ = np.append(X_, weekend, axis=1)
df = np.append(X_, y_, axis=1)
print('df shape:', df.shape)

featureNames = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
                'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues',
                'SpecialDay', 'Month1', 'Month2', 'Month3', 'Month4', 'Month5', 'Month6', 'Month7',
                'Month8', 'Month9', 'Month10', 'Month11', 'Month12', 'OperatingSystems', 'Browser',
                'Region', 'TrafficType', 'VisitorType1', 'VisitorType2', 'VisitorType3',
                'Weekend1', 'Weekend2', 'Revenue']
