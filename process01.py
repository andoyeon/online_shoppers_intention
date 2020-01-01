import pandas as pd
import matplotlib.pyplot as plt

# 1. 데이터 준비
dataset = pd.read_csv("online_shoppers_intention.csv", encoding='utf-8')

# 데이터 확인
print(dataset.shape)
print(dataset.info())
print(dataset.describe())

# 데이터 나누기
# X = dataset.iloc[:, :-1]
# y = dataset.iloc[:, -1:]
# print('X[5] =', X[:5])
# print('y[5] =', y[:5])

# 결측치 확인
print(dataset.isnull().sum())

# 2. 데이터 전처리

# 문자형 데이터 -> 숫자형으로 변환
# Python2 (11) - scratch12 나이브 베이즈(p9)

month = set(dataset.loc[:, 'Month'])
print(month)
# {'Nov', 'Aug', 'Dec', 'Oct', 'Jul', 'June', 'Feb', 'Sep', 'Mar', 'May'}
dataset.loc[dataset['Month'] == 'Jan', 'Month'] = 0
dataset.loc[dataset['Month'] == 'Feb', 'Month'] = 1
dataset.loc[dataset['Month'] == 'Mar', 'Month'] = 2
dataset.loc[dataset['Month'] == 'Apr', 'Month'] = 3
dataset.loc[dataset['Month'] == 'May', 'Month'] = 4
dataset.loc[dataset['Month'] == 'June', 'Month'] = 5
dataset.loc[dataset['Month'] == 'Jul', 'Month'] = 6
dataset.loc[dataset['Month'] == 'Aug', 'Month'] = 7
dataset.loc[dataset['Month'] == 'Sep', 'Month'] = 8
dataset.loc[dataset['Month'] == 'Oct', 'Month'] = 9
dataset.loc[dataset['Month'] == 'Nov', 'Month'] = 10
dataset.loc[dataset['Month'] == 'Dec', 'Month'] = 11
print(dataset.loc[:, 'Month'])

visitor_type = set(dataset.loc[:, 'VisitorType'])
print(visitor_type)
# {'Returning_Visitor', 'Other', 'New_Visitor'}
dataset.loc[dataset['VisitorType'] == 'Returning_Visitor', 'VisitorType'] == 0

# 변수 의미 파악
# 영향력 있는 변수 selecting -> sklearn.feature_selection.SelectBest






