import pandas as pd
import matplotlib.pyplot as plt

# 1. 데이터 준비
dataset = pd.read_csv("online_shoppers_intention.csv",
                                        encoding='utf-8')
# 데이터 확인
print(dataset.shape)
print(dataset.info())
print(dataset.describe())


X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1:]
print('X[5] =', X[:5])
print('y[5] =', y[:5])


# 결측치 확인
print(dataset.isnull().sum())

# 2. 데이터 전처리

# 문자형 데이터 -> 숫자형으로 변환
print(dataset.info())
# Python2 (11) - scratch12 나이브 베이즈(p9)

month = set(dataset.loc[:, 'Month'])
print(month)
# {'Nov', 'Aug', 'Dec', 'Oct', 'Jul', 'June', 'Feb', 'Sep', 'Mar', 'May'}
dataset.loc[dataset['Month'] == 'Jan'] = 0
dataset.loc[dataset['Month'] == 'Feb'] = 1
dataset.loc[dataset['Month'] == 'Mar'] = 2
dataset.loc[dataset['Month'] == 'Apr'] = 3
dataset.loc[dataset['Month'] == 'May'] = 4
dataset.loc[dataset['Month'] == 'June'] = 5
dataset.loc[dataset['Month'] == 'Jul'] = 6
dataset.loc[dataset['Month'] == 'Aug'] = 7
dataset.loc[dataset['Month'] == 'Sep'] = 8
dataset.loc[dataset['Month'] == 'Oct'] = 9
dataset.loc[dataset['Month'] == 'Nov'] = 10
dataset.loc[dataset['Month'] == 'Dec'] = 11
print(month)

visitor_type = set(dataset.loc[:, 'VisitorType'])
print(visitor_type)
# {'Returning_Visitor', 'Other', 'New_Visitor'}




