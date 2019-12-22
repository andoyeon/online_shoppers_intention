import pandas as pd

# 1. 데이터 준비
online_shoppers_intention = pd.read_csv("online_shoppers_intention.csv",
                                        encoding='utf-8')
# 데이터 확인
print(online_shoppers_intention.shape)
print(online_shoppers_intention.info())
print(online_shoppers_intention.describe())

features = list(online_shoppers_intention.columns)
print('features:', features)

X = online_shoppers_intention.iloc[:, :-1]
print(X.columns)
y = online_shoppers_intention.iloc[:, -1:]
print(y.columns)

