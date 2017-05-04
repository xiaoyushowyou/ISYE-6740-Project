import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
import time

df_train = pd.read_csv("data/train.csv", dtype={"place_id": object})

df_train['hour'] = df_train.time // 60 % 24
df_train['weekday'] = df_train.time // (60*24) % 7
df_train["month"] = df_train.time // (60*24*30) % 12
df_train["year"] = df_train.time // (60*24*365)
df_train["day"] = df_train.time // (60*24*7) % 365

df_train.loc[:,'y'] *= 2
df_train.loc[:, 'hour'] *= 0.01
df_train.loc[:, 'weekday'] *= 0.002
df_train.loc[:, 'accuracy'] *= 0.001

df_filter = df_train[(df_train['x'] > 1) & (df_train['x'] < 1.25) & (df_train['y'] > 1)& (df_train['y'] < 1.25) ]

Y = df_filter["place_id"].as_matrix()
X = df_filter.drop(["row_id","place_id","time"], 1).as_matrix()


x_train, x_test, y_train, y_test \
        = train_test_split(X, Y, test_size = 0.2, random_state = 2017 )


start = time.time()
logisticModel = LogisticRegression(multi_class='ovr',penalty ='l1',C=1);
logisticModel.fit(x_train, y_train)
#train_score = logisticModel.predict(x_train, y_train)

test_score = logisticModel.score(x_test, y_test)
end = time.time()
print ("Testing error of Losgitic regression using L1 penalty is:", test_score)
print ("Total Time is:", end-start)

start = time.time()
logisticModel = LogisticRegression(multi_class='ovr',penalty ='l2',C=1);
logisticModel.fit(x_train, y_train)
test_score = logisticModel.score(x_test, y_test)
end = time.time()

print ("Testing error of Losgitic regression using L2 penalty is:", test_score)
print ("Total Time is:", end-start)

