import numpy as np
import pandas as pd
import os
import catboost as cb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance

os.chdir("C:/Users/Ada/Documents/AppMamProject/data/Finaldata" )

df = pd.read_csv("Rfin.csv")
p= df.columns
features=list(p.delete(0))
X= df.loc[:, features].values
X = StandardScaler().fit_transform(X)
y = df.loc[:,['age']].values# Standardizing the target
y=df['age'].tolist()
y = np.array(y)
#print(X, y)

#Train test split
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2, random_state=11)

train_dataset = cb.Pool(X_train, y_train) 
test_dataset = cb.Pool(X_test, y_test)

model = cb.CatBoostRegressor(loss_function='MAE')

grid = {'iterations': [100, 150, 200], 'learning_rate': [0.03, 0.1], 'depth': [2, 4, 6, 8], 'l2_leaf_reg': [0.2, 0.5, 1, 3]}
model.grid_search(grid, train_dataset)

pred = model.predict(X_test)
maee = (mae(y_test, pred))
r2 = r2_score(y_test, pred)
print("Testing performance")
print('RMSE: {:.2f}'.format(maee))
print('R2: {:.2f}'.format(r2))