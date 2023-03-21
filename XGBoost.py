import numpy as np
import pandas as pd
import os
import xgboost as xg
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

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

# Instantiation
xgb_r = xg.XGBRegressor(objective ='reg:absoluteerror', n_estimators = 145, max_depth=5, eta=0.3)

from sklearn.model_selection import GridSearchCV
# set up our search grid
param_grid = {"max_depth":    [4, 5],
              "n_estimators": [100, 145, 170],
              "learning_rate": [0.1, 0.3]}

# try out every combination of the above values
search = GridSearchCV(xgb_r, param_grid, cv=5).fit(X_train, y_train)

print("The best hyperparameters are ",search.best_params_)

regressor= xg.XGBRegressor(learning_rate = search.best_params_["learning_rate"], n_estimators  = search.best_params_["n_estimators"], max_depth     = search.best_params_["max_depth"],)

regressor.fit(X_train, y_train)
pred = regressor.predict(X_test)

# # Fitting the model
# xgb_r.fit(X_train,y_train) 

# # Predict the model
# pred = xgb_r.predict( X_test)
 
# # RMSE Computation
rmse = mae(y_test, pred)
print("RMSE : % f" %(rmse))
