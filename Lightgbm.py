import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import StandardScaler
import lightgbm as ltb 

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

model = ltb.LGBMRegressor()
model.fit(X_train, y_train)
print(); print(model)

expected_y  = y_test
predicted_y = model.predict(X_test)

plt.figure(figsize=(10,10))
sns.regplot(expected_y, predicted_y, fit_reg=True, scatter_kws={"s": 100})
