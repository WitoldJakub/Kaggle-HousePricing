# House Pircing - Random Forest Regression!

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')
testset = pd.read_csv('test.csv')

'''
linear regression on 
1	2d	4	5d	7d	10d	13d	14d	15d	16d	17	18	2018 - 20	25d	26	28d	39d	
40d	41d	46	47	48	49	50	51	52	55d	56	61	62	65d	2018 - 77	78d

to be dummed
dataset -> [2, 5, 7, 10, 13, 14, 15, 16, 25, 28, 39, 40, 41, 55, 65, 78]
indeces X -> [1, 3, 4, 5, 6, 7, 8, 9, 13, 15, 16, 17, 18, 26, 30, 32]
'''
# Selecting supposely relevant future
Xc = dataset.iloc[:, [1, 2, 4, 5, 7, 10, 13, 14, 15, 16, 17, 18, 20, 25, 26, 28, 39, 40, 41, 46, 47, 48, 49, 50, 51, 52, 55, 56, 61, 62, 65, 77, 78]]
Xc_test = testset.iloc[:, [1, 2, 4, 5, 7, 10, 13, 14, 15, 16, 17, 18, 20, 25, 26, 28, 39, 40, 41, 46, 47, 48, 49, 50, 51, 52, 55, 56, 61, 62, 65, 77, 78]]
yc = dataset.iloc[:, 80]

# Encoding categorical data
# concatanating to avoid missing dummies columns
X_all = pd.concat([Xc, Xc_test])
column_names = Xc.columns[[1, 3, 4, 5, 6, 7, 8, 9, 13, 15, 16, 17, 18, 26, 30, 32]]
X_all_dummies = pd.get_dummies(X_all, columns = column_names, drop_first = True)


# Panda to numpy array, spliting again to train and test 
X = np.asarray(X_all_dummies[0:1460], dtype = 'float32')
X_test = np.asarray(X_all_dummies[1460:2919], dtype = 'float32')
y = np.asarray(yc, dtype = 'float32')

# Making the year values as year passed to 2011. 
# Seems more relevant to me for the model feauture.
X[:, [4]] = 2011 - X[:, [4]]
X[:, [16]] = 2011 - X[:, [16]]

X_test[:, [4]] = 2011 - X_test[:, [4]]
X_test[:, [16]] = 2011 - X_test[:, [16]]

# Taking care of missing data for two garrage features
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer = imputer.fit(X[:, 14:16])
X[:, 14:16] = imputer.transform(X[:, 14:16])

imputer = imputer.fit(X_test[:, 14:16])
X_test[:, 14:16] = imputer.transform(X_test[:, 14:16])
#  For Bsmt* two featues another NaNs spotted
imputer = imputer.fit(X_test[:, 7:9])
X_test[:, 7:9] = imputer.transform(X_test[:, 7:9])

# Replacing NaNs in $MasVnrArea
imputer2 = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer2 = imputer2.fit(X[:, 5:6])
X[:, 5:6] = imputer2.transform(X[:, 5:6])

imputer2 = imputer2.fit(X_test[:, 5:6])
X_test[:, 5:6] = imputer2.transform(X_test[:, 5:6])

# Splitting the dataset into the Training set and CrossValidation set
from sklearn.model_selection import train_test_split
X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size = 1/5, random_state = 0)


# Fitting the Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)

# Predicting the CV set results
y_pred_cv = regressor.predict(X_cv)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

regressor.score(X_train, y_train)
regressor.score(X_cv, y_cv)

# Creating output predictions list
y_predv = y_pred.reshape(-1, 1)
IDs = testset.iloc[:, [0]].values
exp_pred = np.concatenate([IDs, y_predv], axis = 1)

# Creating and printing dataframe to .csv with headers and proper formating
out_panda = pd.DataFrame(exp_pred, columns=['Id', 'SalePrice'])
out_panda['Id'] = out_panda['Id'].apply('{:.0f}'.format)
out_panda['SalePrice'] = out_panda['SalePrice'].apply('{:.9f}'.format)
out_panda.to_csv('solutions\HousePricingRF0.csv', index = False, header = True, sep = ',')