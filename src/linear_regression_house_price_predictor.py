import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import math

BEDROOMS = 'bedrooms'
BATHROOMS = 'bathrooms'
SQFT_LIVING = 'sqft_living'
SQFT_LOT = 'sqft_lot'
FLOORS = 'floors'
WATERFRONT = 'waterfront'
VIEW = 'view'
CONDITION = 'condition'
GRADE = 'grade'
SQFT_ABOVE = 'sqft_above'
SQFT_BASEMENT = 'sqft_basement'
YEAR_BUILT = 'yr_built'
YEAR_RENOVATED = 'yr_renovated'
ZIPCODE = 'zipcode'
LATITUDE = 'lat'
LONGITUDE = 'long'
SQFT_LIVING15 = 'sqft_living15'
SQFT_LOT15 = 'sqft_lot15'
PRICE = 'price'

COLON = ":"

data = pd.read_csv("../input/kc_house_data.csv")
print("---\ndata.dtypes:\n" + str(data.dtypes))
print("---\ndata.shape:\n" + str(data.shape))
print("---\ndata.describe():\n" + str(data.describe()))

# draw diagram to analyze the data: 100% - fully linear correlated, 0% - not linear correlated
# draw scatter plot for continuous var (real number)
# draw histogram plot for discrete var (classification, true/false)

# bedrooms: 50%
plt.scatter(data[BEDROOMS], data[PRICE])
plt.title(BEDROOMS + COLON + PRICE)
plt.xlabel(BEDROOMS)
plt.ylabel(PRICE)
plt.show()

# bathrooms: 50%
plt.scatter(data[BATHROOMS], data[PRICE])
plt.title(BATHROOMS + COLON + PRICE)
plt.xlabel(BATHROOMS)
plt.ylabel(PRICE)
plt.show()

# sqft_living: 70%
plt.scatter(data[SQFT_LIVING], data[PRICE])
plt.title(SQFT_LIVING + COLON + PRICE)
plt.xlabel(SQFT_LIVING)
plt.ylabel(PRICE)
plt.show()

# sqft_lot: 0%
plt.scatter(data[SQFT_LOT], data[PRICE])
plt.title(SQFT_LOT + COLON + PRICE)
plt.xlabel(SQFT_LOT)
plt.ylabel(PRICE)
plt.show()

# floors: 0%
plt.scatter(data[FLOORS], data[PRICE])
plt.title(FLOORS + COLON + PRICE)
plt.xlabel(FLOORS)
plt.ylabel(PRICE)
plt.show()

# waterfront: 0%
plt.scatter(data[WATERFRONT], data[PRICE])
plt.title(WATERFRONT + COLON + PRICE)
plt.xlabel(WATERFRONT)
plt.ylabel(PRICE)
plt.show()

# view: 0%
plt.scatter(data[VIEW], data[PRICE])
plt.title(VIEW + COLON + PRICE)
plt.xlabel(VIEW)
plt.ylabel(PRICE)
plt.show()

# condition: 0%
plt.scatter(data[CONDITION], data[PRICE])
plt.title(CONDITION + COLON + PRICE)
plt.xlabel(CONDITION)
plt.ylabel(PRICE)
plt.show()

# grade: 80%
plt.scatter(data[GRADE], data[PRICE])
plt.title(GRADE + COLON + PRICE)
plt.xlabel(GRADE)
plt.ylabel(PRICE)
plt.show()

# sqft_above: 50%
plt.scatter(data[SQFT_ABOVE], data[PRICE])
plt.title(SQFT_ABOVE + COLON + PRICE)
plt.xlabel(SQFT_ABOVE)
plt.ylabel(PRICE)
plt.show()

# sqft_basement: 5%
plt.scatter(data[SQFT_BASEMENT], data[PRICE])
plt.title(SQFT_BASEMENT + COLON + PRICE)
plt.xlabel(SQFT_BASEMENT)
plt.ylabel(PRICE)
plt.show()

# yr_built: 0%
plt.scatter(data[YEAR_BUILT], data[PRICE])
plt.title(YEAR_BUILT + COLON + PRICE)
plt.xlabel(YEAR_BUILT)
plt.ylabel(PRICE)
plt.show()

# yr_renovated: 60%
plt.scatter(data[YEAR_RENOVATED], data[PRICE])
plt.title(YEAR_RENOVATED + COLON + PRICE)
plt.xlabel(YEAR_RENOVATED)
plt.ylabel(PRICE)
plt.show()
# FOUND empty data in YEAR_RENOVATED, fill it with median value
data_copy = data.copy(deep=True)  # good practice to modify deep copy of data, and preserve original data
print('data_copy[YEAR_RENOVATED]:\n' + str(data_copy[YEAR_RENOVATED]))
y = data_copy[YEAR_RENOVATED]
cleaned_y = y[y != 0]
print('cleaned_y:\n' + str(cleaned_y))
print('cleaned_y.median():\n' + str(np.median(cleaned_y)))
data_copy[YEAR_RENOVATED].replace(0, np.median(cleaned_y), inplace=True)
print('cleaned data_copy[YEAR_RENOVATED]:\n' + str(data_copy[YEAR_RENOVATED]))
plt.scatter(data_copy[YEAR_RENOVATED], data_copy[PRICE])
plt.title('cleaned ' + YEAR_RENOVATED + COLON + PRICE)
plt.xlabel('cleaned ' + YEAR_RENOVATED)
plt.ylabel(PRICE)
plt.show()

# zipcode: 0%
plt.scatter(data[ZIPCODE], data[PRICE])
plt.title(ZIPCODE + COLON + PRICE)
plt.xlabel(ZIPCODE)
plt.ylabel(PRICE)
plt.show()

# latitude: 10%
plt.scatter(data[LATITUDE], data[PRICE])
plt.title(LATITUDE + COLON + PRICE)
plt.xlabel(LATITUDE)
plt.ylabel(PRICE)
plt.show()

# longitude: 10%
plt.scatter(data[LONGITUDE], data[PRICE])
plt.title(LONGITUDE + COLON + PRICE)
plt.xlabel(LONGITUDE)
plt.ylabel(PRICE)
plt.show()

# SQFT LIVING15: 70%
plt.scatter(data[SQFT_LIVING15], data[PRICE])
plt.title(SQFT_LIVING15 + COLON + PRICE)
plt.xlabel(SQFT_LIVING15)
plt.ylabel(PRICE)
plt.show()

# SQFT LOT15: -50%
plt.scatter(data[SQFT_LOT15], data[PRICE])
plt.title(SQFT_LOT15 + COLON + PRICE)
plt.xlabel(SQFT_LOT15)
plt.ylabel(PRICE)
plt.show()

# Based on plot, find out the linear correlated independent var
X = data_copy[[BEDROOMS, BATHROOMS, SQFT_LIVING, GRADE, SQFT_ABOVE, YEAR_RENOVATED, SQFT_LIVING15]]
Y = data_copy[PRICE]

# Split train / test data
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=1/3, random_state=0)
print("Shape: \nxtrain.shape - " + str(xtrain.shape) + "\nytrain.shape - " + str(ytrain.shape) + "\nxtest.shape - " +
      str(xtest.shape) + "\nytest.shape - " + str(ytest.shape))

xtrain = np.asmatrix(xtrain)
xtest = np.asmatrix(xtest)
ytrain = np.ravel(ytrain)
ytest = np.ravel(ytest)

# train the model
model = LinearRegression()
model.fit(xtrain, ytrain)  # train

LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
print("\ncoef:\n" + str(pd.DataFrame(list(zip(X.columns, model.coef_)))))  # zip: put 2 list item in the same row

print("show all coefficient: " + str(model.coef_))
print("show intercept: " + str(model.intercept_))

# predict
print("xtest[0, :]: " + str(xtest[0, :]))
print("model.predict(xtest[0, :]: " + str(model.predict(xtest[0, :])))
print("ytest[0]: " + str(ytest[0]))

# calculate MSE(Mean Square Error) of training data set
pred = model.predict(xtrain)
print("MSE: " + str(((pred - ytrain) * (pred - ytrain)).sum() / len(ytrain)))
print("MSE: " + str(metrics.mean_squared_error(ytrain, pred)))

# average relative error
print("ARE: " + str((abs(pred-ytrain)/ytrain).sum() / len(ytrain)))

# MSE of test data set
predtest = model.predict(xtest)
print("MSE: " + str(metrics.mean_squared_error(ytest, predtest)))
# ARE
print("ARE: " + str(abs(predtest-ytest)/ytest).sum() / len(ytest))
