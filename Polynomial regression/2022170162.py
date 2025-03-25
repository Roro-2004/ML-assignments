import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


data = pd.read_csv("D:\\uni\\Machine Learning\\Assignment\\Lab2\\assignment\\assignment1dataset.csv")
print(data.describe())
features = ["NCustomersPerDay", "AverageOrderValue", "WorkingHoursPerDay", "NEmployees", "MarketingSpendPerDay", "LocationFootTraffic"]
X=data.iloc[ :, 1:] # kol al rows l kol al coulmbs ela awel coulmb 3shan da al target
Y=data['RevenuePerDay'] #TARGET VALUE


corr = data.corr()
top_feature = corr.index[abs(corr['RevenuePerDay'])>0.5]
plt.subplots(figsize=(12, 8))
top_corr = data[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()
top_feature = top_feature.delete(0)
X = X[top_feature]


def polynomial_features(X, degree):

    if not isinstance(X, np.ndarray):
        X = np.array(X)
    NumberOfSamples, NumberOfFeatures = X.shape
    Polyfeatures = [np.ones(NumberOfSamples)]
    for degree in range(1, degree + 1):
        for element in itertools.combinations_with_replacement(range(NumberOfFeatures), degree):
            NewFeature = np.prod(X[:, element], axis=1)
            Polyfeatures.append(NewFeature)

    return np.array(Polyfeatures).transpose()

def polynomial_regression_analysis(X, Y, max_degree=8, test_size=0.2, random_state=70):

    mse_train = []
    mse_test = []

    for degree in range(1, max_degree + 1):
        ft = polynomial_features(X, degree)
        X_train, X_test, y_train, y_test = train_test_split(ft, Y, test_size=test_size, shuffle=True, random_state=random_state)
        
        poly_model = linear_model.LinearRegression()
        poly_model.fit(X_train, y_train)

        y_train_predicted = poly_model.predict(X_train)
        tst_prediction = poly_model.predict(X_test)

        mse_train.append(metrics.mean_squared_error(y_train, y_train_predicted))
        mse_test.append(metrics.mean_squared_error(y_test, tst_prediction))

    # Plotting the results
    degrees = range(1, max_degree + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(degrees, mse_train, label="Train MSE", marker='o', linestyle='-')
    plt.plot(degrees, mse_test, label="Test MSE", marker='s', linestyle='--')

    plt.xlabel("Polynomial Degree")
    plt.ylabel("Mean Squared Error")
    plt.title("MSE vs. Polynomial Degree")
    plt.legend()
    plt.grid()
    plt.show()

    return mse_train, mse_test

polynomial_regression_analysis(X, Y, 8, 0.2, 42)