import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("D:\\uni\\Machine Learning\\Assignment\\Lab2\\assignment\\assignment1dataset.csv")
print(data.describe())
features = ["NCustomersPerDay", "AverageOrderValue", "WorkingHoursPerDay", "NEmployees", "MarketingSpendPerDay", "LocationFootTraffic"]
X=data.iloc[ :, 1:] # kol al rows l kol al coulmbs ela awel coulmb 3shan da al target
X = X.to_numpy()
Y=data['RevenuePerDay'] #TARGET VALUE
Y = Y.to_numpy()
total_error_ep = []
mse_all_features = []

def gradient_descent(X, Y, L=0.0000001, epochs=100):
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X) #normalize data 3shan al overflow ally kan by7sal
    Y_norm = scaler.fit_transform(Y.reshape(-1, 1)).flatten()
    n, features = X_normalized.shape
    m = np.zeros(features)
    c = 0  
    MSE = [[] for _ in range(features)] #list of lists shayla l kol feature al mse bta3 kol al points
    predictions = []#list feeha al final pred bta3 kol feature

    for i in range(epochs):

        Y_pred_all = np.dot(X_normalized, m) + c
        D_m = (-2/n) * np.dot(X_normalized.T, (Y_norm - Y_pred_all))
        D_c = (-2/n) * np.sum(Y_norm - Y_pred_all)
        m -= L * D_m
        c -= L * D_c
        for j in range(features):
            Y_pred_j = m[j] * X_normalized[:, j] + c
            MSE[j].append(np.mean((Y_norm - Y_pred_j) ** 2))
        
       
        predictions.append(Y_pred_all)
    return  MSE, predictions
def plot_mse_subplots(MSE_epochs):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for j in range(len(MSE_epochs)):  
        axes[0].plot(range(1, len(MSE_epochs[j]) + 1), MSE_epochs[j], label=f"{features[j]}" if features else f"Feature {j + 1}")
    
    axes[0].set_xlabel("Epochs", fontsize=14)
    axes[0].set_ylabel("Mean Squared Error", fontsize=14)
    axes[0].set_title("MSE vs. Epochs", fontsize=16)
    axes[0].legend()
    axes[0].grid(True)

    for j in range(len(features)):
        mse_values = [mse_all_features[i][j] for i in range(len(L))]
        axes[1].plot(L, mse_values, marker='o', label=features[j])

    axes[1].set_xlabel("Learning Rate", fontsize=14)
    axes[1].set_ylabel("Mean Squared Error (MSE)", fontsize=14)
    axes[1].set_title("MSE vs. Learning Rate for Different Features", fontsize=16)
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_xscale("log")

    plt.tight_layout()
    plt.show()


#epochs observations
epochs = [50, 100, 150, 200, 250, 300, 350, 400]
for i in epochs:
    for j in range(X.shape[1]):
        mse_ep , pred = gradient_descent(X,Y,0.0000001,i)
        total_error_ep = [sum(mse_per_feature) for mse_per_feature in mse_ep]
        print(f"\nfor feature {j + 1} at epochs = {i} the MSE = {total_error_ep[j]} ")
    min_error = min(total_error_ep)
    min_index = total_error_ep.index(min_error)
    print(f"\n minimum error  is {min_error}, selected feature is {features[min_index]}\n \n")
print("*****************************************************************************************************************************************************")

# Learning rate observations
L = [0.00000125, 0.00000125, 0.0000125, 0.000125,  0.00125,  0.0125]
for i in L:
    mse_per_lr = []
    for j in range(X.shape[1]):
        mse_l, pred = gradient_descent(X, Y, i, 100)
        total_error_ep = [sum(mse_per_feature) for mse_per_feature in mse_l]
        mse_per_lr.append(total_error_ep[j])
        print(f" for feature {j + 1} at Learning rate = {i} the MSE = {total_error_ep[j]} ")
    
    mse_all_features.append(mse_per_lr)
    min_error = min(total_error_ep)
    min_index = total_error_ep.index(min_error)
    print(f"\n minimum error  is {min_error}, selected feature is {features[min_index]}\n \n")

print("*****************************************************************************************************************************************************")
plot_mse_subplots(mse_ep)
