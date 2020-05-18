import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import math

def cost_func(theta, X_train, y_train):
    hyp = np.dot(X_train,theta)
    cost = np.sum(pow(hyp - y_train,2))/(2*len(X_train))
    return cost

def gradient_descent(theta, X_train, y_train):
    alpha = 0.001
    hyp = np.dot(X_train, theta)
    total_sum = np.sum(np.dot((hyp - y_train).T,X_train))
    new_theta = theta - (alpha * total_sum)/len(X_train)
    return new_theta

def plot_graph(X_test,y_test,theta):
    hyp = np.dot(X_test,theta)
    #fig = plt.figure()
    #ax = plt.axes(projection='3d')
    #ax.scatter3d(X_test[:,1],X_test[:,2],y_test)
    #ax.plot3d(X_test[:,1],X_test[:,2],hyp)
    plt.scatter(X_test[:,1:],y_test)
    plt.plot(X_test[:,1:],hyp)
    plt.xlabel('X values')
    #plt.ylabel("X2 values")
    #plt.zlabel("hypothesis")
    plt.ylabel("hypothesis")
    plt.show()

def MSE_accuracy(theta,X_test,y_test):
    hyp = np.dot(X_test,theta)
    MSE_sum = np.sum(pow((hyp - y_test),2))
    accuracy = MSE_sum/len(y_test)
    return accuracy

def Rsquared_accuracy(theta,X_test,y_test):
    hyp = np.dot(X_test, theta)
    SSres = np.sum(pow((y_test - hyp),2))
    SStot = np.sum(pow((y_test - np.mean(y_test)),2))
    accuracy = 1 - (SSres/SStot)
    return accuracy

if __name__ == '__main__':
    complete_data = pd.read_csv("BSOM_DataSet_for_HW2.csv", usecols=("all_mcqs_avg_n20", "STEP_1"))
    #complete_data = pd.read_csv("BSOM_DataSet_for_HW2.csv", usecols=("all_mcqs_avg_n20", "all_NBME_avg_n4", "STEP_1"))
    #print(complete_data)

    features = complete_data.shape[1]
    #print("number of features is " + str(features))
    target_var = "STEP_1"
    #print("target variable is " + target_var)

    temp_data = complete_data.to_numpy()
    data = np.append(np.ones([len(temp_data), 1]), temp_data, 1)
    #print("final data " + str(data))

    X = data[:, :-1]
    y = data[:, features]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)
    #print(str(X_train.shape) + str(X_test.shape) + str(y_train.shape) + str(y_test.shape))

    theta_array = []
    theta = np.random.rand(features,1)
    theta_array.append(theta)
    print("initial theta values " + str(theta))
    #print(theta.shape)

    num_iterations = 1
    cost_array = []

    cost = cost_func(theta, X_train, y_train)
    cost_array.append(cost)
    print("initial cost " + str(cost))

    while(num_iterations < 100):
        new_theta = gradient_descent(theta, X_train, y_train)
        #print("new theta values " + str(new_theta))
        new_cost = cost_func(new_theta, X_train, y_train)
        #print("new cost " + str(new_cost))
        cost = new_cost
        cost_array.append(cost)
        theta = new_theta
        theta_array.append(theta)
        num_iterations =num_iterations+1

    print("final theta val " + str(theta))
    print("final cost: "+ str(cost))

    plot_graph(X_test,y_test,theta)
    accuracy = MSE_accuracy(theta,X_test,y_test)
    print("Mean squared error performance is " + str(accuracy))
    accuracy = Rsquared_accuracy(theta,X_test,y_test)
    print("R squared performance value is " + str(accuracy))