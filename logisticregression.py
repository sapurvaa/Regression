import math
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit

def cost_func(theta, X_train, y_train):
    z = np.dot(X_train,theta)
    hyp = np.float64(1. /(1. + pow(math.e,-z)))
    total = np.sum(-(y_train*(np.log(hyp)))-((1-y_train)*(np.log(1-hyp))))
    cost = total/len(X_train)
    return cost

def gradient_descent(theta, X_train, y_train):
    alpha = 0.001
    z = np.dot(X_train, theta)
    hyp = np.float64(1. /(1. + pow(math.e,-z)))
    total = np.dot(X_train.T,(hyp - y_train))
    new_theta = theta - ((alpha * total)/len(X_train))
    return new_theta

def feature_scaling(unscaled_data):
    data = unscaled_data
    data[:,:-1] = (unscaled_data[:,:-1] - np.mean(unscaled_data[:,:-1]))/(np.max(unscaled_data[:,:-1])- np.min(unscaled_data[:,:-1]))
    return data

if __name__ == '__main__':
    complete_data = pd.read_csv("BSOM_DataSet_for_HW2.csv", usecols=("all_mcqs_avg_n20", "all_NBME_avg_n4", "LEVEL"))
    #print(complete_data)
    features = complete_data.shape[1]
    #print("number of features is " + str(features))
    unscaled_data = complete_data.to_numpy()
    #print(unscaled_data)
    temp_data = feature_scaling(unscaled_data)
    #print("scaled data " + str(temp_data))
    data = np.append(np.ones([len(temp_data), 1]), temp_data, 1)
    #print("final data " + str(data))

    X = data[:, :-1]
    y = data[:,features]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=27)
    y_train = y_train[:,None]
    y_test = y_test[:,None]

    labels = ["A","B","C","D"]
    y_master_train = y_train

    final_theta = []
    for label in labels:
        num_iterations = 1
        #print("Label " + label)
        for i in range(len(y_master_train)):
            if y_master_train[i] == label:
                y_train[i,0] = 1
            else:
                y_train[i,0] = 0
        theta = np.random.rand(features, 1)
        #print("initial theta values " + str(theta))
        cost = cost_func(theta, X_train, y_train)
        print("initial cost " + str(cost))
        while(num_iterations < 400):
            new_theta = gradient_descent(theta, X_train, y_train)
            #print("new theta values " + str(new_theta))
            new_cost = cost_func(new_theta, X_train, y_train)
            #print("new cost " + str(cost))
            theta = new_theta
            cost=new_cost
            num_iterations = num_iterations + 1
        #print("final theta val for label " + label + " " + str(theta))
        print("final cost for label "+ label + " "+str(cost))
        final_theta.append(theta.T.flatten().tolist())
    theta_labels = np.array(final_theta)
    #print("Final theta values for all labels")
    #print(theta_labels)

    z = np.dot(X_test,theta_labels.T)
    hyp = np.float64(1. / (1. + pow(math.e, -z)))
    hypothesis_table = np.argmax(hyp,axis =1)
   print("predicted values" ,hypothesis_table)

    y_master_test = y_test
    for i in range(len(y_master_test)):
        y_test[i] = labels.index(y_master_test[i])

    actual = np.array(y_test.T.flatten().tolist())
    print("actual values" ,actual)

    results = confusion_matrix(actual, hypothesis_table)
    print("confusion matrix" + str(results))
    accuracy = accuracy_score(actual, hypothesis_table)
    print(accuracy)
    print(classification_report(actual, hypothesis_table, target_names=['A', 'B', 'C', 'D']))