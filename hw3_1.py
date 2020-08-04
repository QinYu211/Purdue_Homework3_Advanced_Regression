import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt


def main():
    #Importing dataset
    originalData = pd.read_csv('bike-data.csv')
    
    #Cleaning dataset
    precipData, totalData = clean_data(originalData)
    
    #Create the Feature Matrix
    X = # Fill In
    
    #Create the Target Matrix
    ##Initialize an Empty List##
    y = # Fill In
    
    ##Loop to Create Binary Target List Dependent on Attribute##

    
    #Convert the Feature and Target Lists to Numpy Arrays


    #Training and testing split, with 25% of the data reserved as the test set
    [X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size=0.25, random_state=101)

    #Reshaping the Matrices
    X_train = X_train.reshape(-1,1)
    X_test = X_test.reshape(-1,1)

    #Define the range of Regularization Parameters
    regularizationParameters = np.logspace(-8, 1, num=100)  # test case: 10^-8 to 10^1 with 100 number points
    # change this to the values in step 5 when submitting.

    #Initialize Lists
    MODEL = []
    accuracyScores = []
    
    #Train the Regression Models and Find Accuracy Scores with Regularization Parameters
    MODEL, accuracyScores = allActions(regularizationParameters, X_train, y_train, X_test, y_test)

    #Plot the accuracyScores as a function of regularizationParameters
    plt.xscale('log')
    plt.plot(regularizationParameters,accuracyScores)
    plt.xlabel('Regularization Parameter C', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.show()

    #Find best value of lmbda in terms of MSE
    ind = accuracyScores.index(max(accuracyScores))
    [regParam_best,accuracy_best,model_best] = [regularizationParameters[ind],accuracyScores[ind],MODEL[ind]]

    print('Best C tested is ' + str(regParam_best) + ', which yields an Accuracy Score of ' + str(accuracy_best))

    return model_best

#Function that cleans the dataset.
#Input: the original data, originalData
#Output: precipData, a list of the Precipitation dataset without (S) or T
#        totalData, a list of the Total dataset without commas
def clean_data(originalData):
    
    # Extract the columns of data needed

    
    # Clean the data as described in the README

    
    return precipData, totalData

#Function that trains a Logistic Regression model on the input dataset with C = regParam.
#Input: Feature matrix X, target variable vector y, regularization parameter regParam.
#Output: model_log, a numpy object containing the trained Logistic Regression model.
def train_model_log(X,y,regParam):

    # Fill In 

    return model_log

#Function that trains a Support Vector Machines model on the input dataset with C = regParam and kernel = 'rbf'.
#Input: Feature matrix X, target variable vector y, regularization parameter regParam.
#Output: model_svm, a numpy object containing the trained SVM model.
def train_model_svm(X,y,regParam):

    # Fill In

    return model_svm

#Function that calculates the accuracy score of the model on the input dataset.
#Input: Feature matrix X, target variable vector y, numpy model object
#Output: accuracy, the accuracy score
def accuracyFunc(X,y,model):

    # Fill In
    
    return accuracy

#Function that for each regularization parameter, trains the models and finds the accuracy score of the model
#Input: List of regularization parameters regParams, training feature matrix X_train, training target matrix y_train,
#       testing feature matrix X_test, testing target matrix y_test
#Output: List of numpy models, MODEL and list of accuracy scores, accuracyScores 
def allActions(regParams, X_train, y_train, X_test, y_test):
    MODEL = []
    accuracyScores = []
    
    for C in regParams:
        #Train the regression model using a regularization parameter of C
        ###### Uncomment which model you want to train with ######


        #Evaluate the accuracy on the test set


        #Store the model and accuracy in lists for further processing


    return MODEL, accuracyScores

if __name__ == '__main__':
    model_best = main()
