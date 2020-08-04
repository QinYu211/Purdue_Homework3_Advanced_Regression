# Homework 3: Logistic Regression and Support Vector Machines

This homework covers Logistic Regression as well as Support Vector Machines and will give you practice with `numpy` and `sklearn` libraries in Python.

## Goals

In this homework you will:
  1. Build logistic regression models to serve as predictors from input data.
  2. Parse input data into feature matrices and target variables.
  3. Find the best regularization parameter for a dataset.
  4. Formulate a conclusion from analyzing the dataset.

## Background

Before attempting the homework, please review the notes on Logistic Regression. In addition to what is covered there, the following background may be useful:

### CSV Processing in Python

Like .txt, .csv (comma-separated values) is a useful file format for storing data. In a CSV file, each line is a data record, and different fields of the record are separated by commas, making them two-dimensional data tables (i.e. records by fields). Typically, the first row and first column are headings for the fields and records.
Python's `pandas` module helps manage two-dimensional data tables. We can read a CSV as follows:
```
import pandas as pd
data = pd.read_csv('data.csv')
```
To see a small snippet of the data, including the headers, we can write `data.head()`. Once we know which columns we want to use as features (say 'A','B','D') and which to use as a target variable (say 'C'), we can build our feature matrix and target vector by referencing the header:
```
X = data[['A', 'B', 'D']]
y = data[['C']]
```
More details on `pandas.read_csv()` can be found [here](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html).

### Logistic Regression in Python

Python offers several standard machine learning models with optimized implementations in the `sklearn` library. Suppose we have a feature matrix `X` and a target variable vector `y`. Logistic Regression has a single regularization parameter `C` (like the Ridge Regression parameter `λ`). To train a standard logistic regression model with `C = 0.2`, for instance, we can write:
```
from sklearn.linear_model import LogisticRegression
model_log = LogisticRegression(C = 0.2)
model_log.fit(X, y)
```
Then, if we have a feature matrix `Xn` of new samples, we can predict the target variables (if we know the model is performing well) by applying the trained model:
```
yn = model_log.predict(Xn)
```

More regression models in Python can be found [here](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning.).

### Support Vector Machines (SVM) and Support Vector Classification (SVC) in Python

Python offers several standard machine learning models with optimized implementations in the `sklearn` library. Suppose we have a feature matrix `X` and a target variable vector `y`. SVC is one model within SVM. SVC has the same regularization parameter `C`  as Logistic Regression. SVC also has a kernel parameter. To train and predict a standard SVC model with `C = 0.2` and with the kernel being `rbf`, for instance, we can write:
```
from sklearn import svm
model_svm = svm.SVC(C = 0.2, kernel = 'rbf')
model_svm.fit(X, y)
```
Then, if we have a feature matrix `Xn` of new samples, we can predict the target variables (if we know the model is performing well) by applying the trained model:
```
yn = model_svm.predict(Xn)
```

More information on sklearn.svm.SVC module can be found [here](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC).

## Instructions

### Setting Up Your Repository
Click the link on Brightspace to set up your repository for Homework 3, then clone it.
Aside from this README, the repository should contain the following files:
  1. `hw3_1.py` - A starter file with instructions, functions, and a skeleton that you will fill out in Problem 1.
  2. `bike-data.csv` - A data file for Problem 1.
  3. `problem1_writeup_sample.pdf` - A sample writeup for Problem 1.

### Problem 1: Logistic Regression and Support Vector Machines
In this problem, you will complete the starter code `hw3_1.py` that selects the best regularization parameter `C` for a predictor on the given `bike-data.csv` dataset, which gives information on bike traffic across four bridges in New York City. This dataset contains the following: Date, Day, High Temp, Low Temp, Precipitation, bike counts on Brooklyn Bridge, Manhattan Bridge, Williamsburg Bridge, Queensboro Bridge, and Total. From the input data, you will train a Logistic Regression and an SVM Model for different values of `C`, find the best value, and then use the result to make a conclusion, about the dataset, for the following question:
**Can you use this data to predict whether it is raining based on the number of bicyclists on the bridges?** 
More specifically:

  1. Complete the function `clean_data` that cleans the csv document so you can easily analyze the data. This function will need to clean the following values in the corresponding columns:
  
      Precipitation: '0.47(S)' to 0.47 and 'T' to 0.0

      Total: '11,497' to 11497.0

      Notice that the values are no longer strings, they are floats.
  
      Hint: You can use the .replace() function when wanting to change the value of a string. For example, we can write
      ```
      randomString = 'hi*'
      randomString = randomString.replace('*','')
      print(randomString) ## Prints the string 'hi'
      ```
  
  2. Complete the section in `main` that creates the feature matrix, with the attribute Total.
  
  3. Complete the section in `main` that creates the target matrix, with the attribute Precipitation. The data needed to use the Logistic Regression Model and SVC is binary. When Precipitation is greater than or equal to 0.01, it is raining.
  
  4. Complete the section in `main` that converts the Feature and Target Lists to `numpy` Arrays.
  
   5. Define the range of regularization parameters to test. The range should be from _10<sup> -10_ to _10<sup> 4_ with 500 numbers, using the `np.logspace` function. Please refer to [this link](https://numpy.org/doc/stable/reference/generated/numpy.logspace.html) for how to use np.logspace.
      
  6. Complete the function `train_model_log` to fit a Logistic Regression model with regularization parameter `C` on the training dataset. You may use the `LogisticRegression` class in `sklearn` to do this. Note that the partition of the training and testing set has already been done for you in the `main` function.
  
  7. Complete the function `train_model_svm` to use the Support Vector Classification method with regularization parameter `C` and kernel `rbf` on the training dataset. You may use the `svm.SVC` class in `sklearn` to do this. Note that the partition of the training and testing set has already been done for you in the `main` function.
  
  8. Complete the function `accuracyFunc` to find the accuracy of the predicted model when compared to the target variable vector. 
  
  9. Complete the function `allActions` to train the models and find the accuracy scores for each regularization parameter. 
  
  10. Add the two output messages and two plots to your writeup (for the Logistic Regression Model and SVM). Clearly correspond each model to the related output.
  
      Once you have completed `hw3_1.py`, to test your code, set `regularizationParameters` to a range from _10<sup> -8_ to _10<sup> 1_ with 100 numbers, using the `np.logspace` function. Your output messages should be:
      
      ```
      Logistic Regression: 'Best C tested is 1.519911082952933e-07, which yields an Accuracy Score of 0.7962962962962963'
      
      SVM: 'Best C tested is 6.5793322465756825, which yields an Accuracy Score of 0.7962962962962963'
      ```
      (There could be minor differences due to rounding, you won't get points off because of this.)
      After your results for this test case match this readme and the sample write up, set `regularizationParameters` back to the values stated in step 5 when writing your write up and submitting. 
      
  11. Make a conclusion to the proposed question based on your analysis of your output. Explain your conclusion in your writeup.

## What to Submit
For each problem, you must submit:
  1. Your completed version of the starter code.
  2. A writeup as a separate PDF document named `problem1_writeup.pdf`. A sample writeup are available in your GitHub.

## Submitting Your Code
Push your completed `hw3_1.py` and `problem1_writeup.pdf` to your repository before the deadline.
