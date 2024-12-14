#LASSO

import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import *
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')

def findReg(data, iteration):
    x = preprocessing.normalize(data[:, :-1])
    y = data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.7, 
                                                    random_state=iteration,
                                                       stratify=y)
    lasso = LassoCV()
    
    lasso = Lasso()
    lasso.fit(X_train, y_train)

    y_pred = lasso.predict(X_test)
    res = 0
    for i in range(y_pred.size):
        if (abs(y_pred[i] - y_test[i]) < 1):
            res+=1
    #print("mean: ", mean_squared_error(y_test, y_pred))
    return([res / y_pred.size, mean_squared_error(y_test, y_pred)])        

def main():
    data = pd.read_csv('winequalityN.csv').fillna(0)
    data['type'].replace({'white' : 0, 'red' : 1}, inplace = True)
    data = data.values
    
    dataRed = data[data[:, 0] == 1]
    dataWhite = data[data[:, 0] == 0]
    allTemp = 0
    allSqrt = 0
    for i in range(1, 11):
        allList = findReg(data, i)
        #allTemp += findReg(data, i)[0]
        allTemp += allList[0]
        allSqrt += allList[1]
    print("All\tpred: ", allTemp / 10, "\tsqrt: ", allSqrt / 10)

    whiteTemp = 0
    whiteSqrt = 0
    for i in range(1, 11):
        whiteList = findReg(dataWhite, i)
        whiteTemp += whiteList[0]
        whiteSqrt += whiteList[1]
    print("White\tpred: ", whiteTemp / 10, "\tsqrt: ", whiteSqrt / 10)

    redTemp = 0
    redSqrt = 0
    for i in range(1, 11):
        redList = findReg(dataRed, i)
        redTemp += redList[0]
        redSqrt += redList[1]
    print("Red\tpred: ", redTemp / 10, "\tsqrt: ", redSqrt / 10)
    
if __name__ == "__main__":
    main()

#print(dataRed)
