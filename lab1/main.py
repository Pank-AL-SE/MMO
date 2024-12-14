from math import sqrt
import numpy as np
from sklearn.model_selection import train_test_split

def rectangleCore(r):
    ans = True
    if abs(r) <= 1:
        ans = True
    else:
        ans = False
    return ans

def dist(x, y):
    return sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)

def parzenWindow(X, y, h) -> float:
    check_flag = 0
    for i in range(len(X)):
        zero = 0
        one = 0
        for j in range(len(X)):
            if i == j:
                continue
            if rectangleCore(dist(X[i], X[j])/h):
                if y[j] == 1:
                    one += 1
                else:
                    zero += 1
        if (one > zero) :
            if (y[i] == 1):
                check_flag+=1
        else:
            if (y[i] == 0):
                check_flag+=1
    print(check_flag, len(X), check_flag/len(X))
    return check_flag/len(X)



data = np.genfromtxt("data4.csv", delimiter=',', skip_header=True)
X = data[:, :-1]
y = data[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size=0.67, 
                                                    random_state=42)
print(len(X_train))
print(len(X_test))



maxP = -1.0
h = 0
for i in range(1, 11, 1):
    temp = parzenWindow(X_train, y_train, i)
    if temp > maxP:
        maxP = temp
        h = i
print (h, maxP)
parzenWindow(X_test, y_test, h)