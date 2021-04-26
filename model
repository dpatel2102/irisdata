import numpy as np
import matplotlib.pyplot as plt

def __init__(length):
    w = np.zeros((1,length))
    b = 0
    return w,b    

def cf(y, z):
    m = z.shape[1]
    n = (1 / (2 * m)) * np.sum(np.square(y - z))
    return n

def linear_forward(a, b, x): #FROM X TO COST
    y = np.dot(b, a) + x
    return y    

def backpropagation(X, y, z):#TO FIND GRAD
    m = y.shape[1]
    dz = (1 / m) * (z - y)
    dw = np.dot(dz, X.T)
    db = np.sum(dz)
    return dw, db

def grad(w, b, dw, db, learning_rate): # update rule
    w = w - learning_rate * dw
    b = b - learning_rate * db
    return w, b

def model(X_train, Y_train, X_test, Y_test, learning_rate, g):
    length = X_train.shape[0]
    w, b = __init__(length)
    coststraining = []
    mtraining = Y_train.shape[1]
    mval = Y_test.shape[1]
    MeanAbsoluteErrorval = 0

    for i in range (1,g + 1):
        ztraining =linear_forward (X_train, w, b)
        costtraining =cf(ztraining, Y_train)
        dw, db = backpropagation(X_train, Y_train, ztraining)
        w, b = grad(w, b, dw, db, learning_rate)
        
        if (i % 10 == 0):
            coststraining.append(costtraining)
        MeanAbsoluteErrortraining = 1 / (mtraining) * np.sum(np.abs(ztraining - Y_train))
        
        zval = linear_forward(X_test, w, b)
        costval = cf(zval, Y_test)
        MeanAbsoluteErrorval = (1/ (mval)) * np.sum(np.abs(zval - Y_test))
    
   
    return MeanAbsoluteErrorval
