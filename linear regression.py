from Model import model
import numpy as np
import pandas as pd
from statistics import mean
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('iris.data',names =['a','b','c','d','type'])
rows, cols = df.shape 
df = df.sample(frac=1).reset_index(drop=True) #to shuffle dataframe in-place and reset the index
X = df.drop('type', axis= 1)# Data Visualization
Y = df['type']
X = (X - X.mean()) / (X.max() - X.min())
classList = df['type'].unique()
uniquevalues = dict ( zip( classList, list(range(1,len(classList)+1))) )
Y = [uniquevalues[item] for item in Y]
target_df = pd.DataFrame(Y) # Converting Objects to Numerical dtype
def kFLR(kFolds, learningRate): # Splitting the Dataset
    X_split = np.split(X, kFolds)
    Y_split = np.split(target_df, kFolds)
    X_test = []
    X_train = []
  # Instantiating LinearRegression() Model  
    for i in range(len(X_split)):
        X_intermediateTrain = []
        for j in range(len(X_split)):
            if i==j:
                X_test.append(X_split[j])
            else:
                X_intermediateTrain.append(X_split[j])
      # Training/Testing the Model          
        X_train.append(X_intermediateTrain)
    X_trainSet = []
    for i in X_train:
        X_trainSet.append(np.matrix(pd.concat(i)))
    Y_test = []
    Y_train = []
    for i in range(len(Y_split)):
        Y_intermediateTrain = []
        for j in range(len(Y_split)):
            if i==j:
                Y_test.append(Y_split[j])
            else:
                Y_intermediateTrain.append(Y_split[j])
        
        Y_train.append(Y_intermediateTrain)
    Y_trainSet = []
    for i in Y_train:
        Y_trainSet.append(np.matrix(pd.concat(i)))
        
    MeanAbsoluteError=[]
    for i in range(kFolds):    
        X_trainSet[i] = np.array(X_trainSet[i]).T
        Y_trainSet[i] = np.array(Y_trainSet[i]).T
        X_test[i] = np.array(X_test[i]).T
        Y_test[i] = np.array(Y_test[i]).T
        MeanAbsoluteError.append(model(X_trainSet[i], Y_trainSet[i], X_test[i], Y_test[i], learningRate, 150))
    
    totalSum = 0
    for i in MeanAbsoluteError:
        totalSum = totalSum + i
    
    MeanAbsoluteError = totalSum / kFolds
    acc = round((100 - MeanAbsoluteError), 2)
    return acc
# Evaluating Model's Performance
kFolds = int(input("Enter Kfolds (6 Or 10 Or 15): "))
learningRate = float(input("Enter learning rate (0.04 Or 0.05 Or 0.10): "))
acc = kFLR(kFolds, learningRate)
print ("Accuracy : " + str(acc) + "%")
