import numpy as np

def Accuracy(y_test, y_pred):
    acc = 0

    for i in range(len(y_test)):
        acc += 1 if (y_test[i]==y_pred[i]) else 0
    
    return acc / len(y_test)

def Precision(y_test, y_pred):
    numerator = 0
    denominator = 0

    for i in range(len(y_test)):
        numerator += 1 if(y_test[i]==y_pred[i]==1) else 0
        denominator +=1 if(y_test[i]) else 0
    
    return numerator / denominator

def Recall(y_test, y_pred):
    numerator = 0
    denominator = 0

    for i in range(len(y_test)):
        numerator += 1 if(y_test[i]==y_pred[i]==1) else 0
        denominator +=1 if(y_pred[i]) else 0
    
    return numerator / denominator

def F1score(y_test, y_pred):
    pres = Precision(y_test, y_pred)
    rec = Recall(y_test, y_pred)

    return (2*pres*rec)/(pres+rec)