import numpy as np

class one_v_all:

    def __init__(self, k):
        self.label = k
    
    def Accuracy(self,y_test, y_pred):
        acc = 0

        for i in range(len(y_test)):
            act_label = 1 if (y_test[i] == self.label) else 0
            pred_label = 1 if (y_pred[i] == self.label) else 0

            acc += 1 if (act_label == pred_label) else 0
        
        return acc / len(y_test)

    def Precision(self,y_test, y_pred):
        numerator = 0
        denominator = 0

        for i in range(len(y_test)):
            act_label = 1 if (y_test[i] == self.label) else 0
            pred_label = 1 if (y_pred[i] == self.label) else 0

            numerator += 1 if(act_label==pred_label==1) else 0
            denominator +=1 if(pred_label) else 0
        
        return numerator / denominator

    def Recall(self,y_test, y_pred):
        numerator = 0
        denominator = 0

        for i in range(len(y_test)):
            act_label = 1 if (y_test[i] == self.label) else 0
            pred_label = 1 if (y_pred[i] == self.label) else 0

            numerator += 1 if(act_label==pred_label==1) else 0
            denominator +=1 if(pred_label) else 0
        
        return numerator / denominator

    def F1score(self,y_test, y_pred):
        pres = self.Precision(y_test, y_pred)
        rec = self.Recall(y_test, y_pred)

        return (2*pres*rec)/(pres+rec)