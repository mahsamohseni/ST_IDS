import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, balanced_accuracy_score, f1_score,matthews_corrcoef
from imblearn.metrics import geometric_mean_score, sensitivity_score, specificity_score
from sklearn.metrics import roc_curve, roc_auc_score,precision_recall_curve,auc


class Metrics:
    def __init__(self, name, y_test, y_predict, y_predict_prob, minority_label=1):
        self.name = name
        self.y_predict = y_predict
        self.y_predict_prob = y_predict_prob
        self.probs = y_predict_prob[:, 1]
#         self.probs2 = y_predict_prob[:,0]
        self.y_test = y_test
        self.min_label = minority_label
        self.y_zero_one = y_test.replace(-1,0)


   
        
    def result(self):
        # Score the classifier
        acc = accuracy_score(self.y_test.astype(str), self.y_predict.astype(str))
#         pre = precision_score(self.y_test.astype(str), self.y_predict.astype(str), average='binary',
#                               pos_label=str(self.min_label))
        pre_none = precision_score(self.y_test.astype(str), self.y_predict.astype(str), average=None)
        pre_micro = precision_score(self.y_test.astype(str), self.y_predict.astype(str), average='micro')
        pre_macro = precision_score(self.y_test.astype(str), self.y_predict.astype(str), average='macro')
        
        geo_none = geometric_mean_score(self.y_test.astype(str), self.y_predict.astype(str), average=None)
        geo_micro = geometric_mean_score(self.y_test.astype(str), self.y_predict.astype(str), average='micro')
        geo_macro = geometric_mean_score(self.y_test.astype(str), self.y_predict.astype(str), average='macro')
        
        rec_none = recall_score(self.y_test.astype(str), self.y_predict.astype(str), average=None)
        rec_micro = recall_score(self.y_test.astype(str), self.y_predict.astype(str), average='micro')
        rec_macro = recall_score(self.y_test.astype(str), self.y_predict.astype(str), average='macro')
        
        spe_none = specificity_score(self.y_test.astype(str), self.y_predict.astype(str), average=None)
        spe_micro = specificity_score(self.y_test.astype(str), self.y_predict.astype(str), average='micro')
        spe_macro = specificity_score(self.y_test.astype(str), self.y_predict.astype(str), average='macro')
        
        
        f1_none = f1_score(self.y_test.astype(str), self.y_predict.astype(str), average=None)
        f1_micro = f1_score(self.y_test.astype(str), self.y_predict.astype(str), average='micro')
        f1_macro = f1_score(self.y_test.astype(str), self.y_predict.astype(str), average='macro')
        
        mmc = matthews_corrcoef(self.y_test.astype(str), self.y_predict.astype(str))
        
        bal_acc = balanced_accuracy_score(self.y_test.astype(str), self.y_predict.astype(str))
        auc_roc = roc_auc_score(self.y_test, self.probs)
        precision, recall, thresholds = precision_recall_curve(self.y_zero_one, self.probs)
        auc_pre_rec = auc(recall, precision)
        
        cnf_matrix = pd.DataFrame(confusion_matrix(self.y_test.astype(str), self.y_predict.astype(str)))

        print("#", end="")
        print()
        scores = {
            "accuracy": acc,

            "rec_non": rec_none,
            "rec_micro":rec_micro,
            "rec_macro":rec_macro,
            
            "spe_non": spe_none,
            "spe_micro":spe_micro,
            "spe_macro":spe_macro,
            
            "geo-non": geo_none,
            "geo_micro":geo_micro,
            "geo_macro":geo_macro,
            
            
            "f1-non": f1_none,
            "f1_micro":f1_micro,
            "f1_macro":f1_macro,
            
           
            "pre_non":pre_none,
            "pre_micro":pre_micro,
            "pre_macro":pre_macro,
            
            "mmc":mmc,
            "balanced_acc": bal_acc,
            "roc": auc_roc,
            "pre_rec": auc_pre_rec
        }
        return scores, cnf_matrix

