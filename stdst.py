import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from results import *


class StandardSelftraining():
    def __init__(self, name, base_classifier):
        self.name = name
        self.model = base_classifier

    def __str__(self):
        return "Classifier: " + self.name + "\nParameters: " + str(self.model.get_params())
    
        
        
    def training(self,X_lab, X_unlab, y_lab,y_unlab):  # x is X_train and y is y_train
        self.X_l = X_lab
        self.y_l = y_lab
        y_lab[y_lab == -1] = 0
        x = np.concatenate((X_lab,X_unlab))
        y = np.concatenate((y_lab.astype(str), np.full_like(y_unlab.astype(str), "unlabeled")))
        #print(Counter(y))
        y = np.copy(y)  # copy in order not to change original data

        all_labeled = False
        iteration = 0
        max_iterations = 10
        # Iterate until the result is stable or max_iterations is reached
        while not all_labeled and (iteration < max_iterations):
            print("\niteration:",iteration)
            self._fit_iteration(x, y)
            all_labeled = (y != "unlabeled").all()
            iteration += 1
            
        labeled = y != "unlabeled"
        self.X_l = x[labeled]
        self.y_l = y[labeled]
        
        #print("final labeled: ",self.X_l.shape[0],"\t",Counter(self.y_l))

    def _fit_iteration(self, X, y):
        threshold = 0.9

        clf = self.model
        # Fit a classifier on already labeled data
        labeled = y != "unlabeled"
        
        clf.fit(X[labeled], y[labeled])

        probabilities = clf.predict_proba(X)
        prob_max = probabilities.max(axis=1)
        yseri = pd.Series(y)
        unidx = yseri[yseri=="unlabeled"].index
        prlist = []
        for idx in unidx:
            prlist.append(prob_max[idx])
        selection_num = (int(Counter(y)["unlabeled"] * 0.05))
        idmax = sorted(range(len(prlist)),key=lambda i:prlist[i])[-selection_num:]
        for idx in idmax:
            y[[idx]] = clf.predict(X[[idx]])
        
       
        print("num of selected unlabeled:",len(idmax))
       
    def predict(self, X,prob=False):
        if prob == True:
            return self.model.predict(X), self.model.predict_proba(X)
        return self.model.predict(X)

    def score(self, X, y):
        
        y_copy = y.copy(deep=True)
        
        y_zero_one = y_copy.replace(-1,0)
        self.model.fit(self.X_l, self.y_l)
        y_predict_zeroone,y_predict_prob = self.predict(X,prob=True)

        probs = y_predict_prob[:, 1]

        y_predict_zeroone[y_predict_zeroone == '0'] = -1
        y_predict_zeroone[y_predict_zeroone == '1'] = 1
        y_predict_zeroone = y_predict_zeroone.astype('int32')

        y_baraye_metric = y_copy.replace(0,-1)

        res = Metrics(self.name, y_baraye_metric, y_predict_zeroone, y_predict_prob, minority_label=1)

        scores, cnf_matrix = res.result()

        return y_predict_prob,y_zero_one,y_predict_zeroone, scores, cnf_matrix
