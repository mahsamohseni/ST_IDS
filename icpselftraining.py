import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from conformalprediction import CP
import matplotlib.pyplot as plt
from results import *


class ICPSelftraining():

    def __init__(self,name,model,mondrian=False):
        self.model = model
        self.name = name
        self.mondrian = mondrian

    def __str__(self):
        return "Classifier: " + self.name + "\nParameters: " + str(self.model.get_params())

    def selection_index(self, cr, cf, selection_num):
        cc = [a * b for a, b in zip(cr, cf)] 

        selected_unlabled_idx = sorted(range(len(cc)), key=lambda i: cc[i])[-selection_num:]
        selected_unlabled_idx_final = selected_unlabled_idx
        
        print("sele",selected_unlabled_idx_final)
        print("cc",[cc[i] for i in selected_unlabled_idx_final])
        print("\tselected unlabeled: ", len(selected_unlabled_idx_final))
        return selected_unlabled_idx_final


    def selection_unlabeled(self, X_cal, y_cal, X_unlabeled, y_predicted, cr, cf, percent=0.05):

        cc = [a * b for a, b in zip(cr, cf)]
        greaterThanFive = [i for i, j in enumerate(cc) if (j>0.5)]
        selection_num = (int(X_unlabeled.shape[0] * percent))
        
        if len(greaterThanFive) > 0:
            selection = True
            if (selection_num > len(greaterThanFive)):
                selected_idx = greaterThanFive
            else:
                selected_idx = random.sample(greaterThanFive,selection_num)
            

            print("\tselected unlabeled: ", len(selected_idx))
            for idx in selected_idx:
                unl = X_unlabeled[idx]
                X_cal = np.append(X_cal, [unl], axis=0)
                y_cal = np.append(y_cal, [y_predicted[idx]], axis=0)

            X_unlabeled = np.delete(X_unlabeled, selected_idx, axis=0)

        else:
            selection = False

        y_cal = pd.Series(y_cal)
        return X_cal, y_cal, X_unlabeled, selection

    def concatenation(self,X1,X2,y1,y2,split=False):
        X = np.vstack((X1,X2))
        y = pd.concat([y1,y2],ignore_index=True)
        if split == True:
            X1,X2,y1,y2 = train_test_split(X,y,stratify =y,test_size=0.3,random_state=42)
            return X1,X2,y1,y2
        else:
            return X,y


    def training(self,X_labaled,X_unlabeled,y_labeled,y_unlabeled):
        self.X_proper, self.X_cal, self.y_proper, self.y_cal = train_test_split(X_labaled,y_labeled,stratify = y_labeled,
                                                                                 test_size=0.3,random_state=42)
        self.model.fit(self.X_proper,self.y_proper)
        
        selection = True
        iteration = 0
        max_iteration = 10
        classes = np.unique(y_labeled)
        
        while selection and iteration < max_iteration:
            print("iteration:", iteration)
            alpha_cal = self.model.ncm_measure(self.X_cal, self.y_cal)
            pvalues = pd.DataFrame(columns = classes)
            cr = []
            cf = []
            predicted_label = []
            for i, x in enumerate(X_unlabeled):
                for label in np.unique(y_labeled):
                    alpha_un = self.model.ncm_measure([x], [label])
                    pvalue = self.model.compute_p_value(alpha_un, alpha_cal, self.y_cal, label, self.mondrian)
                    pvalues.loc[i, label] = pvalue
                # end for
                cr.append(pvalues.loc[i].max())  # cr = p_1st
                pre_label = pvalues.loc[i].astype('float64')
                pre = pre_label.idxmax()
                predicted_label.append(pre)
                cf.append(1 - (pvalues.loc[i].min()))  # cf = 1 - p_2nd
            # end for
            self.X_cal, self.y_cal, X_unlabeled, selection = self.selection_unlabeled(self.X_cal, self.y_cal, X_unlabeled,
                                                                              predicted_label, cr, cf, 0.05)
            if self.X_cal.shape[0] >= self.X_proper.shape[0]:
                print("concate and split data")
                self.X_proper,self.X_cal,self.y_proper,self.y_cal = self.concatenation(self.X_proper,self.X_cal,
                                                                                  self.y_proper,self.y_cal,split=True)
                self.model.fit(self.X_proper,self.y_proper)

            iteration += 1


    def score(self, X, y):
        X_final,y_final = self.concatenation(self.X_proper,self.X_cal,
                                             self.y_proper,self.y_cal,split=False)
        self.model.fit(X_final, y_final)
        y_predict,y_predict_prob = self.model.predict(X,prob=True)
        probs = y_predict_prob[:, 1]        
        res = Metrics(self.name, y, y_predict, y_predict_prob, minority_label=1)
        scores, cnf_matrix = res.result()
        

        scores, cnf_matrix = res.result()
        return y_predict_prob,y,y_predict, scores, cnf_matrix
