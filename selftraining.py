import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from conformalpredictionn import CP
from results import *



class CPSelfTraining():
    
    def __init__(self,name, base_classifier,mondrian=False):
        self.model = base_classifier
        self.name = name
        self.mondrian = mondrian
        
    def __str__(self):
        return "Classifier: " + self.name + "\nParameters: " + str(self.model.get_params())

    def selection_index(self, cr, cf, selection_num):
        cc = [a * b for a, b in zip(cr, cf)]
        
        selected_unlabled_idx = sorted(range(len(cc)), key=lambda i: cc[i])[-selection_num:]
        print("\tselected unlabeled: ",len(selected_unlabled_idx))
        return selected_unlabled_idx
    
    
    def selection_unlabeled(self, X_labeled, y_labeled, X_unlabeled, y_predicted, cr, cf, percent=0.05):
        
        selection_num = (int(X_unlabeled.shape[0] * percent))
        
        if selection_num > 0:
            selection = True
            selected_idx = self.selection_index(cr, cf, selection_num)
            for idx in selected_idx:
                unl = X_unlabeled[idx]
                X_labeled = np.append(X_labeled,[unl],axis=0)
                y_labeled = np.append(y_labeled,[y_predicted[idx]],axis=0)
            
            X_unlabeled = np.delete(X_unlabeled, selected_idx,axis=0)
            
        else:
            selection = False
        
        y_labeled = pd.Series(y_labeled)
        return X_labeled, y_labeled, X_unlabeled , selection




    def training(self, X_lab, X_unlab,y_lab , y_unlab):

        self.max_iterations = 10
        iteration = 0
        selection = True
        self.X_l = X_lab
        self.y_l = y_lab
        classes = np.unique(y_lab)
        
        while selection and (iteration < self.max_iterations):
            print("\niteration:",iteration)
            cr = []
            cf = []
            predicted_label = []
            pvalues = pd.DataFrame(columns=classes)

            self.model.fit(self.X_l, self.y_l)
            alpha_l = self.model.ncm_measure(self.X_l, self.y_l)

            for i, x in enumerate(X_unlab):
                for label in np.unique(y_lab):
                    alpha_un = self.model.ncm_measure([x], [label])
                    pvalue = self.model.compute_p_value(alpha_un, alpha_l, self.y_l, label,self.mondrian)
                    pvalues.loc[i, label] = pvalue
                #end for
                cr.append(pvalues.loc[i].max())  # cr = p_1st
                pre_label = pvalues.loc[i].astype('float64')
                pre = pre_label.idxmax()
                predicted_label.append(pre)
                cf.append(1 - (pvalues.loc[i].min()))  # cf = 1 - p_2nd
            #end for
            self.X_l, self.y_l, X_unlab, selection = self.selection_unlabeled(self.X_l, self.y_l, X_unlab, predicted_label, cr, cf, 0.05)   
            iteration += 1
            
            
    def score(self, X,y):
        self.model.fit(self.X_l,self.y_l)
        y_predict,y_predict_prob = self.model.predict(X,prob=True)
        probs = y_predict_prob[:, 1]
        
        res = Metrics(self.name, y, y_predict, y_predict_prob, minority_label=1)
        scores, cnf_matrix = res.result()
        
        y_score = self.model.decision(X)
        scores, cnf_matrix = res.result()
        return y_predict_prob,y,y_predict, scores, cnf_matrix


        
        
        


