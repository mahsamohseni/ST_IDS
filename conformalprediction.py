import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter



class CP:
    def __init__(self, base_classifier):
        self.base_classifier = base_classifier

    
    def compute_p_value(self,ncm_unlabeled, ncm_labeled, y_labeled, label, mondrian):
        count = 0
        if mondrian:
            for i, ncm in enumerate(ncm_labeled):
                if ncm >= ncm_unlabeled[0] and y_labeled.iloc[i] == label:
                    count += 1
            total = Counter(y_labeled)[label]
        else:
            for i, ncm in enumerate(ncm_labeled):
                if ncm >= ncm_unlabeled[0]:
                    count += 1
            total = len(ncm_labeled)
         
        pvalue = count / (total + 1)
        return pvalue
    

class SVM_model(CP):
    def __init__(self, base_classifier):
        self.clf = base_classifier

    def fit(self,x, y):
        self.clf.fit(x, y)

    def ncm_measure(self,x, y):
        alpha = []
        for i, y_ in enumerate(y):

            alpha.append(-(y_ * (self.clf.decision_function(x)[i])))  # ncm = -y * d(x)

        return alpha
    
    def predict(self,x,prob=True):
        y_predict = self.clf.predict(x)
        if prob==True:
            y_predict_prob = self.clf.predict_proba(x)
            return y_predict, y_predict_prob
        return y_predict
    
    def decision(self,x):
        return self.clf.decision_function(x)
    
    
class KNN_model(CP):
    def __init__(self, base_classifier):
        self.clf = base_classifier

    def fit(self,x, y):
        self.clf.fit(x, y)
        print("fit",x.shape,y.shape)
        self.n_samples = x.shape[0]
        self.y = y
    def ncm_measure(self,x, y):
        alpha = []
        for i, y_ in enumerate(y):
            
            closest_distances, indices = self.clf.kneighbors(x,n_neighbors=10,return_distance=True)

            count1 = 0
            count_1 = 0
            dist_same = []
            indexs_same = []
            dist_diff = []
            indexs_diff = []
            
            for i,indx in enumerate(indices[0]):
                if self.y.index[indx] == y_:
                    dist_same.append(closest_distances[0][i])
                    indexs_same.append(indx)
                    count1 += 1
                    if count1 == 3:
                        break
    
            for i,indx in enumerate(indices[0]):
                if self.y.index[indx] == -y_:
                    dist_diff.append(closest_distances[0][i])
                    indexs_diff.append(indx)
                    count_1 += 1
                    if count_1 == 3:
                        break
            if(sum(dist_diff)==0):
                alpha.append(0)
            else:
                alpha.append(sum(dist_same)/sum(dist_diff))          
#             alpha.append(-(y_ * (self.clf.decision_function(x)[i])))  # ncm = -y * d(x)
        
        return alpha
    
    def predict(self,x,prob=True):
        y_predict = self.clf.predict(x)
        if prob==True:
            y_predict_prob = self.clf.predict_proba(x)
            return y_predict, y_predict_prob
        return y_predict
