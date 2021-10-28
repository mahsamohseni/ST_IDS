import numpy as np
import pandas as pd



class PreProcessing():
    def __init__(self,labeled_ratio = 0.3,imbalanced_ratio=10):
        self.ir = imbalanced_ratio
        self.labeled_ratio = labeled_ratio
    
    
    # tedade monasebe maj va minor haro ba ir va n(maj+minor) mide
    def calculate_maj_min(self,num, ir):     
        majority = int(num * ir / (ir + 1))
        minority = int(num / (ir + 1))
        return majority, minority

    #train o teste ghadimi ro ba tedade monasebe maj va minor be train va test jadid tabdil mikone
    def modify_ir(self,trainOld,testOld,numTrain,numTest):
        
        numMajorTrain, numMinorTrain = self.calculate_maj_min(numTrain, self.ir)
        numMajorTest, numMinorTest = self.calculate_maj_min(numTest, self.ir)

        majorTrain = trainOld[trainOld.Label == -1].sample(n=numMajorTrain, random_state=42)
        majorTest = testOld[testOld.Label == -1].sample(n=numMajorTest, random_state=42)
        minorTrain = trainOld[trainOld.Label == 1].sample(n=numMinorTrain, random_state=42)
        minorTest = testOld[testOld.Label == 1].sample(n=numMinorTest, random_state=42)

        Train = pd.concat([majorTrain, minorTrain]).sample(frac=1).reset_index(drop=True) # sample(frac=1) ye df be hamun andaze
                                                                                          #barmigardune ba index haye random(jaygasht)
                                                                                           # nemikhastam -0 ha poshte sare ham va 
                                                                                           # be donbalesh 1 ha bashand
        Test = pd.concat([majorTest, minorTest]).sample(frac=1).reset_index(drop=True)    # dar akhar ham hame satr ha ro be tartib index
                                                                                           #zadam ba reset_index
        return  Train,Test
    
    
    def split(self,Train,Test):
        from sklearn.preprocessing import Normalizer
        from sklearn.model_selection import train_test_split
        scale = Normalizer()

        X_train = Train.iloc[:, 0:-1]
        y_train = Train.iloc[:,-1]
        X_test = Test.iloc[:,0:-1]
        y_test = Test.iloc[:,-1]

        X_train = scale.fit_transform(X_train)
        X_test = scale.transform(X_test)
        
        X_unlabeled, X_labeled, y_unlabeled, y_labeled = train_test_split(X_train, y_train, 
                                                                          test_size=self.labeled_ratio,random_state=42)
        return X_unlabeled, X_labeled, y_unlabeled, y_labeled,X_train,y_train, X_test, y_test

