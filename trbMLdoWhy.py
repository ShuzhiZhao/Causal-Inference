import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_recall_fscore_support
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.inspection import DecisionBoundaryDisplay
import dowhy
from dowhy.causal_model import CausalModel
from IPython.display import Image,display

def loadata(data_dir,Met):
    # load data from N1P2.csc
    data = pd.read_csv(data_dir,dtype=object)
    label = data.loc[:,['TimeSerials']].groupby(by='TimeSerials').count()
    label = label.drop(['N1[200:400]','P2[400:600]'],axis=0)
    subNames = data.loc[:,['Name']].groupby(by='Name').count()
    print('++++++++ using the metrics of '+Met+' ++++++++++++++')
    df = pd.DataFrame()
    for ii in range(149):
        df.insert(loc=len(df.columns),column=ii,value=[])
    for name in subNames.index:
        temp = []
        if 'aIFG' in name:
            temp.append(0)
        elif 'pIFG' in name:
            temp.append(1)
        elif 'sham' in name:
            temp.append(2)
        for index in label.index:
            if 'N1[200:400]' in index or 'P2[400:600]' in index :
                print('skip ...')
            else: 
                temp1 = data[(data.Name==name)&(data.TimeSerials==index)][Met].astype(float).values.tolist()[0]
                temp.append(temp1)
    #             SC = data.loc[data['TimeSerials']==index,[Met]]
    #             df.insert(loc=len(df.columns),column=index,value=SC[Met].astype(float).values.tolist())
        df = pd.DataFrame(np.insert(df.values,len(df.index),values=temp,axis=0))
        columns = ['label']
    for index in label.index:
        columns.append(index)
    df.columns = columns
#     df_norm = (df-df.min())/(df.max()-df.min()) 
    df_norm = (df-df.mean())/(df.std())
    print(df,'\n')
    X = df_norm.drop(['label'],axis=1)
    y = df['label']
    return X,y,df_norm

def Classfiers(X,y,Met):
    names = [
        "Nearest Neighbors",
        "Linear SVM",
        "RBF SVM",
        "Gaussian Process",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        "QDA",
    ]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
    ]    
    # iterate over classifiers
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0,shuffle=True)
    for name, clf in zip(names, classifiers):
        print('+++++++++++++++ using classfier of '+name +' ++++++++++++++++++++++')
        clf.fit(X_train, y_train)
        ## predict
        pred_train = clf.predict(X_train)
        pred_test = clf.predict(X_test)
        ## ACC
        train_acc = accuracy_score(y_train,pred_train)
        test_acc = accuracy_score(y_test,pred_test)
        print(" train acc: {0:.2f}   test acc: {1:.2f}".format(train_acc,test_acc))
        # metrics
        precision,recall,F1,_ = precision_recall_fscore_support(y_test,pred_test,average='weighted')
        print(" precision:{0:.2f} recall:{1:2f} F1:{2:.2f}".format(precision,recall,F1)) 
        print('+++++++++++++++ Ending '+name +' ++++++++++++++++++++++')
    
def Donation(X,y,Met):    
        names = [
            "Decision Tree",
            "AdaBoost",
        ]

        classifiers = [
            DecisionTreeClassifier(max_depth=5),
            AdaBoostClassifier(),
        ]    
        # iterate over classifiers
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0,shuffle=True)
        for name, clf in zip(names, classifiers):
            print('+++++++++++++++ using classfier of '+name +' ++++++++++++++++++++++')
            clf.fit(X_train, y_train)
            ## predict
            pred_train = clf.predict(X_train)
            pred_test = clf.predict(X_test)
            ## ACC
            train_acc = accuracy_score(y_train,pred_train)
            test_acc = accuracy_score(y_test,pred_test)
            print(name+" train acc: {0:.2f}   test acc: {1:.2f}".format(train_acc,test_acc))
            # metrics
            precision,recall,F1,_ = precision_recall_fscore_support(y_test,pred_test,average='weighted')
            print(name+" precision:{0:.2f} recall:{1:2f} F1:{2:.2f}".format(precision,recall,F1))
            # feature donation
            features = list(X_test.columns)
            importances = clf.feature_importances_
            indices = np.argsort(importances)[::-1]
            num_features = len(importances)
            f = open('/media/lhj/Momery/causalML/IFG_Source_Causal/data/Importances/'+name+Met+'_importance.txt','w')
            for i in indices:
                if importances[i] > 0:
                    f.write("{0} - {1:.3f} \n".format(features[i],importances[i]))
                    print("{0} - {1:.3f}".format(features[i],importances[i]))  
            print('+++++++++++++++ Ending '+name +' ++++++++++++++++++++++')
            
def CausalInference(df):
    ## causal inference
    data_mpg = df.drop(['label'],axis=1)
    model = CausalModel(
                    data = data_mpg,
                    treatment='G_temp_sup-Plan_polar L',
                    outcome='S_front_sup L',
                    common_causes=data_mpg.columns.drop(['G_temp_sup-Plan_polar L','S_front_sup L']).tolist(),
                    )
    model.view_model()
    display(Image(filename="causal_model.png"))
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    print(identified_estimand)
    estimate = model.estimate_effect(
                                    identified_estimand,
                                    method_name="backdoor.linear_regression",
                                    control_value=0,
                                    treatment_value=1,
                                    confidence_intervals=True,
                                    test_significance=True
                                    )
    print("Causal Estimate is " + str(estimate.value))
    res_random = model.refute_estimate(identified_estimand, estimate, method_name="random_common_cause")
    print(res_random)

def Main():
    data_dir = '/media/lhj/Momery/causalML/IFG_Source_Causal/data/N1P2.csv'
    Metrics = ['Mean','std','median','Max','Power']
    for Met in Metrics :
        X,y,df = loadata(data_dir,Met)
        Classfiers(X,y,Met)
        Donation(X,y,Met)

Main()