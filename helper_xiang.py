
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from IPython.display import clear_output
from sklearn.model_selection import KFold,train_test_split,StratifiedShuffleSplit,ShuffleSplit,cross_val_score,cross_validate,GridSearchCV, StratifiedKFold

from sklearn.metrics import r2_score

def test_func():
    print("test func executed")


#interpoliert [a,b] geometrisch mit n Werten
def rangee(a,b,n):
  if n==1:
    return [a]
  x=np.zeros(n)
  factor=(b/a)**(1/(n-1))
  x[0]=a
  for i in range(1,n):
    x[i]=x[i-1]*factor
  return x  
#wie oben, Array wird gerundet
def rangeeInt(a,b,n):
    if n==1:
        return [a]
    x=[0]*n
    factor=(b/a)**(1/(n-1))
    x[0]=a
    for i in range(1,n):
        x[i]=int((factor**i)*a)
    return x  

class CustomOneHotEncoder(OneHotEncoder):
        """
    
            OneHotEncodes selected columns of the DataFrame 

            :param columns_to_encode (list of strings): list of the names of the columns one wishes to one-hot-encode
            :param X: full DataFrame
        """
      
        def __init__(self,columns_to_encode,**kwargs): #args is not allowed for SKlearn Estimators
            self.columns_to_encode=columns_to_encode
            super().__init__(**kwargs)
        
        def fit(self,X,y=None):

            super().fit(X[self.columns_to_encode])

        def transform(self,X,y=None):
            """
                OneHotEncodes the selected columns of the DataFrame, adds the new one-hot-encoded features to it, and deletes the original feature
            """

            #stores the new features in an array
            array_new_features=super().transform(X[self.columns_to_encode]).toarray()

            #transforms array into DataFrame
            X_new_features=pd.DataFrame(array_new_features,columns=super().get_feature_names_out(),index=X.index)
            
            #adds the new feature columns to the original dataframe
            X_res=pd.concat([X,X_new_features],axis=1)
            #remove the old feature column
            X_res=X_res.drop(self.columns_to_encode,axis=1)
            
            return X_res


'''Parameter tuning with GridsearchCV (nntimes times), nplits-fold CV, outputFormat='list' or 'table', colIndex specifies which value should be in hte column'''
def GSCV(clf, paramSpace, X,y,train_size=0.99,nsplits=2,ntimes=1,outputFormat='list',verbose=0,colIndex=0,called_in_jupyter_notebook=False):
    
    #prep
    arg=(clf,paramSpace)
    #do Gridsearch
    scoreSum=[];stdSum=[];res=0
    for i in range(ntimes):
        X_mod,X_val,y_mod,y_val=train_test_split(X,y,train_size=train_size,random_state=None)
        kFold=StratifiedKFold(n_splits=nsplits,shuffle=True,random_state=None)
        gsCV=GridSearchCV(*arg,cv=kFold,scoring='balanced_accuracy',verbose=verbose,refit=False,n_jobs=-1)
        gsCV.fit(X_mod,y_mod)
        res=gsCV.cv_results_
        if i==0:
            scoreSum=res['mean_test_score']
            stdSum=res['std_test_score']##std does not work. calculates wrong thing
        else:
            scoreSum+=res['mean_test_score']
            stdSum+=res['std_test_score']
            
        res['mean_test_score']=scoreSum/(i+1)
        res['std_test_score']=stdSum/(i+1)


        #output    
        if called_in_jupyter_notebook:
            clear_output(wait=True)
        print("------------------ %d times %d-fold CV ------------------ "%(ntimes,nsplits))
        if outputFormat!='table':        
            for mean_score, std_score, params in zip(res["mean_test_score"],res["std_test_score"],res["params"]):
                mean_score=round(mean_score,4)
                print("After %d iterations: mean score %s with param %s"%(i+1,str(mean_score),str(params)))
                #print(str(mean_score)+ "  in CI " + str([round(mean_score-2*std_score,4),round(mean_score+2*std_score,4)]) +" " +str(params))
            #print("best parameters: " + str(gsCV.best_params_)+ "\n "+ ' with score ' + str(gsCV.best_score_))
        else:
            #parameter[colIndex] is shown in the columns, the others are shown in the rows, k=0 default
            print("After %d iterations:" %(i+1))
            parameters=list(res["params"][0].keys()) #eg ['max_depth','n_estimators']
            for i in range(len(parameters)):
                parameters[i]='param_' + parameters[i]
            res["mean_test_score"]=np.around(res["mean_test_score"],4)
            pvt = pd.pivot_table(pd.DataFrame(res), values='mean_test_score', index=parameters[:colIndex] + parameters[colIndex+1 :], columns=parameters[colIndex])
            pd.set_option('display.max_columns', 20)
            pd.set_option('display.width', 1000)
            #pvt.set_option('display.max_columns', 15)
            pvt.style.apply(pd.io.formats.style.Styler.highlight_max)
            print(pvt)
            #print("\nbest parameters: " + str(gsCV.best_params_)+ "\n "+ ' with score ' + str(gsCV.best_score_))
            #ax = sns.heatmap(pvt,annot=True,cmap=sns.color_palette("Blues"))
        
    print("<<------------------------------------")


'''Goal: Compare Scores for DIFFERENT MODELS How: gives the score of clf by running ntimes times and splitting training/validation by train_size, printallScores mean all scores until this iteration, otherwise just mean
Input: -array clf, wher clf[i] is the i-th classifier. 
-trains ntimes on training dataset (size=train_size) and validation on 1-train_size
-if printAllScores=False: print only mean score . Otherwise all scores until now'''
def scoreMult_regr(clf,X_train,y_train,train_size=0.3,ntimes=1,printAllScores=False,batchDisplay=True):
    itNum=1
    print(len(clf))
    score=np.zeros((len(clf),ntimes))
    sss = ShuffleSplit(n_splits=ntimes, train_size=train_size, random_state=None)
    for train_index, test_index in sss.split(X_train, y_train):
        print("-- %d-th iteration: --------"%(itNum))
        #print("TRAIN:", train_index, "TEST:", test_index)
        if True: #This is for dataframe inputs
            X,y=X_train.iloc[train_index,:],y_train.iloc[train_index]
            X_val,y_val=X_train.iloc[test_index,:],y_train.iloc[test_index]
        if False:
            X,y=np.array([X_train[i] for i in train_index]),np.array([y_train[i] for i in train_index])
            X_val,y_val=np.array([X_train[i] for i in test_index]),np.array([y_train[i] for i in test_index])

        
        for i in range(len(clf)):
            clf[i].fit(X,y)
            pred=clf[i].predict(X_val)
            score[i][itNum-1]=r2_score(y_val,pred) #scor[i][k] ^ score of clf i at iteration k+1
            
            if not batchDisplay:
                scoresTillNow=np.around(np.flip(score[i][0:itNum]),4)
                if printAllScores:
                   print("Clf %d: %.4f -- all scores: %s"%(i,np.mean(score[i][0:itNum]),scoresTillNow))
                else:
                    print("Clf %d: %.4f "%(i,np.mean(score[i][0:itNum])))
        if batchDisplay:
            for i in range(len(clf)):
                #print
                scoresTillNow=np.around(np.flip(score[i][0:itNum]),4)
                if printAllScores:
                   print("Clf %d: %.4f -- all scores: %s"%(i,np.mean(score[i][0:itNum]),scoresTillNow))
                else:
                    print("Clf %d: %.4f "%(i,np.mean(score[i][0:itNum])))
        itNum+=1
       
'''Runs nsplits-fold CV ntimes times. Outputs mean score, std(all individual folds), std faulty!!'''
def CV(clf, X,y,nsplits=3,ntimes=1,printIntRes=True,printRes=True,n_jobs=-1):
    cvScore=[];cvScores=[]
    for i in range(ntimes):
        kFold=KFold(n_splits=nsplits,shuffle=True,random_state=None)
        sc=cross_val_score(clf,X,y,scoring='r2',cv=kFold,n_jobs=n_jobs)
        cvScore.append(np.mean(sc))
        cvScores=np.concatenate([cvScores,sc])
        if printIntRes:
            print("Step %d/%d --- score: %.4f --- mean score after %d Iterations: %.4f --- %s"%(i,ntimes,np.mean(sc),i,np.mean(cvScore),sc))       
    mean_score=round(np.mean(cvScore),4)
    std_score=round(np.std(cvScores),4)
    printOutput=str('---%d times %d-fold CV --- mean: %s---all scores: %s'%(ntimes,nsplits,str(mean_score),np.around(cvScore,4)))
    if printRes:
        print('---%d times %d-fold CV --- mean: %s---all scores: %s'%(ntimes,nsplits,str(mean_score),np.around(cvScore,4)))
        #print('mean: ' + str(mean_score)+ " in CI  " + str([round(mean_score-2*std_score,4),round(mean_score+2*std_score,4)]))
    return mean_score,printOutput#,std_score #std faulty

