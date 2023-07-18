#!/usr/bin/env python
# coding: utf-8

# In[6]:


## Run ML methods on PanPred and panta outputs 
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn import datasets
from sklearn import svm
import random
import os
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from numpy import genfromtxt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
import pandas as pd
import numpy as np


# In[7]:


version = '_v6'  # Remove missing labels, and I = resistance


# In[8]:


def run_ML(X, y, data_set, approach="Default", feature_selection = False, FS_method = 'mutual_info_classif', X2 = None):
    base_dir = '/data/hoan/amromics/prediction/output/predPantaPanPred'+version
    if not os.path.isdir(base_dir):
        os.system('mkdir '+ base_dir)
    score = []
    methods = []
    n_loops = 2
    n_folds = 5
    n_samples = y.shape[0]
    if X2 is not None:
        print("Original shape of input:", X.shape, X2.shape)
    for i in range(n_loops):
        cv = KFold(n_splits=n_folds, shuffle=True, random_state = i)
        for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
            path_dir = base_dir +'/' + data_set + '_run_'+str(i)+'_'+ 'fold_'+str(fold)+'_'+approach
            print('Run: ', i, ', fold: ', fold)
            X_train = X[train_idx]
            X_test = X[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]
            if False:
                if i <= 0:
                    print("Run feature selection", 'method = ', FS_method)
                if FS_method == 'mutual_info_classif':
                    fs_fit = SelectKBest(mutual_info_classif, k=1000).fit(X_train, y_train)
                elif FS_method == 'chi2':
                    fs_fit = SelectKBest(chi2, k=1000).fit(X_train, y_train)
                else:
                    print("Please input correct feature selection method")
                X_train = fs_fit.transform(X_train)
                X_test = fs_fit.transform(X_test)
            if X2 is not None:
                X2_train = X2[train_idx]
                X2_test = X2[test_idx]
                if feature_selection:
                    fs2_fit = SelectKBest(chi2, k=20000).fit(X2_train, y_train)
                    X2_train = fs2_fit.transform(X2_train)
                    X2_test = fs2_fit.transform(X2_test)
                X_train = np.append(X_train, X2_train, axis = 1)
                X_test = np.append(X_test, X2_test, axis = 1)
                # print('Scale the combine data')
                # scaler = StandardScaler()
                # X_train = scaler.fit_transform(X_train)
                # X_test = scaler.fit_transform(X_test)
                
            # print("Standize the data")
            # Save the test true labels
            np.savetxt(path_dir + "_test_true_labels.csv", y_test, delimiter=",")
            if i <= 0 and fold <= 0:
                print("n_samples: ", n_samples)
                print("Reduced shape of the data: ", X_train.shape, X_test.shape)
            print(test_idx[:10])

#             # SVM
#             methods.append('SVM')
#             print(methods[-1], end =', ')
#             clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
#             y_predict = clf.predict(X_test)
#             np.savetxt(path_dir + "_SVM_labels.csv", y_predict, delimiter=",")
#             score.append(f1_score(y_predict, y_test, average='macro'))

#             # Decision Tree
#             methods.append('Decision Tree')
#             print(methods[-1], end =', ')
#             clf = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
#             y_predict = clf.predict(X_test)
#             np.savetxt(path_dir + "_DecisionTree_labels.csv", y_predict, delimiter=",")
#             score.append(f1_score(y_predict, y_test, average='macro'))

#             # RF
#             methods.append('RF')
#             print(methods[-1], end =', ')
#             clf = RandomForestClassifier().fit(X_train, y_train)
#             y_predict = clf.predict(X_test)
#             np.savetxt(path_dir + "_RandomForest_labels.csv", y_predict, delimiter=",")
#             score.append(f1_score(y_predict, y_test, average='macro'))

#             # Neural network
#             methods.append('Neural network')
#             print(methods[-1], end =', ')
#             clf = MLPClassifier(alpha=1, max_iter=2000).fit(X_train, y_train)
#             y_predict = clf.predict(X_test)
#             np.savetxt(path_dir + "_NeuralNet_labels.csv", y_predict, delimiter=",")
#             score.append(f1_score(y_predict, y_test, average='macro'))

#             # Adaboost
#             methods.append('Adaboost')
#             print(methods[-1], end =', ')
#             clf = AdaBoostClassifier().fit(X_train, y_train)
#             y_predict = clf.predict(X_test)
#             np.savetxt(path_dir + "_Adaboost_labels.csv", y_predict, delimiter=",")
#             score.append(f1_score(y_predict, y_test, average='macro'))

#             ## K-NN 
#             methods.append('kNN')
#             print(methods[-1], end =', ')
#             clf = KNeighborsClassifier(n_neighbors=10).fit(X_train, y_train)
#             y_predict = clf.predict(X_test)
#             np.savetxt(path_dir + "_NearestNeighbors_labels.csv", y_predict, delimiter=",")
#             score.append(f1_score(y_predict, y_test, average='macro'))

#             # Naive Bayes
#             methods.append('NaiveBayes')
#             print(methods[-1], end ='\n')
#             clf = GaussianNB().fit(X_train, y_train)
#             y_predict = clf.predict(X_test)
#             np.savetxt(path_dir + "_NaiveBayes_labels.csv", y_predict, delimiter=",")
#             score.append(f1_score(y_predict, y_test, average='macro'))
   
#             # Xgboost
#             clf=XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=500, objective='binary:logistic', booster='gbtree', use_label_encoder=False) #binary
#             methods.append('Xgboost')
#             print(methods[-1], end =', ')
#             XGB=clf.fit(X_train,y_train)
#             y_predict=XGB.predict(X_test)
#             np.savetxt(path_dir + "_Xgboost_labels.csv", y_predict, delimiter=",")
#             score.append(f1_score(y_predict, y_test, average='macro'))
            
            # # GradientBoostingClassifier
            # methods.append('Gradient Boost Decision Tree')
            # print(methods[-1], end =', ')
            # clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=2, random_state=0).fit(X_train, y_train)
            # y_predict = clf.predict(X_test)
            # np.savetxt(path_dir + "_GBDT_labels.csv", y_predict, delimiter=",")
            # score.append(f1_score(y_predict, y_test, average='macro'))
                  
            # LightGBM
            # if X2 is None:
            #     clfG = lgb.LGBMClassifier()
            # else:
            #     clfG = lgb.LGBMClassifier(categorical_feature=list(range(1000,1000+10000)))
            model = lgb.LGBMClassifier()
            model.fit(X_train, y_train)
            methods.append('LightGBM')
            print(methods[-1], end =', ')
            # clfG.fit(X_train, y_train)
            y_predict=model.predict(X_test) 
            np.savetxt(path_dir + "_LightGBM_labels.csv", y_predict, delimiter=",")
            score.append(f1_score(y_predict, y_test, average='macro'))
        
    # Print statistics
    n_methods = len(set(methods))
    score_np = np.array(score)
    # Each column is a method
    print(methods[:n_methods])
    average_score = np.mean(score_np.reshape((n_loops*n_folds, n_methods)), axis=0)
    print(np.round(average_score, 2))


# ### Run PanPred 

# In[9]:


# pandata = pd.read_csv("PanPred/test_data/gene_presence_absence.csv")


# In[10]:


metadata = pd.read_csv('data/Ecoli1936metafiles/PanPred_Metadata.csv')
metadata = metadata.set_index(metadata['Isolate'])


# In[11]:


accessorygene =  pd.read_csv('PanPred/test_data/AccessoryGene.csv', index_col=0)


# In[12]:


populationstructure =  pd.read_csv('PanPred/test_data/PopulationStructure.csv_labelencoded.csv', index_col=0)


# In[13]:


new_accessorygene = accessorygene.loc[metadata['Isolate']]


# #### Run ML models

# In[14]:


# for idx in range(2, 14):
#     y_class = metadata.iloc[:,idx].values
#     print(metadata.columns[idx])
#     y = np.array([1 if y_class[i]=='R' else 0 for i in range(1936)])
#     run_ML(new_accessorygene.values, y, 'Ecoli1936','classic')


# In[15]:


# new_accessorygene.head(2)


# ### Run Panta

# In[16]:


sample_isolate = pd.read_csv('/data/hoan/amromics/prediction/data/Ecoli1936metafiles/sample_isolate.csv')
sample_isolate.head(2)
sample2isolate = {}
for idx in range(len(sample_isolate.index)):
    sample2isolate[sample_isolate.iloc[idx,0]+'.contig'] = sample_isolate.iloc[idx,1]


# In[17]:


# pa_matrix = pd.read_csv('/data/hoan/amromics/prediction/output/pantaEcoli1936/gene_presence_absence.Rtab', sep='\t', index_col=0).T
pa_matrix = pd.read_csv('/data/hoan/amromics/prediction/output/pantaEcoli1936align_v4/gene_presence_absence.Rtab', sep='\t', index_col=0).T


# In[18]:


isolate_index = [sample2isolate[sample] for sample in pa_matrix.index]
metadata_panta = metadata.loc[isolate_index]


# In[19]:


metadata_panta.head(2)


# In[20]:


metadata_panta.to_csv("data/Ecoli1936metafiles/metadata_final.csv", index=False)


# In[21]:


# np.unique(metadata_panta['Year'])


# #### FS for presence-absence matrix

# In[22]:


sel = VarianceThreshold(threshold=0)
pa_matrix = sel.fit_transform(pa_matrix)


# In[23]:


pa_matrix.shape


# In[24]:


pantaout_dir = '/data/hoan/amromics/prediction/output/pantaEcoli1936align_v4/'
# snp_mat = genfromtxt(pantaout_dir + 'amrlabelencodermat_VarianceThreshold.csv', delimiter=',')
# snp_mat = np.load(pantaout_dir + 'amrlabelencodermat_VarianceThreshold.npy')
# snp_mat = np.load(pantaout_dir + 'amrlabelencodermat.npy')
# snp_mat = np.load(pantaout_dir + 'amrlabelencodermat_VT10.npy') # pantaVT10
# snp_mat = np.load(pantaout_dir + 'coregenes.npy') #pantaCoreGene
# snp_mat = np.load(pantaout_dir + 'pantaGFilterHighGeneNeighborVT5.npy') #pantaGFilterHighGeneNeighborVT5
# snp_mat = np.load(pantaout_dir + 'highDegreeGenesEncodermat.npy') # pantaHighGene
# snp_mat = np.load(pantaout_dir + 'differsite.npy') # pantaDifferSite


# In[27]:


# snp_data = 'similarsitecolsum1pct.npy'
# snp_data = 'pantaGFilterHighGeneNeighborVT5.npy'
for foldidx in range(15):
    snp_data = 'genes_fold_VT10_'+str(foldidx)+'.npy'
    snp_mat = np.load(pantaout_dir + snp_data) # pantaDifferSite
    panta_single = 'pantaVT10GeneFold' + str(foldidx)
    panta_combine = 'pantaCombineVT10GeneFold' + str(foldidx)
    print(snp_data)

    y_class = metadata_panta.iloc[:,4].values
    def binary_label(y_class):
        y_bin = []
        nonenan_index = []
        for i in range(len(y_class)):
            if y_class[i]=='R' or y_class[i]=='I':
                y_bin.append(1)
                nonenan_index.append(i)
            elif y_class[i]=='S':
                y_bin.append(0)
                nonenan_index.append(i)
            else:
                y_bin.append(y_class[i])
        return np.array(y_bin), nonenan_index


    # In[22]:


    # https://stackoverflow.com/questions/41458834/how-is-scikit-learn-cross-val-predict-accuracy-score-calculated
    ## No _ in the method name, please
    max_idx_amr = 14; # max value = 14


    # In[23]:


    # for idx in range(2, 3):
    for idx in range(2, max_idx_amr):
        y_class = metadata_panta.iloc[:,idx].values
        print(metadata_panta.columns[idx])
        # y = np.array([1 if y_class[i]=='R' else 0 for i in range(len(y_class))]) version _v5
        y, nonenan_index = binary_label(y_class) # v6
        pa_matrix_new = pa_matrix[nonenan_index, ]
        y_new = y[nonenan_index]
        snp_mat_new = snp_mat[nonenan_index,]
        # Run unimodal gene
        # run_ML(pa_matrix, y, 'Ecoli1936'+'_'+metadata_panta.columns[idx],'pantaPangenome', False, 'mutual_info_classif', None)
        # run_ML(full_matrix, y, 'Ecoli1936'+'_'+metadata_panta.columns[idx],'pantaPangenome', False, 'mutual_info_classif', None)
        # run_ML(snp_mat, y, 'Ecoli1936'+'_'+metadata_panta.columns[idx],'pantaSnp', True, 'chi2')
        # run_ML(pa_matrix, y, 'Ecoli1936'+'_'+metadata_panta.columns[idx],'pantaCombine', False, 'mutual_info_classif', snp_mat)
        # run_ML(pa_matrix, y, 'Ecoli1936'+'_'+metadata_panta.columns[idx],'pantaCombineScale', True, 'chi2', snp_mat)
        # run_ML(pa_matrix, y, 'Ecoli1936'+'_'+metadata_panta.columns[idx],'pantaCombinehighGene', False, 'chi2', snp_mat)
        # run_ML(pa_matrix_new, y_new, 'Ecoli1936'+'_'+metadata_panta.columns[idx],'pantaPangenome', False, 'chi2', None)
        # run_ML(snp_mat_new, y_new, 'Ecoli1936'+'_'+metadata_panta.columns[idx],'pantaVT10', False, 'chi2', None)
        run_ML(snp_mat_new, y_new, 'Ecoli1936'+'_'+metadata_panta.columns[idx], panta_single, False, 'chi2', None)
        run_ML(pa_matrix_new, y_new, 'Ecoli1936'+'_'+metadata_panta.columns[idx], panta_combine, False, 'chi2', snp_mat_new)

