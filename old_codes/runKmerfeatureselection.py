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
from pangraph.utils import binary_label
from sklearn.feature_selection import mutual_info_classif, chi2

# In[39]:

kmer_matrix_VT1_f1 = np.load('/data/hoan/amromics/prediction/data/kmer_Fold1_mat_VT1.npy') # save numpy array
selected_features_f1 = np.load('/data/hoan/amromics/prediction/data/kmer_Fold1_mat_VT1_features.npy') # save numpy array

metadata_panta = pd.read_csv("/data/hoan/amromics/prediction/data/Ecoli1936metafiles/metadata_final.csv")

mutual_mat = []
for idx in range(2, 14):
    y_class = metadata_panta.iloc[600:1200,idx].values
    print(metadata_panta.columns[idx])
    y, nonenan_index = binary_label(y_class) # v6
    pa_matrix_new = kmer_matrix_VT1_f1[nonenan_index, ]
    y_new = y[nonenan_index].astype(int)
    if len(y_new) > 10:
        scores, pvalue = chi2(pa_matrix_new, y_new)
        mutual_mat.append(scores)
mutual_mat = np.array(mutual_mat)

mutual_mat_mean = mutual_mat.mean(axis=0)

top_features = np.argsort(mutual_mat_mean)[::-1][:100000]
kmer_matrix_VT_top_features = kmer_matrix_VT1_f1[:,top_features]
selected_features_top = selected_features_f1[top_features]

data_fold = '1'
np.save('/data/hoan/amromics/prediction/data/kmer_Fold'+data_fold+'_mat_VT1_top_features.npy', kmer_matrix_VT_top_features) # save numpy array
np.save('/data/hoan/amromics/prediction/data/kmer_Fold'+data_fold+'_mat_VT1_features_top_features.npy', selected_features_top) # save numpy array