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

pantaout_dir = '/data/hoan/amromics/prediction/output/pantaEcoli1936align_v4/'
amr_mat = np.load(pantaout_dir + 'amrlabelencodermat_VT10.npy')
mapping = dict()
for i in range(22):
    mapping[i] = i
def one_hot_encode2(seq):
    seq2 = [mapping[i] for i in seq]
    return np.eye(22)[seq2].flatten()

amr_mat = amr_mat.astype(int)
amr_matOnehot = None
for idx in range(amr_mat.shape[0]):
    if idx == 0:
        amr_matOnehot = one_hot_encode2(amr_mat[idx,:])
    else:
        amr_matOnehot = np.vstack([amr_matOnehot, one_hot_encode2(amr_mat[idx,:])])
        
np.save(pantaout_dir + 'onehotencodermat_VT10.npy', amr_matOnehot) # save numpy array
