#%% import library
import numpy as np
from numpy import linalg as LA
import pandas as pd
from scipy.stats import norm, laplace
import torch.nn as nn
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import pickle

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
# %matplotlib inline
from IPython import display

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error, brier_score_loss
from sklearn.metrics import precision_score, recall_score, roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPClassifier

from scipy.stats import linregress
from scipy.optimize import minimize
from scipy.stats import t
from statsmodels.stats.multitest import multipletests

from keras.layers import Dense, Dropout, Input, BatchNormalization
from keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping

from algorithms import *
from utils import *
from KMPE import *
from NN_functions import *
from TIcE import tice_wrapper


from torchvision.datasets import MNIST

from tqdm import tqdm_notebook as tqdm
# from tqdm import tqdm as tqdm

import warnings
warnings.filterwarnings('ignore')

import time
start_time = time.time()



# ===================================================================================
#     adjust these parameters for testing
# ====================================================================================

# data_mode = 'mushroom' # 0.48, (3916, 4208, 8124)
# data_mode = 'landsat' # 0.44, (2841, 3594, 6435)
# data_mode = 'shuttle' # 0.21, (12414, 45586, 58000)
data_mode = 'mnist_17' # 0.51, (35582, 34418, 70000)

# for not irreducible: stepwise - 0.8 - 0.7/0.6
# for irreudcible: constant - 1.0
selection_scheme = 'constant' # choose 'stepwise' and 'constant'

# choose: '1.0' (for irreducible setting)
#         '0.9' (notice this is not labeling frequency, for mushroom, landsat, mnist)
#         '0.8' (for shuttle)
selection_upp_prob = 0.9
selection_low_prob = 0.0


scenario = 'Domain adaptation' # choose 'Domain adaptation' or 'Reported at random'
# scenario = 'Reported at random'


numRepeat = 10 # number of repeated simulation
kappa_star_list = [0.1, 0.25, 0.5, 0.75] 
kappa_max_list = []


print()
print("=======================================================================================")
print("         Testing on real-world dataset:", data_mode)
print("         Practical Scenario:", scenario)
print("with ***", selection_scheme, "*** selection scheme ", "of maximum prob. of being selected = ", selection_upp_prob)
print("=======================================================================================")
print("kappa_star values:", kappa_star_list)
print("number of repeated experiments:", numRepeat)

# post_max_val = 0.8 
# print()
# print("Maximum posterior probability of belong to H =", post_max_val)


regroup_frac = 0.1
print()
print("====================================================================================")
print("                 Regrouping fraction is: ", regroup_frac)
print("====================================================================================")
# function for regrouping
def cus_sort(l):
    d = {i:l[i] for i in range(len(l))}
    s = [(k,d[k]) for k in sorted(d, key=d.get, reverse=False)]
    return s

def get_anchor_index(preds_unlabel,relabel_frac):
    index_p_list = cus_sort(preds_unlabel)
    n = len(index_p_list)
    num_anchors = int(n*relabel_frac)
    min_p = index_p_list[0][1]
    max_p = index_p_list[-1][1]
    min_f_list = []
    max_f_list = []
    for (idx, p) in index_p_list:
        if(len(min_f_list)<num_anchors):
            min_f_list.append(idx)
        else:
            break
    for (idx, p) in reversed(index_p_list):
        if(len(max_f_list)<num_anchors):
            max_f_list.append(idx)
        else:
            break
    return min_f_list, max_f_list



print()
print("=======================================================================================")
print(" Testing Case: G violates irreducibility, know posterior probability for some data points")
print("=======================================================================================")


# ==========================================================================================
#       Resampling method
# ==========================================================================================

# subsample_method = 'determinstic'
subsample_method = 'probabilistic'
rejsample_method = 'linear' # 'linear' or 'quadratic' relation to posterior

# subsample_method = 'oracle'
# oracle_option = 2 # 1 for deterministic, 2 for rejection sampling

print()
print("====================================================================================")
print("                 sub-sampling method is: ", subsample_method)
print("====================================================================================")

if subsample_method == 'determinstic':
    # used for determinstic sampling
    # neg_prob_thresh = 0.7 # unlabelled data with P[unlabeled|X=x] bigger than a certain threshold will be thrown away
    pos_prob_thresh = 0.2 # unlabelled data with P[neg|X=x] smaller than a certain threshold will be thrown away    

    print("with cut-off probability = ", pos_prob_thresh)

if subsample_method == 'probabilistic':
    # used for probabilistic sampling
    if scenario == 'Domain adaptation':
        lower_prob_thresh = 0.5 # choose this value to be bigger than (notice that the value will be over-written) 
    elif scenario == 'Reported at random':
        lower_prob_thresh = 0.6 # avoid thowing away too many sample (0.5, 0.6, 0.7)

    upper_prob_thresh = 1.0 # not throw away samples that are very likely to be unlabeled -> negative, because that has little use (and throwing away to much is not good)

    print("with lower probability = ", lower_prob_thresh, "and upper probability = ", upper_prob_thresh)

    print("Rejection rate is ***", rejsample_method,"*** relation to posterior probability")

if subsample_method == 'oracle':
    lower_prob_thresh = 0.5 # choose this value to be bigger than (notice that the value will be over-written) 
    upper_prob_thresh = 1.0 # not throw away samples that are very likely to be unlabeled -> negative, because that has little use (and throwing away to much is not good)

    print("with lower probability = ", lower_prob_thresh, "and upper probability = ", upper_prob_thresh)
    if oracle_option == 1:
        print("in a brute force/deterministic way")           

    elif oracle_option == 2:
        print("in an oracle rejection sampling way")


# =========================================================================
#               Read Data
# =========================================================================
def read_data(data_mode, truncate=None, random_state=None):

    if data_mode == 'landsat':
        df = pd.read_csv('UCI//landsat//sat.trn.txt', header=None, sep=' ')
        df = pd.concat([df, pd.read_csv('UCI//landsat//sat.tst.txt', header=None, sep=' ')])
        df = normalize_cols(df, columns=[x for x in range(36)])
        df.rename(columns={36: 'target'}, inplace=True)
        df['target'] = mul_to_bin(df['target'])

    elif data_mode == 'mushroom':
        df = pd.read_csv('UCI//mushroom//agaricus-lepiota.data.txt', header=None)
        df = dummy_encode(df)
        df.rename(columns={0: 'target'}, inplace=True)

    elif data_mode == 'shuttle':
        df = pd.read_csv('UCI//shuttle//shuttle.trn', header=None, sep=' ')
        df = pd.concat([df, pd.read_csv('UCI//shuttle//shuttle.tst.txt', header=None, sep=' ')])
        df = normalize_cols(df, columns=[x for x in range(9)])
        df.rename(columns={9: 'target'}, inplace=True)
        df['target'] = mul_to_bin(df['target'], 1)
        
    elif data_mode.startswith('mnist'):
        data = MNIST('mnist', download=True, train=True)
        data_test = MNIST('mnist', download=True, train=False)

        # added by SuMPE authors: pick out 1 and 7
        train_filter = torch.where((data.train_labels == 1) | (data.train_labels == 7))
        test_filter = torch.where((data_test.test_labels == 1) | (data_test.test_labels == 7))

        df = data.train_data[train_filter]
        target = data.train_labels[train_filter]
        df_test = data_test.test_data[test_filter]
        target_test = data_test.test_labels[test_filter]

        df = pd.DataFrame(torch.flatten(df, start_dim=1).detach().numpy())
        df_test = pd.DataFrame(torch.flatten(df_test, start_dim=1).detach().numpy())
        df = pd.concat([df, df_test])
        df = normalize_cols(df)
        
        target = pd.Series(target.detach().numpy())
        target_test = pd.Series(target_test.detach().numpy())
        target = pd.concat([target, target_test])
        
        if data_mode == 'mnist_17':
            target[target == 1] = 0
            target[target == 7] = 1
        
        df['target'] = target
    
    return df



# Yilun modifies: alpha used in DEDPUl, now kappa = 1 - alpha
# n_pos, n_pos_to_mix, n_neg_to_mix = shapes[data_mode][kappa]
shapes = {
          'landsat': {
                   0.1: (1000, 350, 3150), # added by Yilun
                   0.25: (1000, 1000, 3000), 
                   0.50: (1000, 1000, 1000), 
                   0.75: (1000, 1000, int(1000 / 3))
                   },
          'mushroom': {
                   0.1: (1000, 400, 3600),
                   0.25: (1000, 1000, 3000), 
                   0.50: (1000, 1500, 1500), # Yilun modified
                   0.75: (1000, 1500, 500), 
                   0.95: (990, 2926, 154)
                   },
          'shuttle': {
                   0.1: (1000, 400, 3600), # Yilun modified: 4000 sample in total
                   0.25: (1000, 1000, 3000), 
                   0.50: (1000, 2000, 2000), 
                   0.75: (1000, 3000, 1000), 
                   },
           'mnist_17': {
                   0.1: (1000, 400, 3600), # Yilun modified: 4000 sample in total
                   0.25: (1000, 1000, 3000), 
                   0.50: (1000, 2000, 2000), 
                   0.75: (1000, 3000, 1000), 
                    },
        }

LRS = {
    'landsat': 1e-5,
    'mushroom': 1e-4,
    'shuttle': 1e-4,
    'mnist_17': 1e-4,
}


# =====================================================
#       several sampling functions
# ====================================================

# Add 'target_s': selected positive to the whole dataset
def stepwise(array, x_thresh, low_prob, upp_prob):
    return np.piecewise(array, [array < x_thresh, array >= x_thresh], [low_prob, upp_prob])

def add_selection_target(df, data_mode, random_state=None, selection_scheme='stepwise', low_prob=0.6, upp_prob = 0.8):

    # make the datset balanced
    pos_class_size = (df['target'] == 0).sum()
    neg_class_size = (df['target'] == 1).sum()

    if pos_class_size * 3 < neg_class_size: # 3 comes from: 0.25:0.75
        df_pos = df[df['target'] == 0].sample(n=pos_class_size, random_state=random_state, replace=False).reset_index(drop=True) # sample out H from class 0
        df_neg = df[df['target'] == 1].sample(n= 3 * pos_class_size, random_state=random_state, replace=False).reset_index(drop=True) 
        df = pd.concat([df_pos, df_neg]).sample(frac=1).reset_index(drop=True)

    if selection_scheme == 'stepwise':
        features = df.drop(['target'], axis=1).values
        features_norm = LA.norm(features, axis=1)
        features_norm_quantile = np.quantile(features_norm, 0.5)
        selection_prob = stepwise(features_norm, features_norm_quantile, low_prob, upp_prob) 
    elif selection_scheme == 'constant':
        features = df.drop(['target'], axis=1).values
        selection_prob = upp_prob       

    rej_prob = np.random.uniform(0.0, 1.0, len(features))
    selection_mask = np.logical_and(selection_prob >= rej_prob, df['target'] == 0)     

    df['target_s'] = 1 # 0 - selected, 1 - unselected
    df.loc[selection_mask, 'target_s'] = 0

    return df



# get the positive and unlabeled samples
# target 0 - H, target 1 - G

def make_pu(df, data_mode, kappa=0.5, random_state=None):
    df['target_pu'] = df['target']

    df['target_pu_true'] = 2 # labeled H to be 2
    df.loc[df['target'] == 1, 'target_pu_true'] = 1 # unlabeled G to be 1

    n_pos, n_pos_to_mix, n_neg_to_mix = shapes[data_mode][kappa]

    df_pos = df[df['target'] == 0].sample(n=n_pos+n_pos_to_mix, random_state=random_state, replace=False).reset_index(drop=True) # sample out H from class 0
    df_neg = df[df['target'] == 1].sample(n=n_neg_to_mix, random_state=random_state, replace=False).reset_index(drop=True) # sample out G from class 1

    pos_to_mix_idx = df_pos.sample(n=n_pos_to_mix, random_state=random_state, replace=False).index

    df_pos.loc[pos_to_mix_idx, 'target_pu'] = 1 # sample out unlabeled pos, with pu label = 1 - for unlabeled
    df_pos.loc[pos_to_mix_idx, 'target_pu_true'] = 0 # unlabeled H to be 0
    return pd.concat([df_pos, df_neg]).sample(frac=1).reset_index(drop=True), n_pos, (n_pos_to_mix + n_neg_to_mix)


def make_su(df, data_mode, kappa=0.5, random_state=None):
    # now sample from 'target_s', not 'target'
    df['target_pu'] = df['target_s']

    df['target_pu_true'] = 2 # labeled H to be 2
    df.loc[df['target_s'] == 1, 'target_pu_true'] = 1 # unlabeled G to be 1

    n_pos, n_pos_to_mix, n_neg_to_mix = shapes[data_mode][kappa]

    df_pos = df[df['target_s'] == 0].sample(n=n_pos+n_pos_to_mix, random_state=random_state, replace=False).reset_index(drop=True) # sample out H from class 0
    df_neg = df[df['target_s'] == 1].sample(n=n_neg_to_mix, random_state=random_state, replace=False).reset_index(drop=True) # sample out G from class 1

    pos_to_mix_idx = df_pos.sample(n=n_pos_to_mix, random_state=random_state, replace=False).index

    df_pos.loc[pos_to_mix_idx, 'target_pu'] = 1 # sample out unlabeled pos, with pu label = 1 - for unlabeled
    df_pos.loc[pos_to_mix_idx, 'target_pu_true'] = 0 # unlabeled H to be 0
    return pd.concat([df_pos, df_neg]).sample(frac=1).reset_index(drop=True), n_pos, (n_pos_to_mix + n_neg_to_mix)


# get the positive and negative samples
# target 0 - H, target 1 - G
def make_pn(df, data_mode, kappa=0.5, random_state=None):
    _, n_pos, n_neg = shapes[data_mode][kappa]

    # use target_s!
    df_pos = df[df['target_s'] == 0].sample(n=n_pos, random_state=random_state, replace=False).reset_index(drop=True) # sample out H from class 0
    df_neg = df[df['target_s'] == 1].sample(n=n_neg, random_state=random_state, replace=False).reset_index(drop=True) # sample out G from class 1

    return pd.concat([df_pos, df_neg]).sample(frac=1).reset_index(drop=True), n_pos, n_neg

def make_pn_source(df, data_mode, kappa=0.5, random_state=None, neg_sub_prop=0.95): # to have CSPL: P^sr[Y=1|X=x] > P^tg[Y=1|X=x], sample fewer negative
    _, n_pos, n_neg = shapes[data_mode][kappa]

    # use target_s!
    df_pos = df[df['target_s'] == 0].sample(n=n_pos, random_state=random_state, replace=False).reset_index(drop=True) # sample out H from class 0
    df_neg = df[df['target_s'] == 1].sample(n=int(n_neg*neg_sub_prop), random_state=random_state, replace=False).reset_index(drop=True) # sample out G from class 1

    return pd.concat([df_pos, df_neg]).sample(frac=1).reset_index(drop=True), n_pos, n_neg


# ========================================================================================


print()

mean_DEDPUL_all = []
err_DEDPUL_all = []
diff_DEDPUL_all = []

mean_DEDPUL_sub_all = []
err_DEDPUL_sub_all = []
diff_DEDPUL_sub_all = []

mean_DEDPUL_regr_all = []
err_DEDPUL_regr_all = []
diff_DEDPUL_regr_all = []

mean_DEDPUL_rescale_all = []
err_DEDPUL_rescale_all = []
diff_DEDPUL_rescale_all = []



mean_EN_prior_all = []
err_EN_prior_all = []
diff_EN_prior_all = []

mean_EN_prior_sub_all = []
err_EN_prior_sub_all = []
diff_EN_prior_sub_all = []

mean_EN_prior_regr_all = []
err_EN_prior_regr_all = []
diff_EN_prior_regr_all = []

mean_EN_prior_rescale_all = []
err_EN_prior_rescale_all = []
diff_EN_prior_rescale_all = []



mean_EN_post_all = []
err_EN_post_all = []
diff_EN_post_all = []

mean_EN_post_sub_all = []
err_EN_post_sub_all = []
diff_EN_post_sub_all = []

mean_EN_post_regr_all = []
err_EN_post_regr_all = []
diff_EN_post_regr_all = []

mean_EN_post_rescale_all = []
err_EN_post_rescale_all = []
diff_EN_post_rescale_all = []



mean_KM1_all = []
err_KM1_all = []
diff_KM1_all = []

mean_KM1_sub_all = []
err_KM1_sub_all = []
diff_KM1_sub_all = []

mean_KM1_regr_all = []
err_KM1_regr_all = []
diff_KM1_regr_all = []

mean_KM1_rescale_all = []
err_KM1_rescale_all = []
diff_KM1_rescale_all = []



mean_KM2_all = []
err_KM2_all = []
diff_KM2_all = []

mean_KM2_sub_all = []
err_KM2_sub_all = []
diff_KM2_sub_all = []

mean_KM2_regr_all = []
err_KM2_regr_all = []
diff_KM2_regr_all = []

mean_KM2_rescale_all = []
err_KM2_rescale_all = []
diff_KM2_rescale_all = []



mean_tice_all = []
err_tice_all = []
diff_tice_all = []

mean_tice_sub_all = []
err_tice_sub_all = []
diff_tice_sub_all = []

mean_tice_regr_all = []
err_tice_regr_all = []
diff_tice_regr_all = []

mean_tice_rescale_all = []
err_tice_rescale_all = []
diff_tice_rescale_all = []



mean_additional_proportion_all = []
mean_remaining_proportion_all = []




for kappa_star in kappa_star_list:
    print()
    print("==============================================================================================")
    print('            true mixture proportion kappa_star =', kappa_star)
    print("===============================================================================================")

    DEDPUL_all = []
    EN_prior_all = []
    EN_post_all = []
    KM1_all = []
    KM2_all = []
    tice_all = []

    DEDPUL_sub_all = []
    EN_prior_sub_all = []
    EN_post_sub_all = []
    KM1_sub_all = []
    KM2_sub_all = []
    tice_sub_all = []

    DEDPUL_regr_all = []
    EN_prior_regr_all = []
    EN_post_regr_all = []
    KM1_regr_all = []
    KM2_regr_all = []
    tice_regr_all = []


    DEDPUL_rescale_all = []     
    EN_prior_rescale_all = []       
    EN_post_rescale_all = []       
    KM1_rescale_all = []        
    KM2_rescale_all = []        
    tice_rescale_all = []

    additional_proportion_all = []
    remaining_proportion_all = []

    kappa_max_list_all = []

    for i in range(0,numRepeat):
        print()
        print("=============================== Test #", i, " =======================================")
        random.seed(i)

        df = read_data(data_mode, truncate=None, random_state=i) # the whole dataset
        df = add_selection_target(df, data_mode, random_state=2*i, selection_scheme=selection_scheme, low_prob=selection_low_prob, upp_prob = selection_upp_prob) # choose 'stepwise' or 'constant'

        print("===================================================")
        print("             Dataset Summary")
        print("===================================================")
        print("Total Number of samples from H = ", (df['target'] == 0).sum())
        print("Total Number of samples from G = ", (df['target'] == 1).sum())

        print("Total Number of selected H = ", (df['target_s'] == 0).sum())


        df_pu, pos_size, mix_size = make_su(df, data_mode, kappa=kappa_star, random_state=i)

        print("Double-checking kappa_star proportion of data from H are sampled: ", (kappa_star - (df_pu['target_pu_true'] == 0).sum() / mix_size) < 1e-5 )
        
        kappa_max = (np.logical_and(df_pu['target'] == 0, df_pu['target_pu']  == 1)).sum() / mix_size
        kappa_max_list_all.append(kappa_max)

        print("Number of samples from component = ", pos_size)
        print("Number of samples from mixture = ", mix_size)
        print('kappa_star =', kappa_star)
        print('kappa_max =', kappa_max)
        print()

        # extract out the data
        # data_test = df.drop(['target', 'target_pu', 'target_pu_true'], axis=1).values
        data_test = df_pu.drop(['target', 'target_pu', 'target_pu_true', 'target_s'], axis=1).values 
        # drop the labeles, extact out the feature
        target_test = df_pu['target_pu'].values
        target_test_true = df_pu['target_pu_true'].values

        mix_data_test = data_test[ df_pu['target_pu'] == 1 ]
        pos_data_test = data_test[ df_pu['target_pu'] == 0 ]


        # ===========================================================================================================================
        #                                           Regrouping
        # ===========================================================================================================================
        

        # use NTC
        # preds = P[Y=unlabelled|x]: bigger preds -> more likely to be mix -> more likely to belong to neg. class G -> throw away
        preds = estimate_preds_cv(data_test, target_test, cv=5, n_networks=1, lr=LRS[data_mode], hid_dim=512, n_hid_layers=1, l2=1e-4,
                          bn=True,
                          train_nn_options={'n_epochs': 250, 'batch_size': 64,
                                            'n_batches': None, 'n_early_stop': 7, 'disp': False})

        print('regrouping ac', accuracy_score(target_test, preds.round()))
        print('regrouping roc', roc_auc_score(target_test, preds))

        # regrouping
        preds_unlabeled = preds[target_test == 1]
        min_f_idxs, max_f_idxs = get_anchor_index(preds_unlabeled, regroup_frac)

        mix_data_test_list=mix_data_test.tolist()
        estimated_set_A = [ mix_data_test_list[idx] for idx in min_f_idxs ]
        estimated_set_A = np.array(estimated_set_A)
        set_A_size = estimated_set_A.shape[0]


        data_test_regr = np.concatenate((data_test, estimated_set_A), axis=0)
        target_test_regr = np.concatenate((target_test, np.array([0] * set_A_size)), axis=0)
        target_test_true_regr = np.concatenate((target_test_true, np.array([2] * set_A_size)), axis=0)

        mix_data_test_regr = mix_data_test 
        pos_data_test_regr = np.concatenate((pos_data_test, estimated_set_A), axis=0)

        # keep this ratio (for reporting purpose)
        additional_proportion = sum(target_test_regr == 0)/sum(target_test == 0) # notice: only w.r.t unlabelled data!

        print("Number of samples from component (Original) = ", pos_size)
        print("Number of samples from mixture (Original) = ", mix_size)
        print()
        print("Number of samples from component (after Regrouping) = ", sum(target_test_regr == 0))
        print("Number of samples from mixture (after Regrouping) = ", sum(target_test_regr == 1))
        print("Enlarged proportion in labeled dataset =", round(additional_proportion, 2)) # show 2 decimal places
        print()

        additional_proportion_all.append(additional_proportion)


        # ===========================================================================================================================
        #                                           subsampling
        # ===========================================================================================================================

        

        # =========================
        # use sklearn
        # ==========================
        # # clf = LogisticRegression(random_state=0).fit(X_train, y_train)
        # clf = MLPClassifier(solver='lbfgs', alpha=1e-4,
        #                     hidden_layer_sizes=(16, ), random_state=1)
        # clf.fit(X_train, y_train)


        # poster_pos_standard = clf.predict_proba(mix_data_test)[:,1] # get the posterior probability of being positive, notice the prob. is ordered by the index of class: 0,1, etc

        # ============================================================================
        #          choosing the sampling scheme for different cases
        # ============================================================================

        # know the P[Y=1|x] 
        if scenario == 'Domain adaptation':
            # sampling & training a traditional P vs N classifier
            df_pn, train_pos_size, train_neg_size = make_pn_source(df, data_mode, kappa=kappa_star, random_state=2*i, neg_sub_prop=0.95)

            X_train = df_pn.drop(['target', 'target_pu', 'target_pu_true', 'target_s'], axis=1).values 
            y_train = df_pn['target_s'].values # get the true label

            # =========================
            # use implementation from DEDPUL
            # =========================
            # preds_standard = estimate_preds_standard(X_train, y_train, mix_data_test, n_networks=1, lr=LRS[data_mode], hid_dim=512, n_hid_layers=1, l2=1e-4,
            #                         bn=True, training_mode = 'traditional',
            #                         train_nn_options={'n_epochs': 250, 'batch_size': 64,
            #                                             'n_batches': None, 'n_early_stop': 7, 'disp': False})

            # print('ac', accuracy_score(target_test_true[df_pu['target_pu'] == 1], preds_standard.round()))

            # poster_pos_standard = 1 - preds_standard # need to 1 - xxx

            # =========================
            # use implementation from sklearn
            # =========================
            if data_mode.startswith('mnist'):
                clf = MLPClassifier(solver='sgd', alpha=1e-4, hidden_layer_sizes=(512, 512, ), random_state=1, max_iter = 20) # early stopping, prevent overfitting
            else: 
                clf = MLPClassifier(solver='sgd', alpha=1e-4, hidden_layer_sizes=(512, 512, ), random_state=1)
            
            # clf = LogisticRegression(random_state=0)
            clf.fit(X_train, y_train)
            poster_pos_standard = clf.predict_proba(mix_data_test)[:,0] # get the posterior probability of being positive, notice the prob. is ordered by the index of class: 0,1, etc

            print("Training (source) accuracy for domain adaptation:", accuracy_score(y_train, clf.predict(X_train)), ", in comparison bayes risk = ", kappa_max - kappa_star)
            print('Testing (target) accuracy for domain adaptation:', accuracy_score(target_test_true[df_pu['target_pu'] == 1], (1-poster_pos_standard).round()))



        elif scenario == 'Reported at random':
            # sample from positive data (target = 0), and get its reported label (target_s)
            df_pn, train_pos_size, train_neg_size = make_pn(df, data_mode, kappa=kappa_star, random_state=2*i)
            df_pn = df_pn[df_pn['target']==0] # only take out example with target = 0

            X_train = df_pn.drop(['target', 'target_pu', 'target_pu_true', 'target_s'], axis=1).values 
            y_train = df_pn['target_s'].values # get the label of target_s, from sample with target = 0

            # preds_standard = estimate_preds_standard(X_train, y_train, mix_data_test, n_networks=1, lr=LRS[data_mode], hid_dim=512, n_hid_layers=1, l2=1e-4,
            #                 bn=True, training_mode = 'traditional',
            #                 train_nn_options={'n_epochs': 250, 'batch_size': 8,
            #                                     'n_batches': None, 'n_early_stop': 7, 'disp': False}) # smaller batch size

            if data_mode.startswith('mnist'):
                clf = MLPClassifier(solver='sgd', alpha=1e-4, hidden_layer_sizes=(512, 512, ), random_state=1, max_iter = 20) # early stopping, prevent overfitting
            else: 
                clf = MLPClassifier(solver='sgd', alpha=1e-4, hidden_layer_sizes=(512, 512, ), random_state=1)
            # clf = MLPClassifier(solver='sgd', alpha=1e-4, hidden_layer_sizes=(512, 512, ), random_state=1)
            # clf = LogisticRegression(random_state=0)
            clf.fit(X_train, y_train)

            print("Training accuracy for reported at random:", accuracy_score(y_train, clf.predict(X_train)), "in comparison k^*/k_max = ", kappa_star/kappa_max)

            poster_pos_standard = clf.predict_proba(mix_data_test)[:,0] # get the posterior probability of being positive, notice the prob. is ordered by the index of class: 0,1, etc

            

        sampling_scheme = np.ones(len(data_test)) * 100
        sampling_scheme[target_test == 1] = poster_pos_standard

        post_max_val = np.max(poster_pos_standard)

        print()
        print("=====================================================================")
        print("    Case 3: know P[Y=1|x] for some x, with max =", post_max_val, " and min =", np.min(poster_pos_standard), "(expect to be 0)")
        print("=====================================================================")
        print()


        print("===================================================")
        print("                Subsampling                    ")
        print("===================================================")

        if subsample_method == 'probabilistic':
            rej_prob = np.random.uniform(0.0, 1.0, len(data_test)) # here, always set as Unif[0, 1]

            if rejsample_method == 'linear':
                dropping_mask = np.logical_and(rej_prob > sampling_scheme, target_test == 1)
            elif rejsample_method == 'quadratic':
                dropping_mask = np.logical_and(rej_prob > (sampling_scheme ** 2), target_test == 1)
            
            dropping_mask = np.logical_and(np.logical_and(sampling_scheme <= upper_prob_thresh, sampling_scheme > lower_prob_thresh), dropping_mask) # prevent extreme cases from being thrown away
            subsample_mask = np.logical_not(dropping_mask)
            
            print("probabilistic Subsampling: examples with posterior prob. of being positive within range [", lower_prob_thresh, ",", upper_prob_thresh, "] are being sampled")


        # =====================================================
        # subsampling
        # =====================================================
        # preds_sub = preds[subsample_mask]
        data_test_sub = data_test[subsample_mask]
        target_test_sub = target_test[subsample_mask]
        target_test_true_sub = target_test_true[subsample_mask]

        # take out the remaining mixture samples (used in KM, notice the positive samples remain the same)
        mix_data_test_sub = data_test_sub[target_test_sub == 1]
        pos_data_test_sub = pos_data_test

        # keep this ratio, used at the end of any algorithm
        remaining_proportion = sum(target_test_sub == 1)/sum(target_test == 1) # notice: only w.r.t unlabelled data!
        remaining_proportion_all.append(remaining_proportion)

        print("Number of samples from component (Original) = ", pos_size)
        print("Number of samples from mixture (Original) = ", mix_size)
        print()
        print("Number of samples from component (after subsampling) = ", sum(target_test_sub == 0))
        print("Number of samples from mixture (after subsampling) = ", sum(target_test_sub == 1))
        print("The remining proportion of mixture samples = ", remaining_proportion)
        print()
        print("kappa_star =", kappa_star, ", kappa_max =", kappa_max)
        print()

        # =============================================================================================================================
        # DEDPUL (preds used in EN)
        # =============================================================================================================================
        # choose to turn on or off the display
        # notice the name: here it is "xxx_poster_cv"
        all_conv = False # need to turn on when minst

        kappa_neg_DEDPUL, poster, preds = estimate_poster_cv(data_test, target_test, estimator='dedpul', alpha=None,
                                             estimate_poster_options={'disp': False, 'alpha_as_mean_poster': True},
                                            estimate_diff_options={
                                                 'MT': False, 'MT_coef': 0.25, 'decay_MT_coef': False, 'tune': False,
                                                 'bw_mix': 0.05, 'bw_pos': 0.1, 'threshold': 'mid', 
                                                 'n_gauss_mix': 20, 'n_gauss_pos': 10,
                                                 'bins_mix': 20, 'bins_pos': 20, 'k_neighbours': None,},
                                             estimate_preds_cv_options={
                                                 'n_networks': 1,
                                                 'cv': 5,
                                                 'random_state': i,
                                                 'hid_dim': 512,
                                                 'n_hid_layers': 1,
                                                 'lr': LRS[data_mode],
                                                 'l2': 1e-4,
                                                 'bn': True,
                                                 'all_conv': all_conv,
                                             },
                                             train_nn_options = {
                                                 'n_epochs': 250, 'loss_function': 'log', 'batch_size': 64,
                                                 'n_batches': None, 'n_early_stop': 7, 'disp': False,
                                             },
                    )  


        kappa_DEDPUL = 1 - kappa_neg_DEDPUL


        # print result
        # print()
        print("===================================================")
        print("                       DEDPUL                      ")
        print("===================================================")
        print('kappa_estimate =', kappa_DEDPUL)
        print('Abs. Error |kappa_est - kappa_star|:', abs(kappa_DEDPUL - kappa_star))
        # print('Abs. Difference |kappa_est - kappa_max|:', abs(kappa_DEDPUL - kappa_max))    

        DEDPUL_all.append(kappa_DEDPUL)


        # =========================================================================
        # Regrouping DEDPUL
        # =========================================================================
        kappa_neg_DEDPUL_regr, poster_regr, preds_regr = estimate_poster_cv(data_test_regr, target_test_regr, estimator='dedpul', alpha=None,
                                                estimate_poster_options={'disp': False, 'alpha_as_mean_poster': True},
                                            estimate_diff_options={
                                                 'MT': False, 'MT_coef': 0.25, 'decay_MT_coef': False, 'tune': False,
                                                 'bw_mix': 0.05, 'bw_pos': 0.1, 'threshold': 'mid', 
                                                 'n_gauss_mix': 20, 'n_gauss_pos': 10,
                                                 'bins_mix': 20, 'bins_pos': 20, 'k_neighbours': None,},
                                             estimate_preds_cv_options={
                                                 'n_networks': 1,
                                                 'cv': 5,
                                                 'random_state': i,
                                                 'hid_dim': 512,
                                                 'n_hid_layers': 1,
                                                 'lr': LRS[data_mode],
                                                 'l2': 1e-4,
                                                 'bn': True,
                                                 'all_conv': all_conv,
                                             },
                                             train_nn_options = {
                                                 'n_epochs': 250, 'loss_function': 'log', 'batch_size': 64,
                                                 'n_batches': None, 'n_early_stop': 7, 'disp': False,
                                             },
                                            )

        kappa_DEDPUL_regr = 1 - kappa_neg_DEDPUL_regr

        # print result
        print()
        print("===================================================")
        print("              Regrouping DEDPUL                    ")
        print("===================================================")
        print('kappa_estimate =', kappa_DEDPUL_regr)
        print('Abs. Error |kappa_est - kappa_star|:', abs(kappa_DEDPUL_regr - kappa_star))
        # print('Abs. Difference |kappa_est - kappa_max|:', abs(kappa_DEDPUL_regr - kappa_max))
        print()

        DEDPUL_regr_all.append(kappa_DEDPUL_regr)



        # print result
        print()
        kappa_DEDPUL_rescale = kappa_DEDPUL * post_max_val
        print("===================================================")
        print("                  Rescale DEDPUL                      ")
        print("===================================================")
        print('kappa_estimate =', kappa_DEDPUL_rescale)
        print('Abs. Error |kappa_est - kappa_star|:', abs(kappa_DEDPUL_rescale - kappa_star))
        # print('Abs. Difference |kappa_est - kappa_max|:', abs(kappa_DEDPUL - kappa_max))   
        DEDPUL_rescale_all.append(kappa_DEDPUL_rescale)



        # =========================================================================
        # subsampling DEDPUL
        # =========================================================================
        kappa_neg_DEDPUL_sub, poster_sub, preds_sub = estimate_poster_cv(data_test_sub, target_test_sub, estimator='dedpul', alpha=None,
                                                estimate_poster_options={'disp': False, 'alpha_as_mean_poster': True},
                                            estimate_diff_options={
                                                 'MT': False, 'MT_coef': 0.25, 'decay_MT_coef': False, 'tune': False,
                                                 'bw_mix': 0.05, 'bw_pos': 0.1, 'threshold': 'mid', 
                                                 'n_gauss_mix': 20, 'n_gauss_pos': 10,
                                                 'bins_mix': 20, 'bins_pos': 20, 'k_neighbours': None,},
                                             estimate_preds_cv_options={
                                                 'n_networks': 1,
                                                 'cv': 5,
                                                 'random_state': i,
                                                 'hid_dim': 512,
                                                 'n_hid_layers': 1,
                                                 'lr': LRS[data_mode],
                                                 'l2': 1e-4,
                                                 'bn': True,
                                                 'all_conv': all_conv,
                                             },
                                             train_nn_options = {
                                                 'n_epochs': 250, 'loss_function': 'log', 'batch_size': 64,
                                                 'n_batches': None, 'n_early_stop': 7, 'disp': False,
                                             },
                                            )

        kappa_DEDPUL_sub = 1 - kappa_neg_DEDPUL_sub
        kappa_DEDPUL_sub = kappa_DEDPUL_sub * remaining_proportion

        # print result
        print()
        print("===================================================")
        print("              Subsamping DEDPUL                    ")
        print("===================================================")
        print('kappa_estimate =', kappa_DEDPUL_sub)
        print('Abs. Error |kappa_est - kappa_star|:', abs(kappa_DEDPUL_sub - kappa_star))
        # print('Abs. Difference |kappa_est - kappa_max|:', abs(kappa_DEDPUL_sub - kappa_max))
        print()

        DEDPUL_sub_all.append(kappa_DEDPUL_sub)

        # ==========================================================================================================================================================
        # EN
        # ==========================================================================================================================================================
        EN_neg_kappa, EN_neg_poster = estimate_poster_en(preds, target_test, alpha=None, estimator='e1') # here, 'e3' in the orginal paper is used, but it underestimates; consider using 'e1'
        kappa_EN = 1 - EN_neg_kappa
        poster_EN = 1 - EN_neg_poster

        print()
        print("===================================================")
        print("                       EN                    ")
        print("===================================================")
        print('------------------ EN prior --------------------------')
        print('kappa_estimate =', kappa_EN)
        print('Abs. Error |kappa_est - kappa_star|:', abs(kappa_EN - kappa_star))
        # print('Abs. Difference |kappa_est - kappa_max|:', abs(kappa_EN - kappa_max))
        print('--------------- EN posteriors mean --------------------')
        print('kappa_estimate =', np.mean(poster_EN))
        print('Abs. Error |kappa_est - kappa_star|:', abs(np.mean(poster_EN) - kappa_star))
        # print('Abs. Difference |kappa_est - kappa_max|:', abs(np.mean(poster_EN) - kappa_max))

        EN_prior_all.append(kappa_EN)
        EN_post_all.append(np.mean(poster_EN))

         # ==========================================================================================================================================================
        # Regrouping EN
        # ==========================================================================================================================================================
        EN_neg_kappa_regr, EN_neg_poster_regr = estimate_poster_en(preds_regr, target_test_regr, alpha=None, estimator='e1') # here, 'e3' in the orginal paper is used, but it underestimates; consider using 'e1'

        print()
        print("===================================================")
        print("                 Regrouping EN                    ")
        print("===================================================")

        kappa_EN_regr = 1 - EN_neg_kappa_regr        

        poster_EN_regr = 1 - EN_neg_poster_regr        

        # print()
        print('------------------ EN prior --------------------------')
        print('kappa_estimate =', kappa_EN_regr)
        print('Abs. Error |kappa_est - kappa_star|:', abs(kappa_EN_regr - kappa_star))
        # print('Abs. Difference |kappa_est - kappa_max|:', abs(kappa_EN_regr - kappa_max))
        # print()
        print('--------------- EN posteriors mean --------------------')
        print('kappa_estimate =', np.mean(poster_EN_regr))
        print('Abs. Error |kappa_est - kappa_star|:', abs(np.mean(poster_EN_regr) - kappa_star))
        # print('Abs. Difference |kappa_est - kappa_max|:', abs(np.mean(poster_EN_regr) - kappa_max))
        print()

        EN_prior_regr_all.append(kappa_EN_regr)
        EN_post_regr_all.append(np.mean(poster_EN_regr))


        # ==========================================================================================================================================================
        # Rescale EN
        # ==========================================================================================================================================================
        print()
        kappa_EN_rescale = kappa_EN * post_max_val
        poster_EN_rescale = poster_EN * post_max_val
        print("===================================================")
        print("                      Rescale EN                    ")
        print("===================================================")
        print('------------------ EN prior --------------------------')
        print('kappa_estimate =', kappa_EN_rescale)
        print('Abs. Error |kappa_est - kappa_star|:', abs(kappa_EN_rescale - kappa_star))
        # print('Abs. Difference |kappa_est - kappa_max|:', abs(kappa_EN - kappa_max))
        print('--------------- EN posteriors mean --------------------')
        print('kappa_estimate =', np.mean(poster_EN_rescale))
        print('Abs. Error |kappa_est - kappa_star|:', abs(np.mean(poster_EN_rescale) - kappa_star))
        # print('Abs. Difference |kappa_est - kappa_max|:', abs(np.mean(poster_EN) - kappa_max))

        EN_prior_rescale_all.append(kappa_EN)
        EN_post_rescale_all.append(np.mean(poster_EN_rescale))       



        # ==========================================================================================================================================================
        # Subsampling EN
        # ==========================================================================================================================================================
        EN_neg_kappa_sub, EN_neg_poster_sub = estimate_poster_en(preds_sub, target_test_sub, alpha=None, estimator='e1') # here, 'e3' in the orginal paper is used, but it underestimates; consider using 'e1'

        print()
        print("===================================================")
        print("                 Subsampling EN                    ")
        print("===================================================")

        kappa_EN_sub = 1 - EN_neg_kappa_sub
        kappa_EN_sub = kappa_EN_sub * remaining_proportion

        poster_EN_sub = 1 - EN_neg_poster_sub
        poster_EN_sub = poster_EN_sub * remaining_proportion

        # print()
        print('------------------ EN prior --------------------------')
        print('kappa_estimate =', kappa_EN_sub)
        print('Abs. Error |kappa_est - kappa_star|:', abs(kappa_EN_sub - kappa_star))
        # print('Abs. Difference |kappa_est - kappa_max|:', abs(kappa_EN_sub - kappa_max))
        # print()
        print('--------------- EN posteriors mean --------------------')
        print('kappa_estimate =', np.mean(poster_EN_sub))
        print('Abs. Error |kappa_est - kappa_star|:', abs(np.mean(poster_EN_sub) - kappa_star))
        # print('Abs. Difference |kappa_est - kappa_max|:', abs(np.mean(poster_EN_sub) - kappa_max))
        print()

        EN_prior_sub_all.append(kappa_EN_sub)
        EN_post_sub_all.append(np.mean(poster_EN_sub))

        # ==========================================================================================================================================================
        # KM
        # ==========================================================================================================================================================
        # (kappa_KM1, kappa_KM2) = wrapper(mix_data_test, pos_data_test, disp=False, epsilon=0.04, lambda_lower_bound=0.5, lambda_upper_bound=100, 
        #                 KM_1=True, KM_2=True)                  

        if data_mode.startswith('mnist'):

            # apply non-traditional classifier
            mix_data_test_ntc = preds[ target_test == 1 ].reshape([-1, 1])
            pos_data_test_ntc = preds[ target_test == 0 ].reshape([-1, 1]) 

            (kappa_KM1, kappa_KM2) = wrapper(mix_data_test_ntc, pos_data_test_ntc, epsilon=0.04, lambda_lower_bound=0.5, lambda_upper_bound=10, 
                            KM_1=True, KM_2=True)
        else:
            (kappa_KM1, kappa_KM2) = wrapper(mix_data_test, pos_data_test, epsilon=0.04, lambda_lower_bound=0.5, lambda_upper_bound=10, 
                            KM_1=True, KM_2=True)

        print()
        print("===================================================")
        print("                   KM                    ")
        print("===================================================")
 
        print('------------------ KM1 --------------------------')
        print('kappa_estimate =', kappa_KM1)
        print('Abs. Error |kappa_est - kappa_star|:', abs(kappa_KM1 - kappa_star))
        print('Abs. Difference |kappa_est - kappa_max|:', abs(kappa_KM1 - kappa_max))
        print('------------------ KM2 --------------------------')
        print('kappa_estimate =', kappa_KM2)
        print('Abs. Error |kappa_est - kappa_star|:', abs(kappa_KM2 - kappa_star))
        print('Abs. Difference |kappa_est - kappa_max|:', abs(kappa_KM2 - kappa_max))

        KM1_all.append(kappa_KM1)
        KM2_all.append(kappa_KM2)

        # ==========================================================================================================================================================
        # Regrouping KM
        # ==========================================================================================================================================================
        # (kappa_KM1_regr, kappa_KM2_regr) = wrapper(mix_data_test_regr, pos_data_test_regr, disp=False, epsilon=0.04, lambda_lower_bound=0.5, lambda_upper_bound=100, 
                            # KM_1=True, KM_2=True)

        if data_mode.startswith('mnist'):
            # apply non-traditional classifier
            mix_data_test_regr_ntc = preds_regr[ target_test_regr == 1 ].reshape([-1, 1])
            pos_data_test_regr_ntc = preds_regr[ target_test_regr == 0 ].reshape([-1, 1])

            (kappa_KM1_regr, kappa_KM2_regr) = wrapper(mix_data_test_regr_ntc, pos_data_test_regr_ntc, epsilon=0.04, lambda_lower_bound=0.5, lambda_upper_bound=10, 
                            KM_1=True, KM_2=True)
        else:
            (kappa_KM1_regr, kappa_KM2_regr) = wrapper(mix_data_test_regr, pos_data_test_regr, disp=False, epsilon=0.04, lambda_lower_bound=0.5, lambda_upper_bound=10, 
                                    KM_1=True, KM_2=True)

        print()
        print("===================================================")
        print("                 Regrouping KM                    ")
        print("===================================================")

        print('------------------ KM1 --------------------------')
        print('kappa_estimate =', kappa_KM1_regr)
        print('Abs. Error |kappa_est - kappa_star|:', abs(kappa_KM1_regr - kappa_star))
        print('Abs. Difference |kappa_est - kappa_max|:', abs(kappa_KM1_regr - kappa_max))
        print('------------------ KM2 --------------------------')
        print('kappa_estimate =', kappa_KM2_regr)
        print('Abs. Error |kappa_est - kappa_star|:', abs(kappa_KM2_regr - kappa_star))
        print('Abs. Difference |kappa_est - kappa_max|:', abs(kappa_KM2_regr - kappa_max))

        KM1_regr_all.append(kappa_KM1_regr)
        KM2_regr_all.append(np.mean(kappa_KM2_regr))


        print()
        kappa_KM1_rescale = kappa_KM1 * post_max_val
        kappa_KM2_rescale = kappa_KM2 * post_max_val
        print("===================================================")
        print("                  Rescale KM                    ")
        print("===================================================") 
        print('------------------ KM1 --------------------------')
        print('kappa_estimate =', kappa_KM1_rescale)
        print('Abs. Error |kappa_est - kappa_star|:', abs(kappa_KM1_rescale - kappa_star))
        print('Abs. Difference |kappa_est - kappa_max|:', abs(kappa_KM1_rescale - kappa_max))
        print('------------------ KM2 --------------------------')
        print('kappa_estimate =', kappa_KM2_rescale)
        print('Abs. Error |kappa_est - kappa_star|:', abs(kappa_KM2_rescale - kappa_star))
        print('Abs. Difference |kappa_est - kappa_max|:', abs(kappa_KM2_rescale - kappa_max))

        KM1_rescale_all.append(kappa_KM1_rescale)
        KM2_rescale_all.append(kappa_KM2_rescale)

        # ==========================================================================================================================================================
        # Subsampling KM
        # ==========================================================================================================================================================
        # (kappa_KM1_sub, kappa_KM2_sub) = wrapper(mix_data_test_sub, pos_data_test_sub, disp=False, epsilon=0.04, lambda_lower_bound=0.5, lambda_upper_bound=100, 
                            # KM_1=True, KM_2=True)

        if data_mode.startswith('mnist'):
            # apply non-traditional classifier
            mix_data_test_sub_ntc = preds_sub[ target_test_sub == 1 ].reshape([-1, 1])
            pos_data_test_sub_ntc = preds_sub[ target_test_sub == 0 ].reshape([-1, 1]) 
            
            (kappa_KM1_sub, kappa_KM2_sub) = wrapper(mix_data_test_sub_ntc, pos_data_test_sub_ntc, epsilon=0.04, lambda_lower_bound=0.5, lambda_upper_bound=10, 
                            KM_1=True, KM_2=True)
        else:
            (kappa_KM1_sub, kappa_KM2_sub) = wrapper(mix_data_test_sub, pos_data_test_sub, epsilon=0.04, lambda_lower_bound=0.5, lambda_upper_bound=10, 
                            KM_1=True, KM_2=True)

        kappa_KM1_sub = kappa_KM1_sub * remaining_proportion
        kappa_KM2_sub = kappa_KM2_sub * remaining_proportion
        print()
        print("===================================================")
        print("                 Subsampling KM                    ")
        print("===================================================")

        print('------------------ KM1 --------------------------')
        print('kappa_estimate =', kappa_KM1_sub)
        print('Abs. Error |kappa_est - kappa_star|:', abs(kappa_KM1_sub - kappa_star))
        print('Abs. Difference |kappa_est - kappa_max|:', abs(kappa_KM1_sub - kappa_max))
        print('------------------ KM2 --------------------------')
        print('kappa_estimate =', kappa_KM2_sub)
        print('Abs. Error |kappa_est - kappa_star|:', abs(kappa_KM2_sub - kappa_star))
        print('Abs. Difference |kappa_est - kappa_max|:', abs(kappa_KM2_sub - kappa_max))

        KM1_sub_all.append(kappa_KM1_sub)
        KM2_sub_all.append(np.mean(kappa_KM2_sub))

        # ==========================================================================================================================================================
        # TIcE
        # ==========================================================================================================================================================
        # tice_alpha = tice_wrapper(data_test, target_test, k=10, n_folds=5, delta=0.2, n_splits=3)

        if data_mode.startswith('mnist'):
            tice_alpha = tice_wrapper(preds[:,np.newaxis], target_test, k=10, n_folds=5, delta=0.2, n_splits=3) # use NTC
        else: 
            tice_alpha = tice_wrapper(data_test, target_test, k=10, n_folds=5, delta=0.2, n_splits=3) # n_splits = 3 in code, = 4 in paper

        kappa_tice = 1 - tice_alpha

        print()
        print("===================================================")
        print("                       TIcE                    ")
        print("===================================================")
        print('kappa_estimate =', kappa_tice)
        print('Abs. Error |kappa_est - kappa_star|:', abs(kappa_tice - kappa_star))
        print()

        tice_all.append(kappa_tice)



        # ==========================================================================================================================================================
        # Regrouping TIcE
        # ==========================================================================================================================================================
        # tice_alpha_regr = tice_wrapper(data_test_regr, target_test_regr, k=10, n_folds=5, delta=0.2, n_splits=3)

        if data_mode.startswith('mnist'):
            # data_test_regr_pca = pca.transform(data_test_regr)
            # tice_alpha_regr = tice_wrapper(data_test_regr_pca, target_test_regr, k=10, n_folds=5, delta=0.2, n_splits=3) 
            tice_alpha_regr = tice_wrapper(preds_regr[:,np.newaxis], target_test_regr, k=10, n_folds=5, delta=0.2, n_splits=3)
        else: 
            tice_alpha_regr = tice_wrapper(data_test_regr, target_test_regr, k=10, n_folds=5, delta=0.2, n_splits=3)

        kappa_tice_regr = 1 - tice_alpha_regr

        # print()
        print("===================================================")
        print("               Regrouping TIcE                    ")
        print("===================================================")

        print('kappa_estimate =', kappa_tice_regr)
        print('Abs. Error |kappa_est - kappa_star|:', abs(kappa_tice_regr - kappa_star))
        print()

        tice_regr_all.append(kappa_tice_regr)




        print()
        kappa_tice_rescale = kappa_tice * post_max_val
        print("===================================================")
        print("               Rescale TIcE                    ")
        print("===================================================")

        print('kappa_estimate =', kappa_tice_rescale)
        print('Abs. Error |kappa_est - kappa_star|:', abs(kappa_tice_rescale - kappa_star))
        print()

        tice_rescale_all.append(kappa_tice_rescale)



        # ==========================================================================================================================================================
        # subsampling TIcE
        # ==========================================================================================================================================================
        # tice_alpha_sub = tice_wrapper(data_test_sub, target_test_sub, k=10, n_folds=5, delta=0.2, n_splits=3)

        if data_mode.startswith('mnist'):
            tice_alpha_sub = tice_wrapper(preds_sub[:,np.newaxis], target_test_sub, k=10, n_folds=5, delta=0.2, n_splits=3)
        else: 
            tice_alpha_sub = tice_wrapper(data_test_sub, target_test_sub, k=10, n_folds=5, delta=0.2, n_splits=3)

        kappa_tice_sub = 1 - tice_alpha_sub
        kappa_tice_sub = kappa_tice_sub * remaining_proportion
        # print()
        print("===================================================")
        print("               Subsampling TIcE                    ")
        print("===================================================")

        print('kappa_estimate =', kappa_tice_sub)
        print('Abs. Error |kappa_est - kappa_star|:', abs(kappa_tice_sub - kappa_star))
        print()

        tice_sub_all.append(kappa_tice_sub)

        # print("\n \n")
        

    print()
    print("===== Time for", numRepeat, "simulations of a single kappa_star configuration =====")
    print("-------------- %s seconds ----------------" % (time.time() - start_time))

    print("================= Displaying 3 decimal points ==========================")
    mean_remaining_proportion_all.append(np.mean(remaining_proportion_all))
    mean_additional_proportion_all.append(np.mean(additional_proportion_all))

    kappa_max_list.append(np.mean(kappa_max_list_all))

    DEDPUL_all = np.array(DEDPUL_all)
    mean_DEDPUL_all.append(np.mean(DEDPUL_all))
    diff_DEDPUL_all.append(np.mean(np.abs(DEDPUL_all - kappa_max)))
    err_DEDPUL_all.append(np.mean(np.abs(DEDPUL_all - kappa_star)))

    DEDPUL_sub_all = np.array(DEDPUL_sub_all)        
    mean_DEDPUL_sub_all.append(np.mean(DEDPUL_sub_all))     
    diff_DEDPUL_sub_all.append(np.mean(np.abs(DEDPUL_sub_all - kappa_max)))    
    err_DEDPUL_sub_all.append(np.mean(np.abs(DEDPUL_sub_all - kappa_star)))

    DEDPUL_regr_all = np.array(DEDPUL_regr_all)        
    mean_DEDPUL_regr_all.append(np.mean(DEDPUL_regr_all))     
    diff_DEDPUL_regr_all.append(np.mean(np.abs(DEDPUL_regr_all - kappa_max)))    
    err_DEDPUL_regr_all.append(np.mean(np.abs(DEDPUL_regr_all - kappa_star)))

    DEDPUL_rescale_all = np.array(DEDPUL_rescale_all)
    mean_DEDPUL_rescale_all.append(np.mean(DEDPUL_rescale_all))
    diff_DEDPUL_rescale_all.append(np.mean(np.abs(DEDPUL_rescale_all - kappa_max)))
    err_DEDPUL_rescale_all.append(np.mean(np.abs(DEDPUL_rescale_all - kappa_star)))

    print("empirical mean of DEDPUL = ", float("{:.3f}".format(np.mean(DEDPUL_all))), " std of DEDPUL = ", float("{:.3f}".format(np.std(DEDPUL_all))), " Mean absolute difference = ", float("{:.3f}".format(np.mean(np.abs(DEDPUL_all - kappa_max)))),  " Mean absolute error of of DEDPUL = ", float("{:.3f}".format(np.mean(np.abs(DEDPUL_all - kappa_star)))))
    print("empirical mean of DEDPUL_regr = ", float("{:.3f}".format(np.mean(DEDPUL_regr_all))), " std of DEDPUL_regr = ", float("{:.3f}".format(np.std(DEDPUL_regr_all)))," Mean absolute difference = ", float("{:.3f}".format(np.mean(np.abs(DEDPUL_regr_all - kappa_max)))), " Mean absolute error of DEDPUL_regr = ", float("{:.3f}".format(np.mean(np.abs(DEDPUL_regr_all - kappa_star)))))
    print("empirical mean of DEDPUL_rescale = ", float("{:.3f}".format(np.mean(DEDPUL_rescale_all))), " std of DEDPUL_rescale = ", float("{:.3f}".format(np.std(DEDPUL_rescale_all)))," Mean absolute difference = ", float("{:.3f}".format(np.mean(np.abs(DEDPUL_rescale_all - kappa_max)))), " Mean absolute error of DEDPUL_rescale = ", float("{:.3f}".format(np.mean(np.abs(DEDPUL_rescale_all - kappa_star)))))
    print("empirical mean of DEDPUL_sub = ", float("{:.3f}".format(np.mean(DEDPUL_sub_all))), " std of DEDPUL_sub = ", float("{:.3f}".format(np.std(DEDPUL_sub_all)))," Mean absolute difference = ", float("{:.3f}".format(np.mean(np.abs(DEDPUL_sub_all - kappa_max)))), " Mean absolute error of DEDPUL_sub = ", float("{:.3f}".format(np.mean(np.abs(DEDPUL_sub_all - kappa_star)))))
    print("\n") 


    EN_prior_all = np.array(EN_prior_all)
    mean_EN_prior_all.append(np.mean(EN_prior_all))
    diff_EN_prior_all.append(np.mean(np.abs(EN_prior_all - kappa_max)))
    err_EN_prior_all.append(np.mean(np.abs(EN_prior_all - kappa_star)))

    EN_prior_sub_all = np.array(EN_prior_sub_all)    
    mean_EN_prior_sub_all.append(np.mean(EN_prior_sub_all))    
    diff_EN_prior_sub_all.append(np.mean(np.abs(EN_prior_sub_all - kappa_max)))    
    err_EN_prior_sub_all.append(np.mean(np.abs(EN_prior_sub_all - kappa_star)))   

    EN_prior_regr_all = np.array(EN_prior_regr_all)    
    mean_EN_prior_regr_all.append(np.mean(EN_prior_regr_all))    
    diff_EN_prior_regr_all.append(np.mean(np.abs(EN_prior_regr_all - kappa_max)))    
    err_EN_prior_regr_all.append(np.mean(np.abs(EN_prior_regr_all - kappa_star)))  

    EN_prior_rescale_all = np.array(EN_prior_rescale_all)    
    mean_EN_prior_rescale_all.append(np.mean(EN_prior_rescale_all))    
    diff_EN_prior_rescale_all.append(np.mean(np.abs(EN_prior_rescale_all - kappa_max)))    
    err_EN_prior_rescale_all.append(np.mean(np.abs(EN_prior_rescale_all - kappa_star)))

    print("empirical mean of EN_prior = ", float("{:.3f}".format(np.mean(EN_prior_all))), " std of EN_prior = ", float("{:.3f}".format(np.std(EN_prior_all)))," Mean absolute difference = ", float("{:.3f}".format(np.mean(np.abs(EN_prior_all - kappa_max)))), " Mean absolute error of EN_prior = ", float("{:.3f}".format(np.mean(np.abs(EN_prior_all - kappa_star)))))
    print("empirical mean of EN_prior_regr = ", float("{:.3f}".format(np.mean(EN_prior_regr_all))), " std of EN_prior_regr = ", float("{:.3f}".format(np.std(EN_prior_regr_all)))," Mean absolute difference = ", float("{:.3f}".format(np.mean(np.abs(EN_prior_regr_all - kappa_max)))), " Mean absolute error of EN_prior_regr = ", float("{:.3f}".format(np.mean(np.abs(EN_prior_regr_all - kappa_star)))))
    print("empirical mean of EN_prior_rescale = ", float("{:.3f}".format(np.mean(EN_prior_rescale_all))), " std of EN_prior_rescale = ", float("{:.3f}".format(np.std(EN_prior_rescale_all)))," Mean absolute difference = ", float("{:.3f}".format(np.mean(np.abs(EN_prior_rescale_all - kappa_max)))), " Mean absolute error of EN_prior_rescale = ", float("{:.3f}".format(np.mean(np.abs(EN_prior_rescale_all - kappa_star)))))
    print("empirical mean of EN_prior_sub = ", float("{:.3f}".format(np.mean(EN_prior_sub_all))), " std of EN_prior_sub = ", float("{:.3f}".format(np.std(EN_prior_sub_all)))," Mean absolute difference = ", float("{:.3f}".format(np.mean(np.abs(EN_prior_sub_all - kappa_max)))), " Mean absolute error of EN_prior_sub = ", float("{:.3f}".format(np.mean(np.abs(EN_prior_sub_all - kappa_star)))))
    print("\n") 


    EN_post_all = np.array(EN_post_all)
    mean_EN_post_all.append(np.mean(EN_post_all))
    diff_EN_post_all.append(np.mean(np.abs(EN_post_all - kappa_max)))
    err_EN_post_all.append(np.mean(np.abs(EN_post_all - kappa_star)))

    EN_post_sub_all = np.array(EN_post_sub_all)    
    mean_EN_post_sub_all.append(np.mean(EN_post_sub_all))    
    diff_EN_post_sub_all.append(np.mean(np.abs(EN_post_sub_all - kappa_max)))    
    err_EN_post_sub_all.append(np.mean(np.abs(EN_post_sub_all - kappa_star)))  

    EN_post_regr_all = np.array(EN_post_regr_all)    
    mean_EN_post_regr_all.append(np.mean(EN_post_regr_all))    
    diff_EN_post_regr_all.append(np.mean(np.abs(EN_post_regr_all - kappa_max)))    
    err_EN_post_regr_all.append(np.mean(np.abs(EN_post_regr_all - kappa_star)))  

    EN_post_rescale_all = np.array(EN_post_rescale_all)    
    mean_EN_post_rescale_all.append(np.mean(EN_post_rescale_all))    
    diff_EN_post_rescale_all.append(np.mean(np.abs(EN_post_rescale_all - kappa_max)))    
    err_EN_post_rescale_all.append(np.mean(np.abs(EN_post_rescale_all - kappa_star)))  

    print("empirical mean of EN_post = ", float("{:.3f}".format(np.mean(EN_post_all))), " std of EN_post = ", float("{:.3f}".format(np.std(EN_post_all)))," Mean absolute difference = ", float("{:.3f}".format(np.mean(np.abs(EN_post_all - kappa_max)))), " Mean absolute error of EN_post = ", float("{:.3f}".format(np.mean(np.abs(EN_post_all - kappa_star)))))
    print("empirical mean of EN_post_regr = ", float("{:.3f}".format(np.mean(EN_post_regr_all))), " std of EN_post_regr = ", float("{:.3f}".format(np.std(EN_post_regr_all)))," Mean absolute difference = ", float("{:.3f}".format(np.mean(np.abs(EN_post_regr_all - kappa_max)))), " Mean absolute error of EN_post_regr = ", float("{:.3f}".format(np.mean(np.abs(EN_post_regr_all - kappa_star))))) 
    print("empirical mean of EN_post_rescale = ", float("{:.3f}".format(np.mean(EN_post_rescale_all))), " std of EN_post_rescale = ", float("{:.3f}".format(np.std(EN_post_rescale_all)))," Mean absolute difference = ", float("{:.3f}".format(np.mean(np.abs(EN_post_rescale_all - kappa_max)))), " Mean absolute error of EN_post_rescale = ", float("{:.3f}".format(np.mean(np.abs(EN_post_rescale_all - kappa_star)))))
    print("empirical mean of EN_post_sub = ", float("{:.3f}".format(np.mean(EN_post_sub_all))), " std of EN_post_sub = ", float("{:.3f}".format(np.std(EN_post_sub_all)))," Mean absolute difference = ", float("{:.3f}".format(np.mean(np.abs(EN_post_sub_all - kappa_max)))), " Mean absolute error of EN_post_sub = ", float("{:.3f}".format(np.mean(np.abs(EN_post_sub_all - kappa_star)))))
    print("\n") 


    KM1_all = np.array(KM1_all)
    mean_KM1_all.append(np.mean(KM1_all))
    diff_KM1_all.append(np.mean(np.abs(KM1_all - kappa_max)))
    err_KM1_all.append(np.mean(np.abs(KM1_all - kappa_star)))

    KM1_sub_all = np.array(KM1_sub_all)    
    mean_KM1_sub_all.append(np.mean(KM1_sub_all))    
    diff_KM1_sub_all.append(np.mean(np.abs(KM1_sub_all - kappa_max)))    
    err_KM1_sub_all.append(np.mean(np.abs(KM1_sub_all - kappa_star)))

    KM1_regr_all = np.array(KM1_regr_all)    
    mean_KM1_regr_all.append(np.mean(KM1_regr_all))    
    diff_KM1_regr_all.append(np.mean(np.abs(KM1_regr_all - kappa_max)))    
    err_KM1_regr_all.append(np.mean(np.abs(KM1_regr_all - kappa_star)))

    KM1_rescale_all = np.array(KM1_rescale_all)    
    mean_KM1_rescale_all.append(np.mean(KM1_rescale_all))    
    diff_KM1_rescale_all.append(np.mean(np.abs(KM1_rescale_all - kappa_max)))    
    err_KM1_rescale_all.append(np.mean(np.abs(KM1_rescale_all - kappa_star)))

    print("empirical mean of KM1 = ", float("{:.3f}".format(np.mean(KM1_all))), " std of KM1 = ", float("{:.3f}".format(np.std(KM1_all)))," Mean absolute difference = ", float("{:.3f}".format(np.mean(np.abs(KM1_all - kappa_max)))), " Mean absolute error of KM1 = ", float("{:.3f}".format(np.mean(np.abs(KM1_all - kappa_star)))))
    print("empirical mean of KM1_regr = ", float("{:.3f}".format(np.mean(KM1_regr_all))), " std of KM1_regr = ", float("{:.3f}".format(np.std(KM1_regr_all)))," Mean absolute difference = ", float("{:.3f}".format(np.mean(np.abs(KM1_regr_all - kappa_max)))), " Mean absolute error of KM1_regr = ", float("{:.3f}".format(np.mean(np.abs(KM1_regr_all - kappa_star)))))
    print("empirical mean of KM1_rescale = ", float("{:.3f}".format(np.mean(KM1_rescale_all))), " std of KM1_rescale = ", float("{:.3f}".format(np.std(KM1_rescale_all)))," Mean absolute difference = ", float("{:.3f}".format(np.mean(np.abs(KM1_rescale_all - kappa_max)))), " Mean absolute error of KM1_rescale = ", float("{:.3f}".format(np.mean(np.abs(KM1_rescale_all - kappa_star)))))
    print("empirical mean of KM1_sub = ", float("{:.3f}".format(np.mean(KM1_sub_all))), " std of KM1_sub = ", float("{:.3f}".format(np.std(KM1_sub_all)))," Mean absolute difference = ", float("{:.3f}".format(np.mean(np.abs(KM1_sub_all - kappa_max)))), " Mean absolute error of KM1_sub = ", float("{:.3f}".format(np.mean(np.abs(KM1_sub_all - kappa_star)))))
    print("\n") 


    KM2_all = np.array(KM2_all)
    mean_KM2_all.append(np.mean(KM2_all))
    diff_KM2_all.append(np.mean(np.abs(KM2_all - kappa_max)))
    err_KM2_all.append(np.mean(np.abs(KM2_all - kappa_star)))

    KM2_regr_all = np.array(KM2_regr_all)    
    mean_KM2_regr_all.append(np.mean(KM2_regr_all))    
    diff_KM2_regr_all.append(np.mean(np.abs(KM2_regr_all - kappa_max)))    
    err_KM2_regr_all.append(np.mean(np.abs(KM2_regr_all - kappa_star))) 

    KM2_sub_all = np.array(KM2_sub_all)    
    mean_KM2_sub_all.append(np.mean(KM2_sub_all))    
    diff_KM2_sub_all.append(np.mean(np.abs(KM2_sub_all - kappa_max)))    
    err_KM2_sub_all.append(np.mean(np.abs(KM2_sub_all - kappa_star)))    

    KM2_rescale_all = np.array(KM2_rescale_all)    
    mean_KM2_rescale_all.append(np.mean(KM2_rescale_all))    
    diff_KM2_rescale_all.append(np.mean(np.abs(KM2_rescale_all - kappa_max)))    
    err_KM2_rescale_all.append(np.mean(np.abs(KM2_rescale_all - kappa_star)))  

    print("empirical mean of KM2 = ", float("{:.3f}".format(np.mean(KM2_all))), " std of KM2 = ", float("{:.3f}".format(np.std(KM2_all)))," Mean absolute difference = ", float("{:.3f}".format(np.mean(np.abs(KM2_all - kappa_max)))), " Mean absolute error of KM2 = ", float("{:.3f}".format(np.mean(np.abs(KM2_all - kappa_star)))))
    print("empirical mean of KM2_regr = ", float("{:.3f}".format(np.mean(KM2_regr_all))), " std of KM2_regr = ", float("{:.3f}".format(np.std(KM2_regr_all)))," Mean absolute difference = ", float("{:.3f}".format(np.mean(np.abs(KM2_regr_all - kappa_max)))), " Mean absolute error of KM2_regr = ", float("{:.3f}".format(np.mean(np.abs(KM2_regr_all - kappa_star)))))
    print("empirical mean of KM2_rescale = ", float("{:.3f}".format(np.mean(KM2_rescale_all))), " std of KM2_rescale = ", float("{:.3f}".format(np.std(KM2_rescale_all)))," Mean absolute difference = ", float("{:.3f}".format(np.mean(np.abs(KM2_rescale_all - kappa_max)))), " Mean absolute error of KM2_rescale = ", float("{:.3f}".format(np.mean(np.abs(KM2_rescale_all - kappa_star)))))
    print("empirical mean of KM2_sub = ", float("{:.3f}".format(np.mean(KM2_sub_all))), " std of KM2_sub = ", float("{:.3f}".format(np.std(KM2_sub_all)))," Mean absolute difference = ", float("{:.3f}".format(np.mean(np.abs(KM2_sub_all - kappa_max)))), " Mean absolute error of KM2_sub = ", float("{:.3f}".format(np.mean(np.abs(KM2_sub_all - kappa_star)))))
    print("\n") 


    tice_all = np.array(tice_all)
    mean_tice_all.append(np.mean(tice_all))
    diff_tice_all.append(np.mean(np.abs(tice_all - kappa_max)))
    err_tice_all.append(np.mean(np.abs(tice_all - kappa_star)))

    tice_regr_all = np.array(tice_regr_all)    
    mean_tice_regr_all.append(np.mean(tice_regr_all))    
    diff_tice_regr_all.append(np.mean(np.abs(tice_regr_all - kappa_max)))    
    err_tice_regr_all.append(np.mean(np.abs(tice_regr_all - kappa_star)))

    tice_sub_all = np.array(tice_sub_all)    
    mean_tice_sub_all.append(np.mean(tice_sub_all))    
    diff_tice_sub_all.append(np.mean(np.abs(tice_sub_all - kappa_max)))    
    err_tice_sub_all.append(np.mean(np.abs(tice_sub_all - kappa_star)))

    tice_rescale_all = np.array(tice_rescale_all)    
    mean_tice_rescale_all.append(np.mean(tice_rescale_all))    
    diff_tice_rescale_all.append(np.mean(np.abs(tice_rescale_all - kappa_max)))    
    err_tice_rescale_all.append(np.mean(np.abs(tice_rescale_all - kappa_star)))

    print("empirical mean of tice = ", float("{:.3f}".format(np.mean(tice_all))), " std of tice = ", float("{:.3f}".format(np.std(tice_all)))," Mean absolute difference = ", float("{:.3f}".format(np.mean(np.abs(tice_all - kappa_max)))), " Mean absolute error of tice = ", float("{:.3f}".format(np.mean(np.abs(tice_all - kappa_star)))))
    print("empirical mean of tice_regr = ", float("{:.3f}".format(np.mean(tice_regr_all))), " std of tice_regr = ", float("{:.3f}".format(np.std(tice_regr_all)))," Mean absolute difference = ", float("{:.3f}".format(np.mean(np.abs(tice_regr_all - kappa_max)))), " Mean absolute error of tice_regr = ", float("{:.3f}".format(np.mean(np.abs(tice_regr_all - kappa_star)))))
    print("empirical mean of tice_rescale = ", float("{:.3f}".format(np.mean(tice_rescale_all))), " std of tice_rescale = ", float("{:.3f}".format(np.std(tice_rescale_all)))," Mean absolute difference = ", float("{:.3f}".format(np.mean(np.abs(tice_rescale_all - kappa_max)))), " Mean absolute error of tice_rescale = ", float("{:.3f}".format(np.mean(np.abs(tice_rescale_all - kappa_star)))))
    print("empirical mean of tice_sub = ", float("{:.3f}".format(np.mean(tice_sub_all))), " std of tice_sub = ", float("{:.3f}".format(np.std(tice_sub_all)))," Mean absolute difference = ", float("{:.3f}".format(np.mean(np.abs(tice_sub_all - kappa_max)))), " Mean absolute error of tice_sub = ", float("{:.3f}".format(np.mean(np.abs(tice_sub_all - kappa_star)))))

    print("\n \n \n \n ") 

print("===== Total Time for ", len(kappa_star_list), " kappa_star configurations =====")
print("-------------- %s seconds ----------------" % (time.time() - start_time))


print()
kappa_star_list = np.array(kappa_star_list)
kappa_max_list = np.array(kappa_max_list)


# DEDPUL
mean_DEDPUL_all = np.array(mean_DEDPUL_all)
err_DEDPUL_all = np.array(err_DEDPUL_all)
diff_DEDPUL_all = np.array(diff_DEDPUL_all)

mean_DEDPUL_sub_all = np.array(mean_DEDPUL_sub_all)
err_DEDPUL_sub_all = np.array(err_DEDPUL_sub_all)
diff_DEDPUL_sub_all = np.array(diff_DEDPUL_sub_all)

mean_DEDPUL_rescale_all = np.array(mean_DEDPUL_rescale_all)
err_DEDPUL_rescale_all = np.array(err_DEDPUL_rescale_all)
diff_DEDPUL_rescale_all = np.array(diff_DEDPUL_rescale_all)

mean_DEDPUL_regr_all = np.array(mean_DEDPUL_regr_all)
err_DEDPUL_regr_all = np.array(err_DEDPUL_regr_all)
diff_DEDPUL_regr_all = np.array(diff_DEDPUL_regr_all)


# EN_prior
mean_EN_prior_all = np.array(mean_EN_prior_all)
err_EN_prior_all = np.array(err_EN_prior_all)
diff_EN_prior_all = np.array(diff_EN_prior_all)

mean_EN_prior_sub_all = np.array(mean_EN_prior_sub_all)
err_EN_prior_sub_all = np.array(err_EN_prior_sub_all)
diff_EN_prior_sub_all = np.array(diff_EN_prior_sub_all)

mean_EN_prior_rescale_all = np.array(mean_EN_prior_rescale_all)
err_EN_prior_rescale_all = np.array(err_EN_prior_rescale_all)
diff_EN_prior_rescale_all = np.array(diff_EN_prior_rescale_all)

mean_EN_prior_regr_all = np.array(mean_EN_prior_regr_all)
err_EN_prior_regr_all = np.array(err_EN_prior_regr_all)
diff_EN_prior_regr_all = np.array(diff_EN_prior_regr_all)



# EN_post
mean_EN_post_all = np.array(mean_EN_post_all)
err_EN_post_all = np.array(err_EN_post_all)
diff_EN_post_all = np.array(diff_EN_post_all)

mean_EN_post_sub_all = np.array(mean_EN_post_sub_all)
err_EN_post_sub_all = np.array(err_EN_post_sub_all)
diff_EN_post_sub_all = np.array(diff_EN_post_sub_all)

mean_EN_post_rescale_all = np.array(mean_EN_post_rescale_all)
err_EN_post_rescale_all = np.array(err_EN_post_rescale_all)
diff_EN_post_rescale_all = np.array(diff_EN_post_rescale_all)

mean_EN_post_regr_all = np.array(mean_EN_post_regr_all)
err_EN_post_regr_all = np.array(err_EN_post_regr_all)
diff_EN_post_regr_all = np.array(diff_EN_post_regr_all)


# KM1,2
mean_KM1_all = np.array(mean_KM1_all)
err_KM1_all = np.array(err_KM1_all)
diff_KM1_all = np.array(diff_KM1_all)

mean_KM1_sub_all = np.array(mean_KM1_sub_all)
err_KM1_sub_all = np.array(err_KM1_sub_all)
diff_KM1_sub_all = np.array(diff_KM1_sub_all)

mean_KM1_rescale_all = np.array(mean_KM1_rescale_all)
err_KM1_rescale_all = np.array(err_KM1_rescale_all)
diff_KM1_rescale_all = np.array(diff_KM1_rescale_all)

mean_KM1_regr_all = np.array(mean_KM1_regr_all)
err_KM1_regr_all = np.array(err_KM1_regr_all)
diff_KM1_regr_all = np.array(diff_KM1_regr_all)


mean_KM2_all = np.array(mean_KM2_all)
err_KM2_all = np.array(err_KM2_all)
diff_KM2_all = np.array(diff_KM2_all)

mean_KM2_sub_all = np.array(mean_KM2_sub_all)
err_KM2_sub_all = np.array(err_KM2_sub_all)
diff_KM2_sub_all = np.array(diff_KM2_sub_all)

mean_KM2_rescale_all = np.array(mean_KM2_rescale_all)
err_KM2_rescale_all = np.array(err_KM2_rescale_all)
diff_KM2_rescale_all = np.array(diff_KM2_rescale_all)

mean_KM2_regr_all = np.array(mean_KM2_regr_all)
err_KM2_regr_all = np.array(err_KM2_regr_all)
diff_KM2_regr_all = np.array(diff_KM2_regr_all)


# tice
mean_tice_all = np.array(mean_tice_all)
err_tice_all = np.array(err_tice_all)
diff_tice_all = np.array(diff_tice_all)

mean_tice_sub_all = np.array(mean_tice_sub_all)
err_tice_sub_all = np.array(err_tice_sub_all)
diff_tice_sub_all = np.array(diff_tice_sub_all)

mean_tice_rescale_all = np.array(mean_tice_rescale_all)
err_tice_rescale_all = np.array(err_tice_rescale_all)
diff_tice_rescale_all = np.array(diff_tice_rescale_all)

mean_tice_regr_all = np.array(mean_tice_regr_all)
err_tice_regr_all = np.array(err_tice_regr_all)
diff_tice_regr_all = np.array(diff_tice_regr_all)


# convert to numpy array
kappa_star_list = np.array(kappa_star_list)
kappa_max_list = np.array(kappa_max_list)

mean_DEDPUL_all = np.array(mean_DEDPUL_all)
err_DEDPUL_all = np.array(err_DEDPUL_all)
diff_DEDPUL_all = np.array(diff_DEDPUL_all)

mean_DEDPUL_sub_all = np.array(mean_DEDPUL_sub_all)
err_DEDPUL_sub_all = np.array(err_DEDPUL_sub_all)
diff_DEDPUL_sub_all = np.array(diff_DEDPUL_sub_all)

mean_DEDPUL_rescale_all = np.array(mean_DEDPUL_rescale_all)
err_DEDPUL_rescale_all = np.array(err_DEDPUL_rescale_all)
diff_DEDPUL_rescale_all = np.array(diff_DEDPUL_rescale_all)

mean_DEDPUL_regr_all = np.array(mean_DEDPUL_regr_all)
err_DEDPUL_regr_all = np.array(err_DEDPUL_regr_all)
diff_DEDPUL_regr_all = np.array(diff_DEDPUL_regr_all)


mean_EN_prior_all = np.array(mean_EN_prior_all)
err_EN_prior_all = np.array(err_EN_prior_all)
diff_EN_prior_all = np.array(diff_EN_prior_all)

mean_EN_prior_sub_all = np.array(mean_EN_prior_sub_all)
err_EN_prior_sub_all = np.array(err_EN_prior_sub_all)
diff_EN_prior_sub_all = np.array(diff_EN_prior_sub_all)

mean_EN_prior_rescale_all = np.array(mean_EN_prior_rescale_all)
err_EN_prior_rescale_all = np.array(err_EN_prior_rescale_all)
diff_EN_prior_rescale_all = np.array(diff_EN_prior_rescale_all)

mean_EN_prior_regr_all = np.array(mean_EN_prior_regr_all)
err_EN_prior_regr_all = np.array(err_EN_prior_regr_all)
diff_EN_prior_regr_all = np.array(diff_EN_prior_regr_all)


mean_EN_post_all = np.array(mean_EN_post_all)
err_EN_post_all = np.array(err_EN_post_all)
diff_EN_post_all = np.array(diff_EN_post_all)

mean_EN_post_sub_all = np.array(mean_EN_post_sub_all)
err_EN_post_sub_all = np.array(err_EN_post_sub_all)
diff_EN_post_sub_all = np.array(diff_EN_post_sub_all)

mean_EN_post_rescale_all = np.array(mean_EN_post_rescale_all)
err_EN_post_rescale_all = np.array(err_EN_post_rescale_all)
diff_EN_post_rescale_all = np.array(diff_EN_post_rescale_all)

mean_EN_post_regr_all = np.array(mean_EN_post_regr_all)
err_EN_post_regr_all = np.array(err_EN_post_regr_all)
diff_EN_post_regr_all = np.array(diff_EN_post_regr_all)


mean_KM1_all = np.array(mean_KM1_all)
err_KM1_all = np.array(err_KM1_all)
diff_KM1_all = np.array(diff_KM1_all)

mean_KM1_sub_all = np.array(mean_KM1_sub_all)
err_KM1_sub_all = np.array(err_KM1_sub_all)
diff_KM1_sub_all = np.array(diff_KM1_sub_all)

mean_KM1_rescale_all = np.array(mean_KM1_rescale_all)
err_KM1_rescale_all = np.array(err_KM1_rescale_all)
diff_KM1_rescale_all = np.array(diff_KM1_rescale_all)

mean_KM1_regr_all = np.array(mean_KM1_regr_all)
err_KM1_regr_all = np.array(err_KM1_regr_all)
diff_KM1_regr_all = np.array(diff_KM1_regr_all)

mean_KM2_all = np.array(mean_KM2_all)
err_KM2_all = np.array(err_KM2_all)
diff_KM2_all = np.array(diff_KM2_all)

mean_KM2_sub_all = np.array(mean_KM2_sub_all)
err_KM2_sub_all = np.array(err_KM2_sub_all)
diff_KM2_sub_all = np.array(diff_KM2_sub_all)

mean_KM2_rescale_all = np.array(mean_KM2_rescale_all)
err_KM2_rescale_all = np.array(err_KM2_rescale_all)
diff_KM2_rescale_all = np.array(diff_KM2_rescale_all)

mean_KM2_regr_all = np.array(mean_KM2_regr_all)
err_KM2_regr_all = np.array(err_KM2_regr_all)
diff_KM2_regr_all = np.array(diff_KM2_regr_all)


mean_tice_all = np.array(mean_tice_all)
err_tice_all = np.array(err_tice_all)
diff_tice_all = np.array(diff_tice_all)

mean_tice_sub_all = np.array(mean_tice_sub_all)
err_tice_sub_all = np.array(err_tice_sub_all)
diff_tice_sub_all = np.array(diff_tice_sub_all)

mean_tice_rescale_all = np.array(mean_tice_rescale_all)
err_tice_rescale_all = np.array(err_tice_rescale_all)
diff_tice_rescale_all = np.array(diff_tice_rescale_all)

mean_tice_regr_all = np.array(mean_tice_regr_all)
err_tice_regr_all = np.array(err_tice_regr_all)
diff_tice_regr_all = np.array(diff_tice_regr_all)

mean_additional_proportion_all = np.array(mean_additional_proportion_all)
mean_remaining_proportion_all = np.array(mean_remaining_proportion_all)


np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

print("Enlarged proportion in labeled dataset =", mean_additional_proportion_all, "(the ReCPE paper expect negative samples to be regrouped, but I don't think so)")
print("Unlabelled Proportion remained after subsampling:", mean_remaining_proportion_all, "(expect negative samples to be thrown away)")
print()
print("kappa_max  values:", kappa_max_list, "(focus on abs. difference)")
print()
print("DEDPUL       mean:", mean_DEDPUL_all, " abs. difference", diff_DEDPUL_all)
print("EN_prior     mean:", mean_EN_prior_all, " abs. difference", diff_EN_prior_all)
print("EN_post      mean:", mean_EN_post_all, " abs. difference", diff_EN_post_all)
print("KM1          mean:", mean_KM1_all, " abs. difference", diff_KM1_all)
print("KM2          mean:", mean_KM2_all, " abs. difference", diff_KM2_all)
print("tice         mean:", mean_tice_all, " abs. difference", diff_tice_all)
print()
print("kappa_star values:", kappa_star_list, "(focus on abs. error)")

print()
print("DEDPUL_regr   mean:", mean_DEDPUL_regr_all, " abs. error", err_DEDPUL_regr_all)
print("EN_prior_regr mean:", mean_EN_prior_regr_all, " abs. error", err_EN_prior_regr_all)
print("EN_post_regr  mean:", mean_EN_post_regr_all, " abs. error", err_EN_post_regr_all)
print("KM1_regr      mean:", mean_KM1_regr_all, " abs. error", err_KM1_regr_all)
print("KM2_regr      mean:", mean_KM2_regr_all, " abs. error", err_KM2_regr_all)
print("tice_regr     mean:", mean_tice_regr_all, " abs. error", err_tice_regr_all)

# print()
# print("DEDPUL_rescale   mean:", mean_DEDPUL_rescale_all, " abs. error", err_DEDPUL_rescale_all)
# print("EN_prior_rescale mean:", mean_EN_prior_rescale_all, " abs. error", err_EN_prior_rescale_all)
# print("EN_post_rescale  mean:", mean_EN_post_rescale_all, " abs. error", err_EN_post_rescale_all)
# print("KM1_rescale      mean:", mean_KM1_rescale_all, " abs. error", err_KM1_rescale_all)
# print("KM2_rescale      mean:", mean_KM2_rescale_all, " abs. error", err_KM2_rescale_all)
# print("tice_rescale     mean:", mean_tice_rescale_all, " abs. error", err_tice_rescale_all)

print()
print("DEDPUL_sub       mean:", mean_DEDPUL_sub_all, " abs. error", err_DEDPUL_sub_all)
print("EN_prior_sub     mean:", mean_EN_prior_sub_all, " abs. error", err_EN_prior_sub_all)
print("EN_post_sub      mean:", mean_EN_post_sub_all, " abs. error", err_EN_post_sub_all)
print("KM1_sub          mean:", mean_KM1_sub_all, " abs. error", err_KM1_sub_all)
print("KM2_sub          mean:", mean_KM2_sub_all, " abs. error", err_KM2_sub_all)
print("tice_sub         mean:", mean_tice_sub_all, " abs. error", err_tice_sub_all)




print("\nCompare of original method with its subsampling version")
print("DEDPUL           mean:", mean_DEDPUL_all, " abs. error", err_DEDPUL_all, "avg abs. error", np.round(np.mean(err_DEDPUL_all), 3))
print("DEDPUL_regr      mean:", mean_DEDPUL_regr_all, " abs. error", err_DEDPUL_regr_all, "avg abs. error", np.round(np.mean(err_DEDPUL_regr_all), 3))
print("DEDPUL_sub       mean:", mean_DEDPUL_sub_all,  " abs. error", err_DEDPUL_sub_all, "avg abs. error", np.round(np.mean(err_DEDPUL_sub_all), 3))
print()
print("EN_prior         mean:", mean_EN_prior_all, " abs. error", err_EN_prior_all, "avg abs. error", np.round(np.mean(err_EN_prior_all), 3))
print("EN_prior_regr    mean:", mean_EN_prior_regr_all, " abs. error", err_EN_prior_regr_all, "avg abs. error", np.round(np.mean(err_EN_prior_regr_all), 3))
print("EN_prior_sub     mean:", mean_EN_prior_sub_all, " abs. error", err_EN_prior_sub_all, "avg abs. error", np.round(np.mean(err_EN_prior_sub_all), 3))
print()
print("EN_post          mean:", mean_EN_post_all, " abs. error", err_EN_post_all, "avg abs. error", np.round(np.mean(err_EN_post_all), 3))
print("EN_post_regr     mean:", mean_EN_post_regr_all, " abs. error", err_EN_post_regr_all, "avg abs. error", np.round(np.mean(err_EN_post_regr_all), 3))
print("EN_post_sub      mean:", mean_EN_post_sub_all, " abs. error", err_EN_post_sub_all, "avg abs. error", np.round(np.mean(err_EN_post_sub_all), 3))
print()
print("KM1              mean:", mean_KM1_all, " abs. error", err_KM1_all, "avg abs. error", np.round(np.mean(err_KM1_all), 3))
print("KM1_regr         mean:", mean_KM1_regr_all, " abs. error", err_KM1_regr_all, "avg abs. error", np.round(np.mean(err_KM1_regr_all), 3))
print("KM1_sub          mean:", mean_KM1_sub_all, " abs. error", err_KM1_sub_all, "avg abs. error", np.round(np.mean(err_KM1_sub_all), 3))
print()
print("KM2              mean:", mean_KM2_all, " abs. error", err_KM2_all, "avg abs. error", np.round(np.mean(err_KM2_all), 3))
print("KM2_regr         mean:", mean_KM2_regr_all, " abs. error", err_KM2_regr_all, "avg abs. error", np.round(np.mean(err_KM2_regr_all), 3))
print("KM2_sub          mean:", mean_KM2_sub_all, " abs. error", err_KM2_sub_all, "avg abs. error", np.round(np.mean(err_KM2_sub_all), 3))
print()
print("tice             mean:", mean_tice_all, " abs. error", err_tice_all, "avg abs. error", np.round(np.mean(err_tice_all), 3))
print("tice_regr        mean:", mean_tice_regr_all, " abs. error", err_tice_regr_all, "avg abs. error", np.round(np.mean(err_tice_regr_all), 3))
print("tice_sub         mean:", mean_tice_sub_all, " abs. error", err_tice_sub_all, "avg abs. error", np.round(np.mean(err_tice_sub_all), 3))


# old summary print
# print("\nCompare of original method with its subsampling version")
# print("DEDPUL           mean:", mean_DEDPUL_all, " abs. difference", diff_DEDPUL_all, " abs. error", err_DEDPUL_all)
# print("DEDPUL_regr      mean:", mean_DEDPUL_regr_all, " abs. difference", diff_DEDPUL_regr_all, " abs. error", err_DEDPUL_regr_all)
# # print("DEDPUL_rescale   mean:", mean_DEDPUL_rescale_all, " abs. difference", diff_DEDPUL_rescale_all, " abs. error", err_DEDPUL_rescale_all)
# print("DEDPUL_sub       mean:", mean_DEDPUL_sub_all, " abs. difference", diff_DEDPUL_sub_all, " abs. error", err_DEDPUL_sub_all)
# print()
# print("EN_prior         mean:", mean_EN_prior_all, " abs. difference", diff_EN_prior_all, " abs. error", err_EN_prior_all)
# print("EN_prior_regr    mean:", mean_EN_prior_regr_all, " abs. difference", diff_EN_prior_regr_all, " abs. error", err_EN_prior_regr_all)
# # print("EN_prior_rescale mean:", mean_EN_prior_rescale_all, " abs. difference", diff_EN_prior_rescale_all, " abs. error", err_EN_prior_rescale_all)
# print("EN_prior_sub     mean:", mean_EN_prior_sub_all, " abs. difference", diff_EN_prior_sub_all, " abs. error", err_EN_prior_sub_all)
# print()
# print("EN_post          mean:", mean_EN_post_all, " abs. difference", diff_EN_post_all, " abs. error", err_EN_post_all)
# print("EN_post_regr     mean:", mean_EN_post_regr_all, " abs. difference", diff_EN_post_regr_all, " abs. error", err_EN_post_regr_all)
# # print("EN_post_rescale  mean:", mean_EN_post_rescale_all, " abs. difference", diff_EN_post_rescale_all, " abs. error", err_EN_post_rescale_all)
# print("EN_post_sub      mean:", mean_EN_post_sub_all, " abs. difference", diff_EN_post_sub_all, " abs. error", err_EN_post_sub_all)
# print()
# print("KM1              mean:", mean_KM1_all, " abs. difference", diff_KM1_all, " abs. error", err_KM1_all)
# print("KM1_regr         mean:", mean_KM1_regr_all, " abs. difference", diff_KM1_regr_all, " abs. error", err_KM1_regr_all)
# # print("KM1_rescale      mean:", mean_KM1_rescale_all, " abs. difference", diff_KM1_rescale_all, " abs. error", err_KM1_rescale_all)
# print("KM1_sub          mean:", mean_KM1_sub_all, " abs. difference", diff_KM1_sub_all, " abs. error", err_KM1_sub_all)
# print()
# print("KM2              mean:", mean_KM2_all, " abs. difference", diff_KM2_all, " abs. error", err_KM2_all)
# print("KM2_regr         mean:", mean_KM2_regr_all, " abs. difference", diff_KM2_regr_all, " abs. error", err_KM2_regr_all)
# # print("KM2_rescale      mean:", mean_KM2_rescale_all, " abs. difference", diff_KM2_rescale_all, " abs. error", err_KM2_rescale_all)
# print("KM2_sub          mean:", mean_KM2_sub_all, " abs. difference", diff_KM2_sub_all, " abs. error", err_KM2_sub_all)
# print()
# print("tice             mean:", mean_tice_all, " abs. difference", diff_tice_all, " abs. error", err_tice_all)
# print("tice_regr        mean:", mean_tice_regr_all, " abs. difference", diff_tice_regr_all, " abs. error", err_tice_regr_all)
# # print("tice_rescale     mean:", mean_tice_rescale_all, " abs. difference", diff_tice_rescale_all, " abs. error", err_tice_rescale_all)
# print("tice_sub         mean:", mean_tice_sub_all, " abs. difference", diff_tice_sub_all, " abs. error", err_tice_sub_all)




# %%
