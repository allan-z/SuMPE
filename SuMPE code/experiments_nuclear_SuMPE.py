# real testing

# %% import library
import numpy as np
import pandas as pd
from scipy.stats import norm, laplace
import random

from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error, confusion_matrix
from sklearn.metrics import precision_score, recall_score, roc_auc_score, brier_score_loss, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
# %matplotlib inline
from matplotlib.lines import Line2D
from IPython import display

from algorithms import *
from utils import *
from KMPE_discrete import * # SuMPE author's implementation of discrete RV version of KM

from NN_functions import *
from TIcE import tice_wrapper

import warnings
warnings.filterwarnings('ignore')

import csv
# Use interpolation to estimate background contribution in certain channels
from scipy.interpolate import interp1d
from random import choices

import time

start_time = time.time()


# random.seed(10) # set the seed for sampling
# ===================================================================================
#     adjust these parameters for testing
# ====================================================================================


gamma = 0.8 # G = gamma_star * G1 + (1-gamma_star) * G2, need not change

data_mode = 'Sp2021 NaI' # choose between 'Sp2021 NaI', 'Su2022 NaI'

print()
print("=======================================================================================")
print("                  Test on nuclear dataset:", data_mode)
print("=======================================================================================")
print("positive distribution is: Cesium")
print("negative distribution is: ", gamma, " * Cobalt and ", 1-gamma, " * Terrestrial background")


# pos and mix size used in DEDPUL: 1000, 10000
pos_size = 50000
mix_size = 50000

numRepeat = 10 # number of repeated simulation

# kappa_star is proportion of P in U; (1 - kappa_star) is proportion of N in U;
# kappa_star = 0.25
# kappa_star_list = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
kappa_star_list = [0.1, 0.25, 0.5, 0.75]
kappa_max_list = []

print()
print("kappa_star values:", kappa_star_list)
print("number of repeated experiments:", numRepeat)
print("Number of samples from component = ", pos_size, ", Number of samples from mixture = ", mix_size)

regroup_frac = 0.1
print()
print("====================================================================================")
print("                 Regrouping fraction is: ", regroup_frac)
print("====================================================================================")

# %% function for regrouping
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

# helper functions from gamma spectra
def get_pmf(data, numChannel):
    data_hist = np.zeros(numChannel)

    for i in data:
        data_hist[i] += 1

    data_pmf = data_hist / data_hist.sum()
    return data_pmf

def estimate_cons_kappa_nuclear(F, H):
    numChannel = F.size
    ratio = np.zeros(numChannel)
    
    for i in range(numChannel):
        if (H[i] != 0) and (F[i] != 0): # numChannel may be bigger than support of both distributions
            ratio[i] = F[i] / H[i]
        else:
            ratio[i] = 1
    
    return [np.min(ratio), np.argmin(ratio)]


def estimate_poster_pmf(pos_data_train, mix_data_train, data_test, numChannel):
    """ pos_data_train, mix_data_train: used to build empirical pmf
        data_test: used to evaluate the posterior prob.
        return: poster (k*fp(x)/fu(x), posterior probability of being negative, following the convention in DEDPUL) """


    pos_data_train_hist = np.zeros(numChannel) # smoothing
    mix_data_train_hist = np.zeros(numChannel)

    for i in pos_data_train:
        pos_data_train_hist[i] += 1

    for i in mix_data_train:
        mix_data_train_hist[i] += 1

    poster_pos_hist = pos_data_train_hist / (mix_data_train_hist+1e-6)

    poster_neg = 1 - poster_pos_hist[data_test] # 1 - k*h(x)/f(x): follow the convention of DEDPUL paper

    return poster_neg

def estimate_preds_pmf(pos_data_train, mix_data_train, data_test, numChannel):
    """ pos_data_train, mix_data_train: used to build empirical pmf
        data_test: used to evaluate the posterior prob.
        return: preds (fp(x)/[fp(x)+fu(x)], NTC of being mix, following the convention in DEDPUL) """


    pos_data_train_hist = np.zeros(numChannel) # smoothing
    mix_data_train_hist = np.zeros(numChannel)

    for i in pos_data_train:
        pos_data_train_hist[i] += 1

    pos_data_train_hist = pos_data_train_hist / pos_data_train_hist.sum()

    for i in mix_data_train:
        mix_data_train_hist[i] += 1

    mix_data_train_hist = mix_data_train_hist / mix_data_train_hist.sum()

    preds_hist = mix_data_train_hist / (mix_data_train_hist + pos_data_train_hist + 1e-6)

    preds = preds_hist[data_test] # 1 - prob[pos|x]: follow the convention of DEDPUL paper
    preds = preds.ravel()

    return preds


def estimate_diff_pmf(preds, target, bw_mix=0.05, bw_pos=0.1, kde_mode='logit', threshold=None, k_neighbours=None,
                  tune=False, MT=True, MT_coef=0.2, decay_MT_coef=False, kde_type='kde',
                  n_gauss_mix=20, n_gauss_pos=10, bins_mix=20, bins_pos=20):
    """
    Estimates densities of predictions y(x) for P and U and ratio between them f_p / f_u for U sample;
        uses kernel density estimation (kde);
        post-processes difference of estimated densities - imposes monotonicity on lower preds
        (so that diff is partly non-decreasing) and applies rolling median to further reduce variance
    :param preds: predictions of NTC y(x), probability of belonging to U rather than P, np.array with shape (n,)
    :param target: binary vector, 0 if positive, 1 if unlabeled, np.array with shape (n,)
    :param bw_mix: bandwidth for kde of U
    :param bw_pos: bandwidth for kde of P
    :param kde_mode: 'prob', 'log_prob' or 'logit'; default is 'logit'
    :param monotonicity: monotonicity is imposed on density difference for predictions below this number, float in [0, 1]
    :param k_neighbours: difference is relaxed with median rolling window with size k_neighbours * 2 + 1,
        default = int(preds[target == 1].shape[0] // 10)

    :return: difference of densities f_p / f_u for U sample
    """

    # sorting to relax and impose monotonicity
    sorted_mixed = np.sort(preds[target == 1])

    # # Option1: diff estimated using KDE
    # diff = np.apply_along_axis(lambda x: kde_outer_fun(kde_pos, x) / (kde_outer_fun(kde_mix, x) + 10 ** -5), axis=0,
    #                            arr=sorted_mixed)

    diff = preds[target == 1] / (1 - preds[target == 1])
    diff = diff.ravel()

    # # Option2: diff estimated using NTC directly (notice here NTC is f(x)/xxx, not h(x)/xxx)
    # sorted_mixed = np.clip(sorted_mixed, 0.1, 1) # avoiding f(x)/[h(x)+f(x)] being 0, which in theory is not possible
    # diff = 1/sorted_mixed - 1

    diff[diff > 50] = 50
    diff = rolling_apply(diff, 5)
    diff = np.append(
        np.flip(np.maximum.accumulate(np.flip(diff[sorted_mixed <= threshold], axis=0)), axis=0),
        diff[sorted_mixed > threshold])
    diff = rolling_apply(diff, k_neighbours)

    if MT:
        MTrends = MonotonizingTrends(MT_coef=MT_coef)
        diff = np.flip(np.array(MTrends.monotonize_array(np.flip(diff, axis=0), reset=True, decay_MT_coef=decay_MT_coef)), axis=0)

    diff.sort()
    diff = np.flip(diff, axis=0)

    # desorting
    diff = diff[np.argsort(np.argsort(preds[target == 1]))] 

    # # added by SuMPE author
    # diff = diff / np.mean(diff)

    return diff

def estimate_diff_pmf_direct(pos_data_train, mix_data_train, data_test, numChannel):
    """ 
    :return: difference of densities f_p / f_u for U sample
    """


    pos_data_train_hist = np.zeros(numChannel) # smoothing
    mix_data_train_hist = np.zeros(numChannel)

    for i in pos_data_train:
        pos_data_train_hist[i] += 1

    pos_data_train_hist = pos_data_train_hist / pos_data_train_hist.sum()

    for i in mix_data_train:
        mix_data_train_hist[i] += 1

    mix_data_train_hist = mix_data_train_hist / mix_data_train_hist.sum()

    diff_hist = pos_data_train_hist / (mix_data_train_hist + 1e-6)

    diff =  diff_hist[data_test] # 1 - prob[pos|x]: follow the convention of DEDPUL paper
    diff = diff.ravel()

    return diff

def estimate_max_poster_pmf(p1, pm, kappa):
    """ 
    : return: maximum posterior probability 
      P[X~p1|x] = kappa * p1/(kappa * p1 + (1-kappa) * p2) = kappa * p1 / pm
    """
    post_p1 = kappa * p1 / (pm + 1e-5)
    max_post_p1 = np.max(post_p1)
    index = np.where(post_p1 >= max_post_p1 - 1e-5)
    return max_post_p1, index[0][0]

# =========================================================================
#       How to choose sampling mechanism alpha(x) for different cases:
# 1. G violates irreducibility, and we know the component G1 that violates 
# 2. G violates irreducibility, only know the upper bound of posterior probability
# 3. G violates irreducibility, know nothing else
# 4. G satisfies irreducibility 
# =========================================================================

testing_case = 3 # testing scenario  

print()
print("=======================================================================================")
if testing_case == 1:
    print(" Testing Case 1: G violates irreducibility, and we know the component G1 that violates")
elif testing_case == 2:
    print(" Testing Case 2: G violates irreducibility, know the upper bound of posterior probability")
elif testing_case == 3:
    print(" Testing Case 3: G violates irreducibility, know posterior probability for some data points")

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





# # used for probabilistic sampling
# lower_prob_thresh = -1 # choose this value to be bigger than (notice that the value will be over-written) 
# upper_prob_thresh = 1.0 # not throw away samples that are very likely to be unlabeled -> negative, because that has little use (and throwing away to much is not good)


if subsample_method == 'determinstic':
    # used for determinstic sampling
    # neg_prob_thresh = 0.7 # unlabelled data with P[unlabeled|X=x] bigger than a certain threshold will be thrown away
    pos_prob_thresh = 0.2 # unlabelled data with P[neg|X=x] smaller than a certain threshold will be thrown away    

    print("with cut-off probability = ", pos_prob_thresh)

if subsample_method == 'probabilistic':
    # used for probabilistic sampling
    lower_prob_thresh = 1e-4 # choose this value to be bigger than; previous choice 0.5, not just let 1e-100
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


# ===================================================================================
# data used for testing
# ======================================================

if data_mode == 'Sp2021 NaI':
    numChannel = 1024
    numSimulation = 100

    rangeChannel = range(0, numChannel)

    Cs_allData = []
    Co_allData = []
    BG_allData = []
    BG_Cs_allData = []
    BG_Cs_Co_allData = []

    with open('Nuclear Data/Sp2021 NaI/Cs_137_100.csv', 'r') as Cs_137:
        Cs_137_reader = csv.reader(Cs_137)
        next(Cs_137_reader)
        next(Cs_137_reader)
        for row in Cs_137_reader:
            rowdata = [float(string) for string in row[0:numChannel]]        
            Cs_allData.append(rowdata)


    with open('Nuclear Data/Sp2021 NaI/BG_100.csv', 'r') as BG:
        BG_reader = csv.reader(BG)
        next(BG_reader)
        next(BG_reader)
        for row in BG_reader:
            rowdata = [float(string) for string in row[0:numChannel]]        
            BG_allData.append(rowdata)


    with open('Nuclear Data/Sp2021 NaI/Co_60_100.csv', 'r') as Co:
        Co_reader = csv.reader(Co)
        next(Co_reader)
        next(Co_reader)
        for row in Co_reader:
            rowdata = [float(string) for string in row[0:numChannel]]        
            Co_allData.append(rowdata)

    # Get pmf
    Cs_allData = np.array(Cs_allData)
    Co_allData = np.array(Co_allData)
    BG_allData = np.array(BG_allData)

    Cs_pmf = Cs_allData.sum(axis=0) / Cs_allData.sum()
    Co_pmf = Co_allData.sum(axis=0) / Co_allData.sum()
    BG_pmf = BG_allData.sum(axis=0) / BG_allData.sum()

elif data_mode == 'Su2022 NaI':
    Co_allData = []
    Cs_allData = []
    Na_allData = []
    Ba_allData = []
    Am_allData = []
    Y_allData = []
    BG_allData = []

    with open('Nuclear Data/Su2022 NaI/20220815_NaI_Spectra.csv', 'r') as NaI:
        NaI_reader = csv.reader(NaI)
        next(NaI_reader)
        for col in NaI_reader:
            Co_allData.append(float(col[1]))
            Cs_allData.append(float(col[2]))
            Na_allData.append(float(col[3]))
            Ba_allData.append(float(col[4]))
            Am_allData.append(float(col[5]))
            Y_allData.append(float(col[6]))
            BG_allData.append(float(col[7]))

    numChannel = len(Cs_allData)
    rangeChannel = range(0, numChannel)
    
    # %Get pmf
    Cs_allData = np.array(Cs_allData)
    Co_allData = np.array(Co_allData)
    BG_allData = np.array(BG_allData)

    Cs_pmf = Cs_allData / Cs_allData.sum()
    Co_pmf = Co_allData / Co_allData.sum()
    BG_pmf = BG_allData / BG_allData.sum()



# %% Start testing
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

    p1 = Cs_pmf # H
    
    p2 = (1-gamma) * BG_pmf + gamma * Co_pmf # G
    pm = kappa_star * p1 + (1-kappa_star) * p2

    # useful for hand-craft reducible distributions

    kappa_gh, location_gh = estimate_cons_kappa_nuclear(p2, p1)
    print('(Analytical calculation) Proportion of H in G kappa(G|H) =', kappa_gh, ', achieved at x =', location_gh)

    kappa_max, location_fh = estimate_cons_kappa_nuclear(pm, p1) 
    print('(Analytical calculation) maximum mixture proportion kappa_max =', kappa_max, ', achieved at x =', location_fh) #  this function only returns analytical kappa_max

    post_max_val, post_max_loc = estimate_max_poster_pmf(p1, pm, kappa_star)
    print('(Empirical calculation) maximum posterior prob. of being positive =', post_max_val, ', achieved at x =', post_max_loc)

    kappa_max_list.append(kappa_max)


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



    for i in range(0,numRepeat):
        print()
        print("=============================== Test #", i, " =======================================")
        random.seed(i)

        # print("Number of samples from component = ", pos_size)
        # print("Number of samples from mixture = ", mix_size)
        # print()

        # sampling
        pos_data_test = np.array(choices(rangeChannel, p1, k = pos_size))

        mix_data_test = np.concatenate( (np.array(choices(rangeChannel, p1, k = int(mix_size * kappa_star))), np.array(choices(rangeChannel, p2, k = mix_size - int(mix_size * kappa_star)))), axis = None  )



        data_test = np.append(mix_data_test, pos_data_test).reshape((-1, 1))
        target_test = np.append(np.array([1] * mix_size), np.array([0] * pos_size)) # the label for non-traditional classifier: 1 - unlabeled, 0 - labelled (notice!!!)

        target_test_true = np.append(np.array([0] * int(mix_size * kappa_star)), np.array([1] * int(mix_size * (1 - kappa_star)))) # the true label in mixture distribution: 0 - unlabeled H, 1 - unlabelled G 
        target_test_true = np.append(target_test_true, np.array([2] * pos_size)) # notice the "appending"! Pure positive was labelled as 2.

        # reshape the original mix and pos datasets and do shuffling
        mix_data_test = mix_data_test.reshape([-1, 1])
        pos_data_test = pos_data_test.reshape([-1, 1])

        # The remaining lines are all for shuffling purpose.
        data_test_full = np.concatenate((data_test, target_test.reshape(-1, 1), target_test_true.reshape(-1, 1)), axis=1)
        np.random.shuffle(data_test) 
        target_test = data_test_full[:, 1]
        target_test_true = data_test_full[:, 2]
        data_test = data_test_full[:, 0].reshape(-1, 1)

        # pos and mix data set after shuffling
        mix_data_test = data_test[target_test == 1]
        pos_data_test = data_test[target_test == 0]

        # plot empirical pmf
        mix_data_test_hist = np.zeros(numChannel)
        pos_data_test_hist = np.zeros(numChannel)

        for i in mix_data_test:
            mix_data_test_hist[i] += 1

        mix_data_test_pmf = mix_data_test_hist/mix_size

        for i in pos_data_test:
            pos_data_test_hist[i] += 1

        pos_data_test_pmf = pos_data_test_hist/pos_size

        # ===========================================================================================================================
        #                                           Regrouping
        # ===========================================================================================================================
        

        # use NTC
        # preds = P[Y=unlabelled|x]: bigger preds -> more likely to be mix -> more likely to belong to neg. class G -> throw away
        preds = estimate_preds_pmf(pos_data_test, mix_data_test, data_test, numChannel)

        print('ac', accuracy_score(target_test, preds.round()))
        print('roc', roc_auc_score(target_test, preds))

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
        print("Enlarged proportion in labeled dataset =", additional_proportion)
        print()

        additional_proportion_all.append(additional_proportion)


        # ===========================================================================================================================
        #                                           subsampling
        # ===========================================================================================================================


        # ============================================================================
        #          choosing the sampling scheme for different cases
        # ============================================================================

        

        if testing_case == 1: 
            neg_redu_size = pos_size # choose number of samples from (pure) G1 
            neg_redu_data_test = sampler(mu2, s2, int(neg_redu_size)).reshape((-1, 1))

            data_test_redu = np.copy(data_test) # return a deep copy
            data_test_redu[target_test == 0] = neg_redu_data_test # replace the labeled positive samples (target_test = 0) with samples from G2

            kappa_neg_DEDPUL, poster, _ = estimate_poster_cv(data_test_redu, target_test, estimator='dedpul', alpha=None,
                                                estimate_poster_options={'disp': False},
                                                estimate_diff_options={},
                                                estimate_preds_cv_options={'bn': False, 'l2': 1e-4,
                                                    'cv': 5, 'n_networks': 1, 'lr': 1e-3, 'hid_dim': 16, 'n_hid_layers': 0,
                                                },
                                                train_nn_options={
                                                    'n_epochs': 200, 'batch_size': 64,
                                                    'n_batches': None, 'n_early_stop': 10, 'disp': False,
                                                }
                                            )


            kappa_DEDPUL = 1 - kappa_neg_DEDPUL
            poster_redu = 1 - poster
            # preds_redu = 1 - preds

            poster_redu_expand = np.zeros(len(data_test)) * 100 # notice the zero
            poster_redu_expand[target_test == 1] = poster_redu

            sampling_scheme = 1 - poster_redu_expand

            # print result
            print()
            print("===================================================")
            print("   Case 1: Estimate Posterior for G1                    ")
            print("===================================================")
            print("Number of samples from G1 = ", neg_redu_size)
            print("Number of samples from mixture = ", mix_size)
            print()
            print("(1-kappa_star)*gamma_star =", (1-kappa_star) * gamma_star)
            print()
            print('(1-kappa)*gamma estimate =', kappa_DEDPUL)
            print('Abs. Error |est - star|:', abs(kappa_DEDPUL - (1-kappa_star) * gamma_star) )
            print()

        elif testing_case == 2: # know the maximum P[Y=1|x]  
            sampling_scheme = np.ones(len(data_test)) * 100
            sampling_scheme[target_test == 1] = post_max_val
            
            print()
            print("=============================================================")
            print("    Case 2: know maximum P[Y=1|x] = ", post_max_val)
            print("=============================================================")
            print()

        elif testing_case == 3: # know the P[Y=1|x] 

            # ==========================================
            # practical way: background estimation
            # ==========================================

            # all contribution from F that is out of the support ot H are from G
            neg_contribution_hist = np.zeros(numChannel) 
            countThresh = 5 # a threshold to determine out of support
            neg_contribution_hist[pos_data_test_hist < countThresh] = mix_data_test_hist[pos_data_test_hist < countThresh]

            # select the training data for interpolation
            trainChannel = np.argwhere(neg_contribution_hist > 5).ravel() # take out the *non-zero* channels as training data
            # trainChannel = trainChannel[trainChannel < 300]
            trainHist = neg_contribution_hist[trainChannel]

            f_inter = interp1d(trainChannel, trainHist, kind='linear', fill_value=(trainHist[0]), bounds_error=False)

            # rangeChannel_interp = range(0, np.max(trainChannel)) # this prevents blowing up in small channel index
            rangeChannel_interp = range(np.min(trainChannel), np.max(trainChannel)) # new version, only do linear interpolation to peak region
            neg_contribution_hist[rangeChannel_interp] =  f_inter(rangeChannel_interp)

            poster_neg_hist = neg_contribution_hist / (mix_data_test_hist+1e-6)
            poster_neg_hist[poster_neg_hist > 1] = 1
            poster_pos_standard = 1 - poster_neg_hist[mix_data_test]

            sampling_scheme = np.ones(len(data_test)) * 100
            sampling_scheme[target_test == 1] = poster_pos_standard.ravel()

            print()
            print("=====================================================================")
            print("    Case 3: know maximum P[Y=1|x] for some x, with max =", np.max(poster_pos_standard))
            print("=====================================================================")
            print()



        print("===================================================")
        print("                Subsampling                    ")
        print("===================================================")


        if subsample_method == 'determinstic':
            # unlabelled data with posterior bigger than a certain threshold will be thrown away

            # # Option 1: Subsampling based on P vs. U
            # neg_prob_thresh = 0.8 
            # dropping_mask = np.logical_and(preds > neg_prob_thresh, target_test == 1)
            # subsample_mask = np.logical_not(dropping_mask)

            # Option 2: Subsampling based on P vs. N

            dropping_mask = np.logical_and(sampling_scheme < pos_prob_thresh, target_test == 1)
            subsample_mask = np.logical_not(dropping_mask)
            print("Determinstic Subsampling: Posterior threshold for positive = ", pos_prob_thresh)

        elif subsample_method == 'probabilistic':
            # # Option 1: Subsampling based on P vs. U
            # upper_prob_thresh = 0.8 # not throw away samples that are very likely to be unlabeled -> negative, because that has little use (and throwing away to much is not good)
            # lower_prob_thresh = 0.6 # choose this value to be bigger than 
            # rej_prob = np.random.uniform(lower_prob_thresh, upper_prob_thresh, len(preds))    
            # dropping_mask = np.logical_and(np.logical_and(preds < upper_prob_thresh, preds > rej_prob), target_test == 1)  # notice: logical_and only takes 2 conditions!
            # subsample_mask = np.logical_not(dropping_mask)

            # Option 2: Subsampling based on P vs. N

            # rej_prob = np.random.uniform(lower_prob_thresh, upper_prob_thresh, len(data_test))
            # dropping_mask = np.logical_and(rej_prob > poster_pos_expand, target_test == 1)
            # subsample_mask = np.logical_not(dropping_mask)


            # lower_prob_thresh_ratio = 0.5 # refers to the threshold h(x)/f(x) = 0.5
            # lower_prob_thresh = kappa_DEDPUL * lower_prob_thresh_ratio
            # print("Rejection sampling: Lower probability threshold =", lower_prob_thresh_ratio," * kappa(F|H).")

            region_A = numChannel


            rej_prob = np.random.uniform(0.0, 1.0, len(data_test)) # here, always set as Unif[0, 1]

            if rejsample_method == 'linear':
                dropping_mask = np.logical_and(rej_prob > sampling_scheme, target_test == 1)
            elif rejsample_method == 'quadratic':
                dropping_mask = np.logical_and(rej_prob > (sampling_scheme ** 2), target_test == 1)
            
            dropping_mask = np.logical_and(np.logical_and(sampling_scheme <= upper_prob_thresh, sampling_scheme > lower_prob_thresh), dropping_mask) # prevent extreme cases from being thrown away
            dropping_mask = np.logical_and(data_test.ravel() < region_A, dropping_mask) # only subsample instances x \in supp(source distr.)  
            subsample_mask = np.logical_not(dropping_mask)
            
            print("probabilistic Subsampling: examples with posterior prob. of being positive within range [", lower_prob_thresh, ",", upper_prob_thresh, "] are being sampled")

        elif subsample_method == 'oracle':
            if oracle_option == 1:
                # Option 1: determinstic sampling
                if mu1 >= mu2: 
                    # # this works
                    # rej_region_lower = (mu1+mu2)/2 
                    # rej_region_upper = 100
                    # will this work?
                    rej_region_lower = (mu1+mu2)/2 
                    rej_region_upper = mu1
                else: 
                    # # this works
                    # rej_region_lower = -100
                    # rej_region_upper = (mu1+mu2)/2
                    # will this work?
                    rej_region_lower = mu1
                    rej_region_upper = (mu1+mu2)/2 
                dropping_mask = np.logical_and(target_test_true == 1, np.logical_and(data_test > rej_region_lower, data_test < rej_region_upper ).ravel()  ) # notice: the target_test_true is used intentionally!
                subsample_mask = np.logical_not(dropping_mask)
                print("oracle determinstic sampling: dropping region of samples from G is [", rej_region_lower, rej_region_upper, "]")

            if oracle_option == 2:
                # Option 2: rejection sampling
                # upper_prob_thresh = 1.0 + 1e-4 # choose 0.9 or 1.0

                # choosing threshold for small P[Y=1|x] that we do not subsample
                # lower_prob_thresh_ratio = 0.5 # refers to the threshold h(x)/f(x) = 0.5
                # lower_prob_thresh = kappa_DEDPUL * lower_prob_thresh_ratio
                # print("Rejection sampling: Lower probability threshold =", lower_prob_thresh_ratio," * kappa(F|H). = ", lower_prob_thresh)  
                # lower_prob_thresh = 0.5


                poster_pos_oracle = np.reshape(kappa_star * p1(data_test) / pm(data_test), (-1,))
                
                print("Orcale rejection sampling: Lower probability threshold =", lower_prob_thresh)
                print("Number of data experiencing rej-sampling =", np.sum(poster_pos_oracle > lower_prob_thresh))

                rej_prob = np.random.uniform(0.0, 1.0, len(data_test)) # here, always set as Unif[0, 1]
                
                dropping_mask = np.logical_and(rej_prob > poster_pos_oracle, target_test == 1) # here we use the "optimum empirical posterior estimation" by / kappa_DEDPUL * kappa_star; (poster_pos_expand / kappa_DEDPUL * kappa_star)
                dropping_mask = np.logical_and(np.logical_and(poster_pos_oracle <= upper_prob_thresh, poster_pos_oracle > lower_prob_thresh), dropping_mask) # prevent extreme cases from being thrown away
                subsample_mask = np.logical_not(dropping_mask)


        # subsampling
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
        # ============================================================================================================================
        # choose to turn on or off the display
        # notice the name: here it is "xxx_poster_cv"
        diff = estimate_diff_pmf_direct(pos_data_test, mix_data_test, mix_data_test, numChannel)
        kappa_neg_DEDPUL, poster_neg = estimate_poster_em(diff, mode='dedpul', converge=True, nonconverge=True, 
                                        max_diff=0.05, step=0.0025, alpha_as_mean_poster=True)
        # kappa_neg_DEDPUL, poster, preds = estimate_poster_cv(data_test, target_test, estimator='dedpul', alpha=None,
        #                                         estimate_poster_options={'disp': False},
        #                                         estimate_diff_options={},
        #                                         estimate_preds_cv_options={'bn': False, 'l2': 1e-4,
        #                                             'cv': 5, 'n_networks': 1, 'lr': 1e-3, 'hid_dim': 16, 'n_hid_layers': 0,
        #                                         },
        #                                         train_nn_options={
        #                                             'n_epochs': 200, 'batch_size': 64,
        #                                             'n_batches': None, 'n_early_stop': 10, 'disp': False,
        #                                         }
        #                                     )


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
        diff_regr = estimate_diff_pmf_direct(pos_data_test_regr, mix_data_test_regr, mix_data_test_regr, numChannel)

        kappa_neg_DEDPUL_regr, poster_neg_regr = estimate_poster_em(diff_regr, mode='dedpul', converge=True, nonconverge=True, 
                                        max_diff=0.05, step=0.0025, alpha_as_mean_poster=True)
        
        # kappa_neg_DEDPUL_regr, poster_regr, preds_regr = estimate_poster_cv(data_test_regr, target_test_regr, estimator='dedpul', alpha=None,
        #                                         estimate_poster_options={'disp': False},
        #                                         estimate_diff_options={},
        #                                         estimate_preds_cv_options={'bn': False, 'l2': 1e-4,
        #                                             'cv': 5, 'n_networks': 1, 'lr': 1e-3, 'hid_dim': 16, 'n_hid_layers': 0,
        #                                         },
        #                                         train_nn_options={
        #                                             'n_epochs': 200, 'batch_size': 64,
        #                                             'n_batches': None, 'n_early_stop': 10, 'disp': False,
        #                                         }
        #                                     )

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
        diff_sub = estimate_diff_pmf_direct(pos_data_test_sub, mix_data_test_sub, mix_data_test_sub, numChannel)

        kappa_neg_DEDPUL_sub, poster_neg_sub = estimate_poster_em(diff_sub, mode='dedpul', converge=True, nonconverge=True, 
                                        max_diff=0.05, step=0.0025, alpha_as_mean_poster=True)  
        # kappa_neg_DEDPUL_sub, poster_sub, preds_sub = estimate_poster_cv(data_test_sub, target_test_sub, estimator='dedpul', alpha=None,
        #                                         estimate_poster_options={'disp': False},
        #                                         estimate_diff_options={},
        #                                         estimate_preds_cv_options={'bn': False, 'l2': 1e-4,
        #                                             'cv': 5, 'n_networks': 1, 'lr': 1e-3, 'hid_dim': 16, 'n_hid_layers': 0,
        #                                         },
        #                                         train_nn_options={
        #                                             'n_epochs': 200, 'batch_size': 64,
        #                                             'n_batches': None, 'n_early_stop': 10, 'disp': False,
        #                                         }
        #                                     )

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

        # ============================================================================================================================
        # EN
        # ============================================================================================================================
        preds = estimate_preds_pmf(pos_data_test, mix_data_test, data_test, numChannel)
        EN_neg_kappa, EN_neg_poster = estimate_poster_en(preds, target_test, alpha=None, estimator='e3') # here, 'e3' in the original paper is used
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
        preds_regr = estimate_preds_pmf(pos_data_test_regr, mix_data_test_regr, data_test_regr, numChannel)
        EN_neg_kappa_regr, EN_neg_poster_regr = estimate_poster_en(preds_regr, target_test_regr, alpha=None, estimator='e3') # here, 'e3' in the original paper is used

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
        preds_sub = estimate_preds_pmf(pos_data_test_sub, mix_data_test_sub, data_test_sub, numChannel)
        EN_neg_kappa_sub, EN_neg_poster_sub = estimate_poster_en(preds_sub, target_test_sub, alpha=None, estimator='e3') # here, 'e3' in the orginal paper is used

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
        (kappa_KM1, kappa_KM2) = wrapper_discrete(mix_data_test, pos_data_test, epsilon=0.04, lambda_lower_bound=1.0, lambda_upper_bound=10, method='grad')            
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
        (kappa_KM1_regr, kappa_KM2_regr) = wrapper_discrete(mix_data_test_regr, pos_data_test_regr, epsilon=0.04, lambda_lower_bound=1.0, lambda_upper_bound=10, method='grad')

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
        (kappa_KM1_sub, kappa_KM2_sub) = wrapper_discrete(mix_data_test_sub, pos_data_test_sub, epsilon=0.04, lambda_lower_bound=0.5, lambda_upper_bound=10, method='grad')

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
        tice_alpha = tice_wrapper(data_test, target_test, k=10, n_folds=10, delta=0.2, n_splits=40)
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
        tice_alpha_regr = tice_wrapper(data_test_regr, target_test_regr, k=10, n_folds=10, delta=0.2, n_splits=40)

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
        tice_alpha_sub = tice_wrapper(data_test_sub, target_test_sub, k=10, n_folds=10, delta=0.2, n_splits=40)

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


np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

print("Enlarged proportion in labeled dataset =", mean_additional_proportion_all, "(the ReCPE paper expect negative samples to be regrouped, but I don't think so")
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

print()
print("DEDPUL_rescale   mean:", mean_DEDPUL_rescale_all, " abs. error", err_DEDPUL_rescale_all)
print("EN_prior_rescale mean:", mean_EN_prior_rescale_all, " abs. error", err_EN_prior_rescale_all)
print("EN_post_rescale  mean:", mean_EN_post_rescale_all, " abs. error", err_EN_post_rescale_all)
print("KM1_rescale      mean:", mean_KM1_rescale_all, " abs. error", err_KM1_rescale_all)
print("KM2_rescale      mean:", mean_KM2_rescale_all, " abs. error", err_KM2_rescale_all)
print("tice_rescale     mean:", mean_tice_rescale_all, " abs. error", err_tice_rescale_all)

print()
print("DEDPUL_sub       mean:", mean_DEDPUL_sub_all, " abs. error", err_DEDPUL_sub_all)
print("EN_prior_sub     mean:", mean_EN_prior_sub_all, " abs. error", err_EN_prior_sub_all)
print("EN_post_sub      mean:", mean_EN_post_sub_all, " abs. error", err_EN_post_sub_all)
print("KM1_sub          mean:", mean_KM1_sub_all, " abs. error", err_KM1_sub_all)
print("KM2_sub          mean:", mean_KM2_sub_all, " abs. error", err_KM2_sub_all)
print("tice_sub         mean:", mean_tice_sub_all, " abs. error", err_tice_sub_all)


print("\nCompare of original method with its subsampling version")
print("DEDPUL           mean:", mean_DEDPUL_all, " abs. difference", diff_DEDPUL_all, " abs. error", err_DEDPUL_all)
print("DEDPUL_regr      mean:", mean_DEDPUL_regr_all, " abs. difference", diff_DEDPUL_regr_all, " abs. error", err_DEDPUL_regr_all)
print("DEDPUL_rescale   mean:", mean_DEDPUL_rescale_all, " abs. difference", diff_DEDPUL_rescale_all, " abs. error", err_DEDPUL_rescale_all)
print("DEDPUL_sub       mean:", mean_DEDPUL_sub_all, " abs. difference", diff_DEDPUL_sub_all, " abs. error", err_DEDPUL_sub_all)
print()
print("EN_prior         mean:", mean_EN_prior_all, " abs. difference", diff_EN_prior_all, " abs. error", err_EN_prior_all)
print("EN_prior_regr    mean:", mean_EN_prior_regr_all, " abs. difference", diff_EN_prior_regr_all, " abs. error", err_EN_prior_regr_all)
print("EN_prior_rescale mean:", mean_EN_prior_rescale_all, " abs. difference", diff_EN_prior_rescale_all, " abs. error", err_EN_prior_rescale_all)
print("EN_prior_sub     mean:", mean_EN_prior_sub_all, " abs. difference", diff_EN_prior_sub_all, " abs. error", err_EN_prior_sub_all)
print()
print("EN_post          mean:", mean_EN_post_all, " abs. difference", diff_EN_post_all, " abs. error", err_EN_post_all)
print("EN_post_regr     mean:", mean_EN_post_regr_all, " abs. difference", diff_EN_post_regr_all, " abs. error", err_EN_post_regr_all)
print("EN_post_rescale  mean:", mean_EN_post_rescale_all, " abs. difference", diff_EN_post_rescale_all, " abs. error", err_EN_post_rescale_all)
print("EN_post_sub      mean:", mean_EN_post_sub_all, " abs. difference", diff_EN_post_sub_all, " abs. error", err_EN_post_sub_all)
print()
print("KM1              mean:", mean_KM1_all, " abs. difference", diff_KM1_all, " abs. error", err_KM1_all)
print("KM1_regr         mean:", mean_KM1_regr_all, " abs. difference", diff_KM1_regr_all, " abs. error", err_KM1_regr_all)
print("KM1_rescale      mean:", mean_KM1_rescale_all, " abs. difference", diff_KM1_rescale_all, " abs. error", err_KM1_rescale_all)
print("KM1_sub          mean:", mean_KM1_sub_all, " abs. difference", diff_KM1_sub_all, " abs. error", err_KM1_sub_all)
print()
print("KM2              mean:", mean_KM2_all, " abs. difference", diff_KM2_all, " abs. error", err_KM2_all)
print("KM2_regr         mean:", mean_KM2_regr_all, " abs. difference", diff_KM2_regr_all, " abs. error", err_KM2_regr_all)
print("KM2_rescale      mean:", mean_KM2_rescale_all, " abs. difference", diff_KM2_rescale_all, " abs. error", err_KM2_rescale_all)
print("KM2_sub          mean:", mean_KM2_sub_all, " abs. difference", diff_KM2_sub_all, " abs. error", err_KM2_sub_all)
print()
print("tice             mean:", mean_tice_all, " abs. difference", diff_tice_all, " abs. error", err_tice_all)
print("tice_regr        mean:", mean_tice_regr_all, " abs. difference", diff_tice_regr_all, " abs. error", err_tice_regr_all)
print("tice_rescale     mean:", mean_tice_rescale_all, " abs. difference", diff_tice_rescale_all, " abs. error", err_tice_rescale_all)
print("tice_sub         mean:", mean_tice_sub_all, " abs. difference", diff_tice_sub_all, " abs. error", err_tice_sub_all)
# print()



# %%
