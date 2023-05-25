# %%
"""
SuMPE author's efficient implementation of 1D discrete random version of KM, can be used for nuclear data

DO MIXTURE PROPORTION ESTIMATION 
Using gradient thresholding of the $\C_S$-distance
"""
from cvxopt import matrix, solvers, spmatrix
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
import scipy.linalg as scilin


def find_nearest_valid_distribution(u_alpha, kernel, initial=None, reg=0):
    """ (solution,distance_sqd)=find_nearest_valid_distribution(u_alpha,kernel):
    Given a n-vector u_alpha summing to 1, with negative terms, 
    finds the distance (squared) to the nearest n-vector summing to 1, 
    with non-neg terms. Distance calculated using nxn matrix kernel. 
    Regularization parameter reg -- 

    min_v (u_alpha - v)^\top K (u_alpha - v) + reg* v^\top v"""

    P = matrix(2 * kernel)
    n = kernel.shape[0]
    q = matrix(np.dot(-2 * kernel, u_alpha))
    A = matrix(np.ones((1, n)))
    b = matrix(1.)
    G = spmatrix(-1., range(n), range(n))
    h = matrix(np.zeros(n))
    dims = {'l': n, 'q': [], 's': []}
    solvers.options['show_progress'] = False
    solution = solvers.coneqp(
        P,
        q,
        G,
        h,
        dims,
        A,
        b,
        initvals=initial
        )
    distance_sqd = solution['primal objective'] + np.dot(u_alpha.T,
            np.dot(kernel, u_alpha))[0, 0]
    return (solution, distance_sqd)


def get_distance_curve(
    kernel,
    lambda_values,
    X_mix_hist, X_comp_hist,
    N,
    M = None    
    ):
    """ Given number of elements per class, full kernel (with first N rows corr.
    to mixture and the last M rows corr. to component, and set of lambda values
    compute $\hat d(\lambda)$ for those values of lambda"""

    d_lambda = []
    if M == None:
        M = kernel.shape[0] - N
    prev_soln=None    
    for lambda_value in lambda_values:
        # u_lambda = lambda_value / N * np.concatenate((np.ones((N, 1)),
        #         np.zeros((M, 1)))) + (1 - lambda_value) / M \
        #     * np.concatenate((np.zeros((N, 1)), np.ones((M, 1))))
        u_lambda = lambda_value * np.concatenate((X_mix_hist,
                np.zeros((M, 1)))) + (1 - lambda_value) \
            * np.concatenate((np.zeros((N, 1)), X_comp_hist))  # don't divide by M or N!
               
        (solution, distance_sqd) = \
            find_nearest_valid_distribution(u_lambda, kernel, initial=prev_soln)
        prev_soln = solution
        d_lambda.append(sqrt(np.maximum(distance_sqd, 0)))
    d_lambda = np.array(d_lambda)
    return d_lambda





def compute_best_rbf_kernel_width(X_mixture_sub, X_mix_val, X_mix_hist, X_component_sub, X_comp_val, X_comp_hist):    

    # compute the median -> select a range of kernels
    N=X_mixture_sub.shape[0]
    M=X_component_sub.shape[0]
    # compute median of pairwise distances
    X=np.concatenate((X_mixture_sub,X_component_sub))
    dot_prod_matrix=np.dot(X,X.T)
    norms_squared=sum(np.multiply(X,X).T)    
    distance_sqd_matrix=np.tile(norms_squared,(N+M,1)) + \
        np.tile(norms_squared,(N+M,1)).T - 2*dot_prod_matrix            
    kernel_width_median = sqrt(np.median(distance_sqd_matrix))
    kernel_width_vals= np.logspace(-1,1,21) * kernel_width_median    
    
    # calcute the distance using histogram
    N=X_mix_val.shape[0]
    M=X_comp_val.shape[0]
    # compute median of pairwise distances
    X=np.concatenate((X_mix_val,X_comp_val))
    dot_prod_matrix=np.dot(X,X.T)
    norms_squared=sum(np.multiply(X,X).T)    
    distance_sqd_matrix=np.tile(norms_squared,(N+M,1)) + \
        np.tile(norms_squared,(N+M,1)).T - 2*dot_prod_matrix            
    
    
   
    # Find best kernel width
        
    max_dist_RKHS=0
    best_kernel_width=0
    for kernel_width in kernel_width_vals: 
        kernel=np.exp(-distance_sqd_matrix/(2.*kernel_width**2.))        
        # dist_diff = np.concatenate((np.ones((N, 1)) / N, 
        #                             -1 * np.ones((M,1)) / M))
        dist_diff = np.concatenate((X_mix_hist, -1 * X_comp_hist))
        distribution_RKHS_distance = sqrt(np.dot(dist_diff.T, 
                                        np.dot(kernel, dist_diff))[0,0])
        if distribution_RKHS_distance > max_dist_RKHS:
            max_dist_RKHS=distribution_RKHS_distance
            best_kernel_width=kernel_width                
    kernel=np.exp(-distance_sqd_matrix/(2.*best_kernel_width**2.))
    return best_kernel_width,kernel


def mpe(kernel,N,M, X_mix_hist, X_comp_hist, nu, epsilon=0.04, lambda_lower_bound=1., lambda_upper_bound=8., method='grad'):
    """ Do mixture proportion estimation (as in paper)for N  points from  
    mixture F and M points from component H, given kernel of size (N+M)x(N+M), 
    with first N points from  the mixture  and last M points from 
    the component, and return estimate of lambda_star where
    G =lambda_star*F + (1-lambda_star)*H"""

#    dist_diff = np.concatenate((np.ones((N, 1)) / N, -1 * np.ones((M,1)) / M))
    dist_diff = np.concatenate((X_mix_hist, -1 * X_comp_hist))

    distribution_RKHS_distance = sqrt(np.dot(dist_diff.T, 
                                    np.dot(kernel, dist_diff))[0,0])
    lambda_left = lambda_lower_bound
    lambda_right = lambda_upper_bound
            
    while lambda_right-lambda_left>epsilon:

        if method == 'grad':
            lambda_value=(lambda_left+lambda_right)/2.        
            # u_lambda = lambda_value / N * np.concatenate((np.ones((N, 1)),
            #     np.zeros((M, 1)))) + (1 - lambda_value) / M \
            #     * np.concatenate((np.zeros((N, 1)), np.ones((M, 1))))
            u_lambda = lambda_value  * np.concatenate((X_mix_hist,
                    np.zeros((M, 1)))) + (1 - lambda_value) \
                * np.concatenate((np.zeros((N, 1)), X_comp_hist)) 
            
            (solution, distance_sqd) = \
                find_nearest_valid_distribution(u_lambda, kernel)
            d_lambda_1=sqrt(distance_sqd)
            
            lambda_value=(lambda_left+lambda_right)/2. + epsilon/2.        
            # u_lambda = lambda_value / N * np.concatenate((np.ones((N, 1)),
            #     np.zeros((M, 1)))) + (1 - lambda_value) / M \
            #     * np.concatenate((np.zeros((N, 1)), np.ones((M, 1))))
            u_lambda = lambda_value  * np.concatenate((X_mix_hist,
                    np.zeros((M, 1)))) + (1 - lambda_value)  \
                * np.concatenate((np.zeros((N, 1)), X_comp_hist)) 

            (solution, distance_sqd) = \
                find_nearest_valid_distribution(u_lambda, kernel)
            d_lambda_2=sqrt(distance_sqd)
                
            slope_lambda=(d_lambda_2 - d_lambda_1)*2./epsilon                    
        
            if np.abs(slope_lambda) > nu*distribution_RKHS_distance: # SuMPE author: added abs, slope may be negative
                lambda_right=(lambda_left+lambda_right)/2.
            else:
                lambda_left=(lambda_left+lambda_right)/2.
        elif method == 'value':
            lambda_value=(lambda_left+lambda_right)/2.        
            # u_lambda = lambda_value / N * np.concatenate((np.ones((N, 1)),
            #     np.zeros((M, 1)))) + (1 - lambda_value) / M \
            #     * np.concatenate((np.zeros((N, 1)), np.ones((M, 1))))
            u_lambda = lambda_value  * np.concatenate((X_mix_hist,
                    np.zeros((M, 1)))) + (1 - lambda_value) \
                * np.concatenate((np.zeros((N, 1)), X_comp_hist)) 
            
            (solution, distance_sqd) = \
                find_nearest_valid_distribution(u_lambda, kernel)
            d_lambda_1=sqrt(distance_sqd)
            
            lambda_value=(lambda_left+lambda_right)/2. + epsilon/2.        
            # u_lambda = lambda_value / N * np.concatenate((np.ones((N, 1)),
            #     np.zeros((M, 1)))) + (1 - lambda_value) / M \
            #     * np.concatenate((np.zeros((N, 1)), np.ones((M, 1))))
            u_lambda = lambda_value  * np.concatenate((X_mix_hist,
                    np.zeros((M, 1)))) + (1 - lambda_value)  \
                * np.concatenate((np.zeros((N, 1)), X_comp_hist)) 

            (solution, distance_sqd) = \
                find_nearest_valid_distribution(u_lambda, kernel)
            d_lambda_2=sqrt(distance_sqd)

            sum_lambda = d_lambda_2 + d_lambda_1
                                              
        
            if sum_lambda > 2e-4:
                lambda_right=(lambda_left+lambda_right)/2.
            else:
                lambda_left=(lambda_left+lambda_right)/2.
            
    return (lambda_left+lambda_right)/2.
                                        
def wrapper_discrete(X_mixture, X_component, **kwargs):                 
    """ Takes in 2 arrays containing the mixture and component data as 
    numpy arrays, and prints the estimate of kappastars using the two gradient 
    thresholds as detailed in the paper as KM1 and KM2"""
    
    # Preprocessing: calculate histogram from samples

    X_comp_val, X_comp_cnt = np.unique(X_component, return_counts=True) # get the "bins" and "histogram" from samples    

    X_comp_hist = X_comp_cnt / X_component.shape[0] # normalize the counts to probability

    X_comp_val = np.reshape(X_comp_val, (-1, 1))
    X_comp_hist = np.reshape(X_comp_hist, (-1, 1))

    X_mix_val, X_mix_cnt = np.unique(X_mixture, return_counts=True) # get the "bins" and "histogram" from samples
    X_mix_hist = X_mix_cnt / X_mixture.shape[0] # normalize the counts to probability
    X_mix_val = np.reshape(X_mix_val, (-1, 1))
    X_mix_hist = np.reshape(X_mix_hist, (-1, 1))

    N=X_mix_hist.shape[0]                                                                
    M=X_comp_hist.shape[0]  

    # use sub-samples (400, or 800) to estimate best median kernel width
    X_component_sub = np.random.choice(X_component.ravel(), 1200, replace=False)
    X_component_sub = np.reshape(X_component_sub, (-1, 1))

    X_mixture_sub = np.random.choice(X_mixture.ravel(), 1200, replace=False) 
    X_mixture_sub = np.reshape(X_mixture_sub, (-1, 1))


    # if X_mixture.shape[0] > 1000:
    #     X_mixture_sub = np.random.choice(X_mixture.ravel(), 1000, replace=False) 
    #     X_mixture_sub = np.reshape(X_mixture_sub, (-1, 1))
    # else:
    #     X_mixture_sub = X_mixture

    best_width,kernel=compute_best_rbf_kernel_width(X_mixture_sub, X_mix_val, X_mix_hist, X_component_sub, X_comp_val, X_comp_hist)      

    lambda_values=np.array([1.00,1.05])                                
    dists=get_distance_curve(kernel,lambda_values,X_mix_hist, X_comp_hist, N=N,M=M)
    begin_slope=(dists[1]-dists[0])/(lambda_values[1]-lambda_values[0])
#    dist_diff = np.concatenate((np.ones((N, 1)) / N, -1 * np.ones((M,1)) / M))
    dist_diff = np.concatenate((X_mix_hist, -1 * X_comp_hist))

    # SuMPE author: in paper, this is KM2
    distribution_RKHS_dist = sqrt(np.dot(dist_diff.T, np.dot(kernel, dist_diff))[0,0])
    thres_par=0.2    
    nu1=(1-thres_par)*begin_slope + thres_par*distribution_RKHS_dist
    nu1=nu1/distribution_RKHS_dist                                    
    lambda_star_est_1=mpe(kernel,N,M, X_mix_hist, X_comp_hist, nu=0.1 * nu1, **kwargs)
    kappa_star_est_1=(lambda_star_est_1-1)/lambda_star_est_1  

    # SuMPE author: in paper, this is KM1
    nu2=1/sqrt(np.min([X_mixture.shape[0],X_component.shape[0]])) 
    # nu2=1/sqrt(X_mix_num) #note, here should use X_mix_num, not M!
    nu2=nu2/distribution_RKHS_dist
    if nu2>0.9:
        nu2=nu1
    lambda_star_est_2=mpe(kernel,N,M,X_mix_hist, X_comp_hist,nu=0.1 * nu2, **kwargs)                    
    kappa_star_est_2=(lambda_star_est_2-1)/lambda_star_est_2

    return (kappa_star_est_2,kappa_star_est_1)	# SuMPE author: notice that KM1 and KM2 are swapped previously, now swap back

