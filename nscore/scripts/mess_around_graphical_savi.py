import numpy as np 
from scipy.stats import norm 
import copy 

from sequentialized_barnard_tests.nonparametric_nsm import MirroredContinuousNsmTest
from sequentialized_barnard_tests.base import Hypothesis, Decision

from tqdm import tqdm

if __name__ == "__main__":
    
    n_methods = 6
    n_tests = int(n_methods*(n_methods-1) // 2)

    means = 0.5*np.arange(n_methods) + 0.125 + 0.25*np.random.rand(n_methods) - (0.25*n_methods) # Means lie in [-n_methods/4, n_methods/4]
    data_range = n_methods/2. + 1. 

    # p_outside_range_per_datum = 6.66e-16 < 1e-15 (onesided) --> <= 2e-15 (twosided). 
    #      In practice much closer to 1e-15 because clip window always biased one way or the other.
    #
    #  implicit_scale_factor = 1. 

    N = 10000
    # p_outside_range_total = N*n_methods*2e-15

    alpha_desired = 0.05 
    # alpha_effective = alpha_desired - p_outside_range_total

    DATA = np.zeros((N, n_methods))

    for i in tqdm(range(n_methods)):
        for j in range(N):
            DATA[j, i] = 0.5*np.random.standard_normal(1) + means[i]
            while DATA[j, i] <= -(0.5*n_methods + 1.)/2. or DATA[j, i] >= (0.5*n_methods + 1.)/2.:
                DATA[j, i] = np.random.standard_normal(1) + means[i]
            
            # DATA[:, i] = np.clip(np.random.standard_normal(N) + means[i], -n_methods/2 - 8., n_methods/2 + 8.)
    
    # DATA -= (-n_methods/2 - 8.)
    # DATA /= data_range
    DATA += (0.5*n_methods + 1.)/2.
    DATA /= data_range 
    
    assert np.max(DATA) <= 1. 
    assert np.min(DATA) >= 0. 

    RUNNING_MEANS = np.zeros((N, n_methods))
    for i in range(n_methods):
        RUNNING_MEANS[:, i] = np.cumsum(DATA[:, i]) / (1. + np.arange(N))

    P_VALUES = np.ones((N, n_tests))
    IDX = np.zeros((2, n_tests))
    counter = 0
    for i in range(n_methods):
        for j in range(i+1, n_methods):
            IDX[0, counter] = i
            IDX[1, counter] = j

            nsm_test = MirroredContinuousNsmTest(alternative=Hypothesis.P0LessThanP1, alpha=1e-5, c=np.arange(21)/20.)

            for k in tqdm(range(N)):
                result = nsm_test.step(DATA[k, i], DATA[k, j])
                P_VALUES[k, counter] = np.minimum(nsm_test._test_for_alternative._p_value, nsm_test._test_for_null._p_value)
                if P_VALUES[k, counter] <= 1e-5:
                    P_VALUES[k+1:, counter] *= float(P_VALUES[k, counter])
                    break 
            
            counter += 1
    
    RUNNING_MEANS_ARGSORT = np.argsort(RUNNING_MEANS, axis=1)
    
    print(RUNNING_MEANS_ARGSORT.shape)
    print("Mean empirical performance: ", RUNNING_MEANS[-1, :])

    # Now, go through and aggregate empirical p-value cycle
    running_p_value = np.ones(N)
    for k in range(N):
        current_p_value = 0.
        for i in range(n_methods - 1):
            idx0 = np.minimum(RUNNING_MEANS_ARGSORT[k, i], RUNNING_MEANS_ARGSORT[k, i+1])
            idx1 = np.maximum(RUNNING_MEANS_ARGSORT[k, i], RUNNING_MEANS_ARGSORT[k, i+1])
            
            for j in range(n_tests):
                if IDX[0, j] == idx0 and IDX[1, j] == idx1:
                    counter_idx = j
                    break
            # counter_idx = np.argwhere(IDX[0, :]==idx0 and IDX[1, :]==idx1)
            try:
                current_p_value += P_VALUES[k, counter_idx]
            except:
                breakpoint()
        
        running_p_value[k] = float(current_p_value)
    
    critical_alpha_threshold_idx = np.argwhere(running_p_value <= alpha_desired)[0][0]

    print("Time of aggregate decision: ", critical_alpha_threshold_idx)
    print("Cumulative p-values: ", np.sum(P_VALUES[critical_alpha_threshold_idx, :]))
    correct = False 
    if np.isclose(np.linalg.norm(np.arange(n_methods) - np.argsort(RUNNING_MEANS_ARGSORT[critical_alpha_threshold_idx, :])), 0.):
        correct = True
    print("Correct ordering? ", correct)
    print("Running means at time of decision: ", RUNNING_MEANS[critical_alpha_threshold_idx, :])

    




    
