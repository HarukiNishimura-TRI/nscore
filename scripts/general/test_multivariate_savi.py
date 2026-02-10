import numpy as np 
import os 
import sys
add_path = os.getcwd()
sys.path.append(add_path)

from nscore.savi import PartialCreditSaviTest
from nscore.nsm import PartialCreditNsmTest
from nscore.nonparametric_nsm import ContinuousNsmTest

from sequentialized_barnard_tests.base import Hypothesis, Decision
from tqdm import tqdm 

if __name__ == "__main__":
    n_runs = 1000

    c = np.arange(6) / 5.
    alpha = 0.05 
    N=1000
    verbose=False 
    MEANS = np.zeros((n_runs, 2))
    TTD = np.zeros((n_runs, 3))
    CORRECTNESS = np.zeros((n_runs, 3))

    for i in tqdm(range(n_runs)):
        pc_nsm_test = PartialCreditNsmTest(Hypothesis.P0LessThanP1, alpha=alpha, c=c)
        pc_savi_test = PartialCreditSaviTest(Hypothesis.P0LessThanP1, alpha=alpha, c=c)
        cont_nsm_test = ContinuousNsmTest(Hypothesis.P0LessThanP1, alpha=alpha, c=np.arange(23)/22.)

        mean0 = 0.
        mean1 = 0.
        while np.abs(mean1 - mean0) <= 0.07:
            dist0 = np.random.rand(6)
            dist0 /= np.sum(dist0)

            dist1 = np.random.rand(6)
            dist1 /= np.sum(dist1)

            mean0 = np.dot(c, dist0)
            mean1 = np.dot(c, dist1)

        if mean1 > mean0:
            null_p = dist0 
            alt_p = dist1 
        else:
            null_p = dist1
            alt_p = dist0 

        null_mean = np.dot(null_p, c)
        alt_mean = np.dot(alt_p, c)

        MEANS[i, 0] = null_mean
        MEANS[i, 1] = alt_mean

        # print("Mean 0: ", null_mean)
        # print("Mean 1: ", alt_mean)

        idx_null = np.random.choice(6, size=N, replace=True, p=null_p)
        idx_alt = np.random.choice(6, size=N, replace=True, p=alt_p)

        score_null = np.zeros(N)
        score_alt = np.zeros(N)
        for j in range(N):
            score_null[j] = c[idx_null[j]]
            score_alt[j] = c[idx_alt[j]]
        
        # print()
        # print("Empirical Mean 0: ", np.mean(score_null))
        # print("Empirical Mean 1: ", np.mean(score_alt))

        cont_nsm_result = cont_nsm_test.run_on_sequence(score_null, score_alt)
        pc_nsm_result = pc_nsm_test.run_on_sequence(idx_null, idx_alt)
        pc_savi_result = pc_savi_test.run_on_sequence(idx_null, idx_alt)

        # print()
        if cont_nsm_result.decision == Decision.AcceptAlternative:
            CORRECTNESS[i, 2] += 1. 
            # print("Nonparametric NSM made the correct decision")
        else:
            pass
            # print("Nonparametric NSM failed to decide")
        
        TTD[i, 2] = cont_nsm_result.info["Time"]
        # print("Time of decision: ", cont_nsm_result.info["Time"])
        
        # print()
        if pc_nsm_result.decision == Decision.AcceptAlternative:
            CORRECTNESS[i, 1] += 1. 
            # print("Partial credit NSM made the correct decision")
        else:
            pass
            # print("Partial credit NSM failed to decide")

        TTD[i, 1] = pc_nsm_result.info["Time"]  
        # print("Time of decision: ", pc_nsm_result.info["Time"])
        
        # print()
        pc_savi_result = pc_savi_test.run_on_sequence(idx_null, idx_alt)
        if pc_savi_result.decision == Decision.AcceptAlternative:
            CORRECTNESS[i, 0] += 1. 
            # print("Partial credit SAVI made the correct decision")
        else:
            pass
            # print("Partial credit SAVI failed to decide")
        
        TTD[i, 0] = pc_savi_result.info["Time"]
        # print("Time of decision: ", pc_savi_result.info["Time"])

    # Order is [PC_SAVI, PC_NSM, CONT_NSM]
    np.save("data/PARTIAL_CREDIT/CORRECTNESS.npy", CORRECTNESS)
    np.save("data/PARTIAL_CREDIT/TTD.npy", TTD)
    np.save("data/PARTIAL_CREDIT/MEANS.npy", MEANS)
    