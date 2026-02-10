import numpy as np
import os 
import sys
add_path = os.getcwd()
sys.path.append(add_path)

from sequentialized_barnard_tests.base import Decision, Hypothesis
from nscore.nonparametric_nsm import ContinuousNsmTest

if __name__ == "__main__":
    
    # Save the data
    EXPECTED_VALUES = np.load("data/DENSITY/EV.npy")
    CORRECTNESS = np.load("data/DENSITY/CORRECTNESS.npy")
    TTD = np.load("data/DENSITY/TTD.npy")

    # Print some useful info
    print("Number of runs: ", CORRECTNESS.shape[0])
    print("N_max: ", np.max(TTD))
    print()
    print("NSM Approach: ")
    print(f"Empirical Power: {100. * np.mean(CORRECTNESS[:, 0])}")
    print(f"Empirical TTD: {np.mean(TTD[:, 0])}")
    print(f"Empirical TTD (std): {np.std(TTD[:, 0])}")
    print()
    print("WSR Approach: ")
    print(f"Empirical Power: {100. * np.mean(CORRECTNESS[:, 1])}")
    print(f"Empirical TTD: {np.mean(TTD[:, 1])}")
    print(f"Empirical TTD (std): {np.std(TTD[:, 1])}")
    print()
    print(f"NSM vs WSR savings (%): {100. * (np.mean(TTD[:, 1]) - np.mean(TTD[:, 0])) / (np.mean(TTD[:, 1]))}")
    print()

    nsm_test = ContinuousNsmTest(alternative=Hypothesis.P0LessThanP1, alpha=0.05, c=np.arange(21)/20)
    n_max = int(np.max(TTD))
    n_runs = CORRECTNESS.shape[0]
    np.random.seed(31415)
    result_idx = np.random.choice(n_runs, size=n_runs, replace=False)
    data0 = TTD[result_idx[:n_runs // 2], 0] / n_max
    data1 = TTD[result_idx[(n_runs // 2):], 1] / n_max

    print("Mean of NSM subsample: ", np.mean(data0)*n_max)
    print("Mean of WSR subsample: ", np.mean(data1)*n_max)
    breakpoint()

    nsm_result = nsm_test.run_on_sequence(data0, data1)
    print("Decision: ", nsm_result.decision)
    print("Time of Decision: ", nsm_result.info["Time"])