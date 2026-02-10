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

    CORRECTNESS = np.load("data/PARTIAL_CREDIT/CORRECTNESS.npy")
    TTD = np.load("data/PARTIAL_CREDIT/TTD.npy")
    MEANS = np.load("data/PARTIAL_CREDIT/MEANS.npy")

    n_runs = MEANS.shape[0]

    print()
    print(f"Mean (std) of SAVI TTD: {np.mean(TTD[:, 0]):0.2f} ({np.std(TTD[:, 0]):0.2f})")
    print(f"Mean (std) of PC NSM TTD: {np.mean(TTD[:, 1]):0.2f} ({np.std(TTD[:, 1]):0.2f})")
    print(f"Mean (std) of CONT NSM TTD: {np.mean(TTD[:, 2]):0.2f} ({np.std(TTD[:, 2]):0.2f})")
    
    print()
    print(f"Mean of SAVI POWER: {np.mean(CORRECTNESS[:, 0]):0.3f}")
    print(f"Mean of PC NSM POWER: {np.mean(CORRECTNESS[:, 1]):0.3f}")
    print(f"Mean of CONT NSM POWER: {np.mean(CORRECTNESS[:, 2]):0.3f}")