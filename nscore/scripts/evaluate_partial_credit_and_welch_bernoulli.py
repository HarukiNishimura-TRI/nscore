import numpy as np 
from scipy import stats
import copy 
from tqdm import tqdm
import argparse

from sequentialized_barnard_tests.base import Hypothesis, Decision
from sequentialized_barnard_tests.nsm import MirroredPartialCreditNsmTest

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description=(
            "This script runs a meta-comparison of the feedback-based NSM procedure with Welch's t-test."
        )
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help=("Random seed assignment for repeatability. " "Defaults to 42."),
    )
    parser.add_argument(
        "-n",
        "--n_max",
        type=int,
        default=500,
        help=(
            "Maximum amount of data to evaluate power and FPR per sequence. "
            "Defaults to 500."
        ),
    )
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=0.05,
        help=("Tunable false positive rate; must lie in (0., 1.). " 
              "Defaults to 0.05."
        ),
    )
    parser.add_argument(
        "-g",
        "--gap",
        type=float,
        default=0.2,
        help=("Nominal gap in mean performance between policies. Float. " 
              "Must lie in (0., 1.). Defaults to 0.2."
        ),
    )
    parser.add_argument(
        "-nc",
        "--num_cases",
        type=int,
        default=50,
        help=("Number of different generating cases to sample" 
              "Defaults to 50"
        ),
    )
    parser.add_argument(
        "-nt",
        "--num_trials",
        type=int,
        default=100,
        help=("Number of evaluation sequences (trial procedures) per generating case" 
              "Defaults to 100"
        ),
    )

    args = parser.parse_args()
    np.random.seed(args.seed)

    n_outcomes = 6 
    
    c = np.arange(n_outcomes) / (n_outcomes - 1.)
    alpha = args.alpha
    n_cases = args.num_cases
    n_trials_per_case = args.num_trials
    n_max = args.n_max

    pcnsm_test = MirroredPartialCreditNsmTest(Hypothesis.P0LessThanP1, alpha=alpha, c=c)

    P0 = np.zeros((n_cases, n_outcomes))
    P1 = np.zeros((n_cases, n_outcomes))

    MU_0 = np.zeros(n_cases)
    MU_1 = np.zeros(n_cases)

    AVG_TTD = np.zeros((n_cases, 3))
    AVG_POWER = np.zeros((n_cases, 3))
    AVG_FPR = np.zeros((n_cases, 3))

    starting_point = 10

    for i in tqdm(range(n_cases)):
        mu_1 = np.random.rand(1).astype(float)*(1. - args.gap) + args.gap
        mu_0 = mu_1 - args.gap

        MU_0[i] = mu_0[0]
        MU_1[i] = mu_1[0]

        p0 = np.zeros(n_outcomes)
        p1 = np.zeros(n_outcomes)
        p0[-1] = mu_0[0]
        p0[0] = 1. - mu_0[0]
        p1[-1] = mu_1[0]
        p1[0] = 1. - mu_1[0]
        
        p1 *= (1. / np.sum(p1))
        p0 *= (1. / np.sum(p0))

        P0[i, :] = copy.deepcopy(p0)
        P1[i, :] = copy.deepcopy(p1)

        TTD_Welch = np.zeros(n_trials_per_case)
        TTD_Welch_bonferroni = np.zeros(n_trials_per_case)
        TTD_NSM = np.zeros(n_trials_per_case)

        POWER_Welch = np.zeros(n_trials_per_case)
        POWER_Welch_bonferroni = np.zeros(n_trials_per_case)
        POWER_NSM = np.zeros(n_trials_per_case)

        FPR_Welch = np.zeros(n_trials_per_case)
        FPR_Welch_bonferroni = np.zeros(n_trials_per_case)
        FPR_NSM = np.zeros(n_trials_per_case)

        for j in range(n_trials_per_case):
            data0 = np.random.multinomial(1, pvals=p0, size=n_max)
            data0_pcnsm = copy.deepcopy(np.argmax(data0, axis=1))
            data0_standard = np.zeros(n_max)

            data0prime = np.random.multinomial(1, pvals=p0, size=n_max)
            data0prime_pcnsm = copy.deepcopy(np.argmax(data0prime, axis=1))
            data0prime_standard = np.zeros(n_max)
            
            data1 = np.random.multinomial(1, pvals=p1, size=n_max)
            data1_pcnsm = copy.deepcopy(np.argmax(data1, axis=1))
            data1_standard = np.zeros(n_max)
            for k in range(n_max):
                data0_standard[k] = c[data0_pcnsm[k]]
                data1_standard[k] = c[data1_pcnsm[k]]
                data0prime_standard[k] = c[data0prime_pcnsm[k]]
            
            data0_standard = copy.deepcopy(data0_standard)
            data1_standard = copy.deepcopy(data1_standard)
            data0prime_standard = copy.deepcopy(data0prime_standard)

            k = starting_point
            while k <= n_max and POWER_Welch[j] <= 0.5:
                _, p_value_welch = stats.ttest_ind(data0_standard[:k], data1_standard[:k], equal_var=False)
                if p_value_welch <= alpha:
                    TTD_Welch[j] = copy.deepcopy(k)
                    POWER_Welch[j] = 1.
                elif k == n_max:
                    TTD_Welch[j] = n_max + 1
                
                k += 1

            k = starting_point
            while k <= n_max and POWER_Welch_bonferroni[j] <= 0.5:
                _, p_value_welch_bonferroni = stats.ttest_ind(data0_standard[:k], data1_standard[:k], equal_var=False)
                if p_value_welch_bonferroni <= (alpha)/n_max:
                    TTD_Welch_bonferroni[j] = copy.deepcopy(k)
                    POWER_Welch_bonferroni[j] = 1.
                elif k == n_max:
                    TTD_Welch_bonferroni[j] = n_max + 1
                
                k += 1

            k = starting_point
            while k <= n_max and FPR_Welch[j] <= 0.5:
                _, fpr_p_value_welch = stats.ttest_ind(data0_standard[:k], data0prime_standard[:k], equal_var=False)
                if fpr_p_value_welch <= alpha:
                    FPR_Welch[j] = 1.
                
                k += 1

            k = starting_point
            while k <= n_max and FPR_Welch_bonferroni[j] <= 0.5:
                _, fpr_p_value_welch_bonferroni = stats.ttest_ind(data0_standard[:k], data0prime_standard[:k], equal_var=False)
                if fpr_p_value_welch_bonferroni <= alpha/n_max:
                    FPR_Welch_bonferroni[j] = 1.
                
                k += 1
            
            nsm_result = pcnsm_test.run_on_sequence(data0_pcnsm, data1_pcnsm)
            if nsm_result.decision is Decision.AcceptAlternative:
                POWER_NSM[j] = 1.
                TTD_NSM[j] = nsm_result.info["result_for_alternative"].info["Time"]
            else:
                TTD_NSM[j] = n_max + 1
            
            fpr_nsm_result = pcnsm_test.run_on_sequence(data0_pcnsm, data0prime_pcnsm)
            if fpr_nsm_result.decision is Decision.AcceptAlternative:
                FPR_NSM[j] = 1.

        AVG_TTD[i, 0] = np.mean(TTD_Welch)
        AVG_TTD[i, 1] = np.mean(TTD_Welch_bonferroni)
        AVG_TTD[i, 2] = np.mean(TTD_NSM)

        AVG_POWER[i, 0] = np.mean(POWER_Welch)
        AVG_POWER[i, 1] = np.mean(POWER_Welch_bonferroni)
        AVG_POWER[i, 2] = np.mean(POWER_NSM)

        AVG_FPR[i, 0] = np.mean(FPR_Welch)
        AVG_FPR[i, 1] = np.mean(FPR_Welch_bonferroni)
        AVG_FPR[i, 2] = np.mean(FPR_NSM)

    
    # Save off all of the data
    try:
        np.save("data/Welch/P0.npy", P0)
        np.save("data/Welch/P1.npy", P1)
        np.save("data/Welch/MU_0.npy", MU_0)
        np.save("data/Welch/MU_1.npy", MU_1)
        np.save("data/Welch/AVG_TTD.npy", AVG_TTD)
        np.save("data/Welch/AVG_POWER.npy", AVG_POWER)
        np.save("data/Welch/AVG_FPR.npy", AVG_FPR)
    except:
        print("Failed to save to the intended location. Please save the data manually")
        breakpoint()