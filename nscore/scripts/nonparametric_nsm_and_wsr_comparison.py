import argparse
import copy

import numpy as np
from tqdm import tqdm

from sequentialized_barnard_tests.base import Decision, Hypothesis
from sequentialized_barnard_tests.wsr import WsrComparisonTest
from sequentialized_barnard_tests.nonparametric_nsm import ContinuousNsmTest
from matplotlib import pyplot as plt

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "This script runs a meta-evaluation of the feedback-based NSM procedure."
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
        default=2000,
        help=(
            "Maximum amount of data to evaluate power and FPR per sequence. "
            "Defaults to 2000."
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
        "-nta",
        "--n_trials_alt",
        type=int,
        default=1000,
        help=("Number of trials to run per alternative." "Defaults to 1000."),
    )
    parser.add_argument(
        "-ntn",
        "--n_trials_null",
        type=int,
        default=100,
        help=("Number of trials to run per null." "Defaults to 100."),
    )

    args = parser.parse_args()
    np.random.seed(args.seed)

    N_TRIALS_ALT = args.n_trials_alt
    N_TRIALS_NULL = args.n_trials_null

    N = args.n_max
    K = 7
    alpha = args.alpha
    
    c = (np.cumsum(np.ones(K)) - 1) / (K - 1)

    c_continuous_10 = (np.cumsum(np.ones(10)) - 1) / (10 - 1)
    c_wsr = 0.95

    nonparametric_nsm_test_10 = ContinuousNsmTest(alternative=Hypothesis.P0LessThanP1, alpha=alpha, c=c_continuous_10)
    wsr_test = WsrComparisonTest(alternative=Hypothesis.P0LessThanP1, alpha=alpha, c_wsr=c_wsr)

    p0 = 0.3
    p1 = 0.7
    null_case = False
    if np.isclose(p0, p1):
        null_case = True

    if null_case or p1 <= 1.0:
        if null_case:
            N_TRIALS = N_TRIALS_NULL
        else:
            N_TRIALS = N_TRIALS_ALT

        TIMES_TO_DECISION_C_NSM_10 = np.zeros(N_TRIALS)
        TIMES_TO_DECISION_WSR = np.zeros(N_TRIALS)

        LAMBDA_NSM_C_10 = np.zeros((N_TRIALS, N))

        # MUST INCLUDE: AVERAGE TTD + power
        RESULTS = np.zeros((3, 2))
        RESULTS[0, :2] = np.array([p0, p1])
        # Begin the trials for particular (p0, p1)
        for kk in tqdm(range(N_TRIALS)):
            if True:
                data0 = np.random.binomial(1, p0, N)
                data1 = np.random.binomial(1, p1, N)
            else:
                data0 = np.random.normal(0.35, 0.35, size=N)
                data1 = np.random.normal(0.65, 0.35, size=N)
                for i in range(N):
                    while data0[i] > 1. or data0[i] < 0.:
                        data0[i] = np.random.normal(0.35, 0.35, size=1)[0]
                    
                    while data1[i] > 1. or data1[i] < 0.:
                        data1[i] = np.random.normal(0.65, 0.35, size=1)[0]
            
            c_nsm_result_10 = nonparametric_nsm_test_10.run_on_sequence(data0, data1)
            lambdas_nsm_c_10 = np.array(nonparametric_nsm_test_10._store_lambda_params).reshape(-1,)
            LAMBDA_NSM_C_10[kk, :lambdas_nsm_c_10.shape[0]] = copy.deepcopy(lambdas_nsm_c_10)
            if c_nsm_result_10.decision == Decision.AcceptAlternative:
                TIMES_TO_DECISION_C_NSM_10[kk] = c_nsm_result_10.info["Time"]
            else:
                TIMES_TO_DECISION_C_NSM_10[kk] = N + 1
            
            wsr_result = wsr_test.run_on_sequence(data0, data1)
            if wsr_result.decision == Decision.AcceptAlternative:
                TIMES_TO_DECISION_WSR[kk] = wsr_result.info["Time"]
            else:
                TIMES_TO_DECISION_WSR[kk] = N + 1

        tmp_ttd_c_nsm_10 = np.sort(TIMES_TO_DECISION_C_NSM_10)
        tmp_ttd_wsr = np.sort(TIMES_TO_DECISION_WSR)

        # Store summary statistics
        RESULTS[1, 0] = np.mean(TIMES_TO_DECISION_C_NSM_10)
        RESULTS[1, 1] = np.mean(TIMES_TO_DECISION_WSR)

        tmp_c_nsm_10 = np.argwhere(TIMES_TO_DECISION_C_NSM_10 < N + 0.5)
        tmp_wsr = np.argwhere(TIMES_TO_DECISION_WSR < N + 0.5)

        RESULTS[2, 0] = tmp_c_nsm_10.shape[0] / N_TRIALS
        RESULTS[2, 1] = tmp_wsr.shape[0] / N_TRIALS

        print("(p0, p1): ")
        print(RESULTS[0, :2])
        print()
        print("Times to decision (avg): ")
        print("NSM (10): ", RESULTS[1, 0])
        print("WSR: ", RESULTS[1, 1])
        print()
        print("Power (avg): ")
        print("NSM (10): ", RESULTS[2, 0])
        print("WSR: ", RESULTS[2, 1])

        print()

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(np.arange(N), np.mean(LAMBDA_NSM_C_10, axis=0), 'r')
        ax.plot(np.arange(N), np.mean(LAMBDA_NSM_C_10, axis=0)+np.std(LAMBDA_NSM_C_10, axis=0), 'r--')
        ax.set_xlim([0, 200])
        fig.savefig("tmp_nonparametric_comparison.png")