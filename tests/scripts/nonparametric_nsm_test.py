import argparse
import copy

import numpy as np
from tqdm import tqdm

from sequentialized_barnard_tests.base import Decision, Hypothesis
from sequentialized_barnard_tests.savi import SaviTest
from sequentialized_barnard_tests.nsm import BernoulliNsmTest, PartialCreditNsmTest
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
    c_continuous = (np.cumsum(np.ones(101)) - 1) / (101 - 1)

    pc_nsm_test = PartialCreditNsmTest(alternative=Hypothesis.P0LessThanP1, alpha=alpha, c=c)
    nsm_test = BernoulliNsmTest(alternative=Hypothesis.P0LessThanP1, alpha=alpha, c=c)
    savi_test = SaviTest(Hypothesis.P0LessThanP1, alpha=alpha)
    nonparametric_nsm_test = ContinuousNsmTest(alternative=Hypothesis.P0LessThanP1, alpha=alpha, c=c_continuous)

    p0 = 0.3
    # p1 = P1[ii, jj]
    p1 = 0.7
    null_case = False
    if np.isclose(p0, p1):
        null_case = True

    if null_case or p1 <= 1.0:
        if null_case:
            N_TRIALS = N_TRIALS_NULL
        else:
            N_TRIALS = N_TRIALS_ALT

        TIMES_TO_DECISION_NSM = np.zeros(N_TRIALS)
        TIMES_TO_DECISION_PC_NSM = np.zeros(N_TRIALS)
        TIMES_TO_DECISION_SAVI = np.zeros(N_TRIALS)
        TIMES_TO_DECISION_C_NSM = np.zeros(N_TRIALS)

        LAMBDA_NSM = np.zeros((N_TRIALS, N))
        LAMBDA_NSM_PC = np.zeros((N_TRIALS, N))
        LAMBDA_NSM_C = np.zeros((N_TRIALS, N))

        # MUST INCLUDE: AVERAGE TTD + power
        RESULTS = np.zeros((3, 4))
        RESULTS[0, :2] = np.array([p0, p1])
        # Begin the trials for particular (p0, p1)
        for kk in tqdm(range(N_TRIALS)):
            data0 = np.random.binomial(1, p0, N)
            data1 = np.random.binomial(1, p1, N)
            
            data_pc_0 = (K-1)*data0
            data_pc_1 = (K-1)*data1

            # DATA0 = np.zeros((N, K))
            # DATA1 = np.zeros((N, K))

            # tmp0 = np.where(data0 > 0.5, 1.0, 0.0)
            # DATA0[:, 0] = 1.0 - tmp0
            # DATA0[:, -1] = tmp0

            # tmp1 = np.where(data1 > 0.5, 1.0, 0.0)
            # DATA1[:, 0] = 1.0 - tmp1
            # DATA1[:, -1] = tmp1

            savi_result = savi_test.run_on_sequence(data0, data1)
            if savi_result.decision == Decision.AcceptAlternative:
                TIMES_TO_DECISION_SAVI[kk] = savi_result.info["Time"]
            else:
                TIMES_TO_DECISION_SAVI[kk] = N + 1
            
            pc_nsm_result = pc_nsm_test.run_on_sequence(data_pc_0, data_pc_1)
            lambdas_pc_nsm = np.array(pc_nsm_test._store_lambda_params).reshape(-1,)
            try:
                LAMBDA_NSM_PC[kk, :lambdas_pc_nsm.shape[0]] = copy.deepcopy(lambdas_pc_nsm)
            except:
                LAMBDA_NSM_PC[kk, :] = copy.deepcopy(lambdas_pc_nsm[:N])
            
            if pc_nsm_result.decision == Decision.AcceptAlternative:
                TIMES_TO_DECISION_PC_NSM[kk] = pc_nsm_result.info["Time"]
            else:
                TIMES_TO_DECISION_PC_NSM[kk] = N + 1

            nsm_result = nsm_test.run_on_sequence(data0, data1)
            lambdas_nsm = np.array(nsm_test._store_lambda_params).reshape(-1,)
            LAMBDA_NSM[kk, :lambdas_nsm.shape[0]] = copy.deepcopy(lambdas_nsm)
            if nsm_result.decision == Decision.AcceptAlternative:
                TIMES_TO_DECISION_NSM[kk] = nsm_result.info["Time"]
            else:
                TIMES_TO_DECISION_NSM[kk] = N + 1
            
            c_nsm_result = nonparametric_nsm_test.run_on_sequence(data0, data1)
            lambdas_nsm_c = np.array(nonparametric_nsm_test._store_lambda_params).reshape(-1,)
            LAMBDA_NSM_C[kk, :lambdas_nsm_c.shape[0]] = copy.deepcopy(lambdas_nsm_c)
            if nsm_result.decision == Decision.AcceptAlternative:
                TIMES_TO_DECISION_C_NSM[kk] = c_nsm_result.info["Time"]
            else:
                TIMES_TO_DECISION_C_NSM[kk] = N + 1

        tmp_ttd_nsm = np.sort(TIMES_TO_DECISION_NSM)
        tmp_ttd_savi = np.sort(TIMES_TO_DECISION_SAVI)
        tmp_ttd_pc_nsm = np.sort(TIMES_TO_DECISION_PC_NSM)
        tmp_ttd_c_nsm = np.sort(TIMES_TO_DECISION_C_NSM)

        # Store summary statistics
        RESULTS[1, 0] = np.mean(TIMES_TO_DECISION_SAVI)
        RESULTS[1, 1] = np.mean(TIMES_TO_DECISION_NSM)
        RESULTS[1, 2] = np.mean(TIMES_TO_DECISION_PC_NSM)
        RESULTS[1, 3] = np.mean(TIMES_TO_DECISION_C_NSM)

        tmp_savi = np.argwhere(TIMES_TO_DECISION_SAVI < N + 0.5)
        tmp_nsm = np.argwhere(TIMES_TO_DECISION_NSM < N + 0.5)
        tmp_pc_nsm = np.argwhere(TIMES_TO_DECISION_PC_NSM < N + 0.5)
        tmp_c_nsm = np.argwhere(TIMES_TO_DECISION_C_NSM < N + 0.5)

        RESULTS[2, 0] = tmp_savi.shape[0] / N_TRIALS
        RESULTS[2, 1] = tmp_nsm.shape[0] / N_TRIALS
        RESULTS[2, 2] = tmp_pc_nsm.shape[0] / N_TRIALS
        RESULTS[2, 3] = tmp_c_nsm.shape[0] / N_TRIALS

        print("(p0, p1): ")
        print(RESULTS[0, :2])
        print()
        print("Times to decision (avg): ")
        print("SAVI: ", RESULTS[1, 0])
        print("NSM: ", RESULTS[1, 1])
        print("PC_NSM: ", RESULTS[1, 2])
        print("Nonparametric NSM: ", RESULTS[1, 3])
        print()
        print("Power (avg): ")
        print("SAVI: ", RESULTS[2, 0])
        print("NSM: ", RESULTS[2, 1])
        print("PC_NSM: ", RESULTS[2, 2])
        print("Nonparametric NSM: ", RESULTS[2, 3])

        print()

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(np.arange(N), np.mean(LAMBDA_NSM, axis=0), 'k')
        ax.plot(np.arange(N), np.mean(LAMBDA_NSM, axis=0)+np.std(LAMBDA_NSM, axis=0), 'k--')
        ax.plot(np.arange(N), np.mean(LAMBDA_NSM_PC, axis=0), 'r')
        ax.plot(np.arange(N), np.mean(LAMBDA_NSM_PC, axis=0)+np.std(LAMBDA_NSM_PC, axis=0), 'r--')
        ax.plot(np.arange(N), np.mean(LAMBDA_NSM_C, axis=0), 'b')
        ax.plot(np.arange(N), np.mean(LAMBDA_NSM_C, axis=0)+np.std(LAMBDA_NSM_C, axis=0), 'b--')
        fig.savefig("tmp.png")