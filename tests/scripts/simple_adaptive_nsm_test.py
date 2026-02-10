import argparse
import copy

import numpy as np
from tqdm import tqdm

from sequentialized_barnard_tests.base import Decision, Hypothesis
from sequentialized_barnard_tests.savi import SaviTest
from sequentialized_barnard_tests.nsm import BernoulliNsmTest

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
    K = 3
    alpha = args.alpha
    
    c = (np.cumsum(np.ones(K)) - 1) / (K - 1)
    nsm_test = BernoulliNsmTest(alternative=Hypothesis.P0LessThanP1, alpha=alpha, c=c)
    savi_test = SaviTest(Hypothesis.P0LessThanP1, alpha=alpha)

    # Define the trial pairs
    P0, P1 = np.meshgrid((np.arange(10) + 0.5) / 10, (5 * np.arange(10)) / 100)
    # P0, P1 = np.meshgrid((np.arange(10)+0.5)/10, (np.arange(10)+0.5)/10)
    null_case = False
    null_counter = 0
    alt_counter = 0
    RESULTS_NULL = np.zeros((100, 3, 2))
    RESULTS_ALT = np.zeros((100, 3, 2))
    CUMULATIVE_POWER_NSM = -np.ones((2, 200, N))
    CUMULATIVE_POWER_SAVI = -np.ones((2, 100, N))
    for ii in range(P0.shape[0]):
        for jj in tqdm(range(P0.shape[1])):
            null_case = False
            p0 = P0[ii, jj]
            # p1 = P1[ii, jj]
            p1 = P1[ii, jj] + p0
            if np.isclose(p0, p1):
                null_case = True

            if null_case or p1 <= 1.0:
                if null_case:
                    N_TRIALS = N_TRIALS_NULL
                else:
                    N_TRIALS = N_TRIALS_ALT

                TIMES_TO_DECISION_NSM = np.zeros(N_TRIALS)
                TIMES_TO_DECISION_SAVI = np.zeros(N_TRIALS)

                # MUST INCLUDE: AVERAGE TTD + power
                RESULTS = np.zeros((3, 2))
                RESULTS[0, :] = np.array([p0, p1])
                # Begin the trials for particular (p0, p1)
                for kk in tqdm(range(N_TRIALS)):
                    data0 = np.random.binomial(1, p0, N)
                    data1 = np.random.binomial(1, p1, N)

                    DATA0 = np.zeros((N, K))
                    DATA1 = np.zeros((N, K))

                    tmp0 = np.where(data0 > 0.5, 1.0, 0.0)
                    DATA0[:, 0] = 1.0 - tmp0
                    DATA0[:, -1] = tmp0

                    tmp1 = np.where(data1 > 0.5, 1.0, 0.0)
                    DATA1[:, 0] = 1.0 - tmp1
                    DATA1[:, -1] = tmp1

                    savi_result = savi_test.run_on_sequence(data0, data1)

                    if savi_result.decision == Decision.AcceptAlternative:
                        TIMES_TO_DECISION_SAVI[kk] = savi_result.info["Time"]
                    else:
                        TIMES_TO_DECISION_SAVI[kk] = N + 1

                    nsm_result = nsm_test.run_on_sequence(data0, data1)

                    if nsm_result.decision == Decision.AcceptAlternative:
                        TIMES_TO_DECISION_NSM[kk] = nsm_result.info["Time"]
                    else:
                        TIMES_TO_DECISION_NSM[kk] = N + 1

                # Compute cumulative power data
                if null_case:
                    idx_cumulative_power = 0
                    counter_cumulative_power = null_counter
                else:
                    idx_cumulative_power = 1
                    counter_cumulative_power = alt_counter

                tmp_ttd_nsm = np.sort(TIMES_TO_DECISION_NSM)
                tmp_ttd_savi = np.sort(TIMES_TO_DECISION_SAVI)

                current_value_nsm = TIMES_TO_DECISION_NSM[0]
                current_value_savi = TIMES_TO_DECISION_SAVI[0]
                idx_nsm = 0
                idx_savi = 0
                for i in range(N):
                    if i < current_value_nsm:
                        CUMULATIVE_POWER_NSM[
                            idx_cumulative_power, counter_cumulative_power, i
                        ] = idx_nsm
                    else:
                        tmp = np.argwhere(TIMES_TO_DECISION_NSM <= i)
                        idx_nsm = tmp.shape[0]
                        try:
                            current_value_nsm = TIMES_TO_DECISION_NSM[idx_nsm]
                        except:
                            current_value_nsm = N
                        CUMULATIVE_POWER_NSM[
                            idx_cumulative_power, counter_cumulative_power, i
                        ] = idx_nsm

                    if i < current_value_savi:
                        CUMULATIVE_POWER_SAVI[
                            idx_cumulative_power, counter_cumulative_power, i
                        ] = idx_savi
                    else:
                        tmp = np.argwhere(TIMES_TO_DECISION_SAVI <= i)
                        idx_savi = tmp.shape[0]
                        try:
                            current_value_savi = TIMES_TO_DECISION_SAVI[idx_savi]
                        except:
                            current_value_savi = N
                        CUMULATIVE_POWER_SAVI[
                            idx_cumulative_power, counter_cumulative_power, i
                        ] = idx_savi

                CUMULATIVE_POWER_NSM[
                    idx_cumulative_power, counter_cumulative_power, :
                ] /= N_TRIALS
                CUMULATIVE_POWER_SAVI[
                    idx_cumulative_power, counter_cumulative_power, :
                ] /= N_TRIALS

                # Store summary statistics
                RESULTS[1, 0] = np.mean(TIMES_TO_DECISION_SAVI)
                RESULTS[1, 1] = np.mean(TIMES_TO_DECISION_NSM)
                tmp_savi = np.argwhere(TIMES_TO_DECISION_SAVI < N + 0.5)
                tmp_nsm = np.argwhere(TIMES_TO_DECISION_NSM < N + 0.5)

                RESULTS[2, 0] = tmp_savi.shape[0] / N_TRIALS
                RESULTS[2, 1] = tmp_nsm.shape[0] / N_TRIALS

                if null_case:
                    RESULTS_NULL[null_counter, :, :] = copy.deepcopy(RESULTS)
                    null_counter += 1
                else:
                    RESULTS_ALT[alt_counter, :, :] = copy.deepcopy(RESULTS)
                    alt_counter += 1

    try:
        np.save(
            f"data/NSM/Results_Null_alpha_{alpha}.npy",
            RESULTS_NULL[:null_counter, :, :],
        )
        np.save(
            f"data/NSM/Results_Alt_alpha_{alpha}.npy",
            RESULTS_ALT[:alt_counter, :, :],
        )
        np.save(
            f"data/NSM/Cumulative_Power_NSM_null_alpha_{alpha}.npy",
            CUMULATIVE_POWER_NSM[0, :null_counter, :],
        )
        np.save(
            f"data/NSM/Cumulative_Power_NSM_alt_alpha_{alpha}.npy",
            CUMULATIVE_POWER_NSM[1, :alt_counter, :],
        )
        np.save(
            f"data/NSM/Cumulative_Power_SAVI_null_alpha_{alpha}.npy",
            CUMULATIVE_POWER_SAVI[0, :null_counter, :],
        )
        np.save(
            f"data/NSM/Cumulative_Power_SAVI_alt_alpha_{alpha}.npy",
            CUMULATIVE_POWER_SAVI[1, :alt_counter, :],
        )
    except:
        print()
        print(
            "Couldn't save to desired location. Please try to save manually in this breakpoint"
        )
        breakpoint()