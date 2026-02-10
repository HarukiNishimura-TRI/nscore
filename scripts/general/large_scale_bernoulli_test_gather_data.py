import argparse
import copy
import os 
import sys
add_path = os.getcwd()
sys.path.append(add_path)

import numpy as np
from tqdm import tqdm

from sequentialized_barnard_tests.base import Decision, Hypothesis
from sequentialized_barnard_tests.step import StepTest

from nscore.savi import SaviTest
from nscore.nsm import BernoulliNsmTest, PartialCreditNsmTest
from nscore.wsr import WsrComparisonTest
from nscore.nonparametric_nsm import ContinuousNsmTest

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

    # Parse the args
    args = parser.parse_args()
    
    # Instantiate seed for reproducibility
    np.random.seed(args.seed)

    # Set the number of evaluation sequences for null and alternative, respectively
    N_TRIALS_ALT = args.n_trials_alt
    N_TRIALS_NULL = args.n_trials_null

    # Set the number of trials per evaluation sequence
    N = args.n_max
    alpha = args.alpha

    # Initialize the NSM tests
    K_bernoulli = 3
    c_bernoulli = (np.cumsum(np.ones(K_bernoulli)) - 1) / (K_bernoulli - 1)
    bernoulli_nsm_test = ContinuousNsmTest(alternative=Hypothesis.P0LessThanP1, alpha=alpha, c=c_bernoulli)
    
    K_partial_credit = 5
    c_partial_credit = (np.cumsum(np.ones(K_partial_credit)) - 1) / (K_partial_credit - 1)
    partial_credit_nsm_test = PartialCreditNsmTest(alternative=Hypothesis.P0LessThanP1, alpha=alpha, c=c_partial_credit)

    K_nonparametric = 10
    c_nonparametric = (np.cumsum(np.ones(K_nonparametric)) - 1) / (K_nonparametric - 1)
    nonparametric_nsm_test = ContinuousNsmTest(alternative=Hypothesis.P0LessThanP1, alpha=alpha, c=c_nonparametric)
    
    # Initialize the WSR test
    c_wsr = 0.95 
    wsr_test = WsrComparisonTest(alternative=Hypothesis.P0LessThanP1, alpha=alpha, c_wsr=c_wsr)

    # Initialize the SAVI test
    savi_test = SaviTest(Hypothesis.P0LessThanP1, alpha=alpha)
    
    # Initialize the STEP test
    step_test = StepTest(Hypothesis.P0LessThanP1, np.minimum(N, 500), alpha)

    # Define the trial pairs
    n_p0 = 10 
    n_p1 = 6
    max_p1_gap = 0.5
    P0, P1 = np.meshgrid((np.arange(n_p0) + 0.5) / n_p0, ((100. * max_p1_gap / (n_p1 - 1)) * np.arange(n_p1)) / 100)

    # Explicitly specify the number of policies being evaluated 
    n_policies = 6

    null_case = False

    null_counter = 0
    alt_counter = 0

    RESULTS_NULL = np.zeros((n_p0*n_p1, 3, n_policies))
    RESULTS_ALT = np.zeros((n_p0*n_p1, 3, n_policies))

    CUMULATIVE_POWER_BINARY_NSM = -np.ones((2, n_p0*n_p1*2, N))
    CUMULATIVE_POWER_PC_NSM = -np.ones((2, n_p0*n_p1*2, N))
    CUMULATIVE_POWER_CONT_NSM = -np.ones((2, n_p0*n_p1*2, N))
    CUMULATIVE_POWER_WSR = -np.ones((2, n_p0*n_p1*2, N))
    CUMULATIVE_POWER_SAVI = -np.ones((2, n_p0*n_p1, N))
    CUMULATIVE_POWER_STEP = -np.ones((2, n_p0*n_p1, N))

    for ii in range(P0.shape[0]):
        for jj in tqdm(range(P0.shape[1])):
            null_case = False
            p0 = P0[ii, jj]
            p1 = P1[ii, jj] + p0

            if np.isclose(p0, p1):
                null_case = True

            if null_case or p1 <= 1.0:
                if null_case:
                    N_TRIALS = N_TRIALS_NULL
                else:
                    N_TRIALS = N_TRIALS_ALT

                TIMES_TO_DECISION_BINARY_NSM = np.zeros(N_TRIALS)
                TIMES_TO_DECISION_PC_NSM = np.zeros(N_TRIALS)
                TIMES_TO_DECISION_CONT_NSM = np.zeros(N_TRIALS)
                TIMES_TO_DECISION_WSR = np.zeros(N_TRIALS)
                TIMES_TO_DECISION_SAVI = np.zeros(N_TRIALS)
                TIMES_TO_DECISION_STEP = np.zeros(N_TRIALS)

                # MUST INCLUDE: AVERAGE TTD + power
                RESULTS = np.zeros((3, n_policies))
                RESULTS[0, :2] = np.array([p0, p1])
                # Begin the trials for particular (p0, p1)
                for kk in tqdm(range(N_TRIALS)):
                    data0 = np.random.binomial(1, p0, N)
                    data1 = np.random.binomial(1, p1, N)

                    # Bernoulli NSM
                    bernoulli_nsm_result = bernoulli_nsm_test.run_on_sequence(data0, data1)
                    if bernoulli_nsm_result.decision == Decision.AcceptAlternative:
                        TIMES_TO_DECISION_BINARY_NSM[kk] = bernoulli_nsm_result.info["Time"]
                    else:
                        TIMES_TO_DECISION_BINARY_NSM[kk] = N + 1
                    
                    # Partial Credit NSM
                    partial_credit_nsm_result = partial_credit_nsm_test.run_on_sequence(data0*(K_partial_credit-1), data1*(K_partial_credit-1))
                    if partial_credit_nsm_result.decision == Decision.AcceptAlternative:
                        TIMES_TO_DECISION_PC_NSM[kk] = partial_credit_nsm_result.info["Time"]
                    else:
                        TIMES_TO_DECISION_PC_NSM[kk] = N + 1
                    
                    # Nonparametric NSM
                    nonparametric_nsm_result = nonparametric_nsm_test.run_on_sequence(data0, data1)
                    if nonparametric_nsm_result.decision == Decision.AcceptAlternative:
                        TIMES_TO_DECISION_CONT_NSM[kk] = nonparametric_nsm_result.info["Time"]
                    else:
                        TIMES_TO_DECISION_CONT_NSM[kk] = N + 1
                    
                    # WSR Test
                    wsr_result = wsr_test.run_on_sequence(data0, data1)
                    if wsr_result.decision == Decision.AcceptAlternative:
                        TIMES_TO_DECISION_WSR[kk] = wsr_result.info["Time"]
                    else:
                        TIMES_TO_DECISION_WSR[kk] = N + 1
                    
                    # SAVI Test
                    savi_result = savi_test.run_on_sequence(data0, data1)
                    if savi_result.decision == Decision.AcceptAlternative:
                        TIMES_TO_DECISION_SAVI[kk] = savi_result.info["Time"]
                    else:
                        TIMES_TO_DECISION_SAVI[kk] = N + 1
                    
                    # STEP Test
                    step_result = step_test.run_on_sequence(data0, data1)
                    if step_result.decision == Decision.AcceptAlternative:
                        TIMES_TO_DECISION_STEP[kk] = step_result.info["Time"]
                    else:
                        TIMES_TO_DECISION_STEP[kk] = N + 1

                # Compute cumulative power data
                if null_case:
                    idx_cumulative_power = 0
                    counter_cumulative_power = null_counter
                else:
                    idx_cumulative_power = 1
                    counter_cumulative_power = alt_counter

                tmp_ttd_binary_nsm = np.sort(TIMES_TO_DECISION_BINARY_NSM)
                tmp_ttd_pc_nsm = np.sort(TIMES_TO_DECISION_PC_NSM)
                tmp_ttd_cont_nsm = np.sort(TIMES_TO_DECISION_CONT_NSM)
                tmp_ttd_wsr = np.sort(TIMES_TO_DECISION_WSR)
                tmp_ttd_savi = np.sort(TIMES_TO_DECISION_SAVI)
                tmp_ttd_step = np.sort(TIMES_TO_DECISION_STEP)

                current_value_binary_nsm = tmp_ttd_binary_nsm[0]
                current_value_pc_nsm = tmp_ttd_pc_nsm[0]
                current_value_cont_nsm = tmp_ttd_cont_nsm[0]
                current_value_wsr = tmp_ttd_wsr[0]
                current_value_savi = tmp_ttd_savi[0]
                current_value_step = tmp_ttd_step[0]

                idx_binary_nsm = 0
                idx_pc_nsm = 0
                idx_cont_nsm = 0
                idx_wsr = 0
                idx_savi = 0
                idx_step = 0

                for i in range(N):
                    # Bernoulli NSM Test
                    if i < current_value_binary_nsm:
                        CUMULATIVE_POWER_BINARY_NSM[
                            idx_cumulative_power, counter_cumulative_power, i
                        ] = idx_binary_nsm
                    else:
                        tmp = np.argwhere(tmp_ttd_binary_nsm <= i)
                        idx_binary_nsm = tmp.shape[0]
                        try:
                            current_value_binary_nsm = tmp_ttd_binary_nsm[idx_binary_nsm]
                        except:
                            current_value_binary_nsm = N

                        CUMULATIVE_POWER_BINARY_NSM[
                            idx_cumulative_power, counter_cumulative_power, i
                        ] = idx_binary_nsm
                    
                    # Partial Credit NSM Test
                    if i < current_value_pc_nsm:
                        CUMULATIVE_POWER_PC_NSM[
                            idx_cumulative_power, counter_cumulative_power, i
                        ] = idx_pc_nsm
                    else:
                        tmp = np.argwhere(tmp_ttd_pc_nsm <= i)
                        idx_pc_nsm = tmp.shape[0]
                        try:
                            current_value_pc_nsm = tmp_ttd_pc_nsm[idx_pc_nsm]
                        except:
                            current_value_pc_nsm = N
                        
                        CUMULATIVE_POWER_PC_NSM[
                            idx_cumulative_power, counter_cumulative_power, i
                        ] = idx_pc_nsm
                    
                    # Nonparametric NSM Test
                    if i < current_value_cont_nsm:
                        CUMULATIVE_POWER_CONT_NSM[
                            idx_cumulative_power, counter_cumulative_power, i
                        ] = idx_cont_nsm
                    else:
                        tmp = np.argwhere(tmp_ttd_cont_nsm <= i)
                        idx_cont_nsm = tmp.shape[0]
                        try:
                            current_value_cont_nsm = tmp_ttd_cont_nsm[idx_cont_nsm]
                        except:
                            current_value_cont_nsm = N
                        CUMULATIVE_POWER_CONT_NSM[
                            idx_cumulative_power, counter_cumulative_power, i
                        ] = idx_cont_nsm
                    
                    # WSR Comparison Test
                    if i < current_value_wsr:
                        CUMULATIVE_POWER_WSR[
                            idx_cumulative_power, counter_cumulative_power, i
                        ] = idx_wsr
                    else:
                        tmp = np.argwhere(tmp_ttd_wsr <= i)
                        idx_wsr = tmp.shape[0]
                        try:
                            current_value_wsr = tmp_ttd_wsr[idx_wsr]
                        except:
                            current_value_wsr = N
                        CUMULATIVE_POWER_WSR[
                            idx_cumulative_power, counter_cumulative_power, i
                        ] = idx_wsr

                    # SAVI Test
                    if i < current_value_savi:
                        CUMULATIVE_POWER_SAVI[
                            idx_cumulative_power, counter_cumulative_power, i
                        ] = idx_savi
                    else:
                        tmp = np.argwhere(tmp_ttd_savi <= i)
                        idx_savi = tmp.shape[0]
                        try:
                            current_value_savi = tmp_ttd_savi[idx_savi]
                        except:
                            current_value_savi = N
                        CUMULATIVE_POWER_SAVI[
                            idx_cumulative_power, counter_cumulative_power, i
                        ] = idx_savi
                    
                    # STEP Test
                    if i < current_value_step:
                        CUMULATIVE_POWER_STEP[
                            idx_cumulative_power, counter_cumulative_power, i
                        ] = idx_step
                    else:
                        tmp = np.argwhere(tmp_ttd_step <= i)
                        idx_step = tmp.shape[0]
                        try:
                            current_value_step = tmp_ttd_step[idx_step]
                        except:
                            current_value_step = N
                        CUMULATIVE_POWER_STEP[
                            idx_cumulative_power, counter_cumulative_power, i
                        ] = idx_step

                CUMULATIVE_POWER_BINARY_NSM[
                    idx_cumulative_power, counter_cumulative_power, :
                ] /= N_TRIALS
                CUMULATIVE_POWER_PC_NSM[
                    idx_cumulative_power, counter_cumulative_power, :
                ] /= N_TRIALS
                CUMULATIVE_POWER_CONT_NSM[
                    idx_cumulative_power, counter_cumulative_power, :
                ] /= N_TRIALS
                CUMULATIVE_POWER_WSR[
                    idx_cumulative_power, counter_cumulative_power, :
                ] /= N_TRIALS
                CUMULATIVE_POWER_SAVI[
                    idx_cumulative_power, counter_cumulative_power, :
                ] /= N_TRIALS
                CUMULATIVE_POWER_STEP[
                    idx_cumulative_power, counter_cumulative_power, :
                ] /= N_TRIALS

                # Store summary statistics
                RESULTS[1, 0] = np.mean(TIMES_TO_DECISION_BINARY_NSM)
                RESULTS[1, 1] = np.mean(TIMES_TO_DECISION_PC_NSM)
                RESULTS[1, 2] = np.mean(TIMES_TO_DECISION_CONT_NSM)
                RESULTS[1, 3] = np.mean(TIMES_TO_DECISION_WSR)
                RESULTS[1, 4] = np.mean(TIMES_TO_DECISION_SAVI)
                RESULTS[1, 5] = np.mean(TIMES_TO_DECISION_STEP)

                tmp_binary_nsm = np.argwhere(TIMES_TO_DECISION_BINARY_NSM < N + 0.5)
                tmp_pc_nsm = np.argwhere(TIMES_TO_DECISION_PC_NSM < N + 0.5)
                tmp_cont_nsm = np.argwhere(TIMES_TO_DECISION_CONT_NSM < N + 0.5)
                tmp_wsr = np.argwhere(TIMES_TO_DECISION_WSR < N + 0.5)
                tmp_savi = np.argwhere(TIMES_TO_DECISION_SAVI < N + 0.5)
                tmp_step = np.argwhere(TIMES_TO_DECISION_STEP < N + 0.5)

                RESULTS[2, 0] = tmp_binary_nsm.shape[0] / N_TRIALS
                RESULTS[2, 1] = tmp_pc_nsm.shape[0] / N_TRIALS
                RESULTS[2, 2] = tmp_cont_nsm.shape[0] / N_TRIALS
                RESULTS[2, 3] = tmp_wsr.shape[0] / N_TRIALS
                RESULTS[2, 4] = tmp_savi.shape[0] / N_TRIALS
                RESULTS[2, 5] = tmp_step.shape[0] / N_TRIALS

                if null_case:
                    RESULTS_NULL[null_counter, :, :] = copy.deepcopy(RESULTS)
                    null_counter += 1
                else:
                    RESULTS_ALT[alt_counter, :, :] = copy.deepcopy(RESULTS)
                    alt_counter += 1

    try:
        # Summary Results
        np.save(
            f"data/BINARY/Results_Null_alpha_{alpha}.npy",
            RESULTS_NULL[:null_counter, :, :],
        )
        np.save(
            f"data/BINARY/Results_Alt_alpha_{alpha}.npy",
            RESULTS_ALT[:alt_counter, :, :],
        )
        # Bernoulli NSM
        np.save(
            f"data/BINARY/Cumulative_Power_Binary_NSM_null_alpha_{alpha}.npy",
            CUMULATIVE_POWER_BINARY_NSM[0, :null_counter, :],
        )
        np.save(
            f"data/BINARY/Cumulative_Power_Binary_NSM_alt_alpha_{alpha}.npy",
            CUMULATIVE_POWER_BINARY_NSM[1, :alt_counter, :],
        )
        # Partial Credit NSM
        np.save(
            f"data/BINARY/Cumulative_Power_PC_NSM_null_alpha_{alpha}.npy",
            CUMULATIVE_POWER_PC_NSM[0, :null_counter, :],
        )
        np.save(
            f"data/BINARY/Cumulative_Power_PC_NSM_alt_alpha_{alpha}.npy",
            CUMULATIVE_POWER_PC_NSM[1, :alt_counter, :],
        )
        # Nonparametric NSM
        np.save(
            f"data/BINARY/Cumulative_Power_Cont_NSM_null_alpha_{alpha}.npy",
            CUMULATIVE_POWER_CONT_NSM[0, :null_counter, :],
        )
        np.save(
            f"data/BINARY/Cumulative_Power_Cont_NSM_alt_alpha_{alpha}.npy",
            CUMULATIVE_POWER_CONT_NSM[1, :alt_counter, :],
        )
        # WSR
        np.save(
            f"data/BINARY/Cumulative_Power_WSR_null_alpha_{alpha}.npy",
            CUMULATIVE_POWER_WSR[0, :null_counter, :],
        )
        np.save(
            f"data/BINARY/Cumulative_Power_WSR_alt_alpha_{alpha}.npy",
            CUMULATIVE_POWER_WSR[1, :alt_counter, :],
        )
        # SAVI
        np.save(
            f"data/BINARY/Cumulative_Power_SAVI_null_alpha_{alpha}.npy",
            CUMULATIVE_POWER_SAVI[0, :null_counter, :],
        )
        np.save(
            f"data/BINARY/Cumulative_Power_SAVI_alt_alpha_{alpha}.npy",
            CUMULATIVE_POWER_SAVI[1, :alt_counter, :],
        )
        # STEP
        np.save(
            f"data/BINARY/Cumulative_Power_STEP_null_alpha_{alpha}.npy",
            CUMULATIVE_POWER_STEP[0, :null_counter, :],
        )
        np.save(
            f"data/BINARY/Cumulative_Power_STEP_alt_alpha_{alpha}.npy",
            CUMULATIVE_POWER_STEP[1, :alt_counter, :],
        )
    except:
        print()
        print(
            "Couldn't save to desired location. Please try to save manually in this breakpoint"
        )
        breakpoint()