import pandas as pd
import os 

import numpy as np
from sequentialized_barnard_tests.nsm import MirroredPartialCreditNsmTest, BernoulliNsmTest
from sequentialized_barnard_tests.savi import MirroredSaviTest
from sequentialized_barnard_tests.nonparametric_nsm import MirroredContinuousNsmTest
from sequentialized_barnard_tests.wsr import WsrComparisonTest
from sequentialized_barnard_tests.step import MirroredStepTest

from sequentialized_barnard_tests.base import Hypothesis, Decision

if __name__ == "__main__":
    file_path = 'data/LBM/lbm_data.pkl'

    dict_of_dfs = pd.read_pickle(file_path)

    df_keys = list(dict_of_dfs)
    print("Keys: ")
    print(df_keys)
    print(df_keys[1])
    print()

    old_first_df = dict_of_dfs[df_keys[0]]
    first_df_tmp = dict_of_dfs[df_keys[1]]
    first_df = first_df_tmp.reset_index()
    print("Second dataframe: ")
    print(first_df)
    print()

    n_rows = first_df.shape[0]
    assert n_rows % 2 == 0
    n_tasks = n_rows // 2

    df_columns = list(first_df.columns)
    print(df_columns)
    # breakpoint()

    n_resamples = 1
    RESULTS_PART_2 = np.zeros((n_tasks, 7, 2, n_resamples))
    for ii in range(n_tasks):
        critical_idx = 2*ii

        # Success scores
        success_pi_0_base = np.multiply(np.array(first_df[df_columns[3]][critical_idx]), 1)
        success_pi_1_base = np.multiply(np.array(first_df[df_columns[3]][critical_idx + 1]), 1)

        L_success = np.minimum(success_pi_0_base.shape[0], success_pi_1_base.shape[0])
        success_pi_0 = success_pi_0_base[:L_success]
        success_pi_1 = success_pi_1_base[:L_success]
        # print("Mean success rate: ")
        # print("Policy 0: ", np.mean(success_pi_0))
        # print("Policy 1: ", np.mean(success_pi_1))
        # print()

        # Progress scores
        progress_pi_0_base = np.multiply(np.array(first_df[df_columns[4]][critical_idx]), 1)
        progress_pi_1_base = np.multiply(np.array(first_df[df_columns[4]][critical_idx + 1]), 1)

        L_progress = np.minimum(progress_pi_0_base.shape[0], progress_pi_1_base.shape[0])
        progress_pi_0 = progress_pi_0_base[:L_progress]
        progress_pi_1 = progress_pi_1_base[:L_progress]
        # print("Mean progress amounts: ")
        # print("Policy 0: ", np.mean(progress_pi_0))
        # print("Policy 1: ", np.mean(progress_pi_1))
        # print()

        # Turn progress scores into outcomes
        K = np.array(first_df[df_columns[-1]])[critical_idx]
        assert K == np.array(first_df[df_columns[-1]])[critical_idx + 1]
        c = np.arange(K + 1) / K

        n_evals = np.minimum(progress_pi_0.shape[0], progress_pi_1.shape[0])
        idx_list_0 = -np.ones(n_evals).astype(int)
        idx_list_1 = -np.ones(n_evals).astype(int)
        for i in range(n_evals):
            try:
                assert np.isclose(0, np.min(np.abs(progress_pi_0[i]*K - np.arange(K+1))))
                assert np.isclose(0, np.min(np.abs(progress_pi_1[i]*K - np.arange(K+1))))
                try:
                    idx_list_0[i] = int(np.argmin(np.abs(progress_pi_0[i]*K - np.arange(K+1))))
                    idx_list_1[i] = int(np.argmin(np.abs(progress_pi_1[i]*K - np.arange(K+1))))
                except:
                    idx_list_0[i] = int(np.argmin(np.abs(progress_pi_0[i]*K - np.arange(K+1)))[0])
                    idx_list_1[i] = int(np.argmin(np.abs(progress_pi_1[i]*K - np.arange(K+1)))[0])
            except:
                print(f"Bad datum at index {i}: Input is: {progress_pi_0[i]*K}")
    
        # print()
        # print("Made it!")

        assert np.min(idx_list_0) >= 0
        assert np.min(idx_list_1) >= 0 

        # print()
        # print("Made it again!")
        # breakpoint()

        # Run the desired statistical test
        alpha=0.05
        for jj in range(n_resamples):
            # idx_shuffled_success = np.random.choice(L_success, size=L_success, replace=False)
            # idx_shuffled_progress = np.random.choice(L_progress, size=L_progress, replace=False)
            idx_shuffled_success = np.arange(L_success)
            idx_shuffled_progress = np.arange(L_progress)

            # Progress metrics
            nsm_test = MirroredPartialCreditNsmTest(alternative=Hypothesis.P0LessThanP1, alpha=alpha, c=c)
            nonparametric_nsm_test = MirroredContinuousNsmTest(alternative=Hypothesis.P0LessThanP1, alpha=alpha, c=c)
            wsr_test = WsrComparisonTest(alternative=Hypothesis.P0LessThanP1, alpha=alpha)

            # Success metrics evaluation
            binary_savi_test = MirroredSaviTest(alternative=Hypothesis.P0LessThanP1, alpha=alpha)
            binary_nsm_test = MirroredContinuousNsmTest(alternative=Hypothesis.P0LessThanP1, alpha=alpha, c=np.arange(2))
            binary_step_test = MirroredStepTest(alternative=Hypothesis.P0LessThanP1, n_max=50, alpha=alpha)
            binary_wsr_test = WsrComparisonTest(alternative=Hypothesis.P0LessThanP1, alpha=alpha)

            # Progress metric results
            nsm_result = nsm_test.run_on_sequence(idx_list_0.astype(int), idx_list_1.astype(int))
            if nsm_result.decision == Decision.AcceptAlternative:
                nsm_time_of_decision = nsm_result.info["result_for_alternative"].info["Time"]
                nsm_decision_str = "P0LessThanP1"
                RESULTS_PART_2[ii, 0, 1, jj] = nsm_time_of_decision
                
            elif nsm_result.decision == Decision.AcceptNull:
                nsm_time_of_decision = nsm_result.info["result_for_null"].info["Time"]
                RESULTS_PART_2[ii, 0, 0, jj] = 1
                RESULTS_PART_2[ii, 0, 1, jj] = nsm_time_of_decision
                nsm_decision_str = "P0MoreThanP1"
            else:
                nsm_time_of_decision = n_evals
                RESULTS_PART_2[ii, 0, 1, jj] = nsm_time_of_decision
                nsm_decision_str = "FailToDecide"
            
            nonparametric_nsm_result = nonparametric_nsm_test.run_on_sequence(progress_pi_0, progress_pi_1)
            if nonparametric_nsm_result.decision == Decision.AcceptAlternative:
                nonparametric_nsm_time_of_decision = nonparametric_nsm_result.info["result_for_alternative"].info["Time"]
                nonparametric_nsm_decision_str = "P0LessThanP1"
                RESULTS_PART_2[ii, 1, 1, jj] = nonparametric_nsm_time_of_decision
            elif nonparametric_nsm_result.decision == Decision.AcceptNull:
                nonparametric_nsm_time_of_decision = nonparametric_nsm_result.info["result_for_null"].info["Time"]
                RESULTS_PART_2[ii, 1, 1, jj] = nonparametric_nsm_time_of_decision
                RESULTS_PART_2[ii, 1, 0, jj] = 1
                nonparametric_nsm_decision_str = "P0MoreThanP1"
            else:
                nonparametric_nsm_time_of_decision = n_evals
                RESULTS_PART_2[ii, 1, 1, jj] = nonparametric_nsm_time_of_decision
                nonparametric_nsm_decision_str = "FailToDecide"
            
            wsr_result = wsr_test.run_on_sequence(progress_pi_0, progress_pi_1)
            if wsr_result.decision == Decision.AcceptAlternative:
                wsr_time_of_decision = wsr_result.info["Time"]
                RESULTS_PART_2[ii, 2, 1, jj] = wsr_time_of_decision
                wsr_decision_str = "P0LessThanP1"
            elif wsr_result.decision == Decision.AcceptNull:
                wsr_time_of_decision = wsr_result.info["Time"]
                RESULTS_PART_2[ii, 2, 1, jj] = wsr_time_of_decision
                RESULTS_PART_2[ii, 2, 0, jj] = 1
                wsr_decision_str = "P0MoreThanP1"
            else:
                wsr_time_of_decision = n_evals
                RESULTS_PART_2[ii, 2, 1, jj] = wsr_time_of_decision
                wsr_decision_str = "FailToDecide"

            # Success metric results
            binary_step_result = binary_step_test.run_on_sequence(success_pi_0, success_pi_1)
            if binary_step_result.decision == Decision.AcceptAlternative:
                binary_step_time_of_decision = binary_step_result.info["Time"]
                RESULTS_PART_2[ii, 3, 1, jj] = binary_step_time_of_decision
                binary_step_decision_str = "P0LessThanP1"
            elif binary_step_result.decision == Decision.AcceptNull:
                binary_step_time_of_decision = binary_step_result.info["Time"]
                RESULTS_PART_2[ii, 3, 1, jj] = binary_step_time_of_decision
                RESULTS_PART_2[ii, 3, 0, jj] = 1
                binary_step_decision_str = "P0MoreThanP1"
            else:
                binary_step_time_of_decision = n_evals
                RESULTS_PART_2[ii, 3, 1, jj] = binary_step_time_of_decision
                binary_step_decision_str = "FailToDecide"

            binary_nsm_result = binary_nsm_test.run_on_sequence(success_pi_0, success_pi_1)
            if binary_nsm_result.decision == Decision.AcceptAlternative:
                binary_nsm_time_of_decision = binary_nsm_result.info["result_for_alternative"].info["Time"]
                RESULTS_PART_2[ii, 4, 1, jj] = binary_nsm_time_of_decision
                binary_nsm_decision_str = "P0LessThanP1"
            elif binary_nsm_result.decision == Decision.AcceptNull:
                binary_nsm_time_of_decision = binary_nsm_result.info["result_for_null"].info["Time"]
                RESULTS_PART_2[ii, 4, 1, jj] = binary_nsm_time_of_decision
                RESULTS_PART_2[ii, 4, 0, jj] = 1
                binary_nsm_decision_str = "P0MoreThanP1"
            else:
                binary_nsm_time_of_decision = n_evals
                RESULTS_PART_2[ii, 4, 1, jj] = binary_nsm_time_of_decision
                binary_nsm_decision_str = "FailToDecide"
            
            binary_savi_result = binary_savi_test.run_on_sequence(success_pi_0, success_pi_1)
            if binary_savi_result.decision == Decision.AcceptAlternative:
                binary_savi_time_of_decision = binary_savi_result.info["result_for_alternative"].info["Time"]
                RESULTS_PART_2[ii, 5, 1, jj] = binary_savi_time_of_decision
                binary_savi_decision_str = "P0LessThanP1"
            elif binary_savi_result.decision == Decision.AcceptNull:
                binary_savi_time_of_decision = binary_savi_result.info["result_for_null"].info["Time"]
                RESULTS_PART_2[ii, 5, 1, jj] = binary_savi_time_of_decision
                RESULTS_PART_2[ii, 5, 0, jj] = 1
                binary_savi_decision_str = "P0MoreThanP1"
            else:
                binary_savi_time_of_decision = n_evals
                RESULTS_PART_2[ii, 5, 1, jj] = binary_savi_time_of_decision
                binary_savi_decision_str = "FailToDecide"
            
            binary_wsr_result = binary_wsr_test.run_on_sequence(success_pi_0, success_pi_1)
            if binary_wsr_result.decision == Decision.AcceptAlternative:
                binary_wsr_time_of_decision = binary_wsr_result.info["Time"]
                RESULTS_PART_2[ii, 6, 1, jj] = binary_wsr_time_of_decision
                binary_wsr_decision_str = "P0LessThanP1"
            elif binary_wsr_result.decision == Decision.AcceptNull:
                binary_wsr_time_of_decision = binary_wsr_result.info["Time"]
                RESULTS_PART_2[ii, 6, 1, jj] = binary_wsr_time_of_decision
                RESULTS_PART_2[ii, 6, 0, jj] = 1
                binary_wsr_decision_str = "P0MoreThanP1"
            else:
                binary_wsr_time_of_decision = n_evals
                RESULTS_PART_2[ii, 6, 1, jj] = binary_wsr_time_of_decision
                binary_wsr_decision_str = "FailToDecide"

        print("Task: ")
        print(first_df[df_columns[2]][critical_idx])
        print()
        # print("Mean progress amounts: ")
        # print("Policy 0: ", np.mean(progress_pi_0))
        # print("Policy 1: ", np.mean(progress_pi_1))
        # print()
        # print("NSM Decision (Time): ")
        # print(nsm_decision_str, "(", nsm_time_of_decision,")")
        # print()
        # print("Nonparametric NSM Decision (Time): ")
        # print(nonparametric_nsm_decision_str, "(", nonparametric_nsm_time_of_decision,")")
        # print()
        # print("WSR Decision (Time): ")
        # print(wsr_decision_str, "(", wsr_time_of_decision,")")
        # print()
        # print()
        # print("Mean success scores: ")
        # print("Policy 0: ", np.mean(success_pi_0))
        # print("Policy 1: ", np.mean(success_pi_1))
        # print()
        # print("STEP Decision (Time): ")
        # print(binary_step_decision_str, "(", binary_step_time_of_decision,")")
        # print()
        # print("Binary NSM Decision (Time): ")
        # print(binary_nsm_decision_str, "(", binary_nsm_time_of_decision,")")
        # print()
        # print("Binary SAVI Decision (Time): ")
        # print(binary_savi_decision_str, "(", binary_savi_time_of_decision,")")
        # print()
        # print("Binary WSR Decision (Time): ")
        # print(binary_wsr_decision_str, "(", binary_wsr_time_of_decision,")")
        # print()
        # print()

        # breakpoint()
    
    np.save("data/LBM/LBM_RESULTS_PART_2.npy", RESULTS_PART_2)