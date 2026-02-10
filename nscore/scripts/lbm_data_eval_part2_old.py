import pandas as pd
import os 

import numpy as np
from sequentialized_barnard_tests.nsm import MirroredPartialCreditNsmTest
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
    breakpoint()
    # print("Dataframe column names")
    # print(df_columns[2])
    # print(df_columns[3])    
    # print()

    # # Success
    # print(df_columns[2])
    # print(first_df[df_columns[2]])
    # print()

    # # Task progress
    # print(df_columns[3])
    # print(first_df[df_columns[3]])
    # print()

    for ii in range(n_tasks):
        critical_idx = 2*ii

        # Success scores
        success_pi_0 = np.multiply(np.array(first_df[df_columns[3]][critical_idx]), 1)
        success_pi_1 = np.multiply(np.array(first_df[df_columns[3]][critical_idx + 1]), 1)
    
        # print("Mean success rate: ")
        # print("Policy 0: ", np.mean(success_pi_0))
        # print("Policy 1: ", np.mean(success_pi_1))
        # print()

        # Progress scores
        progress_pi_0 = np.multiply(np.array(first_df[df_columns[4]][critical_idx]), 1)
        progress_pi_1 = np.multiply(np.array(first_df[df_columns[4]][critical_idx + 1]), 1)
        
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
        nsm_test = MirroredPartialCreditNsmTest(alternative=Hypothesis.P0LessThanP1, alpha=alpha, c=c)

        nsm_result = nsm_test.run_on_sequence(idx_list_0.astype(int), idx_list_1.astype(int))

        if nsm_result.decision == Decision.AcceptAlternative:
            time_of_decision = nsm_result.info["result_for_alternative"].info["Time"]
            decision_str = "P0LessThanP1"
        elif nsm_result.decision == Decision.AcceptNull:
            time_of_decision = nsm_result.info["result_for_null"].info["Time"]
            decision_str = "P0MoreThanP1"
        else:
            time_of_decision = n_evals
            decision_str = "FailToDecide"

        print("Task: ")
        print(first_df[df_columns[2]][critical_idx])
        print()
        print("Mean progress amounts: ")
        print("Policy 0: ", np.mean(progress_pi_0))
        print("Policy 1: ", np.mean(progress_pi_1))
        print()
        print("NSM Decision: ")
        print(decision_str)
        print()
        print("Time of Decision: ")
        print(time_of_decision)
        print()
        print()

        breakpoint()