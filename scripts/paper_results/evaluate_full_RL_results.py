import os
import sys
add_path = os.getcwd()
sys.path.append(add_path)

from copy import deepcopy
import numpy as np 
from tqdm import tqdm 

from sequentialized_barnard_tests.base import Decision, Hypothesis
from nscore.nonparametric_nsm import MirroredContinuousNsmTest
from nscore.wsr import WsrComparisonTest

if __name__ == "__main__":

    base_load_path = "data/RL/Task_and_Method/"
    
    methods = ["ddpg", "ppo", "sac", "td3"]
    tasks = ["Ant", "Cheetah", "Hopper", "Humanoid", "Pendulum", "Pusher", "Walker"]

    complete_data_base = np.zeros((500000, len(methods)))

    task_string = tasks[0]
    critical_length = 500000
    for i in range(len(methods)):
        method_string = methods[i]
        tmp = np.load(base_load_path + task_string + "_" + method_string + ".npy")
        critical_length = np.minimum(critical_length, tmp.shape[0])
        complete_data_base[:tmp.shape[0], i] = deepcopy(tmp)
        del tmp
    
    print("Critical length: ", critical_length)

    complete_data = deepcopy(complete_data_base[:critical_length, :])

    del complete_data_base

    # Set up the evaluation logic for full ranking
    n_max = int(500)
    n_meta_evals = critical_length // n_max
    n_comparisons = len(methods) * (len(methods)-1) // 2
    n_policies = len(methods)

    # RL task reward parameters 
    upper_bound_on_reward = np.maximum(np.max(complete_data), 6000.)
    lower_bound_on_reward = np.minimum(np.min(complete_data), 0.)

    # Set evaluation parameters 
    total_alpha = 0.05
    alpha_per_hypothesis = total_alpha / n_comparisons

    # Set storage for evaluation metrics 
    time_to_all_but_one = np.zeros((n_meta_evals, 2))
    time_to_all = np.zeros((n_meta_evals, 2))
    
    # Debugging: specify early stopping time if desired 
    modified_stopping_time = n_meta_evals
    iteration_limit = np.minimum(modified_stopping_time, n_meta_evals)

    # Set up the meta-tests
    meta_procedure_test_full_ranking = MirroredContinuousNsmTest(alternative=Hypothesis.P0LessThanP1, alpha=0.005, c=np.arange(101)/100.)
    meta_procedure_test_full_ranking.reset()

    meta_procedure_test_partial_ranking = MirroredContinuousNsmTest(alternative=Hypothesis.P0LessThanP1, alpha=0.005, c=np.arange(101)/100.)
    meta_procedure_test_partial_ranking.reset()

    full_is_finished = False 
    partial_is_finished = False 

    # Loop through the RL data
    tracking_idx = int(0)
    for ii in tqdm(range(iteration_limit)):
        data_for_comparisons = (complete_data[tracking_idx:(tracking_idx+n_max), :] - lower_bound_on_reward) * (1. / (upper_bound_on_reward - lower_bound_on_reward))
        tracking_idx += n_max

        # Run NSCORE evals
        nscore_test = MirroredContinuousNsmTest(alternative=Hypothesis.P0LessThanP1, alpha=alpha_per_hypothesis, c=np.arange(21)/20.)
        comparison_idx = int(0)
        times_to_decision = np.zeros(n_comparisons)
        for jj in range(n_policies-1):
            for kk in range(jj+1, n_policies):
                nscore_test.reset()
                result = nscore_test.run_on_sequence(data_for_comparisons[:, jj], data_for_comparisons[:, kk])
                if result.decision is Decision.AcceptAlternative:
                    times_to_decision[comparison_idx] = result.info["result_for_alternative"].info["Time"]
                elif result.decision is Decision.AcceptNull:
                    times_to_decision[comparison_idx] = result.info["result_for_null"].info["Time"]
                else:
                    times_to_decision[comparison_idx] = n_max 

                comparison_idx += 1
        
        time_to_all[ii, 0] = np.max(times_to_decision)
        time_to_all_but_one[ii, 0] = np.sort(times_to_decision)[-2]

        # Run WSR evals
        wsr_test = WsrComparisonTest(alternative=Hypothesis.P0LessThanP1, alpha=alpha_per_hypothesis, c_wsr=0.75)
        comparison_idx = int(0)
        times_to_decision = np.zeros(n_comparisons)
        for jj in range(n_policies-1):
            for kk in range(jj+1, n_policies):
                wsr_test.reset()
                result = wsr_test.run_on_sequence(data_for_comparisons[:, jj], data_for_comparisons[:, kk])
                if result.decision is Decision.AcceptAlternative or result.decision is Decision.AcceptNull:
                    times_to_decision[comparison_idx] = result.info["Time"]
                else:
                    times_to_decision[comparison_idx] = n_max 

                comparison_idx += 1
        
        time_to_all[ii, 1] = np.max(times_to_decision)
        time_to_all_but_one[ii, 1] = np.sort(times_to_decision)[-2]

        if ii % 2 == 1 and (not full_is_finished or not partial_is_finished):
            if full_is_finished:
                pass
            else:
                result_full = meta_procedure_test_full_ranking.step(time_to_all[ii-1, 0]/float(n_max), time_to_all[ii, 1]/float(n_max))
                if result_full.decision is not Decision.FailToDecide:
                    full_is_finished = True
            
            if partial_is_finished:
                pass
            else:
                result_partial = meta_procedure_test_partial_ranking.step(time_to_all_but_one[ii-1, 0]/float(n_max), time_to_all_but_one[ii, 1]/float(n_max))
                if result_partial.decision is not Decision.FailToDecide:
                    partial_is_finished = True
            
            print(f"On iteration {ii+1}")
            print(f"P-value of full: {np.minimum(meta_procedure_test_full_ranking._test_for_alternative._p_value, meta_procedure_test_full_ranking._test_for_null._p_value)}")
            print(f"P-value of partial: {np.minimum(meta_procedure_test_partial_ranking._test_for_alternative._p_value, meta_procedure_test_partial_ranking._test_for_null._p_value)}")
        
        stop_idx = ii
        if full_is_finished and partial_is_finished:
            print(f"Terminating loop early at step {ii+1} because meta-evaluation has completed")
            break
        

    # Print summary information
    print("Average time to full ranking")
    print(f"NSCORE: {np.mean(time_to_all[:iteration_limit, 0])}")
    print(f"WSR: {np.mean(time_to_all[:iteration_limit, 1])}")
    print()
    print("Average time to separating N-1 policies")
    print(f"NSCORE: {np.mean(time_to_all_but_one[:iteration_limit, 0])}")
    print(f"WSR: {np.mean(time_to_all_but_one[:iteration_limit, 1])}")
    print()
    breakpoint()