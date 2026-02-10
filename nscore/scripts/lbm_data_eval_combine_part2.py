import numpy as np


if __name__ == "__main__":

    # Load the data
    file_path_2 = 'data/LBM/LBM_RESULTS_PART_2.npy'
    RESULTS_2 = np.load(file_path_2)

    n_tasks_2 = RESULTS_2.shape[0]
    n_resamples_2 = RESULTS_2.shape[-1]

    methods = [
        'pc_nsm_progress',
        'cont_nsm_progress', 
        'wsr_progress', 
        'binary_step_success', 
        'binary_nsm_success', 
        'binary_savi_success', 
        'binary_wsr_success',
    ]
    tasks = [
        'BikeRotorInstall',
        'CutAppleIntoSlices',
        'CleanLitterBox',
        'ClearKitchenCounter',
        'SetUpBreakfastTable',
    ]
    correct_idx = 0
    ttd_idx = 1

    for j in range(len(methods)):
        print("Method: ", methods[j])
        print()
        for i in range(n_tasks_2):
            correct_idx_array = np.argwhere(RESULTS_2[i, j, correct_idx, :] >= 0.5)
            print("Task: ", tasks[i])
            print(f"Correctness: {np.mean(RESULTS_2[i, j, correct_idx, :])}; Average TTD: {np.mean(RESULTS_2[i, j, ttd_idx, :])}")
            if np.size(correct_idx_array) > 0:
                print(f"Average TTCD: {np.mean(RESULTS_2[i, j, ttd_idx, correct_idx_array])}")
            else:
                print("Zero correctness")
            print()
        
        breakpoint()