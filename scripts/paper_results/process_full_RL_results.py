import numpy as np 

import sys 
import os
from copy import deepcopy

if __name__ == "__main__":

    base_path = "data/RL/ALL_RUNS/"
    save_path = "data/RL/Task_and_Method/"

    methods = ["ddpg", "ppo", "sac", "td3"]
    tasks = ["Ant", "Cheetah", "Hopper", "Humanoid", "Pendulum", "Pusher", "Walker"]

    # Test subdirectory scan 
    subdirs = [f.path for f in os.scandir(base_path) if f.is_dir()]
    # print(len(subdirs))
    # print(subdirs[0])
    # breakpoint()

    for i in range(len(methods)):
        method_string = methods[i]
        for j in range(len(tasks)):
            task_string = tasks[j]
            print("Save path: ", save_path + task_string + "_" + method_string + ".npy")
            breakpoint()
            try:
                del data_array
            except:
                pass

            data_array = np.zeros((500000, ))
            legit_index = int(0)
            for k in range(10):
                index_string = "_" + str(k+1) + "_"

                found_correct_idx = False
                for ii in range(len(subdirs)):
                    path_string = str(subdirs[ii])

                    if (method_string in path_string) and (task_string in path_string) and (index_string in path_string):
                        found_correct_idx = True 
                        path_string_to_use = path_string
                        for jj in range(50):
                            try:
                                tmp_data = np.load(path_string_to_use + f"/episodic_returns_{jj}.npy")[:, 0]
                                added_size = tmp_data.shape[0]
                                data_array[legit_index:(legit_index+added_size)] = deepcopy(tmp_data)
                                legit_index += int(added_size)
                            except:
                                pass 
                    else:
                        pass

                if found_correct_idx == False:
                    raise ValueError("Had a missing case where a pathstring could not be found")
            
            np.save(save_path + task_string + "_" + method_string + ".npy", data_array[:legit_index])
