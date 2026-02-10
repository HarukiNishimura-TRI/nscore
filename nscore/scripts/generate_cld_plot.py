import numpy as np

from sequentialized_barnard_tests.tools.plotting import (
    nonasymptotically_valid_compare_progress_and_get_cld, 
    plot_model_comparison
)

from tqdm import tqdm 

if __name__ == "__main__":

    # Specify global parameters
    global_confidence_level = 0.95 
    max_sample_size_per_model = 1000 
    shuffle=False 

    have_data = False

    K_progress = 40
    progress_bins = np.arange(K_progress + 1) / K_progress 

    if have_data:
        pass
        # Load your data! 
        # data_for_sac = np.load("sac_eval_scores.npy")
        # data_for_ppo = ...
        # data_for_td3 = ...
        # data_for_ddpg = ...
        # etc

        # You may need to rectify the data for the CLD computations
        # TRY WITHOUT DOING THIS, FIRST!
        # To do this: once you have defined progress_bins = np.arange(K_progress+1) / K_progress, 
        # Simply take the scaled-to-[0, 1] RL rewards, multiply by K_progress, and floor the results. Then, divide by K_progress
        # That is: 
        # Rectified_data_for_sac = np.floor(K_progress * data_for_sac) / K_progress
        # etc. 
    else:
        # In lieu of data to load, generate artificially
        dist_sac = np.ones(41)/41.
        dist_sac /= np.sum(dist_sac)

        dist_ppo = np.sin(2.*np.pi*2*progress_bins) + 1. 
        dist_ppo /= np.sum(dist_ppo)
        
        dist_td3 = progress_bins**2 - 0.4*progress_bins + 0.04
        dist_td3 += np.maximum(-np.min(dist_td3), 0.)
        dist_td3 /= np.sum(dist_td3)

        dist_ddpg = progress_bins*2.
        dist_ddpg /= np.sum(dist_ddpg)

        nominal_means = np.zeros(4)
        empirical_means = np.zeros(4)

        data_for_sac = np.random.choice(41, size=max_sample_size_per_model, replace=True, p=dist_sac) / 40.
        data_for_ppo = np.random.choice(41, size=max_sample_size_per_model, replace=True, p=dist_ppo) / 40.
        data_for_td3 = np.random.choice(41, size=max_sample_size_per_model, replace=True, p=dist_td3) / 40.
        data_for_ddpg = np.random.choice(41, size=max_sample_size_per_model, replace=True, p=dist_ddpg) / 40.

        nominal_means[0] = np.inner(progress_bins, dist_sac)
        nominal_means[1] = np.inner(progress_bins, dist_ppo)
        nominal_means[2] = np.inner(progress_bins, dist_td3)
        nominal_means[3] = np.inner(progress_bins, dist_ddpg)

        print("Nominal Means: ", nominal_means)

    empirical_means[0] = np.mean(data_for_sac)
    empirical_means[1] = np.mean(data_for_ppo)
    empirical_means[2] = np.mean(data_for_td3)
    empirical_means[3] = np.mean(data_for_ddpg)
    
    print("Empirical Means: ", empirical_means)

    print()
    print("Beginning CLD Computations...")
    print()
    # Make sure this matches the order in the 
    model_name_list = ['sac', 'ppo', 'td3', 'ddpg']

    progress_array_list = [
        data_for_sac, 
        data_for_ppo, 
        data_for_td3,
        data_for_ddpg,
    ]

    cld_dict = nonasymptotically_valid_compare_progress_and_get_cld(
        model_name_list,
        progress_array_list,
        global_confidence_level,
        verbose=True,
    )

    cld_list = [cld_dict[key] for key in model_name_list]

    plot_model_comparison(
        model_name_list,
        progress_array_list,
        cld_list,
        rng=np.random.default_rng(123), # For plotting violins
        mode="task_progress",
        progress_bins=progress_bins,
        output_path="tmp_violin.png"
    )
    print("Plot generated. Two models that do not share any CLD letters are significantly different.")

