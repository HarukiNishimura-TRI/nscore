import numpy as np 
from sequentialized_barnard_tests.nsm import MirroredPartialCreditNsmTest, PartialCreditNsmTest
from sequentialized_barnard_tests.base import Hypothesis, Decision
from scipy import stats 
import warnings 

if __name__ == "__main__":
    model_name_list = [
        "Policy A",
        "Policy B",
        "Policy C",
    ]
    progress_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # Define progress bins from 0% to 100% in increments of 20%.

    progress_array_list = [
        np.array([0.0, 0.4, 0.4, 0.2, 0.4, 0.0, 0.2, 0.4, 0.4, 0.4, 0.2, 0.4, 0.0]),  # For Method A.
        np.array([0.8, 0.8, 0.8, 1.0, 0.8, 0.6, 0.8, 0.8, 0.8, 0.4, 1.0, 1.0, 0.6]),  # For Method B.
        np.array([0.6, 0.4, 0.4, 0.4, 0.6, 0.4, 0.8, 0.4, 0.4, 0.2, 0.6, 0.4, 0.2]),  # For Method C.
    ]
    
    print()
    c = np.array(progress_bins)
    K = int(c.shape[0] - 1)

    n_policies = 3
    n_comparisons = n_policies * (n_policies - 1) // 2

    global_confidence_level = 0.95
    
    alpha = 1. - global_confidence_level
    alpha_bonferroni = alpha / n_comparisons

    pcnsm_array_dict = dict()
    progress_array_dict = dict()  # model_name -> progress_array
    for idx in np.arange(n_policies):
        model = model_name_list[idx]
        progress_array = progress_array_list[idx]
        pcnsm_progress_array = np.floor(K*progress_array + 1e-6).astype(int)
        progress_array_dict[model] = progress_array
        pcnsm_array_dict[model] = pcnsm_progress_array
        print(model + f" mean performance: {np.mean(progress_array)}")
        print()
    
    pcnsm_test = MirroredPartialCreditNsmTest(Hypothesis.P0LessThanP1, alpha=alpha_bonferroni, c=c)

    comparisons_dict_nsm = dict()       # (model_name_a, model_name_b) -> Decision
    comparisons_dict_welch = dict()     # (model_name_a, model_name_b) -> Decision
    p_value_dict_nsm = dict()
    p_value_dict_welch = dict()

    for idx_a in np.arange(n_policies - 1):
        for idx_b in np.arange(idx_a + 1, n_policies):
            model_a = model_name_list[idx_a]
            model_b = model_name_list[idx_b]
            array_a = progress_array_dict[model_a]
            array_b = progress_array_dict[model_b]
            min_sample_size = min(len(array_a), len(array_b))
            # Run Welch's t-test for non-binary data.
            # if min_sample_size < 50:
            #     warnings.warn(
            #         f"{model_a} vs. {model_b}: Sample size {min_sample_size} "
            #         "< 50 might be too small to apply Welch's t-test."
            #     )
            _, p_value_welch = stats.ttest_ind(array_a, array_b, equal_var=False)
            p_value_dict_welch[(model_a, model_b)] = p_value_welch
            if p_value_welch <= alpha_bonferroni:
                comparisons_dict_welch[(model_a, model_b)] = Decision.AcceptAlternative
            else:
                comparisons_dict_welch[(model_a, model_b)] = Decision.FailToDecide
            
            pcnsm_array_a = pcnsm_array_dict[model_a]
            pcnsm_array_b = pcnsm_array_dict[model_b]

            pcnsm_result = pcnsm_test.run_on_sequence(pcnsm_array_a, pcnsm_array_b)
            
            p_value_alt = pcnsm_result.info["result_for_alternative"].info["P-Value"]
            p_value_null = pcnsm_result.info["result_for_null"].info["P-Value"]
            
            p_value_dict_nsm[(model_a, model_b)] = np.minimum(p_value_alt, p_value_null)

            if p_value_alt <= alpha_bonferroni:
                comparisons_dict_nsm[(model_a, model_b)] = Decision.AcceptAlternative
            elif p_value_null <= alpha_bonferroni:
                comparisons_dict_nsm[(model_a, model_b)] = Decision.AcceptNull
            else:
                comparisons_dict_nsm[(model_a, model_b)] = Decision.FailToDecide
            
            print()
            print("Result for testing " + model_a + " vs " + model_b + ":")
            print(f"Significance cutoff: {alpha_bonferroni}")
            print(f"Welch p-value: {p_value_welch}")
            print(f"NSM p-value: {np.minimum(p_value_alt, p_value_null)}")

    
