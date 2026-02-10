import numpy as np 
from scipy import stats 
from sequentialized_barnard_tests.nsm import MirroredPartialCreditNsmTest, PartialCreditNsmTest
from sequentialized_barnard_tests.base import Hypothesis, Decision
from matplotlib import pyplot as plt

if __name__ == "__main__":
    model_name_list = [
        "Policy A",
        "Policy B",
        "Policy C",
    ]
    progress_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # Define progress bins from 0% to 100% in increments of 20%.

    progress_array_list = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # For Method A.
        np.array([0.8, 0.8, 0.8, 1.0, 0.8, 0.6, 0.8, 0.8, 0.8, 0.4, 1.0, 1.0, 0.6]),  # For Method B.
        np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),  # For Method C.
    ]
    critical_probability = 0.7995 ** 13
    print()
    print("This is a counterexample designed to illustrate that Welch is not Type-1 Error controlling.")
    print(f"The outcomes (scores) vector is: {progress_bins}")
    print(f"We desire to run Welch's t-test at alpha = 0.05")
    print(f"Policy A's distribution is {np.array([0.7995, 0., 0., 0., 0., 0.2005])}. It's mean performance is 0.2005")
    print(f"Policy C's distribution is {np.array([0.5, 0., 0.5, 0., 0., 0.])}. It's mean performance is 0.2")
    print()
    print("By definition, Policy A has higher mean performance than Policy C")
    print()
    print("However.")
    print()
    print(f"It can be trivially shown that for 13 data points, the probability that Policy A observes ALL 0's is 0.7995 ** 13 = {critical_probability}.")
    print(f"Because {critical_probability} > 0.05, the specified data given in this script occurs MORE than alpha of the time.")
    print("Thus, if Welch's t-test makes an error on this data, it is not Type-1 Error controlling at level alpha.")
    print(f"Further, if it returns a p-value of P < {critical_probability}, it is not Type-1 Error controlling for all p in (P, {critical_probability}), just from this single counterexample. ")

    print()
    print("Running the test")
    print()
    _, p_value_welch = stats.ttest_ind(progress_array_list[0], progress_array_list[2], equal_var=False)

    nsm_test = PartialCreditNsmTest(alternative=Hypothesis.P0MoreThanP1, alpha=0.05, c=np.array(progress_bins))
    mirrored_nsm_test = MirroredPartialCreditNsmTest(alternative=Hypothesis.P0MoreThanP1, alpha=0.05, c=np.array(progress_bins))
    nsm_test.reset()
    mirrored_nsm_test.reset()

    LAMBDAS = []
    for i in range(27):
        nsm_result = nsm_test.step(int(1), int(0))
        mirrored_nsm_result = mirrored_nsm_test.step(int(1), int(0))
        LAMBDAS.append(nsm_test.lambda_parameter)
    
    print("The Welch test p-value is: ", p_value_welch)
    print("The NSM test p-value is: ", nsm_result.info["P-Value"])
    print("The mirrored NSM test p-value is: ", mirrored_nsm_result.info["result_for_alternative"].info["P-Value"])

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(LAMBDAS)
    fig.savefig("tmp_lambda.png", dpi=100)
