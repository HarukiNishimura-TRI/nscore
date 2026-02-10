"""
Docstring for sequentialized_barnard_tests.utils.utils_wsr
"""

import copy
import warnings
from typing import Dict, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

####################
# Waudby-Smith Ramdas (WSR) procedure for confidence sequences
####################

def LOG_MARTINGALE(m, c, lam_t, Z_norm):
    coeff_plus = np.minimum(lam_t, c / m)
    coeff_minus = np.minimum(lam_t, c / (1 - m))
    M1 = np.cumsum(np.log(1 + coeff_plus * (Z_norm - m)))
    M2 = np.cumsum(np.log(1 - coeff_minus * (Z_norm - m)))
    # return 0.5 * np.maximum(M1, M2)
    return np.maximum(M1, M2) - np.log(2)

def mean_cs_eff_corrected_membership_accelerated(
    Z, alpha, c=0.5, L=-1., U=1., previous_lb=None, previous_ub=None,
):
    n = len(Z)
    Z_norm = np.array([(zi - L) / (U - L) for zi in Z])

    initial_mean = np.mean(Z_norm)

    t_n = np.arange(1, n + 1)

    mu_hat = (0.5 + np.cumsum(Z_norm)) / (1.0 + t_n)
    sigma2hat_n = (0.25 + np.cumsum(np.power(Z_norm - mu_hat, 2))) / (1.0 + t_n)
    sigma2hat_tminus1_n = np.append(0.25, sigma2hat_n[:-1])

    lam_t = np.sqrt(
        2.0 * np.log(2.0 / alpha) / ((t_n) * np.log(1.0 + t_n) * sigma2hat_tminus1_n)
    )
    
    # Initialize interval storage
    C_alpha = []

    # Run lower bound
    if previous_lb is not None:
        tightest_valid_lb = (previous_lb - L) / (U - L)
    else:
        tightest_valid_lb = 0.
    
    current_lb = (initial_mean + tightest_valid_lb) / 2.
    gap_lb = np.abs(initial_mean - current_lb)
    while gap_lb > 1e-6:
        gap_lb /= 2.0
        current_martingale = LOG_MARTINGALE(current_lb, c, lam_t, Z_norm)

        if np.max(current_martingale) >= np.log(1.0 / alpha):
            tightest_valid_lb = float(current_lb)
            # print("Current ceiling: ", current_lb + 2*gap_lb)
            # Set new current_lb HIGHER
            current_lb += gap_lb
        else:
            current_lb -= gap_lb

    C_alpha.append(tightest_valid_lb * (U - L) + L)

    # Run upper bound
    if previous_ub is not None:
        tightest_valid_ub = (previous_ub - L) / (U - L)
    else:
        tightest_valid_ub = 1.
    
    current_ub = (tightest_valid_ub + initial_mean) / 2.0
    gap_ub = np.abs(current_ub - initial_mean)
    while gap_ub > 1e-6:
        gap_ub /= 2.0
        current_martingale = LOG_MARTINGALE(current_ub, c, lam_t, Z_norm)
        if np.max(current_martingale) >= np.log(1.0 / alpha):
            tightest_valid_ub = float(current_ub)
            current_ub -= gap_ub
        else:
            current_ub += gap_ub

    C_alpha.append(tightest_valid_ub * (U - L) + L)
    return C_alpha

# def get_confidence_sequence(
#     score_differences, alpha=0.05, c_wsr=0.99, L_wsr=-1, U_wsr=1, n_max=500
# ):
#     interval_lb = L_wsr * np.ones(n_max + 1)
#     interval_ub = U_wsr * np.ones(n_max + 1)
#     result = ["undecided" for _ in range(n_max + 1)]

#     for i in tqdm(range(1, n_max + 1)):
#         if i < n_max:
#             interval_lb[i], interval_ub[i] = (
#                 mean_cs_eff_corrected_membership_accelerated(
#                     score_differences[:i], alpha, c_wsr, L_wsr, U_wsr
#                 )
#             )
#         else:
#             interval_lb[i], interval_ub[i] = (
#                 mean_cs_eff_corrected_membership_accelerated(
#                     score_differences[:], alpha, c_wsr, L_wsr, U_wsr
#                 )
#             )
#         if interval_lb[i] > 0:
#             result[i] = "null"  # null is better
#         elif interval_ub[i] < 0:
#             result[i] = "alternative"  # alternative is better
#     return interval_lb, interval_ub, result


# def plot_example(interval_lb, interval_lb_ci, interval_ub, interval_ub_ci):
#     fig, ax = plt.subplots(figsize=(7, 5))
#     ax.plot(np.arange(n_max + 1), interval_lb, "k-")
#     ax.plot(np.arange(n_max + 1), interval_ub, "k-", label="CS")
#     ax.plot(np.arange(n_max + 1), np.zeros(n_max + 1), "r.", markersize=3)
#     ax.legend()
#     fig.savefig("wsr_test.png", dpi=300)
###############################################
# Partial credit policy comparison using WSR-CS
###############################################
# def compute_score_differences(c, data_policy_0, data_policy_1):
#     score_differences = []
#     for p0, p1 in zip(data_policy_0, data_policy_1):
#         score_diff = np.dot(c, p0 - p1)
#         score_differences.append(score_diff)
#     return np.array(score_differences)

###############################################
# Run main
###############################################

# if __name__ == "__main__":
#     n_cases_partial_credit = 5
#     alpha = 0.05
#     c_wsr = 0.99
#     L_wsr = -1.0
#     U_wsr = 1.0
#     n_max = 500

#     # Get score differences:
#     c = ...
#     data_policy_0 = ...
#     data_policy_1 = ...
#     score_differences = compute_score_differences(c, data_policy_0, data_policy_1)
#     interval_lb, interval_ub, result = get_confidence_sequence(
#         score_differences, alpha, c_wsr=c_wsr, L_wsr=L_wsr, U_wsr=U_wsr, n_max=n_max
#     )
#     plot_example(interval_lb, interval_ub)