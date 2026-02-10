import numpy as np 
from matplotlib import pyplot as plt

if __name__ == "__main__":

    P0 = np.load("data/Welch/P0.npy")
    P1 = np.load("data/Welch/P1.npy")
    MU_0 = np.load("data/Welch/MU_0.npy")
    MU_1 = np.load("data/Welch/MU_1.npy")
    AVG_TTD = np.load("data/Welch/AVG_TTD.npy")
    AVG_POWER = np.load("data/Welch/AVG_POWER.npy")
    AVG_FPR = np.load("data/Welch/AVG_FPR.npy")

    n_cases = AVG_FPR.shape[0]
    n_outcomes = P0.shape[1]
    c = np.arange(n_outcomes) / (n_outcomes - 1.)

    true_mean_0 = np.dot(c, P0[0, :])
    true_mean_1 = np.dot(c, P1[0, :])

    nominal_mean_0 = MU_0[0]
    nominal_mean_1 = MU_1[0]

    print("Nominal means: ", nominal_mean_0, nominal_mean_1)
    print("True means: ", true_mean_0, true_mean_1)

    fig_ttd, ax_ttd = plt.subplots(figsize=(10, 10))
    ax_ttd.hist([AVG_TTD[:, 0], AVG_TTD[:, 1], AVG_TTD[:, 2]])
    ax_ttd.legend(["Welch", "Welch (Bonferroni)", "NSM"])
    ax_ttd.set_xlabel("Average Time to Decision (steps)")
    ax_ttd.set_ylabel("Number of Instances (Counts)")
    fig_ttd.savefig("welch_and_nsm_histogram.png", dpi=200)

    fig_fpr, ax_fpr = plt.subplots(figsize=(10, 10))
    ax_fpr.plot([0.05, 0.05], [0, 50], 'k--')
    ax_fpr.hist([AVG_FPR[:, 0], AVG_FPR[:, 1], AVG_FPR[:, 2]])
    ax_fpr.legend(['alpha', "Welch", "Welch (Bonferroni)", "NSM"])
    ax_fpr.set_xlabel("Average FPR over 100 redraws (proportion)")
    ax_fpr.set_ylabel("Number of Instances (Counts)")
    fig_fpr.savefig("welch_and_nsm_fpr_histogram.png", dpi=200)
    
    fig0, ax0 = plt.subplots(figsize=(10, 10))
    ax0.plot(c, P0[0, :], 'k')
    ax0.plot(c, P1[0, :], 'r')
    fig0.savefig("Check_density.png")
