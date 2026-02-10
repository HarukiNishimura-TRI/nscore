import numpy as np

if __name__ == "__main__":

    alpha = 0.05
    lambda_parameter = 0.5

    RESULTS_NULL = np.load(
        f"data/NSM/Results_Null_alpha_{alpha}.npy"
    )
    RESULTS_ALT = np.load(
        f"data/NSM/Results_Alt_alpha_{alpha}.npy"
    )

    epsilon_offset = 1e-6

    TTD_BINARY_NSM = np.zeros((10, 10))
    TTD_PC_NSM = np.zeros((10, 10))
    TTD_CONT_NSM = np.zeros((10, 10))
    TTD_WSR = np.zeros((10, 10))
    TTD_SAVI = np.zeros((10, 10))
    TTD_STEP = np.zeros((10, 10))

    POWER_BINARY_NSM = np.zeros((10, 10))
    POWER_PC_NSM = np.zeros((10, 10))
    POWER_CONT_NSM = np.zeros((10, 10))
    POWER_WSR = np.zeros((10, 10))
    POWER_SAVI = np.zeros((10, 10))
    POWER_STEP = np.zeros((10, 10))

    for ii in range(RESULTS_NULL.shape[0]):
        assert np.isclose(RESULTS_NULL[ii, 0, 0], RESULTS_NULL[ii, 0, 1])

        p0 = RESULTS_NULL[ii, 0, 0]
        p1 = RESULTS_NULL[ii, 0, 1]

        idx0 = int(np.floor(epsilon_offset + (p0 * 10 - 0.5)))
        idx1 = int(np.floor(epsilon_offset + (p1 * 10)))

        TTD_BINARY_NSM[idx0, idx1] = RESULTS_NULL[ii, 1, 0]
        TTD_PC_NSM[idx0, idx1] = RESULTS_NULL[ii, 1, 1]
        TTD_CONT_NSM[idx0, idx1] = RESULTS_NULL[ii, 1, 2]
        TTD_WSR[idx0, idx1] = RESULTS_NULL[ii, 1, 3]
        TTD_SAVI[idx0, idx1] = RESULTS_NULL[ii, 1, 4]
        TTD_STEP[idx0, idx1] = RESULTS_NULL[ii, 1, 5]

        POWER_BINARY_NSM[idx0, idx1] = RESULTS_NULL[ii, 2, 0]
        POWER_PC_NSM[idx0, idx1] = RESULTS_NULL[ii, 2, 1]
        POWER_CONT_NSM[idx0, idx1] = RESULTS_NULL[ii, 2, 2]
        POWER_WSR[idx0, idx1] = RESULTS_NULL[ii, 2, 3]
        POWER_SAVI[idx0, idx1] = RESULTS_NULL[ii, 2, 4]
        POWER_STEP[idx0, idx1] = RESULTS_NULL[ii, 2, 5]
        

    for jj in range(RESULTS_ALT.shape[0]):
        p0 = RESULTS_ALT[jj, 0, 0]
        p1 = RESULTS_ALT[jj, 0, 1]

        assert p1 >= p0 and p1 <= 1.0

        idx0 = int(np.floor(epsilon_offset + (p0 * 10 - 0.5)))
        idx1 = int(np.floor(epsilon_offset + (p1 * 10)))

        TTD_BINARY_NSM[idx0, idx1] = RESULTS_ALT[jj, 1, 0]
        TTD_PC_NSM[idx0, idx1] = RESULTS_ALT[jj, 1, 1]
        TTD_CONT_NSM[idx0, idx1] = RESULTS_ALT[jj, 1, 2]
        TTD_WSR[idx0, idx1] = RESULTS_ALT[jj, 1, 3]
        TTD_SAVI[idx0, idx1] = RESULTS_ALT[jj, 1, 4]
        TTD_STEP[idx0, idx1] = RESULTS_ALT[jj, 1, 5]

        POWER_BINARY_NSM[idx0, idx1] = RESULTS_ALT[jj, 2, 0]
        POWER_PC_NSM[idx0, idx1] = RESULTS_ALT[jj, 2, 1]
        POWER_CONT_NSM[idx0, idx1] = RESULTS_ALT[jj, 2, 2]
        POWER_WSR[idx0, idx1] = RESULTS_ALT[jj, 2, 3]
        POWER_SAVI[idx0, idx1] = RESULTS_ALT[jj, 2, 4]
        POWER_STEP[idx0, idx1] = RESULTS_ALT[jj, 2, 5]
    
    np.save(f"data/NSM/TTD_BINARY_NSM_alpha_{alpha}.npy", TTD_BINARY_NSM)
    np.save(f"data/NSM/TTD_PC_NSM_alpha_{alpha}.npy", TTD_PC_NSM)
    np.save(f"data/NSM/TTD_CONT_NSM_alpha_{alpha}.npy", TTD_CONT_NSM)
    np.save(f"data/NSM/TTD_WSR_alpha_{alpha}.npy", TTD_WSR)
    np.save(f"data/NSM/TTD_SAVI_alpha_{alpha}.npy", TTD_SAVI)
    np.save(f"data/NSM/TTD_STEP_alpha_{alpha}.npy", TTD_STEP)

    np.save(
        f"data/NSM/POWER_BINARY_NSM_alpha_{alpha}.npy", POWER_BINARY_NSM
    )
    np.save(
        f"data/NSM/POWER_PC_NSM_alpha_{alpha}.npy", POWER_PC_NSM
    )
    np.save(
        f"data/NSM/POWER_CONT_NSM_alpha_{alpha}.npy", POWER_CONT_NSM
    )
    np.save(
        f"data/NSM/POWER_WSR_alpha_{alpha}.npy", POWER_WSR
    )
    np.save(
        f"data/NSM/POWER_SAVI_alpha_{alpha}.npy", POWER_SAVI
    )
    np.save(
        f"data/NSM/POWER_STEP_alpha_{alpha}.npy", POWER_STEP
    )
