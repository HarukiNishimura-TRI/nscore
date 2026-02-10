import numpy as np
from matplotlib import pyplot as plt
import os 
import sys
add_path = os.getcwd()
sys.path.append(add_path)

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "Helvetica"
# })

if __name__ == "__main__":

    alpha = 0.05

    TTD_BINARY_NSM = np.load(f"data/BINARY/TTD_BINARY_NSM_alpha_{alpha}.npy")
    TTD_PC_NSM = np.load(f"data/BINARY/TTD_PC_NSM_alpha_{alpha}.npy")
    TTD_CONT_NSM = np.load(f"data/BINARY/TTD_CONT_NSM_alpha_{alpha}.npy")
    TTD_WSR = np.load(f"data/BINARY/TTD_WSR_alpha_{alpha}.npy")
    TTD_SAVI = np.load(f"data/BINARY/TTD_SAVI_alpha_{alpha}.npy")
    TTD_STEP = np.load(f"data/BINARY/TTD_STEP_alpha_{alpha}.npy")

    POWER_BINARY_NSM = np.load(
        f"data/BINARY/POWER_BINARY_NSM_alpha_{alpha}.npy"
    )
    POWER_PC_NSM = np.load(
        f"data/BINARY/POWER_PC_NSM_alpha_{alpha}.npy"
    )
    POWER_CONT_NSM = np.load(
        f"data/BINARY/POWER_CONT_NSM_alpha_{alpha}.npy"
    )
    POWER_WSR = np.load(
        f"data/BINARY/POWER_WSR_alpha_{alpha}.npy"
    )
    POWER_SAVI = np.load(
        f"data/BINARY/POWER_SAVI_alpha_{alpha}.npy"
    )
    POWER_STEP = np.load(
        f"data/BINARY/POWER_STEP_alpha_{alpha}.npy"
    )

    #####
    # TIMES TO DECISION
    #####

    # BINARY NSM
    fig_ttd_binary_nsm, ax_ttd_binary_nsm = plt.subplots(figsize=(10, 10))
    ax_ttd_binary_nsm.pcolormesh(
        np.arange(11) / (10),
        np.arange(11) / (10),
        np.transpose(TTD_BINARY_NSM),
        cmap="RdYlGn",
        vmin=0.0,
        vmax=500,
    )
    total_binary_nsm_complexity = 0
    total_binary_nsm_power = 0.
    number_alternatives = 0
    for x in range(10):
        for y in range(10):
            if TTD_BINARY_NSM[x, y] > 3:
                if x != y:
                    number_alternatives += 1
                    assert TTD_BINARY_NSM[x, y] <= 850
                    total_binary_nsm_complexity += TTD_BINARY_NSM[x, y]
                    total_binary_nsm_power += POWER_BINARY_NSM[x, y]
                    ax_ttd_binary_nsm.text(
                        (x + 0.5) / 10,
                        (y + 0.5) / 10,
                        f"{TTD_BINARY_NSM[x,y]:2.2f}",  # data[y,x] +0.05 , data[y,x] + 0.05
                        horizontalalignment="center",
                        verticalalignment="center",
                        color="k",
                    )
    ax_ttd_binary_nsm.plot([0, 1], [0, 1], "k--", linewidth=5)
    # ax_ttd_nsm.set_aspect("equal")
    ax_ttd_binary_nsm.grid(True)
    ax_ttd_binary_nsm.set_xlabel("Baseline Policy Performance")
    fig_ttd_binary_nsm.savefig(f"data/BINARY/TTD_BINARY_NSM.png", dpi=300)

    # PC NSM
    fig_ttd_pc_nsm, ax_ttd_pc_nsm = plt.subplots(figsize=(10, 10))
    ax_ttd_pc_nsm.pcolormesh(
        np.arange(11) / (10),
        np.arange(11) / (10),
        np.transpose(TTD_PC_NSM),
        cmap="RdYlGn",
        vmin=0.0,
        vmax=500,
    )
    for x in range(10):
        for y in range(10):
            if TTD_PC_NSM[x, y] > 3:
                if x != y:
                    ax_ttd_pc_nsm.text(
                        (x + 0.5) / 10,
                        (y + 0.5) / 10,
                        f"{TTD_PC_NSM[x,y]:2.2f}",  # data[y,x] +0.05 , data[y,x] + 0.05
                        horizontalalignment="center",
                        verticalalignment="center",
                        color="k",
                    )
    ax_ttd_pc_nsm.plot([0, 1], [0, 1], "k--", linewidth=5)
    # ax_ttd_nsm.set_aspect("equal")
    ax_ttd_pc_nsm.grid(True)
    ax_ttd_pc_nsm.set_xlabel("Baseline Policy Performance")
    fig_ttd_pc_nsm.savefig(f"data/BINARY/TTD_PC_NSM.png", dpi=300)

    # CONT NSM
    fig_ttd_cont_nsm, ax_ttd_cont_nsm = plt.subplots(figsize=(10, 10))
    ax_ttd_cont_nsm.pcolormesh(
        np.arange(11) / (10),
        np.arange(11) / (10),
        np.transpose(TTD_CONT_NSM),
        cmap="RdYlGn",
        vmin=0.0,
        vmax=500,
    )
    total_cont_nsm_complexity = 0
    total_cont_nsm_power = 0.
    for x in range(10):
        for y in range(10):
            if TTD_CONT_NSM[x, y] > 3:
                if x != y:
                    assert TTD_CONT_NSM[x, y] <= 850
                    total_cont_nsm_complexity += TTD_CONT_NSM[x, y]
                    total_cont_nsm_power += POWER_CONT_NSM[x, y]
                    ax_ttd_cont_nsm.text(
                        (x + 0.5) / 10,
                        (y + 0.5) / 10,
                        f"{TTD_CONT_NSM[x,y]:2.2f}",  # data[y,x] +0.05 , data[y,x] + 0.05
                        horizontalalignment="center",
                        verticalalignment="center",
                        color="k",
                    )
    ax_ttd_cont_nsm.plot([0, 1], [0, 1], "k--", linewidth=5)
    # ax_ttd_nsm.set_aspect("equal")
    ax_ttd_cont_nsm.grid(True)
    ax_ttd_cont_nsm.set_xlabel("Baseline Policy Performance")
    fig_ttd_cont_nsm.savefig(f"data/BINARY/TTD_CONT_NSM.png", dpi=300)

    # WSR
    fig_ttd_wsr, ax_ttd_wsr = plt.subplots(figsize=(10, 10))
    ax_ttd_wsr.pcolormesh(
        np.arange(11) / (10),
        np.arange(11) / (10),
        np.transpose(TTD_WSR),
        cmap="RdYlGn",
        vmin=0.0,
        vmax=500,
    )
    total_wsr_complexity = 0
    total_wsr_power = 0.
    for x in range(10):
        for y in range(10):
            if TTD_WSR[x, y] > 3:
                if x != y:
                    assert TTD_WSR[x, y] <= 900
                    total_wsr_complexity += TTD_WSR[x, y]
                    total_wsr_power += POWER_WSR[x, y]
                    ax_ttd_wsr.text(
                        (x + 0.5) / 10,
                        (y + 0.5) / 10,
                        f"{TTD_WSR[x,y]:2.2f}",  # data[y,x] +0.05 , data[y,x] + 0.05
                        horizontalalignment="center",
                        verticalalignment="center",
                        color="k",
                    )
    ax_ttd_wsr.plot([0, 1], [0, 1], "k--", linewidth=5)
    # ax_ttd_nsm.set_aspect("equal")
    ax_ttd_wsr.grid(True)
    ax_ttd_wsr.set_xlabel("Baseline Policy Performance")
    fig_ttd_wsr.savefig(f"data/BINARY/TTD_WSR.png", dpi=300)

    # SAVI
    fig_ttd_savi, ax_ttd_savi = plt.subplots(figsize=(10, 10))
    ax_ttd_savi.pcolormesh(
        np.arange(11) / (10),
        np.arange(11) / (10),
        np.transpose(TTD_SAVI),
        cmap="RdYlGn",
        vmin=0.0,
        vmax=500,
    )
    total_savi_complexity = 0
    total_savi_power = 0.
    for x in range(10):
        for y in range(10):
            if TTD_SAVI[x, y] > 3:
                if x != y:
                    assert TTD_SAVI[x, y] <= 850
                    total_savi_complexity += TTD_SAVI[x, y]
                    total_savi_power += POWER_SAVI[x, y]
                    ax_ttd_savi.text(
                        (x + 0.5) / 10,
                        (y + 0.5) / 10,
                        f"{TTD_SAVI[x,y]:2.2f}",  # data[y,x] +0.05 , data[y,x] + 0.05
                        horizontalalignment="center",
                        verticalalignment="center",
                        color="k",
                    )
    ax_ttd_savi.plot([0, 1], [0, 1], "k--", linewidth=5)
    # ax_ttd_savi.set_aspect("equal")
    ax_ttd_savi.grid(True)
    fig_ttd_savi.savefig(f"data/BINARY/TTD_SAVI.png", dpi=300)

    # STEP
    fig_ttd_step, ax_ttd_step = plt.subplots(figsize=(10, 10))
    ax_ttd_step.pcolormesh(
        np.arange(11) / (10),
        np.arange(11) / (10),
        np.transpose(TTD_STEP),
        cmap="RdYlGn",
        vmin=0.0,
        vmax=500,
    )
    total_step_complexity = 0
    total_step_power = 0.
    for x in range(10):
        for y in range(10):
            if TTD_STEP[x, y] > 3:
                if x != y:
                    assert TTD_STEP[x, y] <= 850
                    total_step_complexity += TTD_STEP[x, y]
                    total_step_power += POWER_STEP[x, y]
                    ax_ttd_step.text(
                        (x + 0.5) / 10,
                        (y + 0.5) / 10,
                        f"{TTD_STEP[x,y]:2.2f}",  # data[y,x] +0.05 , data[y,x] + 0.05
                        horizontalalignment="center",
                        verticalalignment="center",
                        color="k",
                    )
    ax_ttd_step.plot([0, 1], [0, 1], "k--", linewidth=5)
    # ax_ttd_savi.set_aspect("equal")
    ax_ttd_step.grid(True)
    fig_ttd_step.savefig(f"data/BINARY/TTD_STEP.png", dpi=300)

    # SAVI / Binary NSM DIFFERENCE
    fig_ttd_diff, ax_ttd_diff = plt.subplots(figsize=(10, 10))
    ax_ttd_diff.pcolormesh(
        np.arange(11) / (10),
        np.arange(11) / (10),
        np.transpose(TTD_SAVI)-np.transpose(TTD_BINARY_NSM),
        cmap="RdYlGn",
        vmin=-10.0,
        vmax=50.0,
    )
    for x in range(10):
        for y in range(10):
            if TTD_SAVI[x, y] > 3:
                if x != y:
                    ax_ttd_diff.text(
                        (x + 0.5) / 10,
                        (y + 0.5) / 10,
                        f"{(TTD_SAVI[x,y]-TTD_BINARY_NSM[x,y]):2.2f}",  # data[y,x] +0.05 , data[y,x] + 0.05
                        horizontalalignment="center",
                        verticalalignment="center",
                        color="k",
                    )
    ax_ttd_diff.plot([0, 1], [0, 1], "k--", linewidth=5)
    # ax_ttd_savi.set_aspect("equal")
    ax_ttd_diff.grid(True)
    fig_ttd_diff.savefig(f"data/BINARY/TTD_DIFF.png", dpi=300)

    # Nonparametric NSM / WSR DIFFERENCE
    fig_ttd_nonparametric_diff, ax_ttd_nonparametric_diff = plt.subplots(figsize=(10, 10))
    ax_ttd_nonparametric_diff.pcolormesh(
        np.arange(11) / (10),
        np.arange(11) / (10),
        np.transpose(TTD_WSR)-np.transpose(TTD_CONT_NSM),
        cmap="RdYlGn",
        vmin=0.0,
        vmax=300.0,
    )
    for x in range(10):
        for y in range(10):
            if TTD_CONT_NSM[x, y] > 3:
                if x!= y:
                    ax_ttd_nonparametric_diff.text(
                        (x + 0.5) / 10,
                        (y + 0.5) / 10,
                        f"{(TTD_WSR[x,y]-TTD_CONT_NSM[x,y]):2.2f}",  # data[y,x] +0.05 , data[y,x] + 0.05
                        horizontalalignment="center",
                        verticalalignment="center",
                        color="k",
                    )
    ax_ttd_nonparametric_diff.plot([0, 1], [0, 1], "k--", linewidth=5)
    # ax_ttd_savi.set_aspect("equal")
    ax_ttd_nonparametric_diff.grid(True)
    fig_ttd_nonparametric_diff.savefig(f"data/BINARY/TTD_NONPARAMETRIC_DIFF.png", dpi=300)

    #####
    # TERMINAL POWER
    #####

    # BINARY NSM
    fig_binary_NSM_terminal_power, ax_binary_NSM_terminal_power = plt.subplots(figsize=(10, 10))
    ax_binary_NSM_terminal_power.pcolormesh(
        np.arange(11) / (10),
        np.arange(11) / (10),
        np.transpose(POWER_BINARY_NSM),
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
    )
    for x in range(10):
        for y in range(10):
            if POWER_BINARY_NSM[x,y] >= 0.03:
                if x!= y:
                    ax_binary_NSM_terminal_power.text(
                        (x + 0.5) / 10,
                        (y + 0.5) / 10,
                        f"{POWER_BINARY_NSM[x,y]:2.2f}",  # data[y,x] +0.05 , data[y,x] + 0.05
                        horizontalalignment="center",
                        verticalalignment="center",
                        color="k",
                    )

    ax_binary_NSM_terminal_power.set_title(
        f"BINARY NSM Terminal Power: Nmax={500}, alpha={alpha}", fontsize=20
    )
    ax_binary_NSM_terminal_power.plot([0, 1], [0, 1], "k--", linewidth=5)
    ax_binary_NSM_terminal_power.set_xlabel("Baseline Performance", fontsize=24)
    ax_binary_NSM_terminal_power.set_ylabel("Delta Test Policy Performance", fontsize=24)
    ax_binary_NSM_terminal_power.tick_params(labelsize=20)
    # ax_NSM_terminal_power.set_aspect("equal")
    ax_binary_NSM_terminal_power.grid(True)
    fig_binary_NSM_terminal_power.savefig(f"data/BINARY/POWER_BINARY_NSM.png", dpi=450)

    # PC NSM
    fig_pc_NSM_terminal_power, ax_pc_NSM_terminal_power = plt.subplots(figsize=(10, 10))
    ax_pc_NSM_terminal_power.pcolormesh(
        np.arange(11) / (10),
        np.arange(11) / (10),
        np.transpose(POWER_PC_NSM),
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
    )
    for x in range(10):
        for y in range(10):
            if POWER_BINARY_NSM[x,y] >= 0.03:
                if x!= y:
                    ax_pc_NSM_terminal_power.text(
                        (x + 0.5) / 10,
                        (y + 0.5) / 10,
                        f"{POWER_PC_NSM[x,y]:2.2f}",  # data[y,x] +0.05 , data[y,x] + 0.05
                        horizontalalignment="center",
                        verticalalignment="center",
                        color="k",
                    )

    ax_pc_NSM_terminal_power.set_title(
        f"PC NSM Terminal Power: Nmax={500}, alpha={alpha}", fontsize=20
    )
    ax_pc_NSM_terminal_power.plot([0, 1], [0, 1], "k--", linewidth=5)
    ax_pc_NSM_terminal_power.set_xlabel("Baseline Performance", fontsize=24)
    ax_pc_NSM_terminal_power.set_ylabel("Delta Test Policy Performance", fontsize=24)
    ax_pc_NSM_terminal_power.tick_params(labelsize=20)
    # ax_NSM_terminal_power.set_aspect("equal")
    ax_pc_NSM_terminal_power.grid(True)
    fig_pc_NSM_terminal_power.savefig(f"data/BINARY/POWER_PC_NSM.png", dpi=450)

    # CONT NSM
    fig_cont_NSM_terminal_power, ax_cont_NSM_terminal_power = plt.subplots(figsize=(10, 10))
    ax_cont_NSM_terminal_power.pcolormesh(
        np.arange(11) / (10),
        np.arange(11) / (10),
        np.transpose(POWER_CONT_NSM),
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
    )
    for x in range(10):
        for y in range(10):
            if POWER_BINARY_NSM[x,y] >= 0.03:
                if x!= y:
                    ax_cont_NSM_terminal_power.text(
                        (x + 0.5) / 10,
                        (y + 0.5) / 10,
                        f"{POWER_CONT_NSM[x,y]:2.2f}",  # data[y,x] +0.05 , data[y,x] + 0.05
                        horizontalalignment="center",
                        verticalalignment="center",
                        color="k",
                    )

    ax_cont_NSM_terminal_power.set_title(
        f"CONT NSM Terminal Power: Nmax={500}, alpha={alpha}", fontsize=20
    )
    ax_cont_NSM_terminal_power.plot([0, 1], [0, 1], "k--", linewidth=5)
    ax_cont_NSM_terminal_power.set_xlabel("Baseline Performance", fontsize=24)
    ax_cont_NSM_terminal_power.set_ylabel("Delta Test Policy Performance", fontsize=24)
    ax_cont_NSM_terminal_power.tick_params(labelsize=20)
    # ax_NSM_terminal_power.set_aspect("equal")
    ax_cont_NSM_terminal_power.grid(True)
    fig_cont_NSM_terminal_power.savefig(f"data/BINARY/POWER_CONT_NSM.png", dpi=450)

    # WSR
    fig_WSR_terminal_power, ax_WSR_terminal_power = plt.subplots(figsize=(10, 10))
    ax_WSR_terminal_power.pcolormesh(
        np.arange(11) / (10),
        np.arange(11) / (10),
        np.transpose(POWER_WSR),
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
    )
    for x in range(10):
        for y in range(10):
            if POWER_BINARY_NSM[x,y] >= 0.03:
                if x!= y:
                    ax_WSR_terminal_power.text(
                        (x + 0.5) / 10,
                        (y + 0.5) / 10,
                        f"{POWER_WSR[x,y]:2.2f}",  # data[y,x] +0.05 , data[y,x] + 0.05
                        horizontalalignment="center",
                        verticalalignment="center",
                        color="k",
                    )

    ax_WSR_terminal_power.set_title(
        f"WSR Terminal Power: Nmax={500}, alpha={alpha}", fontsize=20
    )
    ax_WSR_terminal_power.plot([0, 1], [0, 1], "k--", linewidth=5)
    ax_WSR_terminal_power.set_xlabel("Baseline Performance", fontsize=24)
    ax_WSR_terminal_power.set_ylabel("Delta Test Policy Performance", fontsize=24)
    ax_WSR_terminal_power.tick_params(labelsize=20)
    # ax_NSM_terminal_power.set_aspect("equal")
    ax_WSR_terminal_power.grid(True)
    fig_WSR_terminal_power.savefig(f"data/BINARY/POWER_WSR.png", dpi=450)

    # SAVI
    fig_SAVI_terminal_power, ax_SAVI_terminal_power = plt.subplots(figsize=(10, 10))
    ax_SAVI_terminal_power.pcolormesh(
        np.arange(11) / (10),
        np.arange(11) / (10),
        np.transpose(POWER_SAVI),
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
    )
    for x in range(10):
        for y in range(10):
            if POWER_BINARY_NSM[x,y] >= 0.03:
                if x!= y:
                    ax_SAVI_terminal_power.text(
                        (x + 0.5) / 10,
                        (y + 0.5) / 10,
                        f"{POWER_SAVI[x,y]:2.2f}",  # data[y,x] +0.05 , data[y,x] + 0.05
                        horizontalalignment="center",
                        verticalalignment="center",
                        color="k",
                    )

    ax_SAVI_terminal_power.set_title(
        f"SAVI Terminal Power: Nmax={500}, alpha={alpha}", fontsize=20
    )

    ax_SAVI_terminal_power.plot([0, 1], [0, 1], "k--", linewidth=5)
    ax_SAVI_terminal_power.set_xlabel("Baseline Performance", fontsize=24)
    ax_SAVI_terminal_power.set_ylabel("Delta Test Policy Performance", fontsize=24)
    ax_SAVI_terminal_power.tick_params(labelsize=20)
    # ax_SAVI_terminal_power.set_aspect("equal")
    ax_SAVI_terminal_power.grid(True)
    fig_SAVI_terminal_power.savefig(f"data/BINARY/POWER_SAVI.png", dpi=450)

    # STEP
    fig_STEP_terminal_power, ax_STEP_terminal_power = plt.subplots(figsize=(10, 10))
    ax_STEP_terminal_power.pcolormesh(
        np.arange(11) / (10),
        np.arange(11) / (10),
        np.transpose(POWER_STEP),
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
    )
    for x in range(10):
        for y in range(10):
            if POWER_BINARY_NSM[x,y] >= 0.03:
                if x!= y:
                    ax_STEP_terminal_power.text(
                        (x + 0.5) / 10,
                        (y + 0.5) / 10,
                        f"{POWER_STEP[x,y]:2.2f}",  # data[y,x] +0.05 , data[y,x] + 0.05
                        horizontalalignment="center",
                        verticalalignment="center",
                        color="k",
                    )

    ax_STEP_terminal_power.set_title(
        f"STEP Terminal Power: Nmax={500}, alpha={alpha}", fontsize=20
    )

    ax_STEP_terminal_power.plot([0, 1], [0, 1], "k--", linewidth=5)
    ax_STEP_terminal_power.set_xlabel("Baseline Performance", fontsize=24)
    ax_STEP_terminal_power.set_ylabel("Delta Test Policy Performance", fontsize=24)
    ax_STEP_terminal_power.tick_params(labelsize=20)
    # ax_STEP_terminal_power.set_aspect("equal")
    ax_STEP_terminal_power.grid(True)
    fig_STEP_terminal_power.savefig(f"data/BINARY/POWER_STEP.png", dpi=450)

    # fig_alt_cumulative_power, ax_alt_cumulative_power = plt.subplots(figsize=(10, 10))
    # ax_alt_cumulative_power.plot(np.arange(N), CUMULATIVE_POWER_NSM[1, alt_counter // 2, :])
    # ax_alt_cumulative_power.plot(np.arange(N), CUMULATIVE_POWER_NSM[1, alt_counter // 2, :])
    # ax_alt_cumulative_power.set_title("Alternative Cumulative Power")
    # fig_alt_cumulative_power.savefig("data/BINARY/Alt_Cumulative_Power.png", dpi=300)

    # fig_null_cumulative_power, ax_null_cumulative_power = plt.subplots(figsize=(10, 10))
    # ax_null_cumulative_power.plot(np.arange(N), CUMULATIVE_POWER_NSM[0, null_counter // 2, :])
    # ax_null_cumulative_power.plot(np.arange(N), CUMULATIVE_POWER_NSM[0, null_counter // 2, :])
    # ax_null_cumulative_power.set_title("Null Cumulative Power")
    # fig_null_cumulative_power.savefig("data/BINARY/Null_Cumulative_Power.png", dpi=300)

    print()
    print("Summary Info for Paper: ")
    print()
    print(f"Total number of alternatives: {number_alternatives}")
    print()
    print(f"Total STEP complexity (e.g., assuming a uniform measure): {total_step_complexity:0.3f}; {total_step_complexity/number_alternatives:0.3f}")
    print(f"Total SAVI complexity (e.g., assuming a uniform measure): {total_savi_complexity:0.3f}; {total_savi_complexity/number_alternatives:0.3f}")
    print(f"Total binary NSM complexity (e.g., assuming a uniform measure): {total_binary_nsm_complexity:0.3f}; {total_binary_nsm_complexity/number_alternatives:0.3f}")
    print(f"Total nonparametric NSM complexity (e.g., assuming a uniform measure): {total_cont_nsm_complexity:0.3f}; {total_cont_nsm_complexity/number_alternatives:0.3f}")
    print(f"Total WSR complexity (e.g., assuming a uniform measure): {total_wsr_complexity:0.3f}; {total_wsr_complexity/number_alternatives:0.3f}")
    print()
    print("Parametric Savings")
    print()
    print(f"SAVI savings (%) over binary NSM: {(100. * (total_binary_nsm_complexity - total_savi_complexity) / total_binary_nsm_complexity):0.3f}")
    print(f"SAVI savings (%) over continuous NSM: {(100. * (total_cont_nsm_complexity - total_savi_complexity) / total_cont_nsm_complexity):0.3f}")
    print()
    print("Nonparametric Savings")
    print()
    print(f"Continuous NSM savings (%) over WSR: {(100. * (total_wsr_complexity - total_cont_nsm_complexity) / total_wsr_complexity):0.3f}")
    print()
    print()
    print(f"Total STEP power (e.g., assuming a uniform measure): {total_step_power-26.:0.3f}; {(total_step_power-26.)/(number_alternatives-26.):0.3f}")
    print(f"Total SAVI power (e.g., assuming a uniform measure): {total_savi_power-26.:0.3f}; {(total_savi_power-26.)/(number_alternatives-26.):0.3f}")
    print(f"Total binary NSM power (e.g., assuming a uniform measure): {total_binary_nsm_power-26.:0.3f}; {(total_binary_nsm_power-26.)/(number_alternatives-26.):0.3f}")
    print(f"Total nonparametric NSM power (e.g., assuming a uniform measure): {total_cont_nsm_power-26.:0.3f}; {(total_cont_nsm_power-26.)/(number_alternatives-26.):0.3f}")
    print(f"Total WSR power (e.g., assuming a uniform measure): {total_wsr_power-26.:0.3f}; {(total_wsr_power-26.)/(number_alternatives-26.):0.3f}")
    