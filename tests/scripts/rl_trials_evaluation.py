import numpy as np 

from sequentialized_barnard_tests.base import Decision, Hypothesis
from sequentialized_barnard_tests.nonparametric_nsm import ContinuousNsmTest
from sequentialized_barnard_tests.wsr import WsrComparisonTest

if __name__ == "__main__":

    policy_0_score = np.load("data/RL/Cartpole/completed_trials_rewards_steps15000.npy").reshape(-1, )
    policy_1_score = np.load("data/RL/Cartpole/completed_trials_rewards_steps50000.npy").reshape(-1, )

    alpha = 0.05 

    nsm_test = ContinuousNsmTest(alternative=Hypothesis.P0LessThanP1, alpha=alpha, c=np.arange(101)/100.)
    wsr_test = WsrComparisonTest(alternative=Hypothesis.P0LessThanP1, alpha=alpha, c_wsr=0.5)

    nsm_result = nsm_test.run_on_sequence(policy_0_score/200., policy_1_score/200.)
    print("Decision: ", nsm_result.decision)
    print("Time of Decision: ", nsm_result.info["Time"])

    wsr_result = wsr_test.run_on_sequence(policy_0_score/200., policy_1_score/200.)
    print("Decision: ", wsr_result.decision)
    print("Time of Decision: ", wsr_result.info["Time"])