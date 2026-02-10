PYTHONPATH="/home/dasnyder/Documents/GitHub/sbt/":"${PYTHONPATH}"
export PYTHONPATH

# python sequentialized_barnard_tests/scripts/simple_adaptive_nsm_test.py -a 0.05 -n 2000 -s 31415 -nta 1000 -ntn 100
# python sequentialized_barnard_tests/scripts/nonparametric_nsm_test.py -a 0.05 -n 2000 -s 31415 -nta 5 -ntn 5
# python sequentialized_barnard_tests/scripts/nonparametric_nsm_scaling_test.py -a 0.05 -n 2000 -s 31415 -nta 10 -ntn 10
# python sequentialized_barnard_tests/scripts/nonparametric_nsm_and_wsr_comparison.py -a 0.05 -n 2000 -s 31415 -nta 100 -ntn 10
python sequentialized_barnard_tests/scripts/full_large_scale_bernoulli_test.py -a 0.05 -n 1000 -s 31415 -nta 50 -ntn 2
python sequentialized_barnard_tests/scripts/analyze_full_large_scale_bernoulli_test.py
python sequentialized_barnard_tests/scripts/viz_full_large_scale_bernoulli_test.py
# python scripts/partial_credit_nsm_debug.py -a 0.05 -n 2000 -s 31415
# python sequentialized_barnard_tests/scripts/synthesize_general_step_policy.py -n 500 -a 0.005
# python sequentialized_barnard_tests/scripts/multitest_heuristic.py 
# python sequentialized_barnard_tests/scripts/twosided_pcnsm_test.py -a 0.05 -n 2000 -s 31415 -nta 100 -ntn 100
# python sequentialized_barnard_tests/scripts/compare_nsm_and_welch.py
# python sequentialized_barnard_tests/scripts/break_welch.py
# python sequentialized_barnard_tests/scripts/evaluate_partial_credit_and_welch.py -a 0.05 -s 31415 
# python sequentialized_barnard_tests/scripts/evaluate_partial_credit_and_welch_bernoulli.py -a 0.05 -s 11111 -g 0.25 -nc 1000 -nt 5 
# python sequentialized_barnard_tests/scripts/welch_and_nsm_data_analysis.py