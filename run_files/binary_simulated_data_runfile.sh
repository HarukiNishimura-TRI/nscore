# Get results for simulated binary data
python scripts/general/large_scale_bernoulli_test_gather_data.py -a 0.05 -n 1000 -s 31415 -nta 50 -ntn 2
python scripts/general/large_scale_bernoulli_test_process_data.py
python scripts/general/large_scale_bernoulli_test_vizualize_data.py