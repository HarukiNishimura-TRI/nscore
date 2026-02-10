import numpy as np
from numpy.polynomial.polynomial import Polynomial
import copy 
from tqdm import tqdm

from sequentialized_barnard_tests.nonparametric_nsm import ContinuousNsmTest
from sequentialized_barnard_tests.wsr import WsrComparisonTest
from sequentialized_barnard_tests.base import Decision, Hypothesis

def poly_minmax(p, x_low, x_high):
    # get local minima and maxima
    x_minmax = p.deriv().roots()
    
    x_candidates = []
    if x_low is not None:
        x_candidates.append(x_low)

    try:
        L = x_minmax.shape[0]
        for i in range(L):
            in_range = True
            if x_low is not None:
                in_range &= (x_low <= x_minmax[i])
            if x_high is not None:
                in_range &= (x_minmax[i] <= x_high)
            
            if in_range and (x_minmax[i].imag <= 1e-6):
                x_candidates.append(x_minmax[i].real)
        
    except:
        pass
    if x_high is not None:
        x_candidates.append(x_high)

    # find the lowest of all possible candidates
    x_candidates = np.array(x_candidates).reshape(-1,)
    return p(x_candidates).min(), p(x_candidates).max()


def construct_polynomial_density_with_mean(order: int=5, domain=np.array([0., 1.])):
    coeffs_0 = 3. * (np.random.rand(order + 1) - 0.5)
    p0 = Polynomial(coeffs_0)
    min0, max0 = poly_minmax(p0, domain[0], domain[1])
    integral_0 = p0.integ(1, domain[0])
    i_val_0 = integral_0(domain[1])
    div_factor = i_val_0 - ((min0 - 1e-8)*(domain[1]-domain[0]))

    coeffs_0[0] += (-min0 + 1e-8)
    if div_factor > 0.:
        coeffs_0 /= div_factor
    
    pdf = Polynomial(coeffs_0)
    cdf = pdf.integ(1, lbnd=domain[0])
    min, _ = poly_minmax(pdf, domain[0], domain[1])
    
    assert min >= 0. and np.isclose(cdf(domain[1]), 1.)
    
    ev_coeffs = np.zeros(order + 2)
    ev_coeffs[1:] = copy.deepcopy(coeffs_0)
    ev_p = Polynomial(ev_coeffs)
    i_ev_p = ev_p.integ(1, lbnd=domain[0])

    expected_value = i_ev_p(domain[1])
    i_cdf = cdf.integ(1, lbnd=domain[0])

    alternative_expected_value = 1. - i_cdf(domain[1])

    return pdf, cdf, expected_value, alternative_expected_value

def sample_from_polynomial_cdf(cdf: Polynomial, n_points: int=1000, domain=np.array([0., 1.])):
    
    random_values = np.zeros(n_points)
    random_args = np.random.rand(n_points)*(domain[1] - domain[0])
    for i in range(n_points):
        random_values[i] = 1. - cdf(random_args[i])
    
    return random_values

if __name__ == "__main__":

    np.random.seed(42)

    n_runs = 1000
    n_max = 1000
    alpha = 0.05

    EXPECTED_VALUES = np.zeros((n_runs, 2, 2))
    POLY_ORDERS = np.random.choice(np.arange(1, 10), size=(n_runs, 2))

    CORRECTNESS = np.zeros((n_runs, 2))
    TTD = np.ones((n_runs, 2))*n_max

    domain = np.array([0., 1.])

    # Begin loop
    for i in tqdm(range(n_runs)):
        if i % 50 == 49:
            # Print some useful info
            print("Number of runs so far: ", i-1)
            print("N_max: ", n_max)
            print()
            print("NSM Approach: ")
            print(f"Empirical Power: {100. * np.mean(CORRECTNESS[:i-1, 0])}")
            print(f"Empirical TTD: {np.mean(TTD[:i-1, 0])}")
            print()
            print("WSR Approach: ")
            print(f"Empirical Power: {100. * np.mean(CORRECTNESS[:i-1, 1])}")
            print(f"Empirical TTD: {np.mean(TTD[:i-1, 1])}")
        
        # Reset the tests
        nsm_test = ContinuousNsmTest(alternative=Hypothesis.P0LessThanP1, alpha=alpha, c=np.arange(21)/20.)
        wsr_test = WsrComparisonTest(alternative=Hypothesis.P0LessThanP1, alpha=alpha, c_wsr=0.95)

        # Sample the polynomials
        keep_regenerating = True
        while keep_regenerating:
            pdf_0, cdf_0, ev_0, alt_ev_0 = construct_polynomial_density_with_mean(POLY_ORDERS[i, 0], domain)
            pdf_1, cdf_1, ev_1, alt_ev_1 = construct_polynomial_density_with_mean(POLY_ORDERS[i, 1], domain)

            try:
                assert np.isclose(ev_0, alt_ev_0) and np.isclose(ev_1, alt_ev_1) and np.abs(ev_0 - ev_1) >= 0.02 and np.abs(ev_0 - ev_1) <= 0.2
                keep_regenerating = False
            except:
                pass
        
        random_draws_0 = sample_from_polynomial_cdf(cdf_0, n_max)
        random_draws_1 = sample_from_polynomial_cdf(cdf_1, n_max)

        if ev_0 <= ev_1:
            EXPECTED_VALUES[i, 0, 0] = ev_0
            EXPECTED_VALUES[i, 0, 1] = np.mean(random_draws_0)
            EXPECTED_VALUES[i, 1, 0] = ev_1
            EXPECTED_VALUES[i, 1, 1] = np.mean(random_draws_1)

            nsm_result = nsm_test.run_on_sequence(random_draws_0, random_draws_1)
            wsr_result = wsr_test.run_on_sequence(random_draws_0, random_draws_1)
        
        else: # ev_0 > ev_1
            EXPECTED_VALUES[i, 0, 0] = ev_1
            EXPECTED_VALUES[i, 0, 1] = np.mean(random_draws_1)
            EXPECTED_VALUES[i, 1, 0] = ev_0
            EXPECTED_VALUES[i, 1, 1] = np.mean(random_draws_0)

            nsm_result = nsm_test.run_on_sequence(random_draws_1, random_draws_0)
            wsr_result = wsr_test.run_on_sequence(random_draws_1, random_draws_0)
        
        if nsm_result.decision == Decision.AcceptAlternative:
            CORRECTNESS[i, 0] = 1. 
            TTD[i, 0] = nsm_result.info["Time"]
        if wsr_result.decision == Decision.AcceptAlternative:
            CORRECTNESS[i, 1] = 1. 
            TTD[i, 1] = wsr_result.info["Time"]
    
    # Save the data
    np.save("data/DENSITY/EV.npy", EXPECTED_VALUES)
    np.save("data/DENSITY/CORRECTNESS.npy", CORRECTNESS)
    np.save("data/DENSITY/TTD.npy", TTD)

    # Print some useful info
    print("Number of runs: ", n_runs)
    print("N_max: ", n_max)
    print()
    print("NSM Approach: ")
    print(f"Empirical Power: {100. * np.mean(CORRECTNESS[:, 0])}")
    print(f"Empirical TTD: {np.mean(TTD[:, 0])}")
    print()
    print("WSR Approach: ")
    print(f"Empirical Power: {100. * np.mean(CORRECTNESS[:, 1])}")
    print(f"Empirical TTD: {np.mean(TTD[:, 1])}")
        
    





        