import numpy as np
from numpy.polynomial.polynomial import Polynomial
import copy 
from matplotlib import pyplot as plt
from tqdm import tqdm 

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

    n_runs = 2
    poly_orders = np.random.choice(np.arange(1, 10), size=(n_runs, 2))
    domain = np.array([0.,  1.])

    pdf_0, cdf_0, ev_0, alt_ev_0 = construct_polynomial_density_with_mean(poly_orders[0, 0], domain)

    print()
    print("Expected value using int (x pdf(x)): ", ev_0)
    print("Expected value using area above CDF: ", alt_ev_0)
    print()

    n_points = 10000

    values = sample_from_polynomial_cdf(cdf_0, n_points)

    print(f"Empirical mean of {n_points} samples: {np.mean(values)}")





        