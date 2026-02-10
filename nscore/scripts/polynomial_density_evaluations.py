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

if __name__ == "__main__":
    n_runs = 500
    EXPECTED_VALUES = np.zeros(n_runs)

    poly_orders = np.random.choice(np.arange(1, 10), size=(n_runs, 2))

    domain = np.array([0., 1.])
    # basic_poly = Polynomial([0, 0, 1])
    # print(basic_poly(3))
    # deriv = np.polyder(basic_poly)
    # print(deriv)

    for i in tqdm(range(n_runs)):
        try: 
            del coeffs_0, coeffs_1
        except: 
            pass
        # Initialize polynomials
        order_0 = poly_orders[i, 0]
        order_1 = poly_orders[i, 1]

        coeffs_0 = np.random.rand(order_0 + 1) - 0.5
        coeffs_1 = np.random.rand(order_1 + 1) - 0.5

        p0 = Polynomial(coeffs_0)
        p1 = Polynomial(coeffs_1)

        min0, max0 = poly_minmax(p0, domain[0], domain[1])
        integral_0 = p0.integ(1, domain[0])
        i_val_0 = integral_0(domain[1])
        div_factor = i_val_0 - ((min0 - 1e-8)*(domain[1]-domain[0]))

        coeffs_0[0] += (-min0 + 1e-8)
        if div_factor > 0.:
            coeffs_0 /= div_factor
        
        p0 = Polynomial(coeffs_0)
        i_p0 = p0.integ(1, lbnd=domain[0])
        min0, _ = poly_minmax(p0, domain[0], domain[1])
        try:
            assert min0 >= 0. and np.isclose(i_p0(1), domain[1])
        except:
            print("Minimum value: ", min0)
            print("Integral: ", i_p0(domain[1]))
            breakpoint()
            # raise ValueError("Invalid polynomial after filtering")
        
        ev_coeffs_0 = np.zeros(order_0 + 2)
        ev_coeffs_0[1:] = copy.deepcopy(coeffs_0)
        ev_p0 = Polynomial(ev_coeffs_0)
        i_ev_p0 = ev_p0.integ(1, lbnd=domain[0])

        EXPECTED_VALUES[i] = i_ev_p0(domain[1])
        # print("Expected Value: ", i_ev_p0(domain[1]))

        # breakpoint()

        # min1, max1 = poly_minmax(p1, domain[0], domain[1])
        
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.hist(EXPECTED_VALUES)
    fig.savefig("tmp_nonparametric_hist.png", dpi=300)




        