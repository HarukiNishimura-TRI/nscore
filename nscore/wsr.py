"""
Docstring for nscore.wsr
"""

import numpy as np
from nscore.utils.utils_wsr import mean_cs_eff_corrected_membership_accelerated
from statistical_comparison_core import Decision, Hypothesis, SequentialTestBase, TestResult

from typing import Union

###############################################
# Partial credit policy comparison using WSR-CS
###############################################

class WsrComparisonTest(SequentialTestBase):
    """
    Docstring for WsrComparisonTest
    """
    def __init__(self, 
        alternative: Hypothesis,
        alpha: float,
        c_wsr: float = 0.95,
        verbose: bool = False,
    ) -> None:
        """
        Docstring for __init__
        
        :param self: Description
        :param alternative: Description
        :type alternative: Hypothesis
        :param alpha: Description
        :type alpha: float
        :param c: Description
        :type c: float
        :param verbose: Description
        :type verbose: bool
        """
        self.alternative = alternative
        self.alpha = alpha
        self.c_wsr = c_wsr

        # Time state for decision information
        self._t = None
        
        # Comparison is undertaken via confidence intervals of the differences
        # These store the current lower_bound and upper_bound
        self._previous_lb = None
        self._previous_ub = None

        # Variable in which to store the observed policy outcome differences
        self._observed_data = None

        # Store history of confidence intervals for the mean
        self._intervals = None

        self.reset(verbose)
    
    def step(self, 
        datum_0: Union[bool, int, float],
        datum_1: Union[bool, int, float],
        verbose: bool = False,
    ) -> TestResult:
        
        datum_0_valid = datum_0 >= 0. and datum_0 <= 1.
        datum_1_valid = datum_1 >= 0. and datum_1 <= 1.
        
        if not (datum_0_valid and datum_1_valid):
            raise (ValueError("Input data are not understood to be in the range [0., 1.]."))
        
        if verbose:
            print(
                (
                    "Update the WSR process given new "
                    f"datum_0 == {datum_0} and datum_1 == {datum_1}."
                )
            )
        
        # Update time increment
        self._t += 1

        # Take difference, conventionally datum_1 - datum_0
        new_datum = datum_1 - datum_0
        
        # Append to aggregated data
        self._observed_data.append(new_datum)

        # Get updated intervals
        if self._t > 2:
            new_lb, new_ub = mean_cs_eff_corrected_membership_accelerated(self._observed_data, self.alpha, self.c_wsr, -1., 1., self._previous_lb, self._previous_ub)
        else:
            new_lb = -1.
            new_ub = 1.
        
        # Update previous lb and ub
        if new_lb >= self._previous_lb:
            self._previous_lb = new_lb
        
        if new_ub <= self._previous_ub:
            self._previous_ub = new_ub
        
        # Update intervals
        new_interval = np.array([new_lb, new_ub])
        self._intervals.append(new_interval)
        
        # Check if decision has been reached
        decision = Decision.FailToDecide
        info = {"Time": self._t, "Current_LB": new_lb, "Current_UB": new_ub}

        if new_lb > 0.:
            # Policy 1 does better whp
            if self.alternative == Hypothesis.P0LessThanP1:
                # Alternative is true
                decision = Decision.AcceptAlternative
            elif self.alternative == Hypothesis.P0MoreThanP1:
                # Alternative is false
                decision = Decision.AcceptNull
        elif new_ub < 0.:
            # Policy 0 does better whp
            if self.alternative == Hypothesis.P0LessThanP1:
                # Alternative is false
                decision = Decision.AcceptNull
            elif self.alternative == Hypothesis.P0MoreThanP1:
                # Alternative is true
                decision = Decision.AcceptAlternative
        
        result = TestResult(decision, info)

        return result


    def reset(self, verbose: bool = False) -> None:
        """Resets the underlying WSR process.

        Args:
            verbose (optional): If True, print the outputs to stdout. Defaults to False.
        """
        if verbose:
            print("Reset the WSR process under the following hypotheses:")
        self._previous_lb = -1.0
        self._previous_ub = 1.0

        self._t = int(0)

        try:
            del self._intervals, self._observed_data
        except:
            pass
        
        self._observed_data = []
        self._intervals = []
        
        if self.alternative == Hypothesis.P0MoreThanP1:
            if verbose:
                print("    Null:        P0 <= P1")
                print("    Alternative: P0 >  P1")
        elif self.alternative == Hypothesis.P0LessThanP1:
            if verbose:
                print("    Null:        P0 >= P1")
                print("    Alternative: P0 <  P1")
