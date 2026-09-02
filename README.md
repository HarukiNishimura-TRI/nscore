# README 
Instructions for installation and usage of NSCORE, a nonparametric sequential procedure for rigorous robot policy comparison. NSCORE has minimal and lightweight computational requirements, and is easy to wrap around existing simulation and hardware evaluation pipelines.  

## Installation

Install NSCORE from the repository root with pip:

```bash
pip install .
```

For local development, use an editable install:

```bash
pip install -e .
```

NSCORE depends on `statistical-comparison-core` and
`statistical-comparison-helpers`. Both are published on PyPI, so pip resolves
them automatically and no manual setup is required.

NSCORE itself is also published, so it can be installed without a checkout:

```bash
pip install nscore
```

### Optional conda environment

Conda is not required to install NSCORE, but it can be used to create an isolated
environment:

```bash
conda create -n nscore --file requirements_conda.txt
conda activate nscore
pip install .
```

## Applicability (Non-Technical)
For additional motivation of N-SCORE and related methods, see Tutorials, below. 

### Understanding the Arguments: Hypotheses
NSCORE performs _mean comparison_ between two different random variables of unknown (possibly nonparametric) distribution shape. We will term the variables $R_0$ and $R_1$, with respective (unknown) means $\mu_0$ and $\mu_1$. 

The first thing that we will need is a _hypothesis_: what do we want the relationship to be between $\mu_0$ and $\mu_1$? For example, if $R_0$ is a measure of a baseline policy's performance and $R_1$ is a measure of the performance of our novel method, we want $\mu_0 < \mu_1$ (under the convention that $R_i$ is a reward, i.e., higher is better). This is specified precisely as `alternative=Hypothesis.P0LessThanP1` (the notation $p$ is derived from the original use case of comparing Bernoulli random variables, where the mean $\mu$ is equivalent to the conventional Bernoulli parameter $p$). Conversely, If the $R_i$ are cost measures, where lower is better, we may wish to specify the alternative $\mu_0 > \mu_1$. This is expressed directly as `alternative=Hypothesis.P0MoreThanP1`.  

A second use also arises in the form of error checking. Returning to _reward measures_: imagine there might be a bug in our codebase. We might value a method that runs both tests simultaneously. The first test checks if we improve upon the baseline (assuming no bugs), while the second test quickly tells us if there might be a bug making us perform significantly worse than the baseline, allowing us to stop early and fix the problem! This corresponds to the case of a `Mirrored` (Two-Sided) Test, where we keep track of both directions. We will get into Mirrored tests a bit more, later.  

### Understanding the Arguments: Alpha
When we compare ourselves to a baseline method, we want to establish that our performance is better. But evaluations have significant randomness, meaning that we cannot simply use empirical success to argue for policy improvement. Instead, we propose the following evidential justification: "the probability that our robot policy's performance is _not_ better than the baseline policy is less than $\alpha$."

This guarantee means that, if our policy was _no better than the baseline_, then the probability of observing as strong empirical evidence in favor of our method is less than $\alpha$. Conversely, we can claim $1-\alpha$ __confidence__ that our new policy is indeed better than the baseline (on the shared distribution of environments that both policies are evaluated on). This is a strong form of generalization that allows us to tune our confidence level to any desired $\alpha \in (0, 1)$. 

### Understanding the Arguments: c
At present, NSCORE takes advantage of certain efficient properties of linear representations of the data-generating distributions. The vector $c \in [0, 1]^K$ corresponds to this representation, which is perhaps best understood as a discretization of the interval $[0, 1]$ into bins (where $c$ encodes the bin edge positions). We then construct an approximation of the distribution law from the empirical counts within each bin.  

### Instantiating and Running an NSCORE Test on Full Datasets (Basic Usage)
To use NSCORE, it is necessary to specify the three aforementioned parameters. 

```python 

alternative: [Hypothesis]
alpha: [float]
c: [np.ndarray]

```

As an example: for Bernoulli data, to test if our policy $\pi_1$ has a higher success rate than a baseline $\pi_0$ at $95$\% confidence, we would specify the test:

```python 
nscore_test = BernoulliNsmTest(alternative=Hypothesis.P0LessThanP1, alpha=0.05, c=np.arange(2)/1.) 
```
and, given an evaluation sequence of success/failure outcomes for each policy, we would run the test as:
```python 
nscore_result = nscore_test.run_on_sequence(outcomes_for_pi_0, outcomes_for_pi_1)
```

### Instantiating and Running an NSCORE Test on Full Datasets (General Metrics)
If we have performance measures that are not bounded in $[0, 1]$, then the measures must be __normalized__ using _a priori_ knowledge of the test domain. For example, on the Mujoco InvertedPendulum-v4 task, the reward is bounded by the time horizon $T$; therefore, normalization can be undertaken simply by scaling down the results. First, we define a more detailed NSM test for the more complex performance measure:
```python 
nscore_inverted_pendulum_test = ContinuousNsmTest(alternative=Hypothesis.P0LessThanP1, alpha=0.05, c=np.arange(101)/100.) 
```
Next, we use the same `run_on_sequence()` functionality, but normalize the scores: 
```python 
nscore_inverted_pendulum_result = nscore_inverted_pendulum_test.run_on_sequence(outcomes_for_pi_0/T, outcomes_for_pi_1/T)
```

### Instantiating and Running an NSCORE Test _Online_ (General Metrics)
NSCORE can be used to save time by allowing the evaluator to update the evaluation _online_ as they gather trials. The test stops precisely
when enough evidence has accumulated to be $1-\alpha$ confident that the intended decision is correct. Taking the InvertedPendulum-v4 task as the running example, we first initialize the test as before:
```python 
nscore_online_evaluation_test = ContinuousNsmTest(alternative=Hypothesis.P0LessThanP1, alpha=0.05, c=np.arange(101)/100.) 
```
and then implement a simple evaluation protocol
```python
time_of_decision = 0
decided = False
maximum_number_of_evals_per_policy = 100 # Optional
while (decided is False) and (time_of_decision < maximum_number_of_evals_per_policy):
    new_datum_pi_0 = run_pi_0_on_new_iid_environment(ics, ...)
    new_datum_pi_1 = run_pi_1_on_new_iid_environment(ics, ...)

    result = nscore_online_evaluation_test.step(new_datum_pi_0, new_datum_pi_1)

    if result.decision is Decision.AcceptAlternative:
        decided = True
    
    time_of_decision += 1

# Print the results
print(f"NSCORE test decided in {time_of_decision} trials per policy")
print(f"NSCORE decision: {result.decision}")
```

### Understanding an NSCORE Test Result
We utilize the same structure for test decisions as our previous work, [STEP](https://github.com/TRI-ML/sequentialized_barnard_tests). Formally, Neyman-Pearson statistical testing only allows for _accepting_ the alternative hypothesis (if sufficient evidence for it is accumulated, of course). This is encoded by a `Decision` object; for a realized test that rejects the null and accepts the alternative, the associated decision would be `test.decision = Decision.AcceptAlternative`. 

Of course, the data might be insufficiently indicative of the alternative. Because the tests are sequential, there is the opportunity to gather more, so we define a placeholder, denoted by `Decision.FailToDecide`. For anytime-valid tests, data may in principle be collected in perpetuity; thus, `Decision.FailToDecide` implies simply that 'not enough information has yet accumulated to make up our mind one way or the other.' 

Finally, as mentioned earlier, there are practical instances where we might wish to monitor both directions of a comparison. Informally, the standard direction seeks certify that our policy has improved over the baseline, while the other direction gives us reliable statistical evidence that we should stop early and 'give up;' that is, it tells us that our novel method is performing significantly worse, and that gathering more trials is unlikely to change that assessment. This can be used as a bug-catcher, as well as to reliable help with design iterations _when used responsibly_. 

__We emphasize that a "kitchen sink" approach of just trying a bunch of different $\pi_1^{[i]}$ until a significant result is found against $\pi_0$ constitutes p-hacking, and invalidates statistical assurances__. More precisely, each $\pi_1^{[i]}$ must be accompanied by an additional union bound correction of $\alpha$ -- concretely: if you test five new policies against $\pi_0$, with each test at level $\alpha$, any significant difference found can only be reported at level $5\alpha$! This is the union bound, or "Bonferroni correction" in classical hypothesis testing. 

However, when used responsibly, the two-sided test can add significant further empirical savings to the practitioner's evaluation burden. In our setup, a two-sided test is termed `Mirrored`, because it will consist of two _mirrored_ one-sided tests; the tests will individually check the hypotheses `Hypothesis.P0LessThanP1` and `Hypothesis.P0MoreThanP1`. In this context, the user will still specify a specific alternative, which then amounts to selecting which hypothesis is the semantic "Alternative" and which is the "Null." From this, we construct a third decision option, `Decision.AcceptNull`, which amounts to accepting the semantic "Null" as the alternative of a second one-sided test. For example: 
```python 
mirrored_nscore_test = MirroredContinuousNsmTest(alternative=Hypothesis.P0LessThanP1, alpha=0.05, c=np.arange(101)/100.) 
```
will induce two one-sided tests: 
```python 
nscore_test_for_alternative: ContinuousNsmTest(alternative=Hypothesis.P0LessThanP1, alpha=0.05, c=np.arange(101)/100.)
nscore_test_for_null: ContinuousNsmTest(alternative=Hypothesis.P0MoreThanP1, alpha=0.05, c=np.arange(101)/100.)
```
Formally `result.decision = Decision.AcceptNull` here is equivalent to accepting the _alternative hypothesis_ of `nscore_test_for_null` (as required by Neyman-Pearson testing -- one is not generally allowed to accept a null hypothesis). Semantically, however, we have _informally_ accepted the null in the sense that we have concluded, with high confidence, that $\mu_0 > \mu_1$.

## Tutorials
Tutorials that illustrate standard use cases of NSCORE can be found as Jupyter Notebooks under `/notebooks`.

## Citation
If you find this code useful, please cite our work: 


```bibtex
@inproceedings{snyder_beyond_2026,
    title = {Beyond {Binary} {Success}: {Sample}-{Efficient} and {Statistically} {Rigorous} {Robot} {Policy} {Comparison}},
    author = {Snyder, David and Badithela, Apurva and Matni, Nikolai and Pappas, George and Majumdar, Anirudha and Itkina, Masha and Nishimura, Haruki},
    booktitle={arXiv preprint arXiv:2603.13616}
    year = {2026},
} 
```
