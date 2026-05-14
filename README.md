# README 
Instructions for installation and usage of NSCORE, a nonparametric sequential procedure for rigorous robot policy comparison. NSCORE has minimal and lightweight computational requirements, and is easy to wrap around existing simulation and hardware evaluation pipelines.  

## Installation (conda)
conda create -n nscore --file requirements_conda.txt  
conda activate nscore  
pip install -r requirements_pip.txt  

## Applicability (Non-Technical)
For additional motivation of N-SCORE and related methods, see Tutorials, below. 

### Understanding the Arguments: Hypotheses
NSCORE performs _mean comparison_ between two different random variables of unknown (possibly nonparametric) distribution shape. We will term the variables $R_0$ and $R_1$, with respective (unknown) means $\mu_0$ and $\mu_1$. 

The first thing that we will need is a _hypothesis_: what do we want the relationship to be between $\mu_0$ and $\mu_1$? For example, if $R_0$ is a baseline procedure and $R_1$ is our novel method, we want $\mu_0 < \mu_1$ (under the convention that $R_i$ is a reward, i.e., higher is better). If $R_i$ is a cost measure, where lower is better, we may wish to specify the alternative $\mu_0 > \mu_1$. 

Returning to _reward measures_: imagine there might be a bug in our codebase. We might value a method that runs both tests simultaneously. The first test checks if we improve upon the baseline (assuming no bugs), while the second test quickly tells us if there might be a bug making us perform significantly worse than the baseline! This corresponds to the case of a Mirrored Test, where we keep track of both directions. 

### Understanding the Arguments: Alpha
When we compare ourselves to a baseline method, we want to establish that our performance is better. But evaluations have significant randomness, meaning that we cannot simply use empirical success to argue for policy improvement. Instead, we propose the following evidential justification: "the probability that our robot policy's performance is _not_ better than the baseline policy is less than $\alpha$."

In other words, if our policy was _no better than the baseline_, then the probability of observing as strong empirical evidence in favor of our method is less than $\alpha$. This is a strong form of generalization that allows us to tune our confidence level to any desired $\alpha \in (0, 1)$. 

### Understanding the Arguments: c
At present, NSCORE takes advantage of certain efficient properties of linear representations of the data-generating distributions. The vector $c \in [0, 1]^K$ corresponds to this representation, which is perhaps best understood as a discretization of the interval $[0, 1]$ into bins (where $c$ encodes the bin edge positions). We then construct an approximation of the distribution law from the empirical counts within each bin.  

### Instantiating an NSCORE Test
To use NSCORE, it is necessary to specify the three aforementioned parameters. 

```python 

alternative: [Hypothesis]
alpha: [float]
c: [np.ndarray]

```

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
