# log concave confidence interval
Python implementation of the paper "Walther G, Ali A, Shen X, Boyd S. Confidence bands for a log-concave density".

Please first install [CVXPY 1.1](http://www.cvxpy.org/).

To test the method in one example, please run the python notebook ```test.ipynb```.

To run the experiments in the paper, please first run ```experiment_platform.ipynb``` to compute and save the confidence intervals, and then run ```compute_coverage_rate_and_width.ipynb``` to generate results in Table 1 in the paper.

## Usage
**Construct object.** Under the directory, import `confint`. 
Given data `X`, to run the method with confidence level specified by `1 - alpha` and ratio of optimized points specified by `opt_pts_ratio`, 
first construct an object of the `confint` class. For example
```python3
from confint import confint
import numpy as np

X = np.random.randn(n)
X = np.sort(X)
opt_pts_ratio = 0.5

conf_int = confint(X, alpha, opt_pts_ratio=opt_pts_ratio)
```

**Solve.** Then call the method `compute_pw_conf_ints` to compute piecewise confidence intervals.
Following the above example, the code is as follows.
```python3
conf_int.compute_pw_conf_ints()
```
The following optional arguments can be passed into the solve method.
* `thread_num` gives the number of threads used to run the parallel algorithm.
* algorithm parameters
    * `tau_max` specifies the maximum value of penalty parameter `tau`, and the default value is `1e3`.
    * `mu` specifies the increase of `tau`, and the default value is `8`.
    * `max_iters` specifies the maximum number of iterations. Default value is `50`.
    * `min_iters` specifies the minimum number of iterations. Default value is `15`.
    * `M` gives a numeric lower bound `exp(-M)` on the density, and a typical value is around `10`.
* `verbose` is a boolean giving the choice of printing information during the iterations. Default value is `False`.

**Retrieve result.**
