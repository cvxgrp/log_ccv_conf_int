# log_ccv_conf_int
Implementation of the paper "Walther G, Ali A, Shen X, Boyd S. Confidence bands for a log-concave density".

Please install [CVXPY 1.1](http://www.cvxpy.org/).

To test the method in one example, please run the python notebook ```test.ipynb```.

To run the experiments in the paper, please first run ```experiment_platform.ipynb``` to compute and save the confidence intervals, and then run ```compute_coverage_rate_and_width.ipynb``` to generate results in Table 1 in the paper.
