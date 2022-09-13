# pyfit3Dcspline

This module is a Python adaptation of the [fit3Dcspline](https://github.com/jries/SMAP/tree/master/fit3Dcspline) main fitting functionality that offers both CPU and GPU accelerated fitting of SMLM data.

- This implementation's performance has not been yet tested against the C compiled fit3Dcspline, thus we appreciate early testers and contributors feedback.

## Helper Functions

- Functions `GPUmleFit_LM` and `CPUmleFit_LM` are the single channel fitters for GPU and CPU, respectively.
- Functions `get_roi_list` and `get_roi_list_CMOS` offer a simple way to get the data and variance array lists from the `points` array within a specific `roi_size`.
- The `ParametersHeaders` is a `dict` that returns the column headers for each fit mode supplied as key. 

## How to Use

**TBA**

# Reference

The original fit3Dcspline code is provided as a part of the Ries group SMAP software, and was published in [paper](https://www.nature.com/articles/nmeth.4661):
  * Yiming Li, Markus Mund, Philipp Hoess, Joran Deschamps, Ulf Matti, Bianca Nijmeijer, Vilma Jimenez Sabinina, Jan Ellenberg, Ingmar Schoen, Jonas Ries.  Real-time 3D single-molecule localization using experimental point spread functions. Nat. Methods 15, 367–369 (2018).
