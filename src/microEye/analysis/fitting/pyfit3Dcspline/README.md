# pyfit3Dcspline

The `pyfit3Dcspline` module is a Python adaptation of the [fit3Dcspline](https://github.com/jries/SMAP/tree/master/fit3Dcspline) main fitting functionality. It offers both CPU and GPU accelerated fitting of Single-Molecule Localization Microscopy (SMLM) data.

> **Note:** The performance of this implementation has not yet been tested against the C-compiled fit3Dcspline. We appreciate feedback from early testers and contributors.

## Helper Functions

- **GPUmleFit_LM**: Single channel fitter for GPU.
- **CPUmleFit_LM**: Single channel fitter for CPU.
- **get_roi_list**: Retrieves data and variance array lists from the `points` array within a specified `roi_size`.
- **get_roi_list_CMOS**: Similar to `get_roi_list`, specifically for CMOS sensors.
- **ParametersHeaders**: A dictionary that returns the column headers for each fit mode supplied as a key.

## How to Use

### To Be Announced

## Reference

The original fit3Dcspline code is part of the Ries group SMAP software, published in the following paper:

> Yiming Li, Markus Mund, Philipp Hoess, Joran Deschamps, Ulf Matti, Bianca Nijmeijer, Vilma Jimenez Sabinina, Jan Ellenberg, Ingmar Schoen, Jonas Ries. Real-time 3D single-molecule localization using experimental point spread functions. Nat. Methods 15, 367–369 (2018). [Link to paper](https://www.nature.com/articles/nmeth.4661)
