# TF Neural Arithmetic Logic Unit

A tensorflow implementation of the paper ["Neural Arithmetic Logic Unit"](https://arxiv.org/pdf/1808.00508.pdf) by Andrew Trask, Felix Hill, Scott Reed, Jack Rae, Chris Dyer, Phil Blunsom.

Explanation of the paper, and a write-up of this code, can be found [here](https://medium.com/@t.j.titcombe/understanding-neural-arithmetic-logic-units-5ca9d0041473).

This code currently provides a NAC module, a NALU module, and a deep network of stacked NALU modules.
The experiments in section 4.1 of the paper have been implemented.
To test, run **test_nalu.py**

### TODO
* Hyperparameter tune
* Perform numerical extrapolation experiment (section 1.1. in original paper)
    - Graph of mean absolute error against input value does not match the paper well
* Test interpolation of static numerical tests
    - NALU and NAC both outperformed by relu6 on interpolation addition. Is this due to poor tuning or a bug?
* Test extrapolation of static numerical tests
    - Succeed in interpolation first

