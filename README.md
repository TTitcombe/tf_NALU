# TF Neural Arithmetic Logic Unit

A tensorflow implementation of the paper ["Neural Arithmetic Logic Unit"](https://arxiv.org/pdf/1808.00508.pdf) by Andrew Trask, Felix Hill, Scott Reed, Jack Rae, Chris Dyer, Phil Blunsom.

Explanation of the paper, and a write-up of this code, can be found [here](https://medium.com/@t.j.titcombe/understanding-neural-arithmetic-logic-units-5ca9d0041473).

This code currently provides a NAC module, a NALU module, and a deep network of stacked NALU modules.
The experiments in section 4.1 of the paper have been implemented.
To test, run **test_nalu.py**

### TODO
* Hyperparameter tune
* Perform numerical extrapolation experiment (section 1.1. in original paper)

