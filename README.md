# TF Neural Arithmetic Logic Unit

A tensorflow implementation of the paper ["Neural Arithmetic Logic Units"](https://arxiv.org/pdf/1808.00508.pdf) by Andrew Trask, Felix Hill, Scott Reed, Jack Rae, Chris Dyer, Phil Blunsom.

Explanation of the paper, and a write-up of this code, can be found [here](https://medium.com/@t.j.titcombe/understanding-neural-arithmetic-logic-units-5ca9d0041473).

This code currently provides a NAC module, a NALU module, and a MLP.

The experiments in section 4.1 of the paper have been implemented. (Static numerical interpolation and extrpolation)

To test, run **test_nalu.py**.

## Results
Currently have only performed the experiments of section 4.1 of the paper, the static numerical tests (addition, subtraction, multiplication, division, square, square root).

In the results directory are several text files, for various experiments.

Naming convention:
* I for interpolation, E for extrapolation
* add/subtract/multiply etc. defines the operation used to calculate the target data
* "small". In small experiments, the x data has 2 dimensions. These dimensions were e.g. multiplied together in the multiplication test.
If small is not present in the name, then x data has 100 dimensions. Two samples of size 10 (without replacement) were taken. The data in each sample are summed together. These sums are then e.g. multiplied together for the multiplication experiment.

## TODO
* Hyperparameter tune
* Perform numerical extrapolation experiment (section 1.1. in original paper)
    - Graph of mean absolute error against input value does not match the paper well
* Test interpolation of static numerical tests
    - Small data: NAC outperforms relu6 on addition and subtraction, but NALU does not. Bug or tuning?
    - Small data: NALU outperforms relu6 and NAC on multiplication, division etc.
    - NALU and NAC both outperformed by relu6 on large interpolation addition.
* Test extrapolation of static numerical tests
    - Succeed in interpolation first
