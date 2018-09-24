# TF Neural Arithmetic Logic Unit

![Extrapolation test](https://github.com/TTitcombe/tf_NALU/blob/master/figures/extrapolation_test.png)

A TensorFlow implementation of the paper ["Neural Arithmetic Logic Units"](https://arxiv.org/pdf/1808.00508.pdf) by Andrew Trask, Felix Hill, Scott Reed, Jack Rae, Chris Dyer, Phil Blunsom.

Explanation of the paper, and a write-up of this code, can be found [here](https://medium.com/@t.j.titcombe/understanding-neural-arithmetic-logic-units-5ca9d0041473).

This code currently provides a NAC module, a NALU module, and a MLP.

To perform the static numerical tests, run **test_nalu.py**.

## Project structure
This code is written in a way to make training and testing as quick and scalable as possible:
* The NAC and NALU models are defined in **nalu.py**. They contain **__init__** and **__call__** methods only.
Calling the object takes a tensor as input (not as a placeholder), and returns an output tensor.
* **models.py** contains the **Model** class. This class is where the x and y tensor placeholders are stored, and the learning hyperparameters such as
learning rate and algorithm are decided. This class passes the data through one of the models in **nalu.py** to define the graph. Loss and error are defined in this class
* **trainer.py** contain the **Trainer** class. This class takes the entire x and y data, splits it into batches, and perform the step and epoch logic by passing a batch to the defined **Model** class.

So **Trainer** contains **Model** contains **NALU**.

## Results
Currently have only performed the experiments of section 4.1 of the paper, the static numerical tests (addition, subtraction, multiplication, division, square, square root).

In the results directory are several text files, for various experiments.

Naming convention:
* I for interpolation (between -1. and 1.), E for extrapolation (between -10. and 10.)
* add/subtract/multiply etc. defines the operation used to calculate the target data
* "small". In small experiments, the x data has 2 dimensions. These dimensions were e.g. multiplied together in the multiplication test.
If small is not present in the name, then x data has 100 dimensions. Two samples of size 10 (without replacement) were taken. The data in each sample are summed together. These sums are then e.g. multiplied together for the multiplication experiment.

### Static Numerical (Small)
**Interpolation**: NAC > relu6 > NALU > random for addition and subtraction. Is this a bug or down to hyperparam tuning?
NALU > relu6 > NAC > random for the remainder of the tests.

Clearly something isn't working perfectly: As a NALU is a generalised NAC, NALU should be able to perform *as well as* NAC in add and sub tasks.

**Extrapolation**: NAC > relu6 > random > NALU in addition and subtraction. NALU is...somehow worse than random? **need to find out why**.

Good news: NAC achieved 0.0 error in addition and subtraction.


## TODO
* Hyperparameter tune
* Perform numerical extrapolation experiment (section 1.1. in original paper)
    - Graph of mean absolute error against input value does not match the paper well
* Test interpolation of static numerical tests
    - NALU and NAC both outperformed by relu6 on large interpolation addition. WHY?!
* Test extrapolation of static numerical tests
    - Succeed in interpolation first
