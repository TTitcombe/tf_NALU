# TF Neural Arithmetic Logic Unit

![Extrapolation test](https://github.com/TTitcombe/tf_NALU/blob/master/figures/extrapolation_test.png)

A TensorFlow implementation of the paper ["Neural Arithmetic Logic Units"](https://arxiv.org/pdf/1808.00508.pdf) by Andrew Trask, Felix Hill, Scott Reed, Jack Rae, Chris Dyer, Phil Blunsom.

Explanation of the paper, and a write-up of this code, can be found [here](https://medium.com/@t.j.titcombe/understanding-neural-arithmetic-logic-units-5ca9d0041473).

This code currently provides a NAC module, a NALU module, and a MLP.

To perform the static numerical tests, run **static_arithmetic_task.py**.

The proof-of-necessity experiment in the paper, trying to learn the identity mapping with a neural network, can be carried out in **test_nn_extrapolation.py**. The results of the quick test done with this code can be seen above.


## Results

### Sanity Check
I completed a quick sanity check by training a 1-layer NALU on 10 2-dimensional data points until loss became unchanged (a smaller version of the static arithmetic task).
Looking at the weight values (W) and g in results/Weights_Sanity_Test.txt, we can see that the NALU learns almost perfectly how to model basic arithmetic functions in this toy example. We're good to continue with the more complex tasks!

### Static Arithmetic
Currently have only performed the experiments of section 4.1 of the paper, the static numerical tests (addition, subtraction, multiplication, division, square, square root).

The results can be seen in results/static_arithmetic_test.txt
We can see that, for all operations except for divide, NAC or NALU achieved the best results. The results for divide are not surprising, as having a/b will be close to 1 unless a >> b or b << a, thus making it quite easy for a MLP to find some local minimum near its initialised state. To combat this one could draw a and b from a greater number of data dimensions, and make the difference between them more distinct i.e. a is the sum of 20 dimensions, b is the sum of 5.

For addition and subtraction, NAC outperformed NALU. While NALU can learn to act as a NAC, these results are to be expected due to the extra complexity of a NALU - its more difficult to learn to become a high-performing NAC than for the NAC to learn to be high-performing.

Note that I did very minimal hyper-perameter tuning, and number of data points and epochs were kept to a minimum, because I am not made of compute. Compared to the paper, it's clear that I've found suboptimal solutions. However, these results are enough to convince me of the power of the NALU.


