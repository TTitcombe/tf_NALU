import tensorflow as tf
import numpy as np
import os

from models import model

class Trainer():
    '''
    A class to handle the batching and training of a defined model
    '''

    def __init__(self, input_dim, output_dim, hidden_dim = None,
                model_type='nac', hyperparams = {}):
        self._Sess = None

        self.model = model(input_dim, output_dim, hidden_dim, model_type, hyperparams)

    def train(self, x, y, x_val, y_val, batchSize, n_epochs):
        self.bs = batchSize
        self.n_steps = x.shape[0] // batchSize
        if x.shape[0] / batchSize != float(self.n_steps):
            self.n_steps += 1

        if self._Sess is None:
            self._Sess = tf.Session()
            init = tf.global_variables_initializer()
            self._Sess.run(init)

        for epoch in range(n_epochs):
            tr_loss, te_loss = self._train_epoch(epoch, x, y, x_val, y_val)
            print("Training err: {}".format(tr_loss))
            print("Testing err: {}".format(te_loss))

    def _train_epoch(self, epoch, x, y, x_val, y_val):
        print("Epoch:{}".format(epoch))

        for step in range(self.n_steps):
            step_start = self.bs * step
            step_end = self.bs * (step + 1)
            x_batch = x[step_start:step_end]
            y_batch = y[step_start:step_end]

            i = epoch * self.n_steps + step
            train_loss = self._train_step(i, x_batch, y_batch)

            if step == 0:
                test_error, test_loss = self._validate(x_val, y_val)

        return train_loss, test_loss

    def _train_step(self, i, x, y):
        feed_dict = {self.model.x: x,
                    self.model.y: y}
        to_run = [self.model.loss, self.model.optimise]
        _loss, _ = self._Sess.run(to_run, feed_dict)
        return _loss

    def _validate(self, x, y):
        feed_dict = {self.model.x: x,
                    self.model.y: y}
        to_run = [self.model.square, self.model.loss]
        _err, _loss = self._Sess.run(to_run, feed_dict)
        return _err, _loss
