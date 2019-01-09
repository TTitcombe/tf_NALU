import numpy as np
import tensorflow as tf

class BaseCell:
    def __init__(self, input_dim, hidden_dim, output_dim, hyper={}):

        self.x = tf.placeholder(tf.float32, [None, input_dim], name='input')
        self.y = tf.placeholder(tf.float32, [None, output_dim], name='ouput')

        optim = hyper.get('optim', 'rms')
        self.lr = hyper.get('lr', 0.01)

        if optim.lower() == 'adam':
            self.optim = tf.train.AdamOptimizer(self.lr)
        elif optim.lower() == 'gd':
            self.optim = tf.train.GradientDescentOptimizer(self.lr)
        elif optim.lower() == 'rms':
            self.optim = tf.train.RMSPropOptimizer(self.lr)
        else:
            raise NotImplementedError("Learning algorithm not recognised")

    def train(self, x, y, x_test=None, y_test=None, N_epochs=100, batchSize=None):
        if not batchSize:
            batchSize = x.shape[0]

        self.batchSize = batchSize
        self.n_steps = round(x.shape[0] / batchSize)

        for epoch in range(N_epochs):
            loss, err = self._train_epoch(x, y)
            if epoch % 1000 == 0:
                print("\nEpoch: {}".format(epoch))
                print("Err: {}".format(err))
            if epoch % 2000 == 0 and x_test is not None:
                loss_test, err_test, y_hat = self.validate(x_test, y_test)
                print("Test error: {}".format(err_test))

        print("Final training error: {}".format(err))
        loss, err, y_hat = self.validate(x_test, y_test)
        print("Final test error: {}".format(err))

        return err

    def _train_epoch(self, x, y):
        for step in range(self.n_steps):
            if step < self.n_steps - 1:
                x_step = x[step*self.batchSize:(step+1)*self.batchSize]
                y_step = y_data[step*self.batchSize:(step+1)*self.batchSize]
            else:
                x_step = x[step*self.batchSize:]
                y_step = y[step*self.batchSize:]

            loss, err = self._train_step(x_step, y_step)

        return loss, err

    def _train_step(self, x_step, y_step):
            feed_dict = {self.x: x_step, self.y: y_step}

            ops = [self.loss, self.error, self.optimise]

            loss, error, _ = self._Sess.run(ops, feed_dict)

            return loss, error

    def validate(self, x, y):
        feed_dict = {self.x: x, self.y: y}
        ops = [self.loss, self.error, self.y_hat]
        loss, err, y_hat = self._Sess.run(ops, feed_dict)

        return loss, err, y_hat
